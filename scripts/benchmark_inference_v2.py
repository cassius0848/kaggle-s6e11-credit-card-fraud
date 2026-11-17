"""Portable experiment entry point."""

def main():
    import os
    if __package__:
        from .project_runtime import prepare_runtime,experiment_dir
    else:
        from project_runtime import prepare_runtime,experiment_dir
    RUNTIME_ROOT,DATA_DIR=prepare_runtime('benchmark',__file__)
    from pathlib import Path
    import sys,time,json,statistics,os
    ROOT=RUNTIME_ROOT
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from catboost import CatBoostClassifier,Pool
    EXP=experiment_dir(ROOT,'v2')
    OUT=ROOT/'outputs'
    start=time.perf_counter()
    raw=pd.read_csv(DATA_DIR/'test.csv').drop(columns='id')
    load_data_seconds=time.perf_counter()-start
    rows=len(raw)
    selection=json.loads((EXP/'selection.json').read_text(encoding='utf-8'))
    records=json.loads((EXP/'results.json').read_text(encoding='utf-8'))
    report={'rows':rows,'repetitions':3,'threads':4,'load_data_seconds':load_data_seconds,'logical_cpu_count':os.cpu_count(),'models':[]}
    predictions={}
    for name in selection['selected_weights']:
        start=time.perf_counter()
        if name.startswith('cat'):
            model=CatBoostClassifier()
            model.load_model(str(EXP/(name+'.cbm')))
        else:
            model=lgb.Booster(model_file=str(EXP/(name+'.txt')))
        model_load_seconds=time.perf_counter()-start
        start=time.perf_counter()
        x=raw.copy()
        if name.endswith('ratios'):
            x['amount_balance_ratio']=x.amount_usd/x.account_balance_usd.where(x.account_balance_usd>0)
            x['amount_per_card_month']=x.amount_usd/(x.card_age_months+1)
            x['amount_x_txn_count']=x.amount_usd*x.txn_count_last_24h
            x['retry_x_txn_count']=x.cvv_retry_count*x.txn_count_last_24h
            x['velocity_x_merchant_risk']=x.velocity_score*x.merchant_risk_score
            x['log_amount']=np.log1p(x.amount_usd.clip(lower=0))
        x=x.replace([np.inf,-np.inf],np.nan)
        cats=x.select_dtypes(include=['object','str','category']).columns.tolist()
        if name.startswith('cat'):
            for c in cats: x[c]=x[c].astype(str).fillna('__MISSING__')
            for c in x.select_dtypes(include='bool').columns: x[c]=x[c].astype(int)
            predict=lambda data,threads:model.predict_proba(data,thread_count=threads)[:,1]
            small=x.iloc[:50000]
        else:
            for c,vocab in zip(cats,model.pandas_categorical):
                x[c]=pd.Categorical(x[c],categories=vocab)
            x=x.loc[:,model.feature_name()]
            predict=lambda data,threads:model.predict(data,num_threads=threads)
            small=x.iloc[:50000]
        preprocess_seconds=time.perf_counter()-start
        predict(x.iloc[:1000],4)
        durations=[]
        for _ in range(3):
            start=time.perf_counter(); p=predict(x,4); durations.append(time.perf_counter()-start)
        expected=np.load(EXP/('full_test_'+name+'.npy'))
        assert np.allclose(p,expected,atol=1e-12,rtol=0),name
        predictions[name]=p
        item={'name':name,'model_load_seconds':model_load_seconds,'preprocess_seconds':preprocess_seconds,'batch_predict_seconds':durations,'median_batch_predict_seconds':statistics.median(durations),'rows_per_second':rows/statistics.median(durations),'file_bytes':(EXP/(name+('.cbm' if name.startswith('cat') else '.txt'))).stat().st_size}
        threading={}
        for threads in [1,4]:
            times=[]
            for _ in range(3):
                start=time.perf_counter(); predict(small,threads); times.append(time.perf_counter()-start)
            threading[str(threads)]=statistics.median(times)
        item['median_50k_seconds_by_threads']=threading
        if name.startswith('cat'):
            start=time.perf_counter()
            pool=Pool(x,cat_features=cats,thread_count=4)
            item['pool_build_seconds']=time.perf_counter()-start
            pool_times=[]
            for _ in range(3):
                start=time.perf_counter(); q=predict(pool,4); pool_times.append(time.perf_counter()-start)
            assert np.allclose(q,p,atol=1e-12,rtol=0)
            item['median_prebuilt_pool_predict_seconds']=statistics.median(pool_times)
            item['resolved_training_params']={k:model.get_all_params().get(k) for k in ['boosting_type','bootstrap_type','one_hot_max_size','max_ctr_complexity','leaf_estimation_iterations','border_count','depth']}
        item['recorded_fit_seconds_total']=sum(r['seconds'] for r in records if r['name']==name)
        report['models'].append(item)
        print(json.dumps(item,ensure_ascii=False),flush=True)
    start=time.perf_counter()
    blended=sum(predictions[n]*w for n,w in selection['selected_weights'].items())
    report['blend_seconds']=time.perf_counter()-start
    reference=pd.read_csv(OUT/'submission_v2_recommended.csv').is_fraud.to_numpy()
    assert np.allclose(reference,blended,atol=1e-12,rtol=0)
    report['total_warm_batch_seconds_sequential']=sum(r['median_batch_predict_seconds'] for r in report['models'])+report['blend_seconds']
    report['total_models_bytes']=sum(r['file_bytes'] for r in report['models'])
    report['prediction_matches_current_submission']=True
    (OUT/'inference_benchmark_v2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE '+json.dumps({k:v for k,v in report.items() if k!='models'},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
