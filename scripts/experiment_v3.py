'Target 0.96: original-data, additive and exact-value experiments.\nIdeas: https://www.kaggle.com/code/alexchilton/credit-card-fraud-lb-0-9542-shallow-tree-stack\nIndependent implementation; no third-party OOF predictions or submissions used.'

def main():
    import os
    if __package__:
        from .project_runtime import prepare_runtime,experiment_dir
    else:
        from project_runtime import prepare_runtime,experiment_dir
    RUNTIME_ROOT,DATA_DIR=prepare_runtime('v3',__file__)
    from pathlib import Path
    import sys,os,json,time,hashlib,warnings
    ROOT=RUNTIME_ROOT
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split,StratifiedKFold
    from sklearn.preprocessing import OneHotEncoder,SplineTransformer,TargetEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy import sparse
    from scipy.special import expit,logit
    from scipy.stats import rankdata
    warnings.filterwarnings('ignore',category=FutureWarning)
    EXP=experiment_dir(ROOT,'v3')
    EXP.mkdir(exist_ok=True)
    OUT=ROOT/'outputs'
    SCRIPT_VERSION='v3.1'
    THREADS=int(os.environ.get("KAGGLE_THREADS","8"))
    def say(s): print(time.strftime('%H:%M:%S')+' '+s,flush=True)
    def jsave(p,v):
        temp=p.with_suffix('.tmp'); temp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8'); temp.replace(p)
    train=pd.read_csv(DATA_DIR/'train.csv')
    test=pd.read_csv(DATA_DIR/'test.csv')
    y=train.is_fraud.astype(int).to_numpy()
    raw=train.drop(columns=['id','is_fraud'])
    rt=test.drop(columns='id')
    catcols=raw.select_dtypes(include=['object','str']).columns.tolist()
    boolcols=raw.select_dtypes(include='bool').columns.tolist()
    numcols=[c for c in raw if c not in catcols+boolcols]
    fit_idx,valid_idx=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=y)
    folds=list(StratifiedKFold(3,shuffle=True,random_state=2026).split(raw,y))
    np.savez_compressed(EXP/'split_indices.npz',holdout_train=fit_idx,holdout_valid=valid_idx,**{f'fold_{i}':v for i,(t,v) in enumerate(folds)})
    data_hash=hashlib.sha256((DATA_DIR/'train.csv').read_bytes()).hexdigest()
    records=json.loads((EXP/'results.json').read_text(encoding='utf-8')) if (EXP/'results.json').exists() else []
    # Existing labels only used for diagnostics; all predictive encodings below are fold-fitted.
    previous=experiment_dir(ROOT,'v2')
    old_weights=json.loads((previous/'selection.json').read_text(encoding='utf-8'))['selected_weights']
    old_names=list(old_weights)
    old_oof={n:np.load(previous/('oof_'+n+'.npy')) for n in old_names}
    old_blend=sum(old_oof[n]*old_weights[n] for n in old_names)
    profile={'target_public_auc':.96,'current_public_auc':.94743,'gap':.01257,'leaderboard_top_observed':.95960,'leaderboard_observed_date':'2026-11-17','leaderboard_source':'https://www.kaggle.com/competitions/playground-series-s6e11/leaderboard','categorical_unique':{c:int(raw[c].nunique()) for c in catcols},'numeric_unique':{c:int(raw[c].nunique()) for c in numcols},'old_blend_oof_auc':float(roc_auc_score(y,old_blend)),'old_prediction_correlation':np.corrcoef(np.column_stack(list(old_oof.values())),rowvar=False).tolist(),'positive_prediction_quantiles':np.quantile(old_blend[y==1],[0,.1,.25,.5,.75,.9,1]).tolist()}
    jsave(EXP/'diagnostics.json',profile)

    # Row-local digit and magnitude features, independent of all labels.
    def digit_features(df):
        z=pd.DataFrame(index=df.index)
        for c in ['amount_usd','account_balance_usd','hours_since_last_txn','distance_from_home_km','velocity_score','merchant_risk_score']:
            scaled=np.rint(df[c].to_numpy()*100).astype(np.int64)
            z['decimal_1_'+c]=((scaled//10)%10).astype(np.int8)
            z['decimal_2_'+c]=(scaled%10).astype(np.int8)
        return z
    D=digit_features(raw); DT=digit_features(rt)

    # Label-free identities for exact numeric values, rounded groups and six selected interactions.
    both=pd.concat([raw,rt],ignore_index=True)
    K={}
    for c in raw:
        K[c]=pd.factorize(both[c],sort=False)[0].astype(np.int32)
    for c in ['amount_usd','hours_since_last_txn','distance_from_home_km','account_balance_usd']:
        for div in [1,10]:
            K[c+'_rounded_'+str(div)]=pd.factorize(np.rint(both[c].to_numpy()/div).astype(np.int64),sort=False)[0].astype(np.int32)
    for a,b in [('cvv_retry_count','auth_method'),('is_foreign_transaction','ip_country_mismatch'),('used_vpn','ip_country_mismatch'),('merchant_category','is_new_merchant'),('auth_method','channel'),('cvv_retry_count','txn_count_last_24h')]:
        K[a+'__'+b]=pd.factorize(pd.MultiIndex.from_frame(both[[a,b]]),sort=False)[0].astype(np.int32)
    keys=pd.DataFrame(K)
    KT=keys.iloc[len(raw):].reset_index(drop=True); keys=keys.iloc[:len(raw)].reset_index(drop=True)
    del both,K

    # Public source data: author CC0, and explicitly linked by competition host.
    orig=pd.read_csv(DATA_DIR/'original/credit_card_fraud_2026.csv')
    orig=orig.drop(columns=['transaction_id','id'],errors='ignore')
    assert set(raw.columns).issubset(orig.columns)
    oy=orig.is_fraud.astype(int)
    prior=float(oy.mean())
    G=pd.DataFrame(index=raw.index); GT=pd.DataFrame(index=rt.index)
    for c in ['merchant_risk_score','velocity_score','customer_age','auth_method','merchant_category',*boolcols]:
        stats=orig.groupby(c).is_fraud.agg(['sum','size'])
        smoothed=(stats['sum']+20*prior)/(stats['size']+20)
        G['original_rate_'+c]=raw[c].map(smoothed).fillna(prior).astype(np.float32)
        GT['original_rate_'+c]=rt[c].map(smoothed).fillna(prior).astype(np.float32)

    def lgb_format(a,b):
        a,b=a.copy(),b.copy()
        for c in catcols:
            vocab=sorted(a[c].dropna().unique().tolist())
            a[c]=pd.Categorical(a[c],categories=vocab)
            b[c]=pd.Categorical(b[c],categories=vocab)
        return a,b
    oa,ob=lgb_format(orig[raw.columns],pd.concat([raw,rt],ignore_index=True))
    om=lgb.LGBMClassifier(n_estimators=500,learning_rate=.03,num_leaves=4,min_child_samples=50,reg_lambda=10,n_jobs=THREADS,verbosity=-1,random_state=42)
    om.fit(oa,oy)
    op=om.predict_proba(ob)[:,1]
    G['original_model']=op[:len(raw)].astype(np.float32)
    GT['original_model']=op[len(raw):].astype(np.float32)
    del oa,ob,om,op

    B=dict(objective='binary',learning_rate=.03,num_leaves=4,min_child_samples=200,reg_lambda=10.,max_bin=127,colsample_bytree=1.,random_state=42,n_jobs=THREADS,verbosity=-1)
    C=dict(loss_function='Logloss',eval_metric='AUC',depth=3,learning_rate=.04,l2_leaf_reg=10.,task_type=os.environ.get('KAGGLE_DEVICE','GPU').upper(),devices='0' if os.environ.get('KAGGLE_DEVICE','GPU').upper()=='GPU' else None,bootstrap_type='Bernoulli',subsample=.8,boosting_type='Plain',max_ctr_complexity=2,border_count=128,thread_count=THREADS,random_seed=42,allow_writing_files=False)
    X=dict(objective='binary:logistic',eval_metric='auc',tree_method='hist',device='cuda' if os.environ.get('KAGGLE_DEVICE','GPU').upper()=='GPU' else 'cpu',max_depth=2,learning_rate=.04,min_child_weight=10.,reg_lambda=10.,max_bin=128,subsample=.9,colsample_bytree=1.,enable_categorical=True,n_jobs=THREADS,random_state=42)
    configs={
     'lgb_leaf4_raw':dict(kind='lgb',mode='raw',params=B),
     'lgb_leaf4_digits':dict(kind='lgb',mode='digits',params=B),
     'lgb_leaf4_encoded':dict(kind='lgb',mode='encoded',params=B),
     'lgb_leaf4_full':dict(kind='lgb',mode='full',params=B),
     'cat_gpu_depth3':dict(kind='cat',mode='raw',params=C),
     'cat_gpu_depth3_full':dict(kind='cat',mode='full',params=C),
     'xgb_depth2_raw':dict(kind='xgb',mode='raw',params=X),
     'xgb_depth2_full':dict(kind='xgb',mode='full',params=X),
     'glm_spline':dict(kind='glm',mode='spline',params=dict(C=1.,max_iter=300,solver='liblinear',tol=1e-5,random_state=42)),
     'glm_exact_values':dict(kind='glm',mode='exact',params=dict(C=.1,max_iter=300,solver='liblinear',tol=1e-5,random_state=42)),
    }

    def matrix(idx, val, mode):
        x=raw.iloc[idx].reset_index(drop=True).copy()
        z=(raw.iloc[val] if val is not None else rt).reset_index(drop=True).copy()
        if mode in ('digits','full'):
            x=pd.concat([x,D.iloc[idx].reset_index(drop=True)],axis=1)
            z=pd.concat([z,(D.iloc[val] if val is not None else DT).reset_index(drop=True)],axis=1)
        if mode in ('encoded','full'):
            tag=hashlib.sha256(np.asarray(idx,dtype=np.int64).tobytes()+(b'test' if val is None else np.asarray(val,dtype=np.int64).tobytes())).hexdigest()[:16]
            cache=EXP/('enc_'+tag+'.npz')
            if cache.exists():
                enc=np.load(cache); a,b=enc['train'],enc['valid']
            else:
                enc=TargetEncoder(target_type='binary',smooth='auto',cv=3,shuffle=True,random_state=42)
                a=enc.fit_transform(keys.iloc[idx],y[idx]).astype(np.float32)
                b=enc.transform(keys.iloc[val] if val is not None else KT).astype(np.float32)
                np.savez_compressed(cache,train=a,valid=b)
            x=pd.concat([x,pd.DataFrame(a,columns=['te_'+c for c in keys])],axis=1)
            z=pd.concat([z,pd.DataFrame(b,columns=['te_'+c for c in keys])],axis=1)
        if mode=='full':
            x=pd.concat([x,G.iloc[idx].reset_index(drop=True)],axis=1)
            z=pd.concat([z,(G.iloc[val] if val is not None else GT).reset_index(drop=True)],axis=1)
        return x,z

    def glm_matrix(idx,val,mode):
        a=raw.iloc[idx]; b=raw.iloc[val] if val is not None else rt
        if mode=='exact':
            enc=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float64)
            return enc.fit_transform(a.astype(str)),enc.transform(b.astype(str))
        continuous=['amount_usd','hours_since_last_txn','distance_from_home_km','account_balance_usd','velocity_score','merchant_risk_score','customer_age','card_age_months']
        discrete=[c for c in raw if c not in continuous]
        spl=SplineTransformer(n_knots=7,degree=3,knots='quantile',include_bias=False,extrapolation='constant',sparse_output=True)
        enc=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float64)
        aa=sparse.hstack([spl.fit_transform(a[continuous]),enc.fit_transform(a[discrete].astype(str))],format='csr')
        bb=sparse.hstack([spl.transform(b[continuous]),enc.transform(b[discrete].astype(str))],format='csr')
        return aa,bb

    def fit(conf,idx,val=None,iterations=None):
        kind=conf['kind']
        if kind=='glm':
            a,b=glm_matrix(idx,val,conf['mode'])
            m=LogisticRegression(**conf['params']); m.fit(a,y[idx])
            p=m.predict_proba(b)[:,1]; rounds=int(m.n_iter_[0])
        else:
            a,b=matrix(idx,val,conf['mode'])
            if kind in ('lgb','xgb'):
                a,b=lgb_format(a,b)
            else:
                for c in catcols:
                    a[c]=a[c].astype(str); b[c]=b[c].astype(str)
                for c in boolcols:
                    a[c]=a[c].astype(int); b[c]=b[c].astype(int)
            if kind=='lgb':
                m=lgb.LGBMClassifier(n_estimators=iterations or 6000,**conf['params'])
                kw={} if val is None else dict(eval_set=[(b,y[val])],eval_metric='auc',callbacks=[lgb.early_stopping(200,first_metric_only=True,verbose=False)])
                m.fit(a,y[idx],**kw); rounds=int(m.best_iteration_) if val is not None else int(iterations)
            elif kind=='cat':
                m=CatBoostClassifier(iterations=iterations or 4000,**conf['params'])
                kw=dict(cat_features=catcols,verbose=500)
                if val is not None: kw.update(eval_set=(b,y[val]),early_stopping_rounds=200)
                m.fit(a,y[idx],**kw); rounds=m.get_best_iteration()+1 if val is not None else int(iterations)
            else:
                kw=dict(n_estimators=iterations or 6000,**conf['params'])
                if val is not None: kw['early_stopping_rounds']=200
                m=xgb.XGBClassifier(**kw)
                m.fit(a,y[idx],**({} if val is None else dict(eval_set=[(b,y[val])],verbose=False)))
                rounds=m.best_iteration+1 if val is not None else int(iterations)
            p=m.predict_proba(b)[:,1]
        assert np.isfinite(p).all() and ((0<=p)&(p<=1)).all()
        return p,int(rounds),m

    def run(name,stage,fold=None):
        conf=configs[name]
        config_hash=hashlib.sha256(json.dumps(dict(conf=conf,version=SCRIPT_VERSION,data=data_hash,fold=fold,stage=stage),sort_keys=True).encode()).hexdigest()[:16]
        key=stage+'_'+name+(('_fold'+str(fold)) if fold is not None else '')
        file=EXP/(key+'_'+config_hash+'.npy')
        old=next((r for r in records if r['key']==key and r['hash']==config_hash),None)
        if old and file.exists(): return old,np.load(file)
        idx,val=(fit_idx,valid_idx) if fold is None else folds[fold]
        say('START '+key)
        start=time.perf_counter()
        p,rounds,m=fit(conf,idx,val)
        r=dict(key=key,hash=config_hash,name=name,stage=stage,fold=fold,auc=float(roc_auc_score(y[val],p)),iterations=rounds,seconds=round(time.perf_counter()-start,2),config=conf,prediction_file=str(file))
        np.save(file,p); records.append(r); jsave(EXP/'results.json',records)
        say('DONE '+key+' AUC='+f'{r["auc"]:.8f}'+' rounds='+str(rounds)+' seconds='+str(r['seconds']))
        return r,p

    say('STAGE SCREEN: additive, exact-value and original-data candidates')
    screen=[]
    for name in configs:
        try:
            r,p=run(name,'screen'); screen.append(r)
        except Exception as error:
            say('FAILED '+name+': '+repr(error))
            jsave(EXP/(name+'_error.json'),dict(error=repr(error),config=configs[name]))
            if configs[name]['kind'] in ('cat','xgb') and ('cuda' in str(error).lower() or 'gpu' in str(error).lower()):
                continue
            raise

    # One representative per learning family, plus the overall best distinct alternative.
    shortlist=[]
    for family in ['lgb','cat','xgb','glm']:
        group=[r for r in screen if r['config']['kind']==family]
        if group: shortlist.append(max(group,key=lambda r:r['auc'])['name'])
    # Limit full CV to candidates with some promise in score or a distinct learning bias.
    shortlist=[n for n in shortlist if next(r['auc'] for r in screen if r['name']==n)>=.943]
    best_two=sorted(screen,key=lambda r:r['auc'],reverse=True)[:2]
    for r in best_two:
        if r['name'] not in shortlist: shortlist.append(r['name'])
    shortlist=shortlist[:5]
    say('SHORTLIST '+json.dumps(shortlist))
    cv_results={}; preds={}
    for name in shortlist:
        pred=np.full(len(y),np.nan); rs=[]
        for fold,(idx,val) in enumerate(folds):
            r,p=run(name,'cv',fold); pred[val]=p; rs.append(r)
        fs=[r['auc'] for r in rs]
        result=dict(name=name,fold_auc=fs,mean_auc=float(np.mean(fs)),pooled_auc=float(roc_auc_score(y,pred)),iterations=[r['iterations'] for r in rs],config=configs[name])
        cv_results[name]=result; preds[name]=pred
        np.save(EXP/('oof_'+name+'.npy'),pred)
        jsave(EXP/'cv_summary.json',cv_results)
        say('CV '+name+' '+json.dumps(result))

    # Coarse development blends. These use validation labels for selection, not an untouched test.
    best_single=max(cv_results,key=lambda n:cv_results[n]['mean_auc'])
    candidates=[]
    def assess(name,weights):
        p=sum((old_blend if n=='v2' else preds[n])*w for n,w in weights.items())
        fs=[float(roc_auc_score(y[v],p[v])) for _,v in folds]
        return dict(name=name,weights=weights,fold_auc=fs,mean_auc=float(np.mean(fs)),pooled_auc=float(roc_auc_score(y,p)))
    for n in preds:
        candidates.append(assess(n,{n:1.}))
        for w in [.25,.5,.75]: candidates.append(assess(n+'_with_v2_'+str(w),{n:w,'v2':1-w}))
    ordered=sorted(cv_results,key=lambda n:cv_results[n]['mean_auc'],reverse=True)[:3]
    for a in range(len(ordered)):
        for b in range(a+1,len(ordered)):
            pair=[ordered[a],ordered[b]]
            candidates.append(assess('blend_'+pair[0]+'_'+pair[1],{n:.5 for n in pair}))
    if len(ordered)>=3: candidates.append(assess('equal_three_v3',{n:1/3 for n in ordered}))
    selected=max(candidates,key=lambda r:r['mean_auc'])
    v2_scores=[float(roc_auc_score(y[v],old_blend[v])) for _,v in folds]
    selection=dict(target_public_auc=.96,current_public_auc=.94743,baseline_v2_cv=float(np.mean(v2_scores)),best_single=best_single,selected=selected,candidates=candidates,shortlist=shortlist,notes='Development selection reuses CV labels. Public target not confirmed until a Kaggle submission is scored. GPU methods have different defaults than CPU. Original public data is CC0 and officially linked.')
    jsave(EXP/'selection.json',selection)
    say('SELECTED '+json.dumps(selected))
    full={}
    for name in dict.fromkeys([best_single,*[n for n in selected['weights'] if n!='v2']]):
        cache=EXP/('full_test_'+name+'.npy')
        if cache.exists(): p=np.load(cache)
        else:
            it=int(np.median(cv_results[name]['iterations']))
            say('FULL REFIT '+name+' rounds='+str(it))
            p,_,m=fit(configs[name],np.arange(len(y)),iterations=it)
            np.save(cache,p)
            if configs[name]['kind']=='lgb':m.booster_.save_model(str(EXP/(name+'.txt')))
            elif configs[name]['kind']=='cat':m.save_model(str(EXP/(name+'.cbm')))
            elif configs[name]['kind']=='xgb':m.save_model(str(EXP/(name+'.ubj')))
        full[name]=p
    v2_test=pd.read_csv(OUT/'submission_v2_recommended.csv').is_fraud.to_numpy()
    recommended=sum((v2_test if n=='v2' else full[n])*w for n,w in selected['weights'].items())
    sample=pd.read_csv(DATA_DIR/'sample_submission.csv')
    for filename,p in [('submission_v3_recommended.csv',recommended),('submission_v3_best_single.csv',full[best_single])]:
        assert len(p)==len(test) and np.isfinite(p).all() and ((0<=p)&(p<=1)).all()
        df=pd.DataFrame({'id':test.id,'is_fraud':p}); df.to_csv(OUT/filename,index=False)
        check=pd.read_csv(OUT/filename)
        assert check.columns.tolist()==sample.columns.tolist() and check.shape==sample.shape and np.array_equal(check.id,sample.id)
    rows=[{k:r.get(k) for k in ['name','stage','fold','auc','iterations','seconds']} for r in records]
    pd.DataFrame(rows).to_csv(OUT/'experiment_results_v3.csv',index=False)
    lines=['# 第三轮：向公开 AUC 0.96 推进','','目标尚未达到；当前用户报告公开成绩为 0.94743。','公开榜首在 2026-11-17 观察为 0.95960，0.96 超过当时榜首。','','## 来源与方法','','- 比赛原始数据： https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026 ，CC0；比赛明确允许公开外部数据。','- 方法线索： https://www.kaggle.com/code/alexchilton/credit-card-fraud-lb-0-9542-shallow-tree-stack （Apache 2.0）；独立实现，未使用其他作者的预测文件。','- 新分支：4 叶 LightGBM、3 层 CatBoost、2 层 XGBoost、样条逻辑回归、精确取值独热逻辑回归。','- 新特征：逐行小数位、折内交叉拟合的目标统计、原始公开数据统计和原始数据模型的预测。','- 目标编码仅在外层训练部分拟合，训练部分再做内部交叉拟合。','- GPU 分支与 CPU 默认设置不同，作为新候选验证。','','## 三折复核','','| 模型 | 第1折 | 第2折 | 第3折 | 平均 AUC |','|---|---:|---:|---:|---:|']
    for n,r in cv_results.items(): lines.append('| '+n+' | '+' | '.join(f'{a:.6f}' for a in [*r['fold_auc'],r['mean_auc']])+' |')
    lines+=['',f'- 上版同一划分平均 AUC：{np.mean(v2_scores):.8f}',f'- 新推荐方案：{selected["name"]}',f'- 推荐开发 CV 均值：{selected["mean_auc"]:.8f}',f'- 组合权重：{json.dumps(selected["weights"])}','- 验证标签用于方案选择，这不是独立测试成绩；公开和私榜成绩均需另行确认。','- 新文件 submission_v3_recommended.csv 已检查格式，原版与第二版文件保留不变。']
    (OUT/'experiment_report_v3.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'submission_description_v3.txt').write_text('Additive-model and exact-value feature experiments, independently implemented with fold-fitted target encodings and public CC0 original-data features. Selected weights: '+json.dumps(selected['weights'])+'. Development 3-fold mean ROC-AUC: '+f'{selected["mean_auc"]:.6f}'+'. Public score pending.',encoding='utf-8')
    jsave(EXP/'final.json',dict(selection=selection,cv=cv_results,public_target_reached=False,submission='submission_v3_recommended.csv'))
    say('ALL DONE '+json.dumps(dict(cv_auc=selected['mean_auc'],submission=str(OUT/'submission_v3_recommended.csv'),public_target_reached=False)))


if __name__ == "__main__":
    main()
