'Reproducible, resumable Kaggle experiments. Run with Python 3.14.\nAll runtime files stay in the project workspace.'

def main():
    import os
    if __package__:
        from .project_runtime import prepare_runtime,experiment_dir
    else:
        from project_runtime import prepare_runtime,experiment_dir
    RUNTIME_ROOT,DATA_DIR=prepare_runtime('v2',__file__)
    from pathlib import Path
    import sys, os, json, time, hashlib, itertools, platform
    ROOT = RUNTIME_ROOT
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from catboost import CatBoostClassifier
    import catboost, sklearn
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import roc_auc_score

    EXP = experiment_dir(ROOT,'v2')
    EXP.mkdir(exist_ok=True)
    OUT = ROOT/'outputs'
    RESULTS = EXP/'results.json'
    SEED = 42
    THREADS = int(os.environ.get("KAGGLE_THREADS","4"))

    def say(s):
        print(time.strftime('%H:%M:%S')+' '+s, flush=True)

    def save_json(p, obj):
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(p)

    records = json.loads(RESULTS.read_text(encoding='utf-8')) if RESULTS.exists() else []
    train = pd.read_csv(DATA_DIR/'train.csv')
    test = pd.read_csv(DATA_DIR/'test.csv')
    sample = pd.read_csv(DATA_DIR/'sample_submission.csv')
    y = train.is_fraud.astype(int).to_numpy()
    raw = train.drop(columns=['id','is_fraud'])
    raw_test = test.drop(columns='id')
    assert raw.columns.tolist() == raw_test.columns.tolist()
    assert np.array_equal(test.id.to_numpy(), sample.id.to_numpy())
    assert set(np.unique(y)) == {0,1}
    fit_idx, valid_idx = train_test_split(np.arange(len(y)),test_size=.2,random_state=SEED,stratify=y)
    folds = list(StratifiedKFold(3,shuffle=True,random_state=2026).split(raw,y))
    np.savez_compressed(EXP/'splits.npz',holdout_train=fit_idx,holdout_valid=valid_idx,**{f'fold_{i}_valid':b for i,(a,b) in enumerate(folds)})

    base = dict(objective='binary',learning_rate=.05,num_leaves=31,max_depth=-1,min_child_samples=100,colsample_bytree=.9,reg_lambda=2.,n_jobs=THREADS,random_state=SEED,verbosity=-1)
    configs = {
     'lgb_baseline':dict(kind='lgb',params=base,features='raw'),
     'lgb_shallow':dict(kind='lgb',params={**base,'num_leaves':15,'min_child_samples':200,'learning_rate':.03,'reg_lambda':10.},features='raw'),
     'lgb_regularized':dict(kind='lgb',params={**base,'learning_rate':.03,'reg_lambda':10.,'reg_alpha':.5,'min_child_samples':150},features='raw'),
     'lgb_flexible':dict(kind='lgb',params={**base,'num_leaves':63,'learning_rate':.03,'reg_lambda':5.,'subsample':.8,'subsample_freq':1},features='raw'),
     'cat_depth6':dict(kind='cat',params=dict(loss_function='Logloss',eval_metric='AUC',depth=6,learning_rate=.06,l2_leaf_reg=5.,random_seed=SEED,thread_count=THREADS,allow_writing_files=False),features='raw')}


    def features(df, group):
        z = df.copy()
        if group in ('ratios','combined'):
            z['amount_balance_ratio'] = z.amount_usd / z.account_balance_usd.where(z.account_balance_usd>0)
            z['amount_per_card_month'] = z.amount_usd / (z.card_age_months+1)
            z['amount_x_txn_count'] = z.amount_usd * z.txn_count_last_24h
            z['retry_x_txn_count'] = z.cvv_retry_count * z.txn_count_last_24h
            z['velocity_x_merchant_risk'] = z.velocity_score * z.merchant_risk_score
            z['log_amount'] = np.log1p(z.amount_usd.clip(lower=0))
        if group in ('context','combined'):
            flags=['is_foreign_transaction','is_new_merchant','used_vpn','ip_country_mismatch','billing_shipping_mismatch','is_ai_generated_scam_attempt']
            z['risk_flag_count'] = z[flags].astype(int).sum(axis=1)
            z['foreign_x_ip_mismatch'] = z.is_foreign_transaction.astype(int)*z.ip_country_mismatch.astype(int)
            z['new_merchant_x_amount'] = z.is_new_merchant.astype(int)*z.amount_usd
            z['foreign_x_distance'] = z.is_foreign_transaction.astype(int)*z.distance_from_home_km
            z['channel_device'] = z.channel.astype(str)+'|'+z.device_type.astype(str)
            z['merchant_auth'] = z.merchant_category.astype(str)+'|'+z.auth_method.astype(str)
        return z.replace([np.inf,-np.inf], np.nan)


    def prepare(a,b,kind):
        # Learn categorical vocabularies on the training fold only.
        a,b=a.copy(),b.copy()
        cats=a.select_dtypes(include=['object','str','category']).columns.tolist()
        if kind=='cat':
            for c in cats:
                a[c]=a[c].astype(str).fillna('__MISSING__')
                b[c]=b[c].astype(str).fillna('__MISSING__')
            for c in a.select_dtypes(include='bool').columns:
                a[c]=a[c].astype(int); b[c]=b[c].astype(int)
        else:
            for c in cats:
                vocabulary=sorted(a[c].dropna().unique().tolist())
                a[c]=pd.Categorical(a[c],categories=vocabulary)
                b[c]=pd.Categorical(b[c],categories=vocabulary)
        return a,b,cats


    def fit_model(conf, idx, val=None, iterations=None):
        x=features(raw,conf['features'])
        a=x.iloc[idx]
        b=x.iloc[val] if val is not None else features(raw_test,conf['features'])
        a,b,cats=prepare(a,b,conf['kind'])
        if conf['kind']=='lgb':
            model=lgb.LGBMClassifier(n_estimators=iterations or 2500,**conf['params'])
            if val is not None:
                model.fit(a,y[idx],eval_set=[(b,y[val])],eval_metric='auc',callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False)])
                rounds=model.best_iteration_
            else:
                model.fit(a,y[idx]); rounds=iterations
        else:
            model=CatBoostClassifier(iterations=iterations or 2000,**conf['params'])
            args=dict(cat_features=cats,verbose=False)
            if val is not None:
                model.fit(a,y[idx],eval_set=(b,y[val]),early_stopping_rounds=100,**args)
                rounds=model.get_best_iteration()+1
            else:
                model.fit(a,y[idx],**args); rounds=iterations
        p=model.predict_proba(b)[:,1]
        assert np.isfinite(p).all() and ((0<=p)&(p<=1)).all()
        return p,int(rounds),model


    def experiment(name, conf, stage, fold=None):
        key=f'{stage}_{name}'+(f'_fold{fold}' if fold is not None else '')
        old=next((r for r in records if r['key']==key),None)
        if old and (EXP/(key+'.npy')).exists():
            say('RESUME '+key); return old,np.load(EXP/(key+'.npy'))
        a,b=(fit_idx,valid_idx) if fold is None else folds[fold]
        say('START '+key)
        start=time.perf_counter()
        p,rounds,_=fit_model(conf,a,b)
        result=dict(key=key,name=name,stage=stage,fold=fold,auc=float(roc_auc_score(y[b],p)),iterations=rounds,seconds=round(time.perf_counter()-start,2),config=conf)
        np.save(EXP/(key+'.npy'),p)
        records.append(result); save_json(RESULTS,records)
        say('DONE '+key+' AUC='+f"{result['auc']:.8f}"+' rounds='+str(rounds)+' seconds='+str(result['seconds']))
        return result,p


    def cv(name, conf):
        oof=np.full(len(y),np.nan)
        scores=[]; rounds=[]
        for fold in range(3):
            r,p=experiment(name,conf,'cv',fold)
            oof[folds[fold][1]]=p
            scores.append(r['auc']); rounds.append(r['iterations'])
        assert np.isfinite(oof).all()
        np.save(EXP/('oof_'+name+'.npy'),oof)
        result=dict(name=name,config=conf,fold_auc=scores,mean_auc=float(np.mean(scores)),std_auc=float(np.std(scores)),pooled_auc=float(roc_auc_score(y,oof)),iterations=rounds)
        say('CV '+name+' '+json.dumps({k:result[k] for k in ['fold_auc','mean_auc','pooled_auc']}))
        return result,oof


    say('STAGE 1: preserve baseline and dataset fingerprints')
    manifest=EXP/'baseline_manifest.json'
    if not manifest.exists():
        hashes={}
        for p in [DATA_DIR/'train.csv',DATA_DIR/'test.csv',DATA_DIR/'sample_submission.csv',OUT/'submission_baseline.csv']:
            hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        save_json(manifest,dict(sha256=hashes,baseline_holdout_auc=.9451824190983399,baseline_public_auc=.94612,train_rows=len(y),test_rows=len(test),seed=SEED,cv_seed=2026,versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,lightgbm=lgb.__version__,catboost=catboost.__version__,sklearn=sklearn.__version__)))

    say('STAGE 2: fixed 80/20 holdout screening')
    screen=[]
    for name,conf in configs.items():
        r,_=experiment(name,conf,'screen'); screen.append(r)
    best_lgb=max((r for r in screen if r['config']['kind']=='lgb'),key=lambda r:r['auc'])['name']
    say('Best holdout LightGBM: '+best_lgb)

    say('STAGE 3: three-fold validation of baseline and candidates')
    cv_results={}; predictions={}
    for name in dict.fromkeys(['lgb_baseline',best_lgb,'cat_depth6']):
        r,p=cv(name,configs[name]); cv_results[name]=r; predictions[name]=p
    best_cv_lgb=max((n for n in cv_results if configs[n]['kind']=='lgb'),key=lambda n:cv_results[n]['mean_auc'])

    say('STAGE 4: feature ablations on the best cross-validated LightGBM')
    feature_screen=[]
    for group in ['ratios','context','combined']:
        name=best_cv_lgb+'_'+group
        configs[name]={**configs[best_cv_lgb],'features':group}
        r,_=experiment(name,configs[name],'screen'); feature_screen.append(r)
    feature_winner=max(feature_screen,key=lambda r:r['auc'])['name']
    r,p=cv(feature_winner,configs[feature_winner]); cv_results[feature_winner]=r; predictions[feature_winner]=p
    save_json(EXP/'cv_summary.json',cv_results)

    say('STAGE 5: small, predefined OOF blending comparison')
    # OOF = each row predicted only by a model that did not train on that row.
    # Early stopping, screening and model selection still reuse validation labels;
    # these are development scores, not an untouched/nested performance estimate.
    best_single=max(cv_results,key=lambda n:cv_results[n]['mean_auc'])
    base_cv=cv_results['lgb_baseline']
    blend_results=[]
    blend_names=list(predictions)
    for pair in itertools.combinations(blend_names,2):
        if pair[0]=='lgb_baseline' and len(blend_names)>3:
            continue
        weights={n:1/len(pair) for n in pair}
        pred=sum(predictions[n]*w for n,w in weights.items())
        fs=[float(roc_auc_score(y[v],pred[v])) for t,v in folds]
        blend_results.append(dict(name='blend_'+'_'.join(pair),weights=weights,fold_auc=fs,mean_auc=float(np.mean(fs)),pooled_auc=float(roc_auc_score(y,pred))))
    if len(predictions)>=3:
        members=list(dict.fromkeys([best_cv_lgb,'cat_depth6',feature_winner]))
        weights={n:1/len(members) for n in members}
        pred=sum(predictions[n]*w for n,w in weights.items())
        fs=[float(roc_auc_score(y[v],pred[v])) for t,v in folds]
        blend_results.append(dict(name='blend_equal_three',weights=weights,fold_auc=fs,mean_auc=float(np.mean(fs)),pooled_auc=float(roc_auc_score(y,pred))))
    winning_blend=max(blend_results,key=lambda r:r['mean_auc'])
    reference=cv_results[best_single]
    blend_ok=(winning_blend['mean_auc']>reference['mean_auc']+.0001 and winning_blend['pooled_auc']>reference['pooled_auc'] and sum(a>b for a,b in zip(winning_blend['fold_auc'],reference['fold_auc']))>=2)
    selected_weights=winning_blend['weights'] if blend_ok else {best_single:1.}
    selected_name=winning_blend['name'] if blend_ok else best_single
    selection=dict(best_single=best_single,best_single_cv=reference,blends=blend_results,blend_selected=bool(blend_ok),selected_name=selected_name,selected_weights=selected_weights,selection_note='Small development search. Not nested CV; improvement must be confirmed on Kaggle. No public-score optimization performed.')
    save_json(EXP/'selection.json',selection)

    say('STAGE 6: full-data refit and checked submission export')
    full_predictions={}
    for name in dict.fromkeys([best_single,*selected_weights]):
        path=EXP/('full_test_'+name+'.npy')
        if path.exists():
            p=np.load(path)
        else:
            iterations=int(np.median(cv_results[name]['iterations']))
            say('FULL REFIT '+name+' rounds='+str(iterations))
            p,_,model=fit_model(configs[name],np.arange(len(y)),iterations=iterations)
            np.save(path,p)
            if configs[name]['kind']=='lgb': model.booster_.save_model(str(EXP/(name+'.txt')))
            else: model.save_model(str(EXP/(name+'.cbm')))
        full_predictions[name]=p

    def export(filename,p):
        assert len(p)==len(sample) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        frame=pd.DataFrame({'id':test.id,'is_fraud':p})
        target=OUT/filename
        frame.to_csv(target,index=False)
        check=pd.read_csv(target)
        assert check.shape==sample.shape and check.columns.tolist()==sample.columns.tolist()
        assert np.array_equal(check.id,sample.id) and check.is_fraud.notna().all()
        assert check.id.is_unique
        return str(target)

    single_file=export('submission_v2_best_single.csv',full_predictions[best_single])
    recommended_p=sum(full_predictions[n]*w for n,w in selected_weights.items())
    recommended_file=export('submission_v2_recommended.csv',recommended_p)
    rows=[]
    for r in records:
        if r['stage']=='screen': rows.append(dict(experiment=r['name'],evaluation='80/20 holdout',auc=r['auc'],std_auc=None,iterations=r['iterations'],seconds=r['seconds']))
    for name,r in cv_results.items(): rows.append(dict(experiment=name,evaluation='3-fold CV mean',auc=r['mean_auc'],std_auc=r['std_auc'],iterations=int(np.median(r['iterations'])),seconds=None))
    for r in blend_results: rows.append(dict(experiment=r['name'],evaluation='OOF blend CV mean',auc=r['mean_auc'],std_auc=float(np.std(r['fold_auc'])),iterations=None,seconds=None))
    pd.DataFrame(rows).to_csv(OUT/'experiment_results_v2.csv',index=False)
    save_json(EXP/'final.json',dict(selection=selection,files=dict(single=single_file,recommended=recommended_file),baseline_cv=base_cv))
    lines=['# 信用卡欺诈预测：第二轮实验报告','','## 基准与方法','',f'- 原版单次验证 AUC：0.94518242；用户提供的公开分数：0.94612。', '- 原版预测文件保持不变，SHA-256 已保存。','- 筛选：沿用原版分层 80/20 划分，随机种子 42。','- 复核：3 折分层交叉验证，随机种子 2026，所有模型使用相同划分。','- 类别词表仅从各训练部分学习；新特征不使用标签。','- 早停、方案选择使用了验证标签，结果属于开发阶段评估，并非独立测试或嵌套交叉验证。','','## 单模型交叉验证','','| 模型 | 第1折 | 第2折 | 第3折 | 平均 AUC | 标准差 |','|---|---:|---:|---:|---:|---:|']
    for name,r in cv_results.items():
        lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in [*r['fold_auc'],r['mean_auc'],r['std_auc']])+' |')
    lines+=['','## 新特征','', '- ratios：金额/余额、金额/卡龄、金额×交易次数、安全码重试×交易次数、速度分数×商户风险分数、金额对数。','- context：风险标志总数、境外×IP不匹配、新商户×金额、境外×距离、渠道与设备组合、商户与验证方式组合。',f'- 在固定验证集筛选后，选 {feature_winner} 做 3 折复核。','','## 融合比较','','| 方案 | 平均 AUC | 相对最佳单模型 |','|---|---:|---:|']
    for r in blend_results: lines.append(f"| {r['name']} | {r['mean_auc']:.6f} | {r['mean_auc']-reference['mean_auc']:+.6f} |")
    lines+=['','## 推荐与提交','',f'- 最佳单模型：{best_single}。',f'- 推荐方案：{selected_name}。',f'- 推荐权重：{json.dumps(selected_weights)}。',f'- 推荐开发 CV 均值：{(winning_blend if blend_ok else reference)["mean_auc"]:.8f}。',f'- 与原版同一套 3 折均值的差值：{(winning_blend if blend_ok else reference)["mean_auc"]-base_cv["mean_auc"]:+.8f}。','- 只有融合的均值至少提升 0.0001、整体 OOF AUC 提升、且至少两折优于最佳单模型时，才推荐融合。','- 最终每个组成模型在全部训练数据上重训，轮次取各折最佳轮次的中位数。','- submission_v2_recommended.csv：建议下一次上传；submission_v2_best_single.csv：单模型备用。','- 所有提交均检查列名、299984 行、编号顺序、唯一编号、缺失值和 0～1 概率范围。','- 尚未提交 Kaggle；公开分数和排名提升均未知。','- experiment_results_v2.csv 包含全部筛选和复核结果。','','## 重现','', '- 从项目目录运行：python work/experiment_v2.py。依赖放在 work/ml_packages。','- work/experiments_v2 保存参数、数据指纹、划分、验证预测、完整模型及选择记录。','- 脚本支持继续已完成实验；如要改变代码或参数，请使用新的实验文件夹，避免沿用旧缓存。']
    (OUT/'experiment_report_v2.md').write_text('\n'.join(lines),encoding='utf-8')
    say('ALL DONE '+json.dumps(dict(recommended=selected_name,cv_mean=(winning_blend if blend_ok else reference)['mean_auc'],baseline_cv=base_cv['mean_auc'],files=[recommended_file,single_file]),ensure_ascii=False))


if __name__ == "__main__":
    main()
