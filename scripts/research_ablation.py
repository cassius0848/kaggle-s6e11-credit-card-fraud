"""Controlled feature ablations; importing does not train."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
if __package__:
    from .research_features import FeatureStore
else:
    from research_features import FeatureStore

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args(); store=FeatureStore(args.workspace);root=store.root
    cache=root/'work/research_v4/ablation';cache.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(store.fingerprint.encode()+Path(__file__).read_bytes()).hexdigest()
    params=dict(objective='binary',learning_rate=.03,num_leaves=4,min_child_samples=200,reg_lambda=10.,max_bin=127,colsample_bytree=1.,random_state=42,n_jobs=8,verbosity=-1)
    records=[]
    def fit(mode,fold=None):
        key='screen_'+mode if fold is None else mode+'_fold'+str(fold)
        meta=cache/(key+'.json');array=cache/(key+'.npy')
        if meta.exists() and array.exists():
            r=json.loads(meta.read_text(encoding='utf-8'))
            if r['signature']==signature:return r,np.load(array)
        idx,val=(store.fit_idx,store.val_idx) if fold is None else store.folds[fold]
        print(time.strftime('%H:%M:%S'),'START',key,flush=True);start=time.perf_counter()
        a,b=store.frames(idx,val,mode);a,b=store.lgb_format(a,b)
        model=lgb.LGBMClassifier(n_estimators=6000,**params)
        model.fit(a,store.y[idx],eval_set=[(b,store.y[val])],eval_metric='auc',callbacks=[lgb.early_stopping(200,first_metric_only=True,verbose=False)])
        p=model.predict_proba(b)[:,1]
        from sklearn.metrics import roc_auc_score
        r=dict(name=key,mode=mode,fold=fold,auc=float(roc_auc_score(store.y[val],p)),iterations=int(model.best_iteration_),seconds=round(time.perf_counter()-start,2),features=a.shape[1],signature=signature,params=params)
        np.save(array,p);meta.write_text(json.dumps(r,indent=2),encoding='utf-8')
        print(time.strftime('%H:%M:%S'),'DONE',key,'AUC',round(r['auc'],8),'seconds',r['seconds'],flush=True)
        return r,p
    modes=['raw','digits','original','encoded','encoded_digits','encoded_original','all']
    for mode in modes:
        r,p=fit(mode);records.append(r)
    scores={r['mode']:r['auc'] for r in records}
    # Existing encoded control is retained even if all additions hurt.
    best=max(modes,key=lambda m:scores[m])
    cv={}
    for mode in list(dict.fromkeys(['encoded',best])):
        pred=np.full(len(store.y),np.nan);fold_records=[]
        for fold,(_,val) in enumerate(store.folds):
            r,p=fit(mode,fold);pred[val]=p;fold_records.append(r)
        from sklearn.metrics import roc_auc_score
        cv[mode]=dict(mean_auc=float(np.mean([r['auc'] for r in fold_records])),fold_auc=[r['auc'] for r in fold_records],pooled_auc=float(roc_auc_score(store.y,pred)),iterations=[r['iterations'] for r in fold_records])
        np.save(cache/('oof_'+mode+'.npy'),pred)
        print('CV',mode,json.dumps(cv[mode]),flush=True)
    (cache/'summary.json').write_text(json.dumps(dict(signature=signature,screen=records,cv=cv,best_mode=best),indent=2),encoding='utf-8')
    pd.DataFrame([{k:r[k] for k in ['mode','auc','iterations','seconds','features']} for r in records]).to_csv(root/'outputs/research_v4_ablation.csv',index=False)
    print('ABLATION DONE',best,flush=True)
if __name__=='__main__':main()
