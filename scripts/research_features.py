"""Fold-safe features for research experiments; no work is run on import."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold

class FeatureStore:
    def __init__(self,workspace):
        self.root=Path(workspace).resolve()
        self.cache=self.root/'work/research_v4/features'
        self.cache.mkdir(parents=True,exist_ok=True)
        self.train=pd.read_csv(self.root/'data/train.csv')
        self.test=pd.read_csv(self.root/'data/test.csv')
        self.y=self.train.is_fraud.astype(int).to_numpy()
        self.raw=self.train.drop(columns=['id','is_fraud'])
        self.test_raw=self.test.drop(columns='id')
        assert self.raw.columns.tolist()==self.test_raw.columns.tolist()
        self.cats=self.raw.select_dtypes(include=['object','str']).columns.tolist()
        self.bools=self.raw.select_dtypes(include='bool').columns.tolist()
        self.nums=[c for c in self.raw if c not in self.cats+self.bools]
        self.fit_idx,self.val_idx=train_test_split(np.arange(len(self.y)),test_size=.2,random_state=42,stratify=self.y)
        self.folds=list(StratifiedKFold(3,shuffle=True,random_state=2026).split(self.raw,self.y))
        self.fingerprint=hashlib.sha256(b''.join(hashlib.sha256((self.root/p).read_bytes()).digest() for p in ['data/train.csv','data/test.csv','data/original/credit_card_fraud_2026.csv'])+Path(__file__).read_bytes()).hexdigest()
        self._keys=None; self._orig=None
    def digit(self,df):
        d={}
        for c in ['amount_usd','account_balance_usd','hours_since_last_txn','distance_from_home_km','velocity_score','merchant_risk_score']:
            z=np.rint(df[c].to_numpy()*100).astype(np.int64)
            d['decimal_1_'+c]=((z//10)%10).astype(np.int8)
            d['decimal_2_'+c]=(z%10).astype(np.int8)
        return pd.DataFrame(d)
    def keys(self):
        if self._keys is not None:return self._keys
        both=pd.concat([self.raw,self.test_raw],ignore_index=True); d={}
        for c in self.raw:d[c]=pd.factorize(both[c],sort=False)[0].astype(np.int32)
        for c in ['amount_usd','hours_since_last_txn','distance_from_home_km','account_balance_usd']:
            for div in [1,10]:
                d[c+'_rounded_'+str(div)]=pd.factorize(np.rint(both[c].to_numpy()/div).astype(np.int64),sort=False)[0].astype(np.int32)
        for a,b in [('cvv_retry_count','auth_method'),('is_foreign_transaction','ip_country_mismatch'),('used_vpn','ip_country_mismatch'),('merchant_category','is_new_merchant'),('auth_method','channel'),('cvv_retry_count','txn_count_last_24h')]:
            d[a+'__'+b]=pd.factorize(pd.MultiIndex.from_frame(both[[a,b]]),sort=False)[0].astype(np.int32)
        d=pd.DataFrame(d)
        self._keys=(d.iloc[:len(self.y)].reset_index(drop=True),d.iloc[len(self.y):].reset_index(drop=True))
        return self._keys
    def original(self):
        if self._orig is not None:return self._orig
        original=pd.read_csv(self.root/'data/original/credit_card_fraud_2026.csv')
        oy=original.is_fraud.astype(int); prior=float(oy.mean())
        g,gt=pd.DataFrame(index=self.raw.index),pd.DataFrame(index=self.test_raw.index)
        for c in ['merchant_risk_score','velocity_score','customer_age','auth_method','merchant_category',*self.bools]:
            stats=original.groupby(c).is_fraud.agg(['sum','size'])
            rates=(stats['sum']+20*prior)/(stats['size']+20)
            g['original_rate_'+c]=self.raw[c].map(rates).fillna(prior).astype(np.float32)
            gt['original_rate_'+c]=self.test_raw[c].map(rates).fillna(prior).astype(np.float32)
        a,b=self.lgb_format(original[self.raw.columns],pd.concat([self.raw,self.test_raw],ignore_index=True))
        teacher=lgb.LGBMClassifier(n_estimators=500,learning_rate=.03,num_leaves=4,min_child_samples=50,reg_lambda=10,n_jobs=8,verbosity=-1,random_state=42)
        teacher.fit(a,oy); p=teacher.predict_proba(b)[:,1]
        g['original_model']=p[:len(self.y)].astype(np.float32);gt['original_model']=p[len(self.y):].astype(np.float32)
        self._orig=(g,gt);return self._orig
    def lgb_format(self,a,b):
        a,b=a.copy(),b.copy()
        for c in self.cats:
            vocabulary=sorted(a[c].dropna().unique().tolist())
            a[c]=pd.Categorical(a[c],categories=vocabulary);b[c]=pd.Categorical(b[c],categories=vocabulary)
        return a,b
    def frames(self,idx,val,mode):
        a=self.raw.iloc[idx].reset_index(drop=True).copy()
        b=(self.raw.iloc[val] if val is not None else self.test_raw).reset_index(drop=True).copy()
        use_digits=mode in ['digits','encoded_digits','all']
        use_encoding=mode in ['encoded','encoded_digits','encoded_original','all']
        use_original=mode in ['original','encoded_original','all']
        if use_digits:
            a=pd.concat([a,self.digit(a)],axis=1);b=pd.concat([b,self.digit(b)],axis=1)
        if use_encoding:
            keys,kt=self.keys()
            tag=hashlib.sha256(self.fingerprint.encode()+np.asarray(idx,dtype=np.int64).tobytes()+(b'test' if val is None else np.asarray(val,dtype=np.int64).tobytes())).hexdigest()[:20]
            file=self.cache/('target_encoder_'+tag+'.npz')
            if file.exists():
                saved=np.load(file);ae,be=saved['train'],saved['valid']
            else:
                encoder=TargetEncoder(target_type='binary',smooth='auto',cv=3,shuffle=True,random_state=42)
                ae=encoder.fit_transform(keys.iloc[idx],self.y[idx]).astype(np.float32)
                be=encoder.transform(keys.iloc[val] if val is not None else kt).astype(np.float32)
                np.savez_compressed(file,train=ae,valid=be)
            names=['te_'+c for c in keys]
            a=pd.concat([a,pd.DataFrame(ae,columns=names)],axis=1)
            b=pd.concat([b,pd.DataFrame(be,columns=names)],axis=1)
        if use_original:
            g,gt=self.original()
            a=pd.concat([a,g.iloc[idx].reset_index(drop=True)],axis=1)
            b=pd.concat([b,(g.iloc[val] if val is not None else gt).reset_index(drop=True)],axis=1)
        return a,b
