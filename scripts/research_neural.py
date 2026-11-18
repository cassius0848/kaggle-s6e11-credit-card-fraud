"""Controlled neural experiments, checkpointed on F:. Importing never trains."""
from pathlib import Path
import os,json,time,hashlib,argparse,gc,math
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
if __package__:
    from .research_features import FeatureStore
else:
    from research_features import FeatureStore

def save_json(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    temp.replace(path)

def log(*x): print(time.strftime('%H:%M:%S'),*x,flush=True)

def setup(root, neural=True):
    for name,folder in [('TEMP','temp'),('TMP','temp'),('TORCH_HOME','torch_cache'),('TORCH_EXTENSIONS_DIR','torch_extensions'),('XDG_CACHE_HOME','cache'),('MPLCONFIGDIR','mpl')]:
        path=root/'work'/folder;path.mkdir(exist_ok=True,parents=True);os.environ[name]=str(path)
    torch.set_num_threads(6)
    if neural: torch.set_float32_matmul_precision('high')
    assert torch.cuda.is_available(),'CUDA GPU is required for this experiment'
    log('GPU',torch.cuda.get_device_name(0))

def prepare(store,idx,val,mode,folder):
    a,b=store.frames(idx,val,mode)
    cats=store.cats+store.bools
    nums=[c for c in a if c not in cats]
    quantile=QuantileTransformer(n_quantiles=min(1000,len(a)),output_distribution='normal',subsample=200000,random_state=42)
    an=quantile.fit_transform(a[nums]).astype(np.float32)
    bn=quantile.transform(b[nums]).astype(np.float32)
    ac,bc,vocab=[],[],{}
    for c in cats:
        levels=sorted(a[c].unique().tolist())
        vocab[c]=levels
        ac.append(pd.Categorical(a[c],categories=levels).codes.astype(np.int64)+1)
        bc.append(pd.Categorical(b[c],categories=levels).codes.astype(np.int64)+1)
    ac=np.stack(ac,axis=1);bc=np.stack(bc,axis=1)
    joblib.dump({'quantile':quantile,'nums':nums,'cats':cats,'vocab':vocab},folder/'preprocessor.joblib')
    return an,ac,bn,bc,[len(vocab[c])+1 for c in cats]

def make_model(an,cards,cfg):
    import tabm,rtdl_num_embeddings as embeddings
    emb=None
    if cfg['representation']=='ple':
        bins=embeddings.compute_bins(torch.from_numpy(an),n_bins=cfg.get('bins',32))
        emb=embeddings.PiecewiseLinearEmbeddings(bins,12,activation=False,version='B')
    elif cfg['representation']=='periodic':
        emb=embeddings.PeriodicEmbeddings(an.shape[1],12,n_frequencies=24,frequency_init_scale=.1,lite=True)
    return tabm.TabM.make(n_num_features=an.shape[1],cat_cardinalities=cards,d_out=1,
                         num_embeddings=emb,k=cfg.get('k',16),n_blocks=cfg.get('blocks',2),
                         d_block=cfg.get('width',128),dropout=.1).cuda()

@torch.inference_mode()
def predict(model,x,c,offset):
    model.eval();result=[]
    for start in range(0,len(x),8192):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model(x[start:start+8192],c[start:start+8192])
        result.append((z.float()+offset).sigmoid().mean(1).squeeze(-1).cpu().numpy())
    return np.concatenate(result)

def train_tabm(store,idx,val,cfg,name,fixed_epochs=None,warm_path=None,arm='bce',test_predict=False):
    folder=store.root/'work/research_v4/neural'/name
    folder.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256((store.fingerprint+json.dumps(cfg,sort_keys=True)+arm+str(fixed_epochs)).encode()+np.asarray(idx).tobytes()+(b'test' if val is None else np.asarray(val).tobytes())).hexdigest()
    result_file=folder/'result.json'
    if result_file.exists():
        result=json.loads(result_file.read_text())
        if result['signature']!=signature:raise RuntimeError('Changed configuration: use a new run name')
        log('CACHED',name,result.get('auc'));return result
    start=time.perf_counter()
    log('START',name,cfg)
    torch.manual_seed(cfg.get('seed',42));np.random.seed(cfg.get('seed',42))
    an,ac,bn,bc,cards=prepare(store,idx,val,cfg.get('features','raw'),folder)
    model=make_model(an,cards,cfg)
    x=torch.from_numpy(an).cuda();c=torch.from_numpy(ac).cuda()
    vx=torch.from_numpy(bn).cuda();vc=torch.from_numpy(bc).cuda()
    y=torch.as_tensor(store.y[idx],device='cuda',dtype=torch.float32)[:,None,None]
    prior=float(store.y[idx].mean());offset=math.log(prior/(1-prior))
    if warm_path is not None:
        warm=torch.load(warm_path,map_location='cuda',weights_only=False)
        model.load_state_dict(warm['model'])
    criterion=None
    if arm=='auc':
        from libauc.losses import AUCMLoss
        from libauc.optimizers import PESG
        criterion=AUCMLoss(version='v2',device='cuda')
        optimizer=PESG(model.parameters(),loss_fn=criterion,lr=.05,mode='sgd',weight_decay=.0003,epoch_decay=.002,device='cuda',verbose=False)
    else: optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.get('lr',.002),weight_decay=.0003)
    max_epoch=fixed_epochs or cfg.get('epochs',48)
    best=-1.;best_epoch=0;epoch_start=1;history=[]
    checkpoint=folder/'last.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cuda',weights_only=False)
        if state['signature']!=signature:raise RuntimeError('Checkpoint signature mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        if criterion is not None:criterion.load_state_dict(state['criterion'])
        epoch_start=state['epoch']+1;best=state['best'];best_epoch=state['best_epoch'];history=state['history']
        log('RESUME',name,'epoch',epoch_start)
    batch=2048
    # Same class-controlled sequence and exposure for both fine-tuning arms.
    pos=np.flatnonzero(store.y[idx]==1);neg=np.flatnonzero(store.y[idx]==0)
    for epoch in range(epoch_start,max_epoch+1):
        epoch_time=time.perf_counter()
        torch.manual_seed(cfg.get('seed',42)+1000+epoch)
        rng=np.random.default_rng(cfg.get('seed',42)+1000+epoch)
        if arm in ['auc','balanced_bce']:
            n_batches=len(idx)//batch
            positive=rng.choice(pos,n_batches*int(batch*.1),replace=True).reshape(n_batches,-1)
            negative=rng.choice(neg,n_batches*(batch-int(batch*.1)),replace=False).reshape(n_batches,-1)
            permutation=torch.as_tensor(np.concatenate([positive,negative],axis=1).reshape(-1),device='cuda')
        else:permutation=torch.as_tensor(rng.permutation(len(idx)),device='cuda')
        model.train();loss_sum=0.;steps=0
        for indices in permutation.split(batch):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(x[indices],c[indices])
            logits=logits.float()+offset
            if arm=='auc':
                loss=criterion(logits.sigmoid().mean(1),y[indices].squeeze(1))
            else:loss=F.binary_cross_entropy_with_logits(logits,y[indices].expand_as(logits))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if arm!='auc':torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step();loss_sum+=loss.detach();steps+=1
        # Fixed-epoch refits do not monitor the outer validation labels.
        auc=None
        if fixed_epochs is None:
            prediction=predict(model,vx,vc,offset)
            auc=float(roc_auc_score(store.y[val],prediction))
        improved=(fixed_epochs is not None) or auc>best+1e-7
        if improved:
            best=auc if auc is not None else -1.;best_epoch=epoch
            torch.save({'model':model.state_dict(),'config':cfg,'offset':offset,'epoch':epoch},folder/'best.pt')
        record={'epoch':epoch,'loss':float(loss_sum/steps),'auc':auc,'seconds':round(time.perf_counter()-epoch_time,2)}
        history.append(record)
        state={'signature':signature,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'criterion':None if criterion is None else criterion.state_dict(),'epoch':epoch,
               'best':best,'best_epoch':best_epoch,'history':history}
        temp=folder/'last.tmp';torch.save(state,temp);temp.replace(checkpoint)
        save_json(folder/'history.json',history)
        log(name,'epoch',epoch,'auc',None if auc is None else round(auc,8),'best',best_epoch,'sec',record['seconds'])
        if fixed_epochs is None and arm not in ['auc','balanced_bce'] and epoch>=cfg.get('min_epochs',12) and epoch-best_epoch>=cfg.get('patience',8):break
    best_state=torch.load(folder/'best.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(best_state['model'])
    prediction=predict(model,vx,vc,offset)
    np.save(folder/'valid.npy',prediction)
    result={'name':name,'signature':signature,'config':cfg,'arm':arm,'best_epoch':best_epoch,
            'epochs_run':len(history),'seconds':round(time.perf_counter()-start,2),
            'auc':None if val is None else float(roc_auc_score(store.y[val],prediction)),
            'validation_protocol':'fixed epochs selected on inner split' if fixed_epochs else 'development holdout early stopping'}
    if test_predict and val is not None:
        # Transform test using exactly the preprocessing fitted on this training partition.
        _,test=store.frames(idx,None,cfg.get('features','raw'))
        prep=joblib.load(folder/'preprocessor.joblib')
        tn=prep['quantile'].transform(test[prep['nums']]).astype(np.float32)
        tc=np.stack([pd.Categorical(test[col],categories=prep['vocab'][col]).codes.astype(np.int64)+1 for col in prep['cats']],axis=1)
        testp=predict(model,torch.from_numpy(tn).cuda(),torch.from_numpy(tc).cuda(),offset)
        np.save(folder/'test.npy',testp)
    save_json(result_file,result);log('DONE',name,result)
    del model,optimizer,x,c,vx,vc,y;gc.collect();torch.cuda.empty_cache()
    return result

def screen_tabm(store):
    results=[]
    common={'k':16,'blocks':2,'width':128,'epochs':48,'patience':8,'min_epochs':12,'seed':42}
    for representation in ['plain','ple','periodic']:
        cfg=dict(common,representation=representation,features='raw')
        results.append(train_tabm(store,store.fit_idx,store.val_idx,cfg,'screen_tabm_'+representation))
    save_json(store.root/'work/research_v4/neural/screen_summary.json',results)
    return results

def realmlp(store, full=False):
    from pytabkit import RealMLP_TD_Classifier
    folder=store.root/'work/research_v4/neural'/('screen_realmlp_default' if full else 'screen_realmlp')
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    a,b=store.frames(store.fit_idx,store.val_idx,'raw')
    for col in store.bools:a[col]=a[col].astype(int);b[col]=b[col].astype(int)
    for col in store.cats:a[col]=a[col].astype(object);b[col]=b[col].astype(object)
    model=RealMLP_TD_Classifier(device='cuda',random_state=42,n_cv=1,n_refit=0,
          n_threads=6,tmp_folder=str(folder/'temp'),verbosity=1,n_epochs=256 if full else 64,
          batch_size=4096 if full else 1024,hidden_sizes=[256,256,256] if full else [128,128,128],val_metric_name='1-auc_ovr',use_ls=False)
    start=time.perf_counter();log('START RealMLP', 'default scale' if full else 'small')
    import pytorch_lightning as pl
    original_trainer=pl.Trainer
    class Progress(pl.Callback):
        def on_validation_epoch_end(self,trainer,pl_module):
            epoch=trainer.current_epoch+1
            if epoch%16==0:
                log(folder.name,'epoch',epoch)
                save_json(folder/'progress.json',{'epoch':epoch,'max_epochs':256 if full else 64,'timestamp':time.strftime('%Y-%m-%d %H:%M:%S')})
                # Save the Lightning model/optimizer snapshot; native pytabkit best-state resume needs an adapter.
                trainer.save_checkpoint(str(folder/'last.ckpt'))
    class ProgressTrainer(original_trainer):
        def __init__(self,*args,**kwargs):
            kwargs['callbacks']=list(kwargs.get('callbacks',[]))+[Progress()]
            kwargs['default_root_dir']=str(folder)
            super().__init__(*args,**kwargs)
    pl.Trainer=ProgressTrainer
    try:
        model.fit(a,store.y[store.fit_idx],X_val=b,y_val=store.y[store.val_idx],cat_col_names=store.cats)
    finally:
        pl.Trainer=original_trainer
    p=model.predict_proba(b)[:,1];np.save(folder/'valid.npy',p)
    joblib.dump(model,folder/'model.joblib')
    result={'name':folder.name,'auc':float(roc_auc_score(store.y[store.val_idx],p)),
            'seconds':round(time.perf_counter()-start,2),'config':model.get_params(),
            'fit_params':str(getattr(getattr(model,'cv_alg_interface_',None),'fit_params',None)),
            'best_epoch':model.cv_alg_interface_.fit_params[0]['stop_epoch']['1-auc_ovr'],
            'resolved_config':model.get_config()}
    save_json(folder/'result.json',result);log('DONE RealMLP',result['auc']);return result

def loss_comparison(store):
    cfg={'representation':'ple','features':'raw','k':1,'blocks':2,'width':128,'seed':42,
         'epochs':32,'patience':8,'min_epochs':16,'lr':.002}
    warm=train_tabm(store,store.fit_idx,store.val_idx,cfg,'loss_warmup',fixed_epochs=8)
    checkpoint=store.root/'work/research_v4/neural/loss_warmup/best.pt'
    results=[]
    control=dict(cfg,patience=32)
    results.append(train_tabm(store,store.fit_idx,store.val_idx,control,'loss_unbalanced_bce',warm_path=checkpoint,arm='bce'))
    for arm in ['balanced_bce','auc']:
        results.append(train_tabm(store,store.fit_idx,store.val_idx,cfg,'loss_'+arm,
                                 warm_path=checkpoint,arm=arm))
    save_json(store.root/'work/research_v4/neural/loss_comparison.json',results)

def cv_tabm(store,representation,features='raw',width=128,members=16,seed=42):
    cfg={'representation':representation,'features':features,'k':16,'blocks':2,'width':128,
         'epochs':48,'patience':8,'min_epochs':12,'seed':42}
    results=[];oof=np.zeros(len(store.y));test=[]
    cfg.update(width=width,k=members,seed=seed)
    label='tabm_'+representation+'_'+features+('' if (width,members,seed)==(128,16,42) else f'_w{width}_k{members}_s{seed}')
    for fold,(idx,val) in enumerate(store.folds):
        fit,inner=train_test_split(idx,test_size=.2,random_state=1731+fold,stratify=store.y[idx])
        selected=train_tabm(store,fit,inner,cfg,label+'_inner'+str(fold))
        trained=train_tabm(store,idx,val,cfg,label+'_fold'+str(fold),fixed_epochs=selected['best_epoch'],test_predict=True)
        results.append(trained)
        folder=store.root/'work/research_v4/neural'/(label+'_fold'+str(fold))
        oof[val]=np.load(folder/'valid.npy');test.append(np.load(folder/'test.npy'))
    folder=store.root/'work/research_v4/neural'
    np.save(folder/('oof_'+label+'.npy'),oof);np.save(folder/('test_'+label+'.npy'),np.mean(test,axis=0))
    summary={'name':label,'folds':results,'mean_auc':float(np.mean([r['auc'] for r in results])),
             'pooled_auc':float(roc_auc_score(store.y,oof)),
             'protocol':'inner validation selects epochs; refit on outer train; outer labels evaluated only after training'}
    save_json(folder/(label+'_cv.json'),summary);log('CV DONE',summary)
    return summary

def finalists(store):
    if __package__:
        from .research_evaluate import tree_predictions
    else:
        from research_evaluate import tree_predictions
    base=tree_predictions(store,'screen');y=store.y[store.val_idx]
    candidates=[]
    folder=store.root/'work/research_v4/neural'
    for path in sorted(folder.glob('screen_tabm_ple*/result.json')):
        r=json.loads(path.read_text())
        if r['config'].get('seed',42)!=42:continue
        p=np.load(path.parent/'valid.npy')
        gain=max(float(roc_auc_score(y,(1-w)*base+w*p)) for w in [.1,.25,.5])
        candidates.append({'name':r['name'],'config':r['config'],'development_blend_auc':gain})
    selected=[]
    for mode in ['raw','encoded']:
        family=[c for c in candidates if c['config']['features']==mode]
        best=max(c['development_blend_auc'] for c in family)
        comparable=[c for c in family if c['development_blend_auc']>=best-.00002]
        selected.append(min(comparable,key=lambda c:c['config']['width']**2*c['config']['k']))
    save_json(folder/'cv_candidate_screening.json',{'candidates':candidates,'selection_rule':'Within 0.00002 of the best development blend AUC, prefer smaller k*width^2. Confirm one raw and one encoded model.'})
    selection=folder/'selection_for_cv.json'
    if selection.exists():
        assert json.loads(selection.read_text())==selected,'Candidate set changed; review selection before resuming'
    else:save_json(selection,selected)
    log('FROZEN CV CANDIDATES',selected)
    # This second seed measures training variation; it does not select hyperparameters.
    for item in selected:
        cfg=dict(item['config'],seed=123)
        train_tabm(store,store.fit_idx,store.val_idx,cfg,item['name']+'_seed123')
    for item in selected:
        cfg=item['config']
        cv_tabm(store,cfg['representation'],cfg['features'],cfg['width'],cfg['k'],cfg['seed'])
    loss_comparison(store)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',required=True)
    parser.add_argument('--stage',choices=['screen','realmlp','realmlp-default','loss','cv','encoded','finalists'],default='screen')
    parser.add_argument('--representation',default='ple',choices=['plain','ple','periodic'])
    parser.add_argument('--features',default='raw',choices=['raw','encoded','encoded_digits'])
    parser.add_argument('--width',type=int,default=128)
    parser.add_argument('--members',type=int,default=16)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();root=Path(args.workspace).resolve();setup(root,args.stage not in ['realmlp','realmlp-default']);store=FeatureStore(root)
    if args.stage=='screen':screen_tabm(store)
    if args.stage=='realmlp':realmlp(store)
    if args.stage=='realmlp-default':realmlp(store,full=True)
    if args.stage=='finalists':finalists(store)
    if args.stage=='loss':loss_comparison(store)
    if args.stage=='cv':cv_tabm(store,args.representation,args.features,args.width,args.members,args.seed)
    if args.stage=='encoded':
        cfg={'representation':args.representation,'features':args.features,'k':16,'blocks':2,'width':128,
             'epochs':48,'patience':8,'min_epochs':12,'seed':42}
        cfg.update(width=args.width,k=args.members,seed=args.seed)
        suffix='' if (args.width,args.members,args.seed)==(128,16,42) else f'_w{args.width}_k{args.members}_s{args.seed}'
        train_tabm(store,store.fit_idx,store.val_idx,cfg,'screen_tabm_'+args.representation+'_'+args.features+suffix)
if __name__=='__main__':main()
