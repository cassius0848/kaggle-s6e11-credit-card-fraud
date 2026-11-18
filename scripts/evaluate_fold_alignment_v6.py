"""Exploratory fold-score alignment after the fixed V6 blend screen. No training."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .evaluate_public_extension_v6 import auc,paired_auc,metrics,write_candidate
from .replicate_public_v5 import logit_rank,read_prediction,sha256


def run(root):
    work=root/'work/extension_v6';previous=json.loads((work/'result.json').read_text())
    protocol={'stage':'Exploratory follow-up: label-free within-model-fold rank alignment',
              'trigger':'Local neural OOF pooled AUC differs from mean per-training-fold AUC, suggesting between-fold score shifts.',
              'models':['tabm_ple_encoded','tabm_ple_raw'],'weights':[0.01,0.025,0.05,0.10,0.20],
              'gate':previous['protocol']['selection_gate'],
              'inference':'Rank-normalize each of the three fold-specific test vectors, average their logit ranks, then rank-normalize the ensemble.',
              'limits':previous['limitations']+['This follow-up was motivated by the first screen and is not a pre-registered independent confirmation.']}
    (work/'fold_alignment_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    y=pd.read_csv(root/'data/train.csv',usecols=['is_fraud']).is_fraud.to_numpy(np.int8)
    ids=pd.read_csv(root/'data/test.csv',usecols=['id']).id.to_numpy()
    folds=[v for _,v in StratifiedKFold(3,shuffle=True,random_state=2026).split(y,y)]
    z=np.load(work/'inputs/s6e11-rank1-q17-meta-bundle/rank1_q17_streams.npz',allow_pickle=False)
    public=[np.flatnonzero(z['folds']==f) for f in range(5)]
    directory=work/'inputs/s6e11-fraud-series-artifacts'
    bases={name:logit_rank(np.load(directory/(name+'_nested.npy'))) for name in ['wide_lm_stack','wide_stack']}
    aligned={};aligned_test={};alignment_metrics={};hashes={}
    neural=root/'work/research_v4/neural'
    for name in protocol['models']:
        raw=np.load(neural/('oof_'+name+'.npy'));scores=np.empty(len(y));tests=[]
        for f,index in enumerate(folds):
            scores[index]=logit_rank(raw[index])
            p=neural/(name+'_fold'+str(f))/'test.npy';v=np.load(p,allow_pickle=False)
            if v.shape!=(len(ids),) or not np.isfinite(v).all():raise ValueError(p)
            hashes[str(p)]=sha256(p);tests.append(logit_rank(v))
        aligned[name]=logit_rank(scores);aligned_test[name]=logit_rank(np.mean(tests,axis=0))
        alignment_metrics[name]={'original_pooled_auc':auc(y,raw),'aligned_pooled_auc':auc(y,scores),'per_training_fold_auc':[auc(y[ix],raw[ix]) for ix in folds]}
    rows=[]
    for base_name,base in bases.items():
        bp=metrics(y,base,public);bl=metrics(y,base,folds)
        for name,scores in aligned.items():
            for weight in protocol['weights']:
                mixture=(1-weight)*base+weight*scores;mp=metrics(y,mixture,public);ml=metrics(y,mixture,folds)
                d=(np.array(mp['fold_auc'])-bp['fold_auc']).tolist();ld=(np.array(ml['fold_auc'])-bl['fold_auc']).tolist()
                rows.append({'base':base_name,'own_model':name,'own_weight':weight,'pooled_auc':mp['pooled_auc'],'pooled_delta':mp['pooled_auc']-bp['pooled_auc'],
                             'mean_five_fold_delta':float(np.mean(d)),'public_fold_deltas':d,'positive_public_folds':sum(v>0 for v in d),'worst_public_fold_delta':min(d),
                             'local_fold_deltas':ld,'positive_local_folds':sum(v>0 for v in ld)})
    eligible=[];gate=protocol['gate']
    for row in rows:
        if row['base']!='wide_lm_stack':continue
        secondary=next(r for r in rows if r['base']=='wide_stack' and r['own_model']==row['own_model'] and r['own_weight']==row['own_weight'])
        row['diagnostic_gate_passed']=False
        if row['mean_five_fold_delta']>=gate['primary_mean_five_fold_delta_at_least'] and row['positive_public_folds']>=4 and row['worst_public_fold_delta']>=-.00002 and row['positive_local_folds']==3 and secondary['pooled_delta']>0:
            w=row['own_weight'];name=row['own_model'];row['paired_interval']=paired_auc(y,(1-w)*bases['wide_lm_stack']+w*aligned[name],bases['wide_lm_stack'])
            if row['paired_interval']['conditional_95_interval'][0]>0:row['diagnostic_gate_passed']=True;eligible.append(row)
    best=None;submissions=[]
    if eligible:
        top=max(row['pooled_auc'] for row in eligible);best=min([row for row in eligible if row['pooled_auc']>=top-.000002],key=lambda row:row['own_weight'])
        weight=best['own_weight'];own=aligned_test[best['own_model']]
        public_test=logit_rank(np.load(directory/'wide_lm_stack_test.npy'))
        submissions.append(write_candidate(root/'outputs/submission_v6_aligned_cv_candidate.csv',ids,(1-weight)*public_test+weight*own))
        v5path=root/'outputs/submission_v5_replica_095447.csv';v5=logit_rank(read_prediction(v5path,previous['protocol']['v5_sha256'],ids))
        submissions.append(write_candidate(root/'outputs/submission_v6_aligned_v5_experimental.csv',ids,(1-weight)*v5+weight*own))
    result={'protocol':protocol,'alignment_metrics':alignment_metrics,'candidates':rows,'selected':best,'submissions':submissions,'additional_input_hashes':hashes}
    # Conditional intervals for the two best primary candidates, including ones that did not pass the gate.
    result['best_diagnostics']=[]
    for name in aligned:
        row=max([r for r in rows if r['base']=='wide_lm_stack' and r['own_model']==name],key=lambda r:r['pooled_auc']).copy();w=row['own_weight']
        row['paired_interval']=paired_auc(y,(1-w)*bases['wide_lm_stack']+w*aligned[name],bases['wide_lm_stack']);result['best_diagnostics'].append(row)
    for folder in [work,root/'repository/reports']:
        (folder/'fold_alignment_v6.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'alignment':alignment_metrics,'best':result['best_diagnostics'],'selected':best},indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True);args=parser.parse_args();run(args.workspace.resolve())
