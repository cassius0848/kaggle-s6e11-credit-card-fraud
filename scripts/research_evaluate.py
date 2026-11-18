"""Evaluate fixed predictions; never starts model fitting."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
if __package__:
    from .research_features import FeatureStore
    from .research_neural import save_json
else:
    from research_features import FeatureStore
    from research_neural import save_json

def paired_auc(y,a,b):
    """DeLong placements for a fixed paired score comparison, including ties.
    Conditional on these predictions; does not correct model selection or CV dependence.
    """
    pos=y==1;neg=~pos
    def placements(p):
        pp=np.sort(p[pos]);pn=np.sort(p[neg])
        vp=(np.searchsorted(pn,p[pos],side='left')+np.searchsorted(pn,p[pos],side='right'))/(2*len(pn))
        vn=1-(np.searchsorted(pp,p[neg],side='left')+np.searchsorted(pp,p[neg],side='right'))/(2*len(pp))
        return vp,vn
    ap,an=placements(a);bp,bn=placements(b)
    delta=float(np.mean(ap-bp))
    se=float(np.sqrt(np.var(ap-bp,ddof=1)/len(ap)+np.var(an-bn,ddof=1)/len(an)))
    return {'delta':delta,'conditional_se':se,'conditional_95_interval':[delta-1.96*se,delta+1.96*se],
            'caveat':'Does not correct development selection, overlapping CV training sets, or training randomness.'}

def tree_predictions(store,stage):
    folder=store.root/'work/experiments_v3'
    if stage=='screen':
        records=json.loads((folder/'results.json').read_text())
        return np.mean([np.load(next(r['prediction_file'] for r in records if r['stage']=='screen' and r['name']==n))
                       for n in ['lgb_leaf4_encoded','xgb_depth2_full']],axis=0)
    return np.mean([np.load(folder/('oof_'+n+'.npy')) for n in ['lgb_leaf4_encoded','xgb_depth2_full']],axis=0)

def development(store):
    folder=store.root/'work/research_v4/neural'
    base=tree_predictions(store,'screen'); y=store.y[store.val_idx]
    rows=[{'model':'v3_tree_blend','neural_weight':0.,'auc':float(roc_auc_score(y,base))}]
    for path in sorted(folder.glob('*/result.json')):
        result=json.loads(path.read_text())
        if not path.parent.name.startswith(('screen_','loss_')) or not (path.parent/'valid.npy').exists():continue
        p=np.load(path.parent/'valid.npy')
        for w in [.1,.25,.5,1.]:
            rows.append({'model':path.parent.name,'neural_weight':w,'auc':float(roc_auc_score(y,(1-w)*base+w*p)),
                         'best_epoch':result.get('best_epoch'),'training_seconds':result.get('seconds')})
    table=pd.DataFrame(rows)
    table.to_csv(store.root/'outputs/research_v4_development.csv',index=False)
    print(table.to_string(index=False),flush=True)
    return rows

def cv(store):
    folder=store.root/'work/research_v4/neural';base=tree_predictions(store,'cv')
    def score(p):
        values=[float(roc_auc_score(store.y[v],p[v])) for _,v in store.folds]
        return {'mean_auc':float(np.mean(values)),'fold_auc':values,'pooled_auc':float(roc_auc_score(store.y,p))}
    base_score=score(base);rows=[{'name':'v3_tree_blend','weight':0.,**base_score}]
    for file in sorted(folder.glob('oof_*.npy')):
        p=np.load(file)
        for w in [.1,.25,.5,1.]:
            blend=(1-w)*base+w*p
            rows.append({'name':file.stem[4:],'weight':w,**score(blend),
                         'paired_comparison':paired_auc(store.y,blend,base)})
    report={'baseline':base_score,'candidates':rows,'note':'Development comparison; families/weights selected on previously explored data. Public scores are unknown.'}
    save_json(store.root/'work/research_v4/cv_comparison.json',report)
    pd.DataFrame([{k:v for k,v in row.items() if k!='paired_comparison'} for row in rows]).to_csv(store.root/'outputs/research_v4_cv.csv',index=False)
    print(json.dumps(report,indent=2),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',required=True);parser.add_argument('--stage',choices=['development','cv'],default='development')
    args=parser.parse_args();store=FeatureStore(Path(args.workspace))
    development(store) if args.stage=='development' else cv(store)
if __name__=='__main__':main()
