"""Create a reviewable research report and an optional validated V4 candidate."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
if __package__:
    from .research_features import FeatureStore
    from .research_evaluate import development,cv,tree_predictions,paired_auc
    from .research_neural import save_json
else:
    from research_features import FeatureStore
    from research_evaluate import development,cv,tree_predictions,paired_auc
    from research_neural import save_json

def finalize(root):
    store=FeatureStore(root);dev=development(store);comparison=cv(store)
    folder=root/'work/research_v4/neural';base=tree_predictions(store,'cv')
    def metrics(p):
        scores=[float(roc_auc_score(store.y[v],p[v])) for _,v in store.folds]
        return dict(mean_auc=float(np.mean(scores)),fold_auc=scores,pooled_auc=float(roc_auc_score(store.y,p)))
    baseline=metrics(base);candidates=[]
    neural={f.stem[4:]:np.load(f) for f in sorted(folder.glob('oof_*.npy'))}
    for name,p in neural.items():
        for w in [.1,.25,.5]:
            pred=base*(1-w)+p*w
            candidates.append({'weights':{'v3':1-w,name:w},**metrics(pred)})
    # A small predefined joint grid, in addition to the single-model blends.
    names=list(neural)
    if len(names)==2:
        for wa,wb in [(.1,.1),(.1,.25),(.25,.25)]:
            p=(1-wa-wb)*base+wa*neural[names[0]]+wb*neural[names[1]]
            candidates.append({'weights':{'v3':1-wa-wb,names[0]:wa,names[1]:wb},**metrics(p)})
    for c in candidates:
        c['fold_delta']=[a-b for a,b in zip(c['fold_auc'],baseline['fold_auc'])]
        c['positive_folds']=sum(x>0 for x in c['fold_delta'])
        c['mean_delta']=c['mean_auc']-baseline['mean_auc']
    eligible=[c for c in candidates if c['mean_delta']>=.00005 and c['positive_folds']>=2 and min(c['fold_delta'])>-.0002]
    selected=None;submission=None
    if eligible:
        best=max(c['mean_auc'] for c in eligible)
        similar=[c for c in eligible if c['mean_auc']>=best-.00002]
        selected=min(similar,key=lambda c:(len(c['weights']),1-c['weights']['v3']))
        oof=sum(weight*(base if name=='v3' else neural[name]) for name,weight in selected['weights'].items())
        selected['paired_comparison']=paired_auc(store.y,oof,base)
        v3=pd.read_csv(root/'outputs/submission_v3_recommended.csv')
        sample=pd.read_csv(root/'data/sample_submission.csv')
        assert v3.columns.tolist()==sample.columns.tolist()
        assert np.array_equal(v3.id.to_numpy(),store.test.id.to_numpy())
        testp=selected['weights']['v3']*v3.is_fraud.to_numpy()
        for name,w in selected['weights'].items():
            if name!='v3':testp+=w*np.load(folder/('test_'+name+'.npy'))
        assert len(testp)==len(store.test) and np.isfinite(testp).all() and ((testp>=0)&(testp<=1)).all()
        sample['is_fraud']=testp
        output=root/'outputs/submission_v4_research.csv'
        if output.exists():output=root/('outputs/submission_v4_research_'+time.strftime('%Y%m%d_%H%M%S')+'.csv')
        sample.to_csv(output,index=False)
        reread=pd.read_csv(output)
        assert reread.columns.tolist()==['id','is_fraud'] and len(reread)==len(store.test)
        assert np.array_equal(reread.id.to_numpy(),store.test.id.to_numpy())
        assert reread.is_fraud.between(0,1).all() and not reread.isna().any().any()
        submission={'file':str(output),'rows':len(reread),'columns':reread.columns.tolist(),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                    'public_auc':None,'status':'local candidate; not submitted'}
    if submission is not None:
        description=('Research blend of the V3 shallow-tree baseline and fold-ensemble TabM with piecewise-linear numeric embeddings. '
                     'Weights: '+json.dumps(selected['weights'])+'. '
                     'Local 3-fold mean ROC-AUC: '+f"{selected['mean_auc']:.8f}"+'. '
                     'Neural epochs selected on inner validation, then fixed-epoch outer-train refits. '
                     'Target encoding and numeric preprocessing fitted within training partitions. '
                     'These are development results; the V4 public score is unknown.')
        (root/'outputs/submission_v4_description.txt').write_text(description+'\n',encoding='utf-8')
        save_json(root/'work/research_v4/submission_verification.json',submission)
    ablation=json.loads((root/'work/research_v4/ablation/summary.json').read_text())
    screens=[]
    for file in sorted(folder.glob('*/result.json')):
        result=json.loads(file.read_text())
        if file.parent.name.startswith(('screen_','loss_')):
            screens.append({k:result.get(k) for k in ['name','auc','seconds','best_epoch','epochs_run','config','fit_params']})
    report={'created':time.strftime('%Y-%m-%d %H:%M:%S'),'v3_user_reported_public_auc':.94910,'target_public_auc':.96,
            'v4_public_auc':None,'baseline_cv':baseline,'selected':selected,'submission':submission,
            'blend_candidates':candidates,'screen_results':screens,'ablation_cv':ablation['cv'],
            'selection_rule':'Mean gain >=0.00005, at least 2 positive folds, no fold worse than -0.0002. Within 0.00002 of best prefer fewer models and less neural weight.',
            'limitations':['Historical data/folds have been explored repeatedly; this is development evidence.',
                          'Neural outer epochs are selected on inner splits; V3 baseline used outer early stopping.',
                          'Blend selection uses OOF scores, so its reported gain has selection bias.',
                          'Conditional paired intervals do not account for model selection or overlapping training partitions.',
                          'Paper-inspired configurations and limited seeds are not complete reproductions of paper benchmarks.']}
    save_json(root/'work/research_v4/final.json',report)
    save_json(root/'repository/reports/research_v4_summary.json',report)
    lines=['# V4 研究实验结果','',
           'V3 的用户报告公开分数为 **0.94910**。本轮 V4 尚未提交 Kaggle，公开分数未知；**尚未达到 0.96**。','',
           '## 三折复验','',
           '| 方法 | 三折平均 AUC | 各折 AUC |','|---|---:|---|',
           '| V3 树模型融合 | '+f"{baseline['mean_auc']:.8f}"+' | '+', '.join(f'{x:.8f}' for x in baseline['fold_auc'])+' |']
    for file in sorted(folder.glob('tabm_*_cv.json')):
        r=json.loads(file.read_text());lines.append('| '+r['name']+' | '+f"{r['mean_auc']:.8f}"+' | '+', '.join(f"{x['auc']:.8f}" for x in r['folds'])+' |')
    if selected:
        lines+=['| 本轮候选融合 | '+f"{selected['mean_auc']:.8f}"+' | '+', '.join(f'{x:.8f}' for x in selected['fold_auc'])+' |','',
                '权重：'+json.dumps(selected['weights'],ensure_ascii=False)+'。',
                '相对 V3 的平均增量：'+f"{selected['mean_delta']:+.8f}"+'。各折增量：'+', '.join(f'{x:+.8f}' for x in selected['fold_delta'])+'。',
                '这是待线上验证的候选，不能视为已确认的排行榜提升。','',
                '提交文件：'+submission['file'],'行数：'+str(submission['rows'])+'；列：id、is_fraud；编号顺序、概率范围、缺失值和重新读入检查均通过。']
    else:lines+=['','本轮没有满足预设收益与稳定性条件的融合，继续保留 V3 推荐提交。']
    lines+=['','## 首轮固定留出实验','','| 方法 | AUC | 最佳轮次 |','|---|---:|---:|']
    for r in screens:lines.append('| '+r['name']+' | '+f"{r['auc']:.8f}"+' | '+str(r['best_epoch'])+' |')
    lines+=['','## 特征消融','','| 特征组 | 固定留出 AUC |','|---|---:|']
    for r in ablation['screen']:lines.append('| '+r['mode']+' | '+f"{r['auc']:.8f}"+' |')
    lines+=['','取值统计是本轮最明确的有效特征。小数位叠加后的三折增益很小；原始公开数据的统计在本次固定骨干上没有带来收益。','',
            '## 验证边界','',
            'TabM 三折先在每个外层训练部分内部选择 epoch，再用全部外层训练部分固定轮次重训，外层标签只在最终计算 AUC 时读取。所有目标编码与数值预处理都在训练部分拟合。','',
            '历史验证集已用于多轮开发；候选族和融合权重仍根据开发结果选择，因此这不是完全独立的盲测。V3 早停协议与本轮神经网络不同。配对区间只对固定预测作条件比较，不消除选择偏差。','',
            '两组损失对照共享 8 轮 BCE 预训练权重，并用相同的 10% 正类批次、32 轮训练预算。AUC 组同时改用了 PESG，不能把差异只归因于损失函数。另有原始类别比例的 BCE 对照。','',
            'RealMLP 默认规模实验使用官方默认的三层 256、256 轮及其优化方案；大数据训练批次调至 4096，并关闭标签平滑，以 AUC 选择模型。小规模筛选使用三层 128、64 轮。','',
            'TabM 较小配置：k=16、两层128；扩容对照：k=32、两层256。数值表示对照的其余骨干参数保持一致。随机种子 123 只作训练波动检查，没有用来选择超参数。','',
            '## 文件','',
            '代码：repository/scripts/research_features.py、research_ablation.py、research_neural.py、research_evaluate.py、research_finalize.py。',
            '模型、预处理、逐轮记录和预测：work/research_v4。',
            '全部环境版本：work/research_v4/environment.txt。',
            '完整开发集对照：outputs/research_v4_development.csv；三折对照：outputs/research_v4_cv.csv。','',
            '本轮未自动提交 Kaggle，也未自动推送 GitHub。']
    text='\n'.join(lines)+'\n'
    (root/'outputs/research_report_v4.md').write_text(text,encoding='utf-8')
    (root/'repository/reports/research_report_v4.md').write_text(text,encoding='utf-8')
    print('FINAL',json.dumps({'selected':selected,'submission':submission},indent=2),flush=True)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    args=p.parse_args();finalize(args.workspace.resolve())
if __name__=='__main__':main()
