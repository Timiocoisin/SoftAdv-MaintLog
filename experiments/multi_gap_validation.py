"""Exploratory repair for the single-gap label-prior confound, identified during analysis.
Two multi-gap settings fixed before their fits. Reuse code, never reuse trained weights.
"""
from pathlib import Path
import os,sys,json,subprocess,time
import numpy as np
import fair_cnn_validation as base

base.OUT=base.ROOT/'outputs/multi_gap_validation_20261106'
base.OUT.mkdir(exist_ok=True)

def main():
    protocol=base.prepare()
    h,e,g,keys=base.data()
    centers=[25,55,85,115,145]
    test=np.zeros(len(g),bool)
    for c in centers:test|=(g>=c-4)&(g<=c+4)
    folds=[]
    for guard in [0,10]:
        excluded=np.zeros(len(g),bool)
        for c in centers:excluded|=(g>=c-4-guard)&(g<=c+4+guard)
        train=~excluded; ts=np.unique(g[train])
        # Rank-based subsampling prevents all sparse middle anchors falling in validation.
        val=ts[np.arange(len(ts))%5==2]
        fit=np.setdiff1d(ts,val)
        fold={'name':f'multi_gap_guard{guard}','kind':'multi_gap','center':-1,'guard':guard,
            'train_stages':ts.tolist(),'test_stages':np.unique(g[excluded]).tolist(),
            'core_test_stages':np.unique(g[test]).tolist(),
            'inner_fit_stages':fit.tolist(),'inner_val_stages':val.tolist()}
        assert len(fold['core_test_stages'])==45 and not set(ts)&set(fold['test_stages'])
        folds.append(fold)
    protocol['folds']=folds
    protocol['status']='exploratory follow-up, designed after detecting single-gap label-prior confound'
    protocol['inner_validation']='every fifth available reference stage by sorted rank, offset2; same for all models'
    protocol['confound_note']='Query stages span 5 separated blocks, so one missing-interval midpoint cannot localize all queries.'
    protocol['evaluation_note']='Evaluate all missing stages: 45 at guard0, 145 at guard10. Also summarize the identical 45 central core stages across settings; guard stages have never entered fit/validation.'
    (base.OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    procs=[]
    for worker in [0,1]:
        log=open(base.OUT/f'worker{worker}.log','a',buffering=1)
        env=dict(os.environ); env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        procs.append((subprocess.Popen([sys.executable,'-u',__file__,'worker',str(worker)],stdout=log,stderr=subprocess.STDOUT,env=env),log))
    base.baseline_run(protocol)
    while any(p.poll() is None for p,l in procs):
        print('PROGRESS',len(list(base.OUT.glob('multi_gap_guard*_seed*.json'))),'/6',flush=True);time.sleep(20)
    codes=[p.wait() for p,l in procs]
    for p,l in procs:l.close()
    assert not any(c!=0 for c in codes),codes
    import pandas as pd
    rows=[];preds=[]
    for fold in folds:
        for seed in base.SEEDS:
            tag=f"{fold['name']}_seed{seed}"
            row=json.load(open(base.OUT/(tag+'.json')));row.pop('history');rows.append(row)
            preds.append(pd.read_csv(base.OUT/(tag+'_predictions.csv')))
    pd.DataFrame(rows).to_csv(base.OUT/'cnn_metrics.csv',index=False)
    pd.concat(preds,ignore_index=True).to_csv(base.OUT/'cnn_predictions.csv',index=False)
    (base.OUT/'COMPLETE').write_text('All six multi-gap jobs completed.\n')
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':base.cnn_worker(int(sys.argv[2]))
    else:main()
