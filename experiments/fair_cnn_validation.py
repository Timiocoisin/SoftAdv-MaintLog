"""Fair-input exploratory follow-up. Fixed 3 source frames/pass across every model.
Outer test is never used to select hyperparameters, epochs, seeds or blending.
Two GPUs are computation workers, not research agents.
"""
from pathlib import Path
import os,sys,json,time,hashlib,random,copy,subprocess
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np,pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/fair_cnn_validation_bf16_20261106'
OUT.mkdir(exist_ok=True)
PREV=ROOT/'outputs/idea_validation_20261106'
MAX_EPOCHS=16
SEEDS=[0,1,2]

def data():
    a=np.load(OUT/'sampled_data.npz')
    return a['handcrafted'],a['embedding'],a['stages'],a['keys']

def score(y,p):
    return float(pd.DataFrame({'g':y,'e':np.abs(y-p)}).groupby('g').e.mean().mean())

def prepare():
    meta=pd.read_csv(ROOT/'outputs/baselines/features/frame_features_meta.csv')
    f=np.load(ROOT/'outputs/baselines/features/frame_features.npz',allow_pickle=True)['X']
    keys=[]; rows=[]; paths=[]; indices=[]
    for (g,p),group in meta.groupby(['grind','pass_id'],sort=True):
        idx=group.sort_values('frame_in_pass').index.to_numpy()
        si=idx[np.linspace(0,len(idx)-1,3).round().astype(int)]
        sf=f[si]
        rows.append(np.concatenate([sf.mean(0),sf.std(0),sf.max(0),sf.min(0)]).astype(np.float32))
        keys.append([int(g),int(p)]); indices.extend(si.tolist()); paths.extend(meta.loc[si,'path'].tolist())
    emb=np.load(PREV/'imagenet_resnet18_pass_features.npz')['X']
    assert np.array_equal(np.load(PREV/'imagenet_resnet18_pass_features.npz')['keys'],np.array(keys))
    assert len(keys)==877 and len(paths)==2631
    np.savez_compressed(OUT/'sampled_data.npz',handcrafted=np.stack(rows),embedding=emb,
                        stages=np.array(keys)[:,0],keys=np.array(keys))
    image_cache=OUT/'sampled_images.npy'
    old_cache=ROOT/'outputs/fair_cnn_validation_20261106/sampled_images.npy'
    if not image_cache.exists() and old_cache.exists():
        os.link(old_cache,image_cache)
    if not image_cache.exists():
        images=np.lib.format.open_memmap(image_cache,mode='w+',dtype=np.uint8,shape=(len(paths),128,768,3))
        for i,path in enumerate(paths):
            with Image.open(path) as im:
                images[i]=np.asarray(im.convert('RGB').resize((768,128),Image.Resampling.BILINEAR))
            if i%500==0: print('CACHE',i,len(paths),flush=True)
        images.flush(); del images
    pd.DataFrame({'cache_index':np.arange(len(paths)),'feature_row':indices,'path':paths}).to_csv(OUT/'sample_manifest.csv',index=False)
    g=np.array(keys)[:,0]
    folds=[]
    for c in [25,55,85,115,145]:
        for guard in [0,10]:
            test=(g>=c-4)&(g<=c+4)
            train=(g<c-4-guard)|(g>c+4+guard)
            folds.append({'name':f'block_c{c}_guard{guard}','kind':'block','center':c,'guard':guard,
                          'train_stages':np.unique(g[train]).tolist(),'test_stages':np.unique(g[test]).tolist()})
    folds.append({'name':'odd_even','kind':'interleaved','center':-1,'guard':0,
                  'train_stages':np.unique(g[g%2==0]).tolist(),'test_stages':np.unique(g[g%2==1]).tolist()})
    for fold in folds:
        ts=np.array(fold['train_stages']); val=ts[ts%5==0]; fit=ts[ts%5!=0]
        fold['inner_fit_stages']=fit.tolist(); fold['inner_val_stages']=val.tolist()
        assert len(val)>5 and not set(ts)&set(fold['test_stages'])
    protocol={'status':'exploratory fixed before new model fits','seeds':SEEDS,'max_epochs':MAX_EPOCHS,
       'folds':folds,'selection_metric':'stage-macro pass-aggregated MAE; smallest value, earliest epoch tie',
       'sample_rule':'3 uniformly spaced source frames/pass; same frames for every method',
       'cnn':{'backbone':'ResNet18 ImageNet IMAGENET1K_V1','input':[128,768],
        'loss':'CE3+CE5+2*MSE(progress sigmoid)','optimizer':'AdamW lr3e-4 weight_decay1e-4',
        'schedule':'cosine T_max=16 for selection and refit','batch_size':64,
        'train_augmentation':'independent horizontal flip p0.5','refit':'fresh ImageNet initialization; selected epoch count; all outer train stages',
        'fp':'BF16 CUDA autocast, float32 weights/optimizer; TF32 disabled; deterministic algorithms warn_only; fixed seeds'},
       'baseline_candidates':{'handcrafted_rf':[8,16,None],'handcrafted_knn':[3,5,9],
        'imagenet_knn':[3,5,9],'imagenet_ridge':[1.,10.,100.]},
       'execution_note':'Initial float32 run stopped before any outer CNN test result existed; changed only numerical execution precision to BF16 to accelerate all CNN jobs. Initial logs retained separately.',
       'limitations':['single specimen reused exploratory corpus','no new independent external test',
        'baseline scaling/parameters fit inside each outer train only','CNN has training flip augmentation; classical features are not augmented',
        'same source frames; feature extraction operations differ','main paper all-frame baseline is not directly compared to this 3-frame study']}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    return protocol

def baseline_run(protocol):
    h,e,g,keys=data(); rows=[]; predictions=[]
    for fold in protocol['folds']:
        fit=np.isin(g,fold['inner_fit_stages']); val=np.isin(g,fold['inner_val_stages'])
        train=fit|val; test=np.isin(g,fold['test_stages'])
        for name,x,params in [('handcrafted_rf',h,[8,16,None]),('handcrafted_knn',h,[3,5,9]),
                              ('imagenet_knn',e,[3,5,9]),('imagenet_ridge',e,[1.,10.,100.])]:
            best=None
            for param in params:
                scaler=StandardScaler().fit(x[fit])
                a,b=scaler.transform(x[fit]),scaler.transform(x[val])
                if name.endswith('rf'): est=RandomForestRegressor(n_estimators=400,max_depth=param,random_state=42,n_jobs=2)
                elif name.endswith('knn'): est=KNeighborsRegressor(n_neighbors=param,weights='distance')
                else: est=Ridge(alpha=param)
                est.fit(a,g[fit]/169.); ph=est.predict(b)*169
                metric=score(g[val],ph)
                if best is None or metric<best[0]: best=(metric,param)
            scaler=StandardScaler().fit(x[train]); a,b=scaler.transform(x[train]),scaler.transform(x[test])
            if name.endswith('rf'): est=RandomForestRegressor(n_estimators=400,max_depth=best[1],random_state=42,n_jobs=2)
            elif name.endswith('knn'): est=KNeighborsRegressor(n_neighbors=best[1],weights='distance')
            else: est=Ridge(alpha=best[1])
            est.fit(a,g[train]/169.); ph=est.predict(b)*169
            rows.append(dict(fold=fold['name'],model=name,seed=42,center=fold['center'],guard=fold['guard'],kind=fold['kind'],
                selected_parameter=best[1],inner_val_mae=best[0],mae_stage_macro=score(g[test],ph),mae_pass=float(np.abs(ph-g[test]).mean())))
            for j,idx in enumerate(np.flatnonzero(test)):
                predictions.append(dict(fold=fold['name'],model=name,seed=42,stage=int(g[idx]),pass_id=int(keys[idx,1]),prediction=float(ph[j]),error=float(abs(ph[j]-g[idx]))))
        print('BASELINES',fold['name'],[(r['model'],round(r['mae_stage_macro'],3)) for r in rows[-4:]],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'baseline_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'baseline_predictions.csv',index=False)

def cnn_worker(worker_id):
    import torch
    from torch import nn
    from torchvision import models
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.cuda.set_device(worker_id)
    device=torch.device(f'cuda:{worker_id}')
    protocol=json.load(open(OUT/'protocol.json'))
    h,e,g,keys=data()
    # Keep decoded uint8 images on GPU for repeatable, low-overhead repeated fits.
    images=torch.from_numpy(np.load(OUT/'sampled_images.npy').copy()).permute(0,3,1,2).contiguous().to(device)
    gp=np.repeat(g,3)
    targets=torch.tensor(gp/169.,dtype=torch.float32,device=device)
    c3=torch.tensor(np.where(gp<=40,0,np.where(gp<=110,1,2)),device=device,dtype=torch.long)
    c5=torch.tensor(np.minimum((gp-1)//34,4),device=device,dtype=torch.long)
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    weight=torch.load('/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth',map_location='cpu',weights_only=True)
    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.backbone=models.resnet18(weights=None)
            self.backbone.load_state_dict(weight); self.backbone.fc=nn.Identity()
            self.head3=nn.Linear(512,3); self.head5=nn.Linear(512,5); self.progress=nn.Linear(512,1)
        def forward(self,x):
            z=self.backbone(x)
            return self.head3(z),self.head5(z),torch.sigmoid(self.progress(z)).squeeze(1)
    def seed_all(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    def normalize(ids,augment=False):
        x=images[ids].float().div_(255.)
        if augment:
            flip=torch.rand(len(ids),device=device)<.5
            x[flip]=x[flip].flip(-1)
        return (x-mean)/std
    def make_net(seed):
        seed_all(seed); model=Net().to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=MAX_EPOCHS)
        return model,opt,sch
    def train_epoch(model,opt,ids):
        model.train(); order=ids[torch.randperm(len(ids),device=device)]
        total=0.
        for batch in order.split(64):
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                o3,o5,op=model(normalize(batch,True))
                loss=nn.functional.cross_entropy(o3,c3[batch])+nn.functional.cross_entropy(o5,c5[batch])+2*nn.functional.mse_loss(op,targets[batch])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total+=float(loss.detach())*len(batch)
        return total/len(ids)
    def predict(model,pass_mask):
        # ascending flat frame IDs keep 3 frames of each pass contiguous
        ids=torch.tensor(np.flatnonzero(np.repeat(pass_mask,3)),device=device)
        model.eval(); parts=[]
        with torch.inference_mode():
            for batch in ids.split(96):
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    output=model(normalize(batch))[2]
                parts.append(output.float().cpu().numpy())
        return np.concatenate(parts).reshape(-1,3).mean(1)*169.
    jobs=[(fold,seed) for fold in protocol['folds'] for seed in SEEDS]
    for job_id,(fold,seed) in enumerate(jobs):
        if job_id%2!=worker_id: continue
        tag=f"{fold['name']}_seed{seed}"
        result_path=OUT/(tag+'.json')
        if result_path.exists():
            print('SKIP_COMPLETE',tag,flush=True); continue
        fit=np.isin(g,fold['inner_fit_stages']); val=np.isin(g,fold['inner_val_stages'])
        train=fit|val; test=np.isin(g,fold['test_stages'])
        assert not set(g[train])&set(g[test])
        fitids=torch.tensor(np.flatnonzero(np.repeat(fit,3)),device=device)
        trainids=torch.tensor(np.flatnonzero(np.repeat(train,3)),device=device)
        start=time.time(); model,opt,sch=make_net(seed); history=[]; best=(float('inf'),None)
        for epoch in range(1,MAX_EPOCHS+1):
            loss=train_epoch(model,opt,fitids); sch.step()
            vm=score(g[val],predict(model,val)); history.append(dict(epoch=epoch,loss=loss,val_mae=vm))
            if vm<best[0]: best=(vm,epoch)
            if epoch%4==0: print('SELECT',worker_id,tag,epoch,round(vm,3),'best',best[1],flush=True)
        del model,opt,sch; torch.cuda.empty_cache()
        model,opt,sch=make_net(seed)
        for epoch in range(1,best[1]+1):
            train_epoch(model,opt,trainids); sch.step()
        ph=predict(model,test)
        checkpoint=OUT/(tag+'.pt')
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
            'fold':fold,'seed':seed,'epochs':best[1],'selection':'inner stage validation only'},checkpoint)
        predrows=[dict(fold=fold['name'],model='finetuned_resnet18',seed=seed,stage=int(g[idx]),pass_id=int(keys[idx,1]),
                      prediction=float(ph[j]),error=float(abs(ph[j]-g[idx]))) for j,idx in enumerate(np.flatnonzero(test))]
        pd.DataFrame(predrows).to_csv(OUT/(tag+'_predictions.csv'),index=False)
        row=dict(fold=fold['name'],model='finetuned_resnet18',seed=seed,center=fold['center'],guard=fold['guard'],kind=fold['kind'],
            selected_epochs=best[1],inner_val_mae=best[0],mae_stage_macro=score(g[test],ph),mae_pass=float(np.abs(ph-g[test]).mean()),
            elapsed_seconds=time.time()-start,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),history=history)
        result_path.write_text(json.dumps(row,indent=2))
        print('COMPLETE_CNN',worker_id,tag,'epochs',best[1],'macro_MAE',round(row['mae_stage_macro'],3),'seconds',round(row['elapsed_seconds'],1),flush=True)
        del model,opt,sch; torch.cuda.empty_cache()

def main():
    protocol=prepare()
    procs=[]
    for worker in [0,1]:
        log=open(OUT/f'worker{worker}.log','a',buffering=1)
        env=dict(os.environ); env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        procs.append((subprocess.Popen([sys.executable,'-u',__file__,'worker',str(worker)],stdout=log,stderr=subprocess.STDOUT,env=env),log))
    baseline_run(protocol)
    while any(p.poll() is None for p,l in procs):
        completed=len(list(OUT.glob('block*_seed*.json')))+len(list(OUT.glob('odd_even_seed*.json')))
        print('PROGRESS',completed,'/',len(protocol['folds'])*len(SEEDS),flush=True)
        time.sleep(20)
    codes=[p.wait() for p,l in procs]
    for p,l in procs:l.close()
    if any(c!=0 for c in codes): raise RuntimeError(f'workers failed: {codes}; inspect worker logs')
    rows=[]; preds=[]
    for fold in protocol['folds']:
        for seed in SEEDS:
            tag=f"{fold['name']}_seed{seed}"
            row=json.load(open(OUT/(tag+'.json'))); row.pop('history'); rows.append(row)
            preds.append(pd.read_csv(OUT/(tag+'_predictions.csv')))
    pd.DataFrame(rows).to_csv(OUT/'cnn_metrics.csv',index=False)
    pd.concat(preds,ignore_index=True).to_csv(OUT/'cnn_predictions.csv',index=False)
    import torch,torchvision,sklearn
    provenance={'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'sklearn':sklearn.__version__,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'manifest_sha256':hashlib.sha256((OUT/'sample_manifest.csv').read_bytes()).hexdigest(),
        'input_features_sha256':hashlib.sha256((OUT/'sampled_data.npz').read_bytes()).hexdigest()}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    (OUT/'COMPLETE').write_text('All 33 independently initialized selection/refit jobs completed.\n')
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker': cnn_worker(int(sys.argv[2]))
    else: main()
