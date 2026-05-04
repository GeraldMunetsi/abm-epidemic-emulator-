import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse, json, pickle
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy import stats as scipy_stats

from step0_model import create_hybrid_mlp_model
from utils_SIR   import compute_metrics, get_device, PARAM_MINS, PARAM_MAXS, BatchWrapper

MODELS_DIR  = Path("experiments/random-sampling/out/trained-models")
OOD_DATA_DIR = Path("experiments/random-sampling/data/Out_of_distribution_data")
RESULTS_DIR = Path("experiments/random-sampling/out/results/out_of_distribution_test_results")
PLOTS_DIR   = Path("experiments/random-sampling/out/plots/out_of_distribution_test_plots")
RATIO = 34.0

# Dataset 
#loops over simulations, converts them to tensors, extracts params and trajectories, normalizes params, computes R0, and stores in a list of dicts. Skips bad sims. Also prints param ranges and R0 distribution.
class OODSimDataset(Dataset):
    def __init__(self, sims, n_timepoints, ratio):
        self.samples = []
        skipped = 0
        mins = PARAM_MINS.astype(np.float32)
        maxs = PARAM_MAXS.astype(np.float32)
        for sim in sims:
            p = sim['params']
            tau   = float(p.get('tau',   p.get('beta',  0.0)))
            gamma = float(p.get('gamma', p.get('mu',    0.0)))
            rho   = float(p.get('rho',   0.0))
            out = sim['output']
            S = np.array(out.get('S', []), dtype=np.float32)
            I = np.array(out.get('I', []), dtype=np.float32)
            R = np.array(out.get('R', []), dtype=np.float32)
            if len(S) != n_timepoints or len(I) != n_timepoints or len(R) != n_timepoints:
                skipped += 1; continue
            traj = np.stack([S, I, R], axis=1)
            raw_params  = np.array([tau, gamma, rho], dtype=np.float32)
            norm_params = (raw_params - mins) / (maxs - mins + 1e-12)
            R0 = (tau / gamma) * ratio if gamma > 0 else float('nan')
            self.samples.append({
                'y'          : torch.tensor(traj,        dtype=torch.float32),
                'params_norm': torch.tensor(norm_params, dtype=torch.float32),
                'params_raw' : torch.tensor(raw_params,  dtype=torch.float32),
                'rho_raw'    : torch.tensor(rho,         dtype=torch.float32),
                'R0'         : torch.tensor(R0,          dtype=torch.float32),
            })
        if skipped: print(f"  Skipped {skipped} bad simulations")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def ood_collate(batch):
    return {
        'y'          : torch.stack([s['y']           for s in batch]),  # (B,T,3)
        'params_norm': torch.stack([s['params_norm'] for s in batch]),  # (B,3)
        'params_raw' : torch.stack([s['params_raw']  for s in batch]),  # (B,3)
        'rho_raw'    : torch.stack([s['rho_raw']     for s in batch]),  # (B,)
        'R0'         : torch.stack([s['R0']          for s in batch]),  # (B,)
    }

# ── Load dataset ──────────────────────────────────────────────────────────────

def load_ood_dataset(pkl_path):
    pkl_path = Path(pkl_path)
    print(f"\nLoading: {pkl_path.name}")
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    network  = raw['network']
    metadata = raw['metadata']
    sims     = raw['simulations']
    N            = int(network['N'])
    ratio        = float(network.get('ratio', RATIO))
    n_timepoints = int(metadata['n_timepoints'])
    print(f"  Simulations : {len(sims)}")
    print(f"  N={N:,}  ratio={ratio:.4f}  T={n_timepoints}")
    dataset = OODSimDataset(sims, n_timepoints, ratio)
    print(f"  Converted   : {len(dataset)} samples")
    p_all = torch.stack([s['params_norm'] for s in dataset.samples])
    names = ['tau','gamma','rho']
    print("\n  Param ranges (training=[0,1]):")
    for i,name in enumerate(names):
        lo,hi = p_all[:,i].min().item(), p_all[:,i].max().item()
        ood = ((p_all[:,i]<0)|(p_all[:,i]>1)).float().mean().item()
        print(f"    {name}: [{lo:+.3f},{hi:+.3f}]  {'OOD '+str(round(ood*100))+'%' if ood>0 else 'OK'}")
    r0s = np.array([s['R0'].item() for s in dataset.samples])
    r0v = r0s[~np.isnan(r0s)]
    print(f"\n  R0 range [{r0v.min():.3f},{r0v.max():.3f}]")
    print(f"  R0<0.8: {(r0v<0.8).sum()}  0.8-1.2: {((r0v>=0.8)&(r0v<=1.2)).sum()}  >1.2: {(r0v>1.2).sum()}")
    ood_count = int(((p_all<0)|(p_all>1)).any(dim=1).sum().item())
    meta = {'N':N,'ratio':ratio,'n_timepoints':n_timepoints,
            'n_samples':len(dataset),'ood_count':ood_count}
    return dataset, meta

# ── Load model ────────────────────────────────────────────────────────────────

def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck['config']
    r2  = ck.get('val_metrics',{}).get('R2',float('nan'))
    print(f"  {path.name}  epoch={ck.get('epoch','?')}  R²={r2:.4f}  N={cfg.get('total_population','?'):,}")
    m = create_hybrid_mlp_model(cfg)
    m.load_state_dict(ck['model_state_dict'], strict=True)
    m.to(device).eval()
    return m, cfg

# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_one(model, dataset, device, n_timesteps):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=ood_collate)
    all_preds, all_targets, all_params, all_r0 = [], [], [], []
    with torch.no_grad():
        for rb in loader:
            B = rb['y'].shape[0]
            # Build BatchWrapper — exactly what training uses
            batch = BatchWrapper(
                params_norm = rb['params_norm'].to(device),  # (B,3)
                rho_raw     = rb['rho_raw'].to(device),      # (B,)
                y           = rb['y'].to(device),            # (B,T,3)
            )
            assert batch.params_norm.shape == (B,3), f"Bad shape: {batch.params_norm.shape}"
            assert batch.rho_raw.shape    == (B,),   f"Bad shape: {batch.rho_raw.shape}"
            pred = model(batch, n_timesteps=n_timesteps)
            all_preds.append(pred.cpu())
            all_targets.append(rb['y'])
            all_params.append(rb['params_raw'])
            all_r0.append(rb['R0'])
    preds   = torch.cat(all_preds,   0)
    targets = torch.cat(all_targets, 0)
    params  = torch.cat(all_params,  0)
    r0s     = torch.cat(all_r0,      0)
    return preds, targets, params, r0s, compute_metrics(preds, targets)

def zone_metrics(preds, targets, r0s):
    r0 = r0s.numpy()
    zones = {'sub_critical':r0<0.8, 'threshold':(r0>=0.8)&(r0<=1.2), 'super_critical':r0>1.2}
    out = {}
    for name, mask in zones.items():
        if mask.sum()==0:
            out[name]={'n':0,'R2':float('nan'),'MAE_I':float('nan'),'MAE_S':float('nan'),'MAE_R':float('nan')}
            continue
        m = compute_metrics(preds[mask], targets[mask])
        out[name]={'n':int(mask.sum()),'R2':float(m['R2']),'MAE_I':float(m['MAE_I']),'MAE_S':float(m['MAE_S']),'MAE_R':float(m['MAE_R'])}
    return out

def evaluate_all(models_dir, dataset, meta, device):
    paths = sorted(models_dir.glob("best_balanced_mlp_model_*.pt"), key=lambda p:int(p.stem.split('_')[-1]))
    if not paths: raise FileNotFoundError(f"No models in {models_dir}")
    print(f"\n{'='*60}\nEVALUATING {len(paths)} REPLICATE(S)\n{'='*60}")
    T = meta['n_timepoints']
    results, pred_list, targets=[], [], None
    params = r0s = None
    for path in paths:
        model, cfg = load_model(path, device)
        preds,tgts,prms,r0v,metrics = evaluate_one(model, dataset, device, T)
        zm = zone_metrics(preds, tgts, r0v)
        pred_list.append(preds)
        if targets is None: targets,params,r0s = tgts,prms,r0v
        results.append({'model_name':path.name,'metrics':metrics,'zone_metrics':zm})
        print(f"  R²={metrics['R2']:.4f}  MAE_I={metrics['MAE_I']:.2f}  sub={zm['sub_critical']['R2']:.4f}  thr={zm['threshold']['R2']:.4f}  sup={zm['super_critical']['R2']:.4f}")
    ens_preds   = torch.stack(pred_list).mean(0)
    ens_metrics = compute_metrics(ens_preds, targets)
    ens_zones   = zone_metrics(ens_preds, targets, r0s)
    print(f"\n  Ensemble R²={ens_metrics['R2']:.4f}  MAE_I={ens_metrics['MAE_I']:.2f}")
    return results, ens_preds, targets, params, r0s, ens_metrics, ens_zones

# ── Aggregate stats ───────────────────────────────────────────────────────────

def agg_stats(results):
    out = {}
    for k in ['R2','MAE','MAE_S','MAE_I','MAE_R']:
        v = np.array([r['metrics'][k] for r in results])
        n = len(v); mean=float(v.mean()); std=float(v.std(ddof=1)) if n>1 else 0.
        sem = float(scipy_stats.sem(v)) if n>1 else 0.
        ci  = scipy_stats.t.interval(0.95,n-1,loc=mean,scale=sem) if n>1 else (mean,mean)
        out[k]={'mean':mean,'std':std,'ci_lower':float(ci[0]),'ci_upper':float(ci[1]),'values':v.tolist()}
    return out

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_preds(ens_preds, targets, params, r0s, plots_dir, n=6):
    plots_dir=Path(plots_dir); plots_dir.mkdir(parents=True,exist_ok=True)
    idx = np.argsort(r0s.numpy())
    picks = idx[np.linspace(0,len(idx)-1,n,dtype=int)]
    T = ens_preds.shape[1]; t = np.linspace(0,T-1,T)
    fig=plt.figure(figsize=(16,3.2*n))
    gs=gridspec.GridSpec(n,3,hspace=0.45,wspace=0.30)
    fig.suptitle("OOD Predictions vs Ground Truth",fontsize=13,fontweight='bold')
    titles=['S','I','R']; cols=['#378ADD','#1D9E75','#EF9F27']
    for row,i in enumerate(picks):
        lbl=f"τ={params[i,0]:.4f} γ={params[i,1]:.3f} ρ={params[i,2]:.4f} R₀={r0s[i]:.3f}"
        for c in range(3):
            ax=fig.add_subplot(gs[row,c])
            ax.plot(t,targets[i,:,c].numpy(),color='#CC3333',lw=1.5,ls='--',alpha=0.85,label='GT' if row==0 and c==1 else '')
            ax.plot(t,ens_preds[i,:,c].numpy(),color=cols[c],lw=1.8,label='Emulator' if row==0 and c==1 else '')
            if c==0: ax.set_ylabel(lbl,fontsize=7.5)
            if row==0: ax.set_title(titles[c],fontsize=10,fontweight='bold')
            if row==n-1: ax.set_xlabel('Time')
            if row==0 and c==1: ax.legend(fontsize=8)
            ax.grid(True,alpha=0.25)
    out=plots_dir/"ood_predictions.png"
    plt.savefig(out,dpi=180,bbox_inches='tight'); plt.close(); print(f"Saved: {out}")

def plot_zones(zm, plots_dir):
    plots_dir=Path(plots_dir); plots_dir.mkdir(parents=True,exist_ok=True)
    z=['sub_critical','threshold','super_critical']
    lbl=['Sub-critical\nR₀<0.8','Threshold\n0.8-1.2','Super-critical\nR₀>1.2']
    r2s=[zm[k]['R2'] for k in z]; ns=[zm[k]['n'] for k in z]
    fig,ax=plt.subplots(figsize=(7,4))
    bars=ax.bar(lbl,[0 if np.isnan(r) else r for r in r2s],color=['#378ADD','#1D9E75','#EF9F27'],alpha=0.85,edgecolor='white')
    for bar,r,n in zip(bars,r2s,ns):
        if not np.isnan(r): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f"R²={r:.3f}\n(n={n})",ha='center',va='bottom',fontsize=9)
    ax.axhline(0.9,color='#E24B4A',ls='--',lw=1.2,label='Target 0.90')
    ax.set_ylim(0,1.08); ax.set_ylabel('R²'); ax.set_title('OOD R² by Zone'); ax.legend(); ax.grid(axis='y',alpha=0.3)
    out=plots_dir/"ood_r2_by_zone.png"; plt.savefig(out,dpi=180,bbox_inches='tight'); plt.close(); print(f"Saved: {out}")

def plot_err(ens_preds, targets, r0s, plots_dir):
    plots_dir=Path(plots_dir); plots_dir.mkdir(parents=True,exist_ok=True)
    mae=(ens_preds[:,:,1]-targets[:,:,1]).abs().mean(dim=1).numpy()
    r0=r0s.numpy()
    fig,ax=plt.subplots(figsize=(8,4))
    sc=ax.scatter(r0,mae,c=r0,cmap='viridis',alpha=0.5,s=20)
    plt.colorbar(sc,ax=ax,label='R₀')
    ax.axvline(0.8,color='grey',ls=':',alpha=0.6); ax.axvline(1.2,color='#E24B4A',ls='--',lw=1.2,label='Threshold')
    ax.set_xlabel('R₀'); ax.set_ylabel('MAE_I'); ax.set_title('OOD Error vs R₀'); ax.legend(); ax.grid(alpha=0.25)
    out=plots_dir/"ood_error_vs_r0.png"; plt.savefig(out,dpi=180,bbox_inches='tight'); plt.close(); print(f"Saved: {out}")

# ── Save ─────────────────────────────────────────────────────────────────────

def save_results(results, agg, ens_m, ens_z, meta, output_dir):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    summary={'ood_meta':meta,'n_replicates':len(results),
             'replicate_metrics':[{'model':r['model_name'],'metrics':r['metrics'],'zone_metrics':r['zone_metrics']} for r in results],
             'aggregate':agg,'ensemble_metrics':ens_m,'ensemble_zones':ens_z}
    with open(output_dir/"ood_results.json",'w') as f: json.dump(summary,f,indent=2)
    rows=[]
    for r in results:
        row={'model':r['model_name']}; row.update(r['metrics'])
        for z,zm in r['zone_metrics'].items(): row[f'{z}_R2']=zm['R2']; row[f'{z}_n']=zm['n']
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir/"ood_results.csv",index=False)
    lines=["="*70,"OOD EVALUATION REPORT","="*70,
           f"  N={meta['N']:,}  T={meta['n_timepoints']}  samples={meta['n_samples']}  ood_params={meta['ood_count']}",""]
    for k,v in agg.items(): lines.append(f"  {k:8s}: {v['mean']:.4f} ± {v['std']:.4f}  CI[{v['ci_lower']:.4f},{v['ci_upper']:.4f}]")
    lines+=["","Zone R²:"]
    for z,zm in ens_z.items(): lines.append(f"  {z:20s}: R²={zm['R2']:.4f}  n={zm['n']}")
    lines+=["","Replicates:"]
    for r in results:
        m=r['metrics']; lines.append(f"  {r['model_name']}: R²={m['R2']:.4f} MAE_I={m['MAE_I']:.2f}")
    report="\n".join(lines)
    (output_dir/"OOD_REPORT.txt").write_text(report,encoding='utf-8')
    print(f"\nSaved to {output_dir}"); print("\n"+report)

# ── Main 

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--models_dir',default=str(MODELS_DIR))
    parser.add_argument('--data',default=str(OOD_DATA_DIR/"out_of_distribution_dataset.pkl"))
    parser.add_argument('--output_dir',default=str(RESULTS_DIR))
    parser.add_argument('--plots_dir',default=str(PLOTS_DIR))
    parser.add_argument('--n_plot',type=int,default=6)
    args=parser.parse_args()
    device=get_device()
    dataset,meta=load_ood_dataset(args.data)
    res,ens_preds,targets,params,r0s,ens_m,ens_z=evaluate_all(Path(args.models_dir),dataset,meta,device)
    ag=agg_stats(res)
    print("\nGenerating plots...")
    plot_preds(ens_preds,targets,params,r0s,args.plots_dir,args.n_plot)
    plot_zones(ens_z,args.plots_dir)
    plot_err(ens_preds,targets,r0s,args.plots_dir)
    save_results(res,ag,ens_m,ens_z,meta,args.output_dir)
    print(f"\nEnsemble R²={ens_m['R2']:.4f}  MAE_I={ens_m['MAE_I']:.2f}")