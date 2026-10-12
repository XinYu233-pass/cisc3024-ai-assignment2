"""AI-written compact diagonal VBLL adaptation (ICLR 2024).
Run: OPENBLAS_NUM_THREADS=1 python run_experiment.py
No network, GPU, or downloaded dataset required.
"""
from pathlib import Path
import json, csv, platform, time, warnings
import numpy as np
import scipy, sklearn
from scipy.optimize import minimize, check_grad
from scipy.special import softmax, logsumexp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def objective(t, h, y, prior, bayes):
    n,d=h.shape; k=10; m=t[:d*k].reshape(d,k)
    z=h@m
    if bayes:
        r=t[d*k:].reshape(d,k); v=np.exp(r)
        a=z+0.5*(h*h)@v
    else: a=z
    p=softmax(a,axis=1)
    loss=np.mean(logsumexp(a,axis=1)-z[np.arange(n),y])
    residual=p.copy(); residual[np.arange(n),y]-=1
    gm=h.T@residual/n+m/(n*prior)
    if bayes:
        loss+=0.5*np.sum((m*m+v)/prior-1+np.log(prior)-r)/n
        gr=0.5*((h*h).T@p)*v/n+0.5*(v/prior-1)/n
        g=np.r_[gm.ravel(),gr.ravel()]
    else:
        loss+=0.5*np.sum(m*m)/(n*prior); g=gm.ravel()
    return loss,g

def fit(h,y,prior,bayes,start=None):
    d=h.shape[1]; m=np.zeros(d*10) if start is None else start
    t=np.r_[m,np.full(d*10,-5.)] if bayes else m
    bounds=[(None,None)]*(d*10)+([(-14,4)]*(d*10) if bayes else [])
    res=minimize(objective,t,args=(h,y,prior,bayes),jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':600,'ftol':1e-10,'gtol':1e-6})
    return res

def predict(t,h,bayes,seed=123,samples=512):
    d=h.shape[1]; m=t[:d*10].reshape(d,10)
    if not bayes:return softmax(h@m,axis=1)
    # Each marginal logit is exactly Gaussian for fixed features and Gaussian weights.
    # Sampling independent logits per image suffices for pointwise predictive probabilities.
    v=np.exp(t[d*10:].reshape(d,10)); mu=h@m; sd=np.sqrt((h*h)@v)
    rng=np.random.default_rng(seed); p=np.zeros_like(mu)
    for _ in range(samples):p+=softmax(mu+sd*rng.normal(size=mu.shape),axis=1)
    return p/samples

def metrics(y,p):
    pred=p.argmax(1); c=p.max(1); correct=pred==y; ece=0.
    for j in range(10):
        mask=(c>=j/10)&((c<(j+1)/10) if j<9 else (c<=1))
        if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-c[mask].mean())
    return dict(accuracy=float(accuracy_score(y,pred)),nll=float(log_loss(y,p,labels=np.arange(10))),ece=float(ece),brier=float(np.mean(np.sum((p-np.eye(10)[y])**2,axis=1))),confidence=float(c.mean()))

def main():
    begin=time.time(); x,y=load_digits(return_X_y=True); x=x/16.
    rng=np.random.default_rng(0); h=rng.normal(size=(12,4)); yy=rng.integers(10,size=12); t=np.r_[rng.normal(size=40)*.1,np.full(40,-3.)]
    err=check_grad(lambda q:objective(q,h,yy,1.,True)[0],lambda q:objective(q,h,yy,1.,True)[1],t)
    assert err<1e-5,err
    rows=[]; logs=[]; counts=[]; feature_logs=[]
    for seed in [0,1,2]:
        it,ie=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
        itr,iv=train_test_split(it,test_size=.25,stratify=y[it],random_state=seed)
        counts.append({'seed':seed,'train':len(itr),'validation':len(iv),'test':len(ie)})
        # Supervised feature learning sees only the training partition.
        feature=MLPClassifier(hidden_layer_sizes=(32,),activation='relu',solver='lbfgs',alpha=1.,max_iter=2000,random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            feature.fit(x[itr],y[itr])
        feature_logs.append(dict(seed=seed,iterations=int(feature.n_iter_),warnings=[str(w.message) for w in caught]))
        def features(xx):return np.c_[np.maximum(0,xx@feature.coefs_[0]+feature.intercepts_[0]),np.ones(len(xx))]
        ht,hv=features(x[itr]),features(x[iv]); chosen={}
        for bayes in [False,True]:
            trials=[]
            for prior in [.1,1.,10.]:
                res=fit(ht,y[itr],prior,bayes)
                val=metrics(y[iv],predict(res.x,hv,bayes,seed=seed+100))['nll']
                trials.append((val,prior,res))
                logs.append(dict(seed=seed,model='VBLL' if bayes else 'MAP',prior=prior,validation_nll=val,success=bool(res.success),message=str(res.message),iterations=int(res.nit),objective=float(res.fun)))
            val,prior,res=min(trials,key=lambda a:a[0]); chosen[bayes]=res.x
            for noise in [0.,.2,.4]:
                corrupted=np.clip(x[ie]+np.random.default_rng(seed+200).normal(0,noise,x[ie].shape),0,1)
                p=predict(res.x,features(corrupted),bayes,seed=seed+300)
                assert np.all(np.isfinite(p)) and np.allclose(p.sum(1),1)
                rows.append(dict(seed=seed,model='VBLL' if bayes else 'MAP',noise=noise,prior=prior,**metrics(y[ie],p)))
                if seed==0 and noise==0 and bayes:
                    cm=confusion_matrix(y[ie],p.argmax(1)); np.savez(ROOT/'predictions_seed0.npz',indices=ie,y=y[ie],probabilities=p)
                    fig,ax=plt.subplots(figsize=(5.4,4)); im=ax.imshow(cm,cmap='Blues'); ax.set(xticks=range(10),yticks=range(10),xlabel='Predicted digit',ylabel='True digit',title='VBLL: clean test set (seed 0)');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(ROOT/'confusion.png',dpi=160);plt.close(fig)
        print('Completed seed',seed,flush=True)
    with open(ROOT/'results.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for model in ['MAP','VBLL']:
        for noise in [0.,.2,.4]:
            group=[r for r in rows if r['model']==model and r['noise']==noise]
            entry=dict(model=model,noise=noise)
            for key in ['accuracy','nll','ece','brier','confidence']:
                vals=[r[key] for r in group];entry[key+'_mean']=float(np.mean(vals));entry[key+'_std']=float(np.std(vals,ddof=1))
            summary.append(entry)
    meta={'dataset':'sklearn load_digits','splits':counts,'gradient_check_absolute_error':float(err),'optimizer_logs':logs,'feature_training_logs':feature_logs,'summary':summary,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'elapsed_seconds':time.time()-begin}
    (ROOT/'results.json').write_text(json.dumps(meta,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(8,3.3))
    for model in ['MAP','VBLL']:
        g=[r for r in summary if r['model']==model]
        for ax,key in zip(axes,['accuracy','nll']):
            ax.errorbar([r['noise'] for r in g],[r[key+'_mean'] for r in g],yerr=[r[key+'_std'] for r in g],marker='o',capsize=3,label=model)
            ax.set(xlabel='Gaussian pixel noise (standard deviation)',ylabel='Accuracy' if key=='accuracy' else 'Negative log likelihood');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'robustness.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2));print('Gradient check:',err)
if __name__=='__main__':main()
