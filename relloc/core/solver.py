"""Gravity-aligned generalized relative pose solvers between two multi-camera rigs.
Unknown T = (R_z(psi), t): X_A = R X_B + t  (maps rig-gravity frame B -> A). v=(c,s,tx,ty,tz).
Correspondence record arrays (all in the rg frames):
  cA (n,3) camera center in A, dA (n,3) unit bearing in A;  XB (n,3) 3D point in B (nan if none)
  cB, dB, XA likewise.  """
import numpy as np, os as _os
from scipy.optimize import least_squares
def skew(v):
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
def _rows_p2p(cA,dA,XB):
    """3 linear rows per correspondence: M v = r with v=(c,s,t)."""
    n=len(dA); S=np.zeros((n,3,3)); S[:,0,1]=-dA[:,2];S[:,0,2]=dA[:,1];S[:,1,0]=dA[:,2];S[:,1,2]=-dA[:,0];S[:,2,0]=-dA[:,1];S[:,2,1]=dA[:,0]
    x,y,z=XB[:,0],XB[:,1],XB[:,2]
    P1=np.stack([x,y,np.zeros(n)],1); P2=np.stack([-y,x,np.zeros(n)],1); P3=np.stack([np.zeros(n),np.zeros(n),z],1)
    M=np.zeros((n,3,5)); M[:,:,0]=np.einsum('nij,nj->ni',S,P1); M[:,:,1]=np.einsum('nij,nj->ni',S,P2); M[:,:,2:]=S
    r=np.einsum('nij,nj->ni',S,cA)-np.einsum('nij,nj->ni',S,P3)
    return M,r
def _rows_33(XA,XB):
    n=len(XA); x,y,z=XB[:,0],XB[:,1],XB[:,2]
    M=np.zeros((n,3,5)); M[:,0,0]=x;M[:,1,0]=y; M[:,0,1]=-y;M[:,1,1]=x; M[:,:,2:]=np.eye(3)
    r=XA.copy(); r[:,2]-=z
    return M,r
def solve_2pt(M,r):
    """Batched: M (N,k,5), r (N,k). Solve with null-space + unit-circle constraint. Returns (N,2,5) candidate v (nan if none)."""
    N=M.shape[0]
    U,s,Vt=np.linalg.svd(M,full_matrices=False)   # (N,k,5),(N,5),(N,5,5)
    sinv=np.where(s>1e-6*s[:,:1],1/np.maximum(s,1e-12),0.0); sinv[:,4]=0  # drop null direction
    v0=np.einsum('nji,nj->ni',Vt,sinv*np.einsum('nkj,nk->nj',U,r))
    v1=Vt[:,4,:]
    a=v1[:,0]**2+v1[:,1]**2; b=2*(v0[:,0]*v1[:,0]+v0[:,1]*v1[:,1]); c=v0[:,0]**2+v0[:,1]**2-1
    disc=b*b-4*a*c; ok=(disc>=0)&(a>1e-12)
    sq=np.sqrt(np.maximum(disc,0)); l1=(-b+sq)/(2*np.maximum(a,1e-12)); l2=(-b-sq)/(2*np.maximum(a,1e-12))
    out=np.full((N,2,5),np.nan); out[ok,0]=v0[ok]+l1[ok,None]*v1[ok]; out[ok,1]=v0[ok]+l2[ok,None]*v1[ok]
    return out
def solve_ls(M,r):
    """M (k,5) stacked rows -> LS v, normalized (c,s)."""
    v,*_=np.linalg.lstsq(M.reshape(-1,5),r.reshape(-1),rcond=None); nrm=np.hypot(v[0],v[1]); v[:2]/=max(nrm,1e-9); return v
def v2T(v):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_euler('z',np.arctan2(v[1],v[0])), np.array(v[2:5])
def apply_v(v,X):
    c,s=v[0],v[1]; return np.stack([c*X[:,0]-s*X[:,1]+v[2], s*X[:,0]+c*X[:,1]+v[3], X[:,2]+v[4]],1)
def inv_v(v):
    c,s=v[0],v[1]; t=v[2:5]; ti=-np.array([c*t[0]+s*t[1], -s*t[0]+c*t[1], t[2]]); return np.array([c,-s,*ti])
def inv_v_batch(H):
    """vectorized inv_v over (N,5); NaN rows stay NaN"""
    c,s=H[:,0],H[:,1]; tx,ty,tz=H[:,2],H[:,3],H[:,4]; return np.stack([c,-s,-(c*tx+s*ty),-(-s*tx+c*ty),-tz],1)
def solve_ls_batch(M,r):
    """batched min-norm LS for stacked rows M (N,k,5), r (N,k) via pinv (same solution as solve_ls); (c,s) normalised"""
    v=np.einsum('nij,nj->ni',np.linalg.pinv(M),r); nrm=np.hypot(v[:,0],v[:,1]); v[:,:2]/=np.maximum(nrm,1e-9)[:,None]; return v
def ang_res(cA,dA,XB_in_A):
    w=XB_in_A-cA; nw=np.linalg.norm(w,axis=1); cos=np.einsum('ij,ij->i',w,dA)/np.maximum(nw,1e-9)
    return np.degrees(np.arccos(np.clip(cos,-1,1)))
class Corr:
    """Pooled correspondences. Each has A-side (cA,dA,XA?) and B-side (cB,dB,XB?)."""
    def __init__(self,cA,dA,XA,cB,dB,XB,camA,camB):
        self.cA=cA;self.dA=dA;self.XA=XA;self.cB=cB;self.dB=dB;self.XB=XB;self.camA=camA;self.camB=camB;self.n=len(dA)
        self.hasA=~np.isnan(XA[:,0]); self.hasB=~np.isnan(XB[:,0])
    def residuals(self,v):
        """angular residual (deg) per corr: min over available directions (B->A, A->B)."""
        res=np.full(self.n,np.inf)
        if self.hasB.any():
            res[self.hasB]=ang_res(self.cA[self.hasB],self.dA[self.hasB],apply_v(v,self.XB[self.hasB]))
        if self.hasA.any():
            vi=inv_v(v); r2=ang_res(self.cB[self.hasA],self.dB[self.hasA],apply_v(vi,self.XA[self.hasA]))
            res[self.hasA]=np.minimum(res[self.hasA],r2)
        return res
    def score(self,v,thr):
        r=self.residuals(v); return (r<thr).sum(), r
def ransac(corr,thr=0.6,iters=3000,seed=0,min_inl=8):
    """2-point gravity-aligned generalized P2P RANSAC over pooled correspondences (uses B-side 3D, and A-side 3D via inverse). """
    rng=np.random.default_rng(seed)
    cands=[]
    idxB=np.where(corr.hasB)[0]; idxA=np.where(corr.hasA)[0]
    hyps=[]
    if len(idxB)>=2:
        S=rng.choice(idxB,(iters,2)); M,r=_rows_p2p(corr.cA[S.ravel()],corr.dA[S.ravel()],corr.XB[S.ravel()])
        M=M.reshape(iters,6,5); r=r.reshape(iters,6); hyps.append(solve_2pt(M,r).reshape(-1,5))
    if len(idxA)>=2:
        S=rng.choice(idxA,(iters,2)); M,r=_rows_p2p(corr.cB[S.ravel()],corr.dB[S.ravel()],corr.XA[S.ravel()])
        M=M.reshape(iters,6,5); r=r.reshape(iters,6); H=solve_2pt(M,r).reshape(-1,5); hyps.append(inv_v_batch(H))
    both=np.where(corr.hasA&corr.hasB)[0]
    if len(both)>=2:
        S=rng.choice(both,(iters//2,2)); M,r=_rows_33(corr.XA[S.ravel()],corr.XB[S.ravel()]); hyps.append(solve_ls_batch(M.reshape(len(S),6,5),r.reshape(len(S),6)))
    if not hyps: return None
    H=np.concatenate(hyps); H=H[np.isfinite(H).all(1)]
    if len(H)==0: return None
    scores=score_many(corr,H,thr)
    bi=int(np.argmax(scores)); v=H[bi]
    inl=corr.residuals(v)<thr
    if inl.sum()<min_inl: return None
    v=(refine_gn if REFINE=='gn' else refine)(corr,v,inl,thr)
    inl=corr.residuals(v)<thr
    return {'v':v,'inl':inl,'ninl':int(inl.sum()),'ninl_raw':int(scores[bi]),'scores':scores,'v_raw':H[bi]}
def score_many(corr,H,thr):
    """count inliers for many hypotheses H (N,5) - vectorized over hypotheses in chunks."""
    N=len(H); out=np.zeros(N,int); CH=256
    for s in range(0,N,CH):
        h=H[s:s+CH]; c=h[:,0:1];sn=h[:,1:2]
        cnt=np.zeros(len(h),int)
        if corr.hasB.any():
            X=corr.XB[corr.hasB]; cA=corr.cA[corr.hasB]; dA=corr.dA[corr.hasB]
            Xa=np.stack([c*X[None,:,0]-sn*X[None,:,1]+h[:,2:3], sn*X[None,:,0]+c*X[None,:,1]+h[:,3:4], X[None,:,2]+h[:,4:5]],2)  # (H,n,3)
            w=Xa-cA[None]; cos=np.einsum('hnj,nj->hn',w,dA)/np.maximum(np.linalg.norm(w,axis=2),1e-9)
            cnt+= (cos>np.cos(np.radians(thr))).sum(1)
        if corr.hasA.any():
            X=corr.XA[corr.hasA]; cB=corr.cB[corr.hasA]; dB=corr.dB[corr.hasA]
            # inverse transform: X_B = R^T (X_A - t)
            tx,ty,tz=h[:,2:3],h[:,3:4],h[:,4:5]; ux=X[None,:,0]-tx; uy=X[None,:,1]-ty; uz=X[None,:,2]-tz
            Xb=np.stack([c*ux+sn*uy, -sn*ux+c*uy, uz],2)
            w=Xb-cB[None]; cos=np.einsum('hnj,nj->hn',w,dB)/np.maximum(np.linalg.norm(w,axis=2),1e-9)
            cnt+= (cos>np.cos(np.radians(thr))).sum(1)
        out[s:s+CH]=cnt
    return out
def refine(corr,v,inl,thr):
    psi0=np.arctan2(v[1],v[0]); x0=np.array([psi0,*v[2:5]])
    sub=Corr(corr.cA[inl],corr.dA[inl],corr.XA[inl],corr.cB[inl],corr.dB[inl],corr.XB[inl],corr.camA[inl],corr.camB[inl])
    def f(x):
        vv=np.array([np.cos(x[0]),np.sin(x[0]),*x[1:4]]); r=sub.residuals(vv); r[~np.isfinite(r)]=0; return r
    sol=least_squares(f,x0,loss='soft_l1',f_scale=thr)
    x=sol.x; return np.array([np.cos(x[0]),np.sin(x[0]),*x[1:4]])

def _signed_res(corr,v,inl):
    """signed residual: for each inlier correspondence the tangent-plane component of unit(w)-d, w = predicted direction (2 dof each; in radians for small angles)"""
    out=[]
    if corr.hasB.any():
        m=inl&corr.hasB; w=apply_v(v,corr.XB[m])-corr.cA[m]; u=w/np.maximum(np.linalg.norm(w,axis=1,keepdims=True),1e-9); d=corr.dA[m]; e=u-np.einsum('ij,ij->i',u,d)[:,None]*d; out.append(e)
    if corr.hasA.any():
        m=inl&corr.hasA&~corr.hasB; vi=inv_v(v); w=apply_v(vi,corr.XA[m])-corr.cB[m]; u=w/np.maximum(np.linalg.norm(w,axis=1,keepdims=True),1e-9); d=corr.dB[m]; e=u-np.einsum('ij,ij->i',u,d)[:,None]*d; out.append(e)
    return np.concatenate(out).ravel() if out else np.zeros(0)
def refine_gn(corr,v,inl,thr,iters=3):
    """Gauss-Newton with Cauchy weights (scale = thr) on the signed residual; numeric Jacobian by central differences, all vectorized."""
    x=np.array([np.arctan2(v[1],v[0]),*v[2:5]]); h=np.array([1e-4,1e-3,1e-3,1e-3]); sc=np.radians(thr)
    def f(x): return _signed_res(corr,np.array([np.cos(x[0]),np.sin(x[0]),*x[1:4]]),inl)
    for _ in range(iters):
        r=f(x); n=len(r)
        if n<8: break
        J=np.zeros((n,4))
        for k in range(4):
            xp=x.copy(); xp[k]+=h[k]; xm=x.copy(); xm[k]-=h[k]; J[:,k]=(f(xp)-f(xm))/(2*h[k])
        rn=np.linalg.norm(r.reshape(-1,3),axis=1); wgt=np.repeat(1.0/(1.0+(rn/sc)**2),3)   # Cauchy
        A=J.T@(wgt[:,None]*J)+1e-9*np.eye(4); g=J.T@(wgt*r)
        try: dx=np.linalg.solve(A,-g)
        except np.linalg.LinAlgError: break
        x=x+dx
        if np.abs(dx).max()<1e-7: break
    return np.array([np.cos(x[0]),np.sin(x[0]),*x[1:4]])
REFINE=_os.environ.get('SOLVER_REFINE','gn')
# ---- torch-accelerated hypothesis scoring (drop-in for score_many) ----
def score_many_torch(corr,H,thr):
    import torch
    dev='cuda'; Ht=torch.from_numpy(np.ascontiguousarray(H,dtype=np.float32)).to(dev); c=Ht[:,0:1];sn=Ht[:,1:2]; cth=float(np.cos(np.radians(thr)))
    out=torch.zeros(len(H),dtype=torch.int32,device=dev)
    def block(X,C,D,inverse):
        X=torch.from_numpy(X.astype(np.float32)).to(dev); C=torch.from_numpy(C.astype(np.float32)).to(dev); D=torch.from_numpy(D.astype(np.float32)).to(dev)
        CH=512
        for s in range(0,len(H),CH):
            h=Ht[s:s+CH]; cc=h[:,0:1]; ss=h[:,1:2]
            if not inverse:
                xa=cc*X[None,:,0]-ss*X[None,:,1]+h[:,2:3]; ya=ss*X[None,:,0]+cc*X[None,:,1]+h[:,3:4]; za=X[None,:,2]+h[:,4:5]
            else:
                ux=X[None,:,0]-h[:,2:3]; uy=X[None,:,1]-h[:,3:4]; uz=X[None,:,2]-h[:,4:5]
                xa=cc*ux+ss*uy; ya=-ss*ux+cc*uy; za=uz
            wx=xa-C[None,:,0]; wy=ya-C[None,:,1]; wz=za-C[None,:,2]
            cos=(wx*D[None,:,0]+wy*D[None,:,1]+wz*D[None,:,2])/torch.sqrt(wx*wx+wy*wy+wz*wz+1e-12)
            out[s:s+CH]+=(cos>cth).sum(1).to(torch.int32)
    if corr.hasB.any(): block(corr.XB[corr.hasB],corr.cA[corr.hasB],corr.dA[corr.hasB],False)
    if corr.hasA.any(): block(corr.XA[corr.hasA],corr.cB[corr.hasA],corr.dB[corr.hasA],True)
    return out.cpu().numpy()
import os as _os
if _os.environ.get('SOLVER_TORCH','1')=='1':
    score_many=score_many_torch


def solution_sigma(corr,v,inl):
    """Self-diagnostic of a 4-DOF solution: (sigma_yaw deg, sigma_t m, ray-spread p90 deg) from the inlier residual Jacobian (finite differences).
    Narrow-FOV / low-depth-spread inlier sets leave yaw and lateral translation nearly ambiguous -> large sigma_yaw and small ray spread."""
    psi=np.degrees(np.arctan2(v[1],v[0])); p=np.r_[psi,v[2:5]]
    def res(p):
        vv=np.r_[np.cos(np.radians(p[0])),np.sin(np.radians(p[0])),p[1:4]]; return corr.residuals(vv)[inl]
    r0=res(p); J=np.zeros((len(r0),4)); h=[0.05,0.005,0.005,0.005]
    for k in range(4):
        pp=p.copy(); pp[k]+=h[k]; pm=p.copy(); pm[k]-=h[k]; J[:,k]=(res(pp)-res(pm))/(2*h[k])
    sig=max(float(np.sqrt(np.mean(r0**2))),0.05)
    try: C=sig**2*np.linalg.inv(J.T@J+1e-9*np.eye(4)); sy=float(np.sqrt(C[0,0])); st=float(np.sqrt(np.trace(C[1:,1:])))
    except np.linalg.LinAlgError: sy,st=np.inf,np.inf
    d=corr.dB[inl]; m=d.mean(0); m/=max(np.linalg.norm(m),1e-9); spread=float(np.percentile(np.degrees(np.arccos(np.clip(np.abs(d@m),-1,1))),90))
    return sy,st,spread
# ---- optimization round 2 (2026-09-06 afternoon): torch hypothesis generation, guided sampling, batched-Jacobian GN ----
def _rows_p2p_t(cA,dA,XB):
    import torch
    n=len(dA); z=torch.zeros(n,device=dA.device,dtype=dA.dtype)
    S=torch.stack([torch.stack([z,-dA[:,2],dA[:,1]],1),torch.stack([dA[:,2],z,-dA[:,0]],1),torch.stack([-dA[:,1],dA[:,0],z],1)],1)  # (n,3,3)
    x,y,zz=XB[:,0],XB[:,1],XB[:,2]
    P1=torch.stack([x,y,z],1); P2=torch.stack([-y,x,z],1); P3=torch.stack([z,z,zz],1)
    M=torch.cat([(S@P1[:,:,None]),(S@P2[:,:,None]),S],2)   # (n,3,5)
    r=(S@cA[:,:,None])[:,:,0]-(S@P3[:,:,None])[:,:,0]
    return M,r
def _rows_33_t(XA,XB):
    import torch
    n=len(XA); x,y,z=XB[:,0],XB[:,1],XB[:,2]; o=torch.zeros(n,device=XA.device,dtype=XA.dtype); e=torch.eye(3,device=XA.device,dtype=XA.dtype)[None].expand(n,3,3)
    M=torch.cat([torch.stack([x,y,o],1)[:,:,None],torch.stack([-y,x,o],1)[:,:,None],e],2); r=XA.clone(); r[:,2]-=z
    return M,r
def solve_2pt_t(M,r):
    """torch batched 2-point solve: M (N,6,5), r (N,6) -> (N,2,5) (nan if none). Null direction from eigh(M^T M)."""
    import torch
    A=M.transpose(1,2)@M; b=(M.transpose(1,2)@r[:,:,None])[:,:,0]
    w,V=torch.linalg.eigh(A)                       # ascending eigenvalues; V (N,5,5) columns
    v1=V[:,:,0]; winv=torch.where(w>1e-6*w[:,-1:],1/w.clamp_min(1e-12),torch.zeros_like(w)); winv[:,0]=0
    v0=(V*winv[:,None,:])@(V.transpose(1,2)@b[:,:,None]); v0=v0[:,:,0]
    a=v1[:,0]**2+v1[:,1]**2; bb=2*(v0[:,0]*v1[:,0]+v0[:,1]*v1[:,1]); c=v0[:,0]**2+v0[:,1]**2-1
    disc=bb*bb-4*a*c; ok=(disc>=0)&(a>1e-12); sq=torch.sqrt(disc.clamp_min(0)); l1=(-bb+sq)/(2*a.clamp_min(1e-12)); l2=(-bb-sq)/(2*a.clamp_min(1e-12))
    out=torch.full((len(M),2,5),float('nan'),device=M.device,dtype=M.dtype)
    out[:,0]=torch.where(ok[:,None],v0+l1[:,None]*v1,out[:,0]); out[:,1]=torch.where(ok[:,None],v0+l2[:,None]*v1,out[:,1])
    return out
def _sample(rng,idx,k,n,w=None):
    """k samples of n distinct indices from idx, optionally weighted (guided sampling)."""
    if w is None: return rng.choice(idx,(k,n))
    p=w[idx].astype(np.float64); p=p/p.sum(); return np.stack([rng.choice(idx,n,replace=False,p=p) for _ in range(k)]) if n>1 and len(idx)<64 else rng.choice(idx,(k,n),p=p)
def hyps_torch(corr,iters,seed=0,w=None):
    """hypothesis generation on the GPU (same sampling scheme as ransac(); different random stream)."""
    import torch
    dev='cuda'; rng=np.random.default_rng(seed); out=[]
    f=lambda a: torch.from_numpy(np.ascontiguousarray(a,dtype=np.float32)).to(dev)
    idxB=np.where(corr.hasB)[0]; idxA=np.where(corr.hasA)[0]
    if len(idxB)>=2:
        S=_sample(rng,idxB,iters,2,w).ravel(); M,r=_rows_p2p_t(f(corr.cA[S]),f(corr.dA[S]),f(corr.XB[S])); out.append(solve_2pt_t(M.reshape(iters,6,5),r.reshape(iters,6)).reshape(-1,5))
    if len(idxA)>=2:
        S=_sample(rng,idxA,iters,2,w).ravel(); M,r=_rows_p2p_t(f(corr.cB[S]),f(corr.dB[S]),f(corr.XA[S])); H=solve_2pt_t(M.reshape(iters,6,5),r.reshape(iters,6)).reshape(-1,5)
        c,s=H[:,0],H[:,1]; tx,ty,tz=H[:,2],H[:,3],H[:,4]; out.append(torch.stack([c,-s,-(c*tx+s*ty),-(-s*tx+c*ty),-tz],1))
    both=np.where(corr.hasA&corr.hasB)[0]
    if len(both)>=2:
        S=_sample(rng,both,iters//2,2,w).ravel(); M,r=_rows_33_t(f(corr.XA[S]),f(corr.XB[S])); M=M.reshape(-1,6,5); r=r.reshape(-1,6)
        A=M.transpose(1,2)@M+1e-9*torch.eye(5,device=dev); v=torch.linalg.solve(A,(M.transpose(1,2)@r[:,:,None]))[:,:,0]; nrm=torch.hypot(v[:,0],v[:,1]).clamp_min(1e-9); v=torch.cat([v[:,:2]/nrm[:,None],v[:,2:]],1); out.append(v)
    if not out: return None
    H=torch.cat(out); H=H[torch.isfinite(H).all(1)]; return H
def score_many_t(corr,Ht,thr,CH=None):
    """like score_many_torch but takes a torch H on the device and scores in as few launches as memory allows."""
    import torch
    dev=Ht.device; cth=float(np.cos(np.radians(thr))); N=len(Ht); out=torch.zeros(N,dtype=torch.int32,device=dev)
    def block(X,C,D,inverse):
        X=torch.from_numpy(X.astype(np.float32)).to(dev); C=torch.from_numpy(C.astype(np.float32)).to(dev); D=torch.from_numpy(D.astype(np.float32)).to(dev)
        ch=CH or max(256,min(N,int(3e7//max(len(X),1))))
        for s in range(0,N,ch):
            h=Ht[s:s+ch]; cc=h[:,0:1]; ss=h[:,1:2]
            if not inverse:
                xa=cc*X[None,:,0]-ss*X[None,:,1]+h[:,2:3]; ya=ss*X[None,:,0]+cc*X[None,:,1]+h[:,3:4]; za=X[None,:,2]+h[:,4:5]
            else:
                ux=X[None,:,0]-h[:,2:3]; uy=X[None,:,1]-h[:,3:4]; uz=X[None,:,2]-h[:,4:5]; xa=cc*ux+ss*uy; ya=-ss*ux+cc*uy; za=uz
            wx=xa-C[None,:,0]; wy=ya-C[None,:,1]; wz=za-C[None,:,2]
            cos=(wx*D[None,:,0]+wy*D[None,:,1]+wz*D[None,:,2])*torch.rsqrt(wx*wx+wy*wy+wz*wz+1e-12)
            out[s:s+ch]+=(cos>cth).sum(1).to(torch.int32)
    if corr.hasB.any(): block(corr.XB[corr.hasB],corr.cA[corr.hasB],corr.dA[corr.hasB],False)
    if corr.hasA.any(): block(corr.XA[corr.hasA],corr.cB[corr.hasA],corr.dB[corr.hasA],True)
    return out
def _signed_res_many(corr,V,inl):
    """signed residuals for K hypotheses V (K,5) at once -> (K, n_res*3)."""
    out=[]
    c=V[:,0:1]; s=V[:,1:2]
    if corr.hasB.any():
        m=inl&corr.hasB; X=corr.XB[m]; C=corr.cA[m]; D=corr.dA[m]
        w=np.stack([c*X[None,:,0]-s*X[None,:,1]+V[:,2:3], s*X[None,:,0]+c*X[None,:,1]+V[:,3:4], X[None,:,2]+V[:,4:5]],2)-C[None]
        u=w/np.maximum(np.linalg.norm(w,axis=2,keepdims=True),1e-9); e=u-np.einsum('knj,nj->kn',u,D)[:,:,None]*D[None]; out.append(e.reshape(len(V),-1))
    if corr.hasA.any():
        m=inl&corr.hasA&~corr.hasB; X=corr.XA[m]; C=corr.cB[m]; D=corr.dB[m]
        ux=X[None,:,0]-V[:,2:3]; uy=X[None,:,1]-V[:,3:4]; uz=X[None,:,2]-V[:,4:5]
        w=np.stack([c*ux+s*uy,-s*ux+c*uy,uz],2)-C[None]
        u=w/np.maximum(np.linalg.norm(w,axis=2,keepdims=True),1e-9); e=u-np.einsum('knj,nj->kn',u,D)[:,:,None]*D[None]; out.append(e.reshape(len(V),-1))
    return np.concatenate(out,1) if out else np.zeros((len(V),0))
def refine_gn_fast(corr,v,inl,thr,iters=3):
    """same as refine_gn (central-difference Jacobian, Cauchy weights) with the 9 residual evaluations per iteration batched into one call."""
    x=np.array([np.arctan2(v[1],v[0]),*v[2:5]]); h=np.array([1e-4,1e-3,1e-3,1e-3]); sc=np.radians(thr)
    def tov(X): return np.stack([np.cos(X[:,0]),np.sin(X[:,0]),X[:,1],X[:,2],X[:,3]],1)
    for _ in range(iters):
        Xs=np.repeat(x[None],9,0)
        for k in range(4): Xs[1+2*k,k]+=h[k]; Xs[2+2*k,k]-=h[k]
        Rm=_signed_res_many(corr,tov(Xs),inl); r=Rm[0]; n=len(r)
        if n<8: break
        J=np.stack([(Rm[1+2*k]-Rm[2+2*k])/(2*h[k]) for k in range(4)],1)
        rn=np.linalg.norm(r.reshape(-1,3),axis=1); wgt=np.repeat(1.0/(1.0+(rn/sc)**2),3)
        A=J.T@(wgt[:,None]*J)+1e-9*np.eye(4); g=J.T@(wgt*r)
        try: dx=np.linalg.solve(A,-g)
        except np.linalg.LinAlgError: break
        x=x+dx
        if np.abs(dx).max()<1e-7: break
    return np.array([np.cos(x[0]),np.sin(x[0]),*x[1:4]])
HYP=_os.environ.get('SOLVER_HYP','np'); GUIDED=float(_os.environ.get('SOLVER_GUIDED','0'))
def ransac2(corr,thr=0.6,iters=2000,seed=0,min_inl=8,w=None):
    """ransac() with the round-2 options: SOLVER_HYP=torch (GPU hypotheses), SOLVER_GUIDED=p (sample weights w**p), batched GN."""
    import torch
    ww=None
    if w is not None and GUIDED>0: ww=np.maximum(w,1e-3)**GUIDED
    if HYP=='torch':
        Ht=hyps_torch(corr,iters,seed,ww)
        if Ht is None or len(Ht)==0: return None
        scores_t=score_many_t(corr,Ht,thr); bi=int(torch.argmax(scores_t)); v=Ht[bi].cpu().numpy().astype(np.float64); scores=scores_t.cpu().numpy(); H=None
    else:
        rng=np.random.default_rng(seed); hyps=[]
        idxB=np.where(corr.hasB)[0]; idxA=np.where(corr.hasA)[0]
        if len(idxB)>=2:
            S=_sample(rng,idxB,iters,2,ww); M,r=_rows_p2p(corr.cA[S.ravel()],corr.dA[S.ravel()],corr.XB[S.ravel()]); hyps.append(solve_2pt(M.reshape(iters,6,5),r.reshape(iters,6)).reshape(-1,5))
        if len(idxA)>=2:
            S=_sample(rng,idxA,iters,2,ww); M,r=_rows_p2p(corr.cB[S.ravel()],corr.dB[S.ravel()],corr.XA[S.ravel()]); hyps.append(inv_v_batch(solve_2pt(M.reshape(iters,6,5),r.reshape(iters,6)).reshape(-1,5)))
        both=np.where(corr.hasA&corr.hasB)[0]
        if len(both)>=2:
            S=_sample(rng,both,iters//2,2,ww); M,r=_rows_33(corr.XA[S.ravel()],corr.XB[S.ravel()]); hyps.append(solve_ls_batch(M.reshape(len(S),6,5),r.reshape(len(S),6)))
        if not hyps: return None
        H=np.concatenate(hyps); H=H[np.isfinite(H).all(1)]
        if len(H)==0: return None
        scores=score_many(corr,H,thr); bi=int(np.argmax(scores)); v=H[bi]
    v_raw=v.copy(); inl=corr.residuals(v)<thr
    if inl.sum()<min_inl: return None
    v=refine_gn_fast(corr,v,inl,thr); inl=corr.residuals(v)<thr
    return {'v':v,'inl':inl,'ninl':int(inl.sum()),'ninl_raw':int(scores[bi]),'scores':scores,'v_raw':v_raw}
# ---- closed-form 2-point solver (no LAPACK/cuSOLVER): tangent-plane rows (4x5), null vector by cofactors, particular solution by Cramer ----
def _det3(a): return a[:,0,0]*(a[:,1,1]*a[:,2,2]-a[:,1,2]*a[:,2,1])-a[:,0,1]*(a[:,1,0]*a[:,2,2]-a[:,1,2]*a[:,2,0])+a[:,0,2]*(a[:,1,0]*a[:,2,1]-a[:,1,1]*a[:,2,0])
def _det4(a):
    return a[:,0,0]*_det3(a[:,1:][:,:,[1,2,3]])-a[:,0,1]*_det3(a[:,1:][:,:,[0,2,3]])+a[:,0,2]*_det3(a[:,1:][:,:,[0,1,3]])-a[:,0,3]*_det3(a[:,1:][:,:,[0,1,2]])
def _tangent_rows(cA,dA,XB,xp):
    """per correspondence 2 rows (n,2,5) and rhs (n,2): e_k^T [P1 P2 I] v = e_k^T (c - P3), e_k basis of the plane normal to the bearing."""
    n=len(dA); z=xp.zeros((n,3)) if xp is np else torch.zeros(n,3,device=dA.device,dtype=dA.dtype)
    ax=z.copy() if xp is np else z.clone(); big=xp.abs(dA[:,2])<0.9
    if xp is np: ax[big,2]=1; ax[~big,0]=1
    else: ax[:,2]=big.to(dA.dtype); ax[:,0]=(~big).to(dA.dtype)
    e1=xp.cross(dA,ax,axis=1) if xp is np else torch.cross(dA,ax,dim=1); e1=e1/(xp.linalg.norm(e1,axis=1,keepdims=True) if xp is np else e1.norm(dim=1,keepdim=True))
    e2=xp.cross(dA,e1,axis=1) if xp is np else torch.cross(dA,e1,dim=1)
    x,y,zz=XB[:,0],XB[:,1],XB[:,2]; o=z[:,0]
    P1=xp.stack([x,y,o],1); P2=xp.stack([-y,x,o],1); rhs=cA-xp.stack([o,o,zz],1)
    rows=[]; r=[]
    for e in (e1,e2):
        rows.append(xp.stack([(e*P1).sum(1),(e*P2).sum(1),e[:,0],e[:,1],e[:,2]],1)); r.append((e*rhs).sum(1))
    return xp.stack(rows,1),xp.stack(r,1)
def solve_2pt_cf(M4,r4,xp):
    """M4 (N,4,5), r4 (N,4) -> (N,2,5) candidates (nan if none). Null vector by 4x4 cofactors; particular solution by Cramer's rule after dropping the column of the null vector's largest entry."""
    N=len(M4); cols=[[1,2,3,4],[0,2,3,4],[0,1,3,4],[0,1,2,4],[0,1,2,3]]
    if xp is np and _LAPACK_DET: return _solve_2pt_np(M4,r4,cols)
    v1=xp.stack([((-1)**k)*_det4(M4[:,:,cols[k]]) for k in range(5)],1)
    nv=xp.linalg.norm(v1,axis=1,keepdims=True) if xp is np else v1.norm(dim=1,keepdim=True); v1=v1/(nv+1e-30)
    kstar=xp.abs(v1).argmax(1); C=xp.array(cols) if xp is np else torch.tensor(cols,device=M4.device)
    sel=C[kstar]                                  # (N,4) columns kept
    ar=xp.arange(N); Mp=M4[ar[:,None,None],(xp.arange(4))[None,:,None],sel[:,None,:]]   # (N,4,4)
    D=_det4(Mp); vp=[]
    for i in range(4):
        Mi=Mp.copy() if xp is np else Mp.clone(); Mi[:,:,i]=r4; vp.append(_det4(Mi)/(D+1e-30*(D==0)))
    vp=xp.stack(vp,1)                             # (N,4) values at columns sel
    v0=xp.zeros((N,5)) if xp is np else torch.zeros(N,5,device=M4.device,dtype=M4.dtype)
    v0[ar[:,None],sel]=vp
    a=v1[:,0]**2+v1[:,1]**2; b=2*(v0[:,0]*v1[:,0]+v0[:,1]*v1[:,1]); c=v0[:,0]**2+v0[:,1]**2-1
    disc=b*b-4*a*c; ok=(disc>=0)&(a>1e-12)&(xp.abs(D)>1e-12)
    sq=xp.sqrt(xp.maximum(disc,0) if xp is np else disc.clamp_min(0)); den=2*(xp.maximum(a,1e-12) if xp is np else a.clamp_min(1e-12)); l1=(-b+sq)/den; l2=(-b-sq)/den
    o1=v0+l1[:,None]*v1; o2=v0+l2[:,None]*v1
    nan=xp.nan if xp is np else float('nan')
    o1=xp.where(ok[:,None],o1,nan); o2=xp.where(ok[:,None],o2,nan)
    return xp.stack([o1,o2],1)
import os as _os
_LAPACK_DET=_os.environ.get('HYP_DET','lapack')=='lapack'   # round 14: batched LAPACK determinants (2 calls) instead of ~800 small cofactor ops per solve; same numbers
def _solve_2pt_np(M4,r4,cols):
    N=len(M4); C=np.array(cols); ar=np.arange(N)
    v1=np.linalg.det(M4[:,:,C].transpose(0,2,1,3))*np.array([1.,-1.,1.,-1.,1.])     # (N,5): signed 4x4 minors -> null vector
    v1=v1/(np.linalg.norm(v1,axis=1,keepdims=True)+1e-30)
    sel=C[np.abs(v1).argmax(1)]; Mp=M4[ar[:,None,None],np.arange(4)[None,:,None],sel[:,None,:]]
    Mi=np.repeat(Mp[:,None],5,1)                                                    # [Mp, Mp with column i replaced by r4 (Cramer)]
    for i in range(4): Mi[:,1+i,:,i]=r4
    dd=np.linalg.det(Mi); D=dd[:,0]; vp=dd[:,1:]/(D+1e-30*(D==0))[:,None]
    v0=np.zeros((N,5)); v0[ar[:,None],sel]=vp
    a=v1[:,0]**2+v1[:,1]**2; b=2*(v0[:,0]*v1[:,0]+v0[:,1]*v1[:,1]); c=v0[:,0]**2+v0[:,1]**2-1
    disc=b*b-4*a*c; ok=(disc>=0)&(a>1e-12)&(np.abs(D)>1e-12)
    sq=np.sqrt(np.maximum(disc,0)); den=2*np.maximum(a,1e-12); l1=(-b+sq)/den; l2=(-b-sq)/den
    o1=np.where(ok[:,None],v0+l1[:,None]*v1,np.nan); o2=np.where(ok[:,None],v0+l2[:,None]*v1,np.nan)
    return np.stack([o1,o2],1)
def _hyp_p2p_cf(cA,dA,XB,S,xp):
    """S (k,2) sampled correspondence indices -> (2k,5) hypotheses."""
    rows,r=_tangent_rows(cA[S.ravel()],dA[S.ravel()],XB[S.ravel()],xp); k=len(S)
    return solve_2pt_cf(rows.reshape(k,4,5),r.reshape(k,4),xp).reshape(-1,5)
def _hyp_33_cf(XA,XB,S,xp):
    """exact 2-point gravity-aligned 3D-3D: yaw from the horizontal difference vectors, t from the centroids."""
    A0=XA[S[:,0]];A1=XA[S[:,1]];B0=XB[S[:,0]];B1=XB[S[:,1]]; da=A1-A0; db=B1-B0
    psi=(xp.arctan2 if xp is np else torch.atan2)(da[:,1],da[:,0])-(xp.arctan2 if xp is np else torch.atan2)(db[:,1],db[:,0]); c=xp.cos(psi); s=xp.sin(psi)
    mA=(A0+A1)/2; mB=(B0+B1)/2; t=xp.stack([mA[:,0]-(c*mB[:,0]-s*mB[:,1]), mA[:,1]-(s*mB[:,0]+c*mB[:,1]), mA[:,2]-mB[:,2]],1)
    return xp.concatenate([xp.stack([c,s],1),t],1) if xp is np else torch.cat([torch.stack([c,s],1),t],1)
def hyps_cf(corr,iters,seed=0,w=None,xp=np,dev='cuda'):
    """hypothesis generation with the closed-form solvers; xp=np (CPU) or torch (GPU). Same sampling scheme as ransac()."""
    rng=np.random.default_rng(seed); out=[]
    if xp is np: f=lambda a: a
    else: f=lambda a: torch.from_numpy(np.ascontiguousarray(a,dtype=np.float32)).to(dev)
    g=(lambda S: S) if xp is np else (lambda S: torch.from_numpy(S).to(dev))
    idxB=np.where(corr.hasB)[0]; idxA=np.where(corr.hasA)[0]
    if len(idxB)>=2: out.append(_hyp_p2p_cf(f(corr.cA),f(corr.dA),f(corr.XB),g(_sample(rng,idxB,iters,2,w)),xp))
    if len(idxA)>=2:
        H=_hyp_p2p_cf(f(corr.cB),f(corr.dB),f(corr.XA),g(_sample(rng,idxA,iters,2,w)),xp); c,s=H[:,0],H[:,1]; tx,ty,tz=H[:,2],H[:,3],H[:,4]; out.append(xp.stack([c,-s,-(c*tx+s*ty),-(-s*tx+c*ty),-tz],1))
    both=np.where(corr.hasA&corr.hasB)[0]
    if len(both)>=2: out.append(_hyp_33_cf(f(corr.XA),f(corr.XB),g(_sample(rng,both,iters//2,2,w)),xp))
    if not out: return None
    H=xp.concatenate(out) if xp is np else torch.cat(out); H=H[xp.isfinite(H).all(1)]
    return H
import torch
def ransac3(corr,thr=0.6,iters=2000,seed=0,min_inl=8,w=None,gpu=None):
    """closed-form hypotheses (CPU numpy or GPU torch, no LAPACK), GPU scoring, batched GN. gpu=None -> SOLVER_HYP=='torch'."""
    if gpu is None: gpu=(HYP=='torch')
    ww=None
    if w is not None and GUIDED>0: ww=np.maximum(w,1e-3)**GUIDED
    if gpu:
        Ht=hyps_cf(corr,iters,seed,ww,torch)
        if Ht is None or len(Ht)==0: return None
        scores_t=score_many_t(corr,Ht,thr); bi=int(torch.argmax(scores_t)); v=Ht[bi].cpu().numpy().astype(np.float64); scores=scores_t.cpu().numpy()
    else:
        H=hyps_cf(corr,iters,seed,ww,np)
        if H is None or len(H)==0: return None
        if torch.cuda.is_available():   # one-shot GPU scoring of CPU-generated hypotheses (the chunked score_many_torch costs 2.5x more on the Orin)
            scores=score_many_t(corr,torch.from_numpy(np.ascontiguousarray(H,dtype=np.float32)).cuda(),thr).cpu().numpy()
        else: scores=score_many(corr,H,thr)
        bi=int(np.argmax(scores)); v=H[bi]
    v_raw=v.copy(); inl=corr.residuals(v)<thr
    if inl.sum()<min_inl: return None
    v=refine_gn_fast(corr,v,inl,thr); inl=corr.residuals(v)<thr
    return {'v':v,'inl':inl,'ninl':int(inl.sum()),'ninl_raw':int(scores[bi]),'scores':scores,'v_raw':v_raw}
def residuals_many(corr,V):
    """unsigned angular residual (deg) for K hypotheses V (K,5) -> (K,n); same definition as Corr.residuals (min over the available directions)."""
    K=len(V); out=np.full((K,corr.n),np.inf); c=V[:,0:1]; s=V[:,1:2]
    def ang(w,D):
        u=w/np.maximum(np.linalg.norm(w,axis=2,keepdims=True),1e-9); return np.degrees(np.arccos(np.clip(np.einsum('knj,nj->kn',u,D),-1,1)))
    if corr.hasB.any():
        m=corr.hasB; X=corr.XB[m]; w=np.stack([c*X[None,:,0]-s*X[None,:,1]+V[:,2:3], s*X[None,:,0]+c*X[None,:,1]+V[:,3:4], X[None,:,2]+V[:,4:5]],2)-corr.cA[m][None]; out[:,m]=ang(w,corr.dA[m])
    if corr.hasA.any():
        m=corr.hasA; X=corr.XA[m]; ux=X[None,:,0]-V[:,2:3]; uy=X[None,:,1]-V[:,3:4]; uz=X[None,:,2]-V[:,4:5]; w=np.stack([c*ux+s*uy,-s*ux+c*uy,uz],2)-corr.cB[m][None]; out[:,m]=np.minimum(out[:,m],ang(w,corr.dB[m]))
    return out
def solution_sigma_fast(corr,v,inl):
    """solution_sigma with the 9 residual evaluations batched (identical numbers)."""
    psi=np.degrees(np.arctan2(v[1],v[0])); p=np.r_[psi,v[2:5]]; h=np.array([0.05,0.005,0.005,0.005])
    P=np.repeat(p[None],9,0)
    for k in range(4): P[1+2*k,k]+=h[k]; P[2+2*k,k]-=h[k]
    V=np.stack([np.cos(np.radians(P[:,0])),np.sin(np.radians(P[:,0])),P[:,1],P[:,2],P[:,3]],1)
    R=residuals_many(corr,V)[:,inl]; r0=R[0]; J=np.stack([(R[1+2*k]-R[2+2*k])/(2*h[k]) for k in range(4)],1)
    sig=max(float(np.sqrt(np.mean(r0**2))),0.05)
    try: C=sig**2*np.linalg.inv(J.T@J+1e-9*np.eye(4)); sy=float(np.sqrt(C[0,0])); st=float(np.sqrt(np.trace(C[1:,1:])))
    except np.linalg.LinAlgError: sy,st=np.inf,np.inf
    d=corr.dB[inl]; m=d.mean(0); m/=max(np.linalg.norm(m),1e-9); spread=float(np.percentile(np.degrees(np.arccos(np.clip(np.abs(d@m),-1,1))),90))
    return sy,st,spread
