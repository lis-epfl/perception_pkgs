"""Static-shape LighterGlue forward captured in a CUDA graph (for launch-bound GPUs such as the Orin NX).
Replicates kornia LightGlue._forward with early stopping and width pruning off; the dynamic part (match filtering, nonzero) runs outside the graph.
Usage: G=LGGraph(xf); IDX,CONF=G.match_many(pairs,min_conf) with pairs = list of (dicA,dicB) like lgbatch.match_many. Falls back to lgbatch for unequal sizes.
Env LG_GRAPH=0 disables capture (same static forward, eager); LG_HALF=1 runs the whole matcher in fp16 (weights and activations)."""
import os, numpy as np, torch
from kornia.feature.lightglue import normalize_keypoints, filter_matches
class LGGraph:
    def __init__(self,xf,use_graph=None,half=None):
        if xf.lighterglue is None:
            k=torch.zeros(1,8,2,device=xf.dev); d=torch.zeros(1,8,64,device=xf.dev); xf.match_lighterglue({'keypoints':k[0],'descriptors':d[0],'image_size':(1280,800)},{'keypoints':k[0],'descriptors':d[0],'image_size':(1280,800)},min_conf=0.2)
        self.net=xf.lighterglue.net; self.dev=xf.dev; self.use_graph=(os.environ.get('LG_GRAPH','1')=='1') if use_graph is None else use_graph; self.graphs={}
        self.net.conf.width_confidence=-1; self.net.conf.depth_confidence=-1
        self.layers=int(os.environ.get('LG_LAYERS','0')) or self.net.conf.n_layers   # fixed-depth exit: use the assignment head of layer L-1 (all heads are in the checkpoint)
        self.half=(os.environ.get('LG_HALF','0')=='1') if half is None else half; self.dt=torch.half if self.half else torch.float
        with torch.inference_mode(): self.net.to(self.dt)   # the model was built under inference_mode (XFeat wrapper): convert there too
    BUCKETS=(256,384,512,768,1024,1280,1536,2048,3072)
    def _static_forward(self,k0,k1,d0,d1,m0=None,m1=None):
        net=self.net
        d0=net.input_proj(d0); d1=net.input_proj(d1); e0=net.posenc(k0); e1=net.posenc(k1)
        L=self.layers
        if m0 is None:
            for i in range(L): d0,d1=net.transformers[i](d0,d1,e0,e1)
        else:   # padded batch: masks (B,L,1) bool -> (B,1,L,L) attention masks (kornia's masked_forward with an explicit head dimension)
            mm=(m0&m1.transpose(-1,-2))[:,None]; mm0=(m0&m0.transpose(-1,-2))[:,None]; mm1=(m1&m1.transpose(-1,-2))[:,None]
            for i in range(L):
                T=net.transformers[i]; d0=T.self_attn(d0,e0,mm0); d1=T.self_attn(d1,e1,mm1); d0,d1=T.cross_attn(d0,d1,mm)
        scores,_=net.log_assignment[L-1](d0,d1); return scores
    MAX_GRAPHS=int(os.environ.get('LG_MAX_GRAPHS','24'))   # round 15: 18+ shapes for mixed-rig peers (rig15.graph_shapes)
    def _get(self,B,M,N,masked=False):
        key=(B,M,N,masked)
        if key in self.graphs: return self.graphs[key]
        dev=self.dev; dt=self.dt
        with torch.inference_mode():
            k0=torch.zeros(B,M,2,device=dev,dtype=dt); k1=torch.zeros(B,N,2,device=dev,dtype=dt); d0=torch.zeros(B,M,64,device=dev,dtype=dt); d1=torch.zeros(B,N,64,device=dev,dtype=dt)
            m0=torch.ones(B,M,1,device=dev,dtype=torch.bool) if masked else None; m1=torch.ones(B,N,1,device=dev,dtype=torch.bool) if masked else None
        ent={'k0':k0,'k1':k1,'d0':d0,'d1':d1,'m0':m0,'m1':m1,'g':None,'out':None}
        if self.use_graph and sum(1 for e in self.graphs.values() if e['g'] is not None)<self.MAX_GRAPHS:
            s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.inference_mode():
                for _ in range(3): self._static_forward(k0,k1,d0,d1,m0,m1)
            torch.cuda.current_stream().wait_stream(s)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode=os.environ.get('CAPTURE_MODE','global')), torch.inference_mode(): ent['out']=self._static_forward(k0,k1,d0,d1,m0,m1)
            ent['g']=g
        self.graphs[key]=ent; return ent
    def scores(self,k0,k1,d0,d1,size0,size1,m0=None,m1=None):
        """k0 (B,M,2) pixel keypoints, d0 (B,M,64) descriptors (any float dtype), optional validity masks (B,M,1)/(B,N,1). Returns log-assignment scores (B,M+1,N+1) on the device."""
        B,M,_=k0.shape; N=k1.shape[1]; ent=self._get(B,M,N,m0 is not None)
        with torch.inference_mode():
            n0=normalize_keypoints(k0.float(),size0); n1=normalize_keypoints(k1.float(),size1)
            ent['k0'].copy_(n0); ent['k1'].copy_(n1); ent['d0'].copy_(d0); ent['d1'].copy_(d1)
            if m0 is not None: ent['m0'].copy_(m0); ent['m1'].copy_(m1)
            if ent['g'] is not None: ent['g'].replay(); return ent['out']
            return self._static_forward(ent['k0'],ent['k1'],ent['d0'],ent['d1'],ent['m0'],ent['m1'])
    def match_many(self,pairs,min_conf=0.2,return_conf=False):
        if not pairs: return ([],[]) if return_conf else []
        nA={len(a['keypoints']) for a,_ in pairs}; nB={len(b['keypoints']) for _,b in pairs}; dev=self.dev
        if len(nA)>1 or len(nB)>1:   # unequal sizes within the batch only: the masked path is exact but its attention cannot use the flash kernel (4x slower on the Orin)
            # unequal (or non-bucket) sizes: pad every side to one bucket length with validity masks (exact: masked attention), one graph per (B, bucket)
            L=max(max(nA),max(nB)); L=next((b for b in self.BUCKETS if b>=L),((L+255)//256)*256); B=len(pairs)
            k0=torch.zeros(B,L,2,device=dev); k1=torch.zeros(B,L,2,device=dev); d0=torch.zeros(B,L,64,device=dev); d1=torch.zeros(B,L,64,device=dev)
            m0=torch.zeros(B,L,1,device=dev,dtype=torch.bool); m1=torch.zeros(B,L,1,device=dev,dtype=torch.bool)
            for k,(a,b) in enumerate(pairs):
                na=len(a['keypoints']); nb=len(b['keypoints']); k0[k,:na]=a['keypoints'].float(); d0[k,:na]=a['descriptors'].float(); m0[k,:na]=True; k1[k,:nb]=b['keypoints'].float(); d1[k,:nb]=b['descriptors'].float(); m1[k,:nb]=True
            s0=torch.tensor([list(a['image_size']) for a,_ in pairs],device=dev); s1=torch.tensor([list(b['image_size']) for _,b in pairs],device=dev)
            sc=self.scores(k0,k1,d0,d1,s0,s1,m0,m1)
            with torch.inference_mode():
                sc=sc.float().clone(); neg=-1e9
                sc[:,:L,:L]=torch.where(m0&m1.transpose(1,2),sc[:,:L,:L],torch.full_like(sc[:,:L,:L],neg))   # padded rows/columns can only go to the dustbin
                m0_,m1_,ms0,ms1=filter_matches(sc,min_conf)
            out=[]; conf=[]
            for k,(a,b) in enumerate(pairs):
                na=len(a['keypoints']); nb=len(b['keypoints']); valid=(m0_[k]>-1); valid[na:]=False; i0=torch.where(valid)[0]; i1=m0_[k][valid]; ok=i1<nb
                out.append(torch.stack([i0[ok],i1[ok]],-1).cpu().numpy()); conf.append(ms0[k][valid][ok].float().cpu().numpy())
            return (out,conf) if return_conf else out
        k0=torch.stack([a['keypoints'] for a,_ in pairs]).float(); k1=torch.stack([b['keypoints'] for _,b in pairs]).float()
        d0=torch.stack([a['descriptors'] for a,_ in pairs]).float(); d1=torch.stack([b['descriptors'] for _,b in pairs]).float()
        s0=torch.tensor([list(a['image_size']) for a,_ in pairs],device=dev); s1=torch.tensor([list(b['image_size']) for _,b in pairs],device=dev)
        sc=self.scores(k0,k1,d0,d1,s0,s1)
        with torch.inference_mode(): m0,m1,ms0,ms1=filter_matches(sc.float(),min_conf)
        out=[]; conf=[]
        for k in range(len(pairs)):
            valid=m0[k]>-1; i0=torch.where(valid)[0]; i1=m0[k][valid]
            out.append(torch.stack([i0,i1],-1).cpu().numpy()); conf.append(ms0[k][valid].float().cpu().numpy())
        return (out,conf) if return_conf else out
