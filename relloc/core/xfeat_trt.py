"""XFeat with the backbone on TensorRT (xfeat_export.py -> trtexec engine) and the original post-processing in torch (heatmap, NMS,
top-k, descriptor interpolation). detect(x) returns the same list of dicts as XFeat.detectAndCompute.
Bench: python3 xfeat_trt.py <engine> -> times eager XFeat, eager backbone, TRT backbone, post-processing; keypoint/descriptor agreement."""
import os, sys, time, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.chdir(os.path.dirname(os.path.abspath(__file__)))
from modules.xfeat import XFeat
from modules.interpolator import InterpolateSparse2d
import lg_trt
class XFeatTRT:
    def __init__(self, xf, engine):
        self.xf = xf; self.E = lg_trt.TRTEngine(engine); self.interp = InterpolateSparse2d('bicubic'); self.near = InterpolateSparse2d('nearest'); self.bil = InterpolateSparse2d('bilinear')
    @torch.inference_mode()
    def detect(self, x, top_k=3000, thr=0.05):
        B, _, H1, W1 = x.shape
        o = self.E(img=x.float()); M1, K1, Hr = o['feats'].float(), o['kpts'].float(), o['heat'].float()
        return self.post(M1, K1, Hr, H1, W1, top_k, thr)
    def post(self, M1, K1, Hr, H1, W1, top_k, thr):
        if os.environ.get('XF_DENSE', '0') == '1': return self.post_dense(M1, K1, Hr, H1, W1, top_k, thr)
        xf = self.xf; B = M1.shape[0]
        M1 = F.normalize(M1, dim=1); K1h = xf.get_kpts_heatmap(K1); mk = xf.NMS(K1h, threshold=thr, kernel_size=5)
        sc = (self.near(K1h, mk, H1, W1) * self.bil(Hr, mk, H1, W1)).squeeze(-1); sc[torch.all(mk == 0, dim=-1)] = -1
        if os.environ.get('XF_TOPK', '1') == '1': idx = torch.topk(sc, min(top_k, sc.shape[-1]), dim=-1).indices   # partial sort (same set and order)
        else: idx = torch.argsort(-sc)
        mx = torch.gather(mk[..., 0], -1, idx)[:, :top_k]; my = torch.gather(mk[..., 1], -1, idx)[:, :top_k]
        mk = torch.cat([mx[..., None], my[..., None]], -1); sc = torch.gather(sc, -1, idx)[:, :top_k]
        fe = F.normalize(xf.interpolator(M1, mk, H=H1, W=W1), dim=-1); valid = sc > 0
        return [{'keypoints': mk[b][valid[b]].float(), 'scores': sc[b][valid[b]], 'descriptors': fe[b][valid[b]]} for b in range(B)]
def _post_dense(self, M1, K1, Hr, H1, W1, top_k, thr):
    """dense NMS + scoring + top-k in one pass (no per-image nonzero, no host sync): local maxima above thr of the keypoint heatmap,
    score = heatmap x reliability (bilinear, as the sparse path samples it), top_k per image by score."""
    xf = self.xf; B = M1.shape[0]
    M1 = F.normalize(M1, dim=1); K1h = xf.get_kpts_heatmap(K1)                                   # (B, 1, H, W)
    lm = F.max_pool2d(K1h, 5, stride=1, padding=2); keep = (K1h == lm) & (K1h > thr)
    rel = F.interpolate(Hr, size=(H1, W1), mode='bilinear', align_corners=False)
    sc = torch.where(keep, K1h * rel, torch.full_like(K1h, -1.0)).view(B, -1)
    val, idx = torch.topk(sc, top_k, dim=1); mx = (idx % W1).float(); my = torch.div(idx, W1, rounding_mode='floor').float()
    mk = torch.stack([mx, my], -1); fe = F.normalize(xf.interpolator(M1, mk, H=H1, W=W1), dim=-1); valid = val > 0
    return [{'keypoints': mk[b][valid[b]], 'scores': val[b][valid[b]], 'descriptors': fe[b][valid[b]]} for b in range(B)]
XFeatTRT.post_dense = _post_dense
if __name__ == '__main__':
    import cv2
    eng = sys.argv[1]; dev = 'cuda'; xf = XFeat(top_k=3000); xf.dev = dev; XT = XFeatTRT(xf, eng)
    frames = sorted(f for f in os.listdir('data') if f.endswith('.jpg'))[:4]; ims = [cv2.imread('data/' + f, 0) for f in frames]
    x = torch.from_numpy(np.stack(ims)).float()[:, None].to(dev)
    def tmed(f, n=15, warm=4):
        for _ in range(warm): f()
        ts = []
        for _ in range(n): torch.cuda.synchronize(); t0 = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        return 1000 * float(np.median(ts))
    res = {}
    with torch.inference_mode():
        res['eager_detect_ms'] = tmed(lambda: xf.detectAndCompute(x, top_k=3000))
        res['eager_backbone_ms'] = tmed(lambda: xf.net(x))
        res['trt_backbone_ms'] = tmed(lambda: XT.E(img=x))
        o = xf.net(x); res['post_ms'] = tmed(lambda: XT.post(o[0], o[1], o[2], 800, 1280, 3000, 0.05))
        res['trt_detect_ms'] = tmed(lambda: XT.detect(x))
        a = xf.detectAndCompute(x, top_k=3000); b = XT.detect(x)
        agree = []; dsim = []
        for fa, fb in zip(a, b):
            ka = fa['keypoints'].cpu().numpy(); kb = fb['keypoints'].cpu().numpy()
            sa = {tuple(v) for v in np.round(ka).astype(int)}; sb = {tuple(v) for v in np.round(kb).astype(int)}; agree.append(len(sa & sb) / max(len(sa), 1))
            n = min(len(ka), len(kb)); same = np.all(np.abs(ka[:n] - kb[:n]) < 0.5, 1)
            if same.any(): dsim.append(float((fa['descriptors'][:n][same] * fb['descriptors'][:n][same]).sum(1).mean()))
        res['kpt_overlap'] = float(np.mean(agree)); res['desc_cos_same_kpt'] = float(np.mean(dsim)) if dsim else None
    print(json.dumps(res)); json.dump(res, open('xfeat_trt_bench.json', 'w'))
