"""Where relloc's files are; importing this puts core/ (the evaluated relloc2 modules) and the vendored XFeat / LighterGlue on sys.path."""
import os, sys
LIVE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(LIVE)
CORE = os.path.join(ROOT, 'core'); AF = os.path.join(ROOT, 'third_party', 'accelerated_features')
PCA = os.path.join(CORE, 'data', 'desc_pca_tumvi.npz')
for _p in (AF, CORE):
    if _p not in sys.path: sys.path.insert(0, _p)
ENGINE = '~/relloc_engines/xfeat_b4_800x1280_fp16.engine'   # where scripts/build_xfeat_engine.py writes this drone's XFeat engine


def xfeat_backend(value, ncam, wh, exists=os.path.exists):
    """RELLOC_XFEAT_ENGINE -> (engine file, how it was chosen), or (None, why XFeat runs in PyTorch). auto (the default; also empty): the
    TensorRT engine at ENGINE when it exists and the rig is 4 cameras of 1280 x 800, else PyTorch. torch: PyTorch. Any other value: that
    engine file, which must exist (the node exits otherwise)."""
    v = (value or '').strip(); rig = ncam == 4 and tuple(wh) == (1280, 800)
    if v.lower() == 'torch': return None, 'RELLOC_XFEAT_ENGINE=torch'
    if v.lower() in ('', 'auto'):
        p = os.path.expanduser(ENGINE)
        if not rig: return None, f'auto: the engine is built for 4 cameras of 1280 x 800, not {ncam} of {wh[0]} x {wh[1]}'
        if not exists(p): return None, f'auto: no engine at {p} (build it on this drone: python3 relloc/scripts/build_xfeat_engine.py)'
        return p, 'auto'
    p = os.path.abspath(os.path.expanduser(v))
    if not exists(p): raise SystemExit(f'RELLOC_XFEAT_ENGINE {v}: no such file (scripts/build_xfeat_engine.py)')
    if not rig: raise SystemExit('RELLOC_XFEAT_ENGINE: the engine is built for 4 cameras of 1280 x 800')
    return p, 'RELLOC_XFEAT_ENGINE'
