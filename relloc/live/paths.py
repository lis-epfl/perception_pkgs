"""Where relloc's files are; importing this puts core/ (the evaluated relloc2 modules) and the vendored XFeat / LighterGlue on sys.path."""
import os, sys
LIVE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(LIVE)
CORE = os.path.join(ROOT, 'core'); AF = os.path.join(ROOT, 'third_party', 'accelerated_features')
PCA = os.path.join(CORE, 'data', 'desc_pca_tumvi.npz')
for _p in (AF, CORE):
    if _p not in sys.path: sys.path.insert(0, _p)
