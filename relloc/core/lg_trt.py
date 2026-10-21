"""TensorRT execution of the exported matcher (lg_export.py -> ONNX -> trtexec engine) and a drop-in replacement for LGGraph.match_many.
TRTEngine: generic wrapper (TensorRT 10 python API, torch tensors in and out on the current CUDA stream).
LGTRT(G): same interface as lggraph.LGGraph.match_many; equal-size batches run on the engine, other batches fall back to G.
Env LG_ENGINE (default lg_L5_fp16.engine)."""
import os, numpy as np, torch, tensorrt as trt
from kornia.feature.lightglue import normalize_keypoints, filter_matches
_DT = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool}
class TRTEngine:
    def __init__(self, path):
        self.logger = trt.Logger(trt.Logger.WARNING); rt = trt.Runtime(self.logger)
        with open(path, 'rb') as f: self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    def __call__(self, **kw):
        keep = {}
        for n in self.inputs:
            t = kw[n].to(_DT[self.engine.get_tensor_dtype(n)]).contiguous(); keep[n] = t
            self.ctx.set_input_shape(n, tuple(t.shape)); self.ctx.set_tensor_address(n, t.data_ptr())
        outs = {}
        for n in self.outputs:
            o = torch.empty(tuple(self.ctx.get_tensor_shape(n)), dtype=_DT[self.engine.get_tensor_dtype(n)], device='cuda'); outs[n] = o
            self.ctx.set_tensor_address(n, o.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return outs
class LGTRT:
    def __init__(self, G, engine=None):
        self.G = G; self.dev = G.dev; self.E = TRTEngine(engine or os.environ.get('LG_ENGINE', 'lg_L5_fp16.engine'))
    def match_many(self, pairs, min_conf=0.2, return_conf=False):
        if not pairs: return ([], []) if return_conf else []
        nA = {len(a['keypoints']) for a, _ in pairs}; nB = {len(b['keypoints']) for _, b in pairs}
        if len(nA) > 1 or len(nB) > 1: return self.G.match_many(pairs, min_conf, return_conf)
        with torch.inference_mode():
            k0 = torch.stack([a['keypoints'] for a, _ in pairs]).float(); k1 = torch.stack([b['keypoints'] for _, b in pairs]).float()
            d0 = torch.stack([a['descriptors'] for a, _ in pairs]).float(); d1 = torch.stack([b['descriptors'] for _, b in pairs]).float()
            s0 = torch.tensor([list(a['image_size']) for a, _ in pairs], device=self.dev); s1 = torch.tensor([list(b['image_size']) for _, b in pairs], device=self.dev)
            sc = self.E(k0=normalize_keypoints(k0, s0), k1=normalize_keypoints(k1, s1), d0=d0, d1=d1)['scores']
            m0, m1, ms0, ms1 = filter_matches(sc.float(), min_conf)
        out = []; conf = []
        for k in range(len(pairs)):
            valid = m0[k] > -1; i0 = torch.where(valid)[0]; i1 = m0[k][valid]
            out.append(torch.stack([i0, i1], -1).cpu().numpy()); conf.append(ms0[k][valid].float().cpu().numpy())
        return (out, conf) if return_conf else out
