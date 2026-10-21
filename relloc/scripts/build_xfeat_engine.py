#!/usr/bin/env python3
"""Build the TensorRT engine of the XFeat backbone for this device (an engine runs only on the GPU and TensorRT version it was built
with): core/data/xfeat_b4_800x1280.onnx (relloc2 xfeat_export.py: static batch 4 = the four cameras, 800 x 1280) -> fp16 engine.
Same as `trtexec --onnx=core/data/xfeat_b4_800x1280.onnx --fp16 --saveEngine=<out>`. Then set RELLOC_XFEAT_ENGINE=<out>.
usage: python3 build_xfeat_engine.py [out=~/relloc_engines/xfeat_b4_800x1280_fp16.engine]"""
import os, sys, time
import tensorrt as trt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ONNX = os.path.join(ROOT, 'core', 'data', 'xfeat_b4_800x1280.onnx')
out = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/relloc_engines/xfeat_b4_800x1280_fp16.engine'); os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
log = trt.Logger(trt.Logger.WARNING); b = trt.Builder(log)
try: net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))   # TensorRT 8; deprecated (ignored) in 10
except Exception: net = b.create_network(0)
p = trt.OnnxParser(net, log)
if not p.parse(open(ONNX, 'rb').read()): raise SystemExit('ONNX parse failed: ' + '; '.join(str(p.get_error(i)) for i in range(p.num_errors)))
cfg = b.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
t0 = time.time(); ser = b.build_serialized_network(net, cfg)
if ser is None: raise SystemExit('engine build failed')
open(out, 'wb').write(ser); print(f'{out}: {os.path.getsize(out) // 1024} KB, TensorRT {trt.__version__}, built in {time.time() - t0:.0f} s')
