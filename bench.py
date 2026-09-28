"""Compare CPU vs NPU inference speed and the full pipeline (features + model)."""
import time, numpy as np
from runtime import make_session
from features import feature_vector, FEAT_DIM, SR, WIN_SEC

N = 1000
x = np.random.randn(1, FEAT_DIM).astype(np.float32)
audio = np.random.randn(SR * WIN_SEC).astype(np.float32) * 0.01

t = time.perf_counter()
for _ in range(50):
    feature_vector(audio, 30.0)
print(f"feature extraction (CPU/numpy): {(time.perf_counter()-t)/50*1000:.2f} ms per 3 s window")

for name, npu in [("CPU", False), ("NPU", True)]:
    sess, label = make_session("model/model.onnx", use_npu=npu)
    for _ in range(20):
        sess.run(None, {"features": x})
    ts = []
    for _ in range(N):
        t = time.perf_counter(); sess.run(None, {"features": x}); ts.append((time.perf_counter() - t) * 1000)
    print(f"{name:3s} requested -> ran on {label:14s} mean {np.mean(ts):.3f} ms  p95 {np.percentile(ts,95):.3f} ms")
