import json
import numpy as np
import onnxruntime as ort
from features import feature_vector


def make_session(path, use_npu=True):
    """Try the Snapdragon NPU (QNN / HTP); fall back to CPU. Returns (session, label)."""
    avail = ort.get_available_providers()
    if use_npu and "QNNExecutionProvider" in avail:
        try:
            sess = ort.InferenceSession(path, providers=[
                ("QNNExecutionProvider", {"backend_path": "QnnHtp.dll",
                                          "enable_htp_fp16_precision": "1"}),
                "CPUExecutionProvider"])
            if sess.get_providers()[0] == "QNNExecutionProvider":
                return sess, "NPU (QNN/HTP)"
        except Exception as e:  # noqa
            print("[warn] NPU session failed, using CPU:", e)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"]), "CPU"


class Scorer:
    def __init__(self, model="model/model.onnx", meta="model/meta.json", use_npu=True):
        self.session, self.backend = make_session(model, use_npu)
        with open(meta) as f:
            self.meta = json.load(f)
        self.threshold = self.meta["threshold"]

    def score(self, audio, cpu_percent):
        x = feature_vector(audio, cpu_percent)[None, :]
        s = float(self.session.run(None, {"features": x})[0][0, 0])
        return s, s / self.threshold      # ratio >= 1  ->  abnormal
