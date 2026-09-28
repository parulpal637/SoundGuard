"""Train the anomaly detector on NORMAL fan sound + CPU load, export model/model.onnx.

Idea: learn what a healthy laptop sounds like *for a given CPU load*. A window whose
sound does not fit that pattern (grinding, rattling, fan too loud/quiet for the load)
gets a high reconstruction error -> flagged as suspicious."""
import argparse, glob, json, os
import numpy as np
from features import feature_vector, FEAT_DIM, N_MELS, SR, WIN_SEC
import onnx_writer

LOAD_WEIGHT = 4.0      # how strongly the sound-vs-CPU-load relationship matters
VAR_KEEP, K_MAX = 0.90, 12


def load_folder(folder):
    sessions = []
    for p in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        d = np.load(p)
        X = np.stack([feature_vector(a, c) for a, c in zip(d["audio"], d["cpu"])])
        sessions.append(X)
    return sessions


def auc(neg, pos):
    allv = np.concatenate([neg, pos])
    ranks = allv.argsort().argsort() + 1
    return (ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="model")
    a = ap.parse_args()

    normal = load_folder(os.path.join(a.data, "normal"))
    if not normal:
        raise SystemExit("No data in data/normal. Run record.py first.")
    tr, va = [], []
    for X in normal:                       # per session: first 80% train, last 20% validation
        cut = max(1, int(len(X) * 0.8))
        tr.append(X[:cut]); va.append(X[cut:])
    Xtr, Xva = np.concatenate(tr), np.concatenate(va)
    print(f"normal windows: train={len(Xtr)} val={len(Xva)}")
    if len(Xtr) < 30:
        print("[warn] very little data - record more minutes per state for a reliable model")

    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    floor = np.full(FEAT_DIM, 0.25, np.float32); floor[-1] = 0.05
    sd = np.maximum(sd, floor)
    w = np.ones(FEAT_DIM, np.float32); w[-1] = LOAD_WEIGHT
    scale = (w / sd).astype(np.float32)

    Z = (Xtr - mu) * scale
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = np.cumsum(S ** 2) / np.sum(S ** 2)
    k = int(min(K_MAX, np.searchsorted(var, VAR_KEEP) + 1, FEAT_DIM - 1))
    V = Vt[:k].T.astype(np.float32)
    print(f"PCA components kept: {k} ({var[k-1]*100:.1f}% variance)")

    def err(X):
        z = (X - mu) * scale
        return ((z - z @ V @ V.T) ** 2).sum(1)

    e_tr, e_va = err(Xtr), err(Xva) if len(Xva) else err(Xtr)
    thr = float(1.25 * max(np.percentile(e_tr, 99), np.percentile(e_va, 99)))
    print(f"threshold = {thr:.2f}   false-alarm rate on normal val: {(e_va > thr).mean()*100:.1f}%")

    ab_dir = os.path.join(a.data, "abnormal")
    if os.path.isdir(ab_dir) and glob.glob(os.path.join(ab_dir, "*.npz")):
        Xab = np.concatenate(load_folder(ab_dir))
        e_ab = err(Xab)
        print(f"simulated-abnormal windows: {len(Xab)}  detected: {(e_ab > thr).mean()*100:.1f}%  "
              f"AUC: {auc(e_va, e_ab):.3f}")

    os.makedirs(a.out, exist_ok=True)
    onnx_writer.save(os.path.join(a.out, "model.onnx"),
                     onnx_writer.build_pca_scorer(mu.astype(np.float32), scale, V))
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"threshold": thr, "k": k, "load_weight": LOAD_WEIGHT,
                   "sr": SR, "win_sec": WIN_SEC, "feat_dim": FEAT_DIM}, f, indent=2)

    # sanity check: ONNX output must match numpy
    import onnxruntime as ort
    s = ort.InferenceSession(os.path.join(a.out, "model.onnx"), providers=["CPUExecutionProvider"])
    got = np.array([s.run(None, {"features": x[None]})[0][0, 0] for x in Xva[:20]])
    assert np.allclose(got, err(Xva[:20]), rtol=1e-3, atol=1e-3), "ONNX/numpy mismatch"
    print("Saved model/model.onnx (verified against numpy)")


if __name__ == "__main__":
    main()
