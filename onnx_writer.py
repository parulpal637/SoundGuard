"""Minimal ONNX writer (no `onnx` package needed, so it works on Windows on ARM).
Builds the PCA reconstruction-error scorer as a graph of plain Sub/Mul/MatMul ops,
which are all supported by the Qualcomm QNN (HTP/NPU) execution provider."""
import numpy as np


def _varint(n):
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field, wire):
    return _varint((field << 3) | wire)


def _len(field, data):
    return _key(field, 2) + _varint(len(data)) + data


def _str(field, s):
    return _len(field, s.encode())


def _int(field, v):
    return _key(field, 0) + _varint(v)


def _tensor(name, arr):
    arr = np.ascontiguousarray(arr, np.float32)
    b = b"".join(_int(1, d) for d in arr.shape)
    b += _int(2, 1)                    # FLOAT
    b += _str(8, name)
    b += _len(9, arr.tobytes())
    return b


def _value_info(name, shape):
    dims = b"".join(_len(1, _int(1, d)) for d in shape)
    tensor_type = _int(1, 1) + _len(2, dims)
    return _str(1, name) + _len(2, _len(1, tensor_type))


def _node(op, inputs, outputs, name):
    b = b"".join(_str(1, i) for i in inputs)
    b += b"".join(_str(2, o) for o in outputs)
    return b + _str(3, name) + _str(4, op)


def build_pca_scorer(mean, scale, V):
    """score = || xs - xs V V^T ||^2  with  xs = (x - mean) * scale.
    mean, scale: [D]; V: [D, k].  Input 'features' [1, D] -> output 'score' [1, 1]."""
    D, k = V.shape
    inits = {
        "mean": mean.reshape(1, D), "scale": scale.reshape(1, D),
        "V": V, "Vt": V.T.copy(), "ones": np.ones((D, 1)),
    }
    nodes = [
        _node("Sub", ["features", "mean"], ["c"], "center"),
        _node("Mul", ["c", "scale"], ["xs"], "scale"),
        _node("MatMul", ["xs", "V"], ["z"], "project"),
        _node("MatMul", ["z", "Vt"], ["r"], "reconstruct"),
        _node("Sub", ["xs", "r"], ["d"], "residual"),
        _node("Mul", ["d", "d"], ["sq"], "square"),
        _node("MatMul", ["sq", "ones"], ["score"], "sum"),
    ]
    graph = b"".join(_len(1, n) for n in nodes)
    graph += _str(2, "fan_health_scorer")
    graph += b"".join(_len(5, _tensor(n, a)) for n, a in inits.items())
    graph += _len(11, _value_info("features", [1, D]))
    graph += _len(12, _value_info("score", [1, 1]))
    model = _int(1, 8) + _str(2, "fan-health") + _len(7, graph) + _len(8, _int(2, 13))
    return model


def save(path, model_bytes):
    with open(path, "wb") as f:
        f.write(model_bytes)
