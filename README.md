# Fan Health Monitor - on-device acoustic anomaly detection for Snapdragon laptops

Listens to the laptop's own fan through the built-in microphone, reads CPU load, and flags
**abnormal fan behaviour** (grinding, rattling, blocked vents, fan too loud for the current load).
Everything runs locally; audio is never stored or sent anywhere in live mode.

## How it works
1. `record.py` captures 3-second windows of mic audio plus CPU load while the laptop is idle / medium / high load.
2. `features.py` turns each window into 65 numbers (mean and std of 32 mel bands in dB, plus CPU load).
3. `train.py` learns what a *healthy* laptop sounds like for a given CPU load (PCA anomaly model,
   trained on normal data only) and exports it as an ONNX graph (`model/model.onnx`).
4. `live.py` scores every window with ONNX Runtime on the **Snapdragon NPU (QNN / HTP)**, falling back to CPU.
   `score / threshold >= 1` means abnormal.

Why normal-only training: real faulty-fan recordings are rare, so the model detects "anything unlike healthy".

## Setup (Snapdragon X laptop, Windows on ARM)
```
python -m venv .venv && .venv\Scripts\activate
pip install numpy scipy psutil sounddevice
pip install onnxruntime-qnn        # NPU build; plain `pip install onnxruntime` gives CPU only
```
Windows Settings > Sound > your microphone > turn **Audio enhancements OFF** (noise suppression removes the fan sound).

## Run
```
python record.py --label normal --state idle   --minutes 4
python record.py --label normal --state medium --minutes 4
python record.py --label normal --state high   --minutes 4
python train.py
python live.py          # or: python live.py --cli
python bench.py         # CPU vs NPU timing
```
Record on the same laptop you will demo on, in a quiet room, on a hard flat surface.

## Simulated faults (for evaluation only)
Optional `--label abnormal` recordings: e.g. run under load with the laptop resting on a pillow/blanket
(reduced airflow, watch temperatures, stop if it gets hot), or play a fan-grinding/rattling sound from a phone
next to the mic. Label these clearly as **simulated** in your write-up. `train.py` then reports detection rate and AUC.

## Honest limitations (put these in your submission)
- Detects abnormal sound patterns, not guaranteed hardware failure.
- Calibrated per laptop; a new device needs a few minutes of recording.
- Loud environments (music, speech) can cause false alarms.
- The model is tiny, so CPU and NPU latencies are both well under 1 ms; the NPU benefit to report is
  low-power always-on operation and freeing the CPU, so measure battery drain / CPU usage, not just latency.
- `make_synthetic.py` is only a smoke test. Never report its numbers.
