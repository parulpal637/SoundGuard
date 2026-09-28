"""Record training data: fan sound + CPU load, in 3-second windows.

Examples (run each ~3-5 minutes, in a quiet room, laptop on a hard flat surface):
  python record.py --label normal --state idle   --minutes 4
  python record.py --label normal --state medium --minutes 4
  python record.py --label normal --state high   --minutes 4
Optional simulated-fault data, ONLY for evaluation (see README):
  python record.py --label abnormal --state high --minutes 2
"""
import argparse, multiprocessing as mp, os, time
import numpy as np, psutil, sounddevice as sd
from features import SR, WIN_SEC


def _burn(stop):
    x = 1
    while not stop.is_set():
        for _ in range(200000):
            x = (x * 1103515245 + 12345) % 2147483648


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["normal", "abnormal"], default="normal")
    ap.add_argument("--state", choices=["idle", "medium", "high"], required=True)
    ap.add_argument("--minutes", type=float, default=3)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    cores = os.cpu_count() or 4
    workers = {"idle": 0, "medium": max(1, cores // 2), "high": cores}[a.state]
    print("Input device:", sd.query_devices(kind="input")["name"])
    print("Tip: turn OFF 'Audio enhancements / noise suppression' for this mic in Windows Sound settings.")
    stop, procs = mp.Event(), []
    for _ in range(workers):
        p = mp.Process(target=_burn, args=(stop,), daemon=True)
        p.start()
        procs.append(p)
    if workers:
        print(f"Started {workers} CPU stress workers, waiting 20 s for the fan to spin up...")
        time.sleep(20)

    n = int(a.minutes * 60 / WIN_SEC)
    audio, cpu = [], []
    psutil.cpu_percent(None)
    try:
        for i in range(n):
            rec = sd.rec(int(SR * WIN_SEC), samplerate=SR, channels=1, dtype="float32")
            sd.wait()
            c = psutil.cpu_percent(None)
            audio.append(rec[:, 0]); cpu.append(c)
            print(f"\rwindow {i+1}/{n}  cpu={c:5.1f}%  rms={np.sqrt((rec**2).mean()):.4f}", end="")
    finally:
        stop.set()
        for p in procs:
            p.join(timeout=2)
    os.makedirs(os.path.join(a.out, a.label), exist_ok=True)
    path = os.path.join(a.out, a.label, f"{a.state}_{int(time.time())}.npz")
    np.savez_compressed(path, audio=np.array(audio, np.float32), cpu=np.array(cpu, np.float32))
    print("\nSaved", path)


if __name__ == "__main__":
    main()
