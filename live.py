"""Live fan-health monitor. Listens to the mic, reads CPU load, scores every 3 s.
  python live.py            # window UI
  python live.py --cli      # console only
  python live.py --cpu      # force CPU instead of NPU"""
import argparse, collections, queue, threading, time
import numpy as np, psutil, sounddevice as sd
from features import SR, WIN_SEC
from runtime import Scorer


def status(r):
    if r < 1.0: return "HEALTHY", "#2e7d32"
    if r < 2.0: return "WATCH", "#f9a825"
    return "SUSPICIOUS", "#c62828"


class Monitor(threading.Thread):
    def __init__(self, scorer, q):
        super().__init__(daemon=True)
        self.scorer, self.q, self.hist = scorer, q, collections.deque(maxlen=3)

    def run(self):
        psutil.cpu_percent(None)
        while True:
            rec = sd.rec(int(SR * WIN_SEC), samplerate=SR, channels=1, dtype="float32")
            sd.wait()
            cpu = psutil.cpu_percent(None)
            t0 = time.perf_counter()
            score, ratio = self.scorer.score(rec[:, 0], cpu)
            ms = (time.perf_counter() - t0) * 1000
            self.hist.append(ratio)
            smooth = float(np.median(self.hist))
            self.q.put(dict(cpu=cpu, score=score, ratio=ratio, smooth=smooth, ms=ms))


def run_cli(scorer):
    q = queue.Queue(); Monitor(scorer, q).start()
    print("backend:", scorer.backend)
    while True:
        d = q.get(); s, _ = status(d["smooth"])
        print(f"cpu={d['cpu']:5.1f}%  score/threshold={d['ratio']:.2f}  -> {s}  ({d['ms']:.1f} ms)")


def run_gui(scorer):
    import tkinter as tk
    q = queue.Queue(); Monitor(scorer, q).start()
    root = tk.Tk(); root.title("Laptop Fan Health Monitor (on-device AI)")
    lbl = tk.Label(root, text="Listening...", font=("Segoe UI", 28, "bold"), width=16, fg="white", bg="#555")
    lbl.pack(padx=10, pady=10)
    info = tk.Label(root, text=f"Backend: {scorer.backend}", font=("Segoe UI", 11)); info.pack()
    cv = tk.Canvas(root, width=440, height=150, bg="white"); cv.pack(padx=10, pady=10)
    hist = collections.deque(maxlen=40)

    def draw():
        cv.delete("all")
        cv.create_line(0, 150 - 50, 440, 150 - 50, fill="#c62828", dash=(4, 3))   # threshold = 1.0
        cv.create_text(6, 90, text="threshold", anchor="w", fill="#c62828", font=("Segoe UI", 8))
        pts = [(i * 11, 150 - min(v, 3.0) * 50) for i, v in enumerate(hist)]
        for p0, p1 in zip(pts, pts[1:]):
            cv.create_line(*p0, *p1, width=2, fill="#1565c0")

    def poll():
        while not q.empty():
            d = q.get(); s, col = status(d["smooth"])
            hist.append(d["ratio"]); lbl.config(text=s, bg=col)
            info.config(text=f"Backend: {scorer.backend}   CPU: {d['cpu']:.0f}%   "
                             f"score/threshold: {d['ratio']:.2f}   inference: {d['ms']:.1f} ms")
            draw()
        root.after(300, poll)
    poll(); root.mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", action="store_true"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    sc = Scorer(use_npu=not a.cpu)
    run_cli(sc) if a.cli else run_gui(sc)
