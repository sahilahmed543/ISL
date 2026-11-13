"""Real-time ISL recognition from a webcam or a video file.

Run from Anaconda Prompt in the folder that holds this file and isl_preprocess.py:
    conda activate isl-prep
    python realtime_demo.py                         (webcam 0, sign-by-sign mode)
    python realtime_demo.py --camera 1              (another camera)
    python realtime_demo.py --mode window           (old continuous 3-second window mode)
    python realtime_demo.py --video "F:/ISL/INCLUDE/Animals/1. Dog/MVI_2978.MOV"
Keys: q or Esc to quit.

How to sign (segment mode, the default):
    1. Start with your hands down, out of the camera view.
    2. Raise your hands and make ONE sign at a normal pace.
    3. Lower your hands out of view. The sign is classified once and shown for 3 seconds.

Why segment mode: every training clip is one complete sign (median 75 frames, about 3 s at 25 fps)
resampled to 30 frames. Segment mode cuts the live stream the same way: from when the hands appear
until they disappear. The older window mode classifies whatever happened in the last 3 seconds,
which often mixes half a sign with idle movement.
"""
import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import isl_preprocess as P


def parse_args():
    ap = argparse.ArgumentParser(description="Real-time ISL recognition")
    ap.add_argument("--model", default="F:/ISL/models/isl_lstm.keras")
    ap.add_argument("--config", default="F:/ISL/models/isl_lstm_config.json")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", default=None, help="run on a video file instead of the webcam")
    ap.add_argument("--mode", choices=["segment", "window"], default="segment",
                    help="segment: classify each sign once, from hands-up to hands-down (default); "
                         "window: classify the last few seconds continuously")
    ap.add_argument("--threshold", type=float, default=0.6, help="minimum confidence to accept a prediction")
    ap.add_argument("--end-gap", type=float, default=0.5, help="segment mode: seconds without hands that end a sign")
    ap.add_argument("--min-sec", type=float, default=0.6, help="segment mode: ignore movements shorter than this")
    ap.add_argument("--max-sec", type=float, default=5.0, help="segment mode: force a prediction after this long")
    ap.add_argument("--hold-sec", type=float, default=3.0, help="segment mode: how long a result stays on screen")
    ap.add_argument("--window-sec", type=float, default=None, help="window mode: seconds per prediction (default from config)")
    ap.add_argument("--every", type=int, default=5, help="window mode: predict every N frames")
    ap.add_argument("--smooth", type=int, default=5, help="window mode: majority vote over the last N accepted predictions")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1], help="MediaPipe model complexity (0 = faster)")
    ap.add_argument("--no-draw", action="store_true", help="do not draw landmarks (slightly faster)")
    ap.add_argument("--log", default="F:/ISL/models/realtime_log.csv")
    return ap.parse_args()


def has_hand(vec):
    return bool(np.abs(vec[P.SL["lh"]]).sum() > 0 or np.abs(vec[P.SL["rh"]]).sum() > 0)


class Segmenter:
    """Cuts a stream of frames into single signs: starts when a hand appears, ends after
    `end_gap` seconds with no hands (or after `max_sec`). Keeps a short pre-roll so the
    start of the movement is included, like the start of a training clip."""

    def __init__(self, end_gap=0.5, min_sec=0.6, max_sec=5.0, pre_roll=5):
        self.end_gap, self.min_sec, self.max_sec = end_gap, min_sec, max_sec
        self.pre = collections.deque(maxlen=pre_roll)
        self.seg, self.t_start, self.t_last_hand = [], None, None

    @property
    def active(self):
        return self.t_start is not None

    def update(self, ts, vec, hand):
        """Returns (segment array, reason) when a sign ends, otherwise (None, status)."""
        if not self.active:
            if hand:
                self.seg = list(self.pre) + [vec]
                self.t_start = self.t_last_hand = ts
                return None, "signing..."
            self.pre.append(vec)
            return None, "waiting: raise your hands and sign"
        self.seg.append(vec)
        if hand:
            self.t_last_hand = ts
        ended = (ts - self.t_last_hand) >= self.end_gap
        too_long = (ts - self.t_start) >= self.max_sec
        if not (ended or too_long):
            return None, "signing..."
        dur = self.t_last_hand - self.t_start
        seg = np.stack(self.seg)
        self.seg, self.t_start, self.t_last_hand = [], None, None
        self.pre.clear()
        if dur < self.min_sec:
            return None, "too short, ignored"
        return seg, "max length reached" if too_long else "sign ended"


def put(img, text, org, scale=0.9):
    """White text with a dark outline, readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    a = parse_args()

    # ---- error handling: files and settings
    for f in (a.model, a.config):
        if not Path(f).exists():
            sys.exit(f"[error] {f} not found. Train and save the model in the Task 5 notebook first, or pass --model/--config.")
    if a.video and not Path(a.video).exists():
        sys.exit(f"[error] video not found: {a.video}")
    if not 0 < a.threshold < 1:
        sys.exit("[error] --threshold must be between 0 and 1")
    cfg = json.load(open(a.config))
    classes, T = cfg["classes"], int(cfg["T"])
    window = a.window_sec or float(cfg.get("window_sec", 3.0))

    import mediapipe as mp
    import tensorflow as tf
    import keras

    print("loading model ...")
    model = keras.saving.load_model(a.model)
    infer = tf.function(lambda x: model(x, training=False), reduce_retracing=True)   # faster than model.predict
    infer(tf.zeros((1, T, P.N_FEATURES_COMPACT)))                                      # warm-up (first call compiles)

    def classify(seq):
        t1 = time.perf_counter()
        p = infer(P.preprocess(seq)[None]).numpy()[0]
        return p, time.perf_counter() - t1

    file_mode = a.video is not None
    cap = cv2.VideoCapture(a.video if file_mode else a.camera)
    if not cap.isOpened():
        what = f"video {a.video}" if file_mode else f"camera {a.camera}"
        sys.exit(f"[error] could not open {what}. Close other apps using the camera, or try --camera 1.")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    hol = mp.solutions.holistic.Holistic(model_complexity=a.complexity,
                                         min_detection_confidence=0.5, min_tracking_confidence=0.5)
    draw, H = mp.solutions.drawing_utils, mp.solutions.holistic

    seg = Segmenter(a.end_gap, a.min_sec, a.max_sec)
    buf = collections.deque()                  # window mode: (timestamp, vector) for the last `window` seconds
    votes = collections.deque(maxlen=a.smooth)
    shown, shown_conf, shown_until, status = "", 0.0, 0.0, "starting"
    top3 = ""
    all_vecs, log, results = [], [], []
    n, fails, fps = 0, 0, 0.0
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if file_mode:
                    break
                fails += 1                                  # webcams can drop a frame; give up after 30 in a row
                if fails > 30:
                    print("[error] camera stopped sending frames")
                    break
                continue
            fails = 0
            ts = n / src_fps if file_mode else time.time()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hol.process(rgb)
            vec = P.frame_to_vector(res)
            hand = has_hand(vec)
            t_ext = time.perf_counter() - t0
            if file_mode:
                all_vecs.append(vec)
            n += 1
            t_model = 0.0

            if a.mode == "segment":
                s, status = seg.update(ts, vec, hand)
                if s is not None:
                    p, t_model = classify(s)
                    order = np.argsort(p)[::-1][:3]
                    top3 = "  ".join(f"{classes[i]} {p[i]:.2f}" for i in order)
                    k = int(order[0])
                    if p[k] >= a.threshold:
                        shown, shown_conf, status = classes[k], float(p[k]), f"recognised ({len(s)} frames)"
                    else:
                        shown, shown_conf, status = "", 0.0, f"not sure, try again ({len(s)} frames)"
                    shown_until = ts + a.hold_sec
                    results.append((round(ts, 2), len(s), classes[k], round(float(p[k]), 3)))
                    print(f"[sign] {len(s)} frames -> {top3}")
                if shown and ts > shown_until:
                    shown = ""                                   # result expires; ready for the next sign
            else:                                                # window mode
                buf.append((ts, vec))
                while buf and ts - buf[0][0] > window:
                    buf.popleft()
                if n % a.every == 0:
                    w = np.stack([v for _, v in buf])
                    if P.hand_rate(w) < 0.5:
                        status = "no hands in view"
                        votes.clear()
                        shown = ""                               # bug fix: do not keep showing an old label
                    elif len(w) < 10:
                        status = "collecting frames"
                    else:
                        p, t_model = classify(w)
                        order = np.argsort(p)[::-1][:3]
                        top3 = "  ".join(f"{classes[i]} {p[i]:.2f}" for i in order)
                        k = int(order[0])
                        if p[k] >= a.threshold:
                            votes.append(k)
                            status = "ok"
                        else:
                            status = "low confidence"
                        if votes:
                            vals, cnt = np.unique(list(votes), return_counts=True)
                            if cnt.max() >= a.smooth // 2 + 1:
                                best = int(vals[cnt.argmax()])
                                shown, shown_conf = classes[best], float(p[best])

            dt = time.perf_counter() - t0
            fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
            log.append((n, round(t_ext * 1000, 2), round(t_model * 1000, 2), round(dt * 1000, 2)))

            if not a.no_draw:
                draw.draw_landmarks(frame, res.pose_landmarks, H.POSE_CONNECTIONS)
                draw.draw_landmarks(frame, res.left_hand_landmarks, H.HAND_CONNECTIONS)
                draw.draw_landmarks(frame, res.right_hand_landmarks, H.HAND_CONNECTIONS)
            scale = 960 / frame.shape[1] if frame.shape[1] > 960 else 1.0
            view = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
            put(view, f"{shown}  ({shown_conf:.2f})" if shown else "-", (15, 45), 1.2)
            put(view, status, (15, 85), 0.7)
            if top3:
                put(view, "top 3: " + top3, (15, 120), 0.55)
            put(view, f"{fps:4.1f} FPS  |  hands: {'yes' if hand else 'no'}", (15, view.shape[0] - 20), 0.7)
            cv2.imshow("ISL recognition (q to quit)", view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        hol.close()
        cv2.destroyAllWindows()

    if file_mode and all_vecs:                       # whole clip, exactly as in training
        s = np.stack(all_vecs)
        step = max(1, int(round(src_fps / 25)))     # training extracted at ~25 fps
        p, _ = classify(s[::step])
        top = np.argsort(p)[::-1][:3]
        print("whole-clip prediction:", ", ".join(f"{classes[i]} {p[i]:.2f}" for i in top))

    if results:
        print("signs recognised this session (time s, frames, top class, confidence):")
        for r in results:
            print("  ", r)
    if log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        with open(a.log, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "extract_ms", "model_ms", "total_ms"])
            w.writerows(log)
        tot = np.array([r[3] for r in log])
        mdl = np.array([r[2] for r in log if r[2] > 0])
        print(f"frames: {len(log)} | median processing {np.median(tot):.1f} ms/frame ({1000 / np.median(tot):.1f} FPS)"
              f" | p95 {np.percentile(tot, 95):.1f} ms"
              + (f" | model median {np.median(mdl):.1f} ms" if len(mdl) else "")
              + f" | log saved to {a.log}")


if __name__ == "__main__":
    main()
