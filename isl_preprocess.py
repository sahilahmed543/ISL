"""Shared preprocessing for the ISL project.

Training (notebook) and real-time use (realtime_demo.py) both import THIS file,
so the live input is processed exactly like the training data.
Keep this file in the same folder as the notebook and the script (e.g. F:\\ISL).
"""
from pathlib import Path
import numpy as np

# MediaPipe Holistic layout: pose 33x4, face 468x3, left hand 21x3, right hand 21x3 = 1,662 values per frame
N_POSE, N_FACE, N_HAND = 33, 468, 21
SL = {"pose": slice(0, 132), "face": slice(132, 1536), "lh": slice(1536, 1599), "rh": slice(1599, 1662)}
N_FEATURES_FULL = 1662
N_FEATURES_COMPACT = 201          # 25 upper-body pose points x 3 + 2 hands x 21 x 3
T_FIXED = 30                      # frames per sample after resampling
VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}


# ---------------------------------------------------------------- data set and extraction
def scan_include(root):
    """One row per video. Category = first folder under root; sign = numbered folder (e.g. '1. Dog')."""
    import pandas as pd
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data folder not found: {root}")
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in VIDEO_EXT:
            rel = p.relative_to(root).parts
            sign = next((s for s in rel[1:-1] if s.split(".")[0].strip().isdigit()),
                        rel[-2] if len(rel) > 2 else "unknown")
            rows.append(dict(path=str(p), category=rel[0], label=sign.split(".", 1)[-1].strip().lower(), file=p.name))
    if not rows:
        raise FileNotFoundError(f"No videos found under {root}. Check that it contains <Category>/<sign>/<video> folders.")
    return pd.DataFrame(rows)


def frame_to_vector(res):
    """One MediaPipe Holistic result -> 1,662 values (zeros for parts not detected)."""
    def arr(lms, n, with_vis=False):
        if lms is None:
            return np.zeros(n * (4 if with_vis else 3), np.float32)
        if with_vis:
            return np.array([[p.x, p.y, p.z, p.visibility] for p in lms.landmark], np.float32).ravel()
        return np.array([[p.x, p.y, p.z] for p in lms.landmark], np.float32).ravel()
    return np.concatenate([arr(res.pose_landmarks, N_POSE, True), arr(res.face_landmarks, N_FACE),
                           arr(res.left_hand_landmarks, N_HAND), arr(res.right_hand_landmarks, N_HAND)])


def extract_video_landmarks(path, target_fps=25, model_complexity=1):
    """Run MediaPipe Holistic over a video -> (frames, 1662). Needs mediapipe==0.10.14."""
    import cv2
    import mediapipe as mp
    if not Path(path).exists():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open {path} (corrupt file or unsupported codec)")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / target_fps))) if target_fps else 1
    out, i = [], 0
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=model_complexity,
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                out.append(frame_to_vector(hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
            i += 1
    cap.release()
    return np.stack(out) if out else np.zeros((0, N_FEATURES_FULL), np.float32)


# ---------------------------------------------------------------- quality
def hand_rate(seq):
    """Share of frames where at least one hand was detected."""
    seq = np.asarray(seq)
    if len(seq) == 0:
        return 0.0
    lh = np.abs(seq[:, SL["lh"]]).sum(1) > 0
    rh = np.abs(seq[:, SL["rh"]]).sum(1) > 0
    return float((lh | rh).mean())


# ---------------------------------------------------------------- preprocessing
def fill_missing(seq):
    """Linear interpolation over frames where a part was not detected (all zeros). Edges held constant."""
    out = np.array(seq, dtype=np.float32, copy=True)
    for key in ("pose", "face", "lh", "rh"):
        blk = out[:, SL[key]]
        miss = np.abs(blk).sum(1) == 0
        if not miss.any() or miss.all():
            continue
        idx = np.arange(len(blk))
        good = ~miss
        for j in range(blk.shape[1]):
            blk[miss, j] = np.interp(idx[miss], idx[good], blk[good, j])
        out[:, SL[key]] = blk
    return out


def normalise(seq, eps=1e-6):
    """Origin = mid-shoulder point in each frame; scale = median shoulder width of the clip."""
    out = np.array(seq, dtype=np.float32, copy=True)
    T = len(out)
    pose = out[:, SL["pose"]].reshape(T, N_POSE, 4)
    ls, rs = pose[:, 11, :3], pose[:, 12, :3]
    origin = (ls + rs) / 2
    width = np.linalg.norm((ls - rs)[:, :2], axis=1)
    scale = float(np.median(width[width > eps])) if (width > eps).any() else 1.0

    def tf(key, d):
        blk = out[:, SL[key]].reshape(T, -1, d)
        nz = np.abs(out[:, SL[key]]).sum(1) > 0
        blk[nz, :, :3] = (blk[nz, :, :3] - origin[nz, None, :]) / scale
        out[:, SL[key]] = blk.reshape(T, -1)
    tf("pose", 4); tf("face", 3); tf("lh", 3); tf("rh", 3)
    return out


def select_features(seq, mode="compact"):
    if mode == "full":
        return np.asarray(seq, np.float32)
    T = len(seq)
    pose = seq[:, SL["pose"]].reshape(T, N_POSE, 4)[:, :25, :3].reshape(T, -1)
    return np.concatenate([pose, seq[:, SL["lh"]], seq[:, SL["rh"]]], 1).astype(np.float32)


def resample(seq, T=T_FIXED):
    """Linear interpolation to exactly T frames.
    Bug fix: an empty sequence used to give np.linspace(0, -1, T) -> negative indices -> IndexError
    with no useful message. It now raises a clear ValueError."""
    seq = np.asarray(seq, np.float32)
    n = len(seq)
    if n == 0:
        raise ValueError("resample: empty sequence (no frames were extracted)")
    pos = np.linspace(0, n - 1, T)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (pos - i0)[:, None]
    return (seq[i0] * (1 - w) + seq[i1] * w).astype(np.float32)


def preprocess(seq, mode="compact", T=T_FIXED):
    """(frames, 1662) raw landmarks -> (T, 201) model input. Same function for training and live use."""
    seq = np.asarray(seq, np.float32)
    if seq.ndim != 2 or seq.shape[1] != N_FEATURES_FULL:
        raise ValueError(f"preprocess: expected (frames, {N_FEATURES_FULL}), got {seq.shape}")
    return resample(select_features(normalise(fill_missing(seq)), mode), T)


# ---------------------------------------------------------------- augmentation (training data only)
_SWAP = list(range(25))
for _a, _b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
               (17, 18), (19, 20), (21, 22), (23, 24)]:
    _SWAP[_a], _SWAP[_b] = _b, _a


def augment(x, rng, rot_deg=8, scale=0.1, shift=0.05, noise=0.01, warp=0.15, mirror=False):
    """x: (T, 201) compact, normalised features = 67 points x (x, y, z). Returns a new (T, 201) array."""
    T = len(x)
    p = np.asarray(x, np.float32).reshape(T, 67, 3).copy()
    nz = np.abs(p).sum(2) > 0                       # points that were never detected stay zero
    a = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    p[..., :2] = p[..., :2] @ R.T
    p *= rng.uniform(1 - scale, 1 + scale)
    p[..., :2] += rng.uniform(-shift, shift, 2)
    p += rng.normal(0, noise, p.shape)
    if mirror:
        p[..., 0] *= -1
        pose, lh, rh = p[:, :25][:, _SWAP], p[:, 25:46].copy(), p[:, 46:67].copy()
        p = np.concatenate([pose, rh, lh], 1)
        nz = np.concatenate([nz[:, :25][:, _SWAP], nz[:, 46:67], nz[:, 25:46]], 1)
    p[~nz] = 0
    q = p.reshape(T, -1)
    u = np.linspace(0, 1, T)
    u = np.clip(u + rng.uniform(-warp, warp) * np.sin(np.pi * u) / np.pi * 2, 0, 1)
    i0 = np.minimum((u * (T - 1)).astype(int), T - 2)
    w = (u * (T - 1) - i0)[:, None]
    return (q[i0] * (1 - w) + q[i0 + 1] * w).astype(np.float32)
