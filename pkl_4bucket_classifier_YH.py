import os
import sys
import csv
import pickle
import numpy as np
from scipy.signal import find_peaks
from collections import Counter

# ================== 你只需要改这两行 ==================
PKL_FOLDER = r"U:\YuanqiHua\High speed\260127 dmrt3 MTZ\pkl_output"
OUT_CSV    = r"U:\YuanqiHua\High speed\260127 dmrt3 MTZ\pkl_output\pkl_4bucket_summary.csv"
# ======================================================


FPS = 1000
PROMINENCE = 20
DISTANCE = 10
SMOOTH_KERNEL = 3
CROP = 10
ENDWIN = 60

# ---------- numpy pkl 兼容 ----------
import numpy.core as ncore
sys.modules["numpy._core"] = ncore
sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath
# ----------------------------------

def smooth_tf(tf, kernel=3, crop=10):
    tf = np.asarray(tf, dtype=float)
    k = np.ones(kernel) / kernel
    out = np.empty_like(tf)
    for p in range(tf.shape[1]):
        out[:, p, 0] = np.convolve(tf[:, p, 0], k, mode="same")
        out[:, p, 1] = np.convolve(tf[:, p, 1], k, mode="same")
    if tf.shape[0] > 2 * crop:
        out = out[crop:-crop]
    return out

def curvature(sm):
    v = sm[:, 1:, :] - sm[:, :-1, :]
    ang = np.degrees(np.arctan2(v[:, :, 1], v[:, :, 0]))
    d = np.abs(ang[:, :-1] - ang[:, 1:])
    d = np.where(d <= 180, d, 360 - d)
    return np.sum(d, axis=1)

def classify(tf):
    sm = smooth_tf(tf, SMOOTH_KERNEL, CROP)
    base = sm[:, 0]
    tip = sm[:, -1]

    dx = tip[:, 0] - base[:, 0]
    dy = tip[:, 1] - base[:, 1]
    defl = np.degrees(np.arctan(np.divide(dx, dy, out=np.zeros_like(dx), where=dy != 0)))
    max_defl = np.nanmax(np.abs(defl))

    curv = curvature(sm)
    max_curv = np.nanmax(curv)

    x = tip[:, 0]
    pr, _ = find_peaks(x, distance=DISTANCE, prominence=PROMINENCE)
    pl, _ = find_peaks(-x, distance=DISTANCE, prominence=PROMINENCE)
    nr_half = len(pr) + len(pl)

    step = np.linalg.norm(np.diff(tip, axis=0), axis=1)
    tail_end = step[-ENDWIN:] if len(step) > ENDWIN else step
    end_repeat = np.mean(tail_end < 1e-3)

    QC_badtracking = int(end_repeat > 0.15)
    has_Cbend = int(max_defl > 50 or max_curv > 200)

    freq = nr_half / (len(sm) / FPS) if len(sm) > 0 else 0
    is_regular_swim = int((QC_badtracking == 0) and (nr_half >= 18) and (freq >= 60))
    is_struggle = int((QC_badtracking == 0) and (is_regular_swim == 0) and (nr_half > 0))

    return {
        "QC_badtracking": QC_badtracking,
        "has_Cbend": has_Cbend,
        "is_regular_swim": is_regular_swim,
        "is_struggle": is_struggle,
        "Nr_half_beats": nr_half,
        "Freq_half_beats_Hz": round(freq, 2),
        "Max_deflection_deg": round(max_defl, 2),
        "Max_curvature": round(max_curv, 2),
        "end_repeat": round(end_repeat, 3)
    }

def main():
    rows = []
    for fn in sorted(os.listdir(PKL_FOLDER)):
        if not fn.endswith(".pkl"):
            continue
        path = os.path.join(PKL_FOLDER, fn)
        with open(path, "rb") as f:
            tf = pickle.load(f)
        res = classify(tf)
        res["file"] = fn
        rows.append(res)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("Saved:", OUT_CSV)
    print("Summary:")
    for k in ["QC_badtracking", "has_Cbend", "is_regular_swim", "is_struggle"]:
        print(k, Counter([r[k] for r in rows]))

if __name__ == "__main__":
    main()
