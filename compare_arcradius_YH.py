import os
import sys
import argparse
import pickle
import numpy as np

import tailclick
import jtailfit2_old
from videowrapper import VideoWrapper

from astropy.convolution import convolve, Box1DKernel
from scipy.signal import find_peaks
import csv


def smooth_tf(tf, kernel_size=3, crop=10):
    """tf: list of frames, each frame shape (num_points,2)"""
    tf = np.array(tf)  # (frames, points, 2)
    tf_swapped = np.swapaxes(tf, 0, 1)  # (points, frames, 2)

    tf_swapped_smoothed = []
    for tailpoint in tf_swapped:
        x_sm = convolve(tailpoint[:, 0], Box1DKernel(kernel_size))
        y_sm = convolve(tailpoint[:, 1], Box1DKernel(kernel_size))

        if crop > 0:
            x_sm = x_sm[crop:-crop]
            y_sm = y_sm[crop:-crop]

        smoothed_coordinates = np.array(list(zip(x_sm, y_sm)))
        tf_swapped_smoothed.append(smoothed_coordinates)

    smoothed_tf = np.swapaxes(tf_swapped_smoothed, 0, 1)  # (frames, points, 2)
    return np.array(smoothed_tf)


def compute_basic_metrics_from_pkl(pkl_path, fps=1000, prominence=20, distance=10, width=1, rel_height=0.5):
    """
    只算你最关心的比较项：
    - Nr_half_beats
    - Freq_half_beats
    - Bias_RL（带inf，不会除0崩）
    同时返回 peaks_right/peaks_left 数量方便诊断
    """
    with open(pkl_path, "rb") as f:
        tf = pickle.load(f)

    sm_tf = smooth_tf(tf, kernel_size=3, crop=10)
    if sm_tf.shape[0] < 5:
        return {"ok": False, "reason": "too_few_frames_after_crop"}

    # deflection per frame (tail base -> tip)
    dx = sm_tf[:, -1, 0] - sm_tf[:, 0, 0]
    dy = sm_tf[:, -1, 1] - sm_tf[:, 0, 1]
    deflection = np.degrees(np.arctan(np.divide(dx, dy)))

    # tailtip x for peaks
    tailtip_x = sm_tf[:, -1, 0].astype(float)

    peaks_right, prop_r = find_peaks(tailtip_x, distance=distance, prominence=prominence, width=width, rel_height=rel_height)
    peaks_left,  prop_l = find_peaks(-tailtip_x, distance=distance, prominence=prominence, width=width, rel_height=rel_height)

    n_r = int(peaks_right.size)
    n_l = int(peaks_left.size)
    nr_half_beats = n_r + n_l

    if nr_half_beats == 0:
        return {
            "ok": False,
            "reason": "no_peaks",
            "Nr_half_beats": 0,
            "Freq_half_beats": np.nan,
            "Bias_RL": np.nan,
            "peaks_right": 0,
            "peaks_left": 0
        }

    # movement duration using combined ips (robust to one-sided peaks)
    left_ips = np.concatenate([prop_r.get("left_ips", np.array([])), prop_l.get("left_ips", np.array([]))])
    right_ips = np.concatenate([prop_r.get("right_ips", np.array([])), prop_l.get("right_ips", np.array([]))])

    if left_ips.size == 0 or right_ips.size == 0:
        # 极少见，但保持安全
        movement_duration = np.nan
    else:
        start_i = int(np.min(left_ips))
        end_i = int(np.max(right_ips))
        end_i = min(end_i, len(tailtip_x))
        movement_duration = (end_i - start_i) / fps

    if movement_duration is None or movement_duration == 0 or np.isnan(movement_duration):
        freq_half_beats = np.nan
    else:
        freq_half_beats = nr_half_beats / movement_duration

    # bias from deflection at peaks (same as your logic)
    defl_peaks = []
    for p in peaks_right:
        defl_peaks.append(deflection[p])
    for p in peaks_left:
        defl_peaks.append(deflection[p])

    bias_right = int(np.sum(np.array(defl_peaks) > 0))
    bias_left  = int(np.sum(np.array(defl_peaks) < 0))

    if bias_left == 0 and bias_right > 0:
        bias_rl = np.inf
    elif bias_right == 0 and bias_left > 0:
        bias_rl = 0.0
    elif (bias_right + bias_left) == 0:
        bias_rl = np.nan
    else:
        bias_rl = bias_right / bias_left

    return {
        "ok": True,
        "Nr_half_beats": nr_half_beats,
        "Movement_duration_s": movement_duration,
        "Freq_half_beats_Hz": freq_half_beats,
        "Bias_RL": bias_rl,
        "peaks_right": n_r,
        "peaks_left": n_l
    }

# =========================
# >>> PATCH: IDE-friendly runner (PyCharm ▶️)
# =========================

def _inject_args_if_run_from_ide():
    """
    If run from IDE with no CLI args, pop up dialogs to select inputs
    and inject sys.argv so argparse works unchanged.
    """
    if len(sys.argv) > 1:
        return  # already have CLI args

    try:
        import tkinter as tk
        from tkinter import filedialog, simpledialog, messagebox
    except Exception as e:
        print("tkinter unavailable, cannot prompt for inputs:", e)
        return

    root = tk.Tk()
    root.withdraw()

    avi = filedialog.askopenfilename(
        title="Select AVI file",
        filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
    )
    if not avi:
        messagebox.showinfo("Cancelled", "No AVI selected. Exit.")
        sys.exit(0)

    outdir = filedialog.askdirectory(title="Select output folder")
    if not outdir:
        messagebox.showinfo("Cancelled", "No output folder selected. Exit.")
        sys.exit(0)

    ar_str = simpledialog.askstring(
        "arcradius list",
        "Enter ar values separated by spaces (default: 30 60):",
        initialvalue="30 60"
    )
    if not ar_str:
        ar_list = ["30", "60"]
    else:
        # allow commas too
        tokens = [t for t in ar_str.replace(",", " ").split() if t.strip()]
        ar_list = []
        for t in tokens:
            try:
                int(t)
                ar_list.append(t)
            except ValueError:
                pass
        if len(ar_list) == 0:
            ar_list = ["30", "60"]

    # Optional: ask whether to display tracking
    use_display = messagebox.askyesno("Display", "Show tracking display? (Yes=--display)")

    sys.argv += ["--avi", avi, "--outdir", outdir, "--ar"] + ar_list
    if use_display:
        sys.argv += ["--display"]

    messagebox.showinfo(
        "Running",
        f"AVI:\n{avi}\n\nOutdir:\n{outdir}\n\nar:\n{' '.join(ar_list)}\n\n"
        f"{'Display ON' if use_display else 'Display OFF'}"
    )


def _popup_summary(out_csv_path, results):
    """
    Show a small pop-up summary after run (no pandas dependency).
    """
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        # fallback to console only
        return

    lines = ["ar compare summary:"]
    lines.append(f"Saved CSV:\n{out_csv_path}\n")
    for r in results:
        ar = r.get("arcradius", "")
        ok = r.get("ok", False)
        if not ok:
            lines.append(f"ar={ar}: FAILED ({r.get('reason','')})")
            continue
        lines.append(
            f"ar={ar}: half_beats={r.get('Nr_half_beats','')}, "
            f"freq={r.get('Freq_half_beats_Hz','')}, "
            f"bias={r.get('Bias_RL','')}, "
            f"peaks(R,L)=({r.get('peaks_right','')},{r.get('peaks_left','')})"
        )

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("ar compare result", "\n".join(lines))

def main():
    _inject_args_if_run_from_ide()   # <<< PATCH: add this line before argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--avi", required=True, help="Path to one AVI file to compare")
    ap.add_argument("--outdir", required=True, help="Output folder to save pkls and comparison csv")
    ap.add_argument("--fps", type=float, default=1000)
    ap.add_argument("--ar", nargs="+", type=int, default=[30, 60], help="List of arcradius values, e.g. --ar 30 60")
    ap.add_argument("--prominence", type=float, default=20)
    ap.add_argument("--distance", type=int, default=10)
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--rel_height", type=float, default=0.5)
    ap.add_argument("--display", action="store_true", help="Show tracking display")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    vid = VideoWrapper(args.avi)
    firstframe = vid.firstframe

    # Pick points ONCE, reuse for all arcradius values
    startpoint, endpoint = tailclick.picktwopoints(firstframe)

    base = os.path.splitext(os.path.basename(args.avi))[0]
    results = []

    for ar in args.ar:
        vid2 = VideoWrapper(args.avi)  # reopen video for each run
        tf = jtailfit2.tailfit_simple(
            vid2, startpoint, endpoint,
            display=args.display,
            arcradius=ar
        )

        pkl_name = f"{base}_ar{ar}.pkl"
        pkl_path = os.path.join(args.outdir, pkl_name)
        with open(pkl_path, "wb") as f:
            pickle.dump(tf, f)

        metrics = compute_basic_metrics_from_pkl(
            pkl_path,
            fps=args.fps,
            prominence=args.prominence,
            distance=args.distance,
            width=args.width,
            rel_height=args.rel_height
        )
        metrics["arcradius"] = ar
        metrics["pkl"] = pkl_name
        results.append(metrics)

    # Save comparison CSV
    out_csv = os.path.join(args.outdir, f"{base}_ar_compare.csv")
    fieldnames = ["arcradius", "pkl", "ok", "reason",
                  "peaks_right", "peaks_left",
                  "Nr_half_beats", "Movement_duration_s", "Freq_half_beats_Hz", "Bias_RL"]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            if "reason" not in r:
                r["reason"] = ""
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print("Saved:", out_csv)
    _popup_summary(out_csv, results)   # <<< PATCH: pop up summary in PyCharm

    print("Rows:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
