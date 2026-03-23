import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# =========================================================
# PARAMETERS
# =========================================================
mode = "single"   # "single" or "batch"

single_pkl = r"U:\YuanqiHua\High speed\Manual_fix_pkl_out\260312 dmrt3 MTZ\d3-MTZ-OMR-3_14_binary_manual_fixed.pkl"
folder = r"U:\YuanqiHua\High speed\Manual_fix_pkl_out\260312 dmrt3 MTZ"
save_folder = os.path.join(folder, "plot1")

os.makedirs(save_folder, exist_ok=True)

show_plot = False
save_plot = True

# smoothing / cropping
do_smooth = True
smooth_window = 3
crop_n = 1       # 设成10就是裁前后10帧；设成0就是不裁

BASELINE_POINT_INDEX = 1
TIP_POINT_INDEX = -2

# peak parameters
peak_distance = 10
peak_prominence = 1
peak_width = 1



# second filter
min_peak_gap_all = 6   # 不分R/L，和上一个保留峰至少间隔的帧
min_deflection_deg = 0
require_amp_ratio = False
amp_ratio = 0.30
require_alternation = False


use_trend_filter = False   # True=用 filter_active_peaks_with_trend
                          # False=用普通 filter_active_peaks
trend_window = 20 # 看“大方向”时，用前后多少帧来判断。
rebound_ratio = 0.05 # “小回弹幅度占上一主峰的比例”。
jitter_gap_same_phase = 10 # 如果一个反向小峰离前一个主峰太近，比如不到 8 帧

os.makedirs(save_folder, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def moving_average_same(x, window=3):
    x = np.asarray(x, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def smooth_tailfit(tf, smooth_window=3, crop_n=0, do_smooth=True):
    tf = np.asarray(tf, dtype=float).copy()

    if do_smooth:
        for p in range(tf.shape[1]):
            tf[:, p, 0] = moving_average_same(tf[:, p, 0], smooth_window)
            tf[:, p, 1] = moving_average_same(tf[:, p, 1], smooth_window)

    if crop_n > 0:
        tf = tf[crop_n:-crop_n]

    return tf


def calc_x_tip(smoothed_tf, tip_idx=-1):
    return np.asarray([frame[tip_idx, 0] for frame in smoothed_tf], dtype=float)


def calc_deflection(smoothed_tf, base_idx=0, tip_idx=-1):
    dx = np.asarray([frame[tip_idx, 0] - frame[base_idx, 0] for frame in smoothed_tf], dtype=float)
    dy = np.asarray([frame[tip_idx, 1] - frame[base_idx, 1] for frame in smoothed_tf], dtype=float)

    # 和你原思路一致：图像坐标y向下增加
    # 保持 arctan2(dx, dy) 写法
    deflection = np.degrees(np.arctan2(dx, dy))
    return deflection


def detect_peaks(deflection):
    # 右摆峰：x_tip局部最大
    peaks_right, _ = find_peaks(
        deflection,
        distance=peak_distance,
        prominence=peak_prominence,
        width=peak_width
    )

    # 左摆峰：x_tip局部最小
    peaks_left, _ = find_peaks(
        -deflection,
        distance=peak_distance,
        prominence=peak_prominence,
        width=peak_width
    )

    return np.asarray(peaks_right, dtype=int), np.asarray(peaks_left, dtype=int)

def merge_raw_peaks(peaks_right, peaks_left):
    items = []
    for p in peaks_right:
        items.append((int(p), "R"))
    for p in peaks_left:
        items.append((int(p), "L"))
    items.sort(key=lambda x: x[0])

    frames_raw = np.asarray([x[0] for x in items], dtype=int)
    dirs_raw = np.asarray([x[1] for x in items], dtype=object)
    return frames_raw, dirs_raw


def filter_active_peaks(deflection, frames_raw, dirs_raw,
                        min_deflection_deg=5.0,
                        require_amp_ratio=True,
                        amp_ratio=0.30,
                        require_alternation=False,
                        min_peak_gap_all=0):
    """
    第二层 filter：
    1) abs(deflection at peak) >= min_deflection_deg
    2) amp_ratio: 当前峰 >= 上一个保留峰 * amp_ratio
    3) alternation: 可选，要求方向交替
    """
    frames_raw = np.asarray(frames_raw, dtype=int)
    dirs_raw = np.asarray(dirs_raw, dtype=object)

    if len(frames_raw) == 0:
        return np.array([], dtype=int), np.array([], dtype=object)

    amps = np.abs(deflection[frames_raw])

    kept_idx = []

    for j in range(len(frames_raw)):
        # 第一个峰
        if len(kept_idx) == 0:
            if amps[j] >= min_deflection_deg:
                kept_idx.append(j)
            continue

        # 新规则0：跨方向最小间隔（不分 R/L）
        if min_peak_gap_all > 0:
            if frames_raw[j] - frames_raw[kept_idx[-1]] < min_peak_gap_all:
                continue

        # 规则1：最小 deflection
        if amps[j] < min_deflection_deg:
            continue

        # 规则2：方向交替
        if require_alternation and dirs_raw[j] == dirs_raw[kept_idx[-1]]:
            continue

        # 规则3：amp_ratio
        if require_amp_ratio:
            thr = amp_ratio * amps[kept_idx[-1]]
            if amps[j] < thr:
                continue

        kept_idx.append(j)

    kept_idx = np.asarray(kept_idx, dtype=int)
    kept_frames = frames_raw[kept_idx]
    kept_dirs = dirs_raw[kept_idx]

    return kept_frames, kept_dirs

def filter_active_peaks_with_trend(deflection, frames_raw, dirs_raw,
                                   min_deflection_deg=0.0,
                                   require_amp_ratio=False,
                                   amp_ratio=0.30,
                                   require_alternation=False,
                                   min_peak_gap_all=0,
                                   trend_window=12,
                                   rebound_ratio=0.35,
                                   jitter_gap_same_phase=8):
    frames_raw = np.asarray(frames_raw, dtype=int)
    dirs_raw = np.asarray(dirs_raw, dtype=object)

    if len(frames_raw) == 0:
        return np.array([], dtype=int), np.array([], dtype=object)

    amps = np.abs(deflection[frames_raw])
    kept_idx = []

    def local_trend(signal, center, w=12):
        left = max(0, center - w)
        right = min(len(signal) - 1, center + w)
        if right <= left:
            return 0.0
        return signal[right] - signal[left]

    for j in range(len(frames_raw)):
        f = frames_raw[j]
        d = dirs_raw[j]
        a = amps[j]

        if a < min_deflection_deg:
            continue

        if len(kept_idx) == 0:
            kept_idx.append(j)
            continue

        prev_j = kept_idx[-1]
        prev_f = frames_raw[prev_j]
        prev_d = dirs_raw[prev_j]
        prev_a = amps[prev_j]

        # 规则0：全局最小峰间隔
        if min_peak_gap_all > 0 and (f - prev_f) < min_peak_gap_all:
            continue

        # 规则1：可选 alternation
        if require_alternation and d == prev_d:
            continue

        # 规则2：可选 amp ratio
        if require_amp_ratio:
            thr = amp_ratio * prev_a
            if a < thr:
                continue

        # ============================
        # 新规则A：主趋势 + 小回弹排除
        # 当前候选与上一个保留峰方向相反时，检查是不是只是小回弹
        # ============================
        if d != prev_d:
            tr = local_trend(deflection, f, w=trend_window)

            # 情况1：当前是 Pos，但附近大趋势仍整体向下，且幅度小 -> 视为回弹
            if d == "R" and tr < 0 and a < rebound_ratio * prev_a:
                continue

            # 情况2：当前是 Neg，但附近大趋势仍整体向上，且幅度小 -> 视为回弹
            if d == "L" and tr > 0 and a < rebound_ratio * prev_a:
                continue

        # ============================
        # 新规则B：近邻抖动排除
        # 反向小峰如果离上一个主峰太近，也排除
        # ============================
        if d != prev_d:
            if (f - prev_f) < jitter_gap_same_phase and a < rebound_ratio * prev_a:
                continue

        kept_idx.append(j)

    kept_idx = np.asarray(kept_idx, dtype=int)
    kept_frames = frames_raw[kept_idx]
    kept_dirs = dirs_raw[kept_idx]
    return kept_frames, kept_dirs

def annotate_peak_labels(ax, xs, ys, label_prefix, color, dx_text=0, dy_text=0):
    for x, y in zip(xs, ys):
        ax.scatter(x, y, s=36, color=color, zorder=5)
        ax.text(
            x + dx_text,
            y + dy_text,
            f"{label_prefix}\nF{x}",
            fontsize=7,
            ha="center",
            va="bottom",
            color=color
        )

def save_parameters(basename):

    param_file = os.path.join(save_folder, basename + "_parameters.txt")

    with open(param_file, "w") as f:

        f.write("Plot_deflc_X_YH parameters\n")
        f.write("==========================\n\n")

        f.write(f"mode = {mode}\n")
        f.write(f"input = {folder}\n")
        f.write(f"save_folder = {save_folder}\n\n")

        f.write("[Smoothing / Cropping]\n")
        f.write(f"do_smooth = {do_smooth}\n")
        f.write(f"smooth_window = {smooth_window}\n")
        f.write(f"crop_n = {crop_n}\n\n")

        f.write("[Tail points]\n")
        f.write(f"BASELINE_POINT_INDEX = {BASELINE_POINT_INDEX}\n")
        f.write(f"TIP_POINT_INDEX = {TIP_POINT_INDEX}\n\n")

        f.write("[Peak detection]\n")
        f.write(f"peak_distance = {peak_distance}\n")
        f.write(f"peak_prominence = {peak_prominence}\n")
        f.write(f"peak_width = {peak_width}\n\n")

        f.write("[Filter mode]\n")
        f.write(f"use_trend_filter = {use_trend_filter}\n\n")

        f.write("[Second filter]\n")
        f.write(f"min_peak_gap_all = {min_peak_gap_all}\n")
        f.write(f"min_deflection_deg = {min_deflection_deg}\n")
        f.write(f"require_amp_ratio = {require_amp_ratio}\n")
        f.write(f"amp_ratio = {amp_ratio}\n")
        f.write(f"require_alternation = {require_alternation}\n\n")

        f.write("[Trend filter parameters]\n")
        f.write(f"trend_window = {trend_window}\n")
        f.write(f"rebound_ratio = {rebound_ratio}\n")
        f.write(f"jitter_gap_same_phase = {jitter_gap_same_phase}\n")

def save_parameters_once():

    param_file = os.path.join(save_folder, "plot_parameters.txt")

    with open(param_file, "w") as f:

        f.write("Plot_deflc_X_YH parameters\n")
        f.write("==========================\n\n")

        f.write(f"mode = {mode}\n")
        f.write(f"input = {folder}\n")
        f.write(f"save_folder = {save_folder}\n\n")

        f.write("[Smoothing / Cropping]\n")
        f.write(f"do_smooth = {do_smooth}\n")
        f.write(f"smooth_window = {smooth_window}\n")
        f.write(f"crop_n = {crop_n}\n\n")

        f.write("[Tail points]\n")
        f.write(f"BASELINE_POINT_INDEX = {BASELINE_POINT_INDEX}\n")
        f.write(f"TIP_POINT_INDEX = {TIP_POINT_INDEX}\n\n")

        f.write("[Peak detection]\n")
        f.write(f"peak_distance = {peak_distance}\n")
        f.write(f"peak_prominence = {peak_prominence}\n")
        f.write(f"peak_width = {peak_width}\n\n")

        f.write("[Filter mode]\n")
        f.write(f"use_trend_filter = {use_trend_filter}\n\n")

        f.write("[Second filter]\n")
        f.write(f"min_peak_gap_all = {min_peak_gap_all}\n")
        f.write(f"min_deflection_deg = {min_deflection_deg}\n")
        f.write(f"require_amp_ratio = {require_amp_ratio}\n")
        f.write(f"amp_ratio = {amp_ratio}\n")
        f.write(f"require_alternation = {require_alternation}\n\n")

        f.write("[Trend filter parameters]\n")
        f.write(f"trend_window = {trend_window}\n")
        f.write(f"rebound_ratio = {rebound_ratio}\n")
        f.write(f"jitter_gap_same_phase = {jitter_gap_same_phase}\n")

# =========================================================
# MAIN PLOT
# =========================================================
def plot_one_file(pkl_path):
    basename = os.path.splitext(os.path.basename(pkl_path))[0]
    print("Processing:", basename)

    with open(pkl_path, "rb") as f:
        tf = pickle.load(f)

    tf2 = smooth_tailfit(
        tf,
        smooth_window=smooth_window,
        crop_n=crop_n,
        do_smooth=do_smooth
    )

    x_tip = calc_x_tip(tf2, tip_idx=TIP_POINT_INDEX)
    # 让第一帧 = 0
    x_tip = x_tip - x_tip[0]

    deflection = calc_deflection(
        tf2,
        base_idx=BASELINE_POINT_INDEX,
        tip_idx=TIP_POINT_INDEX
    )

    # 1) 根据 deflection 找候选峰
    peaks_pos, peaks_neg = detect_peaks(deflection)

    # 2) 合并成时间序列
    raw_frames, raw_dirs = merge_raw_peaks(peaks_pos, peaks_neg)

    # 3) 主趋势 + 回弹 + 近邻抖动筛选
    # 3) 第二层筛选：可切换 普通版 / trend版
    if use_trend_filter:
        kept_frames, kept_dirs = filter_active_peaks_with_trend(
            deflection,
            raw_frames,
            raw_dirs,
            min_deflection_deg=min_deflection_deg,
            require_amp_ratio=require_amp_ratio,
            amp_ratio=amp_ratio,
            require_alternation=require_alternation,
            min_peak_gap_all=min_peak_gap_all,
            trend_window=trend_window,
            rebound_ratio=rebound_ratio,
            jitter_gap_same_phase=jitter_gap_same_phase
        )
    else:
        kept_frames, kept_dirs = filter_active_peaks(
            deflection,
            raw_frames,
            raw_dirs,
            min_deflection_deg=min_deflection_deg,
            require_amp_ratio=require_amp_ratio,
            amp_ratio=amp_ratio,
            require_alternation=require_alternation,
            min_peak_gap_all=min_peak_gap_all
        )

    # 4) 供画图使用
    peaks_pos = kept_frames[kept_dirs == "R"]
    peaks_neg = kept_frames[kept_dirs == "L"]

    frames = np.arange(len(x_tip))

    # =====================================================
    # one figure, two curves, dual y-axis
    # =====================================================
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    # colors
    color_xtip = "tab:blue"
    color_defl = "tab:orange"
    color_R = "tab:red"
    color_L = "tab:green"

    # x_tip line
    ax1.plot(frames, x_tip, color=color_xtip, linewidth=1.8, label="x_tip")

    # x_tip = 0 reference line
    ax1.axhline(0, linestyle="--", linewidth=1, color="blue")

    # deflection line
    ax2.plot(frames, deflection, color=color_defl, linewidth=1.8, alpha=0.9, label="deflection")
    ax2.axhline(0, linestyle="--", linewidth=1, color="red")

    # peak markers on x_tip
    if len(peaks_pos) > 0:
        annotate_peak_labels(
            ax1,
            peaks_pos,
            x_tip[peaks_pos],
            label_prefix="R",
            color=color_R,
            dy_text=8
        )

    if len(peaks_neg) > 0:
        annotate_peak_labels(
            ax1,
            peaks_neg,
            x_tip[peaks_neg],
            label_prefix="L",
            color=color_L,
            dy_text=-10
        )

    print("raw_frames:", raw_frames)
    print("raw_dirs:", raw_dirs)
    print("kept_frames:", kept_frames)
    print("kept_dirs:", kept_dirs)
  ## same peak frames mapped onto deflection
  #if len(peaks_right) > 0:
  #    for x, y in zip(peaks_right, deflection[peaks_right]):
  #        ax2.text(
  #            x,
  #            y,
  #            f"R\nF{x}",
  #            fontsize=7,
  #            ha="center",
  #            va="bottom",
  #            color=color_R
  #        )

  #if len(peaks_left) > 0:
  #    for x, y in zip(peaks_left, deflection[peaks_left]):
  #        ax2.text(
  #            x,
  #            y,
  #            f"L\nF{x}",
  #            fontsize=7,
  #            ha="center",
  #            va="bottom",
  #            color=color_L
  #        )

    # labels and title
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("x_tip", color=color_xtip)
    ax2.set_ylabel("Deflection (deg)", color=color_defl)
    plt.title(basename)

    # combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()

    if save_plot:
        outpath = os.path.join(save_folder, basename + "_QC_overlay.png")
        plt.savefig(outpath, dpi=150, bbox_inches="tight")

        if mode == "single":
            save_parameters(basename)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# RUN
# =========================================================
if mode == "single":
    plot_one_file(single_pkl)

elif mode == "batch":

    save_parameters_once()   # ⭐ 只保存一次参数

    for file in os.listdir(folder):
        if file.endswith(".pkl"):
            plot_one_file(os.path.join(folder, file))

else:
    raise ValueError("mode must be 'single' or 'batch'")