"""
This script calculate different variables from the pkl file which contains the tail coordinates tracked by joetailfit script.

Some parameters should be filled up as required (Line 13-18). Max. deflection angle to be consider a swmming movement in Line 204 &208.

The output will be located in the same folder than the input file (pklfile, Line 18). The first row will be the headers
for each variable, the second row will be the value of each variable written in different columns.

peak-to-peak half-beat interval，不是 zero-to-zero half-beat duration。
"""
from babel.plural import to_python

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

"""PARAMETERS TO FILL UP"""
folder_path = r'\\130.238.36.122\znn\YuanqiHua\High speed\Manual_fix_pkl_out'
output_folder = r'\\130.238.36.122\znn\YuanqiHua\High speed\Final analysis\test'
output_folder2 = r'\\130.238.36.122\znn\YuanqiHua\High speed\Final analysis\Log\test'
output_folder3 = r'\\130.238.36.122\znn\YuanqiHua\High speed\Final analysis\First10hb\test'
output_folder4 = r'\\130.238.36.122\znn\YuanqiHua\High speed\Final analysis\Freqmap_long\test'

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)
os.makedirs(output_folder3, exist_ok=True)
os.makedirs(output_folder4, exist_ok=True)
# peak_mode = "tip"   # "tip" 或 "rel"

decimals_required = 3
frame_for_smoothing = 3
crop_n = 1

RUN_ONLY_THIS_FILE = '' # enter name if you want run specifc ones
START_FROM_FOLDER = '260226 MTZ control'   # 例如 '260311 dmrt3 MTZ'

# 回弹filter
use_trend_filter = False   # True=启用大方向/回弹/jitter规则；False=只用普通 filter
trend_window = 20
rebound_ratio = 0.05
jitter_gap_same_phase = 10

# ============================================
# 子文件夹参数配置（方案A）
# ============================================

FOLDER_CONFIG = {
    "260311 dmrt3 MTZ": {
        "prominence": 1, # 这个峰要“高出周围”多少才算真峰
        "amp_ratio": 0.3,
        "distance": 10, # 两个峰之间最少要隔多少帧
        "swim_thresh": 50,
        "baseline_idx" : 1,   # 例如 0=尾根, 1=第二个点, 2=第三个点
        "tip_idx" : -2  ,     # -1=尾尖
        "min_peak_gap_all": 6
    },

    "260312 dmrt3 MTZ": {
        "prominence": 1,  # 这个峰要“高出周围”多少才算真峰
        "amp_ratio": 0.3,
        "distance": 10,  # 两个峰之间最少要隔多少帧
        "swim_thresh": 50,
        "baseline_idx": 1,  # 例如 0=尾根, 1=第二个点, 2=第三个点
        "tip_idx": -2,  # -1=尾尖
        "min_peak_gap_all": 6

    }
}

DEFAULT_CFG = {
    "prominence": 3,
    "amp_ratio": 0.3,
    "distance": 10,
    "swim_thresh": 50,
    "baseline_idx" : 1,  # 例如 0=尾根, 1=第二个点, 2=第三个点
    "tip_idx" : -2,  # -1=尾尖
    "min_peak_gap_all": 6
}

header = ('Larva ID', 'Genotype','Swim_ID',
          'Peak_sequence', 'Peak_deflection',
          'Movement duration', 'Nr. half beats', 'Freq. half beats','Alternation',
          'Max. deflection', 'Frame at max deflection', 'Mean deflection during half beats',
          'Longest tail trajectory among half beats', 'Cumulative tail trajectory', 'Max. curvature',
          'Frame at max curvature', 'Tail segment with the max. angle at max. curvature', 'Mean curvature',
          'Mean curvature during beats', 'Turning bias to R/L','Laterality_Index',
          # ===== FIRST10 ALL =====
            'HB10_duration',
            'HB10_freq',
            'HB10_interval_mean', # 平均每次半拍隔多久出现一次 peak。
            'HB10_interval_cv', # CV 很小（接近 0）：每次间隔都差不多 → 节律很稳. CV 很大：有的间隔很短、有的很长 → 节律乱/抖动/不稳定
            'HB10_alt',
            'HB10_amp_mean', # 前10次摆尾“平均幅度”（强度）。变小：发力弱、尾摆小, 变大：发力强、摆幅大
            'HB10_amp_max',

            # ===== FIRST10 SWIMMING =====
            'HB10s_duration',
            'HB10s_freq',
            'HB10s_interval_mean',
            'HB10s_interval_cv',
            'HB10s_alt',
            'HB10s_amp_mean',
            'HB10s_amp_max', 'Swimming duration',
          'Nr of half beats during swimming', 'Freq. half beating during swimming', 'Max. deflection during swimming',
          'Frame at max deflection swimming', 'Mean deflection during swimming beats', 'Max. curvature swimming',
          'Frame at max curvature when swimming', 'Tail segment with the max. angle at max. curvature',
          'Mean curvature during swimming', 'Mean curvature during swimming beats', 'Turning bias to R/L during swimming',
          'Laterality_Index_swimming','Swimming_valid', 'Fail_reason')

hidx = {h: i for i, h in enumerate(header)}



def process_one_folder(subfolder, pkl_files, fps, cfg):
    # 1) 解包参数
    PROMINENCE = cfg["prominence"]
    DISTANCE = cfg["distance"]
    Amp_ratio = cfg["amp_ratio"]
    SWIM_THRESH = cfg["swim_thresh"]
    BASELINE_POINT_INDEX = cfg["baseline_idx"]
    TIP_POINT_INDEX = cfg["tip_idx"]
    MIN_PEAK_GAP_ALL = cfg["min_peak_gap_all"]

    # 2) 当前文件夹输出名
    folder_name = os.path.basename(subfolder)
    filename = f'{folder_name}_{int(fps)}fps'

    log_path = os.path.join(output_folder2, f"{folder_name}_{int(fps)}fps_debug.txt")
    log_file = open(log_path, "w")

    def log_print(*args):
        text = " ".join(str(a) for a in args)
        print(text)
        log_file.write(text + "\n")

    # 3) 容器（每个子文件夹都要清空）
    all_results = []
    hb10_long = []
    freqmap_long = []

    for pkl_filename in pkl_files:
        pklfile = os.path.join(subfolder, pkl_filename)

        # ====== 这里粘贴你原本“打开pkl→算变量→result_row”那一大段 =====
        log_print(f'Processing {pkl_filename}')

        basename = os.path.splitext(pkl_filename)[0]  # # wt1-Mtz-control-1_0
        left, swim_id, _ = basename.split('_', 2)
        # left = "wt1-Mtz-control-1"
        # swim_id = "0"
        Swim_ID = int(swim_id)

        parts = left.split('-')  # # ["wt1", "Mtz", "control", "1"]
        Larva_ID = int(parts[-1])
        Genotype = '-'.join(parts[:-1])

        # pklfile = r'C:\Users\huayuanqi\OneDrive - Uppsala universitet\Desktop\PhD\PhD\high speed\scripy\ts3_000052-1.pkl'

        try:
            Swimming_valid = 1
            Fail_reason = ''

            """ TO OPEN tHE pkl FILE:"""
            with open(pklfile, 'rb') as d:
                tf = pickle.load(d)

            # print len(tf) # to print the number of frames in tf (tailfit array)
            # print "shape tf array", np.shape(tf) # to print the dimensions of tf (tailfit array)
            # print tf[:4]

            """TO SMOOTH THE TAIL_FIT DATA in both axis (x,y):"""
            from astropy.convolution import convolve, Box1DKernel
            tf_swapped = np.swapaxes(tf, 0, 1)
            # print "shape tf_swapped array", np.shape(tf_swapped)
            # print tf_swapped[:,10]
            tf_swapped_smoothed = []  # creating a empty list where to store the smoothed data we'll generate in the following loop
            for tailpoint in tf_swapped:
                x_smoothed = convolve(tailpoint[:, 0], Box1DKernel(frame_for_smoothing)) # apply moving average every 15 frame by Boxcar filter
                y_smoothed = convolve(tailpoint[:, 1], Box1DKernel(frame_for_smoothing))

                if crop_n > 0:
                    x_smoothed_cropped = x_smoothed[crop_n:-crop_n] # discard the first and last 5 frames from the smoothed data
                    y_smoothed_cropped = y_smoothed[crop_n:-crop_n]
                else:
                    x_smoothed_cropped = x_smoothed
                    y_smoothed_cropped = y_smoothed

                smoothed_coordinates = np.array(list(zip(x_smoothed_cropped, y_smoothed_cropped)))

                tf_swapped_smoothed.append(
                    smoothed_coordinates)  # including the smoothed coordinates in the empty array

                """
                plt.plot(x, 'y') #plotting original x data in yellow
                plt.plot(x_smoothed_cropped, 'b')  # plotting smoothed x data in blue
                plt.show()
                plt.plot(y, 'r') #plotting original y data in red
                plt.plot(y_smoothed_cropped, 'g')  #plotting smoothed y data in green
                plt.show()
                """

            # print "smoothed coordinates", np.shape(smoothed_coordinates)
            # print "tf_swapped_smoothed shape", np.shape(tf_swapped_smoothed)
            smoothed_tf = np.swapaxes(tf_swapped_smoothed, 0, 1)
            # print "smoothed_tf", np.shape(smoothed_tf)

            """
            plt.plot(smoothed_tf[:, -1,0], 'b')  # smoothed tailtip data in blue
            plt.plot(smoothed_tf[:, 0,0], 'r')  # smoothed tail-basedata in red
            plt.plot(smoothed_tf[:, 9,0], 'g')  # smoothed mid-tail in green
            plt.show()
            """

            """TO CALCULATE THE DEFLECTION ANGLE of the tail tip per each frame:"""
            dx = [frame[TIP_POINT_INDEX, 0] - frame[BASELINE_POINT_INDEX, 0]
                  for frame in
                  smoothed_tf]  # distance in the x axis from tail baseline to the tail tip for each frame
            dy = [frame[TIP_POINT_INDEX, 1] - frame[BASELINE_POINT_INDEX, 1]
                  for frame in
                  smoothed_tf]  # distance in the y axis from tail baseline to the tail tip for each frame

            deflection = np.degrees(np.arctan2(dx,
                                               dy))  # tangent (in degrees) of the tail tip ditance in axis x and axis y from the basal tip point
            # print "max. deflection to the right (in degrees):", max(deflection), "& frame of max. deflection to the right:", np.argmax(deflection)+1
            # print "max. deflection to the left (in degrees):", min(deflection), "& frame of max. deflection to the left:", np.argmin(deflection)+1

            max_deflection = round(max(abs(deflection)), decimals_required)
            frame_max_deflection = np.argmax(abs(deflection)) + 1

            """TO CALCULATE THE CURVATURE ALONG THE TAIL per each frame:"""

            def tail_segment(P1, P2):
                return np.degrees(np.arctan2(P2[1] - P1[1], P2[0] - P1[
                    0]))  # P(Y,X) to get the smallest angle to the reference axis

            atan_segments = []
            # for frame in np.array(zip(smoothed_tf[:, :-1], smoothed_tf[:, 1:])): # loop to calculate the angles of each segment per each frame
            for P1s, P2s in zip(smoothed_tf[:, :-1], smoothed_tf[:, 1:]):
                atan_segments_frame = []
                for P1, P2 in zip(P1s, P2s):
                    # for i in frame[0]:
                    #    for j in frame[1]:
                    #       P1 = i
                    #        P2 = j
                    atan_segments_frame.append(tail_segment(P1, P2))
                atan_segments.append(atan_segments_frame)

            angles_segments = np.array([(angle_frame[0] - angle_frame[1]) for angle_frame in
                                        zip(np.array(atan_segments)[:, :-1], np.array(atan_segments)[
                                            :, 1:])])  # getting the angles between consecutive segments per frame
            angles_between_segments = [
                [abs(angle) if abs(angle) <= 180 else abs(360 - abs(angle)) for angle in frame] for frame in
                angles_segments]  # converting all the angles in absolute values
            k = [sum(frame[:]) for frame in np.array(
                angles_between_segments)]  # Curvature (k), addition of all the angles between segments per frame

            max_curvature = round(max(k), decimals_required)
            frame_max_curvature = np.argmax(k) + 1
            segment_max_angle_when_max_curvature = np.argmax(angles_between_segments[np.argmax(k)])
            # print np.argmax(angles_between_segments, 1)

            """
            #TO PICK THE BEGINING AND END OF THE MOVEMENT & DURATION
            Increment_k = [abs(frame[0] - frame[1]) for frame in zip(k[:-1], k[1:])] #Calculate the difference in curvature from frame to frame
            #print Increment_k

            before_movement_starts = [] #generate an empty array where to store the frames before the movement happens
            for item in Increment_k:
                if item > 2: #I have established changes above 0.5 degrees as the threshold for movements
                    break
                #print item, k.index(item)+1
                before_movement_starts.append(Increment_k.index(item)+1)
            print "movement starts at frame:", before_movement_starts[-1]

            movement_ends = []
            for item in Increment_k[::-1]: #for item in the frame-reversed curvature array
                if item > 2: #I have established cahnges above 0.5 degrees as the threshold for movements
                    break
                #print item, k.index(item)+1
                movement_ends.append(Increment_k.index(item)+1)
            print "movement ends at frame:", movement_ends[-1]

            Movement_duration = movement_ends[-1] - before_movement_starts[-1]
            print "movement duration (in frames):", Movement_duration
            """

     #     """TO FIND THE HALF BEATS by https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks"""
     #     # Peaks: toward the right:

     #     # ===== choose peak signal: tip or rel =====
     #     x_tip = np.asarray([frame[TIP_POINT_INDEX, 0] for frame in smoothed_tf])
     #     x_base = np.asarray([frame[BASELINE_POINT_INDEX, 0]  for frame in smoothed_tf])

     #     if peak_mode == "tip":
     #         smoothed_tf_x = x_tip
     #     elif peak_mode == "rel":
     #         smoothed_tf_x = x_tip - x_base
     #     else:
     #         raise ValueError(f"Unknown peak_mode: {peak_mode}")


     #     from scipy.signal import find_peaks

     #     peaks_right, properties_peaks_right = find_peaks(
     #         smoothed_tf_x,
     #         distance=DISTANCE,
     #         prominence=PROMINENCE,
     #         width=1,
     #         rel_height=0.5
     #     )
     #     # Trough: toward the left:
     #     inverted_smoothed_tf_x = -smoothed_tf_x
     #     peaks_left, properties_peaks_left = find_peaks(inverted_smoothed_tf_x, distance=DISTANCE,
     #                                                    prominence=PROMINENCE, width=1, rel_height=0.5)

            """TO FIND THE HALF BEATS from DEFLECTION (same as plot script)"""
            from scipy.signal import find_peaks

            # 右摆峰：deflection 局部最大
            peaks_right, properties_peaks_right = find_peaks(
                deflection,
                distance=DISTANCE,
                prominence=PROMINENCE,
                width=1,
                rel_height=0.5
            )

            # 左摆峰：deflection 局部最小
            peaks_left, properties_peaks_left = find_peaks(
                -deflection,
                distance=DISTANCE,
                prominence=PROMINENCE,
                width=1,
                rel_height=0.5
            )

            # =====  filter active half-beats (exclude rebound-like peaks) ====================================
            def filter_active_halfbeats(peaks_right, peaks_left, deflection,
                                        amp_ratio=0.3,
                                        require_amp_ratio = False,
                                        require_alternation=False,
                                        interactive_drop=False,
                                        min_peak_gap_all=MIN_PEAK_GAP_ALL,
                                        log_print=print):
                """
                返回：
                  peaks_active, dirs_active, idx_in_right, idx_in_left
                interactive_drop=True 时：当某个峰按规则要 drop，会询问你 keep/drop
                """

                def fmt_peak(tag, j, frame, dir_, defl, amp, extra=""):
                    # j: 3位宽，frame: 5位宽，dir: 1位，defl/amp: 7位宽保留2位（含符号）
                    base = (
                        f"{tag:<18}"  # tag 左对齐，宽18（保证 KEEP/DROP 对齐）
                        f"j={j:>3d}  "
                        f"frame={frame:>5d}  "
                        f"dir={dir_:>1s}  "
                        f"defl={defl:>7.2f}  "
                        f"amp={amp:>7.2f}"
                    )
                    if extra:
                        base += "  " + extra
                    return base

                log_print("DEBUG param require_alternation = ", require_alternation, ", amp_ratio =", amp_ratio)
                print()
                peaks_right = np.asarray(peaks_right, dtype=int)
                peaks_left = np.asarray(peaks_left, dtype=int)

                # 记录每个 peak 来自 right 还是 left，以及它在各自数组里的索引 i
                items = []
                for i, p in enumerate(peaks_right):
                    items.append((p, "R", i))
                for i, p in enumerate(peaks_left):
                    items.append((p, "L", i))

                # 按时间排序
                items.sort(key=lambda x: x[0])

                frames = np.array([x[0] for x in items], dtype=int)
                dirs = np.array([x[1] for x in items], dtype=object)
                raw_i = np.array([x[2] for x in items], dtype=int)  # 在各自 peaks_right/left 里的索引
                defl_at = np.asarray(deflection, float)[frames]  # 峰对应帧的 deflection
                A = np.abs(defl_at)  # 峰的幅度



                #A = np.abs(np.asarray(deflection, dtype=float)[frames])

                kept = []
                history = []  # 记录每次决定: (j, "keep"/"drop")

                j = 0

                def ask_user(frame, d, amp, reason, thr=None, prev=None):
                    """返回 True=保留, False=丢弃"""
                    msg = f"[DROP?] frame={frame} dir={d} amp={amp:.2f} reason={reason}"
                    if thr is not None:
                        msg += f" thr={thr:.2f}"
                    if prev is not None:
                        msg += f" prev_amp={prev:.2f}"
                    log_print(msg)

                    while True:
                        ans = input("Keep this peak? (y=keep / n=drop / u=undo) > ").strip().lower()
                        if ans in ("y", "yes"):
                            log_print("  -> USER KEEP")
                            return True
                        if ans in ("n", "no"):
                            log_print("  -> USER DROP")
                            return False
                        if ans in ("u", "undo"):
                            log_print("  -> USER UNDO")
                            return "undo"
                        print("Please type y, n or u.")


                log_print("DEBUG require_alternation =", require_alternation, "amp_ratio =", amp_ratio,
                          "interactive_drop =", interactive_drop)
                print()


                # plus debug version
                min_amp_deg = 4.0

                while j in range(len(frames)):
                    if len(kept) == 0:
                        kept.append(j)
                        history.append((j, "keep"))

                        # log_print(f"[KEEP first] j={j}, frame={frames[j]}, dir={dirs[j]},defl={defl_at[j]:.2f},  amp={A[j]:.2f}")
                        log_print(fmt_peak("[KEEP first]", j, frames[j], dirs[j], defl_at[j], A[j]))
                        print()
                        j += 1
                        continue

                    # 规则0：跨方向最小间隔（不分 R/L）
                    if min_peak_gap_all > 0 and (frames[j] - frames[kept[-1]] < min_peak_gap_all):
                        log_print(
                            f"[DROP peak_gap] j={j} frame={frames[j]} dir={dirs[j]} "
                            f"defl={defl_at[j]:.2f} amp={A[j]:.2f} "
                            f"< gap {min_peak_gap_all} from prev_kept_frame={frames[kept[-1]]}"
                        )
                        print()
                        j += 1
                        continue

                    # 规则1：交替性
                    if require_alternation and dirs[j] == dirs[kept[-1]]:
                        if interactive_drop:
                            if ask_user(frames[j], dirs[j], A[j], reason = "alternation_same_direction"):
                                kept.append(j)
                        else:
                            # log_print(f"[DROP alternation] j={j} frame={frames[j]} dir={dirs[j]},defl={defl_at[j]:.2f} amp={A[j]:.2f}")
                            log_print(fmt_peak("[DROP alternation]", j, frames[j], dirs[j], defl_at[j], A[j]))
                            print()
                        j += 1
                        continue

                  ## 规则2：第二个峰太小（你原来逻辑）
                  #if len(kept) == 1 and A[j] < min_amp_deg:
                  #    if interactive_drop:
                  #        if ask_user(frames[j], dirs[j], A[j], reason = f"2nd_too_small(<{min_amp_deg})"):
                  #            kept.append(j)
                  #    else:
                  #        log_print(f"[DROP 2nd too small] j={j}, frame={frames[j]}, dir={dirs[j]}, defl={defl_at[j]:.2f}, amp={A[j]:.2f}")
                  #        print()
                  #    continue

                    # 规则3：amp_ratio（相对上一保留峰幅度太小）
                    thr = amp_ratio * A[kept[-1]]
                    if require_amp_ratio and A[j] < thr:
                        # ✅ 新增：如果 deflection 符号与上一个保留峰相反（交替），直接保留
                        if defl_at[j] * defl_at[kept[-1]] < 0:
                            kept.append(j)
                            history.append((j, "keep"))
                            log_print(fmt_peak("[USER KEEP]", j, frames[j], dirs[j], defl_at[j], A[j]))
                            print()
                            j += 1
                            continue

                        log_print(
                            f"[DROP amp_ratio] j={j} frame={frames[j]} ",
                            f"dir={dirs[j]} ",
                            f"defl={defl_at[j]:.2f} ",
                            f"amp={A[j]:.2f} < thr={thr:.2f} (prev_amp={A[kept[-1]]:.2f})"
                        )
                        print()

                        # ===== 新增交互 =====
                        if interactive_drop:
                            resp = ask_user(frames[j], dirs[j], A[j], reason="amp_ratio", thr=thr, prev=A[kept[-1]])

                            if resp == "undo":
                                if history:
                                    last_j, last_action = history.pop()
                                    if last_action == "keep" and kept and kept[-1] == last_j:
                                        kept.pop()
                                    j = last_j
                                    log_print(f"[UNDO] back to j={j} frame={frames[j]}")
                                    print()
                                else:
                                    log_print("[UNDO] nothing to undo")
                                    print()
                                continue

                            elif resp is True:
                                kept.append(j)
                                history.append((j, "keep"))
                            else:
                                history.append((j, "drop"))
                        else:
                            history.append((j, "drop"))

                        j += 1
                        continue

                    kept.append(j)
                    history.append((j, "keep"))
                    log_print(fmt_peak("[KEEP]", j, frames[j], dirs[j], defl_at[j], A[j]))
                    print()
                    j += 1
                    continue

                kept = np.asarray(kept, dtype=int)

                peaks_active = frames[kept]
                dirs_active = dirs[kept]

                # ✅ 关键：返回它在 raw peaks_right/left 中的索引 i（用于 properties）
                idx_in_right = np.array([raw_i[j] if dirs[j] == "R" else -1 for j in kept], dtype=int)
                idx_in_left = np.array([raw_i[j] if dirs[j] == "L" else -1 for j in kept], dtype=int)

                return peaks_active, dirs_active, idx_in_right, idx_in_left

            def merge_raw_peaks(peaks_right, peaks_left):
                items = []
                for i, p in enumerate(peaks_right):
                    items.append((int(p), "R", i))
                for i, p in enumerate(peaks_left):
                    items.append((int(p), "L", i))

                items.sort(key=lambda x: x[0])

                frames_raw = np.asarray([x[0] for x in items], dtype=int)
                dirs_raw = np.asarray([x[1] for x in items], dtype=object)
                raw_i = np.asarray([x[2] for x in items], dtype=int)
                return frames_raw, dirs_raw, raw_i


            def filter_active_peaks_with_trend(peaks_right, peaks_left, deflection,
                                               min_deflection_deg=0.0,
                                               require_amp_ratio=False,
                                               amp_ratio=0.30,
                                               require_alternation=False,
                                               min_peak_gap_all=0,
                                               trend_window=12,
                                               rebound_ratio=0.35,
                                               jitter_gap_same_phase=8,
                                               log_print=print):

                frames_raw, dirs_raw, raw_i = merge_raw_peaks(peaks_right, peaks_left)

                if len(frames_raw) == 0:
                    return (
                        np.array([], dtype=int),
                        np.array([], dtype=object),
                        np.array([], dtype=int),
                        np.array([], dtype=int)
                    )

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

                    # 规则0：最小 deflection
                    if a < min_deflection_deg:
                        continue

                    # 第一个峰
                    if len(kept_idx) == 0:
                        kept_idx.append(j)
                        log_print(f"[KEEP first trend] j={j} frame={f} dir={d} amp={a:.2f}")
                        continue

                    prev_j = kept_idx[-1]
                    prev_f = frames_raw[prev_j]
                    prev_d = dirs_raw[prev_j]
                    prev_a = amps[prev_j]

                    # 规则1：全局最小峰间隔
                    if min_peak_gap_all > 0 and (f - prev_f) < min_peak_gap_all:
                        log_print(f"[DROP peak_gap trend] j={j} frame={f} dir={d} amp={a:.2f}")
                        continue

                    # 规则2：方向交替
                    if require_alternation and d == prev_d:
                        log_print(f"[DROP alternation trend] j={j} frame={f} dir={d} amp={a:.2f}")
                        continue

                    # 规则3：amp_ratio
                    if require_amp_ratio:
                        thr = amp_ratio * prev_a
                        if a < thr:
                            log_print(
                                f"[DROP amp_ratio trend] j={j} frame={f} dir={d} "
                                f"amp={a:.2f} < thr={thr:.2f} (prev_amp={prev_a:.2f})"
                            )
                            continue

                    # 规则4：趋势 + 小回弹排除
                    if d != prev_d:
                        tr = local_trend(deflection, f, w=trend_window)

                        if d == "R" and tr < 0 and a < rebound_ratio * prev_a:
                            log_print(
                                f"[DROP rebound trend] j={j} frame={f} dir={d} "
                                f"amp={a:.2f}, trend={tr:.2f}, prev_amp={prev_a:.2f}"
                            )
                            continue

                        if d == "L" and tr > 0 and a < rebound_ratio * prev_a:
                            log_print(
                                f"[DROP rebound trend] j={j} frame={f} dir={d} "
                                f"amp={a:.2f}, trend={tr:.2f}, prev_amp={prev_a:.2f}"
                            )
                            continue

                    # 规则5：近邻抖动排除
                    if d != prev_d:
                        if (f - prev_f) < jitter_gap_same_phase and a < rebound_ratio * prev_a:
                            log_print(
                                f"[DROP jitter trend] j={j} frame={f} dir={d} "
                                f"amp={a:.2f}, prev_frame={prev_f}, prev_amp={prev_a:.2f}"
                            )
                            continue

                    kept_idx.append(j)
                    log_print(f"[KEEP trend] j={j} frame={f} dir={d} amp={a:.2f}")

                kept_idx = np.asarray(kept_idx, dtype=int)

                peaks_active = frames_raw[kept_idx]
                dirs_active = dirs_raw[kept_idx]

                idx_in_right = np.array(
                    [raw_i[j] if dirs_raw[j] == "R" else -1 for j in kept_idx],
                    dtype=int
                )
                idx_in_left = np.array(
                    [raw_i[j] if dirs_raw[j] == "L" else -1 for j in kept_idx],
                    dtype=int
                )

                return peaks_active, dirs_active, idx_in_right, idx_in_left



            if use_trend_filter:
                peaks_active, dirs_active, idxR_active, idxL_active = filter_active_peaks_with_trend(
                    peaks_right, peaks_left, deflection,
                    min_deflection_deg=0,
                    require_amp_ratio=False,
                    amp_ratio=Amp_ratio,
                    require_alternation=False,
                    min_peak_gap_all=MIN_PEAK_GAP_ALL,
                    trend_window=trend_window,
                    rebound_ratio=rebound_ratio,
                    jitter_gap_same_phase=jitter_gap_same_phase,
                    log_print=log_print
                )
            else:
                peaks_active, dirs_active, idxR_active, idxL_active = filter_active_halfbeats(
                    peaks_right, peaks_left, deflection,
                    amp_ratio=Amp_ratio,
                    require_amp_ratio=False,
                    require_alternation=False,
                    min_peak_gap_all=MIN_PEAK_GAP_ALL,
                    log_print=log_print
                )


            log_print("raw peaks R/L:", len(peaks_right), len(peaks_left))
            log_print("active peaks:", len(peaks_active))
            print()
            log_print("Mean abs deflection in peaks:", np.mean(np.abs(deflection[peaks_active])))
            log_print("Max deflection:", np.max(np.abs(deflection)))
            log_print('-----------------------------------------------')

            Dict_right = dict({'Peaking_frame': peaks_right}, **properties_peaks_right)
            Dict_left = dict({'Peaking_frame': peaks_left}, **properties_peaks_left)

            # print "half beats to the right at frames:",(peaks_right)
            # print "half beats to the left at frames:",(peaks_left)

            # 修改
            ### >>> 修改开始：允许只有一边peaks；两边都没有就跳过，防止IndexError
            ### >>> 修改开始：movement start/end 基于 active half-beats（保持一致）

            # 如果没有 active peaks，直接按你原来逻辑跳过
            if len(peaks_active) == 0:
                print(f"WARNING: No ACTIVE peaks found in {pkl_filename}. Saving NaNs and skipping this file.")
                nan = float('nan')

                # --- build a header-aligned row to avoid column shift ---
                row = [nan] * len(header)
                row[hidx['Larva ID']] = Larva_ID
                row[hidx['Genotype']] = Genotype
                row[hidx['Swim_ID']] = Swim_ID

                row[hidx['Peak_sequence']] = ''
                row[hidx['Peak_deflection']] = ''

                # 仍然保留你已经算出来的全局指标（可选，但很合理）
                row[hidx['Max. deflection']] = max_deflection
                row[hidx['Frame at max deflection']] = frame_max_deflection
                row[hidx['Max. curvature']] = max_curvature
                row[hidx['Frame at max curvature']] = frame_max_curvature
                row[hidx[
                    'Tail segment with the max. angle at max. curvature']] = segment_max_angle_when_max_curvature

                # 没有 half-beats => swimming 也判无效
                row[hidx['Swimming_valid']] = 0
                row[hidx['Fail_reason']] = 'No ACTIVE peaks found'

                all_results.append(tuple(row))
                continue

            left_ips_active = []
            right_ips_active = []

            for d, ir, il in zip(dirs_active, idxR_active, idxL_active):
                if d == "R" and ir >= 0:
                    left_ips_active.append(properties_peaks_right["left_ips"][ir])
                    right_ips_active.append(properties_peaks_right["right_ips"][ir])
                elif d == "L" and il >= 0:
                    left_ips_active.append(properties_peaks_left["left_ips"][il])
                    right_ips_active.append(properties_peaks_left["right_ips"][il])

            left_ips_active = np.asarray(left_ips_active, float)
            right_ips_active = np.asarray(right_ips_active, float)

            # start/end 用 active ips
            start_i = int(np.min(left_ips_active))
            end_i = int(np.max(right_ips_active))
            end_i = min(end_i, len(k))

            ### <<< 修改结束

            # mean_curvature = round(np.mean(k[int(min(properties_peaks_right["left_ips"][0], properties_peaks_left["left_ips"][0])):int(max(properties_peaks_right["left_ips"][-1], properties_peaks_left["left_ips"][-1]))]), decimals_required)
            mean_curvature = round(np.mean(k[start_i:end_i]), decimals_required)

            # print "Tail path per beat:", properties_peaks_right["prominences"], properties_peaks_left["prominences"]
            prom_active = []
            for d, ir, il in zip(dirs_active, idxR_active, idxL_active):
                if d == "R" and ir >= 0:
                    prom_active.append(properties_peaks_right["prominences"][ir])
                elif d == "L" and il >= 0:
                    prom_active.append(properties_peaks_left["prominences"][il])

            prom_active = np.asarray(prom_active, float)

            Longest_tail_trajectory = round(np.max(prom_active), decimals_required)
            Mean_tail_trajectory = round(np.mean(prom_active), decimals_required)
            Cumulative_tail_trajectory = round(np.sum(prom_active), decimals_required)

            """DURATION of THE MOVEMENT"""
            # Movement_duration = round(((max(properties_peaks_right["right_ips"][-1], properties_peaks_left["right_ips"][-1])) - (min(properties_peaks_right["left_ips"][0], properties_peaks_left["left_ips"][0])))/fps,decimals_required)
            Movement_duration = round((end_i - start_i) / fps, decimals_required)

            # print "first frame of the movement:", min(properties_peaks_right["left_ips"][0], properties_peaks_left["left_ips"][0])
            # print "last frame of the movement:", max(properties_peaks_right["left_ips"][-1], properties_peaks_left["left_ips"][-1])

            """TAIL HALF BEAT FREQUENCY"""
            Nr_half_beats = len(peaks_active)  # Total number of half beats (to the right and to the left)
            Freq_half_beats = round(float(Nr_half_beats) / (Movement_duration), decimals_required)

            """DEFLECTION IN THE PEAKS"""
            Deflection_in_peaks = []
            Deflection_in_peaks_right = []
            Deflection_in_peaks_left = []
            Curvature_in_peaks = []

            for element, direction in zip(peaks_active, dirs_active):
                Deflection_in_peaks.append(deflection[element])
                Curvature_in_peaks.append(k[element])

                if direction == "R":
                    Deflection_in_peaks_right.append(deflection[element])
                else:
                    Deflection_in_peaks_left.append(deflection[element])

            Peak_sequence = ''.join(dirs_active.tolist())
            Peak_deflection = ','.join([f"{deflection[p]:.{decimals_required}f}" for p in peaks_active])

            # ===== ALL active-peak intervals for frequency heatmap =====
            # 每一行表示：peak_i 到 peak_(i+1) 这段时间里的频率
            # 时间对齐到 movement start，也就是 start_i -> 0 ms
            if len(peaks_active) >= 2:
                align_frame = start_i   # movement start 对齐为 0 ms

                for ii in range(len(peaks_active) - 1):
                    p1 = int(peaks_active[ii])
                    p2 = int(peaks_active[ii + 1])

                    interval_frames = p2 - p1
                    interval_s = interval_frames / fps if interval_frames > 0 else float('nan')
                    inst_freq_hz = (1.0 / interval_s) if interval_frames > 0 else float('nan')

                    start_ms = (p1 - align_frame) / fps * 1000.0
                    end_ms = (p2 - align_frame) / fps * 1000.0

                    freqmap_long.append((
                        Larva_ID,
                        Genotype,
                        Swim_ID,
                        ii + 1,          # Interval_index
                        p1,              # Peak1_frame
                        p2,              # Peak2_frame
                        round(start_ms, 3),
                        round(end_ms, 3),
                        round(interval_s, 6) if np.isfinite(interval_s) else float('nan'),
                        round(inst_freq_hz, 3) if np.isfinite(inst_freq_hz) else float('nan')
                    ))

            # ===== Alternation Index (use filtered active peaks) =====
            if len(dirs_active) <= 1:
                Alternation = float('nan')
            else:
                Alternation = float(np.mean(dirs_active[1:] != dirs_active[:-1]))

            # ===== FIRST 10 HALF-BEATS (ALL) =====
            # 合并左右 peaks（改为使用过滤后的 active peaks，最小修改）
            all_peaks = list(zip(peaks_active.tolist(), dirs_active.tolist()))

            # peaks_active 一般已经排序了，但保守起见仍排序一次（不影响）
            all_peaks.sort(key=lambda x: x[0])

            # 取前10
            first10 = all_peaks[:10]

            HB10_n = len(first10)

            if HB10_n < 2:
                HB10_duration = float('nan')
                HB10_freq = float('nan')
                HB10_interval_mean = float('nan')
                HB10_interval_cv = float('nan')
                HB10_alt = float('nan')
                HB10_amp_mean = float('nan')
                HB10_amp_max = float('nan')
            else:
                frames10 = [x[0] for x in first10]
                dirs10 = [x[1] for x in first10]

                # duration
                HB10_duration = (frames10[-1] - frames10[0]) / fps

                # frequency (9 intervals for 10 peaks)
                HB10_freq = (HB10_n - 1) / HB10_duration if HB10_duration > 0 else float('nan')

                # intervals
                intervals = [(frames10[i + 1] - frames10[i]) / fps for i in range(HB10_n - 1)]
                for idx, interval in enumerate(intervals):
                    if interval > 0:
                        inst_freq = 1.0 / interval
                    else:
                        inst_freq = float('nan')

                    hb10_long.append((
                        Larva_ID,
                        Genotype,
                        Swim_ID,
                        idx + 1,  # HB_index (1~9)
                        interval,
                        inst_freq
                    ))

                HB10_interval_mean = np.mean(intervals)
                HB10_interval_cv = np.std(intervals) / HB10_interval_mean if HB10_interval_mean > 0 else float(
                    'nan')

                # alternation
                alternations = sum(dirs10[i] != dirs10[i + 1] for i in range(HB10_n - 1))
                HB10_alt = alternations / (HB10_n - 1)

                # amplitude（用 deflection）
                amps = [abs(deflection[f]) for f in frames10]
                HB10_amp_mean = np.mean(amps)
                HB10_amp_max = np.max(amps)

            Mean_deflection_peaks = round(
                np.mean([abs(deflection_element) for deflection_element in Deflection_in_peaks]), decimals_required)
            Mean_curvature_peaks = round(np.mean(Curvature_in_peaks), decimals_required)

            """TURN BIAS (RIGHT/LEFT)"""
            # bias_right = 0
            # bias_left = 0
            # for element in Deflection_in_peaks:
            #    if element > 0:
            #        bias_right = bias_right +1
            #    if element < 0:
            #        bias_left = bias_left + 1
            # Bias_RL = round(float(bias_right)/float(bias_left), decimals_required)

            ### >>> 修改开始：Bias_RL不除0 + Laterality_Index
            bias_right = np.sum(np.array(Deflection_in_peaks) > 0)
            bias_left = np.sum(np.array(Deflection_in_peaks) < 0)

            if bias_left == 0 and bias_right > 0:
                Bias_RL = np.inf
            elif bias_right == 0 and bias_left > 0:
                Bias_RL = 0.0
            elif (bias_right + bias_left) == 0:
                Bias_RL = float('nan')
            else:
                Bias_RL = round(float(bias_right) / float(bias_left), decimals_required)

            if (bias_right + bias_left) == 0:
                Laterality_Index = float('nan')
            else:
                Laterality_Index = round((bias_right - bias_left) / (bias_right + bias_left), decimals_required)
            ### <<< 修改结束

            """SELECTING ONLY SWIMMING -  Swimming duration"""
            ### >>> 修改开始：swimming 判定改用 active peaks（口径一致、排除回弹峰）
            Left_ips_Swimming = []
            Right_ips_Swimming = []
            peaks_swim_active = []
            dirs_swim_active = []

            for p, d, ir, il in zip(peaks_active, dirs_active, idxR_active, idxL_active):
                if abs(deflection[p]) < SWIM_THRESH:
                    peaks_swim_active.append(p)
                    dirs_swim_active.append(d)  # d 是 "R" 或 "L"
                    if d == "R" and ir >= 0:
                        Left_ips_Swimming.append(properties_peaks_right["left_ips"][ir])
                        Right_ips_Swimming.append(properties_peaks_right["right_ips"][ir])
                    elif d == "L" and il >= 0:
                        Left_ips_Swimming.append(properties_peaks_left["left_ips"][il])
                        Right_ips_Swimming.append(properties_peaks_left["right_ips"][il])

            Left_ips_Swimming = np.asarray(Left_ips_Swimming, float)
            Right_ips_Swimming = np.asarray(Right_ips_Swimming, float)
            ### <<< 修改结束

            ### >>> 修改开始：swimming start/end 直接用 active 的 ips
            all_left_swim = Left_ips_Swimming.tolist()
            all_right_swim = Right_ips_Swimming.tolist()
            ### <<< 修改结束

            if len(all_left_swim) == 0 or len(all_right_swim) == 0:
                Swimming_valid = 0
                Fail_reason = 'No swimming beats after threshold (<50deg)'
                # 这里先给 swimming 相关变量占位（推荐 NaN，方便后续统计时自动忽略）
                nan = float('nan')
                Swimming_duration = nan
                Nr_half_beats_swimming = 0
                freq_half_beats_swimming = nan
                max_deflection_swimming = nan
                frame_max_deflection_swimming = nan
                mean_deflection_swimming_peaks = nan
                max_curvature_swimming = nan
                frame_max_curvature_swimming = nan
                segment_max_angle_when_max_curvature_swimming = nan
                mean_curvature_swimming = nan
                mean_curvature_swimming_peaks = nan
                Bias_RL_swimming = nan
                Laterality_Index_swimming = nan
                ### >>> 修改开始：Swimming 无效时 HB10s_* 必须占位，否则 result_row 会 UnboundLocalError
                HB10s_duration = nan
                HB10s_freq = nan
                HB10s_interval_mean = nan
                HB10s_interval_cv = nan
                HB10s_alt = nan
                HB10s_amp_mean = nan
                HB10s_amp_max = nan
                ### <<< 修改结束

            else:
                swim_start = int(np.floor(np.min(Left_ips_Swimming)))
                swim_end = int(np.ceil(np.max(Right_ips_Swimming)))
                swim_end = min(swim_end, len(k))  # 防越界
                Swimming_duration = round((swim_end - swim_start) / fps, decimals_required)

            if Swimming_valid == 1:

                """VARIABLES DURING SWIMMING"""
                # HALF BEATS & TAIL HALF BEAT FREQUENCY:
                peaks_swimming_right = []
                peaks_swimming_left = []
                ### >>> 修改开始：swimming peaks 统一用 swim_start/swim_end（避免单侧为空 IndexError）
                # active swimming peaks split by direction
                peaks_swimming_right = [p for p, d in zip(peaks_swim_active, dirs_swim_active) if d == "R"]
                peaks_swimming_left = [p for p, d in zip(peaks_swim_active, dirs_swim_active) if d == "L"]

                Nr_half_beats_swimming = len(peaks_swimming_right) + len(peaks_swimming_left)
                freq_half_beats_swimming = round(Nr_half_beats_swimming / Swimming_duration,
                                                 decimals_required) if Swimming_duration > 0 else float('nan')

                # ===== FIRST 10 HALF-BEATS (SWIMMING ONLY) =====

                swim_peaks = []

                swim_peaks = [(p, "R") for p in peaks_swimming_right] + [(p, "L") for p in peaks_swimming_left]
                swim_peaks.sort(key=lambda x: x[0])

                first10_swim = swim_peaks[:10]

                HB10s_n = len(first10_swim)

                if HB10s_n < 2:
                    HB10s_duration = float('nan')
                    HB10s_freq = float('nan')
                    HB10s_interval_mean = float('nan')
                    HB10s_interval_cv = float('nan')
                    HB10s_alt = float('nan')
                    HB10s_amp_mean = float('nan')
                    HB10s_amp_max = float('nan')
                else:
                    frames10s = [x[0] for x in first10_swim]
                    dirs10s = [x[1] for x in first10_swim]

                    HB10s_duration = (frames10s[-1] - frames10s[0]) / fps
                    HB10s_freq = (HB10s_n - 1) / HB10s_duration if HB10s_duration > 0 else float('nan')

                    intervals_s = [(frames10s[i + 1] - frames10s[i]) / fps for i in range(HB10s_n - 1)]

                    HB10s_interval_mean = np.mean(intervals_s)
                    HB10s_interval_cv = np.std(
                        intervals_s) / HB10s_interval_mean if HB10s_interval_mean > 0 else float(
                        'nan')

                    alternations_s = sum(dirs10s[i] != dirs10s[i + 1] for i in range(HB10s_n - 1))
                    HB10s_alt = alternations_s / (HB10s_n - 1)

                    amps_s = [abs(deflection[f]) for f in frames10s]
                    HB10s_amp_mean = np.mean(amps_s)
                    HB10s_amp_max = np.max(amps_s)

                # CURVATURE
                k_swimming = k[swim_start:swim_end]
                mean_curvature_swimming = round(np.mean(k_swimming), decimals_required)
                max_curvature_swimming = round(max(k_swimming), decimals_required)
                frame_max_curvature_swimming = swim_start + np.argmax(k_swimming) + 1

                idx_local = int(np.argmax(k_swimming))  # 0..(swim_end-swim_start-1)
                idx_global = swim_start + idx_local  # 全局帧
                segment_max_angle_when_max_curvature_swimming = int(np.argmax(angles_between_segments[idx_global]))

                # DEFLECTION
                deflection_swimming = deflection[swim_start:swim_end]
                ### >>> 修改开始：swimming max deflection 与全局一致（abs），frame 记全局帧号
                max_deflection_swimming = round(np.max(np.abs(deflection_swimming)), decimals_required)
                frame_max_deflection_swimming = swim_start + int(np.argmax(np.abs(deflection_swimming))) + 1
                mean_deflection_swimming = round(np.mean(deflection_swimming), decimals_required)
                ### <<< 修改结束

                Deflection_in_peaks_swimming = []
                # Deflection_in_peaks_right = []
                # Deflection_in_peaks_left = []
                Curvature_in_peaks_swimming = []
                for element in peaks_swimming_right:
                    Deflection_in_peaks_swimming.append(deflection[element])
                    # Deflection_in_peaks_right.append(deflection[element])
                    Curvature_in_peaks_swimming.append(k[element])
                for element in peaks_swimming_left:
                    Deflection_in_peaks_swimming.append(deflection[element])
                    # Deflection_in_peaks_left.append(deflection[element])
                    Curvature_in_peaks_swimming.append(k[element])
                mean_deflection_swimming_peaks = round(
                    np.mean([abs(deflection_element) for deflection_element in Deflection_in_peaks_swimming]),
                    decimals_required)
                mean_curvature_swimming_peaks = round(
                    np.mean([abs(curvature_element) for curvature_element in Curvature_in_peaks_swimming]),
                    decimals_required)

                # BIAS TO TURN RIGHT OR LEFT
                # bias_swimming_right = 0
                # bias_swimming_left = 0
                # for element in Deflection_in_peaks_swimming:
                #    if element > 0:
                #        bias_swimming_right = bias_right +1
                #    if element < 0:
                #        bias_swimming_left = bias_left + 1
                # Bias_RL_swimming = round(float(bias_swimming_right)/float(bias_swimming_left), decimals_required)

                ### >>> 修改开始：swimming bias修复 + 不除0 + Laterality_Index_swimming
                bias_swimming_right = np.sum(np.array(Deflection_in_peaks_swimming) > 0)
                bias_swimming_left = np.sum(np.array(Deflection_in_peaks_swimming) < 0)

                if bias_swimming_left == 0 and bias_swimming_right > 0:
                    Bias_RL_swimming = np.inf
                elif bias_swimming_right == 0 and bias_swimming_left > 0:
                    Bias_RL_swimming = 0.0
                elif (bias_swimming_right + bias_swimming_left) == 0:
                    Bias_RL_swimming = float('nan')
                else:
                    Bias_RL_swimming = round(float(bias_swimming_right) / float(bias_swimming_left),
                                             decimals_required)

                if (bias_swimming_right + bias_swimming_left) == 0:
                    Laterality_Index_swimming = float('nan')
                else:
                    Laterality_Index_swimming = round((bias_swimming_right - bias_swimming_left) /
                                                      (bias_swimming_right + bias_swimming_left), decimals_required)
            ### <<< 修改结束

            """ To save the arrays in txt format. Modify the filepath of the file with the extension .txt"""
            result_row = (
                Larva_ID, Genotype, Swim_ID,

                Peak_sequence, Peak_deflection,

                Movement_duration,
                Nr_half_beats,
                Freq_half_beats,
                Alternation,

                max_deflection,
                frame_max_deflection,
                Mean_deflection_peaks,

                Longest_tail_trajectory,
                Cumulative_tail_trajectory,

                max_curvature,
                frame_max_curvature,
                segment_max_angle_when_max_curvature,
                mean_curvature,
                Mean_curvature_peaks,

                Bias_RL,
                Laterality_Index,

                # ===== FIRST10 ALL =====
                HB10_duration,
                HB10_freq,
                HB10_interval_mean,
                HB10_interval_cv,
                HB10_alt,
                HB10_amp_mean,
                HB10_amp_max,

                # ===== FIRST10 SWIMMING =====
                HB10s_duration,
                HB10s_freq,
                HB10s_interval_mean,
                HB10s_interval_cv,
                HB10s_alt,
                HB10s_amp_mean,
                HB10s_amp_max,

                # ===== SWIMMING ORIGINAL VARIABLES =====
                Swimming_duration,
                Nr_half_beats_swimming,
                freq_half_beats_swimming,
                max_deflection_swimming,
                frame_max_deflection_swimming,
                mean_deflection_swimming_peaks,
                max_curvature_swimming,
                frame_max_curvature_swimming,
                segment_max_angle_when_max_curvature_swimming,
                mean_curvature_swimming,
                mean_curvature_swimming_peaks,
                Bias_RL_swimming,
                Laterality_Index_swimming,
                Swimming_valid,
                Fail_reason
            )

            all_results.append(result_row)

        except Exception as e:
            # ✅ 任何意外都不中断批处理：写一行 “失败原因”
            Swimming_valid = 0
            Fail_reason = f'EXCEPTION: {type(e).__name__}: {e}'
            nan = float('nan')

            # 先放 3 个识别字段，再把剩余列补齐为 nan/0
            row = [nan] * len(header)
            row[0] = Larva_ID if 'Larva_ID' in locals() else -1
            row[1] = Genotype if 'Genotype' in locals() else 'NA'
            row[2] = Swim_ID if 'Swim_ID' in locals() else -1
            row[3] = ''
            row[4] = ''
            row[-2] = Swimming_valid
            row[-1] = Fail_reason

            all_results.append(tuple(row))
            log_print(f'⚠️ Skipped {pkl_filename}: {Fail_reason}')
            continue


        # ===========================================================

    # ✅ 保存（每个子文件夹一次）
    savefilepath = os.path.join(output_folder, f"{filename}.csv")
    with open(savefilepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(all_results)

    long_savepath = os.path.join(output_folder3, f"{filename}_HB10.csv")
    with open(long_savepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Larva_ID','Genotype','Swim_ID','HB_index','Interval_s','Inst_freq_Hz'])
        writer.writerows(hb10_long)

    freqmap_savepath = os.path.join(output_folder4, f"{filename}_freqmap_long.csv")
    with open(freqmap_savepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Larva_ID', 'Genotype', 'Swim_ID', 'Interval_index',
            'Peak1_frame', 'Peak2_frame',
            'Start_ms', 'End_ms',
            'Interval_s', 'Inst_freq_Hz'
        ])
        writer.writerows(freqmap_long)

    print(f"Saved: {savefilepath}")
    print(f"Saved: {long_savepath}")
    print(f"Saved: {freqmap_savepath}")

    log_file.close()
# ========= 单文件模式 =========
if RUN_ONLY_THIS_FILE:
    subfolder = folder_path
    folder_name = os.path.basename(subfolder)
    cfg = {**DEFAULT_CFG, **FOLDER_CONFIG.get(folder_name, {})}
    pkl_files = [RUN_ONLY_THIS_FILE]
    fps = float(input(f"Enter FPS for file [{RUN_ONLY_THIS_FILE}]: "))
    process_one_folder(subfolder, pkl_files, fps,cfg)

# ========= 批处理模式（子文件夹）=========
else:
    start_processing = False if START_FROM_FOLDER else True
    for folder_name in sorted(os.listdir(folder_path)):

        subfolder = os.path.join(folder_path, folder_name)

        if not os.path.isdir(subfolder):
            continue  # 只处理第一层子文件夹

        files = os.listdir(subfolder)
        pkl_files = [f for f in files if f.endswith('.pkl')]

        if not pkl_files:
            continue

        # ===== 新增：从指定 folder 开始 =====
        if not start_processing:
            if folder_name == START_FROM_FOLDER:
                start_processing = True
                print(f"Resume from folder: {folder_name}")
            else:
                continue


        folder_name = os.path.basename(subfolder)

        cfg = {**DEFAULT_CFG, **FOLDER_CONFIG.get(folder_name, {})}

        fps = float(input(f"Enter FPS for folder [{folder_name}]: "))


        print("\n====================================")
        print(f"Processing folder: {folder_name}")
        print(f"Using parameters: fps={fps}, prominence={cfg['prominence']}, amp_ratio={cfg['amp_ratio']}")

        filename = f'{folder_name}_{int(fps)}fps'

        process_one_folder(subfolder, pkl_files,fps, cfg)





"""
    #PLOTS:
    plt.suptitle('Curvature, Deflection & Half Beats')
    plt.subplot(3,1,1)
    plt.ylabel("Curvature (deg)")
    plt.plot(k, 'b')
    plt.subplot(3,1,2)
    plt.ylabel("Deflection (deg)")
    plt.plot(deflection, 'g')
    plt.xlabel("Time (in Frames)")
    plt.subplot(3,1,3)
    
    plt.plot(smoothed_tf_x)
    plt.plot(peaks_right, smoothed_tf_x[peaks_right], "x")
    plt.vlines(x=peaks_right, ymin=(smoothed_tf_x[peaks_right] - properties_peaks_right["prominences"]), ymax = smoothed_tf_x[peaks_right], color = "C3")
    plt.hlines(y=properties_peaks_right["width_heights"], xmin=properties_peaks_right["left_ips"], xmax=properties_peaks_right["right_ips"], color = "C1")
    plt.plot(peaks_left, smoothed_tf_x[peaks_left], "x")
    plt.vlines(x=peaks_left, ymin=smoothed_tf_x[peaks_left], ymax = smoothed_tf_x[peaks_left] + properties_peaks_left["prominences"], color = "C3")
    plt.hlines(y=(-properties_peaks_left["width_heights"]), xmin=properties_peaks_left["left_ips"], xmax=properties_peaks_left["right_ips"], color = "C2")
    plt.hlines(xmin=properties_peaks_left["left_ips"][0],xmax=properties_peaks_left["right_ips"][-1], y=max(smoothed_tf_x[peaks_right]) )
    plt.hlines(xmin=min(Left_ips_Swimming_right[0], Left_ips_Swimming_left[0]),xmax=max(Right_ips_Swimming_right[-1], Right_ips_Swimming_left[-1]), y=max(smoothed_tf_x[peaks_right])-5, color = "C8" )
    plt.ylabel("Tailtip in x axis")
    plt.xlabel("Time (in Frames)")
    plt.show()
    
    
    
    print "#", Larva_ID
    print Genotype
    print "Movement duration (in sec):", Movement_duration
    print "Nr of half beats:",Nr_half_beats
    print "Freq. half beating (in Hz):", Freq_half_beats
    print "Max. deflection (in degrees):", max_deflection
    print "Mean deflection during halft beats (in degrees):", Mean_deflection_peaks
    print "Longest tail trajectory among half beats (in pixels):", Longest_tail_trajectory
    #print "Mean tail trajectory (in pixels):", Mean_tail_trajectory
    print "Cumulative tail trajectory (in pixels):", Cumulative_tail_trajectory
    print "Max. curvature", max_curvature
    print "Tail segment with the max. angle when the max. curvature happen", segment_max_angle_when_max_curvature
    print "Mean curvature", mean_curvature
    print "Mean curvature during beats:", Mean_curvature_peaks
    print "Turning bias to R/L:", Bias_RL
    print ''
    print "Swimming duration (in sec):", Swimming_duration
    print "Nr of half beats:", Nr_half_beats_swimming
    print "Freq. half beating during swimming (in Hz):", freq_half_beats_swimming
    print "Max. deflection (in degrees) during swimming:", max_deflection_swimming
    print "Mean deflection during swimming beats (in degrees):", mean_deflection_swimming_peaks
    print "Max. curvature during swimming:", max_curvature_swimming
    print "Mean curvature during swimming:", mean_curvature_swimming
    print "Mean curvature during swimming beats:", mean_curvature_swimming_peaks
    print "Turning bias to R/L during swimming:", Bias_RL_swimming
    """


"""
====================  SUMMARY NOTE (WHAT TO CHANGE & WHAT IT AFFECTS)  ====================

This script extracts tail kinematics from joetailfit .pkl (tail coordinates) and exports one row per file.

Key places to adjust and what each change means:

-------------------------------------------------------------------------------------------
[1] INPUT / OUTPUT
-------------------------------------------------------------------------------------------
- folder_path / output_folder:
  Change where input .pkl are read from and where output .csv are saved.
  Make sure output_folder exists (recommended: os.makedirs(output_folder, exist_ok=True)).

- filename:
  Output file name prefix (does not affect analysis).

-------------------------------------------------------------------------------------------
[2] SMOOTHING / CROPPING (motion trace quality & edge effects)
-------------------------------------------------------------------------------------------
- Box1DKernel(3)  (or parameter "frame_for_smoothing"):
  Controls moving-average smoothing width.
    Larger kernel => smoother trace, fewer spurious peaks, but may blur fast movements.
    Smaller kernel => preserves fast movement but more noise / more false peaks.

- x_smoothed_cropped = x_smoothed[10:-10] (and y):
  Removes edge artifacts introduced by convolution.
    Increasing 10 => safer edges but shorter usable time window.
    Decreasing 10 => longer window but more edge artifacts.
  IMPORTANT: Cropping changes the frame index reference (all reported frame numbers are in the cropped timeline unless you
  explicitly add the offset back).

-------------------------------------------------------------------------------------------
[3] DEFLECTION ANGLE DEFINITION (geometry reference)
-------------------------------------------------------------------------------------------
- dx / dy used for deflection:
  Current definition uses tail base vs tail tip (or a chosen "base point"):

    dx = tail_point_x - base_point_x
    dy = tail_point_y - base_point_y
    deflection = degrees(arctan2(dx, dy))   # robust vs dy=0

  Where to change the "base point":
    Example:
      dx = [frame[-1,0] - frame[0,0] for frame in smoothed_tf]
      dy = [frame[-1,1] - frame[0,1] for frame in smoothed_tf]

    If you change base from frame[0] to frame[2], e.g.
      dx = [frame[-1,0] - frame[2,0] for frame in smoothed_tf]
      dy = [frame[-1,1] - frame[2,1] for frame in smoothed_tf]
    then you measure deflection relative to a point further down the tail (less sensitive to head/base jitter but changes angle magnitude).

  Changing which tail point you use as the "tip":
    frame[-1] is the last tracked point (tail tip).
    Using frame[-2], frame[-3] etc. measures a more proximal point (reduces noise but changes amplitude).

-------------------------------------------------------------------------------------------
[4] CURVATURE (k) DEFINITION (how bending is quantified)
-------------------------------------------------------------------------------------------
- tail_segment() and angles_between_segments:
  Curvature k is computed as the sum of angles between consecutive tail segments.
  If you change:
    - how many points are included
    - how segment angles are computed
    - whether you use abs() or signed angles
  then max/mean curvature values will change.

-------------------------------------------------------------------------------------------
[5] PEAK DETECTION (how half-beats are found)
-------------------------------------------------------------------------------------------
Peak detection uses scipy.signal.find_peaks on deflection trace.

Parameters and effects:
- distance=10:
  Minimum separation between peaks (in frames).
    Increase => fewer peaks (prevents double counting), but may miss high-frequency beats.
    Decrease => more peaks, higher chance of counting noise / rebound peaks.

- prominence=8:
  Strength threshold (peak must stand out from local baseline).
    Increase => keeps only strong beats, removes small beats/noise, but may miss weak swimming.
    Decrease => detects weaker beats but increases false positives.

- width=1 and rel_height=0.5:
  Controls peak width estimation; affects left_ips/right_ips (movement start/end estimation).
    Increasing width constraints can reject narrow noise peaks but may miss sharp real peaks.

-------------------------------------------------------------------------------------------
[6] ACTIVE HALF-BEAT FILTER (remove rebound-like peaks)
-------------------------------------------------------------------------------------------
After raw peaks (right/left) are detected, they are merged and filtered into active peaks:
- require_alternation=True:
  Enforces R-L-R-L alternation.
    True  => suppresses repeated same-side peaks (often rebound/noise).
    False => allows same-side sequences (may count rebounds as half-beats).

- amp_ratio=0.4:
  Keeps a new peak only if its amplitude is at least (amp_ratio * amplitude of previous kept peak).
    Increase (e.g. 0.6) => stricter, removes small peaks after a big one (more conservative).
    Decrease (e.g. 0.2) => lenient, keeps more peaks (risk of including rebound peaks).

Definition used in this script:
- Number of active peaks == Number of half-beats (Nr_half_beats)
- Full beats (left+right) are approximately Nr_half_beats / 2

-------------------------------------------------------------------------------------------
[7] MOVEMENT START/END (movement duration)
-------------------------------------------------------------------------------------------
Movement start/end are derived from peak widths (left_ips/right_ips) of ACTIVE peaks.
This ensures movement duration is consistent with the half-beats actually counted.

If you change peak detection or active filtering, movement start/end and Movement_duration will change.

-------------------------------------------------------------------------------------------
[8] SWIMMING SELECTION (define swimming subset)
-------------------------------------------------------------------------------------------
Swimming selection uses an amplitude threshold on ACTIVE peaks:
- abs(deflection[p]) < 50  (threshold in degrees)
  Change this threshold to define what counts as "swimming" vs "burst/escape".
    Smaller threshold (e.g. 40) => more strict swimming, fewer swimming beats.
    Larger threshold (e.g. 60) => includes stronger beats as swimming.

Swimming start/end are computed from ACTIVE swimming ips.
All swimming metrics (HB10s, swimming frequency, bias, etc.) should use the same ACTIVE swimming peak set for consistency.

-------------------------------------------------------------------------------------------
[9] FIRST-10 METRICS (HB10, HB10s)
-------------------------------------------------------------------------------------------
HB10_*:
  Based on first 10 ACTIVE peaks (all movement).
HB10s_*:
  Based on first 10 ACTIVE peaks that also pass swimming threshold.

If peak detection / active filter changes, which peaks count as the "first 10" will change.

-------------------------------------------------------------------------------------------
[10] TURNING BIAS / LATERALITY
-------------------------------------------------------------------------------------------
- Bias_RL:
  Count of positive vs negative deflection peaks.
- Laterality_Index:
  (R - L) / (R + L)
If you change the sign convention (e.g. swap dx direction, change coordinate system), bias and LI sign will flip.

-------------------------------------------------------------------------------------------
[11] FRAME INDICES (IMPORTANT)
-------------------------------------------------------------------------------------------
Frame numbers may be:
- in cropped timeline (because of [10:-10] cropping)
- or converted back to global indices if you add offsets.
If you need original-video frame numbers, add back crop offset or store the offset explicitly.

===========================================================================================
"""