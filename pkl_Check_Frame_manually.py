

import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import shutil



# =============================
# 0) 配置：你只需要改这里
# =============================
INPUT_DIR = Path(r"U:\YuanqiHua\High speed\260311 dmrt3 MTZ\pkl_output")  # 你的pkl文件夹
OUT_DIR   = Path(r"U:\YuanqiHua\High speed\Manual_fix_pkl_out\260311 dmrt3 MTZ")  # ✅ 强烈建议本地输出，避免网络盘写权限问题
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ✅ logs 统一放这里
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 从任意指定文件开始（两种方式选一种）
START_FROM_FILENAME = "d3-MTZ-HS-6_13_binary.pkl"     # 例如 "wt1-MTZ-1_4_binaryxxxx.pkl"；留空表示不用它
START_FROM_INDEX_1BASED = 1  # 例如 12 表示从第12个开始（按排序）
# 如果你想“自动从上次进度接着跑”，把下面设为 True（优先于上面两个）
RESUME_FROM_PROGRESS = True

# ---- 标注阈值（review用）----
SHORT_RATIO = 0.60
FOLD90_DEG  = 60.0
TURN150_DEG = 150.0

# ---- 候选修复绿点怎么生成（越保守越好）----
CANDIDATE_BAD_MODE = "all_flags"   # "short_only" / "short_and_selfx" / "all_flags"

# =============================
# 1) 兼容读取 pickle（处理 numpy._core）
# =============================
class NumpyCoreCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

def load_pkl_compat(pkl_path: Path):
    # 用 open(...,'rb') 更直观，也避免某些网络盘对 read_bytes 的怪限制
    with open(pkl_path, "rb") as f:
        tf = NumpyCoreCompatUnpickler(f).load()
    tf = np.asarray(tf, dtype=np.float32)
    if tf.ndim != 3 or tf.shape[2] != 2:
        raise ValueError(f"Unexpected shape {tf.shape} in {pkl_path.name} (expect frames x points x 2)")
    return tf

# =============================
# 2) 指标：弧长 / 最大转角 / 自交叉
# =============================
def tail_arclength(tf):
    d = np.diff(tf, axis=1)
    seg = np.sqrt((d * d).sum(axis=2))
    return seg.sum(axis=1)

def max_turn_angle_per_frame(tf):
    F, P, _ = tf.shape
    max_turn = np.zeros(F, dtype=np.float32)
    eps = 1e-9
    for f in range(F):
        pts = tf[f]
        v1 = pts[1:-1] - pts[:-2]
        v2 = pts[2:]   - pts[1:-1]
        n1 = np.sqrt((v1*v1).sum(axis=1)) + eps
        n2 = np.sqrt((v2*v2).sum(axis=1)) + eps
        cos = (v1*v2).sum(axis=1) / (n1*n2)
        cos = np.clip(cos, -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        max_turn[f] = float(np.max(ang)) if ang.size else 0.0
    return max_turn

def _seg_intersect(a, b, c, d):
    def orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    def onseg(p, q, r):
        return (min(p[0],r[0]) <= q[0] <= max(p[0],r[0]) and
                min(p[1],r[1]) <= q[1] <= max(p[1],r[1]))
    o1 = orient(a,b,c); o2 = orient(a,b,d)
    o3 = orient(c,d,a); o4 = orient(c,d,b)
    if (o1*o2 < 0) and (o3*o4 < 0):
        return True
    if o1 == 0 and onseg(a,c,b): return True
    if o2 == 0 and onseg(a,d,b): return True
    if o3 == 0 and onseg(c,a,d): return True
    if o4 == 0 and onseg(c,b,d): return True
    return False

def self_intersection_mask(tf):
    F, P, _ = tf.shape
    out = np.zeros(F, dtype=bool)
    for f in range(F):
        pts = tf[f]
        for i in range(P-1):
            a = pts[i]; b = pts[i+1]
            for j in range(i+2, P-1):
                if j == i+1:
                    continue
                c = pts[j]; d = pts[j+1]
                if _seg_intersect(a,b,c,d):
                    out[f] = True
                    break
            if out[f]:
                break
    return out

def build_flags(tf):
    arc = tail_arclength(tf)
    med = float(np.median(arc[np.isfinite(arc)]))
    short = arc < (SHORT_RATIO * med)

    max_turn = max_turn_angle_per_frame(tf)
    fold90  = max_turn >= FOLD90_DEG
    turn150 = max_turn >= TURN150_DEG

    selfx = self_intersection_mask(tf)

    flags_mask = short | fold90 | turn150 | selfx
    flagged = np.where(flags_mask)[0].astype(int)

    reasons = {}
    for i in flagged:
        r = []
        if short[i]: r.append("short")
        if turn150[i]: r.append("turn150")
        elif fold90[i]: r.append("fold90")
        if selfx[i]: r.append("self_intersect")
        reasons[int(i)] = r

    return {
        "arc": arc,
        "arc_median": med,
        "max_turn": max_turn,
        "short": short,
        "fold90": fold90,
        "turn150": turn150,
        "selfx": selfx,
        "flagged": flagged,
        "reasons": reasons,
    }

# =============================
# 3) 候选修复：用前后“参考帧”插值（不删帧）
# =============================
def build_candidate_fixed(tf, bad_mask):
    F = tf.shape[0]
    good = ~bad_mask

    prev_good = np.full(F, -1, dtype=int)
    last = -1
    for i in range(F):
        if good[i]:
            last = i
        prev_good[i] = last

    next_good = np.full(F, -1, dtype=int)
    last = -1
    for i in range(F-1, -1, -1):
        if good[i]:
            last = i
        next_good[i] = last

    tf_cand = tf.copy()
    bad_idx = np.where(bad_mask)[0]
    for i in bad_idx:
        a = prev_good[i]
        b = next_good[i]
        if a >= 0 and b >= 0 and a != b:
            t = (i - a) / (b - a)
            tf_cand[i] = (1 - t) * tf[a] + t * tf[b]
        elif a >= 0:
            tf_cand[i] = tf[a]
        elif b >= 0:
            tf_cand[i] = tf[b]
    return tf_cand

# =============================
# 4) 保存（每个文件一套输出）
# =============================
def save_outputs(pkl_path: Path, tf_out, decisions, flagged_idx, info):
    out_pkl = OUT_DIR / (pkl_path.stem + "_manual_fixed.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(tf_out, f)

    out_csv = LOG_DIR / (pkl_path.stem + "_manual_fix_log.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "reasons", "decision", "arc", "max_turn"])
        for fr in flagged_idx:
            rs = ",".join(info["reasons"].get(int(fr), []))
            dec = decisions.get(int(fr), "")
            w.writerow([int(fr), rs, dec, float(info["arc"][fr]), float(info["max_turn"][fr])])

    return out_pkl, out_csv


def write_progress(last_done_filename: str):
    (OUT_DIR / "progress.txt").write_text(last_done_filename, encoding="utf-8")

def read_progress():
    p = OUT_DIR / "progress.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return ""

# =============================
# 5) 单文件交互 review（返回 "done" 或 "quit"）
# =============================

def run_manual_review_fix_one(pkl_path: Path):
    state = {"result": ("done", False)}  # 默认：完成且未修改

    tf = load_pkl_compat(pkl_path)
    info = build_flags(tf)
    flagged = info["flagged"]

    # 没有异常帧：也要写一个“空日志”，并直接保存原样为 manual_fixed，方便后续统一处理
    if flagged.size == 0:
        tf_out = tf.copy()
        decisions = {}
        out_pkl, out_csv = save_outputs(pkl_path, tf_out, decisions, flagged, info)
        return ("done", False)

    if CANDIDATE_BAD_MODE == "short_only":
        bad_for_candidate = info["short"]
    elif CANDIDATE_BAD_MODE == "short_and_selfx":
        bad_for_candidate = info["short"] | info["selfx"]
    else:
        bad_for_candidate = info["short"] | info["fold90"] | info["turn150"] | info["selfx"]

    tf_cand = build_candidate_fixed(tf, bad_for_candidate)

    tf_out = tf.copy()
    decisions = {}  # frame -> "Y"/"N"

    idx_pos = 0

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.18)
    fig.text(
        0.5, 0.06,
        "Keys: Y=accept  N=keep  ←/→=prev/next  S=save  F=finish file  Q/Esc=quit+save",
        ha="center", va="center"
    )

    def draw():
        ax.clear()
        fr = int(flagged[idx_pos])

        pts_o = tf[fr]
        pts_f = tf_cand[fr]

        # 判断 orig 和 fixed 是否几乎相同
        same = np.allclose(pts_o, pts_f, atol=1e-2)

        if same:
            # 只画一层：空心红圈
            ax.scatter(
                pts_o[:, 0], pts_o[:, 1],
                s=60,
                facecolors="none",
                edgecolors="red",
                linewidths=1.8,
                label="orig (same as fixed)"
            )
        else:
            # orig：空心红圈（不会被盖住）
            ax.scatter(
                pts_o[:, 0], pts_o[:, 1],
                s=60,
                facecolors="none",
                edgecolors="red",
                linewidths=1.8,
                label="orig"
            )

            # fixed：实心绿色
            ax.scatter(
                pts_f[:, 0], pts_f[:, 1],
                s=25,
                c="lime",
                alpha=0.9,
                label="candidate fixed"
            )

        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        rs = ",".join(info["reasons"][fr])
        dec = decisions.get(fr, "")
        ax.set_title(
            f"{pkl_path.name}\n"
            f"flagged {idx_pos+1}/{len(flagged)} | frame {fr} | {rs} | decision={dec}\n"
            f"arc={info['arc'][fr]:.2f} (med={info['arc_median']:.2f})   max_turn={info['max_turn'][fr]:.1f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=9)
        fig.canvas.draw_idle()

    def apply_decision(fr, accept: bool):
        if accept:
            tf_out[fr] = tf_cand[fr]
            decisions[fr] = "Y"
        else:
            tf_out[fr] = tf[fr]
            decisions[fr] = "N"

    state = {"result": None}  # "done"/"quit"

    def finish_and_close(result: str):
        save_outputs(pkl_path, tf_out, decisions, flagged, info)
        changed = ("Y" in decisions.values())
        state["result"] = (result, changed)
        plt.close(fig)

    def on_key(event):
        nonlocal idx_pos
        fr = int(flagged[idx_pos])

        if event.key in ["y", "Y"]:
            apply_decision(fr, True)
            if idx_pos < len(flagged) - 1:
                idx_pos += 1
                draw()
            else:
                finish_and_close("done")

        elif event.key in ["n", "N"]:
            apply_decision(fr, False)
            if idx_pos < len(flagged) - 1:
                idx_pos += 1
                draw()
            else:
                finish_and_close("done")

        elif event.key == "right":
            if idx_pos < len(flagged) - 1:
                idx_pos += 1
            draw()

        elif event.key == "left":
            if idx_pos > 0:
                idx_pos -= 1
            draw()

        elif event.key in ["s", "S"]:
            save_outputs(pkl_path, tf_out, decisions, flagged, info)

        elif event.key in ["f", "F"]:
            # 结束当前文件（即使还有未判定帧，也保存当前已判定进度）
            finish_and_close("done")

        elif event.key in ["q", "Q", "escape"]:
            # 退出批处理：保存当前文件进度
            finish_and_close("quit")

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()

    # 窗口关闭后返回
    return state["result"] or ("done", False)


# =============================
# 6) 批处理：打开一个→review→保存→下一个；中途退出可恢复
# =============================
def pick_start_index(files):
    # 1) progress 自动恢复（优先）
    if RESUME_FROM_PROGRESS:
        last = read_progress()
        if last:
            # 从“上次完成文件”的下一个开始
            for i, f in enumerate(files):
                if f.name == last:
                    return min(i + 1, len(files) - 1) if len(files) > 0 else 0

    # 2) 指定文件名
    if START_FROM_FILENAME:
        for i, f in enumerate(files):
            if f.name == START_FROM_FILENAME:
                return i

    # 3) 指定序号（1-based）
    return max(0, int(START_FROM_INDEX_1BASED) - 1)

def main():
    files = sorted(INPUT_DIR.glob("*.pkl"))
    if not files:
        print("No .pkl found in:", INPUT_DIR)
        return

    start_i = pick_start_index(files)
    print(f"Total pkl: {len(files)} | Start from index {start_i+1}: {files[start_i].name}")

    for i in range(start_i, len(files)):
        pkl = files[i]
        print(f"\n[{i+1}/{len(files)}] Reviewing: {pkl.name}")

        result, changed = run_manual_review_fix_one(pkl)

        # 只有文件完成(done)才处理移动逻辑
        if result == "done":
            if changed:
                print("Saved FIXED result to OUT_DIR for:", pkl.name)
            else:
                print("Saved UNCHANGED copy to OUT_DIR for:", pkl.name)

            # ✅ 只记录进度，不动 INPUT_DIR
            write_progress(pkl.name)


        elif result == "quit":
            # 退出：不移动当前文件、不写 progress（方便你继续从它开始）
            print("Quit requested. Current file kept for resume.")
            break

    print("Batch done.")

if __name__ == "__main__":
    main()


'''
OUT_DIR（一个文件夹里同时含“没修”和“已修”）

xxx_manual_fixed.pkl（所有处理过的都会有：没修的内容等同原始，修过的是真修复版）

xxx_manual_fix_log.csv

progress.txt

originals_not_fixed/（被你判定“不需要修复”的原始 pkl 会移到这里）

INPUT_DIR（最后只剩“修过的那些文件的原始 pkl”）

只有 changed=True 的原始文件还留在这里（你要的“证据/备份”）

对于每一个 .pkl 文件：

自动检测异常帧（short / fold90 / turn150 / self_intersect）

弹出窗口逐帧给你看：

🔴 红点 = 原始 tracking 点

🟢 绿点 = 候选修复点（插值得到）

你人工决定：

Y = 用绿点

N = 保留红点

当前文件 review 完成后：

自动保存

自动进入下一个 pkl

中途退出：

已经 review 的文件会保存

当前文件已决定的帧也会保存

下次可以从指定文件继续

1️⃣ 修复后的文件
wt1-MTZ-1_4_binary_manual_fixed.pkl


这个文件：

帧数完全不变

点数完全不变

只有你按 Y 的帧被替换

可以直接拿去跑 Variables_YH.py

2️⃣ 决策记录日志
wt1-MTZ-1_4_binary_manual_fix_log.csv


内容类似：

frame	reasons	decision	arc	max_turn
26	fold90	N	867.22	113.6
120	short	Y	402.11	32.5

你可以：

回溯你当时怎么判断

统计每个文件修了多少帧

写论文时做 QC 说明

3️⃣ 进度文件（用于自动恢复）
progress.txt


里面记录：

wt1-MTZ-1_4_binary.pkl


下次运行时：

自动从下一个文件开始

'''