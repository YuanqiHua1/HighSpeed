# 频率分布作图

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def freq_dist_all(folder, ycol,savename,df_fish_clean):
    os.makedirs(folder, exist_ok=True)

    groups = ["Control","wt1_MTZ","d3_MTZ"]
    # colors = ["#999999", "#E18E96", "#88D498"]
    ycol = ycol
    colors = ["#7A7A7A", "#C65A5A", "#4E9F6D"]
    colors_edge = {"Control":"#212121", "wt1_MTZ":"#8B0000", "d3_MTZ":"#004D40"}  # 深轮廓


    for stim in ["Head","Tail", "OMR"]:

        plt.figure(figsize=(6,4))

         # ⭐ 1️⃣ 先拿到当前 stimulus 所有数据
        all_vals = df_fish_clean[
            df_fish_clean["Stimulus"] == stim
        ][ycol].dropna()

         # ⭐ 2️⃣ 统一 bins（关键修改）
        bins = np.linspace(all_vals.min(), all_vals.max(), 12)

        for g, color in zip(groups, colors):
            vals = df_fish_clean[
                (df_fish_clean["Stimulus"] == stim) &
                (df_fish_clean["Group"] == g)
            ][ycol]

            plt.hist(vals,
                     bins=bins,
                     density=True,
                     weights=np.ones(len(vals)) / len(vals),
                     alpha=0.3,
                     label=g,
                     color=color,
                     edgecolor=colors_edge[g],
                     linewidth=1.5)

        plt.title(f"{stim} stimulus")
        plt.xlabel(f"Swim Frequency (Hz)")
        plt.ylabel("Probability")
        plt.legend()

            # ⭐ 生成完整文件路径
        filename = f"{savename}_{stim}.png"
        full_path = os.path.join(folder, filename)
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        plt.show()

    return


def freq_dist_all2(folder, ycol,savename,df_fish_clean):
    os.makedirs(folder, exist_ok=True)

    groups = ["Control","wt1_MTZ","d3_MTZ"]
    # colors = ["#999999", "#E18E96", "#88D498"]
    ycol = ycol
    color = {
        "Head": "#4C72B0",
        "Tail": "#DD8452",
        "OMR": "#55A868"
    }

    colors_edge = {
        "Head": "#1f3b73",
        "Tail": "#8c3b00",
        "OMR": "#1b5e20"
    }


    for g in groups:   # ⭐ 改这里：先固定 group

        plt.figure(figsize=(6,4))

        # ⭐ 1️⃣ 当前 group 的所有数据
        all_vals = df_fish_clean[
            df_fish_clean["Group"] == g
        ][ycol].dropna()

        bins = np.linspace(all_vals.min(), all_vals.max(), 12)

        for stim in ["Tail","Head","OMR"]:   # ⭐ 改这里：比较 stimulus

            vals = df_fish_clean[
                (df_fish_clean["Group"] == g) &
                (df_fish_clean["Stimulus"] == stim)
            ][ycol]

            plt.hist(vals,
                     bins=bins,
                     density=True,
                     weights=np.ones(len(vals)) / len(vals),
                     alpha=0.3,
                     label=stim,
                     color=color[stim],
                     edgecolor=colors_edge[stim],
                     linewidth=1.5)

        plt.title(f"{g}")
        plt.xlabel(f"Swim Frequency (Hz)")
        plt.ylabel("Probability")
        plt.legend()

            # ⭐ 生成完整文件路径
        filename = f"{savename}_{g}.png"
        full_path = os.path.join(folder, filename)
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        plt.show()

    return

# 单个频率分布作图
def freq_dist_single(folder, ycol,savename,df_fish_clean):

    groups = ["Control","wt1_MTZ","d3_MTZ"]
    ycol = ycol

    for stim in ["Head","Tail", "OMR"]:
        for g in groups:
            vals = df_fish_clean[
                (df_fish_clean["Stimulus"] == stim) &
                (df_fish_clean["Group"] == g)
            ][ycol].dropna().to_numpy()

            plt.figure(figsize=(5,4))
            bins = np.arange(0, 61, 2)   # 每 2 Hz 一个 bin
            plt.hist(vals, bins=bins, density=True, alpha=0.4)
            plt.xlim(0,60)

            if len(vals) >= 3 and vals.min() != vals.max():
                xs = np.linspace(vals.min(), vals.max(), 200)
                kde = gaussian_kde(vals)
                ys = kde(xs)
                plt.plot(xs, ys)

            plt.title(f"{stim} - {g} (n={len(vals)})")
            plt.xticks([0,10,20,30,40,50,60])
            plt.xlabel("Half-beat_frequency (Hz)")
            plt.ylabel("Density")
            plt.tight_layout()

            filename = f"{savename}_{stim}_{g}.png"
            full_path = os.path.join(folder, filename)
            plt.savefig(full_path, dpi=300, bbox_inches="tight")

            plt.show()
    return