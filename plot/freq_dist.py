# 频率分布作图
def Freq_dist(folder, ycol,savename,df_fish_clean):
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