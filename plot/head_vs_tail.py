# =========================================================
# SAME GROUP: plot_head_tail_within_group
# =========================================================
def head_tail(folder, ycol, Ylabel, df_swim_clean, df_fish_clean):
    os.makedirs(folder, exist_ok=True)  # ⭐ 自动创建文件夹

    groups = ["Control", "wt1_MTZ", "d3_MTZ"]
    stims = ["Head", "Tail", "OMR"]
    pvalue_table = []

    for g in groups:

        vals = []
        n_fish = []
        n_swim = []

        df_fish = df_fish_clean

        for stim in stims:

            fish_vals = df_fish[(df_fish["Stimulus"] == stim) & (df_fish["Group"] == g)][ycol]
            swim_vals = df_swim_clean[(df_swim_clean["Stimulus"] == stim) & (df_swim_clean["Group"] == g)][ycol]

            vals.append(fish_vals)
            n_fish.append(len(fish_vals))
            n_swim.append(len(swim_vals))

        plt.figure(figsize=(6, 4))
        ax = plt.gca()

        # ✅ 标签：n=鱼数(泳次)
        labels = [f"{stim}\n(n={nf} ({ns}))" for stim, nf, ns in zip(stims, n_fish, n_swim)]

        # 画箱线图：不显示 outlier 点
        colors = ["#E5E5E5", "#E18E96", "#88D498"]

        bp = ax.boxplot(vals, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # 计算每组 mean
        means = [v.mean() for v in vals]

        # 在每个箱体上方标 mean
        for i, m in enumerate(means):
            ax.text(i+1, m, f"{m:.2f}",
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    color='black')

        for i, (v, color) in enumerate(zip(vals, colors)):
            x = np.random.normal(i+1, 0.04, size=len(v))
            darker = darken(color, 0.6)   # 0.6 越小越深

            ax.scatter(x, v,
                       color=darker,
                       alpha=0.6,
                       s=35,
                       zorder=3)

        ax.set_ylabel(f"{Ylabel}")

        # 去除外方框（右边和上边）
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 只保留下边和左边
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.set_xlabel("")
        ax.text(0.5, -0.18, f"{g} group", transform=ax.transAxes,
                ha="center", va="top", fontsize=12)

        # -------------------
        # 1️⃣ ANOVA
        # -------------------
        if all(len(v) >= 2 for v in vals):
            F_stat, p_anova = f_oneway(*vals)
        else:
            F_stat, p_anova = np.nan, np.nan

        # -------------------
        # 2️⃣ pairwise t-tests
        # -------------------
        pairs = [(0, 1), (0, 2), (1, 2)]
        raw_pvals = []
        pair_info = []

        for i, j in pairs:
            if len(vals[i]) >= 2 and len(vals[j]) >= 2:
                t_stat, p_raw = ttest_ind(vals[i], vals[j], equal_var=False)

                raw_pvals.append(p_raw)
                pair_info.append((i, j, t_stat, p_raw))
            else:
                raw_pvals.append(np.nan)
                pair_info.append((i, j, np.nan, np.nan))

        # -------------------
        # 3️⃣ Holm correction
        # -------------------
        valid_p = [p for p in raw_pvals if not np.isnan(p)]
        adjusted_p = holm_correction(valid_p) if len(valid_p) > 0 else []

        # -------------------
        # 4) 画显著性（用 Holm 后 p）
        # -------------------
        k = 0
        finite_vals = [v for v in vals if len(v) > 0 and np.isfinite(v).any()]
        if len(finite_vals) > 0:
            vmax = max([np.nanmax(v) for v in finite_vals])
            vmin = min([np.nanmin(v) for v in finite_vals])
            y0 = vmax * 1.05
            step = (vmax - vmin) * 0.12 if vmax > vmin else 0.1
        else:
            y0, step = 1.0, 0.1

        adj_index = 0

        for idx, (i, j, t_stat, p_raw) in enumerate(pair_info):

            if not np.isnan(p_raw):
                p_holm = adjusted_p[adj_index]
                adj_index += 1
            else:
                p_holm = np.nan

            significant = p_holm < 0.05 if not np.isnan(p_holm) else False

            # ⭐ 存入表格
            pvalue_table.append({
                "Variable": ycol,
                "Group": g,
                "Test": "ANOVA",
                "F_value": F_stat,
                "p_ANOVA": p_anova,
                "Stimulus1": stims[i],
                "Stimulus2": stims[j],
                "n1": len(vals[i]),
                "n2": len(vals[j]),
                "t_value": t_stat,
                "p_raw": p_raw,
                "p_holm": p_holm,
                "Significant": significant
            })

            # ⭐ 图上只画 Holm 后显著
            if significant:
                add_sig(ax, i+1, j+1, y0 + k*step, star(p_holm))
                k += 1

        plt.tight_layout()

        safe_ycol = ycol.replace("/", "_").replace(" ", "_")
        safe_group = g.replace("/", "_").replace(" ", "_")

        final_path = os.path.join(folder, f"{safe_ycol}_{safe_group}.png")
        plt.savefig(final_path, dpi=300, bbox_inches="tight")

        plt.show()
        print("Saved to:", final_path)

    # 转成 DataFrame
    df_pvalues = pd.DataFrame(pvalue_table)

    print(df_pvalues)

    return df_pvalues