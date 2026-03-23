# 清洗
import pandas as pd

def rm_outliers(s):
    s = s.dropna()
    if len(s) < 4:
        return s
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    return s[(s >= lo) & (s <= hi)]

def make_clean_dfs(df, ycol):
    groups = ["Control", "wt1_MTZ", "d3_MTZ"]
    out_swim = []

    for stim in ["Head", "Tail", "OMR"]:
        data_swim = df[df["Stimulus"] == stim]

         # ——A) swim 层：按 (Stimulus, Group) 分别 IQR，删“行”
        for g in groups:
            tmp = data_swim[data_swim["Group"] == g].copy() # e.g. ， tmp 就是Head/control组的表格

            kept_vals = rm_outliers(tmp[ycol])      # 返回被保留的 Series（带 index）
            tmp = tmp.loc[kept_vals.index]          # ✅ 按标签（index）选行。用 index 保留整行

            out_swim.append(tmp) # 最后，out_swim = [Head-Control 的 tmp, Head-wt1 的 tmp, Head-d3 的 tmp,。。。] 注意：这还是 list，不是表格。

    df_swim_clean  = pd.concat(out_swim, ignore_index=True) # 把那 6 个小表“上下拼起来”，变成一个大的 新的 DataFrame。

    # ——B) fish 层：用 clean swim 计算每鱼均值
    df_fish_clean0 = (
        df_swim_clean.groupby(["UniqueID","Group","Stimulus"])[ycol]
        .mean()
        .reset_index()
    )

    # ——C) fish 层再 IQR：按 (Stimulus, Group) 删“鱼”
    out_fish = []
    for stim in ["Head", "Tail", "OMR"]:
        tmp_stim = df_fish_clean0[df_fish_clean0["Stimulus"] == stim]
        for g in groups:
            tmp = tmp_stim[tmp_stim["Group"] == g].copy()
            kept = rm_outliers(tmp[ycol])
            tmp = tmp.loc[kept.index]
            out_fish.append(tmp)

    df_fish_clean = pd.concat(out_fish, ignore_index=True)

    return df_swim_clean, df_fish_clean0, df_fish_clean
