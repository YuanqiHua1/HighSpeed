'''
    在图中添加显著性标注（significance annotation）。

    该函数会在两个 x 位置之间画一条“括号线”，并在上方标注统计显著性（如 *, **, ns）。
'''
from scipy.stats import f_oneway, ttest_ind
import matplotlib.colors as mcolors
import numpy as np

def holm_correction(pvals):
    pvals = np.array(pvals)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)

    for i, idx in enumerate(order):
        adjusted[idx] = min((m - i) * pvals[idx], 1.0)

    # 保证单调性
    adjusted_sorted = adjusted[order]
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted[order] = adjusted_sorted

    return adjusted

def darken(color, amount=0.3):
    """
    amount < 1 → 变深
    """
    c = mcolors.to_rgb(color)
    return tuple([x * amount for x in c])

def star(p):
    if p < 0.0001: return "****"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def add_sig(ax, x1, x2, y, text):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1,color="black")
    ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom")
