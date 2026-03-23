import pickle

with open('C:\joetailfit2_Improved\CRISPR_\ReScaled.pkl', 'rb') as f:
    tf = pickle.load(f)
# 2016_08_25_fish2.pkl

print len(tf)


# mylist = []
# for frame in tf:
#     print frame[:, 0].mean()
#     mylist.append()
meanx = [frame[:, 0].mean() for frame in tf]

from matplotlib import pyplot as plt


plt.plot(meanx)
tipmean = [frame[-4:, 0].mean() for frame in tf]
from scipy.ndimage.filters import uniform_filter1d
plt.plot(uniform_filter1d(tipmean, 10), 'r')

plt.show()
# plt.savefig('fig.pdf')

