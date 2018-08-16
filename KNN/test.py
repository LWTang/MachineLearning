import numpy as np
a = np.array([0, 1, 2])
b = np.tile(a, 2) #沿x轴复制2倍
c = np.tile(a, (1, 2)) #沿x轴复制2倍，沿y轴复制1倍
d = np.tile(a, (2, 1, 2)) #沿x轴复制2倍，沿y轴复制1倍，沿z轴复制2倍
print(a)
print(b)
print(c)
print(d)