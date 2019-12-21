'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: GaussianPlot.py
@time: 12/19/2019 10:06 PM
'''

import numpy as np
import matplotlib.pyplot as plt

t = np.arange(-4, 4.01, 0.01)
f = np.arange(-4, 4.01, 0.01)
tl, fl = np.meshgrid(t, f)


def Gaussian(x, y):
    x_ = x/np.sqrt(2)
    y_ = y/np.sqrt(2)
    cr = 1
    delt = 0
    delt = x**2 + y**2
    w = np.exp(-x_ ** 2 - cr * y_ ** 2+ delt)
    return w


plt.figure(figsize=(4, 4))
plt.contour(tl, fl, Gaussian(tl, fl))
plt.show()
