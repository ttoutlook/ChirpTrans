'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: map_chirplets.py
@time: 12/17/2019 8:15 PM
'''

import numpy as np
from Optimize import FindBestChirpletRate
from MakeChirplet import MakeChirplet
from MaxCorrChirplet import MaxCorrelation
from numpy import pi


# th main function used to project the signal to gaussian chirplet atom

class MapChirplet:
    def __init__(self, x, M=64, Depth=5, i0=1, radix=2, methodid=1, windowid=0, windowlength = 4):
        self.norm = np.linalg.norm(x).real
        self.N = x.size
        self.x = x / self.norm
        self.M = M  # resolution for newton-raphson refinement (optional, default is 64)
        self.Depth = Depth
        self.i0 = i0  # the first level to rotate the chirplets
        self.radix = radix  # radix of scale
        # default use boxcar window, for strong noisy signal and we recommend
        # hamming, bartlett, flattop, bohman, especially bartharr (the last one)
        self.windowid = windowid
        self.windowlength = windowlength
        self.initial()

    def initial(self):
        """
        estimate the chirplet using MP
        maping the chirplet... initial the structure
        """
        A, alpha, sigma, tc, fc = MaxCorrelation(self.x, self.Depth, self.i0, self.radix).code
        self.Param = [A, alpha, sigma, tc, fc]
        self.bultan2params(alpha, sigma, tc, fc)
        self.refineParams()

    def refineParams(self):
        """
        refine the chirplet with newton-raphson method
        """
        xx = self.window()


    def window(self):
        Z = self.windowlength  # a longer window is useful here (windown length)
        windows = ['boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett', 'flattop', 'parzen', 'bohman',
                  'blackmanharris', 'nuttall', 'barthann']
        t, f, c, d = self.Param[1:]
        rt = int(round(t))
        # box window
        if rt - Z * self.M < 1 and rt + Z * self.M > self.N:
            xx = np.concatenate([np.zeros([Z * self.M - rt + 1]),
                                 self.x,
                                 np.zeros([Z * self.M - self.N + rt])])
        if rt - Z * self.M < 1:
            xx = np.concatenate([np.zeros([Z * self.M - rt + 1]), self.x[0: rt + Z * self.M]])
        elif rt + Z * self.M > self.N:
            xx = np.concatenate([self.x[rt - Z * self.M - 1:], np.zeros([Z * self.M - self.N + rt])])
        else:
            xx = self.x[rt - Z * self.M - 1:rt + Z * self.M]
        xx = xx.flatten()
        return xx

    def bultan2params(self, alpha, sigma, tc, fc):
        """
        refer to bultan and oneill for chirp rate (cr) and duration (dt)
        """
        # time center (assume fs = 1 Hz)
        tc = tc
        # frequency center (rad)
        fc = fc / self.N * 2 * pi
        # chirp rate
        cr = 2 * pi / self.N * np.tan(alpha)
        # time spread / duration
        dt = sigma * np.sqrt(self.N / pi / 2)
        self.Param[1:] = [tc, fc, cr, dt]
