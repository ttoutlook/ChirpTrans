'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: EEWT.py
@time: 12/17/2019 4:44 PM
'''

import numpy as np
from numpy import pi
from scipy.signal import hilbert
from functionSet import functionSet as funs
from ChirpletLocate import ChirpletLocate
from ParamTrans import ParamsTrans


# This is the main program of EEWT (Enhanced Empirical Wavelet Transform)
# This Program first uses the Adaptive Transform to detect the frequency center, time center, duration and chirplet rate
# if chirplet rate is 0, use meyer wavelet to extract this components,
# if chirplet rate is over 0, use chirplet wavelet to approximate this components

class EEWT:
    def __init__(self, sig=1, fs=1, components=5, mode='EWT', lengthfilter=10, sigmafilter=5):
        if np.iscomplex(sig).all:
            self.sig = hilbert(sig)
        else:
            self.sig = sig
        self.fs = fs
        self.N = np.size(sig)
        self.components = components
        self.mode = mode  # The analysis mode
        self.lengthfilter = lengthfilter
        self.sigmafilter = sigmafilter
        self.initial()

    def initial(self):
        # create chirplet family and get feedback parameters
        self.params = ChirpletLocate(self.sig, self.components).Param
        bounds = ParamsTrans(self.params, self.N, self.components)
        self.xbs = bounds.xbs
        self.ybs = bounds.ybs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from testsignal import testsignal

    cross, _ = testsignal().CrossSignal()
    multi, _ = testsignal().MultiSignal()
    eewt = EEWT(multi, components=7)
    params = eewt.params
    tfr, t, f = funs(multi).ACS_plot(params)
    title = 'Chirp atoms'
    funs(multi).contour(tfr, t, f, P=params, reconstruction=True, title=title)

    xbounds = eewt.ybs
    plt.figure()
    for xbs in xbounds:
        color = np.random.rand(3,)
        x1, x2 = xbs
        plt.vlines(x1, ymin=0, ymax=1, colors=color)
        plt.vlines(x2, ymin=0, ymax=1, colors=color)
        plt.hlines(0,0,512)
