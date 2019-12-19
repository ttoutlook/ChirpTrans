'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: EEWT.py
@time: 12/17/2019 4:44 PM
'''

import numpy as np

from scipy.signal import hilbert
from functionSet import functionSet as funs
from ChirpletLocate import ChirpletLocate



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
        self.N = sig.__sizeof__()
        self.components = components
        self.mode = mode  # The analysis mode
        self.lengthfilter = lengthfilter
        self.sigmafilter = sigmafilter
        self.initial()

    def initial(self):

        # create chirplet family and get feedback parameters
        self.params = ChirpletLocate(self.sig, self.components).Param


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from testsignal import testsignal
    cross = testsignal().CrossSignal().clear
    multi = testsignal().MultiSignal().clear
    params = EEWT(cross).params
    tfr, t, f = funs(cross).ACS_plot(params)
    title = 'Chirp atoms'
    funs(cross).contour(tfr, t, f, P=params, reconstruction=True, title=title)




