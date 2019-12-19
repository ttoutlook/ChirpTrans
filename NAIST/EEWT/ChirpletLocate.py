'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: ChirpletLocate.py
@time: 12/19/2019 3:12 PM
'''

import numpy as np
from MakeChirplet import MakeChirplet
from map_chirplets import MapChirplet
from numpy import pi


# this program is used to find the chirplet atom in original signal
# decompose signal with MLE adaptive chirpelt transform

class ChirpletLocate:
    def __init__(self, x, Q, M=64, CP0=np.array([1, pi / 2, 0, 1])[np.newaxis, :], p_type='oneill', i0=1, radix=2,
                 Depth=5, methodid=1, windowid=0, windowlength=4):
        self.x = np.asarray(x).flatten()  # the orginal signal
        self.res = np.asarray(x).flatten()  # the rest signal components
        self.Q = Q  # the target number
        self.M = M  # the length scale of window
        self.CP0 = CP0  # the intial parameters of chirplet atom
        self.p_type = str(p_type).lower()  # the parameter type of chirplet parameters, p_type = {'Oneill', 'Cohen'}
        self.i0 = i0  # the first level to rotate the chirplets
        self.radix = radix  # the radix of scale
        self.Depth = Depth  # the depth of decomposition
        self.methodid = methodid  # the id of solve operator, including, p_type = {'Oneill', 'Cohen'}
        self.windowid = windowid  # the id of window function:
        # default use boxcar window, for strong noisy signal and we recommend
        # hamming, bartlett, flattop, bohman, especially bartharr (the last one)
        # ['boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett', 'flattop', 'parzen', 'bohman',
        #                    'blackmanharris', 'nuttall', 'barthann']
        self.windowlength = windowlength # the length of window
        self.N = self.x.size
        self.initial()

    def inital(self):


