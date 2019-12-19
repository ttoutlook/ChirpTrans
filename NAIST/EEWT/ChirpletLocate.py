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
        self.x = x
        self.Q = Q
        self.M = M
        self.CP0 = CP0
        self.p_type = p_type
        self.i0 = i0
        self.radix = radix
        self.Depth = Depth
        self.methodid = methodid
        self.windowid = windowid
        self.windowlength = windowlength

