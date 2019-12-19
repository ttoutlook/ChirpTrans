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
        self.windowlength = windowlength  # the length of window
        self.N = self.x.size
        self.En = []  # rest signal energy
        self.initial()

    def initial(self):
        # initial parameters
        self.En.append(np.linalg.norm(self.x))
        i = 1
        self.Param = []
        done = False
        while done == False:
            # find a new chirplet with MP + newton-Raphson
            Mapchirp = MapChirplet(self.res, self.M, self.Depth, self.i0, self.radix, self.methodid, self.windowid,
                                   self.windowlength) # output in O'Neill's format
            if i==1:
                self.Param.append(Mapchirp.Param)
            else:
                # self.Param = list(self.Param)
                self.Param.append(Mapchirp.Param)

            sig = MakeChirplet(self.N, self.Param).sig
            self.res = self.x - sig
            self.En.append(np.linalg.norm(self.res))
            print("Component No.",i)
            # termination criteria
            if self.Q == i:
                done = True
            i +=1
        if self.p_type == 'cohen':
            self.Param = self.cohen2oneill(self.Param)


    def cohen2oneill(self, P):
        """
        convert parameters from cohen style to oneill style
        """
        P_on = np.zeros_like(P)
        # amplitude
        P_on[:, 0] = P[:, 0]
        # t center
        P_on[:, 1] = P[:, 1] * self.N
        # f center
        P_on[:, 2] = P[:, 2]
        # chirp rate
        P_on[:, 3] = P[:, 3] / self.N
        # duration
        P_on[:, 4] = np.sqrt(self.N / P[:, 4] / 2)
        return P_on

if __name__ == '__main__':
    from scipy.signal import hilbert
    import time
    import matplotlib.pyplot as plt
    from functionSet import functionSet as funs

    fs = 1
    T = 1000
    # wave III consists of a pulse (E) and a sinusoidal (F) waveform
    # tc_e = 128  # component E
    # s_e = funs().garbor1d(T, fs, 0, 0, tc_e, 2, 0)
    # fc_f = 0.35  # component F
    # A_f = 0.2
    # t2 = np.arange(0, T * fs)
    # s_f = A_f * np.sin(2 * pi * fc_f * t2 / fs)
    # sig = s_e + np.random.randn(s_e.size) * 0.1

    fs = 100
    t = np.arange(1, 10001)
    sig = np.sin(t) * np.hanning(t.size)
    # sig = np.sin(t)
    sigclass = ChirpletLocate(sig, 10, methodid=1, M=1024)
    plt.figure()
    plt.subplot(411)
    plt.plot(sigclass.x.real)
    plt.subplot(412)
    plt.plot(MakeChirplet(sig.size, sigclass.Param).sig.real)
    plt.subplot(413)
    plt.plot(sigclass.res.real)
    plt.subplot(414)
    plt.plot(sigclass.En)