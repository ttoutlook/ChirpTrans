'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: testsignal.py
@time: 12/19/2019 7:59 PM
'''

import numpy as np
from MakeChirplet import MakeChirplet
from numpy import pi
from functionSet import functionSet as funs
from scipy.signal import hilbert, sawtooth
from ChirpletLocate import MapChirplet


class testsignal:
    def __init__(self, randSet=False, randomSeed=0):
        if randSet:
            np.random.seed(randomSeed)

    def CrossSignal(self, N=500, fs=1, snr=0):
        # use cohen equation
        p_type = 'Cohen'
        P1 = [10, 1 / 2, pi / 2, pi, 1 / 18]  # up-chirplet 0 -> pi
        P2 = [10, 1 / 2, pi / 2, -pi, 1 / 18]  # down-chirplet pi -> 0
        s1 = np.real(MakeChirplet(N, P1, p_type).sig)  # the synthesized signal
        s2 = np.real(MakeChirplet(N, P2, p_type).sig)  # the synthesized signal
        s = s1 + s2
        # add guassian noise at the desired_signal-to-noise (SNR)
        # level, |d_snr|, in dB. * You can change |d_snr| for your experiments*

        d_snr = snr  # desired SNR
        spn, ns = funs(s).wgn(d_snr)
        # e_snr = funs(s).SNR(spn)
        # self.clear = s
        # self.noisy = spn
        # self.Param = [P1, P2]
        return s, spn

    def MultiSignal(self, T=500, fs=1, snr=0):
        # construct and display the signals
        # first signal length and sampling frequency
        T = 512  # signal duration
        fs = 1  # sampling frequency

        # wave I consists of a sine(A), a sawtooth(B) and A gabor(C) waveform
        waveI = np.zeros(T)
        # componentA
        dt_a = 46  # length of component A
        tc_a = 90  # time_center of component A
        fc_a = 1 / dt_a  # frequency center of component A
        t = np.arange(0, dt_a)
        s_a = -np.sin(2 * pi * fc_a * t / fs)

        # component B
        tc_b = 422
        t1 = np.arange(0, dt_a / 2)
        s_b1 = -(1 / 2 * sawtooth(2 * pi * 2 * fc_a * t1 / fs, 1 / 2) + 1 / 2)
        s_b2 = -s_b1
        s_b = np.append(s_b1, s_b2)

        # component C
        dt_c = 28  # length of c
        fc_c = 0.4  # frequency center of C
        tc_c = 256  # time center of C
        s_c = funs().garbor1d(T, fs, dt_c, fc_c, tc_c, 1, 0)  # signal C

        # construct wave I
        ind1a = int(round(tc_a - dt_a / 2) - 1)
        ind1b = int(round(tc_b - dt_a / 2) - 1)
        ind2a = int(round(tc_a - dt_a / 2) + dt_a - 1)
        ind2b = int(round(tc_b - dt_a / 2) + dt_a - 1)
        waveI[ind1a:ind2a] = s_a
        waveI[ind1b:ind2b] = s_b
        waveI = waveI + s_c

        # wave II is a Gabor waveform
        dt_d = 2 * dt_c  # length of D
        fc_d = 2 * fc_c / 3  # frequency center of D
        tc_d = 256  # time center of D
        s_d = funs().garbor1d(T, fs, dt_d, fc_d, tc_d, 1, 0)
        waveII = s_d

        # wave III consists of a pulse (E) and a sinusoidal (F) waveform
        tc_e = 128  # component E
        s_e = funs().garbor1d(T, fs, 0, 0, tc_e, 2, 0)
        fc_f = 0.35  # component F
        A_f = 0.2
        t2 = np.arange(0, T * fs)
        s_f = A_f * np.sin(2 * pi * fc_f * t2 / fs)
        waveIII = s_e + s_f

        # wave IV is a upward chirplet (G)
        A_cp = 6  # amplitude, total energy of the chirplet
        tc_cp = 350  # time center
        fc_cp = 0.2 * 2 * pi / fs
        cr = pi / T  # chirp rate
        dt_cp = 70  # size of the chirplet
        P = [A_cp, tc_cp, fc_cp, cr, dt_cp]
        cp = MakeChirplet(T, P).sig  # chirplets out is a complex signal
        waveIV = cp.real

        # finally, we add all the components together
        chirpsim = waveI + waveII + waveIII + waveIV

        d_snr = snr  # desired SNR
        spn, ns = funs(chirpsim).wgn(d_snr)

        return chirpsim, spn
