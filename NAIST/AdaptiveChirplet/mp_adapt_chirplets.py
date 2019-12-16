'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: mp_adaptive_chirplets.py
@time: 12/13/2019 3:02 PM
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from make_chirplets import MakeChirplets
from mp_chirplet import mp_chirplet
from EM import EMChirplets
from MLE import MLEChirplets


class mp_adapt_chirplets:
    """
    mle_adapt_chirplets decompose signal with MLE adaptive chirplet transform
    """

    def __init__(self, x, Q, M=64, CP0=np.array([1, pi / 2, 0, 1])[np.newaxis, :], mnits=2, level=2,
                 ref_alg='expectmax', p_type='Oneill', i0=1, radix=2, D=5, methodid=1):
        self.x = np.asarray(x).flatten()
        self.err = np.asarray(x).flatten()
        self.Q = Q
        self.M = M
        self.D = D  # The depth of decomposition
        self.res = []
        self.i0 = i0  # the first level to rotate the chirplets (default = 1)
        self.radix = radix  # the radix of scale (default =2 )
        self.CP0 = CP0  # CP0 - initial value of chirplet [tc, fc, cr, d] (default = [1 pi/2 0 1])
        self.decf = 1  # the down-sampling rate of STFT
        self.mnits = mnits  # mnits   - maximum number of iterations (optional, default = 5)
        self.level = level  # level   - level of difficulty of MLE
        self.ref_alg = ref_alg  # ref_alg - { 'expectmax', 'maxlikeliest' }
        self.p_type = str(
            p_type).lower()  # PType   - (parameter) type of chirplet parameter P, p_type = {'Oneill', 'Cohen'}
        self.N = self.x.size  # signal length
        self.methodid = methodid
        self.FindChirplets()

    # find chirplets use MPEM algorithm
    def FindChirplets(self):
        # initial parameters
        # tc, fc, cr, d = self.CP0.flatten().tolist()
        self.res.append(np.linalg.norm(self.x))
        i = 1
        self.P = []
        done = False

        while done == False:
            # find a new chirplet with MP + newton-Raphson

            # estimate the i-th chirplet
            mpchirp = mp_chirplet(self.err, self.M, self.D, self.i0, self.radix,
                                  methodid=self.methodid)  # output in O'Neill's format
            if i == 1:
                self.P.append(mpchirp.P)
            else:
                self.P = list(self.P)
                self.P.append(mpchirp.P)

            # refine the estimated chirplets
            # expectation-maximization (EM)
            if self.ref_alg.lower() == 'expectmax':
                EMClass = EMChirplets(self.x, self.P, self.M, self.D, self.i0, self.radix, self.mnits, mstep='oneill')
                self.P = EMClass.P
                self.err = EMClass.err
            else:
                MLEClass = MLEChirplets(self.x, self.level, self.M, self.decf, self.mnits, self.P)
                self.P = MLEClass.P
                self.err = MLEClass.err
            self.res.append(np.linalg.norm(self.err))
            if self.res[-1] > self.res[-2]:
                self.mnits = 0

            # termination criteria
            if self.Q == i:
                done = True
            print(i)
            i += 1

        # convert P format if neccessary
        if self.p_type == 'cohen':
            self.P = self.POn2Co()

    def POn2Co(self):
        """
        POn2Co convert chirplet parameters from O'Neil to Cohen format

        syntax：
        P_co= PCo2Co(N, P_on)

        inputs:
        N: signal length
        P_on: chirplet parameters of O'Neill

        outputs:
        P_co: chirplet parameters of Cohen format

        % Note:
        %   In Cohen format, sampling frequency is normalized to N and the
        %   frequency range is normalized to 2*\pi. In O'Neill format, sampling
        %   frequency is normalized to 1 and frequency 2*\pi.
        """
        P_co = np.zeros_like(self.P)
        # amplitude
        P_co[:, 0] = self.P[:, 0]
        # t center
        P_co[:, 1] = self.P[:, 1] / self.N
        # f center
        P_co[:, 2] = self.P[:, 2]
        # chirp rate
        P_co[:, 3] = self.P[:, 3] * self.N
        # duration
        P_co[:, 4] = self.N / self.P[:, 4] ** 2 / 2
        self.P = P_co


if __name__ == '__main__':
    from scipy.signal import hilbert
    import time
    import matplotlib.pyplot as plt
    from functionSet import functionSet as funs

    fs = 1
    T = 512
    # wave III consists of a pulse (E) and a sinusoidal (F) waveform
    tc_e = 128  # component E
    s_e = funs().garbor1d(T, fs, 0, 0, tc_e, 2, 0)
    fc_f = 0.35  # component F
    A_f = 0.2
    t2 = np.arange(0, T * fs)
    s_f = A_f * np.sin(2 * pi * fc_f * t2 / fs)
    sig = s_e + np.random.randn(s_e.size) * 0.1

    # fs = 100
    # t = np.arange(1, 1001)
    # sig = np.sin(t) * np.hanning(t.size)
    # sig = np.sin(t)
    sigclass = mp_adapt_chirplets(sig, 2, methodid=1)
    plt.figure()
    plt.subplot(411)
    plt.plot(sigclass.x.real)
    plt.subplot(412)
    plt.plot(MakeChirplets(sig.size, sigclass.P).sig.real)
    plt.subplot(413)
    plt.plot(sigclass.err.real)
    plt.subplot(414)
    plt.plot(sigclass.res)
