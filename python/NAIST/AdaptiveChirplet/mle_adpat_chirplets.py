'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: mle_adpat_chirplets.py
@time: 12/8/2019 12:38 PM
'''

import numpy as np
from scipy.signal import hilbert
from numpy import pi
import copy
from make_chirplets import MakeChirplets
import scipy.optimize as opt
from Optimization import FindBestChirpRate
from best_chirplet import best_chirplet
from MLE import MLEChirplets


class mle_adapt_chirplets:
    """
    mle_adapt_chirplets decompose signal with MLE adaptive chirplet transform

    ---------syntax------------
    [P, res] = mle_adapt_chirplets(x,Q)
    [P, res] = mle_adapt_chirplets(____, M)
    [P, res] = mle_adapt_chirplets(____, M, CP0)
    [P, res] = mle_adapt_chirplets(____, M, CP0, verbose)
    [P, res] = mle_adapt_chirplets(____, M, CP0, verbose, mnits)
    [P, res] = mle_adapt_chirplets(____, M, CP0, verbose, mnits, level)
    [P, res] = mle_adapt_chirplets(____, 'RefineAlgorithm', ref_alg)
    [P, res] = mle_adapt_chirplets(____, 'PType', p_type)

    -------inputs---------------
    x: signal
    Q: number of chirplets to look for (if Q=0, until press 'q' to quit)
    M: resolution for Newton-Raphson refinement (optional, default = 64)
    CP0: initial value of chirplet [tc, fc, cr, d] (default = [1, pi/2, 0, 1])
    verbose: verbose flag, 'yes', 'no', 'vv' (default = yes)
    mnits: maximum number of iterations (optional, default = 5)
    level: level of difficulty of MLE
    ref_alg: { 'expectmax', 'maxlikeliest' }
    PType: (parameter) type of chirplet parameter P, p_type = {'Oneill', 'Cohen'}

    ------outputs--------------
    P: Q_by_5 matrix of chirplet parameters (see make_chirplets.m)
    res: norm of the signal and the residuals for 1 to Q chirplets;
         could be used for a selection of stopping criterion
    """

    def __init__(self, x, Q, M=64, CP0=np.array([1, pi / 2, 0, 1])[np.newaxis, :], mnits=2, level=2,
                 ref_alg='maxlikeliest', p_type='Oneill', methodid = 4):
        self.x = np.asarray(x).flatten()
        self.err = np.asarray(x).flatten()
        self.Q = Q
        self.M = M
        self.res = []
        self.CP0 = CP0  # CP0 - initial value of chirplet [tc, fc, cr, d] (default = [1 pi/2 0 1])
        self.decf = 1  # the down-sampling rate of STFT
        self.mnits = mnits  # mnits   - maximum number of iterations (optional, default = 5)
        self.level = level  # level   - level of difficulty of MLE
        self.ref_alg = ref_alg  # ref_alg - { 'expectmax', 'maxlikeliest' }
        self.p_type = str(
            p_type).lower()  # PType   - (parameter) type of chirplet parameter P, p_type = {'Oneill', 'Cohen'}
        self.N = self.x.size  # signal length
        self.methodid = methodid  # (0"Nelder-Mead", 1"Powell", 2"CG", 3"BFGS", 4"L-BFGS-B")
        self.FindChirplets()


    # find chirplets use MLE algorithm
    def FindChirplets(self):

        # initial parameters
        # tc = self.CP0[:, 0]
        # fc = self.CP0[:, 1]
        # cr = self.CP0[:, 2]
        # d = self.CP0[:, 3]
        tc, fc, cr, d = self.CP0.flatten().tolist()
        self.res.append(np.linalg.norm(self.x))  # the residual components

        done = False
        i = 1
        # e = self.x  # residual
        self.P = []
        while done == False:
            # find a new chirplet with MLE
            # if self.verbose == 'yes' or self.verbose == 'vv':
            #     print('Single Chirplet estimation -- MLE algorithm')

            # estimate the i-th chirplet
            # self.best_chirplet(self.x, self.level, self.M, cr, d, tc, fc, vb)
            bestChirp = best_chirplet(self.err, self.level, self.M, self.decf, tc, fc, cr, d, methodid=self.methodid)
            # bestChirp.p_ = np.mod(bestChirp.p_[p])
            if i == 1:
                self.P.append(bestChirp.p_)
            else:
                self.P = list(self.P)
                self.P.append(bestChirp.p_)

            # plt.figure()
            # t, f, c, d = bestChirp.p_[1:].tolist()
            # plt.plot(MakeChirplets(self.N, [1, t, f, c, d]).sig.real)
            # plt.show()

            # refine the estimated chirplets using maximum-likelihood estimation (MLE)
            MLEClass = MLEChirplets(self.x, self.level, self.M, self.decf, self.mnits, self.P)
            self.err = MLEClass.err
            self.P = MLEClass.P
            # get the residual
            # y = MakeChirplets(self.N, self.P).sig.flatten()
            # plt.figure()
            # plt.plot(y)

            # self.err = self.err - y
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
    import matplotlib.pyplot as plt

    fs = 100
    t = np.arange(1, 1001)
    sig = np.sin(t) * np.hanning(t.size)
    # sig = np.sin(t)
    sigclass = mle_adapt_chirplets(sig, 10)
    plt.figure()
    plt.subplot(411)
    plt.plot(t, sigclass.x.real)
    plt.subplot(412)
    plt.plot(MakeChirplets(sig.size, sigclass.P).sig.real)
    plt.subplot(413)
    plt.plot(sigclass.err.real)
    plt.subplot(414)
    plt.plot(sigclass.res)
