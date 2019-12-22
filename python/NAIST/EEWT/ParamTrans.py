'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: ParamTrans.py
@time: 12/19/2019 8:51 PM
'''

# This params is used to translate the parameters of gaussian chirplet atom detected roughly to EWT boundaries
# First, analyze the frequency distance between each other, if no overlap, use meyer wavelet to extract them directly.
# Second, for overlap components, we will use chirplet atoms to approximate the components with varying frequency, and use
# Meyer wavelet components to extract the signal components with constant frequency.

import numpy as np
from functionSet import functionSet as funs
from MakeChirplet import MakeChirplet
from scipy.signal import hilbert
from skimage.measure import find_contours
import matplotlib.pyplot as plt


class ParamsTrans:
    def __init__(self, params, N, Q, Num=200):
        self.params = np.asarray(params).reshape([-1, 5])  # the chirplet parameters
        self.N = N  # the number of data
        self.Q = Q  # the number of components
        self.Num = N  # Discrete number
        self.paramsCohen = self.Oneill2Cohen(self.params, self.N)
        self.paramsOneill = self.cohen2oneill(self.paramsCohen, self.Num)

        self.initial()

    def initial(self):
        componentId = []  # constant frequency components
        ComponentxBoundaries = []  # Variable frequency components
        ComponentyBoundaries = []  # Variable frequency components
        ComponentCenter = []  # Variable frequency components
        xbounds = []
        ybounds = []
        for i in range(self.Q):
            A, t, f, cr, d = self.paramsCohen[i, :].tolist()
            print('Component No.', i + 1)
            print('t:', t, 'f:', f)
            print('cr:', cr, 'd:', d)
            # f: frequency center
            # cr: chirplet angle in oneill format,

            wig, t, f = self.chirpltwvd_explicit(self.paramsOneill[i, :].tolist(), self.Num)
            sig = MakeChirplet(self.Num, self.paramsOneill[i, :].ravel()).sig.real
            funs(sig).contour(wig, t, f)

            # detect the frequency boundaries by using the chirpltwvd_explicit
            contours = self.det_boundaries(wig)
            xboundaries, yboundaries = np.asarray(contours)
            xb1 = xboundaries[0]
            xb2 = xboundaries[-1]
            yb1 = yboundaries[0]
            yb2 = yboundaries[-1]
            xbounds.append([xb1, xb2])
            ybounds.append([yb1, yb2])

            if np.floor(cr) == 0:
                componentId.append(i + 1)
                ComponentxBoundaries.append([xb1, xb2])
                ComponentyBoundaries.append([yb1, yb2])
                ComponentCenter.append([t, f])

        self.xbs = xbounds
        self.ybs = ybounds
        self.
        # plt.xlim([0, self.N])
        # plt.ylim([0, 0.5])

    def det_boundaries(self, wig, threshold=0.01):
        """
        det_boundaries： detect the time and frequency boundaries
        """
        tfr = np.real(wig * np.conj(wig))
        tfr = np.sqrt(tfr)
        _threshold = np.max(tfr) * threshold
        tfr[tfr <= _threshold] = 0
        # find the boundaries at a constant threshold
        xtfr = np.sum(tfr, axis=0)
        ytfr = np.sum(tfr, axis=1)
        tbounds = np.flatnonzero(xtfr)
        fsbounds = np.flatnonzero(ytfr)
        return tbounds, fsbounds

    def Oneill2Cohen(self, P, N):
        """
        PCO2ON convert chirplet parameters from O'Neill to Cohen format
        """
        P_on = np.zeros_like(P)
        # amplitude
        P_on[:, 0] = P[:, 0]
        # t center
        P_on[:, 1] = P[:, 1] / N  #
        # f center
        P_on[:, 2] = P[:, 2]
        # chirp rate
        P_on[:, 3] = P[:, 3] * N
        # duration
        P_on[:, 4] = N / (2 * P[:, 4] ** 2)
        return P_on

    def cohen2oneill(self, P, N):
        """
        convert parameters from cohen style to oneill style
        """
        P_on = np.zeros_like(P)
        # amplitude
        P_on[:, 0] = P[:, 0]
        # t center
        P_on[:, 1] = P[:, 1] * N
        # f center
        P_on[:, 2] = P[:, 2]
        # chirp rate
        P_on[:, 3] = P[:, 3] / N
        # duration
        P_on[:, 4] = np.sqrt(N / P[:, 4] / 2)
        return P_on

    def chirpltwvd_explicit(self, P, N, fs=1):
        """
        Chirpltwvd_explicit: compute chirplet spectrogram with explicit formula
        """
        P = np.asarray(P).reshape([-1, 5])
        t = np.linspace(0, N, 2 * N)  # vector of time range (assume fs = 1Hz)
        f = np.linspace(0, 2 * np.pi, 2 * N)  # vector of frequency range [0, 1] range
        nchirp = np.size(P, axis=0)
        tmat, fmat = np.meshgrid(t, f)
        wig = np.zeros_like(tmat)
        for k in range(nchirp):
            A_k, tc_k, fc_k, c_k, d_k = P[k, :].tolist()
            # WVD = w1 * w2 * w3
            #  w1 = a^2/pi
            #  w2 = exp(-(n-tc)^2/2/d^2)
            #  w3 = exp(-2*d^2*((f-fc)-c*(n-tc))^2)
            a = abs(A_k)
            w1 = a ** 2 / np.pi
            w2 = np.exp(-(tmat - tc_k) ** 2 / 2 / d_k ** 2)
            w3 = np.exp(-2 * d_k ** 2 * ((fmat - fc_k) - c_k * (tmat - tc_k)) ** 2)
            w_k = w1 * w2 * w3
            wig = wig + w_k
        f = np.linspace(0, 1, 2 * N)
        return wig, t, f
