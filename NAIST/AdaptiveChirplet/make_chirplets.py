'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: DecompDeepCrossChirplet.py
@time: 12/7/2019 4:45 PM
'''

"""
Decompose deep cross chirplets
This simulation compares the results of MPEM and MLE algorithm for the
estimation of the components of a signal, consisting of an upward and a downward
chirplet, embeded in noise
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from numpy import pi
import copy


class MakeChirplets:
    """
    make_chirplets construct summation of chirplets
    --------sytax----------------
    x = make_chirplets(N)
    x = make_chirplets(_, p)
    x = make_chirplets(_, 'PeriodEff', p_eff)
    x = make_chirplets(_, 'PType', p_type)
    ---------input-------------
    N: required length of signal
    P: optional M*5 matrix of parameters, where M is the number of
        chirplets and each row is [A t f cr d] (units depending on parameter 'Equation'.
        A: complex amplitude A = |A|e^{j\phi}
        t: time center (sample or unit sampling time)
        f: frequency center (rad)
        cr: chirprate (rad/sample)
        d: chirplet duration (sample)
        (default [1, N/2, 0, 0, sqrt(N/4/pi)])
    PeriodEff: parameter of discretization effect, p_eff = {true, false}
    (default p_eff = true)
    PType: parameter of type of P, p_type={'Oneill', 'Cohen'}
    (default 'ONeill')
    -----------output------------
    x           - constructed signal


    -----------Note--------------
    In O'Neill's equation, 'd' is the standard deviation of the guassian.
    d = \sqrt{\frac{N}{4\pi}} gives an atom with a circular Wigner distribution
    and 2*sqrt(2)*d is the Rayleigh limit. Assume sampling frequency = 1 Hz
    Nyquist = \pi. Use rad for frequency unit.

    In Cohen's equation, the chirplet formula is A(\frac{d}{\pi})^{1/4}
    e^{-\frac{d}{2}(n-t)^2} e^{j[\frac{cr}{2}(n-t)-f](n-t)}. Assume
    sampling frequency is N and Nyquist = \pi.

    -----------Example--------------------------
    N = 100; x = make_chirplets(N, [1 N/2 pi/2 pi/N sqrt(2*N/pi)]);
    N = 100; x = make_chirplets(N, [1, 1/2, pi/2, pi, pi/4], 'PType', 'Cohen');
    """

    def __init__(self, N, P, PeriodEff='PType', PType='ONeill'):
        # if np.asarray(P).size == 5:
        #     P = np.asarray(P)[np.newaxis,:]
        # else:
        #     P = np.asarray(P)
        P = np.asarray(P).reshape([-1, 5])

        self.N = N  # N: signal size
        self.P = P  # P: optional M*5 matrix of parameters, where M is the number of chirplets
        # each row is [A t f cr d](units depending on parameter 'Equation'.)
        #   A: complex amplitude A = |A|e^{j\phi}
        #   t: time center (sample or unit sampling time)
        #   f: frequency center (rad)
        #   cr: chirprate (rad/sample)
        #   d: chirplet duration (sample)
        #   (default [1, N/2, 0, 0, sqrt(N/4/pi)])

        self.PType = str(PType).lower()
        self.dflag = 0 if isinstance(PeriodEff, bool) else 1  # consider the discretization effect (Optional: default 0)
        self.check_inputs()
        self.convert_p()
        self.ONeillEquation()

        # print(self.P)

    def ONeillEquation(self):
        self.sig = np.zeros(self.N, dtype=complex)
        for i in range(np.size(self.P, axis=0)):
            A_k = self.P[i, 0]
            t_k = self.P[i, 1]
            f_k = self.P[i, 2].real
            # if f_k > pi:
            #     self.P[i, 2] = f_k - 2*pi
            cr_k = self.P[i, 3]
            d_k = self.P[i, 4]
            self.chirplet(t_k, f_k, cr_k, d_k)
            # print(self.clet.size)
            # clet_ = self.clet.real
            # clet_ = abs(clet_) * np.real(clet_) / abs(clet_.real)
            # plt.figure()
            # plt.plot(clet_)
            # plt.show()
            # plt.title('1')

            self.sig += A_k * self.clet
            # self.sigorig = copy.deepcopy(self.sig)
        if sum(self.sig.imag.__abs__()) == 0:

            self.sig = self.sig.real.flatten()
        else:
            self.sig = self.sig.flatten()

    def chirplet(self, t, f, cr, d):
        """
        # chirplet build one unitary chirplet

        sytax:
        clet = chirplet(N, t, f, cr, d, dflag)

        inputs:
        N:  signal length
        t: time center
        f: frequency center
        cr: chirp rate
        d: time duration of the chirplet (sample or unit sampling time)
        dflag: consider the discretization effect (Optional: default 0)
        """
        rep = 5  # control the accuracy of discretization
        n = np.arange(1, self.N + 1)

        if self.dflag == True:  # consider effect
            for r in range(-rep, rep + 1):
                acp = np.exp(-((n + r * self.N - t) / 2 / d) ** 2)
                bcp = np.exp(1j * cr / 2 * (n + r * self.N - t) ** 2)
                ccp = np.exp(1j * f * (n + r * self.N - t))
                if r == -rep:
                    dcp = acp * bcp * ccp
                else:
                    dcp += acp * bcp * ccp

            self.clet = dcp / np.linalg.norm(dcp)
        else:  # dont consider periodization
            am = np.exp(-((n - t) / 2 / d) ** 2) * np.sqrt(1 / np.sqrt(2 * pi) / d)  # for normalization
            chirp = np.exp(1j * (cr / 2 * (n - t) ** 2 + f * (n - t)))
            self.clet = am * chirp

    def check_inputs(self):
        self.isValidChirplets(self.P)

    def isValidChirplets(self, P):
        # check if valid chirplet paramter
        if np.size(P, axis=1) == 5:
            # print(np.size(P, axis=1))
            self.P = P
        else:
            self.P = np.asarray([1, self.N / 2, 0, 0, np.sqrt(self.N / 4 / pi)])[np.newaxis, :]

    def convert_p(self):
        # convert cohen paras to O'Neill
        if self.PType == 'cohen':
            self.PCo2On()

    def PCo2On(self):
        """
        PCO2ON convert chirplet parameters from Cohen to O'Neil format

        Syntax:
        P_On = PCo2On(N, P_co)

        inputs:
        N: signal length
        P_co: chirplet parameters of Cohen format

        outputs:
        P_on: chirplet parameters of O'Neill format

        Note:
            In Cohen format, sampling frequency is normalized to N and the
            frequency range is normalized to 2*\pi. In O'Neill format, sampling
            frequency is normalized to 1 and frequency 2*\pi.
        """
        # print(self.P.size)
        P_on = np.zeros_like(self.P)
        # amplitude
        P_on[:, 0] = self.P[:, 0]
        # t center
        P_on[:, 1] = self.P[:, 1] * self.N
        # f center
        P_on[:, 2] = self.P[:, 2]
        # chirp rate
        P_on[:, 3] = self.P[:, 3] / self.N
        # P_on.append(self.P[:, 3] / self.N)
        # duration
        P_on[:, 4] = np.sqrt(self.N / self.P[:, 4] / 2)
        self.P = P_on
        # P_on.append(np.sqrt(self.N / self.P[:, 4] / 2))
        # self.P = np.asarray(P_on)[np.newaxis, :]


if __name__ == '__main__':
    N = 100  # signal size

    # use cohen equation
    p_type = 'ONeill'
    p1 = np.asarray([10, 1 / 2, pi / 2, pi, 1 / 18])

    p2 =  np.asarray([10, 1 / 2, pi / 2, -pi, 1 / 18])
    p3 = [p1, p2]
    sig1 = MakeChirplets(N, p1, 'PType', p_type)

    sig2 = MakeChirplets(N, p2, 'PType', p_type)
    sig3 = MakeChirplets(N, p3, 'PType', p_type).sig
    sig1 = sig1.sig
    sig2 = sig2.sig
    sig = sig1 + sig2
    plt.figure()
    plt.subplot(411)
    plt.plot(np.real(sig))
    plt.subplot(412)
    plt.plot(np.real(sig1))
    plt.subplot(413)
    plt.plot(np.real(sig2))
    plt.subplot(414)
    plt.plot(np.real(sig3))
