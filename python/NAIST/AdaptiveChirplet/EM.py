'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: EM.py
@time: 12/14/2019 6:00 PM
'''

import numpy as np
from numpy import pi
import copy
from make_chirplets import MakeChirplets
from mp_chirplet import mp_chirplet


class EMChirplets:
    """
    EMChirplets: refine multiple chirplets with Expectation-Maximization alogrithm
    inputs:
    x: signal
    P: vector of chirplet parameters
    M: resolution for newton-raphson refinement
    radix: radix of scale
    mnits: maximum number of iteration
    mstep: method of m-step, 'ONeill'(default) or 'FEDER'
    outputs:
    P: vector of chirplet parameters
    err: err = signal-chirplets
    res: residual = norm(signal-chirplets)
    """

    def __init__(self, x, P, M, D, i0, radix, mnits, mstep='oneill'):
        self.x = x
        self.res = []
        self.P = np.asarray(P)
        self.N = len(x)
        self.D = D
        self.mnits = mnits
        self.M = M
        self.i0 = i0
        self.radix = radix
        self.mstep = mstep
        self.initial()

    def initial(self):
        self.Q = np.size(self.P, axis=0)  # the number of chirplets
        P0 = np.zeros([self.Q, 5])  # initial values of chirplet parameters
        Ts = np.ones([self.Q, 1]) * np.asarray([0.1, 0.001, 1e-5, .1])  # tolerance of parameter error

        # expectation-maximization (EM)
        j = 1
        Pe = np.abs(self.P[:, 1:5] - P0[:, 1:5])  # parameter error
        while sum(sum(Pe > Ts)) and j <= self.mnits and self.Q > 1:
            P0 = copy.deepcopy(self.P)
            # E-Step
            z = MakeChirplets(self.N, P0).sig
            self.d = self.x - z

            # M-Step
            self.cal_mstep(j)
            Pe = abs(self.P[:, 1:5] - P0[:, 1:5])
            j += 1

        # update the residual
        y = MakeChirplets(self.N, self.P).sig
        self.err = self.x - y



    def cal_mstep(self, j):
        # calculate m-step
        if self.mstep.lower() == 'oneill':
            # refine only one chirplet at each iteration
            self.mstep_oneill(j)
        else:
            # refine all the chirplets at each iteration
            self.mstep_feder()

    def mstep_feder(self):
        for k in range(self.Q):
            z_k = MakeChirplets(self.N, self.P[k, :]).sig
            y_k = z_k + self.d / self.Q
            P_k = mp_chirplet(y_k, self.M, self.D, self.i0, self.radix).P
            self.P[k, :] = P_k

    def mstep_oneill(self, j):
        # O'Neill method of m-step
        for k in range(self.Q):
            if np.mod(k, j) == 0:  # O'Neill method
                z_k = MakeChirplets(self.N, self.P[k, :]).sig
                y_k = z_k + self.d
                P_k = mp_chirplet(y_k, self.M, self.D, self.i0, self.radix).P
                self.P[k, :] = P_k
