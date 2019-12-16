'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: MLE.py
@time: 12/11/2019 3:38 PM
'''
import numpy as np
from numpy import pi
from make_chirplets import MakeChirplets
from best_chirplet import best_chirplet
import matplotlib.pyplot as plt
import copy


class MLEChirplets:
    def __init__(self, x, level, M, decf, mnits, P, ):
        self.x = x
        self.res = []
        # self.P = np.asarray(P)
        self.P_ = np.asarray(P)
        self.N = len(x)
        self.level = level
        self.mnits = mnits
        self.M = M
        self.decf = decf
        self.mle_chirplets()


    def mle_chirplets(self):
        """
        mle_chirplets estimate chirplets with maximum likelihood approach

        syntax:
        [P, e, res] = mle_chirplets(x, P, res, M, emits)

        inputs:
        x: signal
        P: vector of chirplet parameters (see make_chirplets.m)
        res: residual = norm(signal-chirplets.m)
        level = level of difficulty of MLE
        M: resolution for Newton-Raphson refinement (optional,
        mnits: maximum number of iterations (optional, default = 5)

        outputs:
        P: vector of chirplet parameters
        e: e = signal - chirplets
        res: residual = norm(signal-chirplets)
        """

        Q_ = np.size(self.P_, axis=0)  # the number of chirplets
        P0 = np.zeros([Q_, 5])  # initial values of chirplet parameters
        Ts = np.ones([Q_, 1]) * np.asarray([0.1, 0.001, 1e-5, .1])  # tolerance of parameter error

        # maximum-likelihood estimation (MLE)
        j = 1

        Pe = abs(self.P_[:, 1:5] - P0[:, 1:5])  # parameter error
        while sum(sum(Pe > Ts)) and j <= self.mnits and Q_ > 1:
            # save current as previous
            # P0 = copy.deepcopy(self.P)
            z = MakeChirplets(self.N, self.P_).sig
            # plt.figure()
            # plt.plot(z.real)
            # plt.show()

            # update current with delta residual
            for k in range(Q_):
                z_k = MakeChirplets(self.N, self.P_[k, :]).sig
                # plt.figure()
                # plt.plot(z_k.real)
                # plt.show()
                delta_k = self.x - (z - z_k)
                # delta_k_max = np.max(delta_k)-np.min()
                # delta_k_norm = delta_k/delta_k_max
                t_k = self.P_[k, 1]
                f_k = self.P_[k, 2]
                c_k = self.P_[k, 3]
                d_k = self.P_[k, 4]
                P_k = best_chirplet(delta_k, self.level, self.M, self.decf, t_k, f_k, c_k, d_k).p_
                P_k[0] = P_k[0]
                self.P_[k, :] = P_k

            Pe = abs(self.P_[:, 1:5] - P0[:, 1:5])
            j += 1
        # update the residual
        y = MakeChirplets(self.N, self.P_).sig
        self.err = self.x - y.flatten()
        self.P = self.P_
        # res = np.linalg.norm(err)
        # if res_




