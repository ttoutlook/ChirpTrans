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


class ParamsTrans:
    def __init__(self, params, N, Q):
        self.params = np.asarray(params).reshape([-1, 5])  # the chirplet parameters
        self.N = N  # the number of data
        self.Q = Q  # the number of components
        self.initial()

    def initial(self):
        paramsOneill = self.Oneill2Cohen(self.params)
        constfre = []  # constant frequency components
        varyfre = []  # Variable frequency components
        for i in range(self.Q):
            A, t, f, cr, d = paramsOneill[i, :].tolist()

            print('Component No.', i + 1)
            print('t:', t, 'f:', f)
            print('cr:', cr, 'd:', d)
            # f: frequency center
            # cr: chirplet angle in oneill format,

    def Oneill2Cohen(self, P):
        """
        PCO2ON convert chirplet parameters from Cohen to O'Neill format
        """
        # print(self.P.size)
        P_on = np.zeros_like(P)
        # amplitude
        P_on[:, 0] = P[:, 0]
        # t center
        P_on[:, 1] = P[:, 1] #
        # f center
        P_on[:, 2] = P[:, 2] / np.pi / 2
        # chirp rate
        P_on[:, 3] = P[:, 3] * self.N / np.pi
        # P_on.append(self.P[:, 3] / self.N)
        # duration
        # P_on[:, 4] = np.sqrt(self.N / P[:, 4] / 2)
        # P_on[:, 4] = self.N / (2 * P[:, 4] ** 2)
        P_on[:,4] = P[:, 4]
        return P_on

    # def params2bultan(self, tc, fc, cr, dt):
    #     """
    #     present params in bultan format
    #     """
    #     # time center (assume fs = 1 Hz)
    #     t = tc
    #     # frequency center (rad)
    #     # fc = fc_ / self.N * 2 * pi
    #     angle = fc * self.N * 2 * np.pi
    #     # chirp rate
    #     cr = 2 * pi / self.N * np.tan(alpha)
    #     # time spread / duration
    #     dt = sigma * np.sqrt(self.N / pi / 2)
    #     self.Param[1:] = [tc, fc, cr, dt]
