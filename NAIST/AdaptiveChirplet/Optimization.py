'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: Optimization.py
@time: 12/10/2019 3:03 PM
'''

import numpy as np
from scipy.optimize import minimize, SR1, Bounds, BFGS
from make_chirplets import MakeChirplets
import warnings
import time

warnings.filterwarnings('ignore')


class FindBestChirpRate:
    def __init__(self, x, initialpoint, dboundary, hboundary):
        self.xx = x  # the signal being fitted
        self.N = len(x)
        self.initialpoint = initialpoint
        self.dboundary = dboundary  # down boundary
        self.hboundary = hboundary  # high boundary

        # self.f_points = []
        # self.fprime_points = []

    def fun(self, p):  # the target function
        t, f, c, d = p
        MakeChirplets_ = MakeChirplets(self.N, [1, t, f, c, d])
        # chirp = MakeChirplets_.sig
        fval = -abs(np.dot(np.conj(self.xx), MakeChirplets_.sig))**2
        # fval = -abs(self.xx.dot(MakeChirplets_.sig)) ** 2 # correlation index
        # self.f_points.append((t, f, c, d))
        return fval

    def fprime(self, p):
        t, f, c, d = p
        # self.fprime_points.append((t, f, c, d))

        n = np.arange(1, self.N + 1)
        MakeChirplets_ = MakeChirplets(self.N, [1, t, f, c, d])
        y = MakeChirplets_.sig.flatten()
        z_conj = np.conj(sum(self.xx * np.conj(y)))

        g = np.zeros(4)
        dz_dt = -sum(self.xx * np.conj(y) * ((n - t) / 2 / d ** 2 + 1j * c * (n - t) + 1j * f))
        g[0] = 2 * np.real(dz_dt * z_conj)

        dz_dt = -sum(self.xx * np.conj(y) * (-1j * (n - t)))
        g[1] = 2 * np.real(dz_dt * z_conj)

        dz_dt = -sum(self.xx * np.conj(y) * (-1j / 2 * (n - t) ** 2))
        g[2] = 2 * np.real(dz_dt * z_conj)

        dz_dt = -sum(self.xx * np.conj(y) * ((n - t) ** 2 / 2 / d ** 3 - 1 / 2 / d))
        g[3] = 2 * np.real(dz_dt * z_conj)

        return g

    def fmin(self, method):
        init_point = self.initialpoint
        bound = Bounds(self.dboundary, self.hboundary)
        res = minimize(self.fun,
                       init_point,
                       method=method,
                       jac=self.fprime,
                       hess='2-point',
                       )
        # return res, [np.array(points) for points in (self.f_points, self.fprime_points)]

        return res


if __name__ == '__main__':
    methods = ("Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B")
    # method = 'BFGS'
    x0 = [257, 5.301437602932776, 0.0, 17.88888888888889]
    vlb = [1, 0, -np.inf, 0.25]
    vub = [513, 6.283185307179586, np.inf, 500.0]
    xx = np.loadtxt('file')
    for method in methods:
        time1 = time.time()
        res = FindBestChirpRate(xx, x0, vlb, vub).fmin(method)
        timespent = time.time() - time1
        print(method, timespent, res.x)
