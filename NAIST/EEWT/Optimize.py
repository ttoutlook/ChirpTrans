'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: Optimize.py
@time: 12/17/2019 9:11 PM
'''
# this program is used to optimize the parameters of chirplet wavelets

import numpy as np
from MakeChirplet import MakeChirplet
import warnings
from scipy.optimize import minimize, Bounds

warnings.filterwarnings('ignore')


class FindBestChirpletRate:
    def __init__(self, x, initialpoint, dboundary, hboundary):
        self.xx = x  # the signal being fitted
        self.N = x.size
        self.initialpoint = initialpoint
        self.dboundary = dboundary  # the low boundary
        self.hboundary = hboundary  # the up boundary

    def fun(self, p):  # the target function
        t, f, cr, d = p
        sig = MakeChirplet(self.N, [1, t, f, cr, d]).sig
        fval = -abs(np.dot(np.conj(self.xx), sig)) ** 2
        return fval

    def fprime(self, p):
        t, f, cr, d = p

        n = np.arange(1, self.N + 1)
        y = MakeChirplet(self.N, [1, t, f, cr, d]).sig.flatten()
        z_conj = np.conj(sum(self.xx * np.conj(y)))

        g = np.zeros(4)
        dz_dt = -sum(self.xx * np.conj(y) * ((n - t) / 2 / d ** 2 + 1j * cr * (n - t) + 1j * f))
        g[0] = 2 * np.real(dz_dt * z_conj)

        dz_df = -sum(self.xx * np.conj(y) * (-1j * (n - t)))
        g[1] = 2 * np.real(dz_df * z_conj)

        dz_dc = -sum(self.xx * np.conj(y) * (-1j / 2 * (n - t) ** 2))
        g[2] = 2 * np.real(dz_dc * z_conj)

        dz_dd = -sum(self.xx * np.conj(y) * ((n - t) ** 2 / 2 / d ** 3 - 1 / 2 / d))
        g[3] = 2 * np.real(dz_dd * z_conj)
        return g

    def fmin(self, method):
        init_point = self.initialpoint
        bound = Bounds(self.dboundary, self.hboundary)
        res = minimize(self.fun,
                       init_point,
                       method=method,
                       bounds=bound,
                       jac=self.fprime,
                       hess='2-point')
        return res


if __name__ == '__main__':
    import time

    methods = ("Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B")
    # method = 'BFGS'
    x0 = [257, 5.301437602932776, 0.0, 17.88888888888889]
    vlb = [1, 0, -np.inf, 0.25]
    vub = [513, 6.283185307179586, np.inf, 500.0]
    xx = np.loadtxt('file')
    for method in methods:
        time1 = time.time()
        res = FindBestChirpletRate(xx, x0, vlb, vub).fmin(method)
        timespent = time.time() - time1
        print(method, timespent, res.x)
