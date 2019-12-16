'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: scipylearning_1.py
@time: 12/9/2019 10:46 PM
'''
import numpy as np
from scipy import optimize


def target_function(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


class TargetFunction(object):
    def __init__(self):
        self.f_points = []
        self.fprime_points = []
        self.fhess_points = []

    def f(self, p):
        x, y = p.tolist()
        z = target_function(x, y)
        self.f_points.append((x, y))
        return z

    def fprime(self, p):
        x, y = p.tolist()
        self.fprime_points.append((x, y))
        dx = -2 + 2 * x - 400 * x * (y - x ** 2)
        dy = 200 * y - 200 * x ** 2
        return np.array([dx, dy])

    def fhess(self, p):
        x, y = p.tolist()
        self.fhess_points.append((x, y))
        return np.array([[2 * (600 * x ** 2 - 200 * y + 1), -400 * x],
                         [-400 * x, 200]])

    def fmin_demo(self, method):
        init_point = (-1, -1)
        res = optimize.minimize(self.f,
                                init_point,
                                method=method,
                                jac=self.fprime,
                                hess=self.fhess)
        return res, [np.array(points) for points in
                     (self.f_points, self.fprime_points, self.fhess_points)]



methods = ("Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG", "L-BFGS-B")
for method in methods:
    res, (f_points, fprime_points, fhess_points) = TargetFunction().fmin_demo(method)
    print(
        "{:12s}: min={:12g}, f count={:3d}, fprime count={:3d},fhess count={:3d}".format(
            method, float(res["fun"]), len(f_points), len(fprime_points), len(fhess_points)))