'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: mp_chirplet.py
@time: 12/13/2019 3:28 PM
'''
import numpy as np
from make_chirplets import MakeChirplets
from numpy import pi
from Optimization import FindBestChirpRate
import matplotlib.pyplot as plt
from scipy.signal import stft, spectrogram
from max_chirpmpd import max_chirpmpd


class mp_chirplet:
    def __init__(self, x, M=64, D=5, i0=1, radix=2, methodid = 1):
        """
        # MP_CHIRPLET Estimate a chirplet that best fits the signal using Matching Pursuit algorithm
        inputs:
           x   - signal
           M   - resolution for Newton-Raphson refinement (optional, default is 64)
           D   - The depth of decomposition
           i0  - The first level to rotate the chirplets
           radix - radix of scale
        """
        self.norm = np.linalg.norm(x).real
        self.x = x / self.norm
        self.M = M
        self.D = D
        self.i0 = i0
        self.methodid = methodid
        self.radix = radix
        self.N = len(self.x)
        self.initial()


    def initial(self):
        # estimate the chirplet using MP
        # mping the chirplet...
        # initial the structure
        maxchirp = max_chirpmpd(self.x, self.D, self.i0, self.radix)
        A, k_idx, m_idx, q_idx, p_idx = maxchirp.code
        self.P_ori = [A, k_idx, m_idx, q_idx, p_idx]
        # k_idx: scale index
        # m_idx: rotation index
        # q_idx: time-shift index
        # p_idx: frequency-shift index
        self.bultan2params(k_idx, m_idx, q_idx, p_idx)
        # newton-raphson refinement
        self.P = self.nr_refine()

    def nr_refine(self):
        """
        refine the chirplet with newton-raphson method
        """

        # a longer window is useful here
        Z = 4
        t, f, c, d = self.P_ori[1:]
        rt = int(round(t))
        if rt - Z * self.M < 1 and rt + Z * self.M > self.N:
            xx = np.concatenate([np.zeros([Z * self.M - rt + 1]),
                                 self.x,
                                 np.zeros([Z * self.M - self.N + rt])])
        elif rt - Z * self.M < 1:
            xx = np.concatenate([np.zeros([Z * self.M - rt + 1]), self.x[0: rt + Z * self.M]])
        elif rt + Z * self.M > self.N:
            xx = np.concatenate([self.x[rt - Z * self.M - 1:], np.zeros([Z * self.M - self.N + rt])])
        else:
            xx = self.x[rt - Z * self.M - 1:rt + Z * self.M]

        xx = xx.flatten()
        x0 = [Z * self.M + 1 + (t - rt), f, c, d]
        vlb = [1, 0, -np.inf, 0.25]
        vub = [2 * Z * self.M + 1, 2 * pi, np.inf, self.N / 2]
        methods = ("Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B")
        p_ = FindBestChirpRate(xx.flatten(), x0, vlb, vub).fmin(methods[self.methodid]).x
        p_[0] = rt + p_[0] - (Z * self.M + 1)
        p_[1] = np.mod(p_[1].real, 2 * pi)
        sig = MakeChirplets(self.N, [1, p_[0], p_[1], p_[2], p_[3]]).sig.flatten()
        Amp = np.dot(sig.conj(), self.x)
        Amp = Amp * self.norm
        return np.append(Amp, p_)

    def bultan2params(self, k_idx, m_idx, q_idx, p_idx):
        # refer to bultan and Oneil for cr and dt
        angm = self.getalpham(k_idx, m_idx)
        s = self.radix ** k_idx  # scale_k
        sigma = np.sqrt(np.sin(angm) ** 2 + s ** 4 * np.cos(angm) ** 2) / s  # sigma(scale_k, ang_m)

        # time center = Tc, index --> second (assmue fs = 1 Hz)
        tc = q_idx
        # freq center = Fc, index --> rad
        fc = p_idx / self.N * 2 * pi
        # index --> chirp rate (Cr) and Duration (Dt)
        # chirp rate = Cr
        # xi = ((s^4-1)*cos(angm)*sin(angm))...
        # %     /(sin(angm)^2+s^4*cos(angm)^2);
        # % c = xi; % xi&sigma Bultan's definitions
        cr = 2 * pi / self.N * np.tan(angm)
        # time spread / duration = Dt
        # d = sigma/sqrt(2); % c&d O'Neil's definition
        dt = sigma * np.sqrt(self.N / pi / 2)
        self.P_ori[1:] = [tc, fc, cr, dt]

    def getalpham(self, k, m):
        """
        getalpham: calculate the discrete rotational angles
        # get the discrete angle=alpha_m
        """
        return np.arctan(m / self.radix ** (2 * (k - self.i0)))


if __name__ == '__main__':
    from scipy.signal import hilbert
    import time

    sig = np.sin(np.arange(0, 1000))
    x = hilbert(sig)
    D = 5
    i0 = 1
    radix = 2
    time1 = time.time()
    clss = mp_chirplet(x)
    print(time.time() - time1)
