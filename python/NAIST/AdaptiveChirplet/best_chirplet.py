'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: best_chirplet.py
@time: 12/10/2019 8:49 PM
'''

import numpy as np
from make_chirplets import MakeChirplets
from numpy import pi
from Optimization import FindBestChirpRate
import matplotlib.pyplot as plt
from scipy.signal import stft, spectrogram
import tftb.processing as tf
import time
from scipy.io import loadmat
from scipy.io import matlab
from sklearn.preprocessing import MinMaxScaler


class best_chirplet:
    """
    best_chirplet find the best chirplet with maximum likelihood estimation
    """

    def __init__(self, x, level, M, decf, t_k, f_k, c_k, d_k, methodid=4):

        # self.x = x / self.norm
        # self.max = np.max(abs(x))
        # self.x = x / self.max
        self.norm = np.linalg.norm(x).real
        self.x = x / self.norm
        self.t = t_k
        self.f = f_k
        self.c = c_k
        self.d = d_k
        self.level = level
        self.M = M
        self.N = self.x.size
        self.decf = decf
        self.methodid = methodid

        r = 5  # robustness parameter
        if self.level == 2 or self.level == 3:
            self.est_cd_global(r)

        if self.level == 1 or self.level == 2:
            self.est_tf()  # estimate of the center in the time and frequency.
            self.f = np.mod(self.f, 2 * pi)  # calculate the frequency center
            self.est_c()  # estimate of the chirp rate
            self.est_d()  # estimate of the duration
        elif self.level == 3:
            for i in range(3):
                self.est_tf()
                self.f = np.mod(self.f, 2 * pi)
                self.est_c()
                self.est_d()

        self.QuasiNewtonMax()

    def QuasiNewtonMax(self):
        """
        # do a quasi-newton maximization on the windowed signal
        # a longer window is useful here.
        """
        Z = 4
        rt = int(round(self.t))
        # print(self.t)
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

        # xx_ = loadmat('xx.mat')['xx'].flatten()

        xx = xx.flatten()
        # plt.figure()
        # plt.plot(np.real(xx * np.conj(xx)))
        # plt.plot(np.real(xx_ * np.conj(xx_)))
        # plt.show()
        x0 = [Z * self.M + 1 + (self.t - rt), self.f, self.c, self.d]

        vlb = [1, 0, -np.inf, 0.25]
        vub = [2 * Z * self.M + 1, 2 * pi, np.inf, self.N / 2]
        methods = ("Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B")
        self.p_ = FindBestChirpRate(xx.flatten(), x0, vlb, vub).fmin(methods[self.methodid]).x
        self.p_[0] = rt + self.p_[0] - (Z * self.M + 1)
        self.p_[1] = np.mod(self.p_[1], 2 * pi)
        sig = MakeChirplets(self.N, [1, self.p_[0], self.p_[1], self.p_[2], self.p_[3]]).sig.flatten()
        # plt.figure()
        # plt.plot(sig.real)
        # plt.show()
        Amp = np.dot(sig.conj(), self.x)
        # Amp = sig.dot(self.x)
        Amp = Amp * self.norm
        self.p_ = np.append(Amp, self.p_)

    def est_d(self, dhigh=0, dlow=0.25):  # dhigh = M/2
        """
        est_d: estmiate the duration
        """
        M = self.M
        dhigh = M / 2 if dhigh == 0 else dhigh
        MM = int(self.M / 2)
        rt = int(round(self.N / 2))
        x_ = 0
        if M > self.N:
            xx = np.zeros([M, 1])
            t0 = MM - rt
            t1 = t0 + self.N - 1
            xx[t0 - 1:t1] = self.x
            x_ = xx

        elif M < self.N:
            t0 = rt - MM
            t1 = t0 + M - 1
            x_ = self.x[t0 - 1: t1]

        z = []
        dd = np.linspace(dlow, dhigh, M)
        for i in range(0, M):
            sig_ = MakeChirplets(M, [1, M / 2, self.f, self.c, dd[i]]).sig
            # sig_c = abs(sig_)*(sig_.real/abs(sig_.real))
            # plt.figure()
            # plt.plot(sig_c)
            # plt.show()
            z.append(sig_.flatten() / np.linalg.norm(sig_))

        z = np.asarray(z).transpose()
        # plt.figure()
        # z_ = np.real(z*np.conj(z))
        # plt.pcolor(z_)
        # plt.show()
        # y = loadmat('y.mat')['y']
        # x_test = loadmat('x_test.mat')['x']
        # A_ = x_.transpose().dot(z)
        A_ = np.dot(np.conj(x_).transpose(), z)
        A = np.abs(A_)
        # plt.figure()
        # plt.plot(A.flatten())
        # plt.show()
        ind = np.argmax(A)
        self.d = dd[ind]

    def est_c(self, method='wig'):
        """
        est_c: estimate the chirp rate from a local measure.
        """
        M = 4 * int(np.floor(self.M / 4))

        # center in time and window, window length = M
        # should probably use a non-rectangular window -> gaussian, hanmming etc.
        MM = int(M / 2)
        rt = int(round(self.N / 2))
        x_ = 0
        if M > self.N:
            xx = np.zeros([M, 1])
            t0 = MM - rt
            t1 = t0 + self.N - 1
            xx[t0 - 1:t1] = self.x
            x_ = xx

        elif M < self.N:
            t0 = rt - MM
            t1 = t0 + M - 1
            x_ = self.x[t0 - 1: t1]

        x_ = x_ * np.exp(-1j * np.arange(1, M + 1).transpose() * self.f)
        xfre = np.fft.fftshift(np.fft.fft(np.fft.fftshift(x_))) / np.sqrt(M)

        # estimate the chirp rate
        R1 = np.zeros([MM, 1])
        R2 = np.zeros([MM, 1])
        nn = np.arange(-M // 2, M // 2)

        for a in range(1, MM + 1):
            angle = a * 90 / MM - 45  # [-45, 45], M is the number of components of equally divided angle
            self.angle2cr(angle, M, 2 * pi)

            y = x_ * np.exp(-1j * self.c / 2 * nn ** 2)
            R1[a - 1] = abs(sum(y)) ** 2  # = \int W_y(t,0) dt

            Y = xfre * np.exp(-1j * self.c / 2 * nn ** 2)
            R2[a - 1] = M / 2 / pi * abs(self.c) * abs(sum(Y)) ** 2

        R = np.concatenate((R2[M // 4:], R1, R2[0:M // 4 - 1]), axis=0)
        a = np.argmax(R) + 1
        angle = a * 180 / M - 90
        self.angle2cr(angle, M, 2 * pi)

    def est_tf(self):
        """
        est_tf: estimate the location in time and frequency of the chirp.
        """
        # print(time.time() - time1)
        # x_ = self.x
        # x_ = abs(x_) * x_.real / abs(x_.real)
        # x_ = np.concatenate([np.zeros(self.M//2-1), self.x, np.zeros(self.M//2-1)])

        #
        # time1 = time.time()
        # f, t, S = spectrogram(self.x, 1, nperseg=self.M, noverlap=self.M - 1, nfft=self.M)
        st = tf.ShortTimeFourierTransform(self.x)
        S, t, f = st.run()


        # print(time.time()-time1)
        S = np.real(S * np.conj(S))
        # plt.figure()
        # plt.pcolor(S)
        # plt.show()
        f = np.asarray(f + 0.5)

        # S, f = self.tfdshift(S, f)
        t_ind = np.argmax(np.max(S, axis=0))
        t_ = t[t_ind]
        f_ind = np.argmax(S[:, t_ind])
        f_ = f[f_ind]


        # convert from sample number to real units
        self.t = t_ * self.decf + 1
        self.f = 2 * pi * f_ - pi

    def est_cd_global(self, r=0):
        """
        EST_CD_GLOBAL -- Estimate chirp rate and duration from a global measure.
        """
        xfre = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.x))) / np.sqrt(self.N)
        M = 4 * int(np.floor(self.M / 4))  # want M to be a multiple of 4
        MM = M // 2

        # estimate the chirp rate
        R1 = np.zeros([MM, 1])
        R2 = np.zeros([MM, 1])
        nn = np.arange(-np.floor(self.N / 2), -np.floor(self.N / 2) + self.N)

        for a in range(1, MM + 1):
            angle = a * 90 / MM - 45  # [-45, 45], M is the number of components of equally divided angle
            self.angle2cr(angle, self.N, 2 * pi)

            y = self.x * np.exp(-1j * self.c / 2 * nn ** 2)
            R1[a - 1] = sum(abs(np.fft.fft(y, axis=0) ** 4))

            Y = xfre * np.exp(-1j * self.c / 2 * nn ** 2)
            R2[a - 1] = self.N / 2 / pi * abs(self.c) * sum(abs(np.fft.fft(Y, axis=0) ** 4))

        R = np.concatenate((R2[M // 4:], R1, R2[0:M // 4 - 1]), axis=0)
        a = np.argmax(R) + 1
        angle = a * 180 / M - 90
        self.angle2cr(angle, self.N, 2 * pi)

        # estimate the duration
        y = self.x * np.exp(-1j * self.c / 2 * nn ** 2)
        # y = np.real(y)/abs(np.real(y)) * abs(y)

        ry = abs(np.correlate(y, y, mode='full'))
        ry = abs(ry[self.N + r - 1:]) ** 2

        z = np.zeros([self.N - r, M])
        dd = np.linspace(0.25, self.N / 4, M)
        for i in range(0, M):
            chirpClass = MakeChirplets(self.N - r, [1, 1 - r, 0, 0, dd[i]])
            sig_ = np.real(chirpClass.sig)
            z[:, i] = sig_.flatten() / np.linalg.norm(sig_)
        # A = ry.dot(z)
        A = np.dot(ry.conj(), z)

        ind = np.argmax(A)
        self.d = dd[ind]
        # find the d that gives the maximum of the likelihood function

    def angle2cr(self, angle, t, f):
        """
        angle2cr: convert an angle in the t-f plane to a chirp rate
        """
        self.c = f / t * np.tan(angle * 2 * pi / 360)



if __name__ == '__main__':
    from scipy.io import loadmat

    sig = loadmat('errData.mat')['e'].flatten()
    P = [1, pi / 2, 0, 1]
    t, f, c, d = P
    bestC_ = best_chirplet(sig, 2, 64, 1, t, f, c, d)
    plt.figure()
    plt.plot(sig.real)
    a, t, f, c, d = bestC_.p_.tolist()
    plt.plot(MakeChirplets(1000, [a, t, f, c, d]).sig.real)
    print(t)
