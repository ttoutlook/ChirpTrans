'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: MaxCorrChirplet.py
@time: 12/17/2019 10:00 PM
'''

import numpy as np
from scipy.signal import hilbert
from numpy import pi
from functionSet import functionSet as funs
import matplotlib.pyplot as plt


class MaxCorrelation():
    def __init__(self, x, D, i0, radix):
        self.x = x
        self.depth = D  # depth of chirplet levels
        self.i0 = i0  # the first level to rotate the chirplets
        self.radix = radix  # the radix of scale (default 2)
        self.code = []
        self.size = 1
        self.curpoint = 1
        self.corratio = []
        self.N = self.x.size
        self.inital()

    def inital(self):
        # intial the parameters:
        RE, params = self.mp_chirplet_bultan()
        # RE: the residual energy; params: corresponding parameters
        cc = np.abs(RE) / np.linalg.norm(self.x)  # cc: correlation coefficients

        # store data
        self.code = np.append(RE, params)
        self.corratio.append(cc)
        self.curpoint += 1

    def mp_chirplet_bultan(self):
        """
        mp_chirplet_bultan: implements matching-persuit with bultan chirplet atoms
        """
        self.levels = np.arange(0, self.depth - self.i0)  # the chirplet levels
        RE, atomid, td, fd = self.forloop()
        beta = [self.level, self.rot, td, fd]
        return RE, beta

    def forloop(self):
        nidx = self.i0 + sum(4 * self.radix ** (2 * self.levels) - 1)  # total number of atoms, from the level
        maxabs = 0
        for seq in range(1, nidx + 1):  # seq:  the chirplet id in chirplet set
            self.seq2ind(seq)  # get the corresponding scale and rotation from the sequence
            self.gaussian()  # construct gaussian chirplet atom at scale integral k and rotation integral m

            # calculate the circular correlation between gaussian atom and signal
            tc = np.argmax(np.abs(self.ccorr(self.x, self.atom.conj())))
            # time center
            # plt.figure()
            # plt.plot(self.atom.real)
            # plt.show()
            if tc != 0:
                tc = self.N - tc
            RE_projection = self.x * np.conj(np.roll(self.atom, tc))
            del self.atom
            RE_fft = np.fft.fft(RE_projection)
            RE_abs = np.abs(RE_fft)
            max_ = np.max(RE_abs)
            if max_ > maxabs:
                maxabs = max_
                seq_ = seq
        # after getting the best gaussian chirp atom, refine the frequency shift and time shift (frequency center and time center)
        self.seq2ind(seq_)
        self.gaussian()
        RE_projection = np.zeros([self.N, self.N], dtype=np.complex64)
        for tc in range(self.N):
            RE_projection[tc, :] = self.x * np.conj(np.roll(self.atom, tc))
        del self.atom
        RE_fft = np.fft.fft(RE_projection, axis=1)
        RE_abs = np.abs(RE_fft)
        tc, fc = np.where(RE_abs == np.max(RE_abs))
        RE = RE_fft[tc[0]][fc[0]]
        return RE, seq_, tc[0], fc[0]

    def gaussian(self):
        """
        gaussian: get guassian chirplet atom at certain scale and rotation angle
        """
        # main body
        self.getalpha()  # get the discrete angle = alpha
        self.getsigma()  # get the sigma -- the time domain angular scale
        self.getxi()  # get the xi -- chirp rate
        self.getgaussian()  # get the gaussian atom
        del self.alpha, self.sigma, self.xi

    def getgaussian(self):
        """
        gaussian atom
        """
        d = 5  # this control the accuracy of the gaussian window
        cg = np.zeros(self.N, dtype=np.complex64)
        r = np.arange(-d, d + 1) * self.N

        n = np.ones([self.N, r.size]).transpose() * np.arange(0, self.N)
        r = np.ones([self.N, r.size]) * r
        nr = n + r.transpose()
        cg = np.sum(np.exp(-pi / self.N * (1 / self.sigma ** 2 - 1j * self.xi) * nr ** 2), axis=0)
        # normalization
        g_km = cg / np.linalg.norm(cg).real
        self.atom = g_km.flatten()

    def getxi(self):
        """
        xi: get the chirp rate
        """
        s = self.radix ** self.level
        self.xi = ((s ** 4 - 1) * np.cos(self.alpha) * np.sin(self.alpha)) / (
                np.sin(self.alpha) ** 2 + s ** 4 * np.cos(self.alpha) ** 2)

    def getsigma(self):
        """
        sigma: the time domain angular scale
        sigma = sigkm(a^k,angm)
        """
        s = self.radix ** self.level
        self.sigma = np.sqrt(np.sin(self.alpha) ** 2 + s ** 4 * np.cos(self.alpha) ** 2) / s

    def getalpha(self):
        """
        get discrete rotating angle
        """
        self.alpha = np.arctan(self.rot / self.radix ** (2 * (self.level - self.i0)))

    def seq2ind(self, seq):
        """
        seq2ind: convert sequence of an atom to scale level and rotation (m) index
        """
        if seq <= self.i0:
            level = seq - 1
            rot = 0
        else:
            seq_ = seq - self.i0
            level_ = 0  # to detect the level of this seq
            getscale = False
            floor1 = 0
            while not getscale:  # when the seq is located in a range, the rotation is fixed.
                floor2 = floor1 + 4 * self.radix ** (2 * level_) - 1
                if floor1 < seq_ <= floor2:
                    level = level_ + self.i0
                    getscale = True
                else:
                    level_ += 1
                    floor1 = floor2
            # get the rotation
            level_arry = np.arange(0, level - self.i0)
            temp1 = self.i0 + sum(4 * self.radix ** (2 * level_arry) - 1)
            # add up the rotating angles of different level radix with the start level, and get the eventually waiting levels
            rot = seq - temp1 - 2 * self.radix ** (2 * (level - self.i0))  # get rotating level number
        self.level = level
        self.rot = rot

    def ccorr(self, a, b):
        '''
        Computes the circular correlation (inverse convolution) of the real-valued
        vector a with b.
        '''
        return self.cconv(np.roll(a[::-1], 1), b)
        # return self.cconv(a,b[::-1])

    def cconv(self, a, b):
        '''
        Computes the circular convolution of the (real-valued) vectors a and b.
        '''
        return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))


if __name__ == '__main__':
    import time

    sig = np.sin(np.arange(0, 1000))
    x = hilbert(sig)
    D = 5
    i0 = 1
    radix = 2
    time1 = time.time()
    clss = MaxCorrelation(x, D, i0, radix)
    print(time.time() - time1)
    print(clss.code)
    # print(clss.seq_test)
