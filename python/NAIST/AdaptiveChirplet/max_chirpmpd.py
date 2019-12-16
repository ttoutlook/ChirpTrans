'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: max_chirpmpd.py
@time: 12/13/2019 5:39 PM
'''

import numpy as np
import copy
from scipy.signal import hilbert
from numpy import pi
import time
import torch
from scipy.fftpack import fft


class max_chirpmpd:
    def __init__(self, x, D, i0, radix):
        L = 1
        self.x = x
        self.D = D
        self.i0 = i0
        self.radix = radix
        self.code = []
        self.size = L
        self.curpoint = 1
        self.corratio = []
        self.N = len(self.x)
        self.initial()
        del self.x, self.D, self.i0, self.N, self.size, self.radix

    def initial(self):
        # initialization of the residual energy
        # Rf0 is the energy residue at step 0, initialization, which should be equal to the signal itself.
        # note that signal(x) can be a complex signal

        # self.Rf0 = copy.deepcopy(self.x)  # RF0 stands for the residue energy at each iteration

        # MP of chirpmpd
        Rf0gbetal, beta1 = self.mp_chirplet_bultan()
        cc = np.abs(Rf0gbetal) / np.linalg.norm(self.x)  # cc - correlation coefficients

        # store data
        # curp = self.curpoint
        self.code = np.append(Rf0gbetal, beta1)
        self.corratio.append(cc)
        self.curpoint += 1

    def mp_chirplet_bultan(self):
        """
        MP_CHIRPLET_BULTAN Implements Matching-pursuit with Bultan chirplet atoms
        """
        # parameter setting--loop1
        # self.r = np.arange(0, self.D - self.i0)  # total # of levels
        # Rf0_g = self.forloop()
        # aR = np.abs(Rf0_g)
        # max_ = np.max(aR)
        # seqs, id1, id2 = np.where(aR == max_)
        # Rf0gbetal = Rf0_g[seqs[0]][id1[0]][id2[0]]
        # self.seq2idx(seqs[0] + 1)
        # beta1 = [self.k, self.m, id1[0], id2[0]]

        # loop2:
        self.r = np.arange(0, self.D - self.i0)  # total # of levels
        Rf0gbetal, seq, id1, id2 = self.forloop1()
        self.seq2idx(seq + 1)
        beta1 = [self.k, self.m, id1, id2]
        del self.k, self.m, self.r
        return Rf0gbetal, beta1

        # max_fre = np.argmax(np.max(aR, axis=0))
        # max_del = np.argmax(aR[max_fre,:])
        # if max_ > maxabscoe:
        #     maxabscoe = max_
        #     self.Rf0gbetal = Rf0_g[id1[0]][id2[0]]
        #     self.beta1 = [self.k, self.m, id1[0], id2[0]]
        # del self.k, self.m, self.r
        # return Rf0gbetal, beta1

    def forloop(self):
        nidx = self.i0 + sum(4 * self.radix ** (2 * self.r) - 1)  # total number of atoms, from the level
        Rf0_gkmq = np.zeros([nidx, self.N, self.N], dtype=np.complex64)
        for seq in range(1, nidx + 1):  # for each chirplet in the dictionary
            self.seq2idx(seq)  # get index of scale and rotation from the sequence
            self.gkmn()  # gkmn: gaussian chirplet atom at scale k and rotation m
            # gkms[seq - 1, :] = self.g_km
            for q in range(self.N):
                Rf0_gkmq[seq - 1, q, :] = self.x * np.conj(np.roll(self.g_km, q))
            del self.g_km
        Rf0_g = np.fft.fft(Rf0_gkmq, axis=2)
        # fft compute the last axis # using fft to compute the correlation between signal and gaussian chirplet atom
        return Rf0_g

    def forloop1(self):
        nidx = self.i0 + sum(4 * self.radix ** (2 * self.r) - 1)  # total number of atoms, from the level
        Rf0_gkmq = np.zeros([self.N, self.N], dtype=np.complex64)
        maxabs = 0
        for seq in range(1, nidx + 1):  # for each chirplet in the dictionary
            self.seq2idx(seq)  # get index of scale and rotation from the sequence
            self.gkmn()  # gkmn: gaussian chirplet atom at scale k and rotation m
            # gkms[seq - 1, :] = self.g_km
            for q in range(self.N):
                Rf0_gkmq[q, :] = self.x * np.conj(np.roll(self.g_km, q))
            del self.g_km
            # Rf0_gkmq = self.x[np.newaxis,:]*np.conj(Rf0_gkmq)
            Rf0_g = np.fft.fft(Rf0_gkmq, axis=1)
            aR = np.abs(Rf0_g)
            max_ = np.max(aR)
            if max_ > maxabs:
                maxabs = max_
                id1, id2 = np.where(aR == max_)
                Rf0gbetal = Rf0_g[id1[0]][id2[0]]
                seq_ = seq
        return Rf0gbetal, seq_, id1[0], id2[0]

        # if np.max(aR) == 0:
        # fft compute the last axis # using fft to compute the correlation between signal and gaussian chirplet atom
    def forloop2(self):
        nidx = self.i0 + sum(4 * self.radix ** (2 * self.r) - 1)  # total number of atoms, from the level
        Rf0_gkmq = np.zeros([self.N, self.N], dtype=np.complex64)
        maxabs = 0
        for seq in range(1, nidx + 1):  # for each chirplet in the dictionary
            self.seq2idx(seq)  # get index of scale and rotation from the sequence
            self.gkmn()  # gkmn: gaussian chirplet atom at scale k and rotation m
            # gkms[seq - 1, :] = self.g_km
            for q in range(self.N):
                Rf0_gkmq[q, :] = self.x * np.conj(np.roll(self.g_km, q))
            del self.g_km
            # Rf0_gkmq = self.x[np.newaxis,:]*np.conj(Rf0_gkmq)
            Rf0_g = np.fft.fft(Rf0_gkmq, axis=1)
            aR = np.abs(Rf0_g)
            max_ = np.max(aR)
            if max_ > maxabs:
                maxabs = max_
                id1, id2 = np.where(aR == max_)
                Rf0gbetal = Rf0_g[id1[0]][id2[0]]
                seq_ = seq
        return Rf0gbetal, seq_, id1[0], id2[0]

    def gkmn(self):
        """
        GKMN get gaussian chirplet atom at certain scale and rotation angle
        Output:
        g: the gaussian chirplet atom
        """

        # main function

        # main body
        self.getalpha()  # get the discrete angle = alpha_m
        self.sigkm()  # current sigma (scale_k, ang_m)
        self.xkm()  # current x (scale_k, ang_m)
        self.gkm()  # g(sig_km,xi_km,n)
        del self.ang_m, self.csig, self.cx

    def gkm(self):
        # Gussain atom = gkm(csig,cx,N)
        csig = self.csig
        cx = self.cx
        N = self.N

        d = 5  # this control the accuracy of the gaussian window
        cg = np.zeros(N, dtype=np.complex64)
        r = np.arange(-d, d + 1) * N

        n = np.ones([N, r.size]).transpose() * np.arange(0, N)
        r = np.ones([N, r.size]) * r
        nr = n + r.transpose()
        cg = np.sum(np.exp(-pi / N * (1 / csig ** 2 - 1j * cx) * nr ** 2), axis=0)
        # normalization
        g_km = cg / np.linalg.norm(cg).real
        self.g_km = g_km.flatten()

    def xkm(self):
        # xi = xkm(a^k,angm)
        s = self.radix ** self.k
        angm = self.ang_m
        self.cx = ((s ** 4 - 1) * np.cos(angm) * np.sin(angm)) / (np.sin(angm) ** 2 + s ** 4 * np.cos(angm) ** 2)

    def sigkm(self):
        # sigma = sigkm(a^k,angm)
        s = self.radix ** self.k
        angm = self.ang_m
        self.csig = np.sqrt(np.sin(angm) ** 2 + s ** 4 * np.cos(angm) ** 2) / s

    def getalpha(self):
        """
        getalpha: calculate the discrete rotational angles
        """
        # parse input
        self.ang_m = np.arctan(self.m / self.radix ** (2 * (self.k - self.i0)))

    def seq2idx(self, seq):
        """
        SEQ2IDX convert sequence of an atom to scale and rotation indexes
        """
        if seq <= self.i0:
            k = seq - 1
            m = 0
        else:
            d = seq - self.i0
            r = 0
            findk = False
            t1 = 0
            while not findk:
                t2 = t1 + 4 * self.radix ** (2 * r) - 1
                if t1 < d <= t2:
                    k = r + self.i0
                    findk = True
                else:
                    r = r + 1
                    t1 = t2
            # get m
            r = np.arange(0, k - self.i0)
            t = self.i0 + sum(4 * self.radix ** (2 * r) - 1)
            m = seq - t - 2 * self.radix ** (2 * (k - self.i0))
        self.k = k
        self.m = m


if __name__ == '__main__':
    sig = np.sin(np.arange(0, 1000))
    x = hilbert(sig)
    D = 5
    i0 = 1
    radix = 2
    time1 = time.time()
    clss = max_chirpmpd(x, D, i0, radix)
    print(time.time() - time1)

