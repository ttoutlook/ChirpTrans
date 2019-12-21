'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: MakeChirplet.py
@time: 12/17/2019 8:16 PM
'''

# This program is used to make chirplet wavelet based on the parameters:

import numpy as np


class MakeChirplet:
    def __init__(self, N, P, PType='oneill'):
        self.N = N  # signal size
        self.P = np.asarray(P).reshape([-1, 5])  # Chirplet parameters [t, f, cr, d]
        self.Ptype = PType.lower()
        self.initial()

    def initial(self):
        if self.Ptype == 'cohen':
            self.cohen2oneill()
        # create chirplet wave
        self.sig = np.zeros(self.N, dtype=complex)
        for i in range(self.P.shape[0]):
            A, t, f, cr, d = self.P[i, :]
            wave = self.chirplet(t, f, cr, d)
            self.sig += wave * A

    def chirplet(self, t, f, cr, d):
        """
        create chirplet wave based on four parameters, namely,
        t-time center, f-frequency center, cr- chirplet rate, d- duration
        """
        rep = 10  # control the accuracy of discretization, using to discrete the continuous wavelets
        n = np.arange(1, self.N + 1)
        Chirplet = np.zeros(self.N, dtype=complex)
        for r in range(-rep, rep + 1):
            scale = np.exp(-((n + r * self.N - t) / 2 / d) ** 2)  # scaling function + time shift function
            chirp = np.exp(1j * cr / 2 * (n + r * self.N - t) ** 2)  # chirping function + time shift function
            fshift = np.exp(1j * f * (n + r * self.N - t))  # frequency shift function
            Chirplet += scale * chirp * fshift
        return Chirplet / np.linalg.norm(Chirplet)  # normalize the chirplet wave

    def cohen2oneill(self):
        """
        convert parameters from cohen style to oneill style
        """
        P_on = np.zeros_like(self.P)
        # amplitude
        P_on[:, 0] = self.P[:, 0]
        # t center
        P_on[:, 1] = self.P[:, 1] * self.N
        # f center
        P_on[:, 2] = self.P[:, 2]
        # chirp rate
        P_on[:, 3] = self.P[:, 3] / self.N
        # duration
        P_on[:, 4] = np.sqrt(self.N / self.P[:, 4] / 2)
        self.P = P_on


if __name__ == '__main__':
    # this is debugging test
    import matplotlib.pyplot as plt

    N = 100  # signal size
    # use cohen equation
    p_type = 'cohen'
    p1 = np.asarray([10, 1 / 2, np.pi / 2, np.pi, 1 / 18])

    p2 = np.asarray([10, 1 / 2, np.pi / 2, -np.pi, 1 / 18])
    p3 = [p1, p2]
    sig1 = MakeChirplet(N, p1, p_type)

    sig2 = MakeChirplet(N, p2, p_type)
    sig3 = MakeChirplet(N, p3, p_type).sig
    sig1 = sig1.sig
    sig2 = sig2.sig
    sig = sig1 + sig2
    plt.figure()
    plt.subplot(411)
    plt.plot(np.real(sig))
    plt.subplot(412)
    plt.plot(np.real(sig1))
    plt.subplot(413)
    plt.plot(np.real(sig2))
    plt.subplot(414)
    plt.plot(np.real(sig3))
