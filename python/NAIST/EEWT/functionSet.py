'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: functionSet.py
@time: 12/12/2019 5:29 PM
'''

import numpy as np
import tftb.processing as tf
from MakeChirplet import MakeChirplet
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from numpy import pi
import numpy.matlib


class functionSet:
    def __init__(self, x=1, fs=1):
        self.x = x  # the original signal
        self.N = np.size(x)
        if self.N > 1:
            # self.norm = np.linalg.norm(self.x)
            # self.xnorm = x / self.norm
            self.xhil = hilbert(self.x)
            self.fs = fs

    def wgn(self, snr):
        if snr == np.inf:
            return self.x, []
        else:
            snr = 10 ** (snr / 10.0)
            xpower = np.sum(self.x ** 2) / self.N
            npower = xpower / snr
            noise = np.random.randn(self.N) * np.sqrt(npower)
            noisedata = self.x + noise
            return noisedata, noise

    def SNR(self, noisydata):
        # scale: dB
        signal = np.sum(self.x ** 2)
        noise = np.sum((self.x - noisydata) ** 2)
        snr = 10 * np.log10(signal / noise)
        return snr

    def show_decomp(self, P, n_points=128, id=3, method='explicit', plot=True):
        if id == 3:
            self.STFT_plot(n_points, plot)
            self.WVD_plot(n_points, plot)
            self.ACS_plot(P, method, plot)
        elif id == 0:
            self.STFT_plot(plot)
        elif id == 1:
            self.WVD_plot(plot)
        elif id == 2:
            self.ACS_plot(P, method, plot)

    def STFT_plot(self, n_points=128, plot=False):
        """
        plot the time-frequency distribution of signal by using short-time-frequency-transform
        """
        threshold = 0.05
        fmin, fmax = 0.0, 0.5 * self.fs
        stft = tf.ShortTimeFourierTransform(self.xhil)
        tfr, t, f = stft.run()
        if plot:
            stft.plot(kind='contour', extent=[0, n_points, fmin, fmax])
        return tfr, t, f

    def WVD_plot(self, n_points=128, plot=False):
        """
        plot the time-frequency distribution of signal by using Viger-ville distribution (WVD)
        """
        threshold = 0.05
        fmin, fmax = 0.0, 0.5 * self.fs
        wvd = tf.WignerVilleDistribution(self.xhil)
        tfr, t, f = wvd.run()
        if plot == True:
            wvd.plot(kind='contour', extent=[0, n_points, fmin, fmax])
        return tfr, t, f

    def ACS_plot(self, P, method='explicit', p_type='ONeill', plot=False):
        if p_type.lower() == 'oneill':
            fs = 1
            P = P
        else:
            fs = self.N
            P = self.PCo2On(P)
        if method.lower() == 'explicit':
            tfr, t, f = self.chirpltwvd_explicit(P, fs, plot)
        else:
            tfr, t, f = self.chirpltwvd_direct(P, fs, plot)
        return tfr, t, f

    def chirpltwvd_direct(self, P, fs=1, plot=False):
        """
        CHRPLTWVD_DIRECT compute chirplet spectrogram by estimating chriplet WVD directly
        """
        fs = fs  # sampling frequency set to 1 Hz
        P = np.asarray(P).reshape([-1, 5])
        Q = np.size(P, axis=0)  # the number of chirplets
        tfr = []
        for i in range(Q):
            chirplet_k = MakeChirplet(self.N, P[i, :]).sig.real
            wg = tf.WignerVilleDistribution(chirplet_k)
            tfr, t, f = wg.run()
            # tfr = np.sqrt(np.real(tfr*tfr.conj()))
            if i == 0:
                tfr = tfr
            else:
                tfr += tfr

        return tfr, t, f

    def chirpltwvd_explicit(self, P, fs=1, plot=False):
        """
        Chirpltwvd_explicit: compute chirplet spectrogram with explicit formula

        output:
        wig - length (n) by length(f) WVD
        """

        P = np.asarray(P).reshape([-1, 5])
        t = np.linspace(0, self.N, 2 * self.N)  # vector of time range (assume fs = 1Hz)
        f = np.linspace(0, 2 * pi, 2 * self.N)  # vector of frequency range [0, 1] range

        # f = f * 2 * pi  # convert to rad in [0 2*pi] range
        nchirp = np.size(P, axis=0)

        # nmat = np.matlib.repmat(t, m, l) # time matrix: m by l
        # fmat = np.matlib.repmat(f, l, l) # freq matrix m by l
        tmat, fmat = np.meshgrid(t, f)
        wig = np.zeros_like(tmat)
        for k in range(nchirp):
            # if P[k, 2].real > pi:
            #     P[k, 2] = np.mod(P[k, 2].real, pi)
            A_k, tc_k, fc_k, c_k, d_k = P[k, :].tolist()
            # WVD = w1 * w2 * w3
            #  w1 = a^2/pi
            #  w2 = exp(-(n-tc)^2/2/d^2)
            #  w3 = exp(-2*d^2*((f-fc)-c*(n-tc))^2)
            a = abs(A_k)
            acp = a ** 2 / pi
            bcp = np.exp(-(tmat - tc_k) ** 2 / 2 / d_k ** 2)
            ccp = np.exp(-2 * d_k ** 2 * ((fmat - fc_k) - c_k * (tmat - tc_k)) ** 2)
            w_k = acp * bcp * ccp

            wig = wig + w_k
        f_ = np.linspace(0, 1, 2 * self.N)
        if plot:
            plt.figure()
            self.contour(wig, t, f_)
            plt.show()
        return wig, t, f_

    def reconstruction(self, P):
        sig = MakeChirplet(self.N, P).sig.real
        flab, xfft = self.fft(sig)
        return sig, flab, xfft

    def fft(self, x, plot=False):
        xfft = np.fft.fft(hilbert(x))
        N = len(x.real)
        xfft = abs(xfft) / N * 2
        flab = np.linspace(0, self.fs / 2, N // 2)
        xfft = xfft[0:self.N // 2]
        if plot:
            plt.figure()
            plt.plot(flab, xfft)
            plt.xlabel('Frequency/Hz')
            plt.ylabel('Amplitude')
            plt.show()
        return flab, xfft

    def PCo2On(self, P):
        """
        PCO2ON convert chirplet parameters from Cohen to O'Neil format
        """
        # print(self.P.size)
        P = np.asarray(P).reshape([-1, 5])
        P_on = np.zeros_like(P)
        # amplitude
        P_on[:, 0] = P[:, 0]
        # t center
        P_on[:, 1] = P[:, 1] * self.N
        # f center
        P_on[:, 2] = P[:, 2]
        # chirp rate
        P_on[:, 3] = P[:, 3] / self.N
        # P_on.append(self.P[:, 3] / self.N)
        # duration
        P_on[:, 4] = np.sqrt(self.N / P[:, 4] / 2)
        return P_on

    def contour(self, tfr, t, f, P=0, title='', levels=8, half=True, reconstruction=False):
        ax1, ax2, ax3, fig = self.plotgrid()
        threshold = 0.05
        if half:
            N = int(np.ceil(len(t) / 2))
            tfr = tfr[:N, :]
            f = f[:N]
        if not title == '':
            fig.canvas.set_window_title(title)

        tfr = np.real(tfr * np.conj(tfr))
        tfr = np.sqrt(tfr)
        _threshold = np.amax(tfr) * threshold
        tfr[tfr <= _threshold] = 0.0
        # fig, ax = plt.subplots(figsize=(5, 4.5))
        # T, F = np.meshgrid(t, f)
        ax1.contourf(t, f, tfr, levels=30, cmap=plt.cm.jet)
        ax1.set_xlim([0, t[-1] * 1.004])
        ax1.set_ylim([0, f[-1] * 1.004])
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Normalized Frequency')
        # ax.clabel(cs)
        # ax1.grid()
        ax2.plot(self.x)

        flab, xfft = self.fft(self.x)
        ax3.plot(xfft.flatten(), flab.flatten())

        if reconstruction:
            sig, flab, xfft = self.reconstruction(P)
            ax2.plot(sig, 'r')
            ax3.plot(xfft.flatten(), flab.flatten(), 'r')

        plt.tight_layout()

    def On2PC(self, P):
        """
        PCO2ON convert chirplet parameters from O'Neil to Cohen format
        """
        P = np.asarray(P).reshape([-1, 5])
        PC = np.zeros_like(P)
        # amplitude
        PC[:, 0] = abs(P[:, 0])
        # t center
        PC[:, 1] = P[:, 1] / self.N
        # f center
        PC[:, 2] = P[:, 2]
        # chirp rate
        PC[:, 3] = P[:, 3] * self.N
        # P_on.append(self.P[:, 3] / self.N)
        # duration
        # P_on[:, 4] = np.sqrt(self.N / P[:, 4] / 2)
        PC[:, 4] = 2 * self.N / P[:, 4] ** 2
        return PC

    def plotgrid(self):
        fig = plt.figure(figsize=(6, 5))
        ax1 = plt.subplot2grid((4, 4), (0, 1), colspan=3, rowspan=3)
        ax2 = plt.subplot2grid((4, 4), (3, 1), colspan=3)
        ax3 = plt.subplot2grid((4, 4), (0, 0), rowspan=3)
        return ax1, ax2, ax3, fig

    def garbor1d(self, T, fs, dt, fc, tc, A, phi):
        """
        construct 1-D garbor wave
        Inputs:
        T: signal duration
        fs: sampling frequency
        dt: time-duration/ spread of the Gabor
        fc: frequency center of the garbor
        tc: time center of the garbor
        A: amplitude of the garbor (real number)
        phi: the phase of the Gabor wave (real number, in rad
        """
        # normalize parameters to samples (points):
        t = np.arange(0, T * fs)
        p = tc * fs
        w = dt * fs
        f = fc / fs

        # construct the signal
        if dt == T:  # sinusoid
            g = A * np.cos(2 * pi * f * t + phi)
        elif dt == 0:  # pulse
            g = np.zeros_like(t)
            g[int(round(p)) - 1] = A
        elif fc == 0:  # a gaussian
            g = A * np.exp(-((t - p) / w) ** 2)
        else:
            g = A * np.exp(-((t - p) / w) ** 2) * (np.cos(2 * pi * f * t + phi))
        return g

    def periodic_corr(self, x, y):
        """Periodic correlation, implemented using the FFT.

        x and y must be real sequences with the same length.
        """
        corr = np.fft.ifft(np.fft.fft(x) * np.fft.fft(y).conj())
        if np.sum(corr.imag) == 0:
            return corr.real
        else:
            return corr

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

    def plotSignal(self, x):
        plt.figure(figsize=(5, 3))
        tlab = np.arange(0, len(x)) / self.fs
        plt.plot(tlab, x)
        plt.show()

    def plotXY(self, x, y):
        plt.figure(figsize=(5, 5))
        plt.plot(x, y, 'k')
        plt.show()
