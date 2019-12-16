'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: ChirpletTransform.py
@time: 12/6/2019 2:32 PM
'''

import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt


def Chirplet_Transform(sig, alpha, fLevel=512, WinLen=64, SampFreq=1000):
    """
    # this function is used to calculate the chirplet transform of the signal
    -----------input---------------------
    sig: one-dimensional signal sequence to be analyzed
    fLevel: frequency axis points associated with the Spec (in Bins)
    WinLen: Gauss window width (in Bins)
    SampFreq: sampling frequency (in Hz)
    alpha: chirplet's line frequency modulation rate (in Hz/s)

    ----------output---------------------
    Spec: 2D spectrum results (horizontal time axis, vertical frequency axis
    Freq: vertical frequency axis (in Hz)
    t: horizontal time axis (in Sec)
    """

    # data preparation
    SigLen = sig.size
    t = np.arange(0, SigLen) / SampFreq
    sig = hilbert(sig)  # calculate the analytical signal

    # frequency axis and its half-axis points
    fLevel = np.ceil(fLevel / 2) * 2 + 1
    # round the frequency axis length value fLevel in one direction to the nearest odd number
    Lf = (fLevel - 1) / 2
    # half frequency axis data points (fLevel has been rounded to an odd number)
    # generate gauss window functions
    WinLen = np.ceil(WinLen / 2) * 2 + 1
    # round the length of windows to a nearest odd
    WinFun = np.exp(-6 * np.linspace(-1, 1, WinLen) ** 2)
    # gauss window function, divided into WinLen modulation data points
    Lw = (WinLen - 1) / 2  # half window data points
    # CT spectrum result array ( initialized to 0 )
    spec = np.zeros([int(fLevel), int(SigLen)])

    # sliding window signal, data segmentation preparation
    for iLoop in range(1, SigLen + 1):
        # determine the upper and lower limits of the left and right signal index subscripts
        # note that to prevent the edge width from exceeding the time domain, the retrieval error!
        iLeft = min([iLoop - 1, Lw, Lf])
        iRight = min([SigLen - iLoop, Lw, Lf])
        iIndex = np.arange(-iLeft, iRight + 1, dtype='int')

        iIndex1 = iIndex + iLoop  # subscript index of the orignal signal

        iIndex2 = iIndex + int(Lw) + 1  # subscript index of window function vector
        Index = iIndex + int(
            Lf) + 1  # subscript index of the frequency axis (row number) in the spec two-dimensional array

        R_operator = np.exp(
            -1j * 0.5 * 2 * np.pi * alpha * t[iIndex1 - 1] ** 2)  # frequency rotation operator (shear frequency)
        S_operator = np.exp(
            1j * 2 * np.pi * alpha * t[iIndex1 - 1] * t[iLoop - 1])  # frequency shift operator (shift frequency)

        Sig_Section = sig[iIndex1 - 1] * R_operator * S_operator
        # signal segment after frequency rotation and frequency shift
        spec[iIndex - 1, iLoop - 1] = Sig_Section * np.conj(WinFun[iIndex2 - 1])  # fill the two-dimensional array

    # STFT
    spec = np.fft.fft(spec, axis=0)
    # spec = np.transpose(spec)
    spec = spec * 2 / fLevel  # restores the true magnitude of FT
    spec = spec[0:int(fLevel) // 2, :]  # till the Nyquist frequency

    freq = np.linspace(0, 0.5 * SampFreq, fLevel // 2)  # Output frequency axis for plotting

    return spec, freq, t


if __name__ == '__main__':
    fs = 200
    t = np.arange(0, 15 + 1 / fs, 1 / fs)

    sig = np.sin(2 * np.pi * (10 + 2.5 * t) * t) + np.sin(2 * np.pi * (12 + 2.5 * t) * t)
    IF1 = 10 + 5 * t
    IF2 = 12 + 5 * t

    plt.figure(figsize=(6, 3))
    plt.subplot(211)
    plt.plot(t, sig, 'k')
    plt.xlabel('Time/Sec', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.subplot(212)
    spec = abs(np.fft.fft(sig)) * 2 / sig.size
    spec = spec[0:sig.size // 2]
    freq = np.arange(0, sig.size // 2) / fs
    plt.plot(freq, spec, 'k')
    plt.xlabel('Frequency/Hz', fontsize=10)
    plt.ylabel('Amplitude', fontsize=10)
    plt.tight_layout()
    plt.show()

    # CT
    fLevel = 1024
    WinLen = 512
    alpha = 5  # in Hz/s
    spec, freq, _ = Chirplet_Transform(sig=sig, alpha=alpha, fLevel=fLevel, WinLen=WinLen, SampFreq=fs)
    plt.figure()
    plt.contourf(t, freq, abs(spec), cmap='jet')
    plt.xlabel('Time/Sec', fontsize=10)
    plt.ylabel('Frequency/Hz', fontsize=10)
    plt.show()
