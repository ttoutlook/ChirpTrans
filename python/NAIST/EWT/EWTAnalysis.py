'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: EWTAnalysis.py
@time: 11/11/2019 5:55 PM
'''

import numpy as np
import matplotlib.pyplot as plt
import ewtpy
import scipy.io as sio
from scipy.signal import hilbert, find_peaks
from scipy.signal import decimate
import pandas as pd


def fft(data, fs):
    datafft = abs(np.fft.fft(hilbert(data)))
    ffthalf = datafft[0:len(data) // 2] / len(datafft)
    fslab = np.linspace(0, 0.5, len(ffthalf)) * fs
    return ffthalf, fslab


# T = 1000
# t = np.arange(1, T + 1) / T
# f = np.cos(2 * np.pi * 0.8 * t) + 2 * np.cos(2 * np.pi * 10 * t) + 0.8 * np.cos(2 * np.pi * 100 * t)
# ewt, mfb, boundaries = ewtpy.EWT1D(f, N=3)
# plt.plot(f)
# plt.plot(ewt)

data = sio.loadmat('datafile.mat')
data1 = data['softmaterial_160']

fs = 1 / (data1[2, 0] - data1[1, 0])
# time = np.arange(0, len(data1)) * 1 / fs
wave = data1[:, 1]
ind = 1
wave = decimate(wave, ind, ftype='iir', zero_phase=True)
fs = fs/ind
time = np.arange(0, len(wave)) * 1 / fs
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(time * 1e6, wave, 'k')
plt.ylabel('Amplitude/mV')
plt.xlabel('Time/$\mu$s')
# plt.show()
ffthalf, fslab = fft(wave, fs / 1e6)
plt.subplot(2, 1, 2)
plt.plot(fslab, ffthalf, 'k')
plt.xlabel('Frequency\MHz')
plt.ylabel('Amplitude\mV')
plt.xlim([0, 2])
plt.tight_layout()

ewt, mfb, boundaries = ewtpy.EWT1D(wave, N=11)
plt.subplot(2, 1, 2)
num = np.size(ewt, 1)
fig, axes = plt.subplots(num, 2)
legends = ['A','B','C','D']
# axes = axes.flatten()
for i in range(num):
    axes[i, 0].plot(time * 1e3, ewt[:, i])
    ffthalf, fslab = fft(ewt[:, i], fs / 1e6)
    ind, _ = find_peaks(ffthalf, height=max(ffthalf) * 0.99)
    print("Components Frequency", fslab[ind])
    axes[i, 1].plot(fslab, ffthalf)
    # axes[i, 1].set_xlim([0, 2])

    # df1 = pd.DataFrame()
    # df2 = pd.DataFrame()
    # df1[legends[i],'_Time/us'] = time * 1e6
    # df1[legends[i],'_Amplitude/mV'] = ewt[:, i] * 1e3
    # df2[legends[i],'_Frequency/MHz'] = fslab / 1e6
    # df2[legends[i],'_Freq_Amplitude/mV'] = ffthalf
    # df1.to_csv(legends[i]+'_components.csv')
    # df1.to_csv(legends[i]+'_Frequency.csv')
plt.show()
fig.set_size_inches(6.5 * 1.3, 3.5 * 2)
plt.tight_layout()
i = 1
axes1 = axes.flatten()
for ax in axes1:
    ax.set_ylabel('Amplitude/mV')
    if i == 1:
        ax.set_xlabel('Time/$\mu$s')
        i = 0
    else:
        ax.set_xlabel('Frequency/MHz')
        i = 1
plt.tight_layout()

# df = pd.DataFrame()
# df['A_Time/us'] = time*1e6
# df['A_Amplitude/mV'] = ewt[:, 0]*1e3
# df['A_Frequency/MHz'] = fslab/1e6
# df['A_Freq_Amplitude/mV'] = ffthalf
# df.to_csv('A_components.csv')
import tftb.processing as tf
from functionSet import functionSet as funs
funs(wave,fs =fs).STFT_plot(plot=True)
# wigner.run()
