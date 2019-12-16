import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt

def fft(sig, fs, axes):
    sig = sig - np.mean(sig)
    fsig = np.fft.fft(sig)
    lens = len(sig)
    fsig = abs(fsig) / lens * 2
    fsig = fsig[0:lens // 2]
    ft = np.linspace(0, 1, lens) * fs
    ft = ft[0:lens // 2]
    axes.plot(ft, fsig, 'k')


datafile = 'DS0049.csv'
data = pd.read_csv(datafile)[15:]
sig = np.asarray(data)[:,0:2]
time = np.asarray([np.float(sig[i,0]) for i in range(len(sig))])
fs = 1/(time[2]-time[1])
sig = np.asarray([np.float(sig[i,1]) for i in range(len(sig))])
sig = sig - np.mean(sig)
sig = signal.detrend(sig)
time = np.arange(0,len(time))*1/fs

plt.figure()
plt.plot(time,sig,'k')
plt.show()

fig, axes = plt.subplots()
fft(sig,fs,axes)
#plt.ylim([0, 0.09])
plt.xlim([0, 500])
plt.show()

f, t, zxx = signal.stft(sig,nfft=256)
plt.figure()
plt.pcolormesh(t, f*fs, np.abs(zxx))
plt.ylim([0,500])

wavename = 'cgau8'
totalscal = 512
fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscal
scales = cparam / np.arange(totalscal, 1, -1)
[cwtmatr, frequencies] = pywt.cwt(sig, scales, wavename, 1.0 / fs)
plt.figure()

plt.contourf(time,frequencies,np.abs(cwtmatr))
plt.ylim([0, 500])