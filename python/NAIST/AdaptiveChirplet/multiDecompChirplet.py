'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: multiDecompChirplet.py
@time: 12/15/2019 3:47 PM
'''

"""
This simulation decomstrates the multi-component decomposition with MPEM alogrithm. This simulation signals are
proposed in Duraka and Blinowska
"""
import numpy as np
import matplotlib.pyplot as plt
from make_chirplets import MakeChirplets
from numpy import pi
from functionSet import functionSet as funs
from scipy.signal import hilbert
from mle_adpat_chirplets import mle_adapt_chirplets
from mp_adapt_chirplets import mp_adapt_chirplets
from scipy import signal

# construct and display the signals
# first signal length and sampling frequency
T = 512  # signal duration
fs = 1  # sampling frequency

# wave I consists of a sine(A), a sawtooth(B) and A gabor(C) waveform
waveI = np.zeros(T)
# componentA
dt_a = 46  # length of component A
tc_a = 90  # time_center of component A
fc_a = 1 / dt_a  # frequency center of component A
t = np.arange(0, dt_a)
s_a = -np.sin(2 * pi * fc_a * t / fs)

# component B
tc_b = 422
t1 = np.arange(0, dt_a / 2)
s_b1 = -(1 / 2 * signal.sawtooth(2 * pi * 2 * fc_a * t1 / fs, 1 / 2) + 1 / 2)
s_b2 = -s_b1
s_b = np.append(s_b1, s_b2)

# component C
dt_c = 28  # length of c
fc_c = 0.4  # frequency center of C
tc_c = 256  # time center of C
s_c = funs().garbor1d(T, fs, dt_c, fc_c, tc_c, 1, 0)  # signal C

# construct wave I
ind1a = int(round(tc_a - dt_a / 2) - 1)
ind1b = int(round(tc_b - dt_a / 2) - 1)
ind2a = int(round(tc_a - dt_a / 2) + dt_a - 1)
ind2b = int(round(tc_b - dt_a / 2) + dt_a - 1)
waveI[ind1a:ind2a] = s_a
waveI[ind1b:ind2b] = s_b
waveI = waveI + s_c
# plt.figure()
# plt.plot(waveI)
# plt.show()

# wave II is a Gabor waveform
dt_d = 2 * dt_c  # length of D
fc_d = 2 * fc_c / 3  # frequency center of D
tc_d = 256  # time center of D
s_d = funs().garbor1d(T, fs, dt_d, fc_d, tc_d, 1, 0)
waveII = s_d

# wave III consists of a pulse (E) and a sinusoidal (F) waveform
tc_e = 128  # component E
s_e = funs().garbor1d(T, fs, 0, 0, tc_e, 2, 0)
fc_f = 0.35  # component F
A_f = 0.2
t2 = np.arange(0, T * fs)
s_f = A_f * np.sin(2 * pi * fc_f * t2 / fs)
waveIII = s_e + s_f
# plt.figure()
# plt.plot(waveIII)
# plt.show()

# wave IV is a upward chirplet (G)
A_cp = 6  # amplitude, total energy of the chirplet
tc_cp = 350  # time center
fc_cp = 0.2 * 2 * pi / fs
cr = pi / T  # chirp rate
dt_cp = 70  # size of the chirplet
P = [A_cp, tc_cp, fc_cp, cr, dt_cp]
cp = MakeChirplets(T, P).sig  # chirplets out is a complex signal
waveIV = cp.real

# finally, we add all the components together
chirpsim = waveI + waveII + waveIII + waveIV
fig = plt.figure()
plt.subplot(211)
plt.plot(chirpsim)

# wave I,II, and III are presented in Durka and Blinowska 1995
durkas = waveI + waveII + waveIII
plt.subplot(212)
plt.plot(durkas)
plt.show()
fig.canvas.set_window_title('Analyzed Signal')

# show components
waves = [waveI, waveII, waveIII, waveIV]
fig, axes = plt.subplots(4, 1)
for i in range(4):
    axes[i].plot(waves[i])
plt.tight_layout()
fig.canvas.set_window_title('Components')

# STFT
tfr, t, f = funs(chirpsim).STFT_plot(plot=False)
title = 'STFT -- clean chirpsim'
funs(chirpsim).contour(tfr, t, f, title=title)

spn, ns = funs(chirpsim).wgn(0)
title = 'STFT -- noisy chirpsim'
tfr, t, f = funs(spn).STFT_plot(plot=False)
funs(chirpsim).contour(tfr, t, f, title=title)

# Wigner
tfr, t, f = funs(chirpsim).WVD_plot(plot=False)
title = 'WVD -- clean chirpsim'
funs(chirpsim).contour(tfr, t, f, title=title)

title = 'WVD -- noisy chirpsim'
tfr, t, f = funs(spn).WVD_plot(plot=False)
funs(chirpsim).contour(tfr, t, f, title=title)

# perform adaptive chirplet decomposition with MLE
# set the parameters
Q = 7  # number of atoms desired
i0 = 1  # the first scale to rotate the atoms
D = 5  # decomposition depth = the highest scale
a = 2  # the radix of scale
M = T  # resolution for newton-raphson refinement

# estimation with MLE ACT
title = 'MLE ACT -- noisy chirpsim'
tests = hilbert(spn)
p_mle = mle_adapt_chirplets(tests, Q, methodid=4).P
tfr, t, f = funs(spn).ACS_plot(p_mle)
funs(spn).contour(tfr, t, f, P=p_mle, reconstruction=True, title=title)

title = 'MLE ACT -- clean chirpsim'
tests = hilbert(chirpsim)
p_mle = mle_adapt_chirplets(tests, Q, methodid=4).P
tfr, t, f = funs(chirpsim).ACS_plot(p_mle)
funs(chirpsim).contour(tfr, t, f, P=p_mle, reconstruction=True, title=title)

# estimation with EM ACT
title = 'EM ACT -- noisy chirpsim'
tests = hilbert(spn)
p_mle = mp_adapt_chirplets(tests, Q, methodid=2).P
tfr, t, f = funs(spn).ACS_plot(p_mle)
funs(spn).contour(tfr, t, f, P=p_mle, reconstruction=True, title=title)

title = 'EM ACT -- clean chirpsim'
tests = hilbert(chirpsim)
p_mp = mp_adapt_chirplets(tests, Q, methodid=2).P
tfr, t, f = funs(chirpsim).ACS_plot(p_mp)
funs(chirpsim).contour(tfr, t, f, P=p_mp, reconstruction=True, title=title)
