'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: DecomposeCrossChirp.py
@time: 12/12/2019 5:18 PM
'''

"""
This simulation copares the results of MPEM and MLE algorithm for the estimation of the components of a signal
consisting of an upward and downward chirplet, embedded in noise.
"""
import numpy as np
import matplotlib.pyplot as plt
from make_chirplets import MakeChirplets
from numpy import pi
from functionSet import functionSet as funs
import tftb
from scipy.signal import hilbert
from mle_adpat_chirplets import mle_adapt_chirplets
from mp_adapt_chirplets import mp_adapt_chirplets

# create and display the simulated signal
N = 500  # signal length

# use cohen equation
p_type = 'Cohen'
P1 = [10, 1 / 2, pi / 2, pi, 1 / 18]  # up-chirplet 0 -> pi
P2 = [10, 1 / 2, pi / 2, -pi, 1 / 18]  # down-chirplet pi -> 0
s1 = np.real(MakeChirplets(N, P1, 'PType', p_type).sig)  # the synthesized signal
s2 = np.real(MakeChirplets(N, P2, 'PType', p_type).sig)  # the synthesized signal
s = s1 + s2

# add guassian noise at the desired_signal-to-noise (SNR)
# level, |d_snr|, in dB. * You can change |d_snr| for your experiments*

d_snr = 0  # desired SNR
spn, ns = funs(s).wgn(d_snr)
e_snr = funs(s).SNR(spn)

plt.figure()
plt.subplot(211)
plt.plot(s)
plt.subplot(212)
plt.plot(spn)
plt.show()

# show the data in time-frequency domain,
# including short-time fourier transform (STFT), Viger-ville distribution (WVD)
# adaptive Chirplet spectrum (ACS)
P = [P1, P2]

# first, show the time-frequency distribution of clean signal
# STFT -- clean signal
tfr, t, f = funs(s, fs=1).STFT_plot(plot=False)
title = 'STFT -- clean signal'
funs(s).contour(tfr, t, f, title=title)

# WVD -- clean signal
tfr, t, f = funs(s, fs=1).WVD_plot(plot=False)
title = 'WVD -- clean signal'
funs(s).contour(tfr, t, f, levels=8, half=False, title=title)

# ACS Explicit -- clean signal
tfr, t, f = funs(s, fs=1).ACS_plot(P, method='explicit', p_type='cohen')
title = 'Explicit ACS -- clean signal'
funs(s).contour(tfr, t, f, title=title)

# ACS direct --  clean signal
tfr, t, f = funs(s, fs=1).ACS_plot(P, method='direct', p_type='cohen')
title = "Direct ACS -- clean signal"
funs(s).contour(tfr, t, f, half=False, title=title)

# perform adaptive chirplet decomposition with MP
# decompose the noisy signal with MPEM and MLE algorithm
tests = hilbert(spn)  # convert it into analytic signal
Q = 2  # number of atoms desired

# test with MLE Adaptive Chirplet Transform
title = 'Explicit MLE - ACS -- noisy signal'
p_mle = mle_adapt_chirplets(tests, Q, methodid=4).P
# p_mle = funs(s, fs=1).On2PC(p_mle).real
tfr, t, f = funs(spn).ACS_plot(p_mle)
funs(spn).contour(tfr, t, f, P=p_mle, reconstruction=True, title=title)

# test with EM Adaptive Chirplet Transform
p_mp = mp_adapt_chirplets(tests, Q, methodid=0).P
# p_mle = funs(s, fs=1).On2PC(p_mle).real
tfr, t, f = funs(spn).ACS_plot(p_mp)
title = 'Explicit EM - ACS -- noisy signal'
funs(spn).contour(tfr, t, f, P=p_mp, reconstruction=True, title=title)
