'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: correlation.py
@time: 12/17/2019 12:47 AM
'''
import numpy as np
from scipy import signal
from functionSet import functionSet as funs

sig = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
sig_noise = sig + np.random.randn(len(sig))
corr = signal.correlate(sig_noise, np.ones(128), mode='same') / 128

corr1 = signal.correlate([1 + 1j, 2 + 2j, 3 + 3j], [0 - 1j, 1 - 1j, 0.5 + 1j], mode='same')
t1 = np.asarray([1 + 1j, 2 + 2j, 3 + 3j])
t2 = np.asarray([0 - 1j, 1 - 1j, 0.5 + 1j])
corr2 = funs().ccorr([1 + 1j, 2 + 2j, 3 + 3j], [0 - 1j, 1 - 1j, 0.5 + 1j])
t3 = np.sum(t1 * np.roll(t2, 0))
print(corr2)
print(t3)

import matplotlib.pyplot as plt

clock = np.arange(64, len(sig), 128)
fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
ax_orig.plot(sig)
ax_orig.plot(clock, sig[clock], 'ro')
ax_orig.set_title('Original signal')
ax_noise.plot(sig_noise)
ax_noise.set_title('Signal with noise')
ax_corr.plot(corr)
ax_corr.plot(clock, corr[clock], 'ro')
ax_corr.axhline(0.5, ls=':')
ax_corr.set_title('Cross-correlated with rectangular pulse')
ax_orig.margins(0, 0.1)
fig.tight_layout()
fig.show()
