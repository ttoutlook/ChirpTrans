'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: EEWT.py
@time: 12/17/2019 4:44 PM
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from functionSet import functionSet as funs

# This is the main program of EEWT (Enhanced Empirical Wavelet Transform)
# This Program first uses the Adaptive Transform to detect the frequency center, time center, duration and chirplet rate
# if chirplet rate is 0, use meyer wavelet to extract this components,
# if chirplet rate is over 0, use chirplet wavelet to approximate this components

