'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: ChirpletFamily.py
@time: 12/17/2019 6:16 PM
'''

import numpy as np
from scipy.signal import hilbert

# this program is used to create chirplet family and coarsely estimate four parameters:
# namely, time shift, frequency shift, time duration, chirplet rate
# if the chirplet rate is almost zero, here we'll create meyer wavelet to extract these components

class chirpletfamily:
    def __init__(self, ):
        None