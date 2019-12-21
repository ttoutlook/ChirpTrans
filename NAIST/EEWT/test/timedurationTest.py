'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: timedurationTest.py
@time: 12/20/2019 10:06 AM
'''

import numpy as np
from numpy import pi
from functionSet import functionSet as funs
from MakeChirplet import MakeChirplet

P1 = [10, 1 / 2, pi / 2, pi, 1/18]  # up-chirplet 0 -> pi
P2 = [10, 1 / 2, pi / 2, -pi, 1/18]  # down-chirplet pi -> 0
P = [P1, P2]
sig = MakeChirplet(1000,P,PType='cohen').sig
tfr, t, f = funs(sig.real).ACS_plot(P, p_type='cohen')
title = 'time duration'
funs(sig.real).contour(tfr, t, f, title=title)