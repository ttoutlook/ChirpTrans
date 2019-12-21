'''
@author: Tang Tao
@contact: tangtaooo123@gmail.com
@file: boundariesDet.py
@time: 12/21/2019 7:51 PM
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import imageio
from skimage.morphology import closing

thresh1 = 127
thresh2 = 254

# Load image
im = imageio.imread('jBD9j.png')

# Get threashold mask for different regions
gryim = np.mean(im[:, :, 0:2], 2)
region1 = (thresh1 < gryim)
region2 = (thresh2 < gryim)
nregion1 = ~ region1
nregion2 = ~ region2

# Plot figure and two regions
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(im)
axs[0, 1].imshow(region1)
axs[1, 0].imshow(region2)

# Clean up any holes, etc (not needed for simple figures here)
# region1 = sp.ndimage.morphology.binary_closing(region1)
# region1 = sp.ndimage.morphology.binary_fill_holes(region1)
# region1.astype('bool')
# region2 = sp.ndimage.morphology.binary_closing(region2)
# region2 = sp.ndimage.morphology.binary_fill_holes(region2)
# region2.astype('bool')

# Get location of edge by comparing array to it's
# inverse shifted by a few pixels
shift = -1
edgex1 = (region1 ^ np.roll(nregion1, shift=shift, axis=0))
edgey1 = (region1 ^ np.roll(nregion1, shift=shift, axis=1))
edgex2 = (region2 ^ np.roll(nregion2, shift=shift, axis=0))
edgey2 = (region2 ^ np.roll(nregion2, shift=shift, axis=1))

# Plot location of edge over image
axs[1, 1].imshow(im)
axs[1, 1].contour(edgex1, 2, colors='r')
axs[1, 1].contour(edgey1, 2, colors='r')
axs[1, 1].contour(edgex2, 2, colors='g')
axs[1, 1].contour(edgey2, 2, colors='g')
plt.show()
