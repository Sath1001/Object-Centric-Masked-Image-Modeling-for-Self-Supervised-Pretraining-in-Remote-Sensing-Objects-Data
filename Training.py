# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:25:08 2021

@author: EGC
"""

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential
import cv2
import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data
import cv2
import numpy as np
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

# === GETTING INPUT SIGNAL

Trainfea1 = []

for file in range(0,110):
    print(file)
    file_temp = file+1;
    IMG_pt = 'Dataset\images\IMG ('
    IMG_nm = file_temp
    IMG_ext = ').jpg'
    IMGGl = IMG_pt+str(IMG_nm)+IMG_ext
    
    img = mpimg.imread(IMGGl)


# GLCM Features

    PATCH_SIZE = 21

# open the camera image

    image = img[:,:,0]
    image = cv2.resize(image,(768,1024))

#image = data.camera()

# select some patches from grassy areas of the image
    grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
    grass_patches = []
    for loc in grass_locations:
        grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                   loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
    sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
    sky_patches = []
    for loc in sky_locations:
        sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                 loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
    xs = []
    ys = []
    for patch in (grass_patches + sky_patches):
        glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,symmetric=True)
        xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(greycoprops(glcm, 'correlation')[0, 0])

# create the figure
    fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(image, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    for (y, x) in grass_locations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')

    for (y, x) in sky_locations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
    ax.set_xlabel('Original Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
               label='Grass')
    ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
               label='Sky')
    ax.set_xlabel('GLCM Dissimilarity')
    ax.set_ylabel('GLCM Correlation')
    ax.legend()

# display the image patches
    for i, patch in enumerate(grass_patches):
        ax = fig.add_subplot(3, len(grass_patches), len(grass_patches)*1 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        ax.set_xlabel('Region 1 %d' % (i + 1))

    for i, patch in enumerate(sky_patches):
        ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        ax.set_xlabel('Region 2 %d' % (i + 1))


# display the patches and plot
    fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()
    sky_patches0 = np.mean(sky_patches[0])
    sky_patches1 = np.mean(sky_patches[1])
    sky_patches2 = np.mean(sky_patches[2])
    sky_patches3 = np.mean(sky_patches[3])

    Glcm_fea = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]


#image = data.astronaut()

# Rescale histogram for better display



    Trainfea1.append(Glcm_fea)
    
import pickle
with open('Trainfea1.pickle', 'wb') as f:
    pickle.dump(Trainfea1, f)    