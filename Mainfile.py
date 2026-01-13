# -*- coding: utf-8 -*-
"""

"""


import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential
import cv2
from skimage.io import imshow
import warnings
warnings.filterwarnings("ignore")
plt.gray()
#tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

# === GETTING INPUT SIGNAL

filename = askopenfilename()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread(filename)

plt.imshow(img)
plt.title('ORIGINAL IMAGE')
plt.show()


# PRE-PROCESSING

h1=300
w1=300

dimension = (w1, h1) 
resized_image = cv2.resize(img,(h1,w1))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)

r = resized_image[:,:,0]
g = resized_image[:,:,1]
b = resized_image[:,:,2]


fig = plt.figure()
imshow(r)


plt.imshow(r)
plt.title('RED IMAGE')
plt.show()

plt.imshow(g)
plt.title('GREEN IMAGE')
plt.show()

plt.imshow(b)
plt.title('BLUE IMAGE')
plt.show()

gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

plt.imshow(gray)
plt.title('GRAY IMAGE')
plt.show()


#==============================================
# IMAGE SEGMENTATION
import cv2
from matplotlib import pyplot as plt

img1 = cv2.medianBlur(resized_image[:,:,1],5)

gray1 = 0.2126 * img1[..., 2] + 0.7152 * img1[..., 1] + 0.0722 * img1[..., 0]

ret,th1 = cv2.threshold(img1,80,100,cv2.THRESH_BINARY)

titles = ['Original Image', 'Segmented Image']

images = [img1, th1]

for i in range(2):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    
plt.show()


# select some patches from foreground and background
import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data
import cv2
import numpy as np
from tkinter.filedialog import askopenfilename

PATCH_SIZE = 21

# open the image

image = img[:,:,0]
image = cv2.resize(image,(768,1024))
warnings.filterwarnings("ignore")

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
warnings.filterwarnings("ignore")

# compute some GLCM properties each patch
xs = []
ys = []
for patch in (grass_patches + sky_patches):
    glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,symmetric=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])
warnings.filterwarnings("ignore")

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
warnings.filterwarnings("ignore")

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
        label='Region 1')
ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
        label='Region 2')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()
warnings.filterwarnings("ignore")

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
warnings.filterwarnings("ignore")


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()
sky_patches0 = np.mean(sky_patches[0])
sky_patches1 = np.mean(sky_patches[1])
sky_patches2 = np.mean(sky_patches[2])
sky_patches3 = np.mean(sky_patches[3])

Glcm_fea = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
Tesfea1 = []
Tesfea1.append(Glcm_fea[0])
Tesfea1.append(Glcm_fea[1])
Tesfea1.append(Glcm_fea[2])
Tesfea1.append(Glcm_fea[3])

# Feature Display --
print('***********************************')

print(' GLCM FEATURE ')

print('***********************************')
print('-----------------------------------')

print(Glcm_fea)

print('-----------------------------------')


############################################################################
# SVM

#==============================================
# CLASSIFICATION

#==============================================
# LOAD TRAIN DATA

import pickle
with open('Trainfea1.pickle', 'rb') as f:
    Train_features = pickle.load(f)
    
Labeled = np.arange(0,110)
Labeled[0:50] = 0
Labeled[50:110] = 1

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
#svclassifier.fit(Train_features,Labeled)
svclassifier.fit(Train_features,Labeled)

y_predd = svclassifier.predict([Glcm_fea])

print('\nSVM Predicted Class: %d' % y_predd[0])

if y_predd[0] <= 17:
    
    print('***********************************')

    print('-- Identified as ""Non Spill"" --')
    
    print('***********************************')
    
else:
    
    print('***********************************')
    
    print('-- Identified as ""Spill"" --')

    print('***********************************')

# Labeled[0:37] = 0
# Labeled[0:1] = 1
# Labeled[6:11] = 1


# svclassifier.fit(Train_features,Labeled)

# y_predd = svclassifier.predict([Glcm_fea])

# print('\nSVM Predicted Class: %d' % y_predd[0])

# if y_predd[0] == 0:
    
#     print('***********************************')

#     print('-- Abnormal - Scevere --')
    
#     print('***********************************')
    

#     print('-- Abnormal - hemorrhage --')
    
#     print('***********************************')
    
    
    
# else:
    
#     print('***********************************')
    
#     print('-- Abnormal - Moderate --')

#     print('***********************************')