import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("./image/1.jpg")
plt.imshow(img[:,:,::-1])
plt.show()

kernel = np.ones((10, 10), np.uint8)
cvtop = cv.morphologyEx(img, cv.MORPH_TOPHAT,kernel)
plt.imshow(cvtop[:,:,::-1])
plt.show()

cvblack = cv.morphologyEx(img, cv.MORPH_BLACKHAT,kernel)
plt.imshow(cvblack[:,:,::-1])
plt.show()