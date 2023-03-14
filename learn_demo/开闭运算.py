import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("./image/1.jpg")
plt.imshow(img[:,:,::-1])
plt.show()

kernel = np.ones((10, 10), np.uint8)
cvopen = cv.morphologyEx(img, cv.MORPH_OPEN,kernel)
plt.imshow(cvopen[:,:,::-1])
plt.show()

cvclose = cv.morphologyEx(img, cv.MORPH_CLOSE,kernel)
plt.imshow(cvclose[:,:,::-1])
plt.show()