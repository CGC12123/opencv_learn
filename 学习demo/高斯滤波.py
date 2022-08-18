import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("./image/rain.jpg")
plt.imshow(img[:,:,::-1])
plt.show()

img2 = cv.GaussianBlur(img, (3,3), 1)
plt.imshow(img2[:,:,::-1])
plt.show()