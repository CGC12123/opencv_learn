import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("./image/horse.jpg", 0)
plt.imshow(img, cmap = plt.cm.gray)
plt.show()

res = cv.Laplacian(img, cv.CV_16S)
res = cv.convertScaleAbs(res)
plt.imshow(res, cmap = plt.cm.gray)
plt.show()