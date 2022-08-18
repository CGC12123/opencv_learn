import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("./image/1.jpg",0)
plt.imshow(img,cmap = plt.cm.gray)
plt.show()

dst = cv.equalizeHist(img)
plt.imshow(dst,cmap = plt.cm.gray)
plt.show()