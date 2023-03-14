import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("./image/view.jpg")
plt.imshow(img[:,:,::-1])
plt.show()

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

hist = cv.calcHist([img], [0], None, [256], [0,256])
plt.figure(figsize=(10,10))
plt.plot(hist)
plt.show()