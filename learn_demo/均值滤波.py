import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("./image/view.jpg")
plt.imshow(img[:,:,::-1])
plt.show()

img2 = cv.blur(img,(5,5))   #均值滤波
plt.imshow(img2[:,:,::-1])
plt.show()