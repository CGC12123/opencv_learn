import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("./image/1.jpg",0)
plt.imshow(img,cmap = plt.cm.gray)
plt.show()

cl = cv.createCLAHE(2.0, (8,8)) #对比度阈值2.0，分成8*8
clahe = cl.apply(img)
plt.imshow(clahe,cmap = plt.cm.gray)
plt.show() 