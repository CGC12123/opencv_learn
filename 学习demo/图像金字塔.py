import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("./image/view.jpg")

rows, cols = img.shape[:2]

plt.imshow(img[:,:,::-1])
plt.show()

imgup = cv.pyrUp(img)
plt.imshow(imgup[:,:,::-1])
plt.show()

imgdown = cv.pyrDown(img)
plt.imshow(imgdown[:,:,::-1])
plt.show()
