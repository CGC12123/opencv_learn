import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("./image/1.jpg")
plt.imshow(img[:,:,::-1])
plt.show()

kernel = np.ones((5,5),np.uint8)
img1 =cv.erode(img, kernel)#腐蚀
plt.imshow(img1[:,:,::-1])
plt.show()

img2 = cv.dilate(img,kernel)#膨胀
plt.imshow(img2[:,:,::-1])
plt.show()