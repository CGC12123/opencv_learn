import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("./image/1.jpg",0)
plt.imshow(img,cmap = plt.cm.gray)
plt.show()

mask = np.zeros(img.shape[:2],np.uint8())#创建掩膜 
mask[100:200,100:400] = 1   #设置感兴趣区域

plt.imshow(mask,cmap = plt.cm.gray)
plt.show()

mask_img = cv.bitwise_and(img,img,mask = mask)
plt.imshow(mask_img,cmap = plt.cm.gray)
plt.show()

mask_hist = cv.calcHist([mask_img], [0], None, [256], [0,256])
plt.imshow(mask_hist,cmap = plt.cm.gray)
plt.show()