import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("./image/view.jpg")
rows, cols = img.shape[:2]
M = cv.getRotationMatrix2D((cols/2, rows/2),90,1)#制造旋转矩阵
dst = cv.warpAffine(img,M,(cols,rows))#利用“类平移”使其与原图像进行矩阵乘法

plt.imshow(dst[:,:,::-1])
# fig,axes = plt.subplots(nrows = 1, ncols =2,figsize=(10,8), dpi = 100)
# axes[0].imshow(img[:,:,::-1])
# axes[0].set_titles("原图")
# axes[1].imshow(img[:,:,::-1])
# axes[1].set_titles("旋转后结果")
plt.show()