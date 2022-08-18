import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# img = cv.imread('./image/1.jpg', 0)
# plt.imshow(img[:,:,::-1], cmap=plt.cm.gray)
# plt.show()

img = cv.imread("./image/view.jpg")  
plt.imshow(img[:,:,::-1])
plt.show()

rows, cols = img.shape[:2]
M = np.float32([[1,0,100],[0,1,50]])#平移矩阵，(先列后行?)，即x方向移动100，y方向移动50
res = cv.warpAffine(img, M, (2*cols, 2*rows))#第三个元素为结果图像的尺寸，先列后行，表现为先增行再增列
plt.imshow(res[:,:,::-1])
plt.show()