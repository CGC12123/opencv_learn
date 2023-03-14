import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img =cv.imread('./image/view.jpg')

rows, cols = img.shape[:2]

pts1 = np.float32([[50,50],[200,50],[50,200]])#原图像中选取三个点
pts2 = np.float32([[100,100],[200,50],[100,250]])#对应到仿射变换后的三个点

M = cv.getAffineTransform(pts1, pts2)#构造出仿射的变换矩阵
# print(M)
res = cv.warpAffine(img,M,(cols,rows))
plt.imshow(res[:,:,::-1])
plt.show()