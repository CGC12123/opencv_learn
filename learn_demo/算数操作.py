import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

rain = cv.imread("./image/rain.jpg")
plt.imshow(rain[:,:,::-1])
plt.show()

view = cv.imread("./image/view.jpg")
plt.imshow(view[:,:,::-1])
plt.show()

img = cv.add(rain, view)       #像素相同才可相加
plt.imshow(img[:,:,::-1])
plt.show()

img2 = cv.addWeighted(view, 0.7, rain, 0.3, 0)#按照7：3进行混合，最后的参数为伽马值，作为图像的补充
plt.imshow(img2[:,:,::-1])
plt.show()