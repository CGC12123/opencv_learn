import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
"""
img = np.zeros((256,256,3), np.uint8)
plt.imshow(img[:,:,::-1])
plt.show()

print(img[100,100])#获取（100，100）处像素点的值
print(img[100,100,0])#获取（100，100）0的像素值（BGR中的G值）

img[100,100] = (0,0,255)#修改（100，100）处的像素值为（0，0，255）
plt.imshow(img[:,:,::-1])#-1表示通道值翻转，cv中为BGR，plt中为RGB
print(img[100,100])
"""
"""
#图片获取和输出
img = cv.imread("./image/1.jpg")
plt.imshow(img[:,:,::-1])
plt.show()
"""
img = cv.imread("./image/1.jpg")
b,g,r = cv.split(img)#拆分bgr
plt.imshow(b,cmap=plt.cm.gray)#对b通道输出灰度图
#plt.show()

img2 = cv.merge((b,g,r))#合并bgr
plt.imshow(img2[:,:,::-1])#输出合并后的img2
plt.show()

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
plt.imshow(gray,cmap = plt.cm.gray)#设置cmap结果为输出灰度图
plt.show()

hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)#hsv结果
plt.imshow(hsv)
plt.show()