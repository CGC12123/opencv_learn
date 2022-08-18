import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = np.zeros((512, 512, 3), np.uint8)

cv.line(img, (0,0), (511, 511), (255, 0, 0), 5)  #直线
cv.circle(img, (256, 256), 60, (50, 50,150), 3)  #圆形
cv.rectangle(img, (100, 100), (400, 400), (100, 100, 70), 4)    #矩形
cv.putText(img, "loloo", (160, 480), cv.FONT_HERSHEY_SIMPLEX, 3, (40,20,100), 3)   #文字

plt.imshow(img[:,:,::-1])   #输出图像
plt.show()