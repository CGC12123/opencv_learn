import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("./image/1.jpg", 0)  #读取当前路径下的图像文件lena,jpg

# 图像读取 as cv
# cv.imshow("1",img)        # 显示图像，窗口标题未:lena
# cv.waitKey(0)                #等待用户输入
# cv.destroyAllWindows()       #用户一旦输入任意键后，程序关闭窗口
# 图像读取 as matplotlib
plt.imshow(img,cmap=plt.cm.gray)
plt.show()

# cv.imwrite("./image/11.jpg",img)    #以灰度图保存
