import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('./image/view.jpg')
rows, cols = img.shape[:2]
print("原始尺寸",rows, cols)
plt.imshow(img[:,:,::-1])
plt.show()
# cv.imshow("1",img)        # 显示图像，窗口标题未:lena
# cv.waitKey(0)                #等待用户输入
# cv.destroyAllWindows() 

#绝对尺寸
print("---绝对尺寸---")
res1 = cv.resize(img, (2*cols, 2*rows))
rows_changed1, cols_changed1 = res1.shape[:2]
print("绝对放大后",rows_changed1, cols_changed1)
plt.imshow(res1[:,:,::-1])
plt.show()
# cv.imshow("2",res1)        # 显示图像，窗口标题未:lena
# cv.waitKey(0)                #等待用户输入
# cv.destroyAllWindows()

#相对尺寸
print("---相对尺寸---")
res2 = cv.resize(img, None, fx = 0.5, fy = 0.5)
rows_changed2, cols_changed2 = res2.shape[:2]
print("相对缩小后",rows_changed2, cols_changed2)
plt.imshow(res2[:,:,::-1])
plt.show()
# cv.imshow("3",res2)        # 显示图像，窗口标题未:lena
# cv.waitKey(0)                #等待用户输入
# cv.destroyAllWindows()