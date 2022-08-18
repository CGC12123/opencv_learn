import cv2 as cv
from cv2 import cvtColor
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("./scaning/word.jpg")
# cv.imshow("img", img)
plt.imshow(img[:,:,::-1])
plt.show()

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# mask = np.zeros(img.shape[:2],np.uint8())#创建掩膜 
# mask[100:350,900:1170] = 1   #设置感兴趣区域
# mask_img = cv.bitwise_and(img,img,mask = mask)
# plt.imshow(mask_img[:,:,::-1])
# plt.show()

temp = cv.imread("./scaning/word2.png")
plt.imshow(temp[:,:,::-1])
plt.show()

temp_gray = cvtColor(temp, cv.COLOR_BGR2GRAY)



res = cv.matchTemplate(img_gray, temp_gray, cv.TM_CCORR_NORMED)
plt.imshow(res, plt.cm.gray)
plt.show()

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = max_loc
h,w = temp.shape[:2]
bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(img, top_left, bottom_right, (0,255,0), 2)

plt.imshow(img[:,:,::-1])
plt.show()