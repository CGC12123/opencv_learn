import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("./image/view.jpg")

rows, cols = img.shape[:2]

pst1 = np.float32([[56,65],[368,95],[28,387],[389,390]])
pst2 = np.float32([[100,145],[300,100],[80,290],[310,300]])

T = cv.getPerspectiveTransform(pst1,pst2)

res = cv.warpPerspective(img, T, (cols, rows))

plt.imshow(res[:,:,::-1])
plt.show()