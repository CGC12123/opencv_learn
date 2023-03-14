from re import A
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('./image/1.jpg')
plt.imshow(img)
plt.show()

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(img_gray, 50, 150)
plt.imshow(edges, plt.cm.gray)
plt.show()

lines = cv.HoughLines(edges, 0.8, np.pi/180, 150)
# plt.imshow(lines)
# plt.show()

for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = rho * a
    y0 = rho * b
    x1 = int (x0 + 1000*(-b))
    y1 = int (y0 + 1000*a)
    x2 = int (x0 - 1000*(-b))
    y2 = int (y0 - 1000*a)
    cv.line(img, (x1, y1), (x2, y2), (50, 250, 50))
plt.imshow(img)
plt.show()