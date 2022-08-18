import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
# 1 读取图像

cap = cv.VideoCapture(0)
ret, frame = cap.read()
while True:
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    # 2 角点检测
    corners = cv.goodFeaturesToTrack(gray,1000,0.01,10)  
    # 3 绘制角点
    for i in corners:
        x,y = i.ravel()
        cv.circle(frame,(int(x),int(y)),2,(0,0,255),-1)
    cv.imshow("res", frame)
    cv.waitKey(1)