import cv2 as cv
import math

image = cv.imread("./image/angle.jpg")
# cv.imshow("image", image)
# cv.waitKey(0)

pointlist = []

def mousePoint(event , x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        size =len(pointlist)
        # if size != 0 and size % 3 != 0:
        #     cv.line(image, tuple(pointlist[round((size-1)/3)*3]),(x, y))
        cv.circle(image, (x, y), 5, (0, 0, 255), cv.FILLED)
        pointlist.append([x, y])
        # print(pointlist)
        # print(x, y)

def get_angle(pointlist):
    pt1, pt2, pt3 = pointlist[-3:]
    # print(pt1, pt2, pt3)
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = abs(math.atan((m2-m1)/(1+(m2*m1)))) # 利用公式求弧度
    angD = round(math.degrees(angR)) # 转化为角度

    cv.putText(image, str(angD), (pt1[0] - 40, pt1[1] - 20), cv.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
    print(angD)

def gradient(pt1, pt2):
    return (pt2[1] - pt1[1])/(pt2[0] - pt1[0])

while True:
    if len(pointlist) % 3 == 0 and len(pointlist) != 0: # 每点击三次测量一次角度
        get_angle(pointlist)


    cv.imshow("image", image)
    cv.setMouseCallback("image", mousePoint) # 鼠标点击时输出当前坐标
    if cv.waitKey(1) & 0xff == ord('q'):
        pointlist = []
        image = cv.imread("./image/angle.jpg")


