import cv2
import numpy as np
#更改
Width = 640*0.5
Height = 480*0.5
cap = cv2.VideoCapture(0)
cap.set(3, Width)
cap.set(4, Height)
cap.set(10,150)

class Function():
    def __init__(self) -> None:
        # 自定义阈值  前三位为HSV的min   后三位为HSV的max
        self.pen_color = [[51,153,255],[255,0,255],[0,255,0]]
        #  自定义画笔颜色

    def find_color(self,image):
        low = np.array([87,53,197])
        high = np.array([107,255,255])
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.erode(image,kernel=kernel,iterations=2)
        mask = cv2.dilate(mask,kernel=kernel,iterations=1)
        mask = cv2.inRange(image,low,high)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        try:
            are_max = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(are_max)
            box = cv2.boxPoints(rect)
            cv2.drawContours(image, [np.int0(box)], -1, (0, 255, 255), 2)
        except:
            pass
        cv2.imshow('camera', image)
        cv2.waitKey(1)




if __name__ == "__main__":
    fun = Function()
    while True:
        id,frame = cap.read()
        image = cv2.GaussianBlur(frame, (5, 5), 0)  
        imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        fun.find_color(imgHSV)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# xiugai