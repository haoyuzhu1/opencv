import cv2
import numpy as np
vc=cv2.VideoCapture("C:\\Users\\haoyu\\Desktop\\opencvimg\\video\\2.mp4")

if vc.isOpened():
    # video.read() 一帧一帧地读取
    # open 得到的是一个布尔值，就是 True 或者 False
    # frame 得到当前这一帧的图像
    open, frame = vc.read()
else:
    open = False
while open:
    ret,frame=vc.read()
    if frame is None:
        break
    if ret ==True:
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow("vc",grey)
        if cv2.waitKey(50)&0XFF == 27:
            break
vc.release()
cv2.destroyAllWindows()
    