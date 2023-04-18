import cv2
import numpy as np

img=cv2.imread('C://Users//haoyu//Desktop//opencvimg//origin//frame.png')
def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转成灰度图
ret,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#cv_show(img,'img')
#cv_show(thresh,'thresh')
 
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


draw_img=img.copy() #.copy是复制另存
res=cv2.drawContours(draw_img,contours,-1,(0,0,255),2)
# -1 是所有轮廓都画进来，可以改 （0,0,255）B,G,R的颜色，红色
cv_show(res,'res') 
draw_img = img.copy()
res = cv2.drawContours(draw_img, contours,0,(0, 0, 255),2)
cv_show(res,'res')

