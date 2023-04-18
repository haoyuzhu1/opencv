import cv2
import numpy as np
def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

img=cv2.imread("C:\\Users\\haoyu\\Desktop\\lena.jpg")
cv_show(img,'img')
print (img.shape)
up=cv2.pyrUp(img)
cv_show(up,'up')
print (up.shape)
down=cv2.pyrDown (img)
cv_show(down,'down')
print (down.shape)
up=cv2.pyrUp(img)
up_down=cv2.pyrDown (up)
cv_show(up_down,'up_down')
cv_show(np.hstack((img,up_down)), 'up_down')
down=cv2.pyrDown (img)
down_up=cv2.pyrUp(down)
l_1=img-down_up
cv_show(l_1,'l_1')
