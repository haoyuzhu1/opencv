import cv2
import numpy as np
#梯度=膨胀-腐蚀
pie=cv2.imread('C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\pie.png')
kernel=np.ones((7,7),np.uint8)
dilate = cv2.dilate(pie,kernel,iterations = 5)
erosion = cv2.erode (pie,kernel,iterations = 5)
res=np.hstack((dilate,erosion))
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
gradient = cv2.morphologyEx(pie,cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradient',gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()
