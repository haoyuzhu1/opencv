import cv2
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
img=cv2.imread("C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\car.png")
print(img)
# cv2.imshow("C:\\Users\\haoyu\\Desktop\\1.png",img)
print(img.shape)
c=cv2.imread('C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\car.png',cv2.IMREAD_GRAYSCALE)    #灰度图
cv2.imshow('C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\car.png',c)
cv2.waitKey(10000)
cv2.destroyAllWindows()
cv2.imwrite("C:\\Users\\haoyu\\Desktop\\opencvimg\\gray\\graygraph.png",c)    
print(type(c))
print(c.size)   #像素点个数
print(c.dtype)  #数据类型