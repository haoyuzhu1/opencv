import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread("C:\\Users\\haoyu\\Desktop\\lena.png")
cv2.imshow('1',img)
cv2.waitKey(10000)
cv2.destroyAllWindows()

#均值滤波
#简单的平均卷积操作    3*3  9个像素平均
blur=cv2.blur(img,(3,3))
cv2.imshow('blur',blur)
cv2.waitKey(0)
cv2.destroyAllWindows ()
#方框滤波
#基本和均值一样,可以选择归一化，容易越界,normalize做归一化，指定False越界归为255
box=cv2.boxFilter(img,-1,(3,3), normalize=True)
cv2.imshow('box',box)
cv2.waitKey(0)
cv2.destroyAllWindows ()
#高斯滤波
#高斯模糊的卷积核里的数值是满足高斯分布,相当干更重视中间的
aussian=cv2.GaussianBlur(img,(5, 5), 1)
cv2.imshow('aussian',aussian)
cv2.waitKey(0)
cv2.destroyAllWindows
#中值滤波
#相当于用中值代替
median=cv2.medianBlur(img, 5)#中值滤波
cv2.imshow('median',median)
cv2.waitKey(0)
cv2.destroyAllWindows ()
#展示所有的
res = np.hstack((blur,aussian,median))
cv2.imshow('medianvsaverage',res)
cv2.waitKey(0)
cv2.destroyAllWindows()


