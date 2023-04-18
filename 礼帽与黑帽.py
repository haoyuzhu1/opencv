import cv2
import numpy as np
#礼帽
img=cv2.imread('C:\\Users\\haoyu\\Desktop\\di.png')
kernel = np.ones((5,5),np.uint8)
tophat=cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
cv2.imshow('tophat',tophat)
cv2.waitKey(0)
cv2. destroyAllWindows
#黑帽
img=cv2.imread('C:\\Users\\haoyu\\Desktop\\di.png')
kernel = np.ones((5,5),np.uint8)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('blackhat',blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows ()

