import cv2
import numpy as np

def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

img=cv2.imread('C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\pie.png')
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobelx=cv2.convertScaleAbs(sobelx)
cv_show(sobelx,'sobelx')
sobely = cv2.Sobel (img,cv2.CV_64F,0,1,ksize=3)
sobely=cv2.convertScaleAbs(sobely)
cv_show(sobely,'sobely')
sobelxy = cv2.addWeighted (sobelx,0.5,sobely,0.5,0.5,0)
cv_show(sobelxy,'sobelxy')
sobelxy=cv2.Sobel (img,cv2.CV_64F,1,1,ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy)
cv_show(sobelxy,'sobelxy')


img = cv2.imread("C:\\Users\\haoyu\\Desktop\\lena.png",cv2.IMREAD_GRAYSCALE)
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0, ksize=3)
sobelx=cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted (sobelx,0.5,sobely,0.5,0)
cv_show(sobelxy,'sobelxy')
sobelxy=cv2.Sobel (img,cv2.CV_64F,1,1, ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy)
cv_show(sobelxy,'sobelxy')


