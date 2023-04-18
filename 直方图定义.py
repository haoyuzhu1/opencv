import cv2
import numpy as np
from matplotlib import pyplot as plt
def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()
img=cv2.imread('C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\face2.png',0)#0表示灰度图
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.hist(img.ravel(),256,[0, 256])
plt.show()
img=cv2.imread('C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\face2.png',0)
# color=('b','g','r')
# for i, col in enumerate(color):
#     histr=cv2.calcHist([img], [i], None, [256],[0, 256])
#     plt.plot(histr, color = col)
#     plt.xlim([0,256])
plt.show()
equ = cv2.equalizeHist(img)
plt.hist(equ.ravel(),256)
plt.show()
res=np.hstack((img,equ))
cv_show(res,'res')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
res_clahe = clahe.apply(img)
res = np.hstack((img,equ,res_clahe))
cv_show(res,'res')
