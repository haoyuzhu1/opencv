import cv2
import numpy as np
def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()


img=cv2.imread("C:\\Users\\haoyu\\Desktop\\lena.jpg",cv2. IMREAD_GRAYSCALE)
v1=cv2.Canny(img, 80,150)
v2=cv2.Canny(img, 50,100)
res=np.hstack((v1,v2))
cv_show(res,'res')
