import numpy as np
import cv2
import matplotlib.pyplot as plt
def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()
#创建mast
img=cv2.imread('C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\dog.jpg')
mask=np.zeros(img.shape[:2], np.uint8)
print (mask.shape)
mask[100:700, 100:700] = 255
cv_show(mask,'mask')
masked_img=cv2.bitwise_and(img, img, mask=mask)#与操作
cv_show(masked_img,'masked_img')
hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.imshow(mask,'gray')
plt.subplot(223),plt.imshow(masked_img,'gray')
plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)
plt.xlim([0,256])
plt.show()
