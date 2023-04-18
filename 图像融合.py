import cv2
import matplotlib.pyplot as plt
img=cv2.imread("C:\\Users\\haoyu\\Desktop\\lena.jpg")
res=cv2.resize(img,(200,200))
print(res.shape)
# cv2.imshow("C:\\Users\\haoyu\\Desktop\\lena.jpg",res)
img1=cv2.imread("C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\car.png")
res1=cv2.resize(img1,(200,200))
print(res1.shape)
c=cv2.addWeighted(res,0.6,res1,0.4,0)
plt.imshow(c)
plt.show()