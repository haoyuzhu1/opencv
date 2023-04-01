import cv2
head=cv2.imread('C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\3.png',-1)
data=cv2.split(head)
print(len(head.shape))
print(len(data))