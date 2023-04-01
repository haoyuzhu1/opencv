import cv2
img=cv2.imread("C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\flower.jpg")
img1=cv2.imread("C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\car.jpg")
img_1=img+10
print(img[:5,:,0])
print(img_1[:5,:,0])
print((img+img_1)[:5,:,0])       #超出256部分%256
print(cv2.add(img,img_1)[:5,:,0])    #越界取255，没越界取自身


