import cv2
#%matplotlib inline
img=cv2.imread("C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\flower.jpg")
person=img[450:750,350:650]
# cv2.imshow('person',person)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()
b,g,r=cv2.split(person)
print(b)
print(b.shape)
print(g)
print(g.shape)
print(r) 
print(r.shape)
cur_img=person.copy()
cur_img[:,:,1]=0
cur_img[:,:,2]=0
cv2.imshow('B',cur_img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
# cur_img[:,:,0]=0
# cur_img[:,:,2]=0
# cv2.imshow('G',cur_img)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()
# cur_img[:,:,0]=0
# cur_img[:,:,1]=0
# cv2.imshow('R',cur_img)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()