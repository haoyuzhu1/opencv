import cv2
#%matplotlib inline
img=cv2.imread("C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\flower.jpg")
top_size,bottom_size,left_size,right_size=(50,50,50,50)
# replicate=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REPLICATE) #复制最边缘像素
# cv2.imshow('replicate',replicate)
# reflect=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT)    #反射法      fedcba|abcdefgh|hgfedcb
# cv2.imshow('refelct',reflect)
# reflect_101=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT101)#反射法以最边缘像素为轴      gfedcb|abcdefgh|gfedcba
# cv2.imshow('refelct101',reflect_101)                  
# BORDER_WRAP=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_WRAP)#外包装法
# cv2.imshow('BORDER_WRAP',BORDER_WRAP)
constant=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT,value=0)#常数法
cv2.imshow('BORDER_WRAP',constant)
cv2.waitKey(10000)
cv2.destroyAllWindows()