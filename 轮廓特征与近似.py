import cv2

img=cv2.imread('C://Users//haoyu//Desktop//opencvimg//origin//frame.png')
def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转成灰度图
ret,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#cv_show(img,'img')
#cv_show(thresh,'thresh')
 
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#轮廓特征
cnt = contours[0]
#面积
print(cv2.contourArea(cnt))
#周长,True表示闭合的
print(cv2.arcLength(cnt,True))


img=cv2.imread('C://Users//haoyu//Desktop//opencvimg//origin//contour.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
draw_img = img.copy()
res = cv2.drawContours(draw_img, [cnt], -1,(0, 0, 255), 2)
cv_show(res,'res')

img=cv2.imread('C://Users//haoyu//Desktop//opencvimg//origin//frame.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt=contours[2]
x,y,w,h=cv2.boundingRect(cnt)
img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv_show(img,'img')

area = cv2.contourArea (cnt)
x,y,w,h=cv2.boundingRect(cnt)
rect_area = w * h
extent=float(area)/rect_area
print('轮廓面积与边界矩形比',extent)

