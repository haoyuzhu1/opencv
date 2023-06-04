import cv2
import numpy as np
import matplotlib.pyplot as plt
img1=cv2.imread('C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\pipei1.png',0)
img2=cv2.imread('C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\pipei2.png', 0)
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows ()
cv_show('imgl',img1)
cv_show('img2',img2)
sift=cv2.xfeatures2d. SIFT_create()
kpl,desl =sift.detectAndCompute(img1, None)
kp2,des2=sift.detectAndCompute(img2,None)
#crossCheck表示两个特征点要互相匹,例如A中的第i个特征点与B中的第j个特征点最近的,并且B中的第j个特征点到A中的第i个特征点也是
#NORM_L2:归一化数组的(欧几里德距离),如果其他特征计算方法需要考虑不同的匹配计算方式
bf = cv2.BFMatcher (crossCheck=True)
#1对1的匹配
matches=bf.match(desl,des2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kpl, img2, kp2, matches[:10], None, flags=2)
cv_show('img3',img3)
#k对最佳匹配
bf = cv2.BFMatcher()
matches=bf.knnMatch(desl,des2, k=2)
good=[]
for m,n in matches:
    if m.distance<0.75*n.distance:
       good.append([m])
img3 = cv2.drawMatchesKnn (img1, kpl,img2, kp2, good, None,flags=2)
cv_show('img3',img3)
