import cv2
import numpy as np
v=cv2.VideoCapture(0)
h_=v.get(propId=cv2.CAP_PROP_FRAME_HEIGHT)
w_=v.get(propId=cv2.CAP_PROP_FRAME_WIDTH)
print(h_,w_)
face_detector=cv2.CascadeClassifier('C:\\Users\\haoyu\\Desktop\\haarcascade_frontalface_alt.xml')
# head=cv2.imread('C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\frog.jpg')
while True:
    head=cv2.imread('C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\3.png',-1)
    print('++++++++',head.shape)
    flag,frame=v.read()
    if flag == False:
        break
    frame=cv2.resize(frame,dsize=(int(w_//2),int(h_//2)))
    gray = cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3)
    flag=0
    # print(frame.dtype)
    for x,y,w,h in faces:
        # cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,255],thickness=2)
        # frame[y:y+h,x:x+w]=cv2.resize(head,dsize=(w,h))
        head=cv2.resize(head,dsize=(w,h))
        head_channels=cv2.split(head)
        # print(head_channels)
        frame_channels=cv2.split(frame)
        b,g,r,a=cv2.split(head)
        for c in range(0,3):
            print(head_channels[c],np.array(a/255,dtype=np.uint8))
            k=np.uint8((255.0-a)/255.0)
            frame_channels[c][y:y+h,x:x+w]=frame_channels[c][y:y+h,x:x+w]*k
            print(np.array(a/255,dtype=np.uint8))
            head_channels[c]*= np.array(a/255,dtype=np.uint8)
            frame_channels[c][y:y+h,x:x+w]+=np.array(head_channels[c],dtype=np.uint8)
        ans=cv2.merge(frame_channels)
        flag=1
    if flag:
        cv2.imshow('frame',ans)
    else:
        cv2.imshow('frame',frame)
    key=cv2.waitKey(10)
    if key == ord('q'):
       break
cv2.destroyAllWindows()       