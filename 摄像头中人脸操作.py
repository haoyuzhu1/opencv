import cv2
v=cv2.VideoCapture(0)
h_=v.get(propId=cv2.CAP_PROP_FRAME_HEIGHT)
w_=v.get(propId=cv2.CAP_PROP_FRAME_WIDTH)
print(h_,w_)
face_detector=cv2.CascadeClassifier('C:\\Users\\haoyu\\Desktop\\haarcascade_frontalface_alt.xml')
head=cv2.imread('C:\\Users\\haoyu\\Desktop\\opencvimg\\origin\\frog.png')
while True:
    flag,frame=v.read()
    if flag == False:
        break
    frame=cv2.resize(frame,dsize=(int(w_//2),int(h_//2)))
    gray = cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
    for x,y,w,h in faces:
        # cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,255],thickness=2)
        # frame[y:y+h,x:x+w]=cv2.resize(head,dsize=(w,h))
        head2=cv2.resize(head,dsize=(w,h))
        for i in range(w):
            for j in range(h//10):
                if(head2[j,i]>200).all():
                    pass
                else:
                    frame[j+y,i+x]=head2[j,i]
                
        for i in range(h//10,4*h//5):   #高度
            flag=1
            for j in range(w//2):   #宽度
                if(flag==1)&(head2[i,j]>200).all():
                    pass
                else:
                    frame[i+y,j+x]=head2[i,j]
                    flag=0
            flag=1
            for j in range(w-1,w//2-1,-1):   #宽度
                if(flag==1)&(head2[i,j]>120).all():
                    pass
                else:
                    frame[i+y,j+x]=head2[i,j]
                    flag=0
        for i in range(w):
            flag=1
            for j in range(h-1,4*h//5-1,-1):
                if(flag==1)&(head2[j,i]>120).all():
                    pass
                else:
                    flag=0
                    frame[j+y,i+x]=head2[j,i]
    cv2.imshow('frame',frame)
    key=cv2.waitKey(10)
    if key == ord('q'):
       break
cv2.destroyAllWindows()       