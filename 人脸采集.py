import cv2
import os
def take_photo(path):
    cap=cv2.VideoCapture(0)
    face_detector=cv2.CascadeClassifier('C:\\Users\\haoyu\\Desktop\\haarcascade_frontalface_alt.xml')
    filename=1
    flag_write=False
    while True:
        flag,frame=cap.read()
        if not flag:
            break
        gray=cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)
        faces=face_detector.detectMultiScale(gray,minNeighbors=10)
        for x,y,w,h in faces:
            if flag_write:
                face=gray[y:y+h,x:x+w]
                face=cv2.resize(face,dsize=(64,64))
                cv2.imwrite('./face_dynamic/%s/%d.jpg'%(path,filename),face)
                filename+=1
            cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,255],thickness=2)
        if filename>10:
            break    
        cv2.imshow('face',frame)
        key=cv2.waitKey(1000//24)
        if key==ord('q'):
            break
        if key==ord('w'):
            flag_write=True
    cv2.destroyAllWindows()
    cap.release()
def take_faces():
     while True:
        key=input("请输入文件夹的名字，拼音缩写")
        if key=="Q":
            break
        os.makedirs('./face_dynamic/%s'%(key),exist_ok=True)
    #动态采集人脸
        take_photo(key)
        
if __name__ == '__main__':
    take_faces()        
