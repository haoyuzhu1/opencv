import cv2
import os
import numpy as np
def load_data():
    listdir=os.listdir('./face_dynamic')
    # print(listdir)
    # dirs=[]
    # for d in listdir:
    #     if d.startswith('.'):
    #         pass
    #     else:
    #         dirs.append(d)
    # print(dirs)
    dirs=[d for d in listdir if not d.endswith('.py')]
    print(dirs)
    faces=[]
    target=[i for i in range(len(dirs))]*10
    for dir in dirs:
        for i in range(1,11):
            gray=cv2.imread('./face_dynamic/%s/%d.jpg'%(dir,i))#三维图片
            gray_=gray[:,:,0]
            gray_=cv2.equalizeHist(gray_)
            faces.append(gray_)
    faces=np.asarray(faces)
    target=np.asarray(target)
    target.sort()
    return faces,target,dirs   

def split_data(faces,target):
    a=np.arange(30)
    np.random.shuffle(a)
    faces=faces[a]
    target=target[a]
    X_train,X_test=faces[:25],faces[25:]
    Y_train,Y_test=target[:25],target[25:]
    return X_train,X_test,Y_train,Y_test

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
        key=cv2.waitKey(1000//500)
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
        
def dynamic_recognizer_face(face_recognizer,dirs):
    cap=cv2.VideoCapture(0)
    face_detector=cv2.CascadeClassifier('C:\\Users\\haoyu\\Desktop\\haarcascade_frontalface_alt.xml')
    while True:
        flag,frame=cap.read()
        if not flag:
           break
        gray=cv2.cvtColor(frame,code=cv2.COLOR_BGR2GRAY)
        faces=face_detector.detectMultiScale(gray,minNeighbors=10)
        for x,y,w,h in faces:
            face=gray[y:y+h,x:x+w]
            face=cv2.resize(face,dsize=(64,64))
            face=cv2.equalizeHist(face)
            y_,confidence=face_recognizer.predict(face)#人脸辨识
            label=dirs[y_]
            print('这个人是：%s。置信度是：%0.1f'%(label,confidence))
            cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,255],thickness=2)
            cv2.putText(frame,text=label,
                        org=(x,y-10),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1.5,
                        color=[0,0,255],
                        thickness=2)
        cv2.imshow('face',frame)
        key=cv2.waitKey(1000//24)
        if key==ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    
    
    
if __name__ == '__main__':
    # take_faces()
    #加载数据
    faces,target,dirs=load_data()
    print(faces.shape,target.shape)
    # X_train,X_test,Y_train,Y_test=split_data(faces,target)
    #face_recognizer=cv2.face_EigenFaceRecognizer.create()
    #face_recognizer=cv2.face_FisherFaceRecognizer.create()
    face_recognizer=cv2.face_LBPHFaceRecognizer.create()
    face_recognizer.train(faces,target)
    dynamic_recognizer_face(face_recognizer,dirs)