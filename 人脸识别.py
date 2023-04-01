import cv2
import os
import numpy as np
def load_data():
    listdir=os.listdir('C:\\Users\\haoyu\\Desktop\\opencv')
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
    target=[]
    for index,dir in enumerate(dirs):
        for i in range(1,31):
            gray=cv2.imread('.face_capture/%s/%d.jpg'%(dir,i))#三维图片
            gray_=gray[:,:,0]
            gray_=cv2.resize(gray_,dsize=(64,64))
            faces.append(gray_)
            target.append(index)
    faces=np.asarray(faces)
    target=np.asarray(target)
    return faces,target,dirs   

def split_data(faces,target):
    a=np.arange(30)
    np.random.shuffle(a)
    faces=faces[a]
    target=target[a]
    X_train,X_test=faces[:25],faces[25:]
    Y_train,Y_test=target[:25],target[25:]
    return X_train,X_test,Y_train,Y_test


if __name__ == '__main__':
    #加载数据
    faces,target,dirs=load_data()
    X_train,X_test,Y_train,Y_test=split_data(faces,target)
    face_recognizer=cv2.face_EigenFaceRecognizer.create()
    face_recognizer.train(X_train,Y_train)
    for face in X_test:
        y_,confidence=face_recognizer.predict(face)
        print(y_)
        cc=dirs[y_]
        print('这个人是：-------',cc)
        cv2.imshow('face',face)
        key=cv2.waitKey(0)
        if key==ord("q"):
            break
    cv2.destroyAllWindows()