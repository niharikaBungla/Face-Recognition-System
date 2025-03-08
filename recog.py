import face_recognition
import cv2
import numpy as np
import os

def resize(img ,size):
    width = int(img.shape[1]*size)
    height= int(img.shape[0] *size)
    dimension= (width ,height)
    return cv2.resize(img ,dimension ,interpolation=cv2.INTER_AREA)


path = 'data'
images = []     # LIST CONTAINING ALL THE IMAGES
className = []    # LIST CONTAINING ALL THE CORRESPONDING CLASS Names
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
for x,cl in enumerate(myList):
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        className.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= resize(img ,0.50)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
#print('Encodings Complete')

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("ElonMusktest.jpg")

while True:

     success,img = cap.read()
     imgs = cv2.resize(img,(0,0),None ,0.25,0.25)
     imgs = cv2.cvtColor(imgs , cv2.COLOR_BGR2RGB)

     facecurframe= face_recognition.face_locations(imgs)
     encodecurframe = face_recognition.face_encodings(imgs,facecurframe)

     for encodeface,faceloc in zip(encodecurframe,facecurframe):
        matches = face_recognition.compare_faces(encodeListKnown,encodeface)
        facedis = face_recognition.face_distance(encodeListKnown,encodeface)
        matchindex = np.argmin(facedis)

        if matches[matchindex]:
             name = className[matchindex].upper()
             y1,x2,y2,x1 = faceloc
             y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
             cv2.rectangle(img ,(x1,y1),(x2,y2) ,(0,255,0),2)
             cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
             cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

     cv2.imshow('webcam' ,img)
     
     k=cv2.waitKey(1)
     if k==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()