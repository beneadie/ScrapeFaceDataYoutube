import cv2
import os
import pathlib

#face train data from python library
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"


#must convert it to string
detector = cv2.CascadeClassifier(str(cascade_path))


camera = cv2.VideoCapture(0)

name_person = str(input("enter name: "))

path = "face_images/"+name_person

checkExists = os.path.exists(path)

if checkExists:
     print("name taken. give another")
     name_person = str(input("enter name: "))
     path = "face_images/"+name_person
else:
     os.makedirs(path)
count=0
while True:
     _, frame = camera.read()
     faces = detector.detectMultiScale(frame, 1.3, 5)
     #faces = detector.detectMultiScale(
     #   frame, scaleFactor=1.07, minNeighbors=4, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE
     #   )
     for x, y, w, h in faces:
          count+=1
          name = f"./face_images/{name_person}/{count}.jpg"
          print(f"making image \t{name}")
          cv2.imwrite(name, frame[y:y+h, x:x+w])
          cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 3)
     cv2.imshow("Faces", frame)
     cv2.waitKey(1)
     if count > 250:
          break
camera.release()
cv2.destroyAllWindows()




