import cv2
import os
import pathlib
from pytube import YouTube, Stream
import io


youtube_url = "https://www.youtube.com/watch?v=aNJ_JPkbH2M&ab_channel=GQ"

yt_obj = YouTube(youtube_url)

video_stream = yt_obj.streams.get_highest_resolution()
temp_file = 'temp1.mp4'
print("Donwloading video")
video_stream.download(output_path='.', filename=temp_file)


#video_data = video_stream.stream_to_buffer(buffer=)

#face train data from python library
print("getting training data")
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"


#must convert it to string
detector = cv2.CascadeClassifier(str(cascade_path))


#camera = cv2.VideoCapture(0)
video_data = cv2.VideoCapture(temp_file)

name_person = str(input("enter name: "))

path = "face_images/"+name_person



checkExists = os.path.exists(path)
while True:
     if checkExists:
          print("name taken. give another")
          name_person = str(input("enter name: "))
          path = "face_images/"+name_person
     else:
          os.makedirs(path)
          break
count=0
frame_skip = 25  # Adjust this value to skip more or fewer frames
video_data.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start from the beginning
while True:
     _, frame = video_data.read()
     faces = detector.detectMultiScale(frame, 1.3, 5)
     faces = detector.detectMultiScale(
        frame, scaleFactor=1.07, minNeighbors=4, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE
        )
     for x, y, w, h in faces:
          count+=1
          name = f"./face_images/{name_person}/{count}.jpg"
          print(f"making image \t{name}")
          cv2.imwrite(name, frame[y:y+h, x:x+w])
          cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 3)
     cv2.imshow("Faces", frame)
     k=cv2.waitKey(1)
     if count>= 250:              #ord("q"):
          break
     for _ in range(frame_skip - 1):
        video_data.grab()

#camera.release()
cv2.destroyAllWindows()
os.remove(temp_file)
print(f"'{temp_file}' deleted")




