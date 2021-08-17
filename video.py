import cv2

trainedDataset=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video=cv2.VideoCapture('video/WhatsApp Video 2021-08-16 at 6.26.50 PM.mp4')
while True:
    success,frame = video.read()
    if success==True:
        gray_v = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = trainedDataset.detectMultiScale(gray_v)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('video', frame)
        cv2.waitKey(1)
    else:
        print("Video Completed")
        break;

