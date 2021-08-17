import cv2

trainedDataset=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#readimg
img=cv2.imread('img/20210629_164949.jpg')

#convgray
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face=trainedDataset.detectMultiScale(gray)
print(face)

for x,y,w,h in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

#cv2.imshow('bala',gray)
cv2.imshow('bala',img)
cv2.waitKey()