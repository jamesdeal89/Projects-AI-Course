# a basic face detector from OpenAI - made following an AI course

import cv2

# trained data from OpenAI github
trainedData = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# image to detect faces in
img = cv2.imread('test.jpg')

# AI needs the photos to be in greyscale

imgGrey = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# detects faces and saves the co-ordinates of the face location in the image
coOrds = trainedData.detectMultiScale(imgGrey)
print(coOrds)

# iterate through and get all faces detected to allow for mutliple detecions
for face in coOrds:
    # assign face co-ordinates
    print(face)
    (x,y,w,h) = face
    # draw detection rectangles around the detected faces based on previous output
    # uses a top left corner co-ordinate and a bottom right co-ordinate
    cv2.rectangle(img=img, pt1=(x,y), pt2=(x+w,y+h), color=(0,255,0), thickness=2)

# creates a UI windows and displays the image
cv2.imshow(winname="face detection", mat=img)

# waits for a pressed key to keep imshow open
cv2.waitKey()
