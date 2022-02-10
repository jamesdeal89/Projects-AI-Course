# a basic face detector from OpenAI - made following an AI course

"""
notes on the haar cascade method for facial detection:
the haar method uses simple differences in whites and blacks to determine what a face should look like.
This is why it only needs a greyscaled image as it doesn't detect facial features, 
just typical gradient changes and locations in a typical face.
This would mean, for example, the line of darker colour around the eyeline and eyebrows horizontally, the vertical
gradient of light to dark along the nose, and other defining features. 
for example in this image: https://www.startpage.com/av/proxy-image?piurl=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1400%2F1%2Agl4JHntNPHQt1G7txpiRMA.png&sp=1644455905T0894db642a08598a603fe1f2f3c45ff055c88dfab0d26c19a4442481cddeb8db
"""

import cv2

# trained data from OpenAI github
trainedData = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# image to detect faces in
#img = cv2.imread('test.jpg')
# or webcam input
webcam = cv2.VideoCapture(0)

# loops to keep getting frames
while True:

    # reads webcam data
    ret, frame = webcam.read()

    # AI needs the photos to be in greyscale
    imgGrey = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # detects faces and saves the co-ordinates of the face location in the image
    coOrds = trainedData.detectMultiScale(imgGrey)
    #print(coOrds)

    # iterate through and get all faces detected to allow for mutliple detecions
    for face in coOrds:
        # assign face co-ordinates
        #print(face)
        (x,y,w,h) = face
        # draw detection rectangles around the detected faces based on previous output
        # uses a top left corner co-ordinate and a bottom right co-ordinate
        cv2.rectangle(img=frame, pt1=(x,y), pt2=(x+w,y+h), color=(0,255,0), thickness=2)

    # creates a UI windows and displays the image
    cv2.imshow(winname="faces", mat=frame)


# ends processes
webcam.release()
cv2.destroyAllWindows