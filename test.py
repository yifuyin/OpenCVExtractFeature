import cv2 as cv

img = cv.imread("data/input.png")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')
anime_cascade = cv.CascadeClassifier('haarcascades/lbpcascade_animeface.xml')

faces = anime_cascade.detectMultiScale(gray, 1.3, 4)

print('Number of detected faces:', len(faces))
#cv.imshow("Display window", gray)
#k = cv.waitKey(0) # Wait for a keystroke in the window
