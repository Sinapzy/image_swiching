import numpy as np
import cv2 as cv
img1 = cv.imread('C:/Users/Administrator/Desktop/zy/cv/image/re-photos_4157/before.jpg')
img2 = cv.imread('C:/Users/Administrator/Desktop/zy/cv/image/re-photos_4157/after.jpg')
gray1= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
sift1 = cv.xfeatures2d.SIFT_create()
kp1 = sift1.detect(gray1,None)
img1=cv.drawKeypoints(img1,kp1,img1)

gray2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
sift2 = cv.xfeatures2d.SIFT_create()
kp2 = sift2.detect(gray2,None)
img2=cv.drawKeypoints(img2,kp2,img2)
cv.namedWindow('sift_keypoints1',cv.WINDOW_NORMAL  )
cv.imshow('sift_keypoints1',img1)
cv.namedWindow('sift_keypoints2',cv.WINDOW_NORMAL  )
cv.imshow('sift_keypoints2',img2)

cv.waitKey(0)
cv.destroyAllWindows()

