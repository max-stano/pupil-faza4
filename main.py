import numpy as np
import cv2 as cv
import cv2
#import matplotlib as plt

img = cv.imread('pupil2.jpg', 0)
img0 = cv.imread('pupil2.jpg')


kernel = np.ones((5, 5), np.uint8)
img = cv.medianBlur(img, 5)
img = cv.erode(img, kernel, iterations=1)
edges = cv.Canny(img, 200, 200)
cv.imwrite('pupil-canny.jpg', edges)


img2 = cv.imread('pupil-canny.jpg')
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1, 20,
                          param1=50, param2=30, minRadius=80, maxRadius=200)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    cv.circle(img2,(i[0], i[1]), i[2], (0, 255, 0), 2)
    cv.circle(img2,(i[0], i[1]), 2, (0, 0, 255), 3)
cv.imshow('detected circles',img2)


hsv = cv.cvtColor(img2, cv2.COLOR_BGR2HSV)
lower_bound = np.array([50, 20, 20])
upper_bound = np.array([100, 255, 255])
mask = cv.inRange(hsv, lower_bound, upper_bound)

segmented_img = cv.bitwise_and(img2, img2, mask=mask)

contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
output = cv.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

output = cv.drawContours(img0, contours, -1, (0, 0, 255), 3)

cv2.imshow("Output", output)

cv.waitKey(0)
cv.destroyAllWindows()