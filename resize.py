import numpy as np
import cv2 as cv
import cv2
#import matplotlib as plt

img = cv.imread('oko.jpg', 0)
img0 = cv.imread('oko.jpg')

h = img.shape[0]
w = img.shape[1]

nova_w = 1000
ratio = nova_w / w # (alebo nova_h / h)
nova_h = int(h * ratio)

dimensions = (nova_w, nova_h)
img1 = cv.resize(img, dimensions, interpolation=cv.INTER_LINEAR)
img0 = cv.resize(img, dimensions, interpolation=cv.INTER_LINEAR)

kernel = np.ones((5, 5), np.uint8)
img = cv.medianBlur(img1, 5)
img = cv.erode(img1, kernel, iterations=1)
edges = cv.Canny(img1, 200, 200)
cv.imwrite('pupil-canny.jpg', edges)


img2 = cv.imread('pupil-canny.jpg')
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1, 20,
                          param1=50, param2=30, minRadius=61, maxRadius=87)
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