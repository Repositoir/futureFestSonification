import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('img1.tif')

#plt.imshow(img)
#plt.show()

imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#plt.imshow(imgHSV)
plt.hist(imgHSV.ravel(),256,[0,256], label="hist Data")
plt.plot(cv2.calcHist([imgHSV], [0], None, [256], [0, 256]), color='blue')
plt.plot(cv2.calcHist([imgHSV], [1], None, [256], [0, 256]), color='green')
plt.plot(cv2.calcHist([imgHSV], [2], None, [256], [0, 256]), color='red')
plt.show()

