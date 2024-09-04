import cv2, time
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('eqImg.png')
img2 = img

height, width, channels = img.shape
print(img.shape)
print(img.shape[0] % 11, img.shape[1] % 22)

W_SIZE = 22
H_SIZE = 11
count = 0

for ih in range(H_SIZE ):
   for iw in range(W_SIZE ):
   
      x = width/W_SIZE * iw 
      y = height/H_SIZE * ih
      h = (height / H_SIZE)
      w = (width / W_SIZE )
      #print(x,y,h,w)
      img = img[int(y):int(y+h), int(x):int(x+w)]
      count += 1
      plt.imshow(img)
      plt.show()
      NAME = str(time.time()) 
      cv2.imwrite("Output Images/" + str(ih)+str(iw) +  ".png",img)
      img = img2

print(count)

#plt.show()

