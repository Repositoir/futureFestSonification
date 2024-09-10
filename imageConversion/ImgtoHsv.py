from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

image = Image.open('./imageConversion/eqImg.png')
imArr = np.array(image.convert('RGB'))
#image.show()

def img2SmallImg(img, H_SIZE, W_SIZE):
    listOfImages = []
    for ih in range(H_SIZE):
        for iw in range(W_SIZE):
            x = width / W_SIZE * iw
            y = height / H_SIZE * ih
            h = (height / H_SIZE)
            w = (width / W_SIZE)
            #print(x,y,h,w)  

            img2 = img[int(y):int(y+h), int(x):int(x+w)]
            listOfImages.append(img2)

    return listOfImages


def smallImg2AvgArr(arr):
   lst2npArray = np.array(arr)
   lst = lst2npArray.transpose((3, 0, 1, 2))
   
   avg = lst.mean(axis= 0, dtype= int)

   return lst[1]

def convert2grey(img):
    img = ImageOps.grayscale(img)
    img.show()
    imgArr = img.convert('L')
    return np.array(imgArr)


print(imArr)
height, width, channels = imArr.shape
#print(image.shape[0] % 11, image.shape[1] % 22)

W_SIZE = 22
H_SIZE = 11
count = 0


lst = img2SmallImg(imArr, H_SIZE, W_SIZE)
lst2npArray = np.array(lst)
print(lst2npArray.shape)
print(lst2npArray[0][0][0])   #This is one pixel, the top right most pixel

plt.imshow(lst2npArray[0][0])      #This is one column of pixels in the right most side??
plt.show()
print(smallImg2AvgArr(lst))
#plt.show()


#convert to greyscale then map that to sound parts
greyArr = convert2grey(image)
#cv2.waitKey(5000)  

# Window shown waits for any key pressing event
#cv2.destroyAllWindows()
