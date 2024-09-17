import cv2, time
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('./imageConversion/eqImg.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    return np.array(listOfImages)


# below fxn not done!
# What part isn't done?
def smallImg2AvgArr(arr):
    if len(arr.shape) == 3:
        lst = arr.transpose((2, 0, 1))
        avg = np.mean(lst, axis= 1, dtype= int)
        avg = np.mean(avg, axis=1, dtype= int)
    else:
       return "Wrong Input!"
    
    return avg

def avgArrTranspose(arr):
    arr = np.array(arr)
    arr = arr.transpose((1, 0))
    return arr

def convert2grey(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #plt.imshow(img, cmap="gray")    
    print(img.shape)
    print(np.max(img))
    print(np.min(img))

    return img

def map_values(x, in_min, in_max):
  out_min = 28
  out_max = 7040
  return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

def normalize_array(arr):
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8 )
    return norm_arr

def greyMedianArr(arr):
    grayMedianArr = []

    for i in range(len(arr)):
        for j in range(len(arr[0])):
            grayNmld = normalize_array(arr[i][j])  # Ensure this function is defined
            gNmldAvg = np.median(grayNmld)  # Calculate the median of the normalized array
            grayMedianArr.append(gNmldAvg)  # Append the median to the list

    # Optionally, convert grayMedianArr to a NumPy array if needed
    grayMedianArr = np.array(grayMedianArr)

    return grayMedianArr
    
def normalize_3d_array(arr, y_scale= 0.5):
    
    # Compute min and max along the third dimension
    min_vals = np.min(arr, axis=0, keepdims=True)
    max_vals = np.max(arr, axis=0, keepdims=True)
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Replace zero ranges with one to avoid division by zero
    
    # Perform normalization
    normalized_arr = (arr - min_vals) / range_vals
    # Scale it to spread out more
    normalized_arr **= y_scale


    return normalized_arr

height, width, channels = image.shape

print(image.shape)
print(image.shape[0] % 11, image.shape[1] % 22)
print(image[[0], [0]])

W_SIZE = 22
H_SIZE = 11
count = 0


lst = img2SmallImg(image, H_SIZE, W_SIZE)
print(lst.shape)
print(lst[0][0][0])

#get Average values of the 242 images
lstOfAvgs = [smallImg2AvgArr(lst[i]) for i in range(len(lst))]
print(len(lstOfAvgs))

lstRGB = avgArrTranspose(lstOfAvgs)

# plt.plot(lstRGB[0][0], 'o', color= 'red')
# plt.plot(lstRGB[1][0], 'o', color= 'green')
# plt.plot(lstRGB[2][0], 'o', color= 'blue')
# plt.show()

# Map each Color Value to a frequency? 
# Not sure how useful that would be? 
# I was thinking of more, each R-G-B value to be a Chord, ahhh.
# I don't what a triad is lol




# #convert to greyscale 
grayLst = convert2grey(image)
# 28 to 7040
grayLstSmall = img2SmallImg(grayLst, H_SIZE, W_SIZE)

greyMedArr = normalize_3d_array(grayLstSmall, 0.7)
greyMedArr = greyMedArr.transpose((2, 0, 1))
print(greyMedArr.shape)
print(greyMedArr[0][0])
# plt.plot(greyMedArr[0][0], 'o')
# plt.show()

exportThisArrayRED = np.matmul(lstRGB[0], greyMedArr)
exportThisArrayGREEN = np.matmul(lstRGB[1], greyMedArr)
exportThisArrayBLUE = np.matmul(lstRGB[2], greyMedArr)

print(exportThisArrayRED)