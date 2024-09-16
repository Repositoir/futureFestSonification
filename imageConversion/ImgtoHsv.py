import cv2, time
import matplotlib.pyplot as plt
import numpy as np
from toSound import generate_sine_wave, get_frequency, play_note, play_chord
import sounddevice as sd

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
def smallImg2AvgArr(arr):
   lst2npArray = np.array(arr)
   lst = lst2npArray.transpose((3, 0, 1, 2))
   avg = lst.mean(axis= 1, dtype= int)
   avg = avg.mean(axis=1, dtype= int)
   avg = avg.mean(axis=1, dtype= int)
    # What i did,
    # So i think you wanted an average one RGB pixel value of the small image
    # So i first transposed the orignal matrix to set according to the R, G, B values first
    # Then i took the average over the second column of the matrix, first it averages over all the images
    # second it averages over all the columns in the image
    # third it averages over all the rows in the image
    # Hence finally givng an average of all the pixels!
    # i wish i was smart enough to implement this in a smaeter way, 
    # this definitely is stupid af 
   return avg

def convert2grey(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #plt.imshow(img, cmap="gray")    
    print(img.shape)
    print(np.max(img))
    print(np.min(img))

    return img

def map(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

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

lst[0][0]


#convert to greyscale 
grayLst = convert2grey(image)
print(grayLst[0])
# 28 to 7040
#

normLst = []
for i in range(len(grayLst[0])):
    normNum = map(int(grayLst[0][i]), 0, 255, 28, 7040)
    normLst.append(normNum)
print(normLst)

piano_freqs = np.array([27.5 * (2 ** (i / 12)) for i in range(88)])

def closest_piano_freq(note):
    return piano_freqs[np.abs(piano_freqs - note).argmin()]


def apply_adsr(wave, attack_time, decay_time, sample_rate):
    length = len(wave)
    attack_samples = int(attack_time * sample_rate)
    decay_samples = int(decay_time * sample_rate)

    attack = np.linspace(0, 1, attack_samples)
    sustain = np.ones(length - attack_samples - decay_samples)
    decay = np.linspace(1, 0, decay_samples)
    envelope = np.concatenate([attack, sustain, decay])

    return wave * envelope[:len(wave)]

def generate_harmonic_wave(freq, duration, sample_rate, harmonics=[1, 2, 3], amplitudes=[1, 0.5, 0.25]):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.zeros_like(t)
    for i, harmonic in enumerate(harmonics):
        wave += amplitudes[i] * np.sin(2 * np.pi * freq * harmonic * t)
    wave = wave / np.max(np.abs(wave))
    return wave

def generate_chord(notes, duration, sample_rate):
    chord_wave = np.zeros(int(sample_rate * duration))
    for note in notes:
        freq = closest_piano_freq(note)
        wave = generate_harmonic_wave(freq, duration, sample_rate)
        chord_wave += wave
    chord_wave = chord_wave / len(notes)
    return chord_wave


# for note in normLst:
#     freq = closest_piano_freq(note)
#     wave = generate_harmonic_wave(freq, 1.0, sample_rate=44100)
#     wave_with_adsr = apply_adsr(wave, 0.1, 0.4, sample_rate=44100)
#     sd.play(wave_with_adsr, samplerate=44100)
#     sd.wait()

for i in range(0, len(normLst), 3):
    chord = normLst[i:i+3]
    chord_wave = generate_chord(chord, 1.0, sample_rate=44100)
    chord_wave_with_adsr = apply_adsr(chord_wave, 0.1, 0.4, sample_rate=44100)
    sd.play(chord_wave_with_adsr, samplerate=44100)
    sd.wait()

normLst = [[0 for j in range(len(grayLst[0]))] for i in range(len(grayLst))]
for i in range(len(grayLst)):
    for j in range(len(grayLst[i])):
        normNum = map(int(grayLst[i][j]), 0, 255, 28, 7040)
        normLst[i][j] = normNum
#print(normLst)

