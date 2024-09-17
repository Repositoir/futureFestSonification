import cv2, time
import matplotlib.pyplot as plt
import numpy as np
from midiutil import MIDIFile
from mingus.core import chords

image = cv2.imread('./eqImg.png')
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
   lst = arr.transpose((2, 0, 1))
   avg = np.mean(lst, axis= 1, dtype= int)
   avg = np.mean(avg, axis=1, dtype= int)
   return avg.tolist()

def convert2grey(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #plt.imshow(img, cmap="gray")    
    print(img.shape)
    print(np.max(img))
    print(np.min(img))

    return img

def map(x, in_min, in_max):
  out_min = 28
  out_max = 7040
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

lstOfAvgs = [smallImg2AvgArr(lst[i]) for i in range(len(lst))]
plt.imshow(lstOfAvgs)
plt.show()

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

NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)


def freq_to_midi_note(freq):
    if freq <= 0:
        raise ValueError("Frequency must be a positive value.")
    midi_note = int(69 + 12 * np.log2(freq / 440.0))
    return midi_note

def get_major_triad(midi_note):
    major_third = midi_note + 4
    perfect_fifth = midi_note + 7
    return [midi_note, major_third, perfect_fifth]


midi_notes_from_normLst = []
for freq in normLst:
    midi_note = freq_to_midi_note(freq)
    major_triad = get_major_triad(midi_note)
    midi_notes_from_normLst.append(major_triad)

track = 0
channel = 0
time = 0  # In beats
duration = 1  # In beats
tempo = 180
volume = 127  # 0-127
arpeggio_spacing = 1

MyMIDI = MIDIFile(1)  # One track


MyMIDI.addTempo(track, time, tempo)

for i, triad in enumerate(midi_notes_from_normLst):
    for j, pitch in enumerate(triad):
        MyMIDI.addNote(track, channel, pitch, time + i * 3 + j * arpeggio_spacing, duration, volume)

# Save the MIDI file
with open("normLst_melody.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)

print("MIDI file saved as 'normLst_melody.mid'")
