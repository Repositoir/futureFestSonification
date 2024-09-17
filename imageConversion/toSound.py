import numpy as np
import sounddevice as sd
from ImgtoHsv import exportThisArrayRED as arrRed
from ImgtoHsv import exportThisArrayBLUE as arrBlue
from ImgtoHsv import exportThisArrayGREEN as arrGreen
from midiutil import MIDIFile
from mingus.core import chords
NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)
CHORDS = {
    "C major":["C", "E", "G"],
    "G major":["G", "B", "D"],
    "A minor": ["A", "C", "E"],
    "F major":["F", "A", "C"],
    "D major":["D", "F#", "A"]
}

def freq_to_midi_note(freq):
    if freq <= 0:
        raise ValueError("Frequency must be a positive value.")
    midi_note = int(69 + 12 * np.log2(freq / 440.0))
    return midi_note

def get_major_triad(midi_note):
    major_third = midi_note + 4
    perfect_fifth = midi_note + 7
    return [midi_note, major_third, perfect_fifth]

arrRed = arrRed.flatten()
midi_notes_from_arrRed = []
for freq in arrRed:
    midi_note = freq_to_midi_note(freq)
    major_triad = get_major_triad(midi_note)
    midi_notes_from_arrRed.append(major_triad)

track = 0
channel = 0
time = 0  # In beats
duration = 1  # In beats
tempo = 180
volume = 127  # 0-127
arpeggio_spacing = 1

MyMIDI = MIDIFile(1)  # One track


MyMIDI.addTempo(track, time, tempo)

for i, triad in enumerate(midi_notes_from_arrRed):
    for j, pitch in enumerate(triad):
        MyMIDI.addNote(track, channel, pitch, time + i * 3 + j * arpeggio_spacing, duration, volume)

# Save the MIDI file
with open("normLst_melody_red.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)

print("MIDI file saved as 'normLst_melody_red.mid'")




# Function to generate a sine wave for a given frequency and duration
def generate_sine_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    return wave

# Function to get the frequency of a note
def get_frequency(note, octave):
    # Frequencies of notes in the 4th octave
    A4 = 440.0
    note_frequencies = {
        'C': A4 * 2**(-9/12),
        'C#': A4 * 2**(-8/12),
        'D': A4 * 2**(-7/12),
        'D#': A4 * 2**(-6/12),
        'E': A4 * 2**(-5/12),
        'F': A4 * 2**(-4/12),
        'F#': A4 * 2**(-3/12),
        'G': A4 * 2**(-2/12),
        'G#': A4 * 2**(-1/12),
        'A': A4,
        'A#': A4 * 2**(1/12),
        'B': A4 * 2**(2/12),
    }
    return note_frequencies[note] * (2 ** (octave - 4))

# Play a note
def play_note(note, octave, duration, sample_rate=44100):
    #frequency = get_frequency(note, octave)
    wave = generate_sine_wave(note, duration, sample_rate)
    sd.play(wave, samplerate=sample_rate)
    sd.wait()  # Wait until the sound finishes playing

def play_chord(notes, octave, duration, sample_rate=44100):
    # Generate the wave for each note in the chord
    chord_wave = sum(generate_sine_wave(int(note), duration, sample_rate) for note in notes)

    # Normalize the wave to avoid clipping
    chord_wave /= len(notes)

    # Play the chord
    sd.play(chord_wave, samplerate=sample_rate)
    sd.wait()  # Wait until the sound finishes playing
