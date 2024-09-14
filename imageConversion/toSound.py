import numpy as np
import sounddevice as sd
from ImgtoHsv import normLst

chords = {
    "C major":["C", "E", "G"],
    "G major":["G", "B", "D"],
    "F major":["F", "A", "C"],
    "D major":["D", "F#", "A"]
}

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

# Example usage
# note = "C"
# octave = 4
# duration = 0.7 # Duration regarding crotchet = 0.7, minim = 1.4, quaver = 0.35
# play_note(note, octave, duration)
# play_note("C", octave, duration)
# play_note("G", octave, duration)
# play_note("G", octave, duration)
# play_note("A", octave, duration)
# play_note("A", octave, duration)
# play_note("G", octave, duration * 2)
# play_note("F", octave, duration)
# play_note("F", octave, duration)
# play_note("E", octave, duration)
# play_note("E", octave, duration)
# play_note("D", octave, duration)
# play_note("D", octave, duration)
# play_note("C", octave, duration * 4)

def play_chord(notes, octave, duration, sample_rate=44100):
    # Generate the wave for each note in the chord
    chord_wave = sum(generate_sine_wave(int(note), duration, sample_rate) for note in notes)

    # Normalize the wave to avoid clipping
    chord_wave /= len(notes)

    # Play the chord
    sd.play(chord_wave, samplerate=sample_rate)
    sd.wait()  # Wait until the sound finishes playing

notes = ["A", "C", "E"]
octave = 4
duration = 5.0  # 1 second duration
#play_chord(["C", "E", "G"], octave, duration)
#play_chord(["G", "B", "D"], octave, duration)
#play_chord(["A", "C", "E"], octave, duration)
for i in normLst:
    play_chord(i, octave, duration)