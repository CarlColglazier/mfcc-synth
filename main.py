import jack
import time
import rtmidi
from rtmidi.midiutil import list_output_ports
import random
import numpy
import librosa
import scipy
import sklearn


client = jack.Client('Darwin')
in1 = client.inports.register('input_1')
midiout = rtmidi.MidiOut()
midiout.open_port(6)

sounds = numpy.array([])
data, rate = librosa.load("oboe-g4-PB-loop.wav")
#data, rate = librosa.load("./1st-violins-stc-rr2-g4-PB.wav")


def callback(f):
	global sounds
	a = in1.get_array()
	if len(a) > 0:
		sounds = numpy.append(sounds, a)

client.set_process_callback(callback)

def fitness(sounds):
	windows = 13
	mfcc = librosa.feature.mfcc(data, rate)
	mfcc_in = librosa.feature.mfcc(sounds, 48_000)
	std = numpy.tile(numpy.std(mfcc, axis=1), windows)
	std = numpy.reshape(std, (len(mfcc), windows))
	print()
	diff = numpy.sum(numpy.sqrt(
		numpy.absolute(
			numpy.divide(mfcc[:,0:windows]-mfcc_in[:,0:windows], std)
		))
	)
	return diff/windows

ccs = [
	34, 35, # VCO Pitch
	36, 37, # VCO shape
	48, 49, # VCO octave
	50, 51, # VCO wave
	33,# noise
	39, 40, # gen levels
	43, # filter cutoff
	44, 45, # filter stuff
	41, # cross mod
	42, # pitch EG
	80, 81, # sync/ring
	16, 17, 18, 19, # adsr
	20, 21, 22, 23,
	24, 26, 27,#lfo
	56, 57, 58, #lfo
	82, 83, # key tracking
	84 # filter type
]

def set_cc(params):
	global midiout
	global ccs
	for i, cc in enumerate(ccs):
		cc_msg = [0xB2, cc, params[i]]
		midiout.send_message(cc_msg)

def combine(a, b):
	c = numpy.random.choice([0,1], size=len(a), p=[0.5,0.5])
	child = a * c + b * (1 - c)
	# mutate
	m = numpy.random.choice([0,1], size=len(a), p=[0.9,0.1])
	child = (1 - m) * child + m * numpy.random.randint(0, 128, len(a))
	return child


def run(params):
	global midiout
	global ccs
	# randomize params
	set_cc(params)
	note_on = [0x92, 0x5B, 112] # channel 3, middle C, velocity 112
	note_off = [0x82, 0x5B, 0]
	midiout.send_message(note_on)
	time.sleep(0.5)
	midiout.send_message(note_off)
	time.sleep(0.25)
	midiout.send_message([0xB2, 120, 0])
	return params

POP_SIZE = 13
GENERATIONS = 51

client.activate()
in1.connect("system:capture_3")
params = numpy.random.randint(0, 127, (POP_SIZE, len(ccs)))
for gen in range(25):
	print(f"Generation {gen+1}")
	scores = []
	for i in range(POP_SIZE):
		p = run(params[i])
		f = fitness(sounds)
		print(f)
		sounds = numpy.array([])
		scores.append(1/f)#max(1, f-400))
	s = numpy.array(scores)
	# mess with the scores
	s = s * (s > numpy.median(s)).astype('int')
	s = s / s.sum()
	print(s)
	print("populating the next generation")
	new_params = []
	for i in range(POP_SIZE):
		choices = numpy.random.choice(len(params), size=2, p=s)
		new_params.append(
			combine(params[choices[0]], params[choices[1]])
			#
		)
	params = new_params


client.deactivate()
