import jack
import time
import rtmidi
from rtmidi.midiutil import list_output_ports
import random
import numpy
import librosa
import scipy
from scipy.io import wavfile
import sklearn
import base64

INIT = 'UFJPR0luaXQgUHJvZ3JhbSAgICCAAIAAAID/AAD/AIAAAACA/wAAgAAAgAD/////AED//5CQMDDAMAAgPQDw/MgPIv//5QBmTfr/////////////////////////////U0VRRLAEAhAANgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=='

INIT_PROG = base64.b64decode(INIT.encode('ascii'))

client = jack.Client('Darwin')
in1 = client.inports.register('input_1')
midiout = rtmidi.MidiOut()
midiout.open_port(6)

sounds = numpy.array([])
rate, data = wavfile.read("./oboe.wav")
print(rate)
#data, rate = librosa.load("./oboe.wav") #"oboe-g4-PB-loop.wav")
#data, rate = librosa.load("./1st-violins-stc-rr2-g4-PB.wav")

change = []
change.extend(range(20, 33))
change.extend(range(33, 44))
change.extend(range(49, 63))

MASK = {
	60: 0xC3,
	62: 0x3F
}


def callback(f):
	global sounds
	a = in1.get_array()
	if len(a) > 0:
		sounds = numpy.append(sounds, a)

client.set_process_callback(callback)

def fitness(sounds):
	windows = 85 #13#?
	mfcc = librosa.feature.mfcc(data, sr=rate)
	mfcc_in = librosa.feature.mfcc(sounds, sr=48000)
	print(mfcc.shape, mfcc_in.shape, rate, len(sounds))
	std = numpy.tile(numpy.std(mfcc[1:20], axis=1), windows)
	std = numpy.reshape(std, (len(mfcc)-1, windows))
	diff = numpy.sum(#numpy.sqrt(
		numpy.absolute(
			numpy.divide(mfcc[1:20,0:windows]-mfcc_in[1:20,0:windows], std)
			#mfcc[:,0:windows]-mfcc_in[:,0:windows]
		)#)
	)
	return diff / windows / (len(mfcc) - 1)

def convert_bytes(inp):
	assert(len(inp) % 7 == 0)
	out = []
	# for each seven numbers in here.
	for i in range(int(len(inp) / 7)):
		offset = 7 * i
		new = [0, 0, 0, 0, 0, 0, 0, 0]
		for n in range(7):
			new[0] += ((inp[offset + n] & 0x80) >> 7) << n
			new[n+1] = inp[offset + n] & 0x7F
		out.extend(new)
	return out

def set_sysex(r):
	global midiout
	global ccs
	global change
	cc_msg = [
		0xF0, 0x42, 0x32, 0x00, 0x01, 0x2c, 0x40
	]
	ne = list(INIT_PROG)
	for i, v in enumerate(change):
		if v in MASK:
			#print(v, r[i] & MASK[v])
			ne[v] = r[i] & MASK[v]
		else:
			ne[v] = r[i] #random.randint(0, 127)
	cc_msg.extend(convert_bytes(ne))
	cc_msg.append(0xF7)
	midiout.send_message(cc_msg)

def combine(a, b):
	c = numpy.random.choice([0,1], size=len(a), p=[0.5,0.5])
	child = a * c + b * (1 - c)
	# mutate
	m = numpy.random.choice([0,1], size=len(a), p=[0.9,0.1])
	child = (1 - m) * child + m * numpy.random.randint(0, 127, len(a))
	return child


def run(params):
	global midiout
	global ccs
	# randomize params
	#set_cc(params)
	set_sysex(params)
	#time.sleep(0.05)
	note_on = [0x92, 0x43, 112]
	note_off = [0x82, 0x43, 0]
	midiout.send_message(note_on)
	time.sleep(0.4)
	midiout.send_message(note_off)
	time.sleep(0.1)
	note_on = [0x92, 0x54, 112]
	note_off = [0x82, 0x54, 0]
	midiout.send_message(note_on)
	time.sleep(0.4)
	midiout.send_message(note_off)
	time.sleep(0.1)
	midiout.send_message([0xB2, 120, 0])
	return params

POP_SIZE = 51
GENERATIONS = 51

client.activate()
in1.connect("system:capture_3")
params = numpy.random.randint(0, 127, (POP_SIZE, len(change)))
for gen in range(GENERATIONS):
	print(f"Generation {gen+1}")
	scores = []
	for i in range(POP_SIZE):
		p = run(params[i])
		f = fitness(sounds)
		print(i, f)
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
