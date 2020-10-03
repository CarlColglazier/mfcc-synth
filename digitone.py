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

client = jack.Client('Darwin')
in1 = client.inports.register('input_1')
midiout = rtmidi.MidiOut()
midiout.open_port(5)

sounds = numpy.array([])

data, rate = librosa.load("./oboe-g4-PB-loop.wav")


def callback(f):
	global sounds
	a = in1.get_array()
	if len(a) > 0:
		sounds = numpy.append(sounds, a)

client.set_process_callback(callback)

def fitness(sounds):
	windows = 75#?
	mfcc = librosa.feature.mfcc(data, sr=rate)
	mfcc_in = librosa.feature.mfcc(sounds, sr=48000)
	#print(mfcc.shape, mfcc_in.shape, rate, len(sounds))
	std = numpy.tile(numpy.std(mfcc[1:20], axis=1), windows)
	std = numpy.reshape(std, (len(mfcc)-1, windows))
	diff = numpy.sum(#numpy.sqrt(
		numpy.absolute(
			#numpy.divide(mfcc[1:20,0:windows]-mfcc_in[1:20,0:windows], std)
			mfcc[:,0:windows]-mfcc_in[:,0:windows]
		)#)
	)
	return diff / windows #/ (len(mfcc) - 1)

def send_nrpn(m_out, msb, lsb, value):#, msb, lsb, value):
	# test: algorithm
	msb_v = value >> 7
	lsb_v = value & 0x07
	m_out.send_message([0xB0, 99, msb])
	m_out.send_message([0xB0, 98, lsb])
	m_out.send_message([0xB0, 6, msb_v])
	m_out.send_message([0xB0, 38, lsb_v])

class Digitone:
	params = {
		'algorithm': {'msb': 1, 'lsb': 72, 'range': 8},
		'ratio_c': {'msb': 1, 'lsb': 73, 'range': 20},
		'ratio_a': {'msb': 1, 'lsb': 74, 'range': 36},
		'ratio_b': {'msb': 1, 'lsb': 75, 'range': 128, 'l': True},
		'harmonics': {'msb': 1, 'lsb': 76},
		'detune': {'msb': 1, 'lsb': 77},
		'feedback': {'msb': 1, 'lsb': 78},
		'mix': {'msb': 1, 'lsb': 79},
		'c_offset': {'msb': 1, 'lsb': 95},
		'a_offset': {'msb': 1, 'lsb': 96},
		'b1_offset': {'msb': 1, 'lsb': 97},
		'b2_offset': {'msb': 1, 'lsb': 98},
		'a_env_att': {'msb': 1, 'lsb': 80},
		'a_env_dec': {'msb': 1, 'lsb': 81},
		'a_env_end': {'msb': 1, 'lsb': 82},
		'a_level': {'msb': 1, 'lsb': 83},
		'b_env_att': {'msb': 1, 'lsb': 84},
		'b_env_dec': {'msb': 1, 'lsb': 85},
		'b_env_end': {'msb': 1, 'lsb': 86},
		'b_level': {'msb': 1, 'lsb': 87},
		'a_delay': {'msb': 1, 'lsb': 88},
		'b_delay': {'msb': 1, 'lsb': 91},
		'filt_freq': {'msb': 1, 'lsb': 20},
		'filt_res': {'msb': 1, 'lsb': 21},
		'filt_type': {'msb': 1, 'lsb': 22, 'range': 4},
		'filt_att': {'msb': 1, 'lsb': 16, 'range': 127},
		'filt_dec': {'msb': 1, 'lsb': 17, 'range': 127},
		'filt_sus': {'msb': 1, 'lsb': 18, 'range': 127},
		'filt_rel': {'msb': 1, 'lsb': 19, 'range': 127},
		'filt_env_dep': {'msb': 1, 'lsb': 24},
		'filt_base': {'msb': 1, 'lsb': 24, 'range': 127},
		'filt_width': {'msb': 1, 'lsb': 25, 'range': 127},
		'amp_att': {'msb': 1, 'lsb': 32, 'range': 127},
		'amp_dec': {'msb': 1, 'lsb': 33, 'range': 127},
		'amp_sus': {'msb': 1, 'lsb': 34, 'range': 127},
		'amp_rel': {'msb': 1, 'lsb': 35, 'range': 127},
		'amp_drive': {'msb': 1, 'lsb': 36}
	}
	def randomize(self):
		nums = []
		for key, v in self.params.items():
			if 'range' in v:
				if 'l' in v:
					r = random.randint(0, v['range'])
				else:
					r = random.randint(0, v['range']) << 7
			else:
				r = random.randint(0, 16383)
			nums.append(r)
		return nums

	def send(self, nums):
		assert(len(nums) == len(self.params))
		i = 0
		for key, v in self.params.items():
			send_nrpn(midiout, v['msb'], v['lsb'], nums[i])
			i += 1

client.activate()
in1.connect("system:capture_4")
dt = Digitone()

def run(n):
	global sounds
	dt.send(n)
	#print(n)
	time.sleep(0.2)
	note_on = [0x90, 0x43, 112]
	note_off = [0x80, 0x43, 0]
	sounds = numpy.array([])
	midiout.send_message(note_on)
	time.sleep(0.9)
	midiout.send_message(note_off)
	time.sleep(0.5)
	return n

def combine(a, b):
	c = numpy.random.choice([0,1], size=len(a), p=[0.5,0.5])
	child = a * c + b * (1 - c)
	return child

# main function
POP_SIZE = 151
GENERATIONS = 51

mymin= 100.0
mmv = []
params = []
for i in range(POP_SIZE):
	params.append(dt.randomize())
for gen in range(GENERATIONS):
	print(f"Gen {gen+1}")
	scores = []
	for i in range(POP_SIZE):
		p = run(params[i])
		f = fitness(sounds)
		sounds = numpy.array([])
		scores.append(1/f)
		print(f)
	s = numpy.array(scores)
	s = s * (s > numpy.median(s)).astype('int')
	s = s / s.sum()
	print("populating the next generation")
	new_params = []
	for i in range(POP_SIZE):
		choices = numpy.random.choice(len(params), size=2, p=s)
		new_params.append(
			combine(params[choices[0]], params[choices[1]])
		)
	params = new_params
	print(f"Mean: {1/numpy.mean(scores)}")

#dt.send(mmv)
#print(mymin)
client.deactivate()
