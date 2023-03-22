#! /Users/toddingalls/Developer/Python/synthesis-env/env/bin/python


from typing import Optional
from contextlib import contextmanager
import subprocess
import functools
from functools import partial
from sndlib import *

import numpy as np
import numpy.typing as npt

import os

import types
from enum import Enum, IntEnum

mus_initialize()
mus_sound_initialize() 


# ENUMS
class Interp(IntEnum):
	NONE = MUS_INTERP_NONE
	LINEAR = MUS_INTERP_LINEAR
	SINUSOIDAL = MUS_INTERP_SINUSOIDAL
	ALL_PASS = MUS_INTERP_ALL_PASS
	LAGRANGE = MUS_INTERP_LAGRANGE
	BEZIER = MUS_INTERP_BEZIER
	HERMITE = MUS_INTERP_HERMITE
	
class Window(IntEnum):
	RECTANGULAR = MUS_RECTANGULAR_WINDOW
	HANN = MUS_HANN_WINDOW
	WELCH = MUS_WELCH_WINDOW
	PARZEN = MUS_PARZEN_WINDOW
	BARTLETT = MUS_BARTLETT_WINDOW
	HAMMING = MUS_HAMMING_WINDOW
	BLACKMAN2 = MUS_BLACKMAN2_WINDOW
	BLACKMAN3 = MUS_BLACKMAN3_WINDOW
	BLACKMAN4 = MUS_BLACKMAN4_WINDOW
	EXPONENTIAL = MUS_EXPONENTIAL_WINDOW
	RIEMANN = MUS_RIEMANN_WINDOW
	KAISER = MUS_KAISER_WINDOW
	CAUCHY = MUS_CAUCHY_WINDOW
	POISSON = MUS_POISSON_WINDOW
	GAUSSIAN = MUS_GAUSSIAN_WINDOW
	TUKEY = MUS_TUKEY_WINDOW
	DOLPH_CHEBYSHEV = MUS_DOLPH_CHEBYSHEV_WINDOW
	HANN_POISSON = MUS_HANN_POISSON_WINDOW
	CONNES = MUS_CONNES_WINDOW
	SAMARAKI = MUS_SAMARAKI_WINDOW
	ULTRASPHERICAL = MUS_ULTRASPHERICAL_WINDOW
	BARTLETT_HANN = MUS_BARTLETT_HANN_WINDOW
	BOHMAN = MUS_BOHMAN_WINDOW
	FLAT_TOP = MUS_FLAT_TOP_WINDOW
	BLACKMAN5 = MUS_BLACKMAN5_WINDOW
	BLACKMAN6 = MUS_BLACKMAN6_WINDOW
	BLACKMAN7 = MUS_BLACKMAN7_WINDOW
	BLACKMAN8 = MUS_BLACKMAN8_WINDOW
	BLACKMAN9 = MUS_BLACKMAN9_WINDOW
	BLACKMAN10 = MUS_BLACKMAN10_WINDOW
	RV2 = MUS_RV2_WINDOW
	RV3 = MUS_RV3_WINDOW
	RV4 = MUS_RV4_WINDOW
	MLT_SINE = MUS_MLT_SINE_WINDOW
	PAPOULIS = MUS_PAPOULIS_WINDOW
	DPSS = MUS_DPSS_WINDOW
	SINC = MUS_SINC_WINDOW
	
class Spectrum(IntEnum):
	IN_DB = MUS_SPECTRUM_IN_DB
	NORMALIZED = MUS_SPECTRUM_NORMALIZED
	RAW = MUS_SPECTRUM_RAW
	
class Polynomial(IntEnum):
	EITHER_KIND = MUS_CHEBYSHEV_EITHER_KIND 
	FIRST_KIND = MUS_CHEBYSHEV_FIRST_KIND
	SECOND_KIND = MUS_CHEBYSHEV_SECOND_KIND 
	BOTH_KINDS = MUS_CHEBYSHEV_BOTH_KINDS

class Header(IntEnum):
	UNKNOWN_HEADER = MUS_UNKNOWN_HEADER
	NEXT = MUS_NEXT
	AIFC = MUS_AIFC
	RIFF = MUS_RIFF
	RF64 = MUS_RF64
	BICSF = MUS_BICSF
	NIST = MUS_NIST
	INRS = MUS_INRS
	ESPS = MUS_ESPS
	SVX = MUS_SVX
	VOC = MUS_VOC
	SNDT = MUS_SNDT
	RAW = MUS_RAW
	SMP = MUS_SMP
	AVR = MUS_AVR
	IRCAM = MUS_IRCAM
	SD1 = MUS_SD1
	SPPACK = MUS_SPPACK
	MUS10 = MUS_MUS10
	HCOM = MUS_HCOM
	PSION = MUS_PSION
	MAUD = MUS_MAUD
	IEEE = MUS_IEEE
	MATLAB = MUS_MATLAB
	ADC = MUS_ADC
	MIDI = MUS_MIDI
	SOUNDFONT = MUS_SOUNDFONT
	GRAVIS = MUS_GRAVIS
	COMDISCO = MUS_COMDISCO
	GOLDWAVE = MUS_GOLDWAVE
	SRFS = MUS_SRFS
	MIDI_SAMPLE_DUMP = MUS_MIDI_SAMPLE_DUMP
	DIAMONDWARE = MUS_DIAMONDWARE
	ADF = MUS_ADF
	SBSTUDIOII = MUS_SBSTUDIOII
	DELUSION = MUS_DELUSION
	FARANDOLE = MUS_FARANDOLE
	SAMPLE_DUMP = MUS_SAMPLE_DUMP
	ULTRATRACKER = MUS_ULTRATRACKER
	YAMAHA_SY85 = MUS_YAMAHA_SY85
	YAMAHA_TX16W = MUS_YAMAHA_TX16W
	DIGIPLAYER = MUS_DIGIPLAYER
	COVOX = MUS_COVOX
	AVI = MUS_AVI
	OMF = MUS_OMF
	QUICKTIME = MUS_QUICKTIME
	ASF = MUS_ASF
	YAMAHA_SY99 = MUS_YAMAHA_SY99
	KURZWEIL_2000 = MUS_KURZWEIL_2000
	AIFF = MUS_AIFF
	PAF = MUS_PAF
	CSL = MUS_CSL
	FILE_SAMP = MUS_FILE_SAMP
	PVF = MUS_PVF
	SOUNDFORGE = MUS_SOUNDFORGE
	TWINVQ = MUS_TWINVQ
	AKAI4 = MUS_AKAI4
	IMPULSETRACKER = MUS_IMPULSETRACKER
	KORG = MUS_KORG
	NVF = MUS_NVF
	CAFF = MUS_CAFF
	MAUI = MUS_MAUI
	SDIF = MUS_SDIF
	OGG = MUS_OGG
	FLAC = MUS_FLAC
	SPEEX = MUS_SPEEX
	MPEG = MUS_MPEG
	SHORTEN = MUS_SHORTEN
	TTA = MUS_TTA
	WAVPACK = MUS_WAVPACK
	SOX = MUS_SOX
	NUM_HEADERS = MUS_NUM_HEADERS
	
class Sample(IntEnum):
	UNKNOWN_SAMPLE = MUS_UNKNOWN_SAMPLE
	BSHORT = MUS_BSHORT
	MULAW = MUS_MULAW
	BYTE = MUS_BYTE
	BFLOAT = MUS_BFLOAT
	BINT = MUS_BINT
	ALAW = MUS_ALAW
	UBYTE = MUS_UBYTE
	B24INT = MUS_B24INT
	BDOUBLE = MUS_BDOUBLE
	LSHORT = MUS_LSHORT
	LINT = MUS_LINT
	LFLOAT = MUS_LFLOAT
	LDOUBLE = MUS_LDOUBLE
	UBSHORT = MUS_UBSHORT
	ULSHORT = MUS_ULSHORT
	L24INT = MUS_L24INT
	BINTN = MUS_BINTN
	LINTN = MUS_LINTN
	BFLOAT_UNSCALED = MUS_BFLOAT_UNSCALED
	LFLOAT_UNSCALED = MUS_LFLOAT_UNSCALED
	BDOUBLE_UNSCALED = MUS_BDOUBLE_UNSCALED
	LDOUBLE_UNSCALED = MUS_LDOUBLE_UNSCALED
	NUM_SAMPLES = MUS_NUM_SAMPLES


INPUTCALLBACK = CFUNCTYPE(c_double, c_void_p, c_int)
EDITCALLBACK = CFUNCTYPE(c_int, c_void_p)
ANALYSISCALLBACK = CFUNCTYPE(c_bool, c_void_p, CFUNCTYPE(c_double, c_void_p, c_int))
SYNTHESISCALLBACK = CFUNCTYPE(c_double, c_void_p)
#FF = CFUNCTYPE(c_double, c_double)

MUS_CLM_DEFAULT_TABLE_SIZE = 512

# is this a hack? i don't know. this seems to add the properties i want
MUS_ANY_POINTER = POINTER(mus_any)

	
get_mus_frequency = lambda s: mus_frequency(s)
set_mus_frequency = lambda s,v : mus_set_frequency(s,v)
get_mus_phase = lambda s: mus_phase(s)
set_mus_phase = lambda s,v : mus_set_phase(s,v)
get_mus_length = lambda s: mus_length(s)
set_mus_length = lambda s,v : mus_set_length(s,v)
get_mus_increment = lambda s: mus_increment(s)
set_mus_increment =  lambda s,v : mus_set_increment(s,v)
get_mus_location = lambda s: mus_location(s)
set_mus_location = lambda s,v : mus_set_location(s,v)

get_mus_offset = lambda s: mus_offset(s)
set_mus_offset = lambda s,v : mus_set_offset(s,v)
get_mus_channels = lambda s: mus_channels(s)
set_mus_channels = lambda s,v : mus_set_channels(s,v)
get_mus_interp_type = lambda s: mus_interp_type(s)
get_mus_width = lambda s: mus_width(s)
set_mus_width = lambda s,v : mus_set_width(s,v)
get_mus_order = lambda s: mus_order(s)
set_mus_order = lambda s,v : mus_set_order(s,v)
get_mus_scaler = lambda s: mus_scaler(s)
set_mus_scaler = lambda s,v : mus_set_scaler(s,v)
get_mus_feedback = lambda s: mus_feedback(s)
set_mus_feedback = lambda s,v : mus_set_feedback(s,v)
get_mus_feedforward = lambda s: mus_feedforward(s)
set_mus_feedforward = lambda s,v : mus_set_feedforward(s,v)
get_mus_hop = lambda s: mus_hop(s)
set_mus_hop = lambda s,v : mus_set_hop(s,v)
get_mus_ramp = lambda s: mus_ramp(s)
set_mus_ramp = lambda s,v : mus_set_ramp(s,v)
get_mus_filename =  lambda s: mus_filename(s)

def get_mus_data(gen: MUS_ANY_POINTER):
	size = mus_length(gen)
	p = np.ctypeslib.as_array(mus_data(gen), shape=size)
	data = np.copy(p)
	return data
	
def set_mus_data(gen: mus_any, data: npt.NDArray[np.float64]):
	return mus_set_data(gen, data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
	
def get_mus_xcoeffs(gen: MUS_ANY_POINTER):
	size = mus_length(gen)
	p = np.ctypeslib.as_array(mus_xcoeffs(gen), shape=size)
	xcoeffs = np.copy(p)
	return xcoeffs
	
def get_mus_ycoeffs(gen: MUS_ANY_POINTER):
	size = mus_length(gen)
	p = np.ctypeslib.as_array(mus_ycoeffs(gen), shape=size)
	ycoeffs = np.copy(p)
	return ycoeffs


MUS_ANY_POINTER.mus_frequency = property(get_mus_frequency, set_mus_frequency, None)
MUS_ANY_POINTER.mus_phase = property(get_mus_phase, set_mus_phase, None)
MUS_ANY_POINTER.mus_length = property(get_mus_length, set_mus_length, None)
MUS_ANY_POINTER.mus_increment = property(get_mus_increment, set_mus_increment, None)
MUS_ANY_POINTER.mus_location = property(get_mus_location, set_mus_location, None)
MUS_ANY_POINTER.mus_data = property(get_mus_data, set_mus_data)
MUS_ANY_POINTER.mus_xcoeffs = property(get_mus_xcoeffs)
MUS_ANY_POINTER.mus_ycoeffs = property(get_mus_ycoeffs)
MUS_ANY_POINTER.get_xcoeff = lambda s,i : mus_xcoeff(s, i)
MUS_ANY_POINTER.set_xcoeff = lambda s,i,v : mus_set_xcoeff(s, i, v)
MUS_ANY_POINTER.get_ycoeff = lambda s,i : mus_ycoeff(s, i)
MUS_ANY_POINTER.set_ycoeff = lambda s,i,v : mus_set_ycoeff(s, i, v)
MUS_ANY_POINTER.mus_offset = property(get_mus_offset, set_mus_offset, None)
MUS_ANY_POINTER.mus_channels = property(get_mus_channels, set_mus_channels, None)
MUS_ANY_POINTER.mus_interp_type = property(get_mus_interp_type,None, None) # not setable
MUS_ANY_POINTER.mus_width = property(get_mus_width, set_mus_width, None)
MUS_ANY_POINTER.mus_order = property(get_mus_order, set_mus_order, None)
MUS_ANY_POINTER.mus_scaler = property(get_mus_scaler, set_mus_scaler, None)
MUS_ANY_POINTER.mus_feedback = property(get_mus_feedback, set_mus_feedback, None)
MUS_ANY_POINTER.mus_feedforward = property(get_mus_feedforward, set_mus_feedforward, None)
MUS_ANY_POINTER.mus_hop = property(get_mus_hop, set_mus_hop, None)
MUS_ANY_POINTER.mus_ramp = property(get_mus_ramp, set_mus_ramp, None)
MUS_ANY_POINTER.mus_filename = property(get_mus_filename, None, None) # not setable

def ff(s):
	print(f'free {mus_describe(s)}')
	mus_free(s)


MUS_ANY_POINTER.__del__ = lambda s : ff(s)
#MUS_ANY_POINTER.__str__ = lambda s : str(mus_describe(s).data)
# 
# def mus_any_free(s):
# 	print("free")
# 	mus_free(s)



# MUS_EXPORT const char *mus_interp_type_to_string(int type);

def radians2hz(radians: float):
	return mus_radians_to_hz(radians)

def hz2radians(hz: float):
	return mus_hz_to_radians(hz)

def degrees2radians(degrees: float):
	return mus_degrees_to_radians(degrees)
	
def radians2degrees(radians: float):
	return mus_radians_to_degrees(radians)
	
def db2linear(x: float):
	return mus_db_to_linear(x)
	
def linear2db(x: float):
	return mus_linear_to_db(x)
	
def odd_multiple(x: float, y: float):
	return mus_odd_multiple(x,y)
	
def even_multiple(x: float, y: float):
	return mus_even_multiple(x,y)
	
def odd_weight(x: float):
	return mus_odd_weight(x)
	
def even_weight(x: float):
	mus_even_weight(x)
	
def get_srate():
	return mus_srate()
	
def set_srate(r: float):
	return mus_set_srate(r)
	
def seconds2samples(secs: float):
	return mus_seconds_to_samples(secs)

def samples2seconds(samples: int):
	return mus_samples_to_seconds(samples)

def get_mus_float_equal_fudge_factor():
	return mus_float_equal_fudge_factor()
	
def get_mus_array_print_length():
	return mus_array_print_length()
	
def set_mus_array_print_length(x: int):
	return mus_set_array_print_length(x)
	
def ring_modulate(s1: float, s2: float):
	return mus_ring_modulate(s1, s2)
	
def amplitude_modulate(s1: float, s2: float, s3: float):
	return mus_amplitude_modulate(s1, s2, s3)
	
def contrast_enhancement(sig: float, index: float):
	return mus_contrast_enhancement(sig, index)
	
def dot_product(data1: npt.NDArray[np.float64], data2: npt.NDArray[np.float64]):
	if isinstance(data1, list):
		data1 = np.array(data1, dtype=np.double)	
		
	if isinstance(data2, list):
		data2 = np.array(data2, dtype=np.double)	
		
	return mus_dot_product(data1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), data2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(data1))
	
def polynomial(coeffs: npt.NDArray[np.float64], x: float):
	if isinstance(coeffs, list):
		coeffs = np.array(coeffs, dtype=np.double)
	return mus_polynomial(coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), x, len(coeffs))

def array_interp(fn: npt.NDArray[np.float64], x: float, size: int):
	if isinstance(fn, list):
		fn = np.array(fn, dtype=np.double)

	return mus_array_interp(fn.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), x, size)

def bessi0(x: float):
	mus_bessi0(x)
	
def mus_interpolate(type, x: float, v: npt.NDArray[np.float64], size: int, y1: float):
	if isinstance(v, list):
		v = np.array(v, dtype=np.double)
	return mus_interpolate(x, v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size, y1)
	
	
def mus_fft(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64], fftsize: int, sign: int):
	if isinstance(rdat, list):
		rdat = np.array(rdat, dtype=np.double)
	if isinstance(idat, list):
		idat = np.array(idat, dtype=np.double)
	return mus_fft(rdat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), idat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), fftsize, sign)

def make_fft_window(type: int, size: int, beta: Optional[float]=0.0, alpha: Optional[float]=0.0):
	win = np.zeros(size, dtype=np.double)
	mus_make_fft_window_with_window(type, size, beta, alpha, win.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
	return win

def rectangular2polar(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64]):
	if isinstance(rdat, list):
		rdat = np.array(rdat, dtype=np.double)
	if isinstance(idat, list):
		idat = np.array(idat, dtype=np.double)
	size = len(rdat)
	rl = np.copy(rdat)
	im = np.copy(idat)
	mus_rectangular_to_polar(rl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size)
	return rl, img
	
def rectangular2magnitudes(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64]):
	if isinstance(rdat, list):
		rdat = np.array(rdat, dtype=np.double)
	if isinstance(idat, list):
		idat = np.array(idat, dtype=np.double)
	size = len(rdat)
	rl = np.copy(rdat)
	im = np.copy(idat)
	mus_rectangular_to_magnitudes(rl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size)
	return rl, img
	
def polar2rectangular(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64]):
	if isinstance(rdat, list):
		rdat = np.array(rdat, dtype=np.double)
	if isinstance(idat, list):
		idat = np.array(idat, dtype=np.double)
	size = len(rdat)
	rl = np.copy(rdat)
	im = np.copy(idat)
	mus_polar_to_rectangular(rl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size)
	return rl, img

def spectrum(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64], window: npt.NDArray[np.float64], norm_type: int):
	if isinstance(rdat, list):
		rdat = np.array(rdat, dtype=np.double)
	if isinstance(idat, list):
		idat = np.array(idat, dtype=np.double)
	if isinstance(window, list):
		window = np.array(window, dtype=np.double)
	size = len(rdat)
	rl = np.copy(rdat)
	mus_spectrum(rl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), idat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), window.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size, norm_type)
	return rl

def convolution(rl1: npt.NDArray[np.float64], rl2: npt.NDArray[np.float64]):
	if isinstance(rl1, list):
		rl1 = np.array(rl1, dtype=np.double)
	if isinstance(rl2, list):
		rl2 = np.array(rl2, dtype=np.double)
	size = len(rl1)
	rl1_1 = np.copy(rl1)
	rl2_1 = np.copy(rl2)
	mus_convolution(rl1_1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), rl2_1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size)
	return rl1_1

def autocorrelate(data: npt.NDArray[np.float64]):
	if isinstance(data, list):
		data = np.array(data, dtype=np.double)
	size = len(data)
	dt = data.copy()
	mus_autocorrelate(dt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size)
	return dt
	
def correlate(data1: npt.NDArray[np.float64], data2: npt.NDArray[np.float64]):
	if isinstance(data1, list):
		data1 = np.array(data1, dtype=np.double)
	if isinstance(data2, list):
		data2 = np.array(data2, dtype=np.double)
	size = len(data1)
	dt1 = data1.copy()
	dt2 = data2.copy()
	mus_correlate(dt1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), dt2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size)
	return dt

def cepstrum(data: npt.NDArray[np.float64]):
	if isinstance(data, list):
		data = np.array(data, dtype=np.double)
	size = len(data)
	dt1 = data.copy()
	mus_cepstrum(dt1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size)
	return dt1
	
	
#########################################
def file2array(filename):
	length = mus_sound_framples(filename)
	chans = mus_sound_chans(filename)
	out = np.zeros((chans, length), dtype=np.double)
	for i in range(chans):
		mus_file_to_array(filename,i, 0, length, out[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
	return out
	
#MUS_EXPORT const char *mus_array_to_file_with_error(const char *filename, mus_float_t *ddata, mus_long_t len, int srate, int channels);
	
#  mus_array_to_file.argtypes = [String, POINTER(mus_float_t), mus_long_t, c_int, c_int]
def array2file(arr, filename, sr=None):
	if not sr:
		sr = mus_srate()
	chans = np.shape(arr)[0]
	length = np.shape(arr)[1]
	flatarray = arr.flatten(order='F')
	return mus_array_to_file(filename, flatarray.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),length, sr, chans)

	
class Sound(object):
	output = None
	reverb = None
	
	
	
	def __init__(self, filename, 
						channels=1, 
						srate=44100., 
						sample_type = Sample.L24INT,
						header_type = Header.AIFF,
						comment = False,
						verbose = False,
						reverb = None,
						revfile = "test.rev",
						reverb_data = None,
						reverb_channels = 1,
						continue_old_file = False,
						statistics = False,
						scaled_to = False,
						scaled_by = False,
						play = True,
						clipped = False,
						notehook = False,
						ignore_output = False):
		self.filename = filename
		self.channels = channels
		self.srate = srate
		self.sample_type = sample_type
		self.header_type = header_type
		self.comment = comment
		self.verbose = verbose
		self.reverb = reverb
		self.revfile = "test.rev"
		self.reverb_data = reverb_data
		self.reverb_channels = reverb_channels
		self.continue_old_file = continue_old_file
		self.statistics = statistics
		self.scaled_to = scaled_to
		self.scaled_by = scaled_by
		self.play = play
		self.clipped = clipped
		self.notehook = notehook
		self.ignore_output = ignore_output
		self.output_to_file = isinstance(self.filename, str)
		self.reverb_to_file = self.reverb and isinstance(self.filename, str)
		self.old_srate = get_srate()
	def __enter__(self):
		print("enter")
		# in original why use reverb-1?
		if self.output_to_file :
			# writing to File
			#continue_sample2file
			if self.continue_old_file:
				Sound.output = continue_sample2file(self.filename)
				set_srate(mus_sound_srate(self.filename))
				
			else:
				set_srate(self.srate)
				Sound.output = make_sample2file(self.filename,self.channels, format=self.sample_type , type=self.header_type)
		else:
			print("can't write to array yet")
			
			
		if self.reverb_to_file:
			if self.continue_old_file:
				Sound.reverb = continue_sample2file(self.revfile)
			else:
				Sound.reverb = make_sample2file(self.revfile,self.reverb_channels, format=self.sample_type , type=self.header_type)

		return self
		
	def __exit__(self, *args):
		print("exit")
		
		
		if self.reverb: 
			if self.reverb_to_file:
				mus_close(Sound.reverb)
				Sound.reverb = make_file2sample(self.revfile)
				#print(self.reverb)
				self.reverb()
				mus_close(Sound.reverb)
				
		if self.output_to_file:
			mus_close(Sound.output)
			
			
		# Statistics and scaling go here	
						
		if self.play :
			print("play")
			subprocess.run(["afplay",self.filename])



		# need some safety if errors
		
		
		set_srate(self.old_srate)











##########################################	

# oscil
# MUS_EXPORT mus_float_t mus_oscil_unmodulated(mus_any *ptr);
# MUS_EXPORT mus_float_t mus_oscil_fm(mus_any *ptr, mus_float_t fm);
# MUS_EXPORT mus_float_t mus_oscil_pm(mus_any *ptr, mus_float_t pm);
def make_oscil(	frequency: Optional[float]=0., initial_phase: Optional[float] = 0.0):
	return mus_make_oscil(frequency, initial_phase)
	
def oscil(os: MUS_ANY_POINTER, fm: Optional[float]=None, pm: Optional[float]=None):
	if not fm:
		if not pm:
			return mus_oscil_unmodulated(os)
		else: 
			return mus_oscil_pm(os, pm)
	else:
		return mus_oscil_fm(os, fm)

	
def is_oscil(os: MUS_ANY_POINTER):
	return mus_is_oscil(os)
	
	
# oscil bank
def make_oscil_bank(freqs: npt.NDArray[np.float64], 
					phases: npt.NDArray[np.float64], 
					amps: npt.NDArray[np.float64], 
					stable: Optional[bool]=False):
	if isinstance(freqs, list):
		freqs = np.array(freqs, dtype=np.double)
		
	if isinstance(phases, list):
		phases = np.array(phases, dtype=np.double)
		
	if isinstance(amps, list):
		amps = np.array(amps, dtype=np.double)	
			
	return mus_make_oscil_bank(freqs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), phases.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), amps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), stable)

def oscil_bank(os: MUS_ANY_POINTER, fms: npt.NDArray[np.float64]):
	if isinstance(fms, list):
		fms = np.array(fms, dtype=np.double)
		
	return mus_oscil_bank(os, fms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
	
def is_oscil_bank(os: MUS_ANY_POINTER):
	return mus_is_oscil_bank(os)
	
# env
#    mus_make_env.argtypes = [POINTER(mus_float_t), c_int, mus_float_t, mus_float_t, mus_float_t, mus_float_t, mus_long_t, POINTER(mus_float_t)]
#MUS_EXPORT mus_any *mus_make_env(mus_float_t *brkpts, int npts, mus_float_t scaler, mus_float_t offset, mus_float_t base, mus_float_t duration, mus_long_t end, mus_float_t *odata);
def make_env(envelope: npt.NDArray[np.float64], 
				scaler: Optional[float]=1.0, 
				duration: Optional[float]=1.0, 
				offset: Optional[float]=0.0, 
				base: Optional[float]=1.0, 
				length: Optional[int]=0):

	if length > 0:
		duration = samples2seconds(length)
	

	if isinstance(envelope, list):
		contents = (c_double * len(envelope))(*envelope)
	
	if isinstance(envelope, np.ndarray):
		contents = envelope.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

	return mus_make_env(contents, len(envelope) // 2, scaler, offset, base, duration, 0, None)
	
def env(e: MUS_ANY_POINTER):
	return mus_env(e)
	
def is_env(e: MUS_ANY_POINTER):
	return mus_is_env(e)
	
def env_interp(x: float, env: MUS_ANY_POINTER):
	return mus_env_interp(x, env)
	
def env_any(e: MUS_ANY_POINTER, connection_function):
	return mus_env_any(e, FF(connection_function))
	
def make_pulsed_env(envelope: npt.NDArray[np.float64], duration, frequency):
	if isinstance(envelope, list):
		contents = (c_double * len(envelope))(*envelope)
	
	if isinstance(envelope, np.ndarray):
		contents = envelope.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

	pl = mus_make_pulse_train(frequency, 1.0, 0.0)
	ge = make_env(contents, scaler=1.0, duration=duration)
	return mus_make_pulsed_env(ge, pl)
	
def pulse_env(gen: MUS_ANY_POINTER, fm: Optional[float]=None):
	if(fm):
		return mus_pulsed_env(gen, fm)
	else:
		return mus_pulsed_env_unmodulated(gen)
		
# TODO envelope-interp different than env-interp

# table lookup

def make_table_lookup(frequency: Optional[float]=0.0, 
						initial_phase: Optional[float]=0.0, 
						wave: npt.NDArray[np.float64]=None, 
						size: Optional[int]=512, 
						type: Optional[int]=Interp.LINEAR):
	if isinstance(wave, list):
		contents = (c_double * len(wave))(*wave)
	
	if isinstance(wave, np.ndarray):
		contents = wave.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
					
		
	return mus_make_table_lookup(frequency, initial_phase, contents, size, type)
	
def table_lookup(tl: MUS_ANY_POINTER, fm_input: Optional[float]=None):
	if fm_input:
		return mus_table_lookup(tl, fm_input)
	else:
		return mus_table_lookup_unmodulated(tl)
		
def is_table_lookup(tl: MUS_ANY_POINTER):
	return mus_is_table_lookup(tl)

# TODO make-table-lookup-with-env

def partials2wave(partials: npt.NDArray[np.float64], 
					wave: Optional[npt.NDArray[np.float64]]=None, 
					norm: Optional[bool]=True ):
					
	if isinstance(partials, list):
		partials = np.array(partials, dtype=np.double)
		
	if isinstance(wave, list):
		wave = np.array(wave, dtype=np.double)
						
	if (not wave):
		wave = np.zeros(MUS_CLM_DEFAULT_TABLE_SIZE)
	mus_partials_to_wave(partials.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(partials) // 2, wave.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(wave), norm)
	return wave
	
def phase_partials2wave(partials: npt.NDArray[np.float64], 
						wave: Optional[npt.NDArray[np.float64]]=None, 
						norm: Optional[bool]=True ):
	if isinstance(partials, list):
		partials = np.array(partials, dtype=np.double)					
	if (not wave):
		wave = np.zeros(MUS_CLM_DEFAULT_TABLE_SIZE)
	mus_partials_to_wave(partials.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(partials) // 3, wave.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(wave), norm)
	return wave
	
	
	
#polywave polyshape

def make_polywave(frequency: float, 
					partials: Optional[npt.NDArray[np.float64]]=np.array([1.,1.]), 
					type: Optional[int]=Polynomial.FIRST_KIND, 
					xcoeffs: npt.NDArray[np.float64]=None, 
					ycoeffs: npt.NDArray[np.float64]=None):
	
	if isinstance(partials, list):
		prtls = (c_double * len(partials))(*partials)
			
	if isinstance(xcoeffs, list):
		xc = (c_double * len(xcoeffs))(*xcoeffs)
		
	if isinstance(ycoeffs, list):
		yc = (c_double * len(ycoeffs))(*ycoeffs)
		
	if isinstance(partials, np.ndarray):
		prtls = partials.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
			
	if isinstance(xcoeffs, np.ndarray):
		xc = xcoeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		
	if isinstance(ycoeffs, np.ndarray):
		yc = ycoeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
			
			

	
	
					
	if(xcoeffs and ycoeffs): # should check they are same length
		return mus_make_polywave_tu(frequency,xc,yc, len(xcoeffs))
	else:
		print(*partials)
		return mus_make_polywave(frequency, prtls, len(partials), type)
		
def polywave(w: MUS_ANY_POINTER, fm: Optional[float]=None):
	if fm:
		return mus_polywave(w, fm)
	else:
		return mus_polywave_unmodulated(w)
		
def is_polywave(w: POINTER(mus_any)):
	return mus_is_polywave(w)


# seems like in docs should be a coeffs argument but don't see how
def make_polyshape(frequency: float, initial_phase: float, 
					partials: Optional[npt.NDArray[np.float64]]=np.array([1.,1.]), 
					kind: Optional[int]=Polynomial.FIRST_KIND):
	
		
	if isinstance(partials, list):
		prtls = (c_double * len(partials))(*partials)
	
	if isinstance(partials, np.ndarray):
		prtls = partials.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

	p = mus_partials_to_polynomial(len(partials), prtls, kind)
	return mus_make_polyshape(frequency, initial_phase, p, len(partials))


def polyshape(w: MUS_ANY_POINTER, index: Optional[float]=1.0, fm: Optional[float]=None):
	if fm:
		return mus_polyshape(w, index, fm)
	else:
		return mus_polyshape_unmodulated(w, index)
		
def is_polyshape(w: MUS_ANY_POINTER):
	return mus_is_polyshape(w)
	

def partials2polynomial(partials: npt.NDArray[np.float64], kind: Optional[int]=MUS_CHEBYSHEV_FIRST_KIND):
	if isinstance(partials, list):
		partials = np.array(partials, dtype=np.double)	
	mus_partials_to_polynomial(len(partials), partials.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) ,kind)
	return partials

def normalize_partials(partials: npt.NDArray[np.float64]):
	if isinstance(partials, list):
		partials = np.array(partials, dtype=np.double)	
	mus_normalize_partials(len(partials), partials.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
	return partials
	
	
def chebyshev_tu_sum(x: float, t_coeffs: npt.NDArray[np.float64], u_coeffs: npt.NDArray[np.float64]):
	if isinstance(t_coeffs, list):
		t_coeffs = np.array(t_coeffs, dtype=np.double)	
	if isinstance(u_coeffs, list):
		u_coeffs = np.array(u_coeffs, dtype=np.double)	
	return mus_chebyshev_tu_sum(x, 
	t_coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
	u_coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

	
def chebyshev_t_sum(x: float, t_coeffs: npt.NDArray[np.float64]):
	if isinstance(t_coeffs, list):
		t_coeffs = np.array(t_coeffs, dtype=np.double)	
	return mus_chebyshev_tu_sum(x, 
	t_coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

	
	
def mus_chebyshev_u_sum(x: float, u_coeffs: npt.NDArray[np.float64]):
	if isinstance(u_coeffs, list):
		u_coeffs = np.array(u_coeffs, dtype=np.double)
	return mus_chebyshev_tu_sum(x, 
	u_coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
	
	
#sawtooth-wave, triangle-wave, pulse-train, square-wave
	
def make_triangle_wave(frequency: float, amplitude: Optional[float]=1.0, phase: Optional[float]=0.0):
	return mus_make_triangle_wave(frequency, amplitude, phase)
	
def triangle_wave(s: MUS_ANY_POINTER, fm: float=None):
	if fm:
		return mus_triangle_wave(s)
	else:
		return mus_triangle_wave_unmodulated(s)
	
def is_triangle_wave(s: MUS_ANY_POINTER):
	return mus_is_triangle_wave(s)


def make_square_wave(frequency: float, amplitude: Optional[float]=1.0, phase: Optional[float]=0.0):
	return mus_make_square_wave(frequency, amplitude, phase)
	
def square_wave(s: MUS_ANY_POINTER, fm: float=None):
	if fm:
		return mus_square_wave(s)
	else:
		return mus_square_wave_unmodulated(s)
	
def is_square_wave(s: MUS_ANY_POINTER):
	return mus_is_square_wave(s)
	

def make_sawtooth_wave(frequency: float, amplitude: Optional[float]=1.0, phase: Optional[float]=0.0):
	return mus_mus_make_sawtooth_wave(frequency, amplitude, phase)
	
def sawtooth_wave(s: MUS_ANY_POINTER):
	if fm:
		return mus_sawtooth_wave(s)
	else:
		return mus_sawtooth_wave_unmodulated(s)
	
def is_sawtooth_wave(s: MUS_ANY_POINTER):
	return mus_is_sawtooth_wave(s)
	
def make_pulse_train(frequency: float, amplitude: Optional[float]=1.0, phase: Optional[float]=0.0):
	return mus_make_pulse_train(frequency, amplitude, phase)
	
def pulse_train(s: MUS_ANY_POINTER, fm: float):
	if fm:
		return mus_sawtooth_wave(s)
	else:
		return mus_sawtooth_wave_unmodulated(s)
	 
def is_pulse_train(s: MUS_ANY_POINTER):
	return mus_is_pulse_train()
	
	
# ncos nsin

def make_ncos(frequency: float, n: Optional[int]=1):
	return mus_make_ncos(frequency, n)
	
def ncos(nc: MUS_ANY_POINTER, fm: Optional[float]=0.0):
	return mus_ncos(nc, fm)

def is_ncos(nc: MUS_ANY_POINTER):
	return mus_is_ncos(nc)
	
def make_nsin(frequency: float, n: Optional[int]=1):
	return mus_make_nsin(frequency, n)
	
def nsin(nc: MUS_ANY_POINTER, fm: Optional[float]=0.0):
	return mus_nsin(nc, fm)
	
def is_nsin(nc: MUS_ANY_POINTER):
	return mus_is_nsin(nc)
	
	
# nrxysin and nrxycos

def make_nrxysin(frequency: float, ratio: Optional[float]=1., n: Optional[int]=1, r: Optional[float]=.5):
	return mus_make_nrxysin(frequency, ratio, n, r)
	
def nrxysin(s: MUS_ANY_POINTER, fm: Optional[float]=0.):
	return mus_nrxysin(s, fm)
	
def is_nrxysin(s: MUS_ANY_POINTER):
	return mus_is_nrxysin(s)
	
	
def make_nrxycos(frequency: float, ratio: Optional[float]=1., n: Optional[int]=1, r: Optional[float]=.5):
	return mus_make_nrxycos(frequency, ratio, n, r)
	
def nrxycos(s: MUS_ANY_POINTER, fm: Optional[float]=0.):
	return mus_nrxycos(s, fm)
	
def is_nrxycos(s: MUS_ANY_POINTER):
	return mus_is_nrxycos(s)
	
	
# ssb_am

def make_ssb_am(frequency: float, n: Optional[int]=40):
	return mus_make_ssb_am(frequency, n)
	
def ssb_am(gen: MUS_ANY_POINTER, insig: Optional[float]=0.0, fm: Optional[float]=None):
	if(fm):
		return mus_ssb_am(gen, insig, fm)
	else:
		return mus_ssb_am_unmodulated(gen, insig)
		
def is_ssb_am(gen: MUS_ANY_POINTER):
	return mus_is_ssb_am(gen)


#wave_train
#MUS_EXPORT mus_any *mus_make_wave_train(mus_float_t freq, mus_float_t phase, mus_float_t *wave, mus_long_t wsize, mus_interp_t type);
def make_wave_train(frequency: float, wave: npt.NDArray[np.float64], phase: Optional[float]=0., type=MUS_INTERP_LINEAR):
	
	if isinstance(wave, list):
		wv = (c_double * len(wave))(*wave)
							
	if isinstance(wave, np.ndarray):
		wv = wave.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

	return mus_make_wave_train(frequency, phase, wv, len(wave), type)
	
def wave_train(w: MUS_ANY_POINTER, fm: Optional[float]=None):
	if fm:
		return mus_wave_train(w, fm)
	else:
		return mus_wave_train_unmodulated(w, fm)
	
def is_wave_train(w: MUS_ANY_POINTER):
	return mus_is_wave_train(w)
	
# TODO make-wave-train-with-env
def make_wave_train_with_env():
	pass




#rand rand_interp

# TODO what is distribution envelope
def make_rand(frequency: float, amplitude: float,distribution: Optional[npt.NDArray[np.float64]]=None):

	if isinstance(distribution, list):
		dst = (c_double * len(distribution))(*distribution)
		
								
	if isinstance(distribution, np.ndarray):
		dst = distribution.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

	if (distribution):
		return mus_make_rand_with_distribution(frequency, amplitude, dst, len(distribution))
	else:
		return mus_make_rand(frequency, amplitude)

	
def rand(r: MUS_ANY_POINTER, sweep: Optional[float]=None):
	if(sweep):
		return mus_rand(r, sweep)
	else:
		return mus_rand_unmodulated(r)
	
def is_rand(r: MUS_ANY_POINTER):
	return mus_is_rand(r)

def make_rand_interp(frequency: float, amplitude: float,distribution: Optional[npt.NDArray[np.float64]]=None):
	
	if isinstance(distribution, list):
		dst = (c_double * len(distribution))(*distribution)
		
								
	if isinstance(distribution, np.ndarray):
		dst = distribution.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		
	if (distribution):
		return mus_make_rand_interp_with_distribution(frequency, amplitude, dst, len(distribution))
	else:
		return mus_make_rand_interp(frequency, amplitude)
	
def rand_interp(r: MUS_ANY_POINTER, sweep: Optional[float]=0.):
	if(sweep):
		return mus_rand_interp(r, sweep)
	else:
		return mus_rand_interp_unmodulated(r)
	
def is_rand_interp(r: MUS_ANY_POINTER):
	return mus_is_rand_interp(r)	
	
	
def mus_random(amplitude: float):
	return mus_random(amplitude)

# TODO 
# def mus_rand_seed():
# 	return mus_rand_seed()
# 	
# def set_mus_rand_seed()
	
# one-pole, one-zero, two-pole, two-zero

def make_one_pole(a0: float, b1: float):
	return mus_make_one_pole(a0, b1)
	
def one_pole(f: MUS_ANY_POINTER, input: float):
	return mus_one_pole(f, input)
	
def is_one_pole(f: mus_any):
	return mus_is_one_pole(f)
	
def make_one_zero(a0: float, a1: float):
	return mus_make_one_zero(a0, a1)
	
def one_zero(f: MUS_ANY_POINTER, input: float):
	return mus_one_zero(f, input)
	
def is_one_zero(f: MUS_ANY_POINTER):
	return mus_is_one_zero(f)	
	
def make_two_pole(*args):
	if(len(args) == 2):
		return mus_make_two_pole_from_frequency_and_radius(args[0], args[1])
	elif(len(args) == 3):
		return mus_make_two_pole(args[0], args[1], args[2])
	else:
		print("error") # make this real error

def two_pole(f: MUS_ANY_POINTER, input: float):
	return mus_two_pole(f, input)
	
def is_two_pole(f: MUS_ANY_POINTER):
	return mus_is_two_pole(f)

def make_two_zero(*args):
	if(len(args) == 2):
		return mus_make_two_zero_from_frequency_and_radius(args[0], args[1])
	elif(len(args) == 3):
		return mus_make_two_zero(args[0], args[1], args[2])
	else:
		print("error") # make this real error

def two_zero(f: MUS_ANY_POINTER, input: float):
	return mus_two_zero(f, input)
	
def is_two_zero(f: MUS_ANY_POINTER):
	return mus_is_two_zero(f)

# formant

def make_formant(frequency: float, radius: float):
	return mus_make_formant(frequency, radius)

def formant(f: MUS_ANY_POINTER, input: float, radians: Optional[float]=None):
	if fm:
		return mus_formant_with_frequency(f, input, radians)
	else:
		return mus_formant(f, input)
	
def is_formant(f: MUS_ANY_POINTER):
	return mus_is_formant(f)
	
def make_formant_bank(filters, amps: npt.NDArray[np.float64]):

	if isinstance(amps, list):
		a = (c_double * len(amps))(*amps)
		
								
	if isinstance(amps, np.ndarray):
		a = amps.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


	filt_array = (POINTER(mus_any) * len(filters))()
	filt_array[:] = [filters[i] for i in range(len(filters))]
	return mus_make_formant_bank(len(filters),filt_array, a)
	

def formant_bank(f: MUS_ANY_POINTER, inputs):
	if type(inputs) is np.ndarray:
		return mus_formant_bank_with_inputs(f, inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
	else:
		return mus_formant_bank(f, inputs)
	
	
def is_formant_bank(f: MUS_ANY_POINTER):
	return mus_is_formant_bank(f)
	
# filts = [1,2,3,4]		
# for i in range(0, 4) :
# 	filts[i] = make_formant(100*i, .2)	
# a = make_formant_bank(filts, np.array([.4,.3,.5,.5]))
# print(formant_bank(a, .2))
# print(mus_describe(a))
# print(is_formant_bank(a))



def make_firmant(frequency: float, radius: float):
	return mus_make_firmant(frequency, radius)

#radians is freq in radians

def firmant(f: MUS_ANY_POINTER, input: float,radians: Optional[float]=None ):
	if radians:
		return mus_firmant_with_frequency(f, input, frequency, radians)
	else: 
		return mus_firmant(f, input)
			
def is_firmant(f: MUS_ANY_POINTER):
	return mus_is_firmant(f)


#;; the next two are optimizations that I may remove
#mus-set-formant-frequency f frequency
#mus-set-formant-radius-and-frequency f radius frequency

#filter, iir-filter, fir-filter

def make_filter(order: int, xcoeffs: npt.NDArray[np.float64], ycoeffs: npt.NDArray[np.float64]):
	if isinstance(xcoeffs, list):
		xc = (c_double * len(xcoeffs))(*xcoeffs)
		
	if isinstance(ycoeffs, list):
		yc = (c_double * len(ycoeffs))(*ycoeffs)

	if isinstance(xcoeffs, np.ndarray):
		xc = xcoeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		
	if isinstance(ycoeffs, np.ndarray):
		yc = ycoeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		
	return mus_make_filter(order, xc, yc)

	
def filter(fl: MUS_ANY_POINTER, input: float):
	return mus_filter(fl, input)
	
def is_filter(fl: MUS_ANY_POINTER):
	return mus_is_filter(fl)
	
def make_fir_filter(order: int, xcoeffs: npt.NDArray[np.float64]):
	if isinstance(xcoeffs, list):
		xc = (c_double * len(xcoeffs))(*xcoeffs)
		
	if isinstance(xcoeffs, np.ndarray):
		xc = xcoeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

	return mus_make_fir_filter(order, xc)
	
def fir_filter(fl: MUS_ANY_POINTER, input: float):
	return mus_fir_filter(fl, input)
	
def is_fir_filter(fl: MUS_ANY_POINTER):
	return mus_is_fir_filter(fl)

def make_iir_filter(order: int, ycoeffs: npt.NDArray[np.float64]):
	if isinstance(ycoeffs, list):
		yc = (c_double * len(ycoeffs))(*ycoeffs)
		
	if isinstance(ycoeffs, np.ndarray):
		yc = ycoeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		
	return mus_make_iir_filter(order, yc, None)
	
def iir_filter(fl: MUS_ANY_POINTER, input: float ):
	return mus_iir_filter(fl, input)
	
def is_iir_filter(fl: MUS_ANY_POINTER):
	return mus_is_iir_filter(fl)
	
def make_fir_coeffs(order: int, v: npt.NDArray[np.float64]):
	if isinstance(v, list):
		v = np.array(v, dtype=np.double)
	
	if isinstance(v, list):
		fre = (c_double * len(ycoeffs))(*xcoeffs)
		
	if isinstance(v, np.ndarray):
		fre = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))	
	
	coeffs = (c_double * (order + 1))()

	
	mus_make_fir_coeffs(order, fre, coeffs)
	return coeffs



#delay, tap
#if max!=size should be linear?
def make_delay(size: int, 
				initial_contents: Optional[npt.NDArray[np.float64]]=None, 
				initial_element: Optional[float]=0.0, 
				max_size:Optional[int]=None,
				type=Interp.NONE):
	
	contents = None
		
	if not max_size:
		max_size = size
	
	if isinstance(initial_contents, list):
		contents = (c_double * len(initial_contents))(*initial_contents)
	
	if isinstance(initial_contents, np.ndarray):
		contents = initial_contents.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		
# 	if not initial_contents:
# 		contents = (c_double * max_size)()
# 		for i in range(max_size):
# 			contents[i] = initial_element
		
	return mus_make_delay(size, contents, max_size, type)
	
def delay(d: MUS_ANY_POINTER, input: float, pm: Optional[float]=None):
	if pm:
		return mus_delay(d, input, pm)
	else: 
		return mus_delay_unmodulated(d, input)
		
	
def is_delay(d: MUS_ANY_POINTER):
	return mus_is_delay(d)

	
def tap(d: MUS_ANY_POINTER, offset: Optional[float]=None):
	if offset:
		return mus_tap(d, offset)
	else:
		return mus_tap_unmodulated(d)
	
def is_tap(d: MUS_ANY_POINTER):
	return mus_is_tap(d)
	
def delay_tick(d: MUS_ANY_POINTER, input: float):
	return mus_delay_tick(d, input)

# comb notch



def make_comb(scaler: float,
				size: int, 
				initial_contents: Optional[npt.NDArray[np.float64]]=None, 
				initial_element: Optional[float]=0.0, 
				max_size:Optional[int]=None,
				type=MUS_INTERP_NONE):
	contents = None
				
	if not max_size:
		max_size = size
	
	if isinstance(initial_contents, list):
		contents = (c_double * len(initial_contents))(*initial_contents)
	
	if isinstance(initial_contents, np.ndarray):
		contents = initial_contents.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		
# 	if not initial_contents:
# 		contents = (c_double * max_size)()
# 		for i in range(max_size):
# 			contents[i] = initial_element
	#x.append(contents)	
# 	for i in range(max_size):
# 			print(contents[i])# = initial_element
	# print(scaler, size, contents, max_size, type)	
	return mus_make_comb(scaler, size, contents, max_size, type)	

	

def comb(cflt: MUS_ANY_POINTER, input: float, pm: Optional[float]=None):
	if pm:
		return mus_comb(cflt, input, pm)
	else:
		#print(input)	
		return mus_comb_unmodulated(cflt, input)
	
def is_comb(cflt: MUS_ANY_POINTER):
	return mus_is_comb(cflt)	


def make_comb_bank(combs: list):
	comb_array = (MUS_ANY_POINTER * len(combs))()
	comb_array[:] = [combs[i] for i in range(len(combs))]
	return mus_make_comb_bank(len(combs), comb_array)

def comb_bank(combs: MUS_ANY_POINTER, input: float):
	return mus_comb_bank(combs, input)
	
def is_comb_bank(combs: MUS_ANY_POINTER):
	return mus_is_comb_bank(combs)


def make_filtered_comb(scaler: float,
				size: int, 
				filter: Optional[mus_any]=None, #not really optional
				initial_contents: Optional[npt.NDArray[np.float64]]=None, 
				initial_element: Optional[float]=0.0, 
				max_size:Optional[int]=None,
				type=Interp.NONE):
	contents = None
	
	if not filter:
		print("error") # need error			
				
	if not max_size:
		max_size = size
		
	if isinstance(initial_contents, list):
		contents = (c_double * len(initial_contents))(*initial_contents)
	
	if isinstance(initial_contents, np.ndarray):
		contents = initial_contents.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		
# 	if not initial_contents:
# 		contents = (c_double * max_size)()
# 		for i in range(max_size):
# 			contents[i] = initial_element
		
	return mus_make_filtered_comb(scaler, size, contents, max_size, type, filter)
	

def filtered_comb(cflt: MUS_ANY_POINTER, input: float, pm: Optional[float]=None):
	if pm:
		return mus_filtered_comb(cflt, input, pm)
	else:
		return mus_filtered_comb_unmodulated(cflt, input)
	
	
def is_filtered_comb(cflt: MUS_ANY_POINTER):
	return mus_is_filtered_comb(cflt)
	
	
	
def make_filtered_comb_bank(fcombs: list):
	fcomb_array = (POINTER(mus_any) * len(fcombs))()
	fcomb_array[:] = [fcombs[i] for i in range(len(fcombs))]
	return mus_make_filtered_comb_bank(len(fcombs), fcomb_array)

def filtered_comb_bank(fcomb: MUS_ANY_POINTER):
	return mus_filtered_comb_bank(fcombs, input)
	
def is_filtered_comb_bank(fcombs: MUS_ANY_POINTER):
	return mus_is_filtered_comb_bank(fcombs)

def make_notch(scaler: float,
				size: int, 
				initial_contents: Optional[npt.NDArray[np.float64]]=None, 
				initial_element: Optional[float]=0.0, 
				max_size:Optional[int]=None,
				type=Interp.NONE):
				
	contents = None
	
	
	if not max_size:
		max_size = size

	if isinstance(initial_contents, list):
		contents = (c_double * len(initial_contents))(*initial_contents)
	
	if isinstance(initial_contents, np.ndarray):
		contents = initial_contents.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		
# 	if not initial_contents:
# 		contents = (c_double * max_size)()
# 		for i in range(max_size):
# 			contents[i] = initial_element
		
	return mus_make_notch(scaler, size, contents, max_size, type)
	

def notch(cflt: MUS_ANY_POINTER, input: float, pm: Optional[float]=None):
	if pm:
		return mus_notch(cflt, input, pm)
	else:
		return mus_notch_unmodulated(cflt, input)
	
def is_notch(cflt: MUS_ANY_POINTER):
	return mus_is_notch(cflt)
	
	
# all pass

def make_all_pass(feedback: float, 
				feedforward: float,
				size: int, 
				initial_contents: Optional[npt.NDArray[np.float64]]=None, 
				initial_element: Optional[float]=0.0, 
				max_size:Optional[int]=None,
				type=Interp.NONE):
	
	contents = None
	
	if not max_size:
		max_size = size

	if isinstance(initial_contents, list):
		contents = (c_double * len(initial_contents))(*initial_contents)
	
	if isinstance(initial_contents, np.ndarray):
		contents = initial_contents.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		
# 	if not initial_contents:
# 		contents = (c_double * max_size)()
# 		for i in range(max_size):
# 			contents[i] = initial_element
		
	return mus_make_all_pass(feedback, feedforward, size, contents, max_size, type)
	
	
def all_pass(f: MUS_ANY_POINTER, input: float, pm: Optional[float]=None):
	if pm:
		return mus_all_pass(f, input, pm)
	else:
		return mus_all_pass_unmodulated(f, input)
	
def is_all_pass(f: MUS_ANY_POINTER):
	return mus_is_all_pass(f)
	

def make_all_pass_bank(all_passes: list):
	all_passes_array = (POINTER(mus_any) * len(all_passes))()
	all_passes_array[:] = [all_passes[i] for i in range(len(all_passes))]
	return mus_make_all_pass_bank(len(all_passes), all_passes_array)

def all_pass_bank(all_passes: MUS_ANY_POINTER, input: float):
	return mus_all_pass_bank(all_passes, input)
	
def is_all_pass_bank(o: MUS_ANY_POINTER):
	return mus_is_all_pass_bank(o)
		
		
# make-one-pole-all-pass size coeff
# one-pole-all-pass f input 
# one-pole-all-pass? f



#moving-average, moving-max, moving-norm

def make_moving_average(size: int, initial_contents: Optional[npt.NDArray[np.float64]]=None, initial_element: Optional[float]=0.0):
					
					
	contents = None
	
	if isinstance(initial_contents, list):
		contents = (c_double * len(initial_contents))(*initial_contents)
	
	if isinstance(initial_contents, np.ndarray):
		contents = initial_contents.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
# 		
# 	if not initial_contents:
# 		contents = (c_double * size)()
# 		for i in range(size):
# 			contents[i] = initial_element
		
	return mus_make_moving_average(size, contents)
	
def moving_average(f: MUS_ANY_POINTER, input: float):
	return mus_moving_average(f, input)
	
def is_moving_average(f: MUS_ANY_POINTER):
	return mus_is_moving_average(f)
	

def make_moving_max(size: int, 
				initial_contents: Optional[npt.NDArray[np.float64]]=None, 
				initial_element: Optional[float]=0.0):

	contents = None

	if isinstance(initial_contents, list):
		contents = (c_double * len(initial_contents))(*initial_contents)
	
	if isinstance(initial_contents, np.ndarray):
		contents = initial_contents.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
		
# 	if not initial_contents:
# 		contents = (c_double * size)()
# 		for i in range(size):
# 			contents[i] = initial_element
		
	return mus_make_moving_max(size, initial_contents.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

	
def moving_max(f: MUS_ANY_POINTER, input: float):
	return mus_moving_max(f, input)
	
def is_moving_max(f: MUS_ANY_POINTER):
	return mus_is_moving_max(f)
	
	
def make_moving_norm(size: int,scaler: Optional[float]=1.):
	initial_contents = (c_double * size)()
	return mus_make_moving_norm(size, initial_contents, scaler)
	
def moving_norm(f: MUS_ANY_POINTER, input: float):
	return mus_moving_norm(f, input)
	
def is_moving_norm(f: MUS_ANY_POINTER):
	return mus_is_moving_norm(f)
	

	
#asymmetric-fm


def make_asymmetric_fm(frequency: float, initial_phase: Optional[float]=0.0, r: Optional[float]=1.0, ratio: Optional[float]=1.):
	return mus_make_asymmetric_fm(frequency, initial_phase, r, ratio)
	
def asymmetric_fm(af: MUS_ANY_POINTER, index: float, fm: Optional[float]=None):
	if fm:
		return mus_asymmetric_fm(af, index, fm)
	else:
		return mus_asymmetric_fm_unmodulated(af, index)
	
def is_asymmetric_fm(af: MUS_ANY_POINTER):
	return mus_is_asymmetric_fm(af)
	
	

	
#@ sound io

def make_file2sample(name, buffer_size: Optional[int]=8192):
	return mus_make_file_to_sample_with_buffer_size(name, buffer_size)
	
def file2sample(obj: MUS_ANY_POINTER, loc: int, chan: int):
	return mus_file_to_sample(obj, loc, chan)
	
def is_file2sample(gen: MUS_ANY_POINTER):
	return mus_is_file_to_sample(gen)
	
	

def make_sample2file(name, chans: Optional[int]=1, format: Optional[Sample]=Sample.L24INT, type: Optional[Header]=Header.AIFF, comment: Optional[str]=None):
	if comment:
		return mus_make_sample_to_file(name, chans, format, type)
	else:
		return mus_make_sample_to_file_with_comment(name, chans, format, type, comment)

def sample2file(obj: MUS_ANY_POINTER,samp: int, chan:int , val: float):
	return mus_sample_to_file(obj, samp, chan, val)
	
def is_sample2file(obj: MUS_ANY_POINTER):
	return mus_is_sample_to_file(obj)
	

def continue_sample2file(name):
	return mus_continue_sample_to_file(name)
	
	

	
def make_file2frample(name, buffer_size: Optional[int]=8192):
	return mus_make_file_to_frample_with_buffer_size(name, buffer_size)
	
def file2frample(obj: MUS_ANY_POINTER, loc: int):
	outf = np.zeros(mus_channels(obj), dtype=np.double);
	mus_file_to_frample(obj, loc, outf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
	return outf
	
def is_file2frample(gen: MUS_ANY_POINTER):
	return mus_is_file_to_frample(gen)
	
	

def make_frample2file(name, chans: Optional[int]=1, format: Optional[Sample]=Sample.LFLOAT, type: Optional[Header]=Header.NEXT, comment: Optional[str]=None):
	if comment:
		return mus_make_frample_to_file(name, chans, format, type)
	else:
		return mus_make_frample_to_file_with_comment(name, chans, format, type, comment)

def frample2file(obj: MUS_ANY_POINTER,samp: int, chan:int , val: npt.NDArray[np.float64]):
	mus_sample_to_file(obj, samp, chan, val.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
	return val
	
def is_frample2file(obj: MUS_ANY_POINTER):
	return mus_is_sample_to_file(obj)
	

def continue_frample2file(name):
	return mus_continue_sample_to_file(name)
	
	
#def file2array(file, channel: Optional[int]=1, beg: int, dur: int):
#def array2file(file, data, len, srate, channels)	

def mus_close(obj):
	return mus_close_file(obj)
	
def mus_is_output(obj):
	return mus_is_output(obj)
	
def mus_is_input(obj):
	return mus_is_input(obj)


#readin

def make_readin(filename: str, chan: int=0, start: int=0, direction: Optional[int]=1, buffersize: Optional[int]=512):
	return mus_make_readin_with_buffer_size(filename, chan, start, direction, buffersize)
	
def readin(rd: MUS_ANY_POINTER):
	return mus_readin(rd)
	
	
def is_readin(rd: MUS_ANY_POINTER):
	return mus_is_readin(rd)
	
	
	
# src

def make_src(input, srate: Optional[float]=1.0, width: Optional[int]=5):
	if(isinstance(input, MUS_ANY_POINTER)):
		@INPUTCALLBACK
		def ifunc(gen, inc):
			#obj = cast(gen, POINTER(mus_any))
			#mus_set_increment(input,inc) # doing this to avoid releasing pointer that is just copy
			return mus_apply(input,inc, 0.)
		
		res = mus_make_src(ifunc, srate, width, input)
	
	elif isinstance(input, types.FunctionType):
		@INPUTCALLBACK
		def ifunc(gen, inc):
			return input(inc)
			
		res = mus_make_src(ifunc, srate, width, None)
	else:
		print("error") # need error
		
	
	res._inputcallback = ifunc
	return res

	
def src(s: MUS_ANY_POINTER, sr_change: Optional[float]=0.0):
	return mus_src(s, sr_change, s._inputcallback)
	
def is_src(s: MUS_ANY_POINTER):
	return mus_is_src(s)
	
	
#convolve

def make_convolve(input, filter: npt.NDArray[np.float64], fft_size: int, filter_size: Optional[int]=None ):

	if isinstance(filter, list):
		filter = np.array(filter, dtype=np.double)
		
	if not filter_size:
		filter_size = len(filter)

	if(isinstance(input, MUS_ANY_POINTER)):
		@INPUTCALLBACK
		def ifunc(gen, inc):
			obj = cast(gen, MUS_ANY_POINTER)
			mus_set_increment(obj,inc)
			return mus_apply(obj, inc, 0.)
		
		res = mus_make_convolve(ifunc, filter.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), fft_size, filter_size)
	
	elif isinstance(input, types.FunctionType):
		@INPUTCALLBACK
		def ifunc(gen, inc):
			return input(inc)
			
		res = mus_make_convolve(ifunc, srate, width, None)
	else:
		print("error") # need error
		
	res._inputcallback = ifunc
	return res

	
def convolve(gen: MUS_ANY_POINTER):
	return mus_convolve(gen, gen._inputcallback)
	
def is_convolve(gen: MUS_ANY_POINTER):
	return mus_is_convolve(gen)
	
	
def convolve_files(file1, file2, maxamp: float, outputfile):
	mus_convolve_files(file1, file2, maxamp,outputfile )


def cepstrum(data: npt.NDArray[np.float64], n):
	if isinstance(data, list):
		data = np.array(data, dtype=np.double)
	size = len(data)
	dt = data.copy()
	mus_cepstrum(dt.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n)
	return dt

# TODO convolve-files file1 file2 (maxamp 1.0) (output-file "tmp.snd")
	
#granulate

def make_granulate(input, 
					expansion: Optional[float]=1.0, 
					length: Optional[float]=.15,
					scaler: Optional[float]=.6,
					hop: Optional[float]=.05,
					ramp: Optional[float]=.4,
					jitter: Optional[float]=1.0,
					max_size: Optional[int]=0,
					edit=None):
	
	if edit:
		@EDITCALLBACK	
		def efunc(gen):
			return edit(gen)
		
	if(isinstance(input, MUS_ANY_POINTER)):
		@INPUTCALLBACK
		def ifunc(gen, inc):
			#obj = cast(gen, MUS_ANY_POINTER)
			#mus_set_increment(input,inc)
			return mus_apply(input, 0., 0.)
	
		res = mus_make_granulate(ifunc, expansion, length, scaler, hop, ramp, jitter, max_size, efunc if edit else cast(None, EDITCALLBACK), input)
	
	elif isinstance(input, types.FunctionType):
		@INPUTCALLBACK
		def ifunc(gen, inc):
			return input(inc)
				
		res = mus_make_granulate(ifunc, expansion, length, scaler, hop, ramp, jitter, max_size, efunc if edit else cast(None, EDITCALLBACK), None)
	else:
		print("error") # need error
		
	res._inputcallback = ifunc
	res._editcallback  = None
	if edit:
		res._editcallback = efunc
		#print(edit)
	return res
	
	
#mus_granulate_grain_max_length
def granulate(e: MUS_ANY_POINTER):
	if e._editcallback:
		return mus_granulate_with_editor(e, e._inputcallback, e._editcallback)
	else:
		return mus_granulate(e, e._inputcallback)

def is_granulate(e: MUS_ANY_POINTER):
	return mus_is_granulate(e)
	
	
	
# phase vocoder 

def make_phase_vocoder(input, 
						fft_size: Optional[int]=512, 
						overlap: Optional[int]=4, 
						interp: Optional[int]=128, 
						pitch: Optional[float]=1.0, 
						analyze=None, 
						edit=None, 
						synthesize=None):
	
	if analyze:
		@ANALYSISCALLBACK
		def afunc(gen, inputfunc):
			return analyze(gen, inputfunc)
				
	if edit:
		@EDITCALLBACK	
		def efunc(gen):
			return edit(gen)

	if synthesize:
		@SYNTHESISCALLBACK
		def sfunc(gen):
			return synthesize(gen)
		
	if(isinstance(input, MUS_ANY_POINTER)):
		@INPUTCALLBACK
		def ifunc(gen, inc):
# 			obj = cast(gen, MUS_ANY_POINTER)
# 			mus_set_increment(obj,inc)
# 			return mus_apply(obj, inc, 0.)
 			mus_set_increment(input,inc)
 			return mus_apply(input, inc, 0.)
	
		res = mus_make_phase_vocoder(ifunc, fft_size, overlap, interp, pitch, 
						afunc if analyze else cast(None, ANALYSISCALLBACK),
						efunc if edit else cast(None, EDITCALLBACK),
						sfunc if synthesize else cast(None, SYNTHESISCALLBACK),
						input)
	
	
	elif isinstance(input, types.FunctionType):
		@INPUTCALLBACK
		def ifunc(gen, inc):
			return input(inc)
				
		res = mus_make_phase_vocoder(ifunc, fft_size, overlap, interp, pitch, 
				afunc if analyze else cast(None, ANALYSISCALLBACK),
				efunc if edit else cast(None, EDITCALLBACK),
				sfunc if synthesize else cast(None, SYNTHESISCALLBACK),
				None)
	else:
		print("error") # need error
		
	setattr(res,'_inputcallback', ifunc)		
	setattr(res,'_analyzecallback', afunc if analyze else None)
	setattr(res,'_synthesizecallback', sfunc if analyze else None)
	setattr(res,'_editcallback', efunc if edit else None )
	
	return res

	
def phase_vocoder(pv: MUS_ANY_POINTER):
	if pv._analyzecallback or pv._synthesizecallback or pv._editcallback :
		return mus_phase_vocoder_with_editors(pv, pv._inputcallback, pv._analyzecallback, pv._editcallback, pv._synthesizecallback)
	else:
		return mus_phase_vocoder(pv, pv._inputcallback)
	
def is_phase_vocoder(pv: MUS_ANY_POINTER):
	return mus_is_phase_vocoder(pv)
	

def phase_vocoder_amps(gen: MUS_ANY_POINTER):
	size = mus_length(gen)
	p = np.ctypeslib.as_array(mus_phase_vocoder_amps(gen), shape=size)
	amps = np.copy(p)
	return amps

def phase_vocoder_amp_increments(gen: MUS_ANY_POINTER):
	size = mus_length(gen)
	p = np.ctypeslib.as_array(mus_phase_vocoder_amp_increments(gen), shape=size)
	amp_increments = np.copy(p)
	return amp_increments
	
def phase_vocoder_freqs(gen: MUS_ANY_POINTER):
	size = mus_length(gen)
	p = np.ctypeslib.as_array(mus_phase_vocoder_freqs(gen), shape=size)
	freqs = np.copy(p)
	return freqs
	

def phase_vocoder_phases(gen: MUS_ANY_POINTER):
	size = mus_length(gen)
	p = np.ctypeslib.as_array(mus_phase_vocoder_phases(gen), shape=size)
	phases = np.copy(p)
	return phase
	
def phase_vocoder_phase_increments(gen: MUS_ANY_POINTER):
	size = mus_length(gen)
	p = np.ctypeslib.as_array(mus_phase_vocoder_phase_increments(gen), shape=size)
	phases_increments = np.copy(p)
	return phases_increments
	
#print(isinstance(np.array([1,2,3]), np.ndarray))
#in-any, out-any

# out-any loc data channel (output *output*)
# def out_any(loc: int, data: float, channel, output):
# 	if isinstance(output, np.ndarray):
# 		output[channel][loc] = data
# 	else:
# 		mus_out_any(loc, data, channel, output)
# 	
# def outa(loc: int, data: float, output):
# 	out_any(loc, data, 0, output)
# 	
# def outb(loc: int, data: float, output):
# 	out_any(loc, data, 1, output)
# 	
# def outc(loc: int, data: float, output):
# 	out_any(loc, data, 2, output)
# 	
# def outd(loc: int, data: float, output):
# 	out_any(loc, data, 3, output)
	
#  TODO : output to array out-any loc data channel (output *output*)	
def out_any(loc: int, data: float, channel,  output=None):
	if output:
		mus_out_any(loc, data, channel, output)		
	else:
		mus_out_any(loc, data, channel, Sound.output)	

def outa(loc: int, data: float, output=None):
	if output:
		out_any(loc, data, 0, output)		
	else:
		out_any(loc, data, 0, Sound.output)	
	
def outb(loc: int, data: float, output=None):
	if output:
		out_any(loc, data, 1, output)		
	else:
		out_any(loc, data, 1, Sound.output)
	
def outc(loc: int, data: float, output=None):
	if output:
		out_any(loc, data, 2, output)		
	else:
		out_any(loc, data, 2, Sound.output)		
	
def outd(loc: int, data: float, output=None):
	if output:
		out_any(loc, data, 3, output)		
	else:
		out_any(loc, data, 3, Sound.output)	
	
def out_bank(gens, loc, input):
	for i in range(len(gens)):
		out_any(loc, mus_apply(gens[i], input, 0.), i)	
	

#out_bank ? TODO - what is this

def in_any(loc: int, channel: int, input):
	if isinstance(input, np.ndarray):
		return input[channel][loc]
	else:
		return mus_in_any(loc, channel, input)
	
def ina(loc: int, input):
	if isinstance(input, np.ndarray):
		return input[channel][loc]
	else:
		return mus_in_any(loc, 0, input)
	
def inb(loc: int, input):
	if isinstance(input, np.ndarray):
		return input[channel][loc]
	else:
		return mus_in_any(loc, 1, input)





# locsig
# TODO:  output could be array 
# TODO : revchans vs chans
#     mus_make_locsig.argtypes = [mus_float_t, mus_float_t, mus_float_t, c_int, POINTER(mus_any), c_int, POINTER(mus_any), mus_interp_t]
# def make_locsig(degree: Optional[float]=0.0, 
# 	distance: Optional[float]=1., 
# 	reverb: Optional[float]=0.0, 
# 	output: Optional[MUS_ANY_POINTER]=None, 
# 	revout: Optional[MUS_ANY_POINTER]=None, 
# 	channels: Optional[int]=2, 
# 	type: Optional[Interp]=Interp.LINEAR):
# 	
# 	
# 	if not output or not reverb:
# 		print("error")
# 		return None
# 	if not channels:
# 		channels = mus_channels(output)	
# 	return mus_make_locsig(degree, distance, reverb, channels, output, channels, revout,  type)

def make_locsig(degree: Optional[float]=0.0, 
	distance: Optional[float]=1., 
	reverb: Optional[float]=0.0, 
	output: Optional[MUS_ANY_POINTER]=None, 
	revout: Optional[MUS_ANY_POINTER]=None, 
	channels: Optional[int]=2, 
	type: Optional[Interp]=Interp.LINEAR):
	
	if not output:
		output = Sound.output
	
	if not revout:
		revout = Sound.reverb

	if not channels:
		channels = mus_channels(output)	
		
	return mus_make_locsig(degree, distance, reverb, channels, output, channels, revout,  type)
		
#(mus_any *ptr, mus_long_t loc, mus_float_t val)
def locsig(gen: MUS_ANY_POINTER, loc: int, val: float):
	mus_locsig(gen, loc, val)
	
def is_locsig(gen: MUS_ANY_POINTER):
	return mus_is_locsig(gen)
	
def locsig_ref(gen: MUS_ANY_POINTER, chan: int):
	return mus_locsig_ref(gen, chan)
	
def locsig_set(gen: MUS_ANY_POINTER, chan: int, val:float):
	return mus_locsig_set(gen, chan, val)
	
def locsig_reverb_ref(gen: MUS_ANY_POINTER, chan: int):
	return mus_locsig_reverb_ref(gen, chan)
	loc
def locsig_reverb_set(gen: MUS_ANY_POINTER, chan: int, val: float):
	return mus_locsig_reverb_set(gen, chan, val)
	
def move_locsig(gen: MUS_ANY_POINTER, degree: float, distance: float):
	mus_move_locsig(gen, degree, distance)
	

# locsig-type ()	

# TODO: move-sound  need dlocsig

# def make_move_sound():
# 	pass
# 	
# def move_sound():
# 	pass
# 	
# def is_move_sound():
# 	pass
# 	
# o = make_oscil(260)	
# print(o)
# print(o.__class__.__name__)

# 