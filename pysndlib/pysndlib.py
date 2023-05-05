from contextlib import contextmanager
from enum import Enum, IntEnum
import functools
from functools import singledispatch
import os
from typing import Optional
import subprocess
import time
import tempfile
import types
import math
import numpy as np
import numpy.typing as npt
import musx
from .sndlib import *

mus_initialize()
mus_sound_initialize() 

# TODO: one_pole_all_pass listed in clm2xen.c
# TODO: pink_noise  listed in clm2xen.c
# TODO: piano_noise listed in clm2xen.c
# TODO: singer_filter listed in clm2xen.c
# TODO: singer_nose_filter listed in clm2xen.c


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
    
class Error(IntEnum):
    NO_ERROR = MUS_NO_ERROR
    NO_FREQUENCY = MUS_NO_FREQUENCY
    NO_PHASE = MUS_NO_PHASE
    NO_GEN = MUS_NO_GEN
    NO_LENGTH = MUS_NO_LENGTH
    NO_DESCRIBE = MUS_NO_DESCRIBE
    NO_DATA = MUS_NO_DATA
    NO_SCALER = MUS_NO_SCALER
    MEMORY_ALLOCATION_FAILED = MUS_MEMORY_ALLOCATION_FAILED
    CANT_OPEN_FILE = MUS_CANT_OPEN_FILE
    NO_SAMPLE_INPUT = MUS_NO_SAMPLE_INPUT
    NO_SAMPLE_OUTPUT = MUS_NO_SAMPLE_OUTPUT
    NO_SUCH_CHANNEL = MUS_NO_SUCH_CHANNEL
    NO_FILE_NAME_PROVIDED = MUS_NO_FILE_NAME_PROVIDED
    NO_LOCATION = MUS_NO_LOCATION
    NO_CHANNEL = MUS_NO_CHANNEL
    NO_SUCH_FFT_WINDOW = MUS_NO_SUCH_FFT_WINDOW
    UNSUPPORTED_SAMPLE_TYPE = MUS_UNSUPPORTED_SAMPLE_TYPE
    HEADER_READ_FAILED = MUS_HEADER_READ_FAILED
    UNSUPPORTED_HEADER_TYPE = MUS_UNSUPPORTED_HEADER_TYPE
    FILE_DESCRIPTORS_NOT_INITIALIZED = MUS_FILE_DESCRIPTORS_NOT_INITIALIZED
    NOT_A_SOUND_FILE = MUS_NOT_A_SOUND_FILE
    FILE_CLOSED = MUS_FILE_CLOSED
    WRITE_ERROR = MUS_WRITE_ERROR
    HEADER_WRITE_FAILED = MUS_HEADER_WRITE_FAILED
    CANT_OPEN_TEMP_FILE = MUS_CANT_OPEN_TEMP_FILE
    INTERRUPTED = MUS_INTERRUPTED
    BAD_ENVELOPE = MUS_BAD_ENVELOPE
    AUDIO_CHANNELS_NOT_AVAILABLE = MUS_AUDIO_CHANNELS_NOT_AVAILABLE
    AUDIO_SRATE_NOT_AVAILABLE = MUS_AUDIO_SRATE_NOT_AVAILABLE
    AUDIO_SAMPLE_TYPE_NOT_AVAILABLE = MUS_AUDIO_SAMPLE_TYPE_NOT_AVAILABLE
    AUDIO_NO_INPUT_AVAILABLE = MUS_AUDIO_NO_INPUT_AVAILABLE
    AUDIO_CONFIGURATION_NOT_AVAILABLE = MUS_AUDIO_CONFIGURATION_NOT_AVAILABLE
    AUDIO_WRITE_ERROR = MUS_AUDIO_WRITE_ERROR
    AUDIO_SIZE_NOT_AVAILABLE = MUS_AUDIO_SIZE_NOT_AVAILABLE
    AUDIO_DEVICE_NOT_AVAILABLE = MUS_AUDIO_DEVICE_NOT_AVAILABLE
    AUDIO_CANT_CLOSE = MUS_AUDIO_CANT_CLOSE
    AUDIO_CANT_OPEN = MUS_AUDIO_CANT_OPEN
    AUDIO_READ_ERROR = MUS_AUDIO_READ_ERROR
    AUDIO_CANT_WRITE = MUS_AUDIO_CANT_WRITE
    AUDIO_CANT_READ = MUS_AUDIO_CANT_READ
    AUDIO_NO_READ_PERMISSION = MUS_AUDIO_NO_READ_PERMISSION
    CANT_CLOSE_FILE = MUS_CANT_CLOSE_FILE
    ARG_OUT_OF_RANGE = MUS_ARG_OUT_OF_RANGE
    NO_CHANNELS = MUS_NO_CHANNELS
    NO_HOP = MUS_NO_HOP
    NO_WIDTH = MUS_NO_WIDTH
    NO_FILE_NAME = MUS_NO_FILE_NAME
    NO_RAMP = MUS_NO_RAMP
    NO_RUN = MUS_NO_RUN
    NO_INCREMENT = MUS_NO_INCREMENT
    NO_OFFSET = MUS_NO_OFFSET
    NO_XCOEFF = MUS_NO_XCOEFF
    NO_YCOEFF = MUS_NO_YCOEFF
    NO_XCOEFFS = MUS_NO_XCOEFFS
    NO_YCOEFFS = MUS_NO_YCOEFFS
    NO_RESET = MUS_NO_RESET
    BAD_SIZE = MUS_BAD_SIZE
    CANT_CONVERT = MUS_CANT_CONVERT
    READ_ERROR = MUS_READ_ERROR
    NO_FEEDFORWARD = MUS_NO_FEEDFORWARD
    NO_FEEDBACK = MUS_NO_FEEDBACK
    NO_INTERP_TYPE = MUS_NO_INTERP_TYPE
    NO_POSITION = MUS_NO_POSITION
    NO_ORDER = MUS_NO_ORDER
    NO_COPY = MUS_NO_COPY
    CANT_TRANSLATE = MUS_CANT_TRANSLATE
    NUM_ERRORS = MUS_NUM_ERRORS

# 
# 
#these may need to be set based on system type etc
DEFAULT_OUTPUT_SRATE = 44100
DEFAULT_OUTPUT_CHANS = 1
DEFAULT_OUTPUT_SAMPLE_TYPE = Sample.B24INT
DEFAULT_OUTPUT_HEADER_TYPE = Header.AIFC


# maybe this needs to be singleton class so
# if srate is changed it calls mus_set_srate??
# no i don't think that is necessary ...
CLM  = types.SimpleNamespace(
    file_name = 'test.snd',
    srate = DEFAULT_OUTPUT_SRATE,
    channels = DEFAULT_OUTPUT_CHANS,
    sample_type = DEFAULT_OUTPUT_SAMPLE_TYPE,
    header_type = DEFAULT_OUTPUT_HEADER_TYPE,
    verbose = False,
    play = False,
    statistics = False,
    reverb = False,
    reverb_channels = 1,
    reverb_data = None,
    reverb_file_name = 'test.rev',
    table_size = 512,
    buffer_size = 65536,
    locsig_tyoe = Interp.LINEAR,
    clipped = True,
    player = False,
    notehook = False,
    to_snd = True,
    output = False,
    delete_reverb = False
)

CLM.player = 'afplay'
# set srate to default 
#mus_sound_set_srate(CLM.srate)





MUS_ANY_POINTER = POINTER(mus_any) 

INPUTCALLBACK = CFUNCTYPE(c_double, c_void_p, c_int)
EDITCALLBACK = CFUNCTYPE(c_int, c_void_p)
ANALYSISCALLBACK = CFUNCTYPE(c_bool, c_void_p, CFUNCTYPE(c_double, c_void_p, c_int))
SYNTHESISCALLBACK = CFUNCTYPE(c_double, c_void_p)
LOCSIGDETOURCALLBACK = CFUNCTYPE(UNCHECKED(None), c_void_p, mus_long_t) #making void to avoid creating pointe objec


# these getters and setters can be used
#  set_mus_scaler(gen, 1.) instead of
#  (set! (mus-scaler gen) 1.)
#  a = set_mus_scaler(gen) instead of
#  (set! a (mus-scaler gen))
# can also use the underlying 
# sndlib functions mus_scaler and mus_set_scaler
# but adding them as properties
# allows for this type of thing
#  gen.mus_scaler = 1.0
# and
# a = gen.scaler
# seems like property style setting and getting
# more in line with python style

# is this a hack? i don't know. this seems to add the properties i want
# I think in the end will need to hand craft a class to be used instead of
# the autogenerated stuff in sndlib.py


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

# TODO: mus_set_data does not make a copy so need to add to cache
# but if called repeatedly cache will keep growing. 
# maybe cache idea needs to be refined. data could
# go into other instance variable (_data) and just keep replacing
def set_mus_data(gen: mus_any, data):
    data_ptr = get_array_ptr(data)
    return mus_set_data(gen, data_ptr)
    
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

# call free 
# i had some issues when implementing some callbacks
# where void pointers or mus_any pointers keep 
# needing to be cast or calling POINTER on them 
# in python. the gc would collect the python objects
# even though they referred to the same underlying 
# c pointer. so need to be careful with this in future
# just putting note here so i a remember if seeing
# any errors that could be related.
MUS_ANY_POINTER.__del__ = lambda s :  mus_free(s)
#MUS_ANY_POINTER.__del__ = lambda s :  print("freeme", s)

# this could use some work but good enough for the moment.
# maybe want to be able to switch between more verbose printing and terser
MUS_ANY_POINTER.__str__ = lambda s : f'{MUS_ANY_POINTER} {str(mus_describe(s).data, "utf-8")}'




# NOTE from what I understand about numpy.ndarray.ctypes 
# it says that when data_as is used that it keeps a reference to 
# the original array. that should mean as long as I cache the
# result of data_as I should not need to cache the original 
# numpy array. right?

# these could be singledispatch but is manageable this way for now

def get_array_ptr(arr):
    if isinstance(arr, list):
        res = (c_double * len(arr))(*arr)
    
    elif isinstance(arr, np.ndarray):
        res = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # check shape
    else:
        raise TypeError(f'{arr} is not a list or ndarry but is a {type(arr)}')

    return res    


# check if something is iterable 
def is_iterable(x):
    return hasattr(x, '__iter__')

def is_zero(n):
    return n == 0
    
def is_number(n):
    return type(n) == int or type(n) == float   
    
def is_power_of_2(x):
    return (((x) - 1) & (x)) == 0

def next_power_of_2(x):
    return 2**int(1 + (float(math.log(x + 1)) / math.log(2.0)))
    
### some utilites as generic functions


# TODO: what other clm functions need. 
#  using clm_* to avoid clashing with 
# what are likely typically used variable names
# 

@singledispatch
def clm_channels(x):
    pass
    
@clm_channels.register
def _(x: str):  #assume it is a file
    return mus_sound_chans(x)
    
@clm_channels.register
def _(x: MUS_ANY_POINTER):  #assume it is a gen
    return x.mus_channels
    
@clm_channels.register
def _(x: list):  
    return len(x)
    
@clm_channels.register
def _(x: np.ndarray):  
    if x.ndim == 1:
        return 1
    elif x.ndim == 2:
        return np.shape(x)[0]
    else:
        print("error") # raise error
        

@singledispatch
def clm_length(x):
    pass
    
@clm_length.register
def _(x: str):# assume file
    return mus_sound_length(x)

@clm_length.register
def _(x: MUS_ANY_POINTER):# assume file
    return x.mus_length
        
@clm_length.register
def _(x: list):
    return len(x[0])

@clm_length.register
def _(x: np.ndarray):
    if x.ndim == 1:
        return np.shape(x)[0]
    elif x.ndim == 2:
        return np.shape(x)[1]
    else:
        print("error") # raise error

@singledispatch
def clm_srate(x):
    pass

#TODO: do we need others ?
@clm_srate.register
def _(x: str): #file
    mus_sound_srate(file)
  
  
@singledispatch
def clamp(x, lo, hi):
    pass
    
@clamp.register
def _(x: float, lo, hi):
    return float(max(min(x, hi),lo))
    
@clamp.register
def _(x: int, lo, hi):
    return int(max(min(x, hi),lo))
    
@singledispatch
def clip(x, lo, hi):
    pass
    
@clip.register
def _(x: float, lo, hi):
    return float(max(min(x, hi),lo))
    
@clip.register
def _(x: int, lo, hi):
    return int(max(min(x, hi),lo))

@singledispatch
def fold(x, lo, hi):
    pass
    
@fold.register
def _(x: float, lo, hi):
    r = hi-lo
    v = (x-lo)/r
    return r * (1.0 - math.fabs(math.fmod(v,2.0) - 1.0)) + lo
    
@fold.register
def _(x: int, lo, hi):
    r = hi-lo
    v = (x-lo)/r
    return int(r * (1.0 - math.fabs(math.fmod(v,2.0) - 1.0)) + lo)

@singledispatch    
def wrap(x, lo, hi):
    pass

@wrap.register
def _(x: float, lo, hi):
    r = hi-lo
    if x >= lo and x <= hi:
        return x
    if x < lo:
        return hi + (math.fmod((x-lo), r))
    if x > hi:
        return lo + (math.fmod((x-hi), r))
 
@wrap.register
def _(x: int, lo, hi):
    r = hi-lo
    if x >= lo and x <= hi:
        return x
    if x < lo:
        return int(hi + (math.fmod((x-lo), r)))
    if x > hi:
        return int(lo + (math.fmod((x-hi), r)))

        

    
    
    
    
#     length = mus_sound_framples(filename)
#     chans = mus_sound_chans(filename)
#     srate = mus_sound_srate(filename)
#     out = np.zeros((chans, length), dtype=np.double)
#     for i in range(chans):
#         mus_file_to_array(filename,i, 0, length, out[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
#     return out, srate    

# CLM utility functions
# these could all be done in pure python
# not sure calling c - functions is any better
# but keeps things synced with sndlib even if seems
# doubtful any of this stuff would change

# calls to ctypes functions that try to pass wrong type 
# will generate their own exceptions so don't need to check here
# one case this is not true is if None is passed for a pointer. 

def radians2hz(radians: float):
    """Convert radians per sample to frequency in "Hz: rads * srate / (2 * pi)"""
    return mus_radians_to_hz(radians)

def hz2radians(hz: float):
    """Convert frequency in Hz to radians per sample: hz * 2 * pi / srate"""
    return mus_hz_to_radians(hz)

def degrees2radians(degrees: float):
    """Convert degrees to radians: deg * 2 * pi / 360"""
    return mus_degrees_to_radians(degrees)
    
def radians2degrees(radians: float):
    """Convert radians to degrees: rads * 360 / (2 * pi)"""
    return mus_radians_to_degrees(radians)
    
def db2linear(x: float):
    """Convert decibel value db to linear value: pow(10, db / 20)"""
    return mus_db_to_linear(x)
    
def linear2db(x: float):
    """Convert linear value to decibels: 20 * log10(lin)"""
    return mus_linear_to_db(x)
    
def odd_multiple(x: float, y: float):
    """Return the odd multiple of x and y"""
    return mus_odd_multiple(x,y)
    
def even_multiple(x: float, y: float):
    """Return the even multiple of x and y"""
    return mus_even_multiple(x,y)
    
def odd_weight(x: float):
    """Return the odd weight of x"""
    return mus_odd_weight(x)
    
def even_weight(x: float):
    """Return the even weight of x"""
    mus_even_weight(x)
    
def get_srate():
    """Return current sample rate"""
    return mus_srate()
    
def set_srate(r: float):
    """Set current sample rate"""
    return mus_set_srate(r)
    
def seconds2samples(secs: float):
    """Use mus_srate to convert seconds to samples."""
    return mus_seconds_to_samples(secs)

def samples2seconds(samples: int):
    """Use mus_srate to convert samples to seconds."""
    return mus_samples_to_seconds(samples)
#
# TODO : do we need these
# def get_mus_float_equal_fudge_factor():
#     return mus_float_equal_fudge_factor()
#     
# def get_mus_array_print_length():
#     return mus_array_print_length()
#     
# def set_mus_array_print_length(x: int):
#     return mus_set_array_print_length(x)
    
def ring_modulate(s1: float, s2: float):
    """Return s1 * s2 (sample by sample multiply)"""
    return mus_ring_modulate(s1, s2)
    
def amplitude_modulate(s1: float, s2: float, s3: float):
    """Carrier in1 in2): in1 * (carrier + in2)"""
    return mus_amplitude_modulate(s1, s2, s3)
    
def contrast_enhancement(sig: float, index: float):
    """Returns sig (index 1.0)): sin(sig * pi / 2 + index * sin(sig * 2 * pi))"""
    return mus_contrast_enhancement(sig, index)
    
def dot_product(data1, data2):
    """Returns v1 v2 (size)): sum of v1[i] * v2[i] (also named scalar product)"""
    data1_ptr = get_array_ptr(data1)    
    data2_ptr = get_array_ptr(data2)    
    return mus_dot_product(data1_ptr, data2_ptr, len(data1))
    
def polynomial(coeffs, x: float):
    """Evaluate a polynomial at x.  coeffs are in order of degree, so coeff[0] is the constant term."""
    coeffs_ptr = get_array_ptr(coeffs)
    return mus_polynomial(coeffs_ptr, x, len(coeffs))

def array_interp(fn, x: float, size: int):
    """Taking into account wrap-around (size is size of data), with linear interpolation if phase is not an integer."""    
    fn_ptr = get_array_ptr(fn)
    return mus_array_interp(fn_ptr, x, size)

def bessi0(x: float):
    """Bessel function of zeroth order"""
    mus_bessi0(x)
    
def mus_interpolate(type, x: float, v, size: int, y1: float):
    """Interpolate in data ('v' is a ndarray) using interpolation 'type', such as Interp.LINEAR."""
    v_ptr = get_array_ptr(v)
    return mus_interpolate(x, v_ptr, size, y1)
    
def mus_fft(rdat, idat, fftsize: int, sign: int):
    """Return the fft of rl and im which contain the real and imaginary parts of the data; len should be a power of 2, dir = 1 for fft, -1 for inverse-fft"""    
    rdat_ptr = get_array_ptr(rdat)
    idat_ptr = get_array_ptr(idat)    
    return mus_fft(rdat_ptr, idat_ptr, fftsize, sign)

def make_fft_window(type: int, size: int, beta: Optional[float]=0.0, alpha: Optional[float]=0.0):
    """fft data window (a ndarray). type is one of the sndlib fft window identifiers such as Window.KAISER, beta is the window family parameter, if any."""
    win = np.zeros(size, dtype=np.double)
    mus_make_fft_window_with_window(type, size, beta, alpha, win.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return win

def rectangular2polar(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64]):
    """Convert real/imaginary data in s rl and im from rectangular form (fft output) to polar form (a spectrum)"""
    size = len(rdat)
    mus_rectangular_to_polar(rdat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), idat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size)


def rectangular2magnitudes(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64]):
    """Convert real/imaginary  data in rl and im from rectangular form (fft output) to polar form, but ignore the phases"""
    size = len(rdat)
    mus_rectangular_to_magnitudes(rdat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), idat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size)


def polar2rectangular(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64]):
    """Convert real/imaginary data in rl and im from polar (spectrum) to rectangular (fft)"""
    size = len(rdat)
    # want to return new ndarrays instead of changing in-place
    mus_polar_to_rectangular(rdat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), idat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size)


def spectrum(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64], window, norm_type: int):
    """Real and imaginary data in ndarrays rl and im, returns (in rl) the spectrum thereof; window is the fft data window (a ndarray as returned by 
        make_fft_window  and type determines how the spectral data is scaled:
              0 = data in dB,
              1 (default) = linear and normalized
              2 = linear and un-normalized."""
    if isinstance(window, list):
        window = np.array(window, dtype=np.double)
    size = len(rdat)
    # want to return new ndarray instead of changing in-place

    mus_spectrum(rdat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), idat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), window.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size, norm_type)
    return rdat

def convolution(rl1: npt.NDArray[np.float64], rl2: npt.NDArray[np.float64]):
    """Convolution of ndarrays v1 with v2, using fft of size len (a power of 2), result in v1"""
    size = len(rl1)
    # want to return new ndarrays instead of changing in-place
    mus_convolution(rl1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), rl2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size)
    return rl1

def autocorrelate(data: npt.NDArray[np.float64]):
    """In place autocorrelation of data (a ndarray)"""
    size = len(data)
    mus_autocorrelate(data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size)
    return data
    
def correlate(data1: npt.NDArray[np.float64], data2: npt.NDArray[np.float64]):
    """In place cross-correlation of data1 and data2 (both ndarrays)"""
    size = len(data1)
    # want to return new ndarrays instead of changing in-place
    mus_correlate(data1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), data2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size)
    return data1

def cepstrum(data: npt.NDArray[np.float64]):
    """Return cepstrum of signal"""
    size = len(data)
    mus_cepstrum(data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), size)
    return data
    
def partials2wave(partials, wave=None, norm: Optional[bool]=True ):
    """Take a list of partials (harmonic number and associated amplitude) and produce a waveform for use in table_lookup"""
    partials_ptr = get_array_ptr(partials)                
    if isinstance(wave, list):
        wave = np.array(wave, dtype=np.double)
                        
    if (not wave):
        wave = np.zeros(CLM.table_size)
    
    mus_partials_to_wave(partials_ptr, len(partials) // 2, wave.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(wave), norm)
    # return a ndarray
    return wave
    
def phase_partials2wave(partials, wave=None, norm: Optional[bool]=True ):
    """Take a list of partials (harmonic number, amplitude, initial phase) and produce a waveform for use in table_lookup"""
    if isinstance(partials, list):
        partials = np.array(partials, dtype=np.double)        
        
    partials_ptr = get_array_ptr(partials)        
                    
    if (not wave):
        wave = np.zeros(CLM.table_size)
        
    mus_partials_to_wave(partials_ptr, len(partials) // 3, wave.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(wave), norm)
    return wave
    
def partials2polynomial(partials, kind: Optional[int]=Polynomial.FIRST_KIND):
    """Returns a Chebyshev polynomial suitable for use with the polynomial generator to create (via waveshaping) the harmonic spectrum described by the partials argument."""
    if isinstance(partials, list):
        partials = np.array(partials, dtype=np.double)
    mus_partials_to_polynomial(len(partials), partials.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) ,kind)
    return partials

def normalize_partials(partials):
    """Scales the partial amplitudes in the list/array or list 'partials' by the inverse of their sum (so that they add to 1.0)."""
    if isinstance(partials, list):
        partials = np.array(partials, dtype=np.double)    
    mus_normalize_partials(len(partials), partials.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return partials
    
def chebyshev_tu_sum(x: float, tcoeffs, ucoeffs):
    """Returns the sum of the weighted Chebyshev polynomials Tn and Un (vectors or " S_vct "s), with phase x"""
    
    tcoeffs_ptr = get_array_ptr(tcoeffs)
    ucoeffs_ptr = get_array_ptr(ucoeffs)

    return mus_chebyshev_tu_sum(x, tcoeffs_ptr, ucoeffs_ptr)
    
def chebyshev_t_sum(x: float, tcoeffs):
    """returns the sum of the weighted Chebyshev polynomials Tn"""
    tcoeffs_ptr = get_array_ptr(tcoeffs)
    return mus_chebyshev_t_sum(x,tcoeffs_ptr)

def chebyshev_u_sum(x: float, ucoeffs):
    """returns the sum of the weighted Chebyshev polynomials Un"""
    ucoeffs_ptr = get_array_ptr(ucoeffs)
    return mus_chebyshev_tu_sum(x, ucoeffs_ptr)
        
#########################################
# basic file reading/writing to/from nympy arrays
# note all of this assumes numpy dtype is np.double
# also assumes shape is (chan, length)
# a mono file that is 8000 samples long should
# be a numpy array created with something like
# arr = np.zeroes((1,8000), dtype=np.double)
# librosa follows this convention
#  TODO : look at allowing something like np.zeroes((8000), dtype=np.double)
# to just be treated as mono sound buffer 
# the python soundfile library does this differently 
# and would use 
# arr = np.zeros((8000,1), dtype=np.double))
# this seems less intuitive to me 
# very issue to translate with simple np.transpose()


def sndinfo(filename):
    """Returns a dictionary of info about a sound file including write date (data), sample rate (srate),
    channels (chans), length in samples (samples), length in second (length), comment (comment), and loop information (loopinfo)"""
    date = mus_sound_write_date(filename)
    srate = mus_sound_srate(filename)
    chans = mus_sound_chans(filename)
    samples = mus_sound_samples(filename)
    comment = mus_sound_comment(filename) 
    length = samples / (chans * srate)

    header_type = Header(mus_sound_header_type(filename))
    sample_type = Sample(mus_sound_sample_type(filename))
    
    loop_info = mus_sound_loop_info(filename)
    if loop_info:
        loop_modes = [loop_info[6], loop_info[7]]
        loop_starts = [loop_info[0], loop_info[2]]
        loop_ends = [loop_info[1], loop_info[3]]
        base_note = loop_info[4]
        base_detune = loop_info[5]
    
        loop_info = {'loop_modes' : loop_modes, 'loop_start' : loop_starts, 'loop_ends' : loop_ends, 
                       'base_note' : base_note,  'base_detune' : base_detune}
    
    info = {'date' : time.localtime(date), 'srate' : srate, 'chans' : chans, 'samples' : samples,
            'comment' : comment, 'length' : length, 'header_type' : header_type, 'sample_type' : sample_type,
            'loop_info' : loop_info}
            
    
    return info


def file2array(filename):
    """Return an ndarray with samples from file and the sample rate of the data"""
    length = mus_sound_framples(filename)
    chans = mus_sound_chans(filename)
    srate = mus_sound_srate(filename)
    out = np.zeros((chans, length), dtype=np.double)
    for i in range(chans):
        mus_file_to_array(filename,i, 0, length, out[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return out, srate


def channel2array(filename):
    length = mus_sound_framples(filename)
    chans = mus_sound_chans(filename)
    srate = mus_sound_srate(filename)
    out = np.zeros((1, length), dtype=np.double)
    mus_file_to_array(filename,i, 0, length, out[0].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return out, srate

    
def array2file(arr, filename: str, sr=None,sample_type=CLM.sample_type, header_type=CLM.header_type, comment=None ):
    """Write an ndarray of samples to file"""
    if not sr:
        sr = CLM.srate
    
    chans = np.shape(arr)[0]
    length = np.shape(arr)[1]
    fd = mus_sound_open_output(filename, int(sr), chans, sample_type, header_type, comment)
 
    obuftype = POINTER(c_double) * chans
    obuf = obuftype()
    
    for i in range(chans):
        obuf[i] = arr[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    err = mus_sound_write(fd, 0, length-1, chans, obuf)
    
    mus_sound_close_output(fd, length*mus_bytes_per_sample(sample_type)*chans)


########context with sound#######    

# TODO: Clipping etc, 

class Sound(object):
    output = None
    reverb = None

    def __init__(self, output, 
                        channels=None, 
                        srate=None, 
                        sample_type = None,
                        header_type = None,
                        comment = False,
                        verbose = False,
                        reverb = None,
                        revfile = None,
                        reverb_data = None,
                        reverb_channels = None,
                        continue_old_file = False,
                        statistics = None,
                        scaled_to = False,
                        scaled_by = False,
                        play = None,
                        clipped = None,
                        ignore_output = False):
        self.output = output
        self.channels = channels or CLM.channels
        self.srate = srate or CLM.srate
        self.sample_type = sample_type or CLM.sample_type
        self.header_type = header_type or CLM.header_type
        self.comment = comment
        self.verbose = verbose or CLM.verbose
        self.reverb = reverb  or CLM.reverb
        self.revfile = revfile or CLM.reverb_file_name
        self.reverb_data = reverb_data or CLM.reverb_data
        self.reverb_channels = reverb_channels or CLM.reverb_channels
        self.continue_old_file = continue_old_file
        self.statistics = statistics or CLM.statistics
        self.scaled_to = scaled_to
        self.scaled_by = scaled_by
        self.play = play or CLM.play
        self.clipped = clipped or CLM.clipped
        self.ignore_output = ignore_output
        self.output_to_file = isinstance(self.output, str)
        self.reverb_to_file = self.reverb and isinstance(self.output, str)
        self.old_srate = get_srate()

    def __enter__(self):
        
    
        #TODO: this all needs to be cleaned up. I think it could be simplified and does not need all the options in scheme version
    
        set_srate(self.srate)

        # in original why use reverb-1?
        if  self.statistics :
            self.tic = time.perf_counter()
        
        if self.output_to_file :
            # writing to File
            #continue_sample2file
            if self.continue_old_file:
                Sound.output = continue_sample2file(self.filename)
                set_srate(mus_sound_srate(self.filename)) # maybe print warning or at least note 
            else:
                
                Sound.output = make_sample2file(self.output,self.channels, sample_type=self.sample_type , header_type=self.header_type)
        elif is_iterable(self.output):
            Sound.output = self.output

        else:
            print("not supported writing to", self.output)
            
        if self.reverb_to_file:
            if self.continue_old_file:
                Sound.reverb = continue_sample2file(self.revfile)
            else:
                Sound.reverb = make_sample2file(self.revfile,self.reverb_channels, sample_type=self.sample_type , header_type=self.header_type)
        
        if self.reverb and not self.reverb_to_file and is_iterable(self.output):
            Sound.reverb = np.zeros((self.reverb_channels, np.shape(Sound.output)[1]), dtype=Sound.output.dtype)
    

        return self
        
    def __exit__(self, *args):

        if self.reverb: 
            if self.reverb_to_file:
                mus_close(Sound.reverb)
                Sound.reverb = make_file2sample(self.revfile)
                
                if self.reverb_data:
                    self.reverb(**self.reverb_data)
                else:
                    self.reverb()
                mus_close(Sound.reverb)

            if is_iterable(Sound.reverb):

                if self.reverb_data:
                    print("applying reverb with data")
                    self.reverb(**self.reverb_data)
                else:
                    self.reverb()          
                
        if self.output_to_file:
            mus_close(Sound.output)
            
            
        # Statistics and scaling go here    
        if  self.statistics :
            toc = time.perf_counter()
            print(f"Total processing time {toc - self.tic:0.4f} seconds")
            
         
        if self.play and self.output_to_file:
            subprocess.run([CLM.player,self.output])
        # need some safety if errors
        
        set_srate(self.old_srate)
        



### Passing None to any generator function will cause a segmentation fault
### so including check for that

def raise_none_error(gen):
    raise TypeError (f"cannot pass None in place of gen {gen}.") # going to call this a type error instead of a ValueError ?

# ---------------- oscil ---------------- #
def make_oscil( frequency: Optional[float]=0., initial_phase: Optional[float] = 0.0):
    """Return a new oscil (sinewave) generator"""
    return mus_make_oscil(frequency, initial_phase)
    
def oscil(os: MUS_ANY_POINTER, fm: Optional[float]=None, pm: Optional[float]=None):
    """Return next sample from oscil  gen: val = sin(phase + pm); phase += (freq + fm)"""
    if os == None:
        raise_none_error('oscil')
    if not fm:
        if not pm:
            return mus_oscil_unmodulated(os)
        else: 
            return mus_oscil_pm(os, pm)
    else:
        return mus_oscil_fm(os, fm)
    
def is_oscil(os: MUS_ANY_POINTER):
    """Returns True if gen is an oscil"""
    return mus_is_oscil(os)

    
# ---------------- oscil-bank ---------------- #

def make_oscil_bank(freqs, phases, amps, stable: Optional[bool]=False):
    """Return a new oscil-bank generator. (freqs in radians)"""
    freqs_ptr = get_array_ptr(freqs)
    phases_ptr = get_array_ptr(phases)
    amps_ptr = get_array_ptr(amps)

    gen =  mus_make_oscil_bank(len(freqs), freqs_ptr, phases_ptr, amps_ptr, stable)
    gen._cache = [freqs_ptr, phases_ptr, amps_ptr]
    return gen

def oscil_bank(os: MUS_ANY_POINTER):
    """sum an array of oscils"""
    if os == None:
        raise TypeError (f"cannot pass None in place of gen oscil_bank.")
    #fms_prt = get_array_ptr(fms)
    return mus_oscil_bank(os)
    
def is_oscil_bank(os: MUS_ANY_POINTER):
    """Returns True if gen is an oscil_bank"""
    return mus_is_oscil_bank(os)
    
# ---------------- env ---------------- #
def make_env(envelope, scaler: Optional[float]=1.0, duration: Optional[float]=1.0, offset: Optional[float]=0.0, base: Optional[float]=1.0, length: Optional[int]=0):
    """Return a new envelope generator.  'envelope' is a list/array of break-point pairs. To create the envelope, these points are offset by 'offset', scaled by 'scaler', and mapped over the time interval defined by
        either 'duration' (seconds) or 'length' (samples).  If 'base' is 1.0, the connecting segments 
        are linear, if 0.0 you get a step function, and anything else produces an exponential connecting segment."""
    if length > 0:
        duration = samples2seconds(length)
    envelope_ptr = get_array_ptr(envelope)

    gen =  mus_make_env(envelope_ptr, len(envelope) // 2, scaler, offset, base, duration, 0, None)
    gen._cache = [envelope_ptr]
    return gen
    
def env(e: MUS_ANY_POINTER):
    if e == None:
        raise_none_error('env')
    return mus_env(e)
    
def is_env(e: MUS_ANY_POINTER):
    """Returns True if gen is an env"""
    return mus_is_env(e)
    
def env_interp(x: float, env: MUS_ANY_POINTER):
    if e == None:
        raise_none_error('env')
    return mus_env_interp(x, env)
    
def envelope_interp(x: float, env: MUS_ANY_POINTER):
    if e == None:
        raise_none_error('env')
    return mus_env_interp(x, env)

# TODO needs testing    
# def env_any(e: MUS_ANY_POINTER, connection_function):
#     return mus_env_any(e, FF(connection_function))

# ---------------- pulsed-env ---------------- #    
def make_pulsed_env(envelope, duration, frequency):    
    pl = mus_make_pulse_train(frequency, 1.0, 0.0)
    ge = make_env(envelope, scaler=1.0, duration=duration)
    gen = mus_make_pulsed_env(ge, pl)
    gen._cache = [pl, ge]
    return 
    
def pulsed_env(gen: MUS_ANY_POINTER, fm: Optional[float]=None):
    if gen == None:
        raise_none_error('pulsed_env')
    if(fm):
        return mus_pulsed_env(gen, fm)
    else:
        return mus_pulsed_env_unmodulated(gen)
        
# TODO envelope-interp different than env-interp

def is_pulsedenv(e: MUS_ANY_POINTER):
    """Returns True if gen is a pulsed_env"""
    return mus_is_pulsed_env(e)

# ---------------- table-lookup ---------------- #
def make_table_lookup(frequency: Optional[float]=0.0, 
                        initial_phase: Optional[float]=0.0, 
                        wave=None, 
                        size: Optional[int]=512, 
                        type: Optional[int]=Interp.LINEAR):        
    """Return a new table_lookup generator. The default table size is 512; use :size to set some other size, or pass your own list/array as the 'wave'."""                        
                        
    wave_ptr = get_array_ptr(wave)            
    gen =  mus_make_table_lookup(frequency, initial_phase, wave_ptr, size, type)
    gen._cache = [wave_ptr]
    return gen
    
def table_lookup(tl: MUS_ANY_POINTER, fm_input: Optional[float]=None):
    if tl == None:
        raise_none_error('table_lookup')
    if fm_input:
        return mus_table_lookup(tl, fm_input)
    else:
        return mus_table_lookup_unmodulated(tl)
        
def is_table_lookup(tl: MUS_ANY_POINTER):
    """Returns True if gen is a table_lookup"""
    return mus_is_table_lookup(tl)

# TODO make-table-lookup-with-env

# ---------------- polywave ---------------- #
def make_polywave(frequency: float, 
                    partials = [0.,1.], 
                    type: Optional[int]=Polynomial.FIRST_KIND, 
                    xcoeffs = None, 
                    ycoeffs =None):
    """Return a new polynomial-based waveshaping generator. make_polywave(440.0, partials=[1.0,1.0]) is the same in effect as make_oscil"""

    if(xcoeffs and ycoeffs): # should check they are same length
        xcoeffs_ptr = get_array_ptr(xcoeffs)
        ycoeffs_ptr = get_array_ptr(ycoeffs)
        gen = mus_make_polywave_tu(frequency,xcoeffs_ptr,ycoeffs_ptr, len(xcoeffs))
        gen._cache = [xcoeffs_ptr,ycoeffs_ptr]
        return gen
    else:
        prtls = normalize_partials(partials)
        prtls_ptr = get_array_ptr(prtls)
        gen = mus_make_polywave(frequency, prtls_ptr, len(partials), type)
        gen._cache = [prtls_ptr]
        return gen
    
    
def polywave(w: MUS_ANY_POINTER, fm: Optional[float]=None):
    """Next sample of polywave waveshaper"""
    if w == None:
        raise_none_error('polywave')
    if fm:
        return mus_polywave(w, fm)
    else:
        return mus_polywave_unmodulated(w)
        
def is_polywave(w: MUS_ANY_POINTER):
    """Returns True if gen is a polywave"""
    return mus_is_polywave(w)


# ---------------- polyshape ---------------- #
# TODO like in docs should be a coeffs argument but don't see how
def make_polyshape(frequency: float, 
                    initial_phase: float, 
                    partials: [1.,1.], 
                    kind: Optional[int]=Polynomial.FIRST_KIND):
                    
    """Return a new polynomial-based waveshaping generator."""

    poly = partials2polynomial(partials)    
    poly_ptr = get_array_ptr(poly)

    gen = mus_make_polyshape(frequency, initial_phase, poly_ptr, len(partials))
    gen._cache = [poly_ptr]
    return gen
    
def polyshape(w: MUS_ANY_POINTER, index: Optional[float]=1.0, fm: Optional[float]=None):
    """Next sample of polynomial-based waveshaper"""
    if w == None:
        raise_none_error('polyshape')
    if fm:
        return mus_polyshape(w, index, fm)
    else:
        return mus_polyshape_unmodulated(w, index)
        
def is_polyshape(w: MUS_ANY_POINTER):
    """Returns True if gen is a polyshape"""
    return mus_is_polyshape(w)
    

# ---------------- triangle-wave ---------------- #    
def make_triangle_wave(frequency: float, amplitude: Optional[float]=1.0, phase: Optional[float]=0.0):
    """return a new triangle_wave generator."""
    return mus_make_triangle_wave(frequency, amplitude, phase)
    
def triangle_wave(s: MUS_ANY_POINTER, fm: float=None):
    """next triangle wave sample from generator"""
    if s == None:
        raise_none_error('triangle_wave')
    if fm:
        return mus_triangle_wave(s)
    else:
        return mus_triangle_wave_unmodulated(s)
    
def is_triangle_wave(s: MUS_ANY_POINTER):
    """Returns True if gen is a triangle_wave"""
    return mus_is_triangle_wave(s)

# ---------------- square-wave ---------------- #    
def make_square_wave(frequency: float, amplitude: Optional[float]=1.0, phase: Optional[float]=0.0):
    """Return a new square_wave generator."""
    return mus_make_square_wave(frequency, amplitude, phase)
    
def square_wave(s: MUS_ANY_POINTER, fm: float=None):
    """next square wave sample from generator"""
    if s == None:
        raise_none_error('square_wave')
    if fm:
        return mus_square_wave(s)
    else:
        return mus_square_wave_unmodulated(s)
    
def is_square_wave(s: MUS_ANY_POINTER):
    """Returns True if gen is a square_wave"""
    return mus_is_square_wave(s)
    
# ---------------- sawtooth-wave ---------------- #    
def make_sawtooth_wave(frequency: float, amplitude: Optional[float]=1.0, phase: Optional[float]=0.0):
    """Return a new sawtooth_wave generator."""
    return mus_mus_make_sawtooth_wave(frequency, amplitude, phase)
    
def sawtooth_wave(s: MUS_ANY_POINTER):
    """next sawtooth wave sample from generator"""
    if s == None:
        raise_none_error('sawtooth_wave')
    if fm:
        return mus_sawtooth_wave(s)
    else:
        return mus_sawtooth_wave_unmodulated(s)
    
def is_sawtooth_wave(s: MUS_ANY_POINTER):
    """Returns True if gen is a sawtooth_wave"""
    return mus_is_sawtooth_wave(s)

# ---------------- pulse-train ---------------- #        
def make_pulse_train(frequency: float, amplitude: Optional[float]=1.0, phase: Optional[float]=0.0):
    """return a new pulse_train generator. This produces a sequence of impulses."""
    return mus_make_pulse_train(frequency, amplitude, phase)
    
def pulse_train(s: MUS_ANY_POINTER, fm: Optional[float]=None):
    """next pulse train sample from generator"""
    if s == None:
        raise_none_error('pulse_train')
    if fm:
        return mus_pulse_train(s, fm)
    else:
        return mus_pulse_train_unmodulated(s)
     
def is_pulse_train(s: MUS_ANY_POINTER):
    """Returns True if gen is a pulse_train"""
    return mus_is_pulse_train()
    
    
# ---------------- ncos ---------------- #

def make_ncos(frequency: float, n: Optional[int]=1):
    """return a new ncos generator, producing a sum of 'n' equal amplitude cosines."""
    return mus_make_ncos(frequency, n)
    
def ncos(nc: MUS_ANY_POINTER, fm: Optional[float]=0.0):
    """Get the next sample from 'gen', an ncos generator"""
    if nc == None:
        raise_none_error('ncos')
    return mus_ncos(nc, fm)

def is_ncos(nc: MUS_ANY_POINTER):
    """Returns True if gen is a ncos"""
    return mus_is_ncos(nc)
    
    
# ---------------- nsin ---------------- #
def make_nsin(frequency: float, n: Optional[int]=1):
    """return a new nsin generator, producing a sum of 'n' equal amplitude sines"""
    return mus_make_nsin(frequency, n)
    
def nsin(nc: MUS_ANY_POINTER, fm: Optional[float]=0.0):
    """Get the next sample from 'gen', an nsin generator"""
    if nc == None:
        raise_none_error('nsin')
    return mus_nsin(nc, fm)
    
def is_nsin(nc: MUS_ANY_POINTER):
    """Returns True if gen is a nsin"""
    return mus_is_nsin(nc)
    
# ---------------- nrxysin and nrxycos ---------------- #

def make_nrxysin(frequency: float, ratio: Optional[float]=1., n: Optional[int]=1, r: Optional[float]=.5):
    """Return a new nrxysin generator."""
    return mus_make_nrxysin(frequency, ratio, n, r)
    
def nrxysin(s: MUS_ANY_POINTER, fm: Optional[float]=0.):
    """next sample of nrxysin generator"""
    if s == None:
        raise_none_error('nrxysin')
    return mus_nrxysin(s, fm)
    
def is_nrxysin(s: MUS_ANY_POINTER):
    """Returns True if gen is a nrxysin"""
    return mus_is_nrxysin(s)
    
    
def make_nrxycos(frequency: float, ratio: Optional[float]=1., n: Optional[int]=1, r: Optional[float]=.5):
    """Return a new nrxycos generator."""
    return mus_make_nrxycos(frequency, ratio, n, r)
    
def nrxycos(s: MUS_ANY_POINTER, fm: Optional[float]=0.):
    """next sample of nrxycos generator"""
    if s == None:
        raise_none_error('nrxycos')
    return mus_nrxycos(s, fm)
    
def is_nrxycos(s: MUS_ANY_POINTER):
    """Returns True if gen is a nrxycos"""
    return mus_is_nrxycos(s)
    
    
# ---------------- rxykcos and rxyksin ---------------- #    
def make_rxykcos(frequency: float, phase: float, r: Optional[float]=.5, ratio: Optional[float]=1.):
    """Return a new rxykcos generator."""
    return mus_make_rxykcos(frequency, phase, r, ratio)
    
def rxykcos(s: MUS_ANY_POINTER, fm: Optional[float]=0.):
    """next sample of rxykcos generator"""
    if s == None:
        raise_none_error('rxykcos')
    return mus_rxykcos(s, fm)
    
def is_rxykcos(s: MUS_ANY_POINTER):
    """Returns True if gen is a rxykcos"""
    return mus_is_rxykcos(s)

def make_rxyksin(frequency: float, phase: float, r: Optional[float]=.5, ratio: Optional[float]=1.):
    """Return a new rxyksin generator."""
    return mus_make_rxyksin(frequency, phase, r, ratio)

def rxyksin(s: MUS_ANY_POINTER, fm: Optional[float]=0.):
    """next sample of rxyksin generator"""
    if s == None:
        raise_none_error('rxyksin')
    return mus_rxyksin(s, fm)
    
def is_rxyksin(s: MUS_ANY_POINTER):
    """Returns True if gen is a rxyksin"""
    return mus_is_rxyksin(s)
        
# ---------------- ssb-am ---------------- #    

def make_ssb_am(frequency: float, n: Optional[int]=40):
    """Return a new ssb_am generator."""
    return mus_make_ssb_am(frequency, n)
    
def ssb_am(gen: MUS_ANY_POINTER, insig: Optional[float]=0.0, fm: Optional[float]=None):
    """get the next sample from ssb_am generator"""
    if gen == None:
        raise_none_error('ssb_am')
    if(fm):
        return mus_ssb_am(gen, insig, fm)
    else:
        return mus_ssb_am_unmodulated(gen, insig)
        
def is_ssb_am(gen: MUS_ANY_POINTER):
    """Returns True if gen is a ssb_am"""
    return mus_is_ssb_am(gen)



# ---------------- wave-train ----------------#
def make_wave_train(frequency: float, wave, phase: Optional[float]=0., type=Interp.LINEAR):
    """Return a new wave-train generator (an extension of pulse-train). Frequency is the repetition rate of the wave found in wave. Successive waves can overlap."""

    wave_ptr = get_array_ptr(wave)
    gen = mus_make_wave_train(frequency, phase, wave_ptr, len(wave), type)
    gen._cache = [wave_ptr]
    return gen
    
def wave_train(w: MUS_ANY_POINTER, fm: Optional[float]=None):
    """next sample of wave_train"""
    if w == None:
        raise_none_error('wave_train')
    if fm:
        return mus_wave_train(w, fm)
    else:
        return mus_wave_train_unmodulated(w)
    
def is_wave_train(w: MUS_ANY_POINTER):
    """Returns True if gen is a wave_train"""    
    return mus_is_wave_train(w)


def make_wave_train_with_env(frequency: float, pulse_env, size=None):
    size = size or CLM.table_size
    ve = np.zero(size)
    e = make_env(pulse_env, length=size)
    for i in range(size):
        ve[i] = env(e)
    return make_wave_train(frequency, ve)    

# ---------------- rand, rand_interp ---------------- #
def make_rand(frequency: float, amplitude: Optional[float]=1.0, distribution=None):
    """Return a new rand generator, producing a sequence of random numbers (a step  function). frequency is the rate at which new numbers are chosen."""
    if (distribution):
        distribution_ptr = get_array_ptr(distribution)
        gen =  mus_make_rand_with_distribution(frequency, amplitude, distribution_ptr, len(distribution))
        gen._cache= [distribution_ptr]
        return gen
    else:
        return mus_make_rand(frequency, amplitude)

def rand(r: MUS_ANY_POINTER, sweep: Optional[float]=None):
    """gen's current random number. fm modulates the rate at which the current number is changed."""
    if r == None:
        raise_none_error('rand')
    if(sweep):
        return mus_rand(r, sweep)
    else:
        return mus_rand_unmodulated(r)
    
def is_rand(r: MUS_ANY_POINTER):
    """Returns True if gen is a rand"""
    return mus_is_rand(r)

def make_rand_interp(frequency: float, amplitude: float,distribution=None):
    """Return a new rand_interp generator, producing linearly interpolated random numbers. frequency is the rate at which new end-points are chosen."""
    
    if (distribution):
        distribution_ptr = get_array_ptr(distribution)
        gen = mus_make_rand_interp_with_distribution(frequency, amplitude, distribution_ptr, len(distribution))
        gen._cache = [distribution_ptr]
        return gen
    else:
        return mus_make_rand_interp(frequency, amplitude)
    
def rand_interp(r: MUS_ANY_POINTER, sweep: Optional[float]=0.):
    """gen's current (interpolating) random number. fm modulates the rate at which new segment end-points are chosen."""
    if r == None:
        raise_none_error('rand_interp')
    if(sweep):
        return mus_rand_interp(r, sweep)
    else:
        return mus_rand_interp_unmodulated(r)
    
def is_rand_interp(r: MUS_ANY_POINTER):
    """Returns True if gen is a rand_interp"""
    return mus_is_rand_interp(r)    
    
    
# ---------------- simple filters ---------------- #

def make_one_pole(a0: float, b1: float):
    """Return a new one_pole filter; a0*x(n) - b1*y(n-1)"""

    return mus_make_one_pole(a0, b1)
    
def one_pole(f: MUS_ANY_POINTER, input: float):
    """One pole filter of input."""
    if f == None:
        raise_none_error('one_pole')
    return mus_one_pole(f, input)
    
def is_one_pole(f: mus_any):
    """Returns True if gen is an one_pole"""
    return mus_is_one_pole(f)
    
def make_one_zero(a0: float, a1: float):
    """Return a new one_zero filter; a0*x(n) + a1*x(n-1)"""
    return mus_make_one_zero(a0, a1)
    
def one_zero(f: MUS_ANY_POINTER, input: float):
    """One zero filter of input."""
    if f == None:
        raise_none_error('one_zero')
    return mus_one_zero(f, input)
    
def is_one_zero(f: MUS_ANY_POINTER):
    """Returns True if gen is an one_zero"""
    return mus_is_one_zero(f)    

# TODO implement keyword args as well
def make_two_pole(*args):
    """Return a new two_pole filter; a0*x(n) - b1*y(n-1) - b2*y(n-2)"""

    if(len(args) == 2):
        return mus_make_two_pole_from_frequency_and_radius(args[0], args[1])
    elif(len(args) == 3):
        return mus_make_two_pole(args[0], args[1], args[2])
    else:
        print("error") # make this real error

def two_pole(f: MUS_ANY_POINTER, input: float):
    """Return a new two_pole filter; a0*x(n) - b1*y(n-1) - b2*y(n-2)"""
    if f == None:
        raise_none_error('two_pole')
    return mus_two_pole(f, input)
    
def is_two_pole(f: MUS_ANY_POINTER):
    return mus_is_two_pole(f)

# TODO implement keyword args as well
def make_two_zero(*args):
    """Return a new two_zero filter; a0*x(n) + a1*x(n-1) + a2*x(n-2)"""
    if(len(args) == 2):
        return mus_make_two_zero_from_frequency_and_radius(args[0], args[1])
    elif(len(args) == 3):
        return mus_make_two_zero(args[0], args[1], args[2])
    else:
        print("error") # make this real error

def two_zero(f: MUS_ANY_POINTER, input: float):
    """Two zero filter of input."""
    if f == None:
        raise_none_error('two_zero')
    return mus_two_zero(f, input)
    
def is_two_zero(f: MUS_ANY_POINTER):
    """Returns True if gen is a two_zero"""
    return mus_is_two_zero(f)

# ---------------- formant ---------------- #

def make_formant(frequency: float, radius: float):
    """Return a new formant generator (a resonator). radius sets the pole radius (in terms of the 'unit circle').
        frequency sets the resonance center frequency (Hz)."""
    return mus_make_formant(frequency, radius)

def formant(f: MUS_ANY_POINTER, input: float, radians: Optional[float]=None):
    """Next sample from resonator generator."""
    if f == None:
        raise_none_error('formant')
    if fm:
        return mus_formant_with_frequency(f, input, radians)
    else:
        return mus_formant(f, input)
    
def is_formant(f: MUS_ANY_POINTER):
    """Returns True if gen is a formant"""
    return mus_is_formant(f)
    
    
# ---------------- formant-bank ---------------- #
    
def make_formant_bank(filters, amps):
    """Return a new formant-bank generator."""
    if False in (is_formant(v) for v in filters):
        raise TypeError(f'filter list contains at least one element that is not a formant.')
    filt_array = (POINTER(mus_any) * len(filters))()
    filt_array[:] = [filters[i] for i in range(len(filters))]    
    amps_ptr = get_array_ptr(amps)
    
    gen = mus_make_formant_bank(len(filters),filt_array, amps_ptr)
    gen._cache = [filt_array, amps_ptr]
    return gen
    
def formant_bank(f: MUS_ANY_POINTER, inputs):
    """Sum a bank of formant generators"""
    if f == None:
        raise_none_error('formant_bank')
    if inputs:
        inputs_ptr = get_array_ptr(inputs)
        gen =  mus_formant_bank_with_inputs(f, inputs_ptr)
        gen._cache.append(inputs_ptr)
        return gen
    else:
        return mus_formant_bank(f, inputs)
        
def is_formant_bank(f: MUS_ANY_POINTER):
    """Returns True if gen is a formant_bank"""
    return mus_is_formant_bank(f)
    

# ---------------- firmant ---------------- #

def make_firmant(frequency: float, radius: float):
    """Return a new firmant generator (a resonator).  radius sets the pole radius (in terms of the 'unit circle').
        frequency sets the resonance center frequency (Hz)."""

    return mus_make_firmant(frequency, radius)

def firmant(f: MUS_ANY_POINTER, input: float,radians: Optional[float]=None ):
    """Next sample from resonator generator."""
    if f == None:
        raise_none_error('firmant')
    if radians:
        return mus_firmant_with_frequency(f, input, frequency, radians)
    else: 
        return mus_firmant(f, input)
            
def is_firmant(f: MUS_ANY_POINTER):
    """Returns True if gen is a firmant"""
    return mus_is_firmant(f)


#;; the next two are optimizations that I may remove
#mus-set-formant-frequency f frequency
#mus-set-formant-radius-and-frequency f radius frequency

# ---------------- filter ---------------- #

def make_filter(order: int, xcoeffs, ycoeffs):   
    """Return a new direct form FIR/IIR filter, coeff args are list/ndarray"""     
    xcoeffs_ptr = get_array_ptr(xcoeffs)    
    ycoeffs_ptr = get_array_ptr(ycoeffs)    
    gen =  mus_make_filter(order, xcoeffs_ptr, ycoeffs_ptr, None)
    gen._cache = [xcoeffs_ptr, ycoeffs_ptr]
    return gen
    
def clm_filter(fl: MUS_ANY_POINTER, input: float): # TODO : conflicts with buitl in function
    """next sample from filter"""
    if fl == None:
        raise_none_error('filter')
    return mus_filter(fl, input)
    
def is_filter(fl: MUS_ANY_POINTER):
    """Returns True if gen is a filter"""
    return mus_is_filter(fl)

# ---------------- fir-filter ---------------- #
    
def make_fir_filter(order: int, xcoeffs):
    """return a new FIR filter, xcoeffs a list/ndarray"""
    xcoeffs_ptr = get_array_ptr(xcoeffs)    
    gen =  mus_make_fir_filter(order, xcoeffs_ptr, None)
    gen._cache = [xcoeffs_ptr]
    return gen
    
def fir_filter(fl: MUS_ANY_POINTER, input: float):
    """next sample from fir filter"""
    if fl == None:
        raise_none_error('fir_filter')
    return mus_fir_filter(fl, input)
    
def is_fir_filter(fl: MUS_ANY_POINTER):
    """Returns True if gen is a fir_filter"""
    return mus_is_fir_filter(fl)


# ---------------- iir-filter ---------------- #

def make_iir_filter(order: int, ycoeffs):
    """return a new IIR filter, ycoeffs a list/ndarray"""
    ycoeffs_ptr = get_array_ptr(ycoeffs)    
        
    gen = mus_make_iir_filter(order, ycoeffs_ptr, None)
    gen._cache = [ycoeffs_ptr]
    return gen
    
def iir_filter(fl: MUS_ANY_POINTER, input: float ):
    """next sample from iir filter"""
    if fl == None:
        raise_none_error('iir_filter')
    return mus_iir_filter(fl, input)
    
def is_iir_filter(fl: MUS_ANY_POINTER):
    """Returns True if gen is an iir_filter"""
    return mus_is_iir_filter(fl)



# TODO : This is all wrong!!    
# def make_fir_coeffs(order: int, v: npt.NDArray[np.float64]):
#     if isinstance(v, list):
#         v = np.array(v, dtype=np.double)
#     
#     if isinstance(v, list):
#         fre = (c_double * len(ycoeffs))(*xcoeffs)
#         
#     if isinstance(v, np.ndarray):
#         fre = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))    
#     
#     coeffs = (c_double * (order + 1))()
# 
#     
#     mus_make_fir_coeffs(order, fre, coeffs)
#     return coeffs



# ---------------- delay ---------------- #
# TODO: if max!=size type should default to linear

def make_delay(size: int, 
                initial_contents=None, 
                initial_element: Optional[float]=None, 
                max_size:Optional[int]=None,
                type=Interp.NONE):
    """Return a new delay line of size elements. If the delay length will be changing at run-time, max-size sets its maximum length"""

    
    initial_contents_ptr = None
    
    if not max_size:
        max_size = size
    
    if initial_contents:
        initial_contents_ptr = get_array_ptr(intial_contents)

    elif initial_element:
        initial_contents = np.zeros(max_size)
        initial_contents.fill(initial_element)
        initial_contents_ptr = get_array_ptr(intial_contents)
        

    gen = mus_make_delay(size, None, max_size, type)
    gen._cache = [initial_contents_ptr]
    return gen
    
def delay(d: MUS_ANY_POINTER, input: float, pm: Optional[float]=None):
    """Delay val according to the delay line's length and pm ('phase-modulation').
        If pm is greater than 0.0, the max-size argument used to create gen should have accommodated its maximum value. """
    if d == None:
        raise_none_error('delay')
    if pm:
        return mus_delay(d, input, pm)
    else: 
        return mus_delay_unmodulated(d, input)
        
    
def is_delay(d: MUS_ANY_POINTER):
    """Returns True if gen is a delay"""
    return mus_is_delay(d)

    
def tap(d: MUS_ANY_POINTER, offset: Optional[float]=None):
    """tap the delay generator offset by pm"""
    if d == None:
        raise_none_error('delay')
    if offset:
        return mus_tap(d, offset)
    else:
        return mus_tap_unmodulated(d)
    
def is_tap(d: MUS_ANY_POINTER):
    """Returns True if gen is a tap"""
    return mus_is_tap(d)
    
def delay_tick(d: MUS_ANY_POINTER, input: float):
    """Delay val according to the delay line's length. This merely 'ticks' the delay line forward.
        The argument 'val' is returned."""
    if d == None:
        raise_none_error('delay')
    return mus_delay_tick(d, input)



# ---------------- comb ---------------- #
def make_comb(scaler: Optional[float]=1.0,
                size: Optional[int]=None, 
                initial_contents=None, 
                initial_element: Optional[float]=None, 
                max_size:Optional[int]=None,
                type=Interp.NONE):
    """Return a new comb filter (a delay line with a scaler on the feedback) of size elements. 
        If the comb length will be changing at run-time, max-size sets its maximum length."""                
                
    initial_contents_ptr = None
    
    if not max_size:
        max_size = size
    
    if initial_contents:
        initial_contents_ptr = get_array_ptr(initial_contents)

    elif initial_element:
        initial_contents = np.zeros(max_size)
        initial_contents.fill(initial_element)
        initial_contents_ptr = get_array_ptr(intial_contents)
                

    gen = mus_make_comb(scaler, size, initial_contents_ptr, max_size, type)
    gen._cache = [initial_contents_ptr]
    return gen    
        
        
def comb(cflt: MUS_ANY_POINTER, input: float, pm: Optional[float]=None):
    """Comb filter val, pm changes the delay length."""

    if cflt == None:
        raise_none_error('comb')
    if pm:
        return mus_comb(cflt, input, pm)
    else:
        #print(input)    
        return mus_comb_unmodulated(cflt, input)
    
def is_comb(cflt: MUS_ANY_POINTER):
    """Returns True if gen is a comb"""
    return mus_is_comb(cflt)    

# ---------------- comb-bank ---------------- #
def make_comb_bank(combs: list):
    """Return a new comb-bank generator."""

    if False in (is_comb(v) for v in combs):
        raise TypeError(f'comb list contains at least one element that is not a comb.')
    comb_array = (MUS_ANY_POINTER * len(combs))()
    comb_array[:] = [combs[i] for i in range(len(combs))]
    
    gen = mus_make_comb_bank(len(combs), comb_array)
    gen._cache = comb_array
    return gen

def comb_bank(combs: MUS_ANY_POINTER, input: float):
    """Sum an array of comb filters."""
    if combs == None:
        raise_none_error('comb_bank')
    return mus_comb_bank(combs, input)
    
def is_comb_bank(combs: MUS_ANY_POINTER):
    """Returns True if gen is a comb_bank"""
    return mus_is_comb_bank(combs)




# ---------------- filtered-comb ---------------- #

def make_filtered_comb(scaler: float,
                size: int, 
                filter: Optional[mus_any]=None, #not really optional
                initial_contents=None, 
                initial_element: Optional[float]=0.0, 
                max_size:Optional[int]=None,
                type=Interp.NONE):
                
    """Return a new filtered comb filter (a delay line with a scaler and a filter on the feedback) of size elements.
        If the comb length will be changing at run-time, max-size sets its maximum length."""

    initial_contents_ptr = None
    
    if not max_size:
        max_size = size
    
    if initial_contents:
        initial_contents_ptr = get_array_ptr(intial_contents)

    elif initial_element:
        initial_contents = np.zeros(max_size)
        initial_contents.fill(initial_element)
        initial_contents_ptr = get_array_ptr(intial_contents)
                
    gen = mus_make_filtered_comb(scaler, size, initial_contents_ptr, max_size, type, filter)
    gen._cache = [initial_contents_ptr]
    return gen    
        
    
def filtered_comb(cflt: MUS_ANY_POINTER, input: float, pm: Optional[float]=None):
    """Filtered comb filter val, pm changes the delay length."""
    if cflt == None:
        raise_none_error('filtered_comb')
    if pm:
        return mus_filtered_comb(cflt, input, pm)
    else:
        return mus_filtered_comb_unmodulated(cflt, input)
        
def is_filtered_comb(cflt: MUS_ANY_POINTER):
    """Returns True if gen is a filtered_comb"""
    return mus_is_filtered_comb(cflt)
    
# ---------------- filtered-comb-bank ---------------- #    
# TODO: cache if initial contents    
def make_filtered_comb_bank(fcombs: list):
    """Return a new filtered_comb-bank generator."""
    if False in (is_filtered_comb(v) for v in filters):
        raise TypeError(f'filter list contains at least one element that is not a filtered_comb.')
    fcomb_array = (POINTER(mus_any) * len(fcombs))()
    fcomb_array[:] = [fcombs[i] for i in range(len(fcombs))]
    gen =  mus_make_filtered_comb_bank(len(fcombs), fcomb_array)
    gen._cache = [fcomb_array]
    return gen

def filtered_comb_bank(fcomb: MUS_ANY_POINTER):
    """sum an array of filtered_comb filters."""
    if fcomb == None:
        raise_none_error('filtered_comb_bank')
    return mus_filtered_comb_bank(fcombs, input)
    
def is_filtered_comb_bank(fcombs: MUS_ANY_POINTER):
    """Returns True if gen is a filtered_comb_bank"""
    return mus_is_filtered_comb_bank(fcombs)

# ---------------- notch ---------------- #

def make_notch(scaler: Optional[float]=1.0,
                size: Optional[int]=None, 
                initial_contents=None, 
                initial_element: Optional[float]=0.0, 
                max_size:Optional[int]=None,
                type=Interp.NONE):

    """return a new notch filter (a delay line with a scaler on the feedforward) of size elements.
        If the notch length will be changing at run-time, max-size sets its maximum length"""
    
    initial_contents_ptr = None
    
    if not max_size:
        max_size = size
    
    if initial_contents:
        initial_contents_ptr = get_array_ptr(intial_contents)

    elif initial_element:
        initial_contents = np.zeros(max_size)
        initial_contents.fill(initial_element)
        initial_contents_ptr = get_array_ptr(intial_contents)
    

    gen = mus_make_notch(scaler, size, initial_contents_ptr, max_size, type)
    gen._cache = [initial_contents_ptr]
    return gen    
    
    

def notch(cflt: MUS_ANY_POINTER, input: float, pm: Optional[float]=None):
    """Notch filter val, pm changes the delay length."""
    if cflt == None:
        raise_none_error('notch')
    if pm:
        return mus_notch(cflt, input, pm)
    else:
        return mus_notch_unmodulated(cflt, input)
    
def is_notch(cflt: MUS_ANY_POINTER):
    """Returns True if gen is a notch"""
    return mus_is_notch(cflt)
    
    
# ---------------- all-pass ---------------- #
# TODO: cache if initial contents
def make_all_pass(feedback: float, 
                feedforward: float,
                size: int, 
                initial_contents=None, 
                initial_element: Optional[float]=0.0, 
                max_size:Optional[int]=None,
                type=Interp.NONE):

    """Return a new allpass filter (a delay line with a scalers on both the feedback and the feedforward).
        length will be changing at run-time, max-size sets its maximum length."""

    initial_contents_ptr = None
    
    if not max_size:
        max_size = size
    
    if initial_contents:
        initial_contents_ptr = get_array_ptr(intial_contents)

    elif initial_element:
        initial_contents = np.zeros(max_size)
        initial_contents.fill(initial_element)
        initial_contents_ptr = get_array_ptr(intial_contents)
            

    gen = mus_make_all_pass(feedback,feedforward,  size, initial_contents_ptr, max_size, type)    
    gen._cache = [initial_contents_ptr]
    return gen
    
def all_pass(f: MUS_ANY_POINTER, input: float, pm: Optional[float]=None):
    """All-pass filter val, pm changes the delay length."""
    if f == None:
        raise_none_error('all_pass')
    if pm:
        return mus_all_pass(f, input, pm)
    else:
        return mus_all_pass_unmodulated(f, input)
    
def is_all_pass(f: MUS_ANY_POINTER):
    """Returns True if gen is an all_pass"""
    return mus_is_all_pass(f)
    
# ---------------- all-pass-bank ---------------- #

def make_all_pass_bank(all_passes: list):
    """Return a new all_pass-bank generator."""
    if False in (is_all_pass(v) for v in all_passes):
        raise TypeError(f'allpass list contains at least one element that is not a all_pass.')
    all_passes_array = (POINTER(mus_any) * len(all_passes))()
    all_passes_array[:] = [all_passes[i] for i in range(len(all_passes))]
    gen =  mus_make_all_pass_bank(len(all_passes), all_passes_array)
    gen._cache = [all_passes_array]
    return gen

def all_pass_bank(all_passes: MUS_ANY_POINTER, input: float):
    """Sum an array of all_pass filters."""
    if all_passes == None:
        raise_none_error('all_pass_bank')
    return mus_all_pass_bank(all_passes, input)
    
def is_all_pass_bank(o: MUS_ANY_POINTER):
    """Returns True if gen is an all_pass_bank"""
    return mus_is_all_pass_bank(o)
        
        
# make-one-pole-all-pass size coeff
# one-pole-all-pass f input 
# one-pole-all-pass? f



# ---------------- moving-average ---------------- #
# TODO: cache if initial contents
def make_moving_average(size: int, initial_contents=None, initial_element: Optional[float]=0.0):
    """Return a new moving_average generator. """

    initial_contents_ptr = None
    
    if initial_contents:
        initial_contents_ptr = get_array_ptr(intial_contents)

    elif initial_element:
        initial_contents = np.zeros(max_size)
        initial_contents.fill(initial_element)
        initial_contents_ptr = get_array_ptr(intial_contents)
                        
    gen = mus_make_moving_average(size, initial_contents_ptr)
    gen._cache = [initial_contents_ptr]
    return gen
        

def moving_average(f: MUS_ANY_POINTER, input: float):
    """Moving window average."""
    if f == None:
        raise_none_error('moving_average')
    return mus_moving_average(f, input)
    
def is_moving_average(f: MUS_ANY_POINTER):
    """Returns True if gen is a moving_average"""
    return mus_is_moving_average(f)

# ---------------- moving-max ---------------- #
# TODO: cache if initial contents
def make_moving_max(size: int, 
                initial_contents=None, 
                initial_element: Optional[float]=0.0):
                
    """Return a new moving-max generator."""                
    
    initial_contents_ptr = None
    
    if not max_size:
        max_size = size
    
    if initial_contents:
        initial_contents_ptr = get_array_ptr(intial_contents)

    elif initial_element:
        initial_contents = np.zeros(max_size)
        initial_contents.fill(initial_element)
        initial_contents_ptr = get_array_ptr(intial_contents)
                    
    gen = mus_make_moving_max(size, initial_contents_ptr)
    gen._cache = [initial_contents_ptr]
    return gen
    
    
def moving_max(f: MUS_ANY_POINTER, input: float):
    """Moving window max."""
    if f == None:
        raise_none_error('moving_max')
    return mus_moving_max(f, input)
    
def is_moving_max(f: MUS_ANY_POINTER):
    """Returns True if gen is a moving_max"""
    return mus_is_moving_max(f)
    
# ---------------- moving-norm ---------------- #
def make_moving_norm(size: int,scaler: Optional[float]=1.):

    """Return a new moving-norm generator."""

    initial_contents_ptr = (c_double * size)()
    gen = mus_make_moving_norm(size, initial_contents_ptr, scaler)
    gen._cache = [initial_contents_ptr]
    
def moving_norm(f: MUS_ANY_POINTER, input: float):
    """Moving window norm."""
    if f == None:
        raise_none_error('moving_norm')
    return mus_moving_norm(f, input)
    
def is_moving_norm(f: MUS_ANY_POINTER):
    """Returns True if gen is a moving_norm"""
    return mus_is_moving_norm(f)
    

    
# ---------------- asymmetric-fm ---------------- #


def make_asymmetric_fm(frequency: float, initial_phase: Optional[float]=0.0, r: Optional[float]=1.0, ratio: Optional[float]=1.):
    """Return a new asymmetric_fm generator."""
    return mus_make_asymmetric_fm(frequency, initial_phase, r, ratio)
    
def asymmetric_fm(af: MUS_ANY_POINTER, index: float, fm: Optional[float]=None):
    """Next sample from asymmetric fm generator."""
    if af == None:
        raise_none_error('asymmetric_fm')
    if fm:
        return mus_asymmetric_fm(af, index, fm)
    else:
        return mus_asymmetric_fm_unmodulated(af, index)
    
def is_asymmetric_fm(af: MUS_ANY_POINTER):
    """Returns True if gen is an asymmetric_fm"""
    return mus_is_asymmetric_fm(af)
    
    

    
# ---------------- file-to-sample ---------------- #
def make_file2sample(filename, buffer_size: Optional[int]=None):
    """Return an input generator reading 'filename' (a sound file)"""
    buffer_size = buffer_size or CLM.buffer_size
    return mus_make_file_to_sample_with_buffer_size(filename, buffer_size)
    
def file2sample(obj: MUS_ANY_POINTER, loc: int, chan: Optional[int]=0):
    """sample value in sound file read by 'obj' in channel chan at frample"""
    if obj == None:
        raise_none_error('file2sample')
    return mus_file_to_sample(obj, loc, chan)
    
def is_file2sample(gen: MUS_ANY_POINTER):
    """Returns True if gen is an file2sample"""
    return mus_is_file_to_sample(gen)
    
    
# ---------------- sample-to-file ---------------- #

def make_sample2file(filename, chans: Optional[int]=1, sample_type: Optional[Sample]=None, header_type: Optional[Header]=None, comment: Optional[str]=None):
    """Return an output generator writing the sound file 'filename' which is set up to have chans' channels of 'sample_type' 
        samples with a header of 'header_type'.  The latter should be sndlib identifiers:"""
    sample_type = sample_type or CLM.sample_type
    header_type = header_type or CLM.header_type
    if comment:
        return mus_make_sample_to_file(filename, chans, sample_type, header_type)
    else:
        return mus_make_sample_to_file_with_comment(filename, chans, sample_type, header_type, comment)

def sample2file(obj: MUS_ANY_POINTER, samp: int, chan:int , val: float):
    """add val to the output stream handled by the output generator 'obj', in channel 'chan' at frample 'samp'"""
    if obj == None:
        raise_none_error('sample2file')
    return mus_sample_to_file(obj, samp, chan, val)
    
def is_sample2file(obj: MUS_ANY_POINTER):
    """Returns True if gen is an sample2file"""
    return mus_is_sample_to_file(obj)
    

def continue_sample2file(name):
    return mus_continue_sample_to_file(name)
    

# ---------------- file-to-frample ---------------- #
def make_file2frample(filename, buffer_size: Optional[int]=None):
    """Return an input generator reading 'filename' (a sound file)"""
    buffer_size = buffer_size or CLM.buffer_size
    return mus_make_file_to_frample_with_buffer_size(filename, buffer_size)
    
def file2frample(obj: MUS_ANY_POINTER, loc: int):
    """frample of samples at frample 'samp' in sound file read by 'obj'"""
    if obj == None:
        raise_none_error('file2frample')
    outf = np.zeros(mus_channels(obj), dtype=np.double);
    mus_file_to_frample(obj, loc, outf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return outf
    
def is_file2frample(gen: MUS_ANY_POINTER):
    """Returns True if gen is an file2frample"""
    return mus_is_file_to_frample(gen)
    
    
# ---------------- frample-to-file ---------------- #
def make_frample2file(filename, chans: Optional[int]=1, sample_type: Optional[Sample]=None, header_type: Optional[Header]=None, comment: Optional[str]=None):
    """Return an output generator writing the sound file 'filename' which is set up to have 'chans' channels of 'sample_type' 
        samples with a header of 'header_type'.  The latter should be sndlib identifiers."""
    sample_type = sample_type or CLM.sample_type
    header_type = header_type or CLM.header_type
    if comment:
        return mus_make_frample_to_fileheader_type
    else:
        return mus_make_frample_to_file_with_comment(filename, chans, sample_type, header_type, comment)

def frample2file(obj: MUS_ANY_POINTER,samp: int, chan:int , val):
    """add frample 'val' to the output stream handled by the output generator 'obj' at frample 'samp'"""
    if obj == None:
        raise_none_error('frample2file')
    frample_ptr = get_array_ptr(val)
    mus_sample_to_file(obj, samp, chan, frample_ptr)
    return val
    
def is_frample2file(obj: MUS_ANY_POINTER):
    """Returns True if gen is a frample2file"""
    if obj == None:
        raise_none_error('frample2file')
    return mus_is_sample_to_file(obj)
    

def continue_frample2file(name):
    return mus_continue_sample_to_file(name)
    
    


# TODO : move these
def mus_close(obj):
    return mus_close_file(obj)
    
def mus_is_output(obj):
    """Returns True if gen is a type of output"""
    return mus_is_output(obj)
    
def mus_is_input(obj):
    """Returns True if gen is a type of output"""
    return mus_is_input(obj)


# ---------------- readin ---------------- #

def make_readin(filename: str, chan: int=0, start: int=0, direction: Optional[int]=1, buffer_size: Optional[int]=None):
    """return a new readin (file input) generator reading the sound file 'file' starting at frample 'start' in channel 'channel' and reading forward if 'direction' is not -1"""
    buffer_size = buffer_size or CLM.buffer_size
    return mus_make_readin_with_buffer_size(filename, chan, start, direction, buffer_size)
    
def readin(rd: MUS_ANY_POINTER):
    """next sample from readin generator (a sound file reader)"""
    if rd == None:
        raise_none_error('readin')
    return mus_readin(rd)
    
def is_readin(rd: MUS_ANY_POINTER):
    """Returns True if gen is a readin"""
    return mus_is_readin(rd)
    
    
    
# ---------------- src ---------------- #

def make_src(input, srate: Optional[float]=1.0, width: Optional[int]=10):
    """Return a new sampling-rate conversion generator (using 'warped sinc interpolation'). 'srate' is the ratio between the new rate and the old. 'width' is the sine 
        width (effectively the steepness of the low-pass filter), normally between 10 and 100. 'input' if given is an open file stream."""
    if(isinstance(input, MUS_ANY_POINTER)):
        @INPUTCALLBACK
        def ifunc(gen, inc):
            return mus_apply(input,inc, 0.) # doing this to avoid releasing pointer that is just copy
        
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
    """next sampling rate conversion sample. 'pm' can be used to change the sampling rate on a sample-by-sample basis.  
        'input-function' is a function of one argument (the current input direction, normally ignored) that is called internally 
        whenever a new sample of input data is needed.  If the associated make_src included an 'input' argument, input-function is ignored."""
    if s == None:
        raise_none_error('src')
    return mus_src(s, sr_change, s._inputcallback)
    
def is_src(s: MUS_ANY_POINTER):
    """Returns True if gen is a type of src"""
    return mus_is_src(s)
    
    
# ---------------- convolve ---------------- #

def make_convolve(input, filt: npt.NDArray[np.float64], fft_size: Optional[int]=512, filter_size: Optional[int]=None ):
    """Return a new convolution generator which convolves its input with the impulse response 'filter'."""
    
    filter_ptr = get_array_ptr(filt)

    if not filter_size:
        filter_size = clm_length(filt)
     
    fft_len = 0
    
    if fft_size < 0 or fft_size == 0:
        print('error') #TODO Raise eeror
        
    if fft_size > mus_max_malloc():
         print('error') #TODO Raise eeror
        
    if is_power_of_2(filter_size):
        fft_len = filter_size * 2
    else :
        fft_len = next_power_of_2(filter_size)
        
    if fft_size < fft_len:
        fft_size = fft_len

    if(isinstance(input, MUS_ANY_POINTER)):
        @INPUTCALLBACK
        def ifunc(gen, inc):
            return mus_apply(input, inc, 0.)

        res = mus_make_convolve(ifunc, filter_ptr, fft_size, filter_size, input)
    
    elif isinstance(input, types.FunctionType):
        @INPUTCALLBACK
        def ifunc(gen, inc):
            return input(inc)

        res = mus_make_convolve(ifunc, filter_ptr, fft_size, filter_size, None)
    else:
        print("error") # need error
    
    res._inputcallback = ifunc 
    res._cache = [filter_ptr]
    return res

    
def convolve(gen: MUS_ANY_POINTER):
    """next sample from convolution generator"""
    if gen == None:
        raise_none_error('convolve')   
    return mus_convolve(gen, gen._inputcallback)  
    
def is_convolve(gen: MUS_ANY_POINTER):
    """Returns True if gen is a convolve"""
    return mus_is_convolve(gen)
    
    
# Added some options . TODO: what about sample rate conversion    
def convolve_files(file1, file2, maxamp: Optional[float]=1., outputfile='test.aif', sample_type=CLM.sample_type, header_type=CLM.header_type):
    if header_type == Header.NEXT:
        mus_convolve_files(file1, file2, maxamp,outputfile)
    else:
        temp_file = tempfile.gettempdir() + '/' + 'temp-' + outputfile
        mus_convolve_files(file1, file2, maxamp, temp_file)
        with Sound(outputfile, header_type=header_type, sample_type=sample_type):
            length = seconds2samples(mus_sound_duration(temp_file))
            reader = make_readin(temp_file)
            for i in range(0, length):
                outa(i, readin(reader))
    return outputfile

    
# --------------- granulate ---------------- #

def make_granulate(input, 
                    expansion: Optional[float]=1.0, 
                    length: Optional[float]=.15,
                    scaler: Optional[float]=.6,
                    hop: Optional[float]=.05,
                    ramp: Optional[float]=.4,
                    jitter: Optional[float]=0.0,
                    max_size: Optional[int]=0,
                    edit=None):
    """Return a new granular synthesis generator.  'length' is the grain length (seconds), 'expansion' is the ratio in timing
         between the new and old (expansion > 1.0 slows things down), 'scaler' scales the grains
        to avoid overflows, 'hop' is the spacing (seconds) between successive grains upon output.
        'jitter' controls the randomness in that spacing, 'input' can be a file pointer. 'edit' can
        be a function of one arg, the current granulate generator.  It is called just before
        a grain is added into the output buffer. The current grain is accessible via mus_data.
        The edit function, if any, should return the length in samples of the grain, or 0."""
    
    if edit:
        @EDITCALLBACK    
        def efunc(gen):
            return edit(gen)
        
    if(isinstance(input, MUS_ANY_POINTER)):
        @INPUTCALLBACK
        def ifunc(gen, inc):
            return mus_apply(input,inc, 0.)
        
        res = mus_make_granulate(ifunc, expansion, length, scaler, hop, ramp, jitter, max_size, efunc if edit else cast(None, EDITCALLBACK), input)
    
    elif isinstance(input, types.FunctionType):
        @INPUTCALLBACK
        def ifunc(gen, inc):
            return input(inc)
            
        res = mus_make_granulate(ifunc, expansion, length, scaler, hop, ramp, jitter, max_size, efunc if edit else cast(None, EDITCALLBACK), None)
    else:
        print("error") # TODO: need error
        
    res._inputcallback = ifunc
    res._editcallback  = None
    if edit:
        res._editcallback = efunc
    return res
    
    
#TODO: mus_granulate_grain_max_length
def granulate(e: MUS_ANY_POINTER):
    """next sample from granular synthesis generator"""
    if e == None:
        raise_none_error('granulate')
    if e._editcallback:
        return mus_granulate_with_editor(e, e._inputcallback, e._editcallback)
    else:
        return mus_granulate(e, e._inputcallback)

def is_granulate(e: MUS_ANY_POINTER):
    """Returns True if gen is a granulate"""
    return mus_is_granulate(e)
    
    
    
#--------------- phase-vocoder ----------------#

def make_phase_vocoder(input, 
                        fft_size: Optional[int]=512, 
                        overlap: Optional[int]=4, 
                        interp: Optional[int]=128, 
                        pitch: Optional[float]=1.0, 
                        analyze=None, 
                        edit=None, 
                        synthesize=None):
                        
    """Return a new phase-vocoder generator; input is the input function (it can be set at run-time), analyze, edit,
        and synthesize are either None or functions that replace the default innards of the generator, fft-size, overlap
        and interp set the fftsize, the amount of overlap between ffts, and the time between new analysis calls.
        'analyze', if given, takes 2 args, the generator and the input function; if it returns True, the default analysis 
        code is also called.  'edit', if given, takes 1 arg, the generator; if it returns True, the default edit code 
        is run.  'synthesize' is a function of 1 arg, the generator; it is called to get the current vocoder 
        output."""

    
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
        print("error") # TODO: need error
        
    setattr(res,'_inputcallback', ifunc)        
    setattr(res,'_analyzecallback', afunc if analyze else None)
    setattr(res,'_synthesizecallback', sfunc if analyze else None)
    setattr(res,'_editcallback', efunc if edit else None )
    
    return res

    
def phase_vocoder(pv: MUS_ANY_POINTER):
    """next phase vocoder value"""
    if v == None:
        raise_none_error('phase_vocoder')
    if pv._analyzecallback or pv._synthesizecallback or pv._editcallback :
        return mus_phase_vocoder_with_editors(pv, pv._inputcallback, pv._analyzecallback, pv._editcallback, pv._synthesizecallback)
    else:
        return mus_phase_vocoder(pv, pv._inputcallback)
    
def is_phase_vocoder(pv: MUS_ANY_POINTER):
    """Returns True if gen is a phase_vocoder"""
    return mus_is_phase_vocoder(pv)
    

def phase_vocoder_amps(gen: MUS_ANY_POINTER):
    """Returns a ndarray containing the current output sinusoid amplitudes"""
    if gen == None:
        raise_none_error('phase_vocoder')
    size = mus_length(gen)
    p = np.ctypeslib.as_array(mus_phase_vocoder_amps(gen), shape=size)
    amps = np.copy(p)
    return amps

def phase_vocoder_amp_increments(gen: MUS_ANY_POINTER):
    """Returns a ndarray containing the current output sinusoid amplitude increments per sample"""    
    size = mus_length(gen)
    p = np.ctypeslib.as_array(mus_phase_vocoder_amp_increments(gen), shape=size)
    amp_increments = np.copy(p)
    return amp_increments
    
def phase_vocoder_freqs(gen: MUS_ANY_POINTER):
    """Returns a ndarray containing the current output sinusoid frequencies"""
    size = mus_length(gen)
    p = np.ctypeslib.as_array(mus_phase_vocoder_freqs(gen), shape=size)
    freqs = np.copy(p)
    return freqs
    
def phase_vocoder_phases(gen: MUS_ANY_POINTER):
    """Returns a ndarray containing the current output sinusoid phases"""
    size = mus_length(gen)
    p = np.ctypeslib.as_array(mus_phase_vocoder_phases(gen), shape=size)
    phases = np.copy(p)
    return phase
    
def phase_vocoder_phase_increments(gen: MUS_ANY_POINTER):
    """Returns a ndarray containing the current output sinusoid phase increments"""
    size = mus_length(gen)
    p = np.ctypeslib.as_array(mus_phase_vocoder_phase_increments(gen), shape=size)
    phases_increments = np.copy(p)
    return phases_increments
    
# --------------- out-any ---------------- #
#  TODO : output to array out-any loc data channel (output *output*)    
def out_any(loc: int, data: float, channel,  output=None):
    if output is not None:
        if is_iterable(output):
            output[channel][loc] += data
        else:
            mus_out_any(loc, data, channel, output)        
    else:
        mus_out_any(loc, data, channel, Sound.output)    
        
def outa(loc: int, data: float, output=None):
    if output is not None:
        out_any(loc, data, 0, output)        
    else:
        out_any(loc, data, 0, Sound.output)    
# --------------- outa ---------------- #
def outa(loc: int, data: float, output=None):
    if output is not None:
        out_any(loc, data, 0, output)        
    else:
        out_any(loc, data, 0, Sound.output)    
# --------------- outb ---------------- #    
def outb(loc: int, data: float, output=None):
    if output is not None:
        out_any(loc, data, 1, output)        
    else:
        out_any(loc, data, 1, Sound.output)
# --------------- outc ---------------- #    
def outc(loc: int, data: float, output=None):
    if output is not None:
        out_any(loc, data, 2, output)        
    else:
        out_any(loc, data, 2, Sound.output)        
# --------------- outd ---------------- #    
def outd(loc: int, data: float, output=None):
    if output is not None:
        out_any(loc, data, 3, output)        
    else:
        out_any(loc, data, 3, Sound.output)    
# --------------- out-bank ---------------- #    
def out_bank(gens, loc, input):
    for i in range(len(gens)):
        out_any(loc, mus_apply(gens[i], input, 0.), i)    
    

#--------------- in-any ----------------#
def in_any(loc: int, channel: int, input):
    """input stream sample at frample in channel chan"""
    if is_iterable(input):
        return input[channel][loc]
    elif isinstance(input, types.GeneratorType):
        return next(input)
    elif isinstance(input, types.FunctionType):
        return input(loc, channel)
    else:
        return mus_in_any(loc, channel, input)
#--------------- ina ----------------#
def ina(loc: int, input):
    return in_any(loc, 0, input)
#--------------- inb ----------------#    
def inb(loc: int, input):
    return in_any(loc, 1, input)






# --------------- locsig ---------------- #

# outn, revn 
# 
# outf 
# 
# revf



def make_locsig(degree: Optional[float]=0.0, 
    distance: Optional[float]=1., 
    reverb: Optional[float]=0.0, 
    output: Optional[MUS_ANY_POINTER]=None, 
    revout: Optional[MUS_ANY_POINTER]=None, 
    channels: Optional[int]=None, 
    reverb_channels: Optional[int]=None,
    type: Optional[Interp]=Interp.LINEAR):
    
    """Return a new generator for signal placement in n channels.  Channel 0 corresponds to 0 degrees."""
    
    if not output:
        output = Sound.output  #TODO : check if this exists
    
    if not revout:
        if Sound.reverb:
            revout = Sound.reverb
        else: 
            revout = cast(None, MUS_ANY_POINTER)

    if not channels:
        channels = clm_channels(output)
    
    if not reverb_channels:
        reverb_channels = clm_channels(revout)
        #print(reverb_channels)
        
   # print(np.shape(output))
    
    # TODO: What if revout is not an iterable? While possible not going to deal with it right now :)   
    if is_iterable(output):
        if not reverb_channels:
            reverb_channels = 0
            

        res = mus_make_locsig(degree, distance, reverb, channels, None, reverb_channels, None, type)
        
                
        @LOCSIGDETOURCALLBACK
        def locsig_to_array(gen, loc):
            outf = mus_locsig_outf(res)
            revf = mus_locsig_revf(res)
            for i in range(channels):
                 Sound.output[i][loc] += outf[i] # += correct?
            for i in range(reverb_channels):
                 Sound.reverb[i][loc] += revf[i]  # 

        mus_locsig_set_detour(res, locsig_to_array)

        setattr(res,'_locsigdetour', locsig_to_array)   

        return res
            
    else:
        return mus_make_locsig(degree, distance, reverb, channels, output, reverb_channels, revout,  type)
        
def locsig(gen: MUS_ANY_POINTER, loc: int, val: float):
    """locsig 'gen' channel 'chan' scaler"""
    if gen == None:
        raise_none_error('locsig')
    mus_locsig(gen, loc, val)
    
def is_locsig(gen: MUS_ANY_POINTER):
    """Returns True if gen is a locsig"""
    return mus_is_locsig(gen)
    
def locsig_ref(gen: MUS_ANY_POINTER, chan: int):
    """locsig 'gen' channel 'chan' scaler"""
    return mus_locsig_ref(gen, chan)
    
def locsig_set(gen: MUS_ANY_POINTER, chan: int, val:float):
    """set the locsig generator's channel 'chan' scaler to 'val'"""
    return mus_locsig_set(gen, chan, val)
    
def locsig_reverb_ref(gen: MUS_ANY_POINTER, chan: int):
    """ locsig reverb channel 'chan' scaler"""
    return mus_locsig_reverb_ref(gen, chan)

def locsig_reverb_set(gen: MUS_ANY_POINTER, chan: int, val: float):
    """set the locsig reverb channel 'chan' scaler to 'val"""
    return mus_locsig_reverb_set(gen, chan, val)
    
def move_locsig(gen: MUS_ANY_POINTER, degree: float, distance: float):
    """move locsig gen to reflect degree and distance"""
    mus_move_locsig(gen, degree, distance)
    

# locsig-type ()    

# TODO: move-sound  need dlocsig




def calc_length(start, dur):
    st = seconds2samples(start)
    nd = seconds2samples(start+dur)
    return st, nd

# attempting some kind of defgenerator


def convert_frequency(gen):
    gen.frequency = hz2radians(gen.frequency)
    return gen

def make_generator(name, slots, wrapper=None, methods=None):
    class mus_gen():
        pass
    def make_generator(**kwargs):
        gen = mus_gen()
        setattr(gen, 'name', name)
        for k, v  in kwargs.items():
            setattr(gen, k, v)
        if methods:
            for k, v  in methods.items():
                setattr(mus_gen, k, property(v[0], v[1], None) )
        
        return gen if not wrapper else wrapper(gen)
    def is_a(gen):
        return isinstance(gen, mus_gen) and gen.name == name
    return functools.partial(make_generator, **slots), is_a


def array_reader(arr, chan, loop=0):
    ind = 0
    if chan > (clm_channels(arr)):
        raise ValueError(f'array has {clm_channels(arr)} channels but {chan} asked for')
    length = clm_length(arr)
    if loop:
        def reader(direction):
            nonlocal ind
            v = arr[chan][ind]
            ind += direction
            ind = wrap(ind, 0, length-1)
            return v
            
    else: 
        def reader(direction):
            nonlocal ind
            v = arr[chan][ind]
            ind += direction
            ind = clip(ind, 0, length-1)
            return v
    return reader    
    
# musx integration
# TODO move to another file

#returns 
def clm_instrument(func):
    @functools.wraps(func)
    def call(time, *args, **kwargs):
        obj = functools.partial(func,time, *args, **kwargs)
        obj.time = time
        return obj
    return call 
    
def render_clm(seq: musx.seq, filename, **kwargs):
    s = seq.events.copy()
    with Sound(filename, **kwargs):
        for event in s:
            event()

# what should the clm score file be? 
# TODO make this something useful that could be read.  i am assuming some type of dictionary
# with header for info and then entries in seq
def write_clm(seq):
    s = seq.events.copy()
    for v in s:
        funcname = v.func.__name__
        args = ("{},"*(len(v.args))).format(*v.args)
        kwargs = ','.join([f'{k}={v}' for k,v in v.keywords.items()])
        #print(args)
        print(f'({funcname} {args}{kwargs})') #TODO: write to actual file