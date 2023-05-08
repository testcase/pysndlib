from contextlib import contextmanager
import functools
from functools import singledispatch
import os
from typing import Optional
import tempfile
import time
import types
import math
import numpy as np
import numpy.typing as npt

from .internals.sndlib import *
from .internals.enums import *
from .internals.clm import *
from .internals.mus_any_pointer import *
from .internals.utils import *

mus_initialize()
mus_sound_initialize() 
mus_set_rand_seed(int(time.time()))

# these could be singledispatch but is manageable this way for now
### some utilites as generic functions


# TODO: what other clm functions need. 
#  using clm_* to avoid clashing with 
# what are likely typically used variable names
# 

# --------------- generic functions ---------------- #
# prepending clm to functions to avoid name classhes


# --------------- clm_channels ---------------- #
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
        
# --------------- clm_length ---------------- #
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
  


# --------------- range utils ---------------- #

# --------------- clamp ---------------- #
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

# --------------- clip ---------------- #
# same as clamp
@clip.register
def _(x: float, lo, hi):
    return float(max(min(x, hi),lo))
    
@clip.register
def _(x: int, lo, hi):
    return int(max(min(x, hi),lo))

# --------------- fold ---------------- #

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


# --------------- wrap ---------------- #

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
    
        loop_info = {'sustain_start' : loop_starts[0], 'sustain_end' : loop_ends[0], 
                    'release_start' : loop_starts[2], 'release_end' : loop_ends[1],
                    'base_note' : base_note, 'base_detune' : base_detune, 
                    'sustain_mode' : loop_modes[0], 'release_mode' : loop_modes[1]}
    
    
    info = {'date' : time.localtime(date), 'srate' : srate, 'chans' : chans, 'samples' : samples,
            'comment' : comment, 'length' : length, 'header_type' : header_type, 'sample_type' : sample_type,
            'loop_info' : loop_info}
            
    
    return info

def sound_loop_info(filename):
    """Returns a dictionary of info about a sound file including write date (data), sample rate (srate),
    channels (chans), length in samples (samples), length in second (length), comment (comment), and loop information (loopinfo)"""
    
    loop_info = mus_sound_loop_info(filename)
    if loop_info:
        loop_modes = [loop_info[6], loop_info[7]]
        loop_starts = [loop_info[0], loop_info[2]]
        loop_ends = [loop_info[1], loop_info[3]]
        base_note = loop_info[4]
        base_detune = loop_info[5]
    
        loop_info = {'sustain_start' : loop_starts[0], 'sustain_end' : loop_ends[0], 
                    'release_start' : loop_starts[2], 'release_end' : loop_ends[1],
                    'base_note' : base_note, 'base_detune' : base_detune, 
                    'sustain_mode' : loop_modes[0], 'release_mode' : loop_modes[1]}
    
    
    return info
    
    
# slightly different than clm version can read multiple channels 
def file2array(filename: str, channel: Optional[int]=None, beg: Optional[int]=None, dur: Optional[int]=None):
    """Return an ndarray with samples from file and the sample rate of the data"""
    length = dur or mus_sound_framples(filename)
    chans = mus_sound_chans(filename)
    srate = mus_sound_srate(filename)
    bg = beg or 0
    out = np.zeros((1 if channel else chans, length), dtype=np.double)
        
    if not channel:
        # read in all channels
        for i in range(chans):
            mus_file_to_array(filename,i, bg, length, out[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    else:
        mus_file_to_array(filename,i, bg, length, out[0].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return out, srate


def channel2array(filename: str, channel: Optional[int]=0, beg: Optional[int]=None, dur: Optional[int]=None): 
    length = dur or mus_sound_framples(filename)
    srate = mus_sound_srate(filename)
    bg = beg or 0
    out = np.zeros((1, length), dtype=np.double)
    mus_file_to_array(filename,channel, bg, length, out[0].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return out, srate

    
def array2file(filename: str, arr, length=None, sr=None, #channels=None, 
    sample_type=CLM.sample_type, header_type=CLM.header_type, comment=None ):
    """Write an ndarray of samples to file"""
    if not sr:
        sr = CLM.srate
    
    chans = np.shape(arr)[0]
    length = length or np.shape(arr)[1]
    fd = mus_sound_open_output(filename, int(sr), chans, sample_type, header_type, comment)
 
    obuftype = POINTER(c_double) * chans
    obuf = obuftype()
    
    for i in range(chans):
        obuf[i] = arr[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    err = mus_sound_write(fd, 0, length, chans, obuf)
    
    mus_sound_close_output(fd, length*mus_bytes_per_sample(sample_type)*chans)




    

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

# TODO: maybe add an exception that have to use keyword args
def make_generator(name, slots, wrapper=None, methods=None, docstring=None):
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
    g =  functools.partial(make_generator, **slots)
    if docstring:
        g.__doc__ = docstring
    
    return g, is_a


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
    
