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

mus_initialize()
mus_sound_initialize() 
mus_set_rand_seed(int(time.time()))

class SNDLIBError(Exception):
    """This is general class to raise an print errors as defined in sndlib. It is to be used internally by the 
        defined error handler registered with sndlib
    """
    def ___init___(self, message):
        self.message = message + mus_error_to_string(error_type)
        super().__init__(self.message)
        
# CFUNCTYPE(UNCHECKED(None), c_int, String)# sndlib.h: 158
# from sndlib.py
@mus_error_handler_t
def clm_error_handler(error_type: int, msg: String):
    message =  msg + ". "  +  mus_error_type_to_string(error_type)
    raise SNDLIBError(message) 

mus_error_set_handler(clm_error_handler)



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
    
def file2array(filename: str, channel: Optional[int]=0, beg: Optional[int]=None, dur: Optional[int]=None):
    """Return an ndarray with samples from file"""
    length = dur or mus_sound_framples(filename)
    chans = mus_sound_chans(filename)
    srate = mus_sound_srate(filename)
    bg = beg or 0
    out = np.zeros(length, dtype=np.double)
        
    mus_file_to_array(filename,channel, bg, length, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return out
    
    

def channel2array(filename: str, channel: Optional[int]=0, beg: Optional[int]=None, dur: Optional[int]=None): 
    length = dur or mus_sound_framples(filename)
    srate = mus_sound_srate(filename)
    bg = beg or 0
    out = np.zeros((1, length), dtype=np.double)
    mus_file_to_array(filename,channel, bg, length, out[0].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return out

    

    
    





    

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
    
