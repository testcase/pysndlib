# cython: c_string_type=unicode
# cython: c_string_encoding=utf8
import ctypes
from functools import singledispatch
import math
import random
import subprocess
import tempfile
import time
import types

import cython
from cython.cimports.cpython.mem import PyMem_Malloc, PyMem_Realloc, PyMem_Free
import numpy as np
cimport numpy as np
import numpy.typing as npt
cimport pysndlib.cclm as cclm
cimport pysndlib.csndlib as csndlib
from pysndlib.sndlib import Sample, Header




# --------------- clm enums ---------------- #
cpdef enum Interp:
    NONE, LINEAR,SINUSOIDAL, ALL_PASS, LAGRANGE, BEZIER, HERMITE
    
cpdef enum Window:
    RECTANGULAR, HANN, WELCH, PARZEN, BARTLETT, HAMMING, BLACKMAN2, BLACKMAN3, BLACKMAN4, EXPONENTIAL, RIEMANN, KAISER, CAUCHY, POISSON, GAUSSIAN, TUKEY, DOLPH_CHEBYSHEV, HANN_POISSON, CONNES, SAMARAKI, ULTRASPHERICAL, BARTLETT_HANN, BOHMAN, FLAT_TOP, BLACKMAN5, BLACKMAN6, BLACKMAN7, BLACKMAN8, BLACKMAN9, BLACKMAN10, RV2, RV3, RV4, MLT_SINE, PAPOULIS, DPSS, SINC,

cpdef enum Spectrum:
    IN_DB, NORMALIZED, RAW
    
cpdef enum Polynomial:
    EITHER_KIND, FIRST_KIND, SECOND_KIND, BOTH_KINDS



# --------------- function types for ctypes ---------------- #
INPUTCALLBACK = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p, ctypes.c_int)
EDITCALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
ANALYZECALLBACK = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p, ctypes.c_int))
SYNTHESIZECALLBACK = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p)
LOCSIGDETOURCALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_longlong) 
ENVFUNCTION = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double) # for env_any    

#these may need to be set based on system type etc
DEFAULT_OUTPUT_SRATE = 44100
DEFAULT_OUTPUT_CHANS = 1
DEFAULT_OUTPUT_SAMPLE_TYPE = Sample.BFLOAT
DEFAULT_OUTPUT_HEADER_TYPE = Header.AIFC



# --------------- main clm prefs ---------------- #

CLM  = types.SimpleNamespace(
    file_name = 'test.aiff',
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


# --------------- initializations ---------------- #

cclm.mus_initialize()
cclm.mus_set_rand_seed(int(time.time()))


cdef void clm_error_handler(int error_type, char* msg):
    message =  msg + ". "  +  csndlib.mus_error_type_to_string(error_type)
    raise SNDLIBError(message) 
    
class SNDLIBError(Exception):
    """This is general class to raise an print errors as defined in sndlib. It is to be used internally by the 
        defined error handler registered with sndlib
    """
    def ___init___(self, message):
        #self.message = message + csndlib.mus_error_type_to_string(error_type)
        super().__init__(self.message)
        

csndlib.mus_error_set_handler(<csndlib.mus_error_handler_t *>clm_error_handler)
    
    

# --------------- extension types ---------------- #

cdef class mus_any:
    """A wrapper class for mus_any pointers"""
    cdef cclm.mus_any *_ptr
    cdef bint ptr_owner
    cdef cclm.input_cb _inputcallback
    cdef cclm.edit_cb _editcallback
    cdef cclm.analyze_cb _analyzecallback
    cdef cclm.synthesize_cb _synthesizecallback
    _cache: list
    
    def __cinit__(self):
        self.ptr_owner = False
        self._cache = []
        self._inputcallback = NULL
        self._editcallback = NULL
        self._analyzecallback = NULL
        self._synthesizecallback = NULL

        
    def __delalloc__(self):
        if self._ptr is not NULL and self.ptr_owner is True:
            cclm.mus_free(self._ptr)
            self._ptr = NULL

    def __init__(self):
        # Prevent accidental instantiation from normal Python code
        # since we cannot pass a struct pointer into a Python constructor.
        raise TypeError("This class cannot be instantiated directly.")

    @staticmethod
    cdef mus_any from_ptr(cclm.mus_any *_ptr, bint owner=True):
        """Factory function to create mus_any objects from
        given mus_any pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated."""
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef mus_any wrapper = mus_any.__new__(mus_any)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        return wrapper
       
    cdef cache_append(self, obj):
        self._cache.append(obj) 

    cdef cache_extend(self, obj):
        self._cache.append(obj) 

    def __str__(self):
        return f'{mus_any} {cclm.mus_describe(self._ptr)}'
        
    @property
    def mus_frequency(self):
        return cclm.mus_frequency(self._ptr)
    
    @mus_frequency.setter
    def mus_frequency(self, v):
        cclm.mus_set_frequency(self._ptr, v)
    
    @property
    def mus_phase(self):
        return cclm.mus_phase(self._ptr)
    
    @mus_phase.setter
    def mus_phase(self, v):
        cclm.mus_set_phase(self._ptr, v)
        
    @property
    def mus_length(self):
        return cclm.mus_length(self._ptr)
    
    @mus_length.setter
    def mus_length(self, v):
        cclm.mus_set_length(self._ptr, v)  
        
    @property
    def mus_increment(self):
        return cclm.mus_increment(self._ptr)
    
    @mus_increment.setter
    def mus_increment(self, v):
        cclm.mus_set_increment(self._ptr, v)  
        
    @property
    def mus_location(self):
        return cclm.mus_location(self._ptr)
    
    @mus_location.setter
    def mus_location(self, v):
        cclm.mus_set_location(self._ptr, v)  
    
    @property
    def mus_offset(self):
        return cclm.mus_offset(self._ptr)
    
    @mus_offset.setter
    def mus_offset(self, v):
        cclm.mus_set_offset(self._ptr, v)  
        
    @property
    def mus_channels(self):
        return cclm.mus_channels(self._ptr)
    
    @mus_channels.setter
    def mus_channels(self, v):
        cclm.mus_set_channels(self._ptr, v)  
        
    @property
    def mus_interp_type(self):
        return cclm.mus_interp_type(self._ptr)
    
    @property
    def mus_width(self):
        return cclm.mus_width(self._ptr)
    
    @mus_width.setter
    def mus_width(self, v):
        cclm.mus_set_width(self._ptr, v)  
        
    @property
    def mus_order(self):
        return cclm.mus_order(self._ptr)

    @property
    def mus_scaler(self):
        return cclm.mus_scaler(self._ptr)
    
    @mus_scaler.setter
    def mus_scaler(self, v):
        cclm.mus_set_scaler(self._ptr, v)  
        
    @property
    def mus_feedback(self):
        return cclm.mus_feedback(self._ptr)
    
    @mus_feedback.setter
    def mus_feedback(self, v):
        cclm.mus_set_feedback(self._ptr, v)  
        
    @property
    def mus_feedforward(self):
        return cclm.mus_feedforward(self._ptr)
    
    @mus_feedforward.setter
    def mus_feedforward(self, v):
        cclm.mus_set_feedforward(self._ptr, v) 

    @property
    def mus_hop(self):
        return cclm.mus_hop(self._ptr)
    
    @mus_hop.setter
    def mus_hop(self, v):
        cclm.mus_set_hop(self._ptr, v) 
        
    @property
    def mus_ramp(self):
        return cclm.mus_ramp(self._ptr)
    
    @mus_ramp.setter
    def mus_ramp(self, v):
        cclm.mus_set_ramp(self._ptr, v) 
    
    @property
    def mus_channels(self):
        return cclm.mus_channel(self._ptr)
        
    def get_mus_data(gen: mus_any):
        size = cclm.mus_length(gen._ptr)
        cdef cclm.mus_float_t* ptr = cclm.mus_data(gen._ptr)
        p = np.asarray(<np.float64_t [:size]> ptr)
        return p


    def set_mus_data(gen: mus_any, data):
        size = cclm.mus_length(gen._ptr)
        data_ptr = mus_float_array.from_this(data)
        cdef cclm.mus_float_t* ptr = cclm.mus_set_data(gen._ptr, <cclm.mus_float_t*>data_ptr.data)
        p = np.asarray(<np.float64_t [:size]> ptr)
        return p
    
    def get_mus_xcoeffs(gen: mus_any):
        size = cclm.mus_length(gen._ptr)
        cdef cclm.mus_float_t* ptr = cclm.mus_xcoeffs(gen._ptr)
        p = np.asarray(<np.float64_t [:size]> ptr)
        return p
    
    def get_mus_ycoeffs(gen: mus_any):
        size = cclm.mus_length(gen._ptr)
        cdef cclm.mus_float_t* ptr = cclm.mus_ycoeffs(gen._ptr)
        p = np.asarray(<np.float64_t [:size]> ptr)
        return p
    
   
   
cdef class mus_any_array:
    """A wrapper class arrays of mus_any objects"""
    cdef  cclm.mus_any_ptr_ptr data
    item_count: cython.size_t
    def __cinit__(self, number: cython.size_t):
        # allocate some memory (uninitialised, may contain arbitrary data)
        if number > 0:
            self.data = cython.cast(cclm.mus_any_ptr_ptr, PyMem_Malloc(
                number * cython.sizeof(cclm.mus_any_ptr)))
            if not self.data:
                raise MemoryError()
        self.item_count = number


    def __dealloc__(self):
        PyMem_Free(self.data)  # no-op if self.data is NULL
    
    def __len__(self):
        return self.item_count
        
    def __str__(self):
        return f"count {self.item_count}"

    @staticmethod
    cdef mus_any_array from_pylist(lst: list):
        """Factory function to create a wrapper around mus_float_t (double) and fill it with values from the list"""
        cdef int i
        
        cdef mus_any_array wrapper = mus_any_array(len(lst))
        for i in range(len(lst)):
            wrapper.data[i] = (<mus_any>lst[i])._ptr
        return wrapper
        

cdef class mus_float_array:
    """A wrapper class arrays mus_float_t (double)"""
    data: cython.p_double
    item_count: cython.size_t
    def __cinit__(self, number: cython.size_t):
        # allocate some memory (uninitialised, may contain arbitrary data)
        if number > 0:
            self.data = cython.cast(cython.p_double, PyMem_Malloc(
                number * cython.sizeof(cython.double)))
            if not self.data:
                raise MemoryError()
        self.item_count = number

    def __dealloc__(self):
        PyMem_Free(self.data)  # no-op if self.data is NULL
    
    def __len__(self):
        return self.item_count
        
    def __str__(self):
        p = []
        for i in range(self.item_count):
            p.append(self.data[i])
        return f"{p} count {self.item_count}"

    @staticmethod
    cdef mus_float_array from_pylist(lst: list):
        """Factory function to create a wrapper around mus_float_t (double) and fill it with values from the list"""
        cdef int i
        cdef mus_float_array wrapper = mus_float_array(len(lst))
        for i in range(len(lst)):
            wrapper.data[i] = lst[i]
        return wrapper
    
    @staticmethod
    cdef mus_float_array from_ndarray(np.ndarray[double, ndim=1, mode="c"] np_array):
        """Factory function to create a wrapper around np.ndarray and fill it with values from the list"""
        cdef mus_float_array wrapper = mus_float_array(0)
        wrapper.data = <double*> np_array.data
        wrapper.item_count = len(np_array)
        return wrapper
        
    @staticmethod
    cdef mus_float_array from_this(thing):
        """Factory function to create a wrapper around np.ndarray or python list"""
        if isinstance(thing, list):
            return mus_float_array.from_pylist(thing)
        if isinstance(thing, np.ndarray): 
            return mus_float_array.from_ndarray(thing)
        else:
            raise TypeError(f'{type(thing)} cannot be converted to a mus_float_array')
            
            
cdef class mus_long_array:
    """A wrapper class arrays mus_long_t (long long)"""
    data: cython.p_longlong
    item_count: cython.size_t
    def __cinit__(self, number: cython.size_t):
        # allocate some memory (uninitialised, may contain arbitrary data)
        if number > 0:
            self.data = cython.cast(cython.p_longlong, PyMem_Malloc(
                number * cython.sizeof(cython.p_longlong)))
            if not self.data:
                raise MemoryError()
        self.item_count = number

    def __dealloc__(self):
        PyMem_Free(self.data)  # no-op if self.data is NULL
    
    def __len__(self):
        return self.item_count
        
    def __str__(self):
        p = []
        for i in range(self.item_count):
            p.append(self.data[i])
        return f"{p} count {self.item_count}"

    @staticmethod
    cdef mus_long_array from_pylist(lst: list):
        """Factory function to create a wrapper around mus_long_t (long long) and fill it with values from the list"""
        cdef int i
        cdef mus_long_array wrapper = mus_long_array(len(lst))
        for i in range(len(lst)):
            wrapper.data[i] = lst[i]
        return wrapper
    
    @staticmethod
    cdef mus_long_array from_ndarray(np.ndarray[longlong, ndim=1, mode="c"] np_array):
        """Factory function to create a wrapper around np.ndarray and fill it with values from the list"""
        cdef mus_long_array wrapper = mus_long_array(0)
        wrapper.data = <long long*> np_array.data
        wrapper.item_count = len(np_array)
        return wrapper
        
    @staticmethod
    cdef mus_long_array from_this(thing):
        """Factory function to create a wrapper around np.ndarray or python list"""
        if isinstance(thing, list):
            return mus_long_array.from_pylist(thing)
        if isinstance(thing, np.ndarray): 
            return mus_long_array.from_ndarray(thing)
        else:
            raise TypeError(f'{type(thing)} cannot be converted to a mus_long_array')


# --------------- callbacks ---------------- #
        
cdef double input_callback_func(void* arg, int direction):
    cdef cclm.mus_any_ptr gen = <cclm.mus_any_ptr>arg
    return cclm.mus_apply(gen, direction, 0.)        
    

cdef void locsig_detour_callback_func(cclm.mus_any *ptr, cclm.mus_long_t val):
    cdef int channels = cclm.mus_locsig_channels(ptr)
    cdef int reverb_channels = cclm.mus_locsig_reverb_channels(ptr)
    cdef cclm.mus_float_t* outf = cclm.mus_locsig_outf(ptr)
    cdef cclm.mus_float_t* revf = cclm.mus_locsig_revf(ptr)
    for i in range(channels):
         CLM.output[i][val] += outf[i] # += correct?
    for i in range(reverb_channels):
         CLM.reverb[i][val] += revf[i]  # 
         
         
# --------------- file2ndarray, array2file ---------------- #                 
cpdef file2ndarray(filename: str, channel: Optional[int]=None, beg: Optional[int]=None, dur: Optional[int]=None):
    """Return an ndarray with samples from file and the sample rate of the data"""
    length = dur or csndlib.mus_sound_framples(filename)
    chans = csndlib.mus_sound_chans(filename)
    srate = csndlib.mus_sound_srate(filename)
    bg = beg or 0
    out = np.zeros((1 if (channel != None) else chans, length), dtype=np.double)
        
    if not channel:
        # read in all channels
        for i in range(chans):
            arr_ptr = mus_float_array.from_ndarray(out[i])
            csndlib.mus_file_to_array(filename, i, bg, length, arr_ptr.data)
    else:
        arr_ptr = mus_float_array.from_ndarray(out[0])
        csndlib.mus_file_to_array(filename,0, bg, length, arr_ptr.data)
    return out, srate
    
cpdef array2file(filename: str, arr, length=None, sr=None, #channels=None, 
    sample_type=CLM.sample_type, header_type=CLM.header_type, comment=None ):
    """Write an ndarray of samples to file"""
    
    if not sr:
        sr = CLM.srate

    chans = np.shape(arr)[0]
    length = length or np.shape(arr)[1]
    fd = csndlib.mus_sound_open_output(filename, int(sr), chans, sample_type, header_type, comment)

    cdef cclm.mus_float_t **obuf = <cclm.mus_float_t**>PyMem_Malloc(chans * sizeof(cclm.mus_float_t*))

    if not obuf:
        raise MemoryError()
    
    cdef double [:] arr_view = arr[0]
    
    try:
        for i in range(chans):
            arr_view = arr[i]
            obuf[i] = &arr_view[0]
    
    finally:
        err = csndlib.mus_file_write(fd, 0, length, chans, obuf)
        csndlib.mus_sound_close_output(fd, length * csndlib.mus_bytes_per_sample(sample_type)*chans)
        PyMem_Free(obuf)
        
    return length
         
# --------------- with Sound context manager ---------------- #      
class Sound(object):

    def __init__(self, output=None, 
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
                        finalize = None,
                        ignore_output = False):
        self.output = output if output is not None else CLM.file_name
        self.channels = channels if channels is not None else CLM.channels
        self.srate = srate if srate is not None else CLM.srate
        self.sample_type = sample_type if sample_type is not None else CLM.sample_type
        self.header_type = header_type if header_type is not None else  CLM.header_type
        self.comment = comment
        self.verbose = verbose if verbose is not None else CLM.verbose
        self.reverb = reverb
        self.revfile = revfile if revfile is not None else CLM.reverb_file_name
        self.reverb_data = reverb_data if reverb_data is not None else CLM.reverb_data
        self.reverb_channels = reverb_channels if reverb_channels is not None else CLM.reverb_channels
        self.continue_old_file = continue_old_file
        self.statistics = statistics if statistics is not None else CLM.statistics
        self.scaled_to = scaled_to
        self.scaled_by = scaled_by
        self.play = play if play is not None else CLM.play
        self.clipped = clipped if clipped is not None else CLM.clipped
        self.ignore_output = ignore_output
        self.output_to_file = isinstance(self.output, str)
        self.reverb_to_file = self.reverb is not None and isinstance(self.output, str)
        self.old_srate = get_srate()
        self.finalize = finalize
        
    def __enter__(self):
                
        if not self.clipped:
            if (self.scaled_by or self.scaled_to) and (self.sample_type in [Sample.BFLOAT, Sample.LFLOAT, Sample.BDOUBLE, Sample.LDOUBLE]):
                csndlib.mus_set_clipping(False)
            else:
                csndlib.mus_set_clipping(CLM.clipped)
        else:
            csndlib.mus_set_clipping(self.clipped)
        
        set_srate(self.srate)

        # in original why use reverb-1?
        if  self.statistics :
            self.tic = time.perf_counter()
        
        if self.output_to_file :
            if self.continue_old_file:
                CLM.output = continue_sample2file(self.filename)
                set_srate(csndlib.mus_sound_srate(self.filename)) # maybe print warning or at least note 
            else:
                CLM.output = make_sample2file(self.output,self.channels, sample_type=self.sample_type , header_type=self.header_type)
        elif is_list_or_ndarray(self.output):
            CLM.output = self.output
        else:
            raise TypeError(f"Writing to  {type(self.output)} not supported")
            
        if self.reverb_to_file:
            if self.continue_old_file:
                CLM.reverb = continue_sample2file(self.revfile)
            else:
                CLM.reverb = make_sample2file(self.revfile,self.reverb_channels, sample_type=self.sample_type , header_type=self.header_type)
        
        if self.reverb and not self.reverb_to_file and is_list_or_ndarray(self.output):
            CLM.reverb = np.zeros((self.reverb_channels, np.shape(CLM.output)[1]), dtype=CLM.output.dtype)
    
        return self
        
    def __exit__(self, *args):

        if self.reverb: 
            if self.reverb_to_file:
                mus_close(CLM.reverb)
                CLM.reverb = make_file2sample(self.revfile)
                
                if self.reverb_data:
                    self.reverb(**self.reverb_data)
                else:
                    self.reverb()
                mus_close(CLM.reverb)

            if is_list_or_ndarray(CLM.reverb):
                if self.reverb_data:
                    self.reverb(**self.reverb_data)
                else:
                    self.reverb()          

        if self.output_to_file:
            mus_close(CLM.output)
            
    
        if  self.statistics :
            toc = time.perf_counter()
            
            statstr = ''
            if isinstance(self.output, str):
                statstr = f"{self.output}: "
            if self.output_to_file:
                    chans = clm_channels(self.output)
                    vals = np.zeros(chans, dtype=np.double)
                    times = np.zeros(chans, dtype=np.int_)
                    vals_arr = mus_float_array.from_ndarray(vals)
                    times_arr = mus_long_array.from_ndarray(times)
                    maxamp = csndlib.mus_sound_maxamps(self.output, chans, vals_arr.data, times_arr.data)
                    statstr += f": maxamp: {vals} {times} "
            else:
                chans = clm_channels(self.output)
                vals = np.zeros(chans, dtype=np.double)
                times = np.zeros(chans, dtype=np.int_)
                for i in range(chans):
                    mabs = np.abs(self.output[i])
                    vals[i] = np.amax(mabs)
                    times[i] = np.argmax(mabs)
                statstr += f"maxamp: {vals} {times} "
                
              
            if self.scaled_by or self.scaled_to:
                statstr += "(before scaling) "
            
            statstr += f"compute time: {toc - self.tic:0.8f} seconds. "
            
            
            if self.reverb_to_file:
                    chans = clm_channels(self.revfile)
                    vals = np.zeros(chans, dtype=np.double)
                    times = np.zeros(chans, dtype=np.int_)
                    vals_arr = mus_float_array.from_ndarray(vals)
                    times_arr = mus_long_array.from_ndarray(times)
                    maxamp = csndlib.mus_sound_maxamps(self.revfile, chans, vals_arr.data, times_arr.data)
                    statstr += f"revmax: {vals} {times}"
            elif self.reverb and not self.reverb_to_file and is_list_or_ndarray(self.output):
                chans = clm_channels(CLM.reverb)
                
                vals = np.zeros(chans, dtype=np.double)
                times = np.zeros(chans, dtype=np.int_)
                for i in range(chans):
                    mabs = np.abs(CLM.reverb[i])
                    vals[i] = np.amax(mabs)
                    times[i] = np.argmax(mabs)
                statstr += f"revmax: {vals} {times}"
            
            print(statstr)
         
        if self.scaled_to:
            if self.output_to_file:
                arr, _ = file2ndarray(self.output)
                arr *= (self.scaled_to / np.max(np.abs(arr)))
                # handle error
                array2file(self.output, arr)
            else:
                self.output *= (self.scaled_to / np.max(np.abs(self.output)))
        elif self.scaled_by:
            if self.output_to_file:
                arr, _ = file2ndarray(self.output)
                arr *= self.scaled_by
                array2file(self.output, arr)
            else:
                self.output *= self.scaled_by
        if self.play and self.output_to_file:
            subprocess.run([CLM.player,self.output])
        # need some safety if errors
        
        set_srate(self.old_srate)
        
        if self.finalize:
            self.finalize(self.output)
         

cpdef is_list_or_ndarray(x):
    return isinstance(x, list) or isinstance(x, np.ndarray)
    
cpdef is_power_of_2(x):
    return (((x) - 1) & (x)) == 0

cpdef next_power_of_2(x):
    return 2**int(1 + (float(math.log(x + 1)) / math.log(2.0)))
    
    
    
    
# --------------- generic functions ---------------- #

cpdef int mus_close(obj: mus_any):
    return cclm.mus_close_file(obj._ptr)
    
cpdef bint mus_is_output(obj: mus_any):
    """Returns True if gen is a type of output"""
    return cclm.mus_is_output(obj._ptr)
    
cpdef bint mus_is_input(obj: mus_any):
    """Returns True if gen is a type of output"""
    return cclm.mus_is_input(obj._ptr)


# prepending clm to functions to avoid name clashes
    
# --------------- clm_length ---------------- #
@singledispatch
def clm_length(x):
    pass
    
@clm_length.register
def _(x: str):# assume file
    return csndlib.mus_sound_length(x)

@clm_length.register
def _(x: mus_any):# assume gen
    return x.mus_length
        
@clm_length.register
def _(x: list):
    if isinstance(x[0], list):
        return len(x[0])
    else:
        return len(x)

@clm_length.register
def _(x: np.ndarray):
    if x.ndim == 1:
        return np.shape(x)[0]
    elif x.ndim == 2:
        return np.shape(x)[1]
    else:
        raise RuntimeError(f'ndarray must have 1 or 2 dimensions not {x.ndim}. ')



# --------------- clm_channels ---------------- #
@singledispatch
def clm_channels(x):
    pass
    
@clm_channels.register
def _(x: str):  #assume it is a file
    return csndlib.mus_sound_chans(x)
    
@clm_channels.register
def _(x: mus_any):  #assume it is a gen
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
        raise RuntimeError(f'ndarray must have 1 or 2 dimensions not {x.ndim}. ')
        
# --------------- clm_srate ---------------- #
@singledispatch
def clm_srate(x):
    pass

@clm_srate.register
def _(x: str): #file
    return csndlib.mus_sound_srate(x)
  
# --------------- clm_framples ---------------- #
@singledispatch
def clm_framples(x):
    pass
    
@clm_framples.register
def _(x: str):# assume file
    return csndlib.mus_sound_framples(x)

@clm_framples.register
def _(x: mus_any):# 
    return x.mus_length
        
@clm_framples.register
def _(x: list):
    return len(x[0])

@clm_framples.register
def _(x: np.ndarray):
    if x.ndim == 1:
        return np.shape(x)[0]
    elif x.ndim == 2:
        return np.shape(x)[1]
    else:
        raise RuntimeError(f'ndarray must have 1 or 2 dimensions not {x.ndim}. ')
        
# --------------- clm random ---------------- # 
@singledispatch
def _random(x):
    pass

@_random.register
def _(x: float):
    return random.random()
        
@_random.register
def _(x: int):
    return random.randrange(x)

@singledispatch
def _random2(x, y):
    pass
    
@_random2.register
def _(x:float , y: float):
    return random.uniform(x,y)
    
@_random2.register
def _(x:int , y: int):
    return random.randrange(x, y)

def clm_random(*args):
    if len(args) == 0:
        return random.random()
    elif len(args) == 1:
        return _random(args[0])
    else:
        return _random2(args[0], args[1])
 
 
# --------------- just some extra utils ---------------- #    

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
    
@clamp.register
def _(x:  np.ndarray, lo, hi):
    return np.clip(x, lo, hi)


@clamp.register
def _(x:  list, lo, hi):    
    return list(map(lambda z : clamp(z,lo,hi), x))
    
# --------------- clip ---------------- #
    
@singledispatch
def clip(x, lo, hi):
    pass

# same as clamp
@clip.register
def _(x: float, lo, hi):
    return float(max(min(x, hi),lo))
    
@clip.register
def _(x: int, lo, hi):
    return int(max(min(x, hi),lo))
    
@clip.register
def _(x:  np.ndarray, lo, hi):
    return np.clip(x, lo, hi)


@clip.register
def _(x:  list, lo, hi):    
    return list(map(lambda z : clip(z,lo,hi), x))

# --------------- fold ---------------- #
# TODO: fix as not working for negative numbers
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
    

_vfold = np.vectorize(fold)
    
@fold.register
def _(x:  np.ndarray, lo, hi):
    return _vfold(x, lo, hi)

@fold.register
def _(x:  list, lo, hi):    
    return list(map(lambda z : fold(z,lo,hi), x))


# --------------- wrap ---------------- #

# TODO fix for negative nuymbers

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
        
        
_vwrap = np.vectorize(wrap)

@wrap.register
def _(x:  np.ndarray, lo, hi):
    return _vwrap(x, lo, hi)


@wrap.register
def _(x:  list, lo, hi):    
    return list(map(lambda z : wrap(z,lo,hi), x))



# --------------- smoothstep ---------------- #
@singledispatch
def smoothstep(x, edge0, edge1):
    pass

@smoothstep.register
def _(x: float, edge0: float, edge1:float):
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

_vsmoothstep = np.vectorize(smoothstep)

@smoothstep.register
def _(x:  np.ndarray, edge0: float, edge1: float):
    return _vsmoothstep(x, edge0, edge1)
    
@smoothstep.register
def _(x: list, edge0: float, edge1:float):
    return list(map(lambda z : smoothstep(z, edge0, edge1)))


# --------------- step ---------------- #
@singledispatch
def step(x, edge):
    pass

@step.register
def _(x: float, edge:float):
    return 0.0 if x < edge else 1.0
    
_vstep = np.vectorize(step)

@step.register
def _(x: np.ndarray, edge:float):
    return _vstep(x, edge)
    
@step.register
def _(x: list, edge:float):
    return list(map(lambda z : step(z, edge), x))


# --------------- mix ---------------- #

@singledispatch
def mix(x, y , a):
    pass
    
@mix.register
def _(x: float, y: float , a: float):
    return x * (1 - a) + y * a
    
@mix.register
def _(x: np.ndarray, y: np.ndarray , a: float):
    res = np.zeros_like(x)
    for i in range(len(x)):
        res[i] = mix(x[i], y[i], a)
    return res
        
@mix.register
def _(x: list, y: list , a: float):
    res = [None] * len(x)
    for i in range(len(x)):
        res[i] = mix(x[i], y[i], a)
    return res

# --------------- sign ---------------- #

@singledispatch
def sign(x):
    pass

@sign.register
def _(x: int ):
    if x == 0:
        return 0
    if x < 0:
        return -1
    else: 
        return 1

@sign.register
def _(x: float ):
    if x == 0:
        return 0.0
    if x < 0:
        return -1.0
    else: 
        return 1.0
 
@sign.register
def _(x: list):
    return list(map(sign, x))

@sign.register
def _(x: np.ndarray):
    return np.sign(x)
    


# --------------- CLM utility functions ---------------- #

cpdef cython.double  radians2hz(radians: cython.double):
    """Convert radians per sample to frequency in "Hz: rads * srate / (2 * pi)"""
    return cclm.mus_radians_to_hz(radians)

cpdef cython.double  hz2radians(hz: cython.double):
    """Convert frequency in Hz to radians per sample: hz * 2 * pi / srate"""
    return cclm.mus_hz_to_radians(hz)

cpdef cython.double degrees2radians(degrees: cython.double):
    """Convert degrees to radians: deg * 2 * pi / 360"""
    return cclm.mus_degrees_to_radians(degrees)
    
cpdef cython.double  radians2degrees(radians: cython.double):
    """Convert radians to degrees: rads * 360 / (2 * pi)"""
    return cclm.mus_radians_to_degrees(radians)
    
cpdef cython.double  db2linear(x: cython.double  ):
    """Convert decibel value db to linear value: pow(10, db / 20)"""
    return cclm.mus_db_to_linear(x)
    
cpdef cython.double  linear2db(x: cython.double  ):
    """Convert linear value to decibels: 20 * log10(lin)"""
    return cclm.mus_linear_to_db(x)
    
cpdef cython.double  odd_multiple(x: cython.double , y: cython.double ):
    """Return the odd multiple of x and y"""
    return cclm.mus_odd_multiple(x,y)
    
cpdef cython.double  even_multiple(x: cython.double , y: cython.double ):
    """Return the even multiple of x and y"""
    return cclm.mus_even_multiple(x,y)
    
cpdef cython.double  odd_weight(x: cython.double):
    """Return the odd weight of x"""
    return cclm.mus_odd_weight(x)
    
cpdef cython.double  even_weight(x: cython.double ):
    """Return the even weight of x"""
    return cclm.mus_even_weight(x)
    
cpdef cython.double get_srate():
    """Return current sample rate"""
    return cclm.mus_srate()
    
cpdef cython.double set_srate(r: cython.double ):
    """Set current sample rate"""
    return cclm.mus_set_srate(r)
    
cpdef cython.double seconds2samples(secs: cython.double ):
    """Use mus_srate to convert seconds to samples."""
    return cclm.mus_seconds_to_samples(secs)

cpdef cython.double samples2seconds(samples: int):
    """Use mus_srate to convert samples to seconds."""
    return cclm.mus_samples_to_seconds(samples)
    
cpdef cython.double ring_modulate(s1: cython.double , s2: cython.double ):
    """Return s1 * s2 (sample by sample multiply)"""
    return cclm.mus_ring_modulate(s1, s2)
    
cpdef cython.double  amplitude_modulate(s1: cython.double , s2: cython.double , s3: cython.double):
    """Carrier in1 in2): in1 * (carrier + in2)"""
    return cclm.mus_amplitude_modulate(s1, s2, s3)
    
cpdef cython.double  contrast_enhancement(sig: cython.double , fm_index: cython.double ):
    """Returns sig (index 1.0)): sin(sig * pi / 2 + fm_index * sin(sig * 2 * pi))"""
    return cclm.mus_contrast_enhancement(sig, fm_index)
    
cpdef cython.double dot_product(data1, data2):
    """Returns v1 v2 (size)): sum of v1[i] * v2[i] (also named scalar product)"""
    data1_ptr = mus_float_array.from_this(data1)    
    data2_ptr = mus_float_array.from_this(data2)    
    return cclm.mus_dot_product(data1_ptr.data, data2_ptr.data, len(data1))
    
cpdef cython.double polynomial(coeffs, x: cython.double ):
    """Evaluate a polynomial at x.  coeffs are in order of degree, so coeff[0] is the constant term."""
    coeffs_ptr = mus_float_array.from_this(coeffs)
    return cclm.mus_polynomial(coeffs_ptr.data, x, len(coeffs_ptr))

cpdef cython.double array_interp(fn, x: cython.double , size: int):
    """Taking into account wrap-around (size is size of data), with linear interpolation if phase is not an integer."""    
    fn_ptr = mus_float_array.from_this(fn)
    return cclm.mus_array_interp(fn_ptr.data, x, size)

cpdef cython.double bessi0(x: cython.double):
    """Bessel function of zeroth order"""
    return cclm.mus_bessi0(x)
    
cpdef cython.double mus_interpolate(interp_type: Interp, x: cython.double, table, size: int, y1: cython.double):
    """Interpolate in data ('table' is a ndarray) using interpolation 'type', such as Interp.LINEAR."""
    table_ptr = mus_float_array.from_this(table)
    return cclm.mus_interpolate(<cclm.mus_interp_t>interp_type, x, table_ptr.data, size, y1)
   
cpdef np.ndarray fft(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64], fftsize: int, sign: int):
    """Return the fft of rl and im which contain the real and imaginary parts of the data; len should be a power of 2, dir = 1 for fft, -1 for inverse-fft"""    
    cdef double [:] rdat_view = rdat
    cdef double [:] idat_view = idat
    res = cclm.mus_fft(&rdat_view[0], &idat_view[0], fftsize, sign)
    return rdat

cpdef np.ndarray make_fft_window(window_type: Window, size: int, beta: Optional[float]=0.0, alpha: Optional[float]=0.0):
    """fft data window (a ndarray). type is one of the sndlib fft window identifiers such as Window.KAISER, beta is the window family parameter, if any."""
    win = np.zeros(size, dtype=np.double)
    cdef double [:] win_view = win
    cclm.mus_make_fft_window_with_window(<cclm.mus_fft_window_t>window_type, size, beta, alpha, &win_view[0])
    return win

cpdef np.ndarray rectangular2polar(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64]):
    """Convert real/imaginary data in s rl and im from rectangular form (fft output) to polar form (a spectrum)"""
    size = len(rdat)
    cdef double [:] rdat_view = rdat
    cdef double [:] idat_view = idat
    res = cclm.mus_rectangular_to_polar(&rdat_view[0], &idat_view[0], size)
    return rdat

cpdef np.ndarray rectangular2magnitudes(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64]):
    """Convert real/imaginary  data in rl and im from rectangular form (fft output) to polar form, but ignore the phases"""
    size = len(rdat)
    cdef double [:] rdat_view = rdat
    cdef double [:] idat_view = idat
    cclm.mus_rectangular_to_magnitudes(&rdat_view[0], &idat_view[0], size)
    return rdat

cpdef np.ndarray polar2rectangular(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64]):
    """Convert real/imaginary data in rl and im from polar (spectrum) to rectangular (fft)"""
    size = len(rdat)
    cdef double [:] rdat_view = rdat
    cdef double [:] idat_view = idat
    cclm.mus_polar_to_rectangular(&rdat_view[0], &idat_view[0], size)
    return rdat

cpdef np.ndarray spectrum(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64], window: npt.NDArray[np.float64], norm_type: Spectrum):
    """Real and imaginary data in ndarrays rl and im, returns (in rl) the spectrum thereof; window is the fft data window (a ndarray as returned by 
        make_fft_window  and type determines how the spectral data is scaled:
              0 = data in dB,
              1 (default) = linear and normalized
              2 = linear and un-normalized."""
    if isinstance(window, list):
        window = np.array(window, dtype=np.double)
    size = len(rdat)
    cdef double [:] rdat_view = rdat
    cdef double [:] idat_view = idat
    cdef double [:] window_view = idat
    cclm.mus_spectrum(&rdat_view[0], &idat_view[0], &window_view[0], size, <cclm.mus_spectrum_t>norm_type)
    return rdat

cpdef np.ndarray convolution(rl1: npt.NDArray[np.float64], rl2: npt.NDArray[np.float64]):
    """Convolution of ndarrays v1 with v2, using fft of size len (a power of 2), result in v1"""
    size = len(rl1)
    cdef double [:] rl1_view = rl1
    cdef double [:] rl2_view = rl2
    cclm.mus_convolution(&rl1_view[0], &rl2_view[0], size)
    return rl1

cpdef np.ndarray autocorrelate(data: npt.NDArray[np.float64]):
    """In place autocorrelation of data (a ndarray)"""
    size = len(data)
    cdef double [:] data_view = data
    cclm.mus_autocorrelate(&data_view[0], size)
    return data
    
cpdef np.ndarray correlate(data1: npt.NDArray[np.float64], data2: npt.NDArray[np.float64]):
    """In place cross-correlation of data1 and data2 (both ndarrays)"""
    size = len(data1)
    cdef double [:] data1_view = data1
    cdef double [:] data2_view = data2
    cclm.mus_correlate(&data1_view[0], &data2_view[0], size)
    return data1

cpdef np.ndarray cepstrum(data: npt.NDArray[np.float64]):
    """Return cepstrum of signal"""
    size = len(data)
    cdef double [:] data_view = data
    cclm.mus_cepstrum(&data_view[0], size)
    return data
    
cpdef np.ndarray partials2wave(partials, wave=None, table_size: Optional[int]=None, norm: Optional[bool]=True ):
    """Take a list of partials (harmonic number and associated amplitude) and produce a waveform for use in table_lookup"""
    partials_ptr = mus_float_array.from_this(partials)
    
    if isinstance(wave, list):
        wave = np.array(wave, dtype=np.double)
                        
    if (not wave):
        if table_size:
            wave = np.zeros(table_size)
        else:
            wave = np.zeros(CLM.table_size)
    else:
        table_size = len(wave)
        
    cdef double [:] wave_view = wave
    
    cclm.mus_partials_to_wave(partials_ptr.data, len(partials) // 2, &wave_view[0], table_size, norm)
    return wave
    
cpdef np.ndarray phase_partials2wave(partials, wave=None, table_size: Optional[int]=None, norm: Optional[bool]=True ):
    """Take a list of partials (harmonic number, amplitude, initial phase) and produce a waveform for use in table_lookup"""
    partials_ptr = mus_float_array.from_this(partials)     
    
    if isinstance(partials, list):
        partials = np.array(partials, dtype=np.double)        
            
    if (not wave):
        if table_size:
            wave = np.zeros(table_size)
        else:
            wave = np.zeros(CLM.table_size)
    else:
        table_size = len(wave)
        
    cdef double [:] wave_view = wave
    
    cclm.mus_partials_to_wave(partials_ptr.data, len(partials) // 3, &wave_view[0], table_size, norm)
    return wave
    
cpdef np.ndarray partials2polynomial(partials, kind: Optional[int]=Polynomial.FIRST_KIND):
    """Returns a Chebyshev polynomial suitable for use with the polynomial generator to create (via waveshaping) the harmonic spectrum described by the partials argument."""
    if isinstance(partials, list):
        partials = np.array(partials, dtype=np.double)
        
    p = to_partials(partials)
    
    cdef double [:] p_view = p
    
    cclm.mus_partials_to_polynomial(len(p), &p_view[0], kind)
    return p

cpdef np.ndarray normalize_partials(partials):
    """Scales the partial amplitudes in the list/array or list 'partials' by the inverse of their sum (so that they add to 1.0)."""
    if isinstance(partials, list):
        partials = np.array(partials, dtype=np.double)
    
    cdef double [:] partials_view = partials
        
    cclm.mus_normalize_partials(len(partials), &partials_view[0])
    
    return partials

cpdef cython.double chebyshev_tu_sum(x: cython.double, tcoeffs, ucoeffs):
    """Returns the sum of the weighted Chebyshev polynomials Tn and Un (vectors or " S_vct "s), with phase x"""
    
    tcoeffs_ptr = mus_float_array.from_this(tcoeffs)
    ucoeffs_ptr = mus_float_array.from_this(ucoeffs)

    return cclm.mus_chebyshev_tu_sum(x, len(tcoeffs), tcoeffs_ptr.data, ucoeffs_ptr.data)
    
cpdef cython.double chebyshev_t_sum(x: cython.double, tcoeffs):
    """returns the sum of the weighted Chebyshev polynomials Tn"""
    tcoeffs_ptr = mus_float_array.from_this(tcoeffs)
    return cclm.mus_chebyshev_t_sum(x,len(tcoeffs), tcoeffs_ptr.data)

cpdef cython.double chebyshev_u_sum(x: cython.double, ucoeffs):
    """returns the sum of the weighted Chebyshev polynomials Un"""
    ucoeffs_ptr = mus_float_array.from_this(ucoeffs)
    return cclm.mus_chebyshev_u_sum(x,len(ucoeffs),  ucoeffs_ptr.data)

# ---------------- oscil ---------------- #
cpdef mus_any make_oscil( frequency: Optional[cython.double]=0., initial_phase: Optional[cython.double] = 0.0):
    """Return a new oscil (sinewave) generator"""
    return mus_any.from_ptr(cclm.mus_make_oscil(frequency, initial_phase))
    
cpdef cython.double oscil(os: mus_any, fm: Optional[float]=None, pm: Optional[float]=None):
    """Return next sample from oscil  gen: val = sin(phase + pm); phase += (freq + fm)"""    
    if not (fm or pm):
        return cclm.mus_oscil_unmodulated(os._ptr)
    elif fm and not pm:
        return cclm.mus_oscil_fm(os._ptr, fm)
    else:
        return cclm.mus_oscil(os._ptr, fm, pm)
               
cpdef bint is_oscil(os: mus_any):
    """Returns True if gen is an oscil"""
    return cclm.mus_is_oscil(os._ptr)
    
# ---------------- oscil-bank ---------------- #

cpdef mus_any make_oscil_bank(freqs, phases, amps=None, stable: Optional[bool]=False):
    """Return a new oscil-bank generator. (freqs in radians)"""
    
    freqs_ptr = mus_float_array.from_this(freqs)
    phases_ptr = mus_float_array.from_this(phases)
    if amps is not None:
        amps_ptr = mus_float_array.from_this(amps)
    else:
         amps_ptr = None
    # TODO make sure same length
    #print(freqs_ptr)
    gen = mus_any.from_ptr(cclm.mus_make_oscil_bank(len(freqs_ptr), freqs_ptr.data, phases_ptr.data, amps_ptr.data, stable))
    gen.cache_extend([freqs_ptr, phases_ptr, amps_ptr])
    return gen

cpdef cython.double oscil_bank(os: mus_any):
    """sum an array of oscils"""
    return cclm.mus_oscil_bank(os._ptr)
    
cpdef bint is_oscil_bank(os: mus_any):
    """Returns True if gen is an oscil_bank"""
    return cclm.mus_is_oscil_bank(os._ptr)


# ---------------- env ---------------- #
cpdef mus_any make_env(envelope, scaler: Optional[cython.double]=1.0, duration: Optional[cython.double]=1.0, offset: Optional[cython.double]=0.0, base: Optional[cython.double]=1.0, length: Optional[int]=0):
    """Return a new envelope generator.  'envelope' is a list/array of break-point pairs. To create the envelope, these points are offset by 'offset', scaled by 'scaler', and mapped over the time interval defined by
        either 'duration' (seconds) or 'length' (samples).  If 'base' is 1.0, the connecting segments 
        are linear, if 0.0 you get a step function, and anything else produces an exponential connecting segment."""
    if length > 0:
        duration = samples2seconds(length)
    envelope_ptr = mus_float_array.from_this(envelope)

    gen =  mus_any.from_ptr(cclm.mus_make_env(envelope_ptr.data, len(envelope_ptr) // 2, scaler, offset, base, duration, 0, NULL))
    gen.cache_append(envelope_ptr)
    return gen
    
cpdef cython.double env(e: mus_any):
    return cclm.mus_env(e._ptr)
    
cpdef bint is_env(e: mus_any):
    """Returns True if gen is an env"""
    return cclm.mus_is_env(e._ptr)
    
cpdef cython.double env_interp(x: cython.double, env: mus_any):
    return cclm.mus_env_interp(x, env._ptr)
    
cpdef cython.double envelope_interp(x: cython.double, env: mus_any):
    return cclm.mus_env_interp(x, env._ptr)

# this is slow because of needing ctypes to define function at runtime
# but no other way to do it with cython without changing sndlib code
cpdef cython.double env_any(e: mus_any, connection_function):
    f = ENVFUNCTION(connection_function)
    cdef cclm.connect_points_cb cy_f_ptr = (<cclm.connect_points_cb*><size_t>ctypes.addressof(f))[0]
    return cclm.mus_env_any(e._ptr, cy_f_ptr)
    
    
# ---------------- pulsed-env ---------------- #    
cpdef mus_any make_pulsed_env(envelope, duration: cython.double, frequency: cython.double):    
    pl = mus_any.from_ptr(cclm.mus_make_pulse_train(frequency, 1.0, 0.0))
    envelope_ptr = mus_float_array.from_this(envelope)
    ge =  mus_any.from_ptr(cclm.mus_make_env(envelope_ptr.data, len(envelope_ptr) // 2, 1.0, 0, 1.0, duration, 0, NULL))
    gen = mus_any.from_ptr(cclm.mus_make_pulsed_env(ge._ptr, pl._ptr))
    gen.cache_extend([pl, ge])
    return gen
    
cpdef cython.double pulsed_env(gen: mus_any, fm: Optional[float]=None):
    if(fm):
        return cclm.mus_pulsed_env(gen._ptr, fm)
    else:
        return cclm.mus_pulsed_env_unmodulated(gen._ptr)
        
cpdef bint is_pulsedenv(e: mus_any):
    """Returns True if gen is a pulsed_env"""
    return cclm.mus_is_pulsed_env(e._ptr)
        
# TODO envelope-interp different than env-interp

# ---------------- table-lookup ---------------- #
cpdef mus_any make_table_lookup(frequency: Optional[cython.double]=0.0, 
                        initial_phase: Optional[cython.double]=0.0, 
                        wave=None, 
                        size: Optional[int]=512, 
                        interp_type: Optional[int]=Interp.LINEAR):        
    """Return a new table_lookup generator. The default table size is 512; use :size to set some other size, or pass your own list/array as the 'wave'."""                                               
    wave_ptr = mus_float_array.from_this(wave)       
    gen =  mus_any.from_ptr(cclm.mus_make_table_lookup(frequency, initial_phase, wave_ptr.data, size, interp_type))
    gen.cache_append(wave_ptr)
    return gen
    
cpdef cython.double table_lookup(tl: mus_any, fm_input: Optional[float]=None):
    if fm_input:
        return cclm.mus_table_lookup(tl._ptr, fm_input)
    else:
        return cclm.mus_table_lookup_unmodulated(tl._ptr)
        
cpdef bint is_table_lookup(tl: mus_any):
    """Returns True if gen is a table_lookup"""
    return cclm.mus_is_table_lookup(tl._ptr)

# TODO make-table-lookup-with-env


cpdef to_partials(harms):
    if isinstance(harms, list):
        p = harms[::2]
        maxpartial = max(p)
        partials = [0.0] * int(maxpartial+1)
        for i in range(0, len(harms),2):
            partials[int(harms[i])] = harms[i+1]
        return partials
    elif isinstance(harms[0], np.double): 
        p = harms[::2]
        maxpartial = np.max(p)
        partials = np.zeros(int(maxpartial)+1, dtype=np.double)
        for i in range(0, len(harms),2):
            partials[int(harms[i])] = harms[i+1]
        return partials
    else: 
        raise TypeError(f'{type(harms)} cannot be converted to a mus_float_array')
        
         
# ---------------- polywave ---------------- #
cpdef mus_any make_polywave(frequency: cython.double, 
                    partials = [0.,1.], 
                    kind: Optional[int]=Polynomial.FIRST_KIND, 
                    xcoeffs = None, 
                    ycoeffs = None):
    """Return a new polynomial-based waveshaping generator. make_polywave(440.0, partials=[1.0,1.0]) is the same in effect as make_oscil"""

    if(isinstance(xcoeffs, np.ndarray | list) ) and (isinstance(ycoeffs, np.ndarray | list)): # should check they are same length
        xcoeffs_ptr = mus_float_array.from_this(xcoeffs)
        ycoeffs_ptr = mus_float_array.from_this(ycoeffs)
        gen = mus_any.from_ptr(cclm.mus_make_polywave_tu(frequency,xcoeffs_ptr.data,ycoeffs_ptr.data, len(xcoeffs_ptr)))
        gen.cache_extend([xcoeffs_ptr,ycoeffs_ptr])
        return gen
    else:
        prtls = to_partials(partials)
        prtls_ptr = mus_float_array.from_this(prtls)
        gen = mus_any.from_ptr(cclm.mus_make_polywave(frequency, prtls_ptr.data, len(prtls_ptr), kind))
        gen.cache_append(prtls_ptr)
        return gen
    
    
cpdef cython.double polywave(w: mus_any, fm: Optional[float]=None):
    """Next sample of polywave waveshaper"""
    if fm:
        return cclm.mus_polywave(w._ptr, fm)
    else:
        return cclm.mus_polywave_unmodulated(w._ptr)
        
cpdef bint is_polywave(w: mus_any):
    """Returns True if gen is a polywave"""
    return cclm.mus_is_polywave(w._ptr)



# ---------------- polyshape ---------------- #

cpdef mus_any make_polyshape(frequency: cython.double,
                    initial_phase: Optional[cython.double]=0.0,
                    coeffs=None,
                    partials= [1.,1.], 
                    kind: Optional[int]=Polynomial.FIRST_KIND):
                    
    """Return a new polynomial-based waveshaping generator."""

    if coeffs:
        coeffs_ptr = mus_float_array.from_this(coeffs)
    else:
        p = partials2polynomial(partials, kind)
        coeffs_ptr = mus_float_array.from_this(p)
    
    gen = mus_any.from_ptr(cclm.mus_make_polyshape(frequency, initial_phase, coeffs_ptr.data, len(p), kind))
    gen.cache_append(coeffs_ptr)
    return gen
    
cpdef cython.double polyshape(w: mus_any, index: Optional[float]=1.0, fm: Optional[float]=None):
    """Next sample of polynomial-based waveshaper"""
    if fm:
        return cclm.mus_polyshape(w._ptr, index, fm)
    else:
        return cclm.mus_polyshape_unmodulated(w._ptr, index)
        
cpdef bint is_polyshape(w: mus_any):
    """Returns True if gen is a polyshape"""
    return cclm.mus_is_polyshape(w._ptr)


# ---------------- triangle-wave ---------------- #    
cpdef mus_any make_triangle_wave(frequency: cython.double, amplitude: Optional[cython.double]=1.0, phase: Optional[cython.double]=0.0):
    """return a new triangle_wave generator."""
    return mus_any.from_ptr(cclm.mus_make_triangle_wave(frequency, amplitude, phase))
    
cpdef cython.double triangle_wave(s: mus_any, fm: cython.double=None):
    """next triangle wave sample from generator"""
    if fm:
        return cclm.mus_triangle_wave(s._ptr, fm)
    else:
        return cclm.mus_triangle_wave_unmodulated(s._ptr)
    
cpdef bint is_triangle_wave(s: mus_any):
    """Returns True if gen is a triangle_wave"""
    return cclm.mus_is_triangle_wave(s._ptr)

# ---------------- square-wave ---------------- #    
cpdef mus_any make_square_wave(frequency: cython.double, amplitude: Optional[cython.double]=1.0, phase: Optional[cython.double]=0.0):
    """Return a new square_wave generator."""
    return mus_any.from_ptr(cclm.mus_make_square_wave(frequency, amplitude, phase))
    
cpdef cython.double square_wave(s: mus_any, fm: cython.double=0.0):
    """next square wave sample from generator"""
    return cclm.mus_square_wave(s._ptr, fm)

    
cpdef bint is_square_wave(s: mus_any):
    """Returns True if gen is a square_wave"""
    return cclm.mus_is_square_wave(s._ptr)
    
# ---------------- sawtooth-wave ---------------- #    
cpdef mus_any make_sawtooth_wave(frequency: cython.double, amplitude: Optional[cython.double]=1.0, phase: Optional[cython.double]=0.0):
    """Return a new sawtooth_wave generator."""
    return mus_any.from_ptr(cclm.mus_make_sawtooth_wave(frequency, amplitude, phase))
    
cpdef cython.double sawtooth_wave(s: mus_any, fm: cython.double=0.0):
    """next sawtooth wave sample from generator"""
    return cclm.mus_sawtooth_wave(s._ptr, fm)

    
cpdef bint is_sawtooth_wave(s: mus_any):
    """Returns True if gen is a sawtooth_wave"""
    return cclm.mus_is_sawtooth_wave(s._ptr)

# ---------------- pulse-train ---------------- #        
cpdef mus_any make_pulse_train(frequency: cython.double, amplitude: Optional[cython.double]=1.0, phase: Optional[cython.double]=0.0):
    """return a new pulse_train generator. This produces a sequence of impulses."""
    return mus_any.from_ptr(cclm.mus_make_pulse_train(frequency, amplitude, phase))
    
cpdef cython.double pulse_train(s: mus_any, fm: Optional[cython.double]=None):
    """next pulse train sample from generator"""
    if fm:
        return cclm.mus_pulse_train(s._ptr, fm)
    else:
        return cclm.mus_pulse_train_unmodulated(s._ptr)
     
cpdef bint is_pulse_train(s: mus_any):
    """Returns True if gen is a pulse_train"""
    return cclm.mus_is_pulse_train(s._ptr)
    
# ---------------- ncos ---------------- #

cpdef mus_any make_ncos(frequency: cython.double, n: Optional[int]=1):
    """return a new ncos generator, producing a sum of 'n' equal amplitude cosines."""
    return mus_any.from_ptr(cclm.mus_make_ncos(frequency, n))
    
cpdef cython.double ncos(nc: mus_any, fm: Optional[cython.double]=0.0):
    """Get the next sample from 'gen', an ncos generator"""
    return cclm.mus_ncos(nc._ptr, fm)

cpdef bint is_ncos(nc: mus_any):
    """Returns True if gen is a ncos"""
    return cclm.mus_is_ncos(nc._ptr)
    
    
# ---------------- nsin ---------------- #
cpdef mus_any make_nsin(frequency: cython.double, n: Optional[int]=1):
    """return a new nsin generator, producing a sum of 'n' equal amplitude sines"""
    return mus_any.from_ptr(cclm.mus_make_nsin(frequency, n))
    
cpdef cython.double nsin(nc: mus_any, fm: Optional[cython.double]=0.0):
    """Get the next sample from 'gen', an nsin generator"""
    return cclm.mus_nsin(nc._ptr, fm)
    
cpdef bint is_nsin(nc: mus_any):
    """Returns True if gen is a nsin"""
    return cclm.mus_is_nsin(nc._ptr)
    
# ---------------- nrxysin and nrxycos ---------------- #

cpdef mus_any make_nrxysin(frequency: cython.double, ratio: Optional[float]=1., n: Optional[int]=1, r: Optional[float]=.5):
    """Return a new nrxysin generator."""
    return mus_any.from_ptr(cclm.mus_make_nrxysin(frequency, ratio, n, r))
    
cpdef cython.double nrxysin(s: mus_any, fm: Optional[cython.double]=0.):
    """next sample of nrxysin generator"""
    return cclm.mus_nrxysin(s._ptr, fm)
    
cpdef bint is_nrxysin(s: mus_any):
    """Returns True if gen is a nrxysin"""
    return cclm.mus_is_nrxysin(s._ptr)
    
    
cpdef mus_any make_nrxycos(frequency: cython.double, ratio: Optional[cython.double]=1., n: Optional[int]=1, r: Optional[cython.double]=.5):
    """Return a new nrxycos generator."""
    return mus_any.from_ptr(cclm.mus_make_nrxycos(frequency, ratio, n, r))
    
cpdef cython.double nrxycos(s: mus_any, fm: Optional[cython.double]=0.):
    """next sample of nrxycos generator"""
    return cclm.mus_nrxycos(s._ptr, fm)
    
cpdef bint is_nrxycos(s: mus_any):
    """Returns True if gen is a nrxycos"""
    return cclm.mus_is_nrxycos(s._ptr)
    
    
# ---------------- rxykcos and rxyksin ---------------- #    
cpdef mus_any make_rxykcos(frequency: cython.double, phase: cython.double, r: Optional[cython.double]=.5, ratio: Optional[cython.double]=1.):
    """Return a new rxykcos generator."""
    return mus_any.from_ptr(cclm.mus_make_rxykcos(frequency, phase, r, ratio))
    
cpdef cython.double rxykcos(s: mus_any, fm: Optional[float]=0.):
    """next sample of rxykcos generator"""
    return cclm.mus_rxykcos(s._ptr, fm)
    
cpdef bint is_rxykcos(s: mus_any):
    """Returns True if gen is a rxykcos"""
    return cclm.mus_is_rxykcos(s._ptr)

cpdef mus_any make_rxyksin(frequency: cython.double, phase: cython.double, r: Optional[cython.double]=.5, ratio: Optional[cython.double]=1.):
    """Return a new rxyksin generator."""
    return mus_any.from_ptr(cclm.mus_make_rxyksin(frequency, phase, r, ratio))

cpdef cython.double rxyksin(s:  mus_any, fm: Optional[cython.double]=0.):
    """next sample of rxyksin generator"""
    return cclm.mus_rxyksin(s._ptr, fm)
    
cpdef bint  is_rxyksin(s: mus_any):
    """Returns True if gen is a rxyksin"""
    return cclm.mus_is_rxyksin(s._ptr)
        
# ---------------- ssb-am ---------------- #    
cpdef mus_any make_ssb_am(frequency: cython.double, n: Optional[int]=40):
    """Return a new ssb_am generator."""
    return mus_any.from_ptr(cclm.mus_make_ssb_am(frequency, n))
    
cpdef cython.double ssb_am(gen: mus_any, insig: cython.double, fm: Optional[cython.double]=None):
    """get the next sample from ssb_am generator"""
    if(fm):
        return cclm.mus_ssb_am(gen._ptr, insig, fm)
    else:
        return cclm.mus_ssb_am_unmodulated(gen._ptr, insig)
        
cpdef bint is_ssb_am(gen: mus_any):
    """Returns True if gen is a ssb_am"""
    return cclm.mus_is_ssb_am(gen._ptr)



# ---------------- wave-train ----------------#
cpdef mus_any make_wave_train(frequency: cython.double, wave, phase: Optional[cython.double]=0., interp_type=Interp.LINEAR):
    """Return a new wave-train generator (an extension of pulse-train). Frequency is the repetition rate of the wave found in wave. Successive waves can overlap."""
    wave_ptr = mus_float_array.from_this(wave)
    gen = mus_any.from_ptr(cclm.mus_make_wave_train(frequency, phase, wave_ptr.data, len(wave), interp_type))
    gen.cache_append(wave_ptr)
    return gen
    
cpdef cython.double wave_train(w: mus_any, fm: Optional[float]=None):
    """next sample of wave_train"""
    if fm:
        return cclm.mus_wave_train(w._ptr, fm)
    else:
        return cclm.mus_wave_train_unmodulated(w._ptr)
    
cpdef bint is_wave_train(w: mus_any):
    """Returns True if gen is a wave_train"""    
    return cclm.mus_is_wave_train(w._ptr)


cpdef mus_any make_wave_train_with_env(frequency: cython.double, pulse_env, size=None):
    size = size or CLM.table_size
    wave = np.zeros(size)  
    e = make_env(pulse_env, length=size)
    for i in range(size):
        wave[i] = env(e)
    cdef double [:] wave_view = wave
    gen = mus_any.from_ptr(cclm.mus_make_wave_train(frequency, 0.0, &wave_view[0], size, <cclm.mus_interp_t>Interp.LINEAR))
    gen.cache_append(wave)
    return gen

# ---------------- rand, rand_interp ---------------- #
cpdef mus_any make_rand(frequency: cython.double, amplitude: Optional[cython.double]=1.0, distribution=None):
    """Return a new rand generator, producing a sequence of random numbers (a step  function). frequency is the rate at which new numbers are chosen."""
    if distribution:
        distribution_ptr = mus_float_array.from_this(distribution)
        gen =  mus_any.from_ptr(cclm.mus_make_rand_with_distribution(frequency, amplitude, distribution_ptr.data, len(distribution)))
        gen.cache_append(distribution_ptr)
        return gen
    else:
        return mus_any.from_ptr(cclm.mus_make_rand(frequency, amplitude))

cpdef cython.double rand(r: mus_any, sweep: Optional[cython.double]=None):
    """gen's current random number. fm modulates the rate at which the current number is changed."""
    if(sweep):
        return cclm.mus_rand(r._ptr, sweep)
    else:
        return cclm.mus_rand_unmodulated(r._ptr)
    
cpdef bint is_rand(r: mus_any):
    """Returns True if gen is a rand"""
    return cclm.mus_is_rand(r._ptr)

cpdef mus_any make_rand_interp(frequency: cython.double, amplitude: cython.double,distribution=None):
    """Return a new rand_interp generator, producing linearly interpolated random numbers. frequency is the rate at which new end-points are chosen."""
    if distribution:
        distribution_ptr = mus_float_array.from_this(distribution)
        gen = mus_any.from_ptr(cclm.mus_make_rand_interp_with_distribution(frequency, amplitude, distribution_ptr.data, len(distribution)))
        gen.cache_append(distribution_ptr)
        return gen
    else:
        return mus_any.from_ptr(cclm.mus_make_rand_interp(frequency, amplitude))
    
cpdef rand_interp(r: mus_any, sweep: Optional[cython.double]=0.):
    """gen's current (interpolating) random number. fm modulates the rate at which new segment end-points are chosen."""
    if(sweep):
        return cclm.mus_rand_interp(r._ptr, sweep)
    else:
        return cclm.mus_rand_interp_unmodulated(r._ptr)
    
cpdef bint is_rand_interp(r: mus_any):
    """Returns True if gen is a rand_interp"""
    return cclm.mus_is_rand_interp(r._ptr)    
    
    
# ---------------- simple filters ---------------- #
cpdef mus_any make_one_pole(a0: cython.double, b1: cython.double):
    """Return a new one_pole filter; a0*x(n) - b1*y(n-1)"""
    return mus_any.from_ptr(cclm.mus_make_one_pole(a0, b1))
    
cpdef cython.double one_pole(f: mus_any, insig: cython.double):
    """One pole filter of input."""
    return cclm.mus_one_pole(f._ptr, insig)
    
cpdef bint is_one_pole(f: mus_any):
    """Returns True if gen is an one_pole"""
    return cclm.mus_is_one_pole(f._ptr)
    
cpdef mus_any make_one_zero(a0: cython.double, a1: cython.double):
    """Return a new one_zero filter; a0*x(n) + a1*x(n-1)"""
    return mus_any.from_ptr(cclm.mus_make_one_zero(a0, a1))
    
cpdef cython.double one_zero(f: mus_any, insig: cython.double):
    """One zero filter of input."""
    return cclm.mus_one_zero(f._ptr, insig)
    
cpdef bint is_one_zero(f: mus_any):
    """Returns True if gen is an one_zero"""
    return cclm.mus_is_one_zero(f._ptr)    

# make def for *args. are there other ways
def make_two_pole(*args):
    """Return a new two_pole filter; a0*x(n) - b1*y(n-1) - b2*y(n-2)"""
    if(len(args) == 2):
        return mus_any.from_ptr(cclm.mus_make_two_pole_from_frequency_and_radius(args[0], args[1]))
    elif(len(args) == 3):
        return mus_any.from_ptr(cclm.mus_make_two_pole(args[0], args[1], args[2]))
    else:
        raise RuntimeError("Requires either 2 or 3 args but received {len(args)}.")
       
cpdef cython.double two_pole(f: mus_any, insig: cython.double):
    """Return a new two_pole filter; a0*x(n) - b1*y(n-1) - b2*y(n-2)"""
    return cclm.mus_two_pole(f._ptr, insig)
    
cpdef bint is_two_pole(f: mus_any):
    return cclm.mus_is_two_pole(f._ptr)

# make def for *args. are there other ways
def make_two_zero(*args):
    """Return a new two_zero filter; a0*x(n) + a1*x(n-1) + a2*x(n-2)"""
    if(len(args) == 2):
        return mus_any.from_ptr(cclm.mus_make_two_zero_from_frequency_and_radius(args[0], args[1]))
    elif(len(args) == 3):
        return mus_any.from_ptr(cclm.mus_make_two_zero(args[0], args[1], args[2]))
    else:
        raise RuntimeError("Requires either 2 or 3 args but received {len(args)}.")

cpdef cython.double two_zero(f: mus_any, input: float):
    """Two zero filter of input."""
    return cclm.mus_two_zero(f._ptr, input)
    
cpdef bint is_two_zero(f: mus_any):
    """Returns True if gen is a two_zero"""
    return cclm.mus_is_two_zero(f._ptr)
    
    
# ---------------- one_pole_all_pass ---------------- #
cpdef mus_any make_one_pole_all_pass(size: int, coeff: cython.double):
    """Return a new one_pole all_pass filter size, coeff"""
    return mus_any.from_ptr(cclm.mus_make_one_pole_all_pass(size, coeff))
    
cpdef cython.double one_pole_all_pass(f: mus_any, insig: cython.double):
    """One pole all pass filter of input."""
    return cclm.mus_one_pole_all_pass(f._ptr, insig)
    
cpdef bint is_one_pole_all_pass(f: mus_any):
    """Returns True if gen is an one_pole_all_pass"""
    return cclm.mus_is_one_pole_all_pass(f._ptr)


# ---------------- formant ---------------- #
cpdef mus_any make_formant(frequency: cython.double, radius: cython.double):
    """Return a new formant generator (a resonator). radius sets the pole radius (in terms of the 'unit circle').
        frequency sets the resonance center frequency (Hz)."""
    return mus_any.from_ptr(cclm.mus_make_formant(frequency, radius))

cpdef cython.double formant(f: mus_any, insig: cython.double, radians: Optional[float]=None):
    """Next sample from resonator generator."""
    if radians:
        return cclm.mus_formant_with_frequency(f._ptr, insig, radians)
    else:
        return cclm.mus_formant(f._ptr, insig)
    
cpdef bint is_formant(f: mus_any):
    """Returns True if gen is a formant"""
    return cclm.mus_is_formant(f._ptr)

# ---------------- formant-bank ---------------- #   
cpdef mus_any make_formant_bank(filters: list, amps=None):
    """Return a new formant-bank generator."""

    p = list(map(is_formant, filters))
    if not all(p):
        raise TypeError(f'filter list contains at least one element that is not a formant.')
    
    filt_array = mus_any_array.from_pylist(filters)
    amps_ptr = None
    if amps is not None:
        amps_ptr = mus_float_array.from_this(amps)
        gen = mus_any.from_ptr(cclm.mus_make_formant_bank(len(filters),filt_array.data, amps_ptr.data))
    else: 
        gen = mus_any.from_ptr(cclm.mus_make_formant_bank(len(filters),filt_array.data, NULL))    
    gen.cache_extend([filt_array, amps_ptr])
    return gen
    
cpdef cython.double formant_bank(f: mus_any, inputs):
    """Sum a bank of formant generators"""
    if isinstance(inputs, (list, np.ndarray)):
        inputs_ptr = mus_float_array.from_this(inputs)
        gen =  cclm.mus_formant_bank_with_inputs(f._ptr, inputs_ptr.data)
        gen.cache_append(inputs_ptr)
        return gen
    else:
        res = cclm.mus_formant_bank(f._ptr, inputs)

    return res
     
cpdef bint is_formant_bank(f: mus_any):
    """Returns True if gen is a formant_bank"""
    return cclm.mus_is_formant_bank(f._ptr)

# ---------------- firmant ---------------- #
cpdef mus_any make_firmant(frequency: cython.double, radius: cython.double):
    """Return a new firmant generator (a resonator).  radius sets the pole radius (in terms of the 'unit circle').
        frequency sets the resonance center frequency (Hz)."""
    return mus_any.from_ptr(cclm.mus_make_firmant(frequency, radius))

cpdef firmant(f: mus_any, insig: cython.double, radians: Optional[cython.double]=None ):
    """Next sample from resonator generator."""
    if radians:
        return cclm.mus_firmant_with_frequency(f._ptr, insig, radians)
    else: 
        return cclm.mus_firmant(f._ptr, insig)
            
cpdef is_firmant(f: mus_any):
    """Returns True if gen is a firmant"""
    return cclm.mus_is_firmant(f._ptr)

# ---------------- filter ---------------- #
cpdef mus_any make_filter(order: int, xcoeffs, ycoeffs):   
    """Return a new direct form FIR/IIR filter, coeff args are list/ndarray"""     
    xcoeffs_ptr = mus_float_array.from_this(xcoeffs)    
    ycoeffs_ptr = mus_float_array.from_this(ycoeffs)    
    gen =  mus_any.from_ptr(cclm.mus_make_filter(order, xcoeffs_ptr.data, ycoeffs_ptr.data, NULL))
    gen.cache_extend([xcoeffs_ptr, ycoeffs_ptr])
    return gen
    
cpdef cython.double filter(fl: mus_any, insig: cython.double): # TODO : conflicts with buitl in function
    """next sample from filter"""
    return cclm.mus_filter(fl._ptr, insig)
    
cpdef bint is_filter(fl: mus_any):
    """Returns True if gen is a filter"""
    return cclm.mus_is_filter(fl._ptr)

# ---------------- fir-filter ---------------- #
cpdef mus_any make_fir_filter(order: int, xcoeffs):
    """return a new FIR filter, xcoeffs a list/ndarray"""
    xcoeffs_ptr = mus_float_array.from_this(xcoeffs)
    gen =  mus_any.from_ptr(cclm.mus_make_fir_filter(order, xcoeffs_ptr.data, NULL))
    gen.cache_append([xcoeffs_ptr])
    return gen
    
cpdef cython.double fir_filter(fl: mus_any, insig: cython.double):
    """next sample from fir filter"""
    return cclm.mus_fir_filter(fl._ptr, insig)
    
cpdef bint is_fir_filter(fl: mus_any):
    """Returns True if gen is a fir_filter"""
    return cclm.mus_is_fir_filter(fl._ptr)


# ---------------- iir-filter ---------------- #
cpdef mus_any make_iir_filter(order: int, ycoeffs):
    """return a new IIR filter, ycoeffs a list/ndarray"""
    ycoeffs_ptr = mus_float_array.from_this(ycoeffs) 
        
    gen = mus_any.from_ptr(cclm.mus_make_iir_filter(order, ycoeffs_ptr.data, NULL))
    gen.cache_append([ycoeffs_ptr])
    return gen
    
cpdef cython.double iir_filter(fl: mus_any, insig: cython.double ):
    """next sample from iir filter"""
    return cclm.mus_iir_filter(fl._ptr, insig)
    
cpdef bint is_iir_filter(fl: mus_any):
    """Returns True if gen is an iir_filter"""
    return cclm.mus_is_iir_filter(fl._ptr)

cpdef np.ndarray make_fir_coeffs(order: int, envelope):
    envelope_ptr = mus_float_array.from_this(envelope)
    coeffs = np.zeros(order+1, dtype=np.double)
    cdef double [:] coeffs_view = coeffs
    cclm.mus_make_fir_coeffs(order, envelope_ptr.data, &coeffs_view[0])
    return coeffs


# ---------------- delay ---------------- #
cpdef mus_any make_delay(size: int, 
                initial_contents=None, 
                initial_element: Optional[cython.double]=None, 
                max_size:Optional[int]=None,
                interp_type=Interp.NONE):
    """Return a new delay line of size elements. If the delay length will be changing at run-time, max-size sets its maximum length"""
  
    initial_contents_ptr = None
    contents = initial_contents
    
    if not max_size:
        max_size = size
        
    if max_size != size and interp_type == Interp.NONE:
        interp_type = Interp.LAGRANGE #think this is correct from clm2xen.c
        
    if initial_contents is not None:
        initial_contents_ptr = mus_float_array.from_this(contents)

    elif initial_element:
        initial_contents = np.zeros(max_size, dtype=np.double)
        initial_contents.fill(initial_element)
        initial_contents_ptr = mus_float_array.from_this(contents)

    if initial_contents_ptr is not None:
        gen = mus_any.from_ptr(cclm.mus_make_delay(size, initial_contents_ptr.data, max_size, interp_type))
    else:   
        gen = mus_any.from_ptr(cclm.mus_make_delay(size, NULL, max_size, interp_type))
    
    gen.cache_append(initial_contents_ptr)
    return gen
    
cpdef cython.double delay(d: mus_any, insig: cython.double, pm: Optional[cython.double]=None):
    """Delay val according to the delay line's length and pm ('phase-modulation').
        If pm is greater than 0.0, the max-size argument used to create gen should have accommodated its maximum value. """
    if pm:
        return cclm.mus_delay(d._ptr, insig, pm)
    else: 
        return cclm.mus_delay_unmodulated(d._ptr, insig)
        
cpdef bint is_delay(d: mus_any):
    """Returns True if gen is a delay"""
    return cclm.mus_is_delay(d._ptr)

cpdef cython.double tap(d: mus_any, offset: Optional[cython.double ]=None):
    """tap the delay mus_any offset by pm"""
    if offset:
        return cclm.mus_tap(d._ptr, offset)
    else:
        return cclm.mus_tap_unmodulated(d._ptr)
    
cpdef bint is_tap(d: mus_any):
    """Returns True if gen is a tap"""
    return cclm.mus_is_tap(d._ptr)
    
cpdef cython.double delay_tick(d: mus_any, insig: cython.double ):
    """Delay val according to the delay line's length. This merely 'ticks' the delay line forward.
        The argument 'val' is returned."""
    return cclm.mus_delay_tick(d._ptr, insig)


# ---------------- comb ---------------- #
cpdef mus_any make_comb(scaler: Optional[cython.double]=1.0,
                size: Optional[int]=None, 
                initial_contents=None, 
                initial_element: Optional[cython.double]=None, 
                max_size:Optional[int]=None,
                interp_type=Interp.NONE):
    """Return a new comb filter (a delay line with a scaler on the feedback) of size elements. 
        If the comb length will be changing at run-time, max-size sets its maximum length."""                
                
    initial_contents_ptr = None
    contents = initial_contents
    
    if not max_size:
        max_size = size
        
    if max_size != size and interp_type == Interp.NONE:
        interp_type = Interp.BEZIER #think this is correct from clm2xen.c
    
    if initial_contents:
        initial_contents_ptr = mus_float_array.from_this(contents)

    elif initial_element:
        initial_contents = np.zeros(max_size)
        initial_contents.fill(initial_element)
        initial_contents_ptr = mus_float_array.from_this(contents)
                
    gen = mus_any.from_ptr(cclm.mus_make_comb(scaler, size, initial_contents_ptr.data, max_size, interp_type))
    gen.cache_append(initial_contents_ptr)
    return gen    
           
cpdef cython.double comb(cflt: mus_any, insig: cython.double, pm: Optional[cython.double]=None):
    """Comb filter val, pm changes the delay length."""
    if pm:
        return cclm.mus_comb(cflt._ptr, insig, pm)
    else:
        return cclm.mus_comb_unmodulated(cflt._ptr, insig)
    
cpdef bint is_comb(cflt: mus_any):
    """Returns True if gen is a comb"""
    return cclm.mus_is_comb(cflt._ptr)    

# ---------------- comb-bank ---------------- #
cpdef mus_any make_comb_bank(combs: list):
    """Return a new comb-bank generator."""
    
    p = list(map(is_comb, combs))
    if not all(p):
        raise TypeError(f'filter list contains at least one element that is not a formant.')

    comb_array = mus_any_array.from_pylist(combs)
    
    gen = mus_any.from_ptr(cclm.mus_make_comb_bank(len(combs), comb_array.data))
    gen.cache_append(comb_array)
    return gen

cpdef cython.double comb_bank(combs: mus_any, insig: cython.double):
    """Sum an array of comb filters."""
    return cclm.mus_comb_bank(combs._ptr, input)
    
cpdef bint is_comb_bank(combs: mus_any):
    """Returns True if gen is a comb_bank"""
    return cclm.mus_is_comb_bank(combs._ptr)


# ---------------- filtered-comb ---------------- #
cpdef mus_any make_filtered_comb(scaler:  cython.double,
                size: int, 
                filter: mus_any, 
                initial_contents=None, 
                initial_element: Optional[ cython.double]=0.0, 
                max_size:Optional[int]=None,
                interp_type=Interp.NONE):
                
    """Return a new filtered comb filter (a delay line with a scaler and a filter on the feedback) of size elements.
        If the comb length will be changing at run-time, max-size sets its maximum length."""

    initial_contents_ptr = None
    contents = initial_contents
    
    if not max_size:
        max_size = size
        
    if max_size != size and interp_type == Interp.NONE:
        interp_type = Interp.BEZIER #think this is correct from clm2xen.c
    
    if initial_contents:
        initial_contents_ptr = mus_float_array.from_this(contents)

    elif initial_element:
        initial_contents = np.zeros(max_size)
        initial_contents.fill(initial_element)
        initial_contents_ptr = mus_float_array.from_this(contents)
                
    gen = mus_any.from_ptr(cclm.mus_make_filtered_comb(scaler, int(size), initial_contents_ptr.data, int(max_size), interp_type, filter._ptr))
    gen.cache_append(initial_contents_ptr)
    return gen    
        
cpdef cython.double filtered_comb(cflt: mus_any, insig: cython.double, pm: Optional[ cython.double]=None):
    """Filtered comb filter val, pm changes the delay length."""
    if pm:
        return cclm.mus_filtered_comb(cflt._ptr, insig, pm)
    else:
        return cclm.mus_filtered_comb_unmodulated(cflt._ptr, insig)
        
cpdef bint is_filtered_comb(cflt: mus_any):
    """Returns True if gen is a filtered_comb"""
    return cclm.mus_is_filtered_comb(cflt._ptr)
    
# ---------------- filtered-comb-bank ---------------- #      
cpdef mus_any make_filtered_comb_bank(fcombs: list):
    """Return a new filtered_comb-bank generator."""
    p = list(map(is_formant, fcombs))
    if not all(p):
        raise TypeError(f'filter list contains at least one element that is not a filtered_comb.')

    fcomb_array = mus_any_array.from_pylist(fcombs)
    
    gen =  mus_any.from_ptr(cclm.mus_make_filtered_comb_bank(len(fcombs), fcomb_array.data))
    gen.cache_append(fcomb_array)
    return gen

cpdef cython.double filtered_comb_bank(fcombs: mus_any):
    """sum an array of filtered_comb filters."""
    return cclm.mus_filtered_comb_bank(fcombs._ptr, input)
    
cpdef bint is_filtered_comb_bank(fcombs: mus_any):
    """Returns True if gen is a filtered_comb_bank"""
    return cclm.mus_is_filtered_comb_bank(fcombs._ptr)

# ---------------- notch ---------------- #
cpdef mus_any make_notch(scaler: Optional[cython.double]=1.0,
                size: Optional[int]=None, 
                initial_contents=None, 
                initial_element: Optional[cython.double]=0.0, 
                max_size:Optional[int]=None,
                interp_type=Interp.NONE):

    """return a new notch filter (a delay line with a scaler on the feedforward) of size elements.
        If the notch length will be changing at run-time, max-size sets its maximum length"""
    
    initial_contents_ptr = None
    contents = initial_contents
    
    if not max_size:
        max_size = size
        
    if max_size != size and interp_type == Interp.NONE:
        interp_type = Interp.BEZIER #think this is correct from clm2xen.c
     
    
    if initial_contents:
        initial_contents_ptr = mus_float_array.from_this(contents)
        
    elif initial_element:
        initial_contents = np.zeros(max_size)
        initial_contents.fill(initial_element)
        initial_contents_ptr = mus_float_array.from_this(contents)
    
    gen = mus_any.from_ptr(cclm.mus_make_notch(scaler, size, initial_contents_ptr.data, max_size, interp_type))
    gen.cache_append(initial_contents_ptr)
    return gen    

cpdef cython.double notch(cflt: mus_any, insig: cython.double, pm: Optional[cython.double]=None):
    """Notch filter val, pm changes the delay length."""
    if pm:
        return cclm.mus_notch(cflt._ptr, input, pm)
    else:
        return cclm.mus_notch_unmodulated(cflt._ptr, insig)
    
cpdef bint is_notch(cflt: mus_any):
    """Returns True if gen is a notch"""
    return cclm.mus_is_notch(cflt._ptr)
    
    
# ---------------- all-pass ---------------- #
cpdef mus_any make_all_pass(feedback: cython.double, 
                feedforward: cython.double,
                size: int, 
                initial_contents=None, 
                initial_element: Optional[cython.double]=0.0, 
                max_size:Optional[int]=None,
                interp_type=Interp.NONE):

    """Return a new allpass filter (a delay line with a scalers on both the feedback and the feedforward).
        length will be changing at run-time, max-size sets its maximum length."""

    initial_contents_ptr = None
    contents = initial_contents
    
    if not max_size:
        max_size = size
        
    if max_size != size and interp_type == Interp.NONE:
        interp_type = Interp.HERMITE #think this is correct from clm2xen.c
     
    
    if initial_contents:
        initial_contents_ptr = mus_float_array.from_this(contents)

    elif initial_element:
        initial_contents = np.zeros(max_size)
        initial_contents.fill(initial_element)
        initial_contents_ptr = mus_float_array.from_this(contents)
            

    gen = mus_any.from_ptr(cclm.mus_make_all_pass(feedback,feedforward, size, initial_contents_ptr.data, max_size, interp_type))
    gen.cache_append(initial_contents_ptr)
    return gen
    
cpdef all_pass(f: mus_any, insig: float, pm: Optional[float]=None):
    """All-pass filter val, pm changes the delay length."""
    if pm:
        return cclm.mus_all_pass(f._ptr, insig, pm)
    else:
        return cclm.mus_all_pass_unmodulated(f._ptr, insig)
    
cpdef is_all_pass(f: mus_any):
    """Returns True if gen is an all_pass"""
    return cclm.mus_is_all_pass(f._ptr)
    
# ---------------- all-pass-bank ---------------- #
cpdef mus_any make_all_pass_bank(all_passes: list):
    """Return a new all_pass-bank generator."""
    p = list(map(is_formant, all_passes))
    if not all(p):
        raise TypeError(f'allpass list contains at least one element that is not a all_pass.')
        
    all_passes_array = mus_any_array.from_pylist(all_passes)
    gen =  mus_any.from_ptr(cclm.mus_make_all_pass_bank(len(all_passes), all_passes_array.data))
    gen.cache_append(all_passes_array)
    return gen

cpdef cython.double all_pass_bank(all_passes: mus_any, insig: cython.double):
    """Sum an array of all_pass filters."""
    return cclm.mus_all_pass_bank(all_passes._ptr, insig)
    
cpdef bint is_all_pass_bank(o: mus_any):
    """Returns True if gen is an all_pass_bank"""
    return cclm.mus_is_all_pass_bank(o._ptr)
        
# ---------------- moving-average ---------------- #
cpdef mus_any make_moving_average(size: int, initial_contents=None, initial_element: Optional[cython.double]=0.0):
    """Return a new moving_average generator. """

    initial_contents_ptr = None
    contents = initial_contents
    
    if initial_contents:
        initial_contents_ptr = mus_float_array.from_this(contents)

    elif initial_element:
        initial_contents = np.zeros(size)
        initial_contents.fill(initial_element)
        initial_contents_ptr = mus_float_array.from_this(contents)
                        
    gen = mus_any.from_ptr(cclm.mus_make_moving_average(size, initial_contents_ptr.data))
    gen.cache_append(initial_contents_ptr)
    return gen
        
cpdef cython.double moving_average(f: mus_any, input: cython.double):
    """Moving window average."""
    return cclm.mus_moving_average(f._ptr, input)
    
cpdef bint is_moving_average(f: mus_any):
    """Returns True if gen is a moving_average"""
    return cclm.mus_is_moving_average(f._ptr)

# ---------------- moving-max ---------------- #
cpdef mus_any make_moving_max(size: int, 
                initial_contents=None, 
                initial_element: Optional[cython.double]=0.0):
                
    """Return a new moving-max generator."""                
    
    initial_contents_ptr = None
    contents = initial_contents
    
    if initial_contents:
        initial_contents_ptr = mus_float_array.from_this(contents)

    elif initial_element:
        initial_contents = np.zeros(size)
        initial_contents.fill(initial_element)
        initial_contents_ptr = mus_float_array.from_this(contents)
                    
    gen = mus_any.from_ptr(cclm.mus_make_moving_max(size, initial_contents_ptr.data))
    gen.cache_append(initial_contents_ptr)
    return gen
    
cpdef cython.double moving_max(f: mus_any, insig: float):
    """Moving window max."""
    return cclm.mus_moving_max(f._ptr, insig)
    
cpdef bint is_moving_max(f: mus_any):
    """Returns True if gen is a moving_max"""
    return cclm.mus_is_moving_max(f._ptr)
    
# ---------------- moving-norm ---------------- #
cpdef mus_any make_moving_norm(size: int,scaler: Optional[float]=1.):
    """Return a new moving-norm generator."""
    initial_contents = np.zeros(size, dtype=np.double)
    initial_contents_ptr = mus_float_array.from_ndarray(initial_contents)
    gen = mus_any.from_ptr(cclm.mus_make_moving_norm(size, initial_contents_ptr.data, scaler))
    gen.cache_append(initial_contents_ptr)
    return gen
    
cpdef cython.double moving_norm(f: mus_any, insig: cython.double):
    """Moving window norm."""
    return cclm.mus_moving_norm(f._ptr, insig)
    
cpdef is_moving_norm(f: mus_any):
    """Returns True if gen is a moving_norm"""
    return cclm.mus_is_moving_norm(f._ptr)
    
    
# ---------------- asymmetric-fm ---------------- #
cpdef mus_any make_asymmetric_fm(frequency: cython.double, initial_phase: Optional[cython.double]=0.0, r: Optional[cython.double]=1.0, ratio: Optional[cython.double]=1.):
    """Return a new asymmetric_fm generator."""
    return mus_any.from_ptr(cclm.mus_make_asymmetric_fm(frequency, initial_phase, r, ratio))
    
cpdef cython.double asymmetric_fm(af: mus_any, insig: cython.double, fm: Optional[cython.double]=None):
    """Next sample from asymmetric fm generator."""
    if fm:
        return cclm.mus_asymmetric_fm(af._ptr, insig, fm)
    else:
        return cclm.mus_asymmetric_fm_unmodulated(af._ptr, insig)
    
cpdef bint is_asymmetric_fm(af: mus_any):
    """Returns True if gen is an asymmetric_fm"""
    return cclm.mus_is_asymmetric_fm(af._ptr)
    
# ---------------- file-to-sample ---------------- #
cpdef mus_any make_file2sample(filename, buffer_size: Optional[int]=None):
    """Return an input generator reading 'filename' (a sound file)"""
    buffer_size = buffer_size or CLM.buffer_size
    return mus_any.from_ptr(cclm.mus_make_file_to_sample_with_buffer_size(filename, buffer_size))
    
cpdef cython.double file2sample(obj: mus_any, loc: int, chan: Optional[int]=0):
    """sample value in sound file read by 'obj' in channel chan at frample"""
    return cclm.mus_file_to_sample(obj._ptr, loc, chan)
    
cpdef bint is_file2sample(gen: mus_any):
    """Returns True if gen is an file2sample"""
    return cclm.mus_is_file_to_sample(gen._ptr)
    
# ---------------- sample-to-file ---------------- #
cpdef mus_any make_sample2file(filename, chans: Optional[int]=1, sample_type: Optional[Sample]=None, header_type: Optional[Header]=None, comment: Optional[str]=None):
    """Return an output generator writing the sound file 'filename' which is set up to have chans' channels of 'sample_type' 
        samples with a header of 'header_type'.  The latter should be sndlib identifiers:"""
    sample_type = sample_type or CLM.sample_type
    header_type = header_type or CLM.header_type
    if comment is None:
        return mus_any.from_ptr(cclm.mus_make_sample_to_file_with_comment(filename, chans, sample_type, header_type, NULL))
    else:
        return mus_any.from_ptr(cclm.mus_make_sample_to_file_with_comment(filename, chans, sample_type, header_type, comment))

cpdef cython.double sample2file(obj: mus_any, samp: int, chan:int , val: cython.double):
    """add val to the output stream handled by the output generator 'obj', in channel 'chan' at frample 'samp'"""
    return cclm.mus_sample_to_file(obj._ptr, samp, chan, val)
    
cpdef bint is_sample2file(obj: mus_any):
    """Returns True if gen is an sample2file"""
    return cclm.mus_is_sample_to_file(obj._ptr)
    
cpdef mus_any continue_sample2file(name: str):
    return mus_any.from_ptr(cclm.mus_continue_sample_to_file(name))
    
    
# ---------------- file-to-frample ---------------- #
cpdef mus_any make_file2frample(filename, buffer_size: Optional[int]=None):
    """Return an input generator reading 'filename' (a sound file)"""
    buffer_size = buffer_size or CLM.buffer_size
    return  mus_any.from_ptr(cclm.mus_make_file_to_frample_with_buffer_size(filename, buffer_size))
    
cpdef file2frample(obj: mus_any, loc: int):
    """frample of samples at frample 'samp' in sound file read by 'obj'"""
    outf = np.zeros(cclm.mus_channels(obj._ptr), dtype=np.double)
    outf_ptr = mus_float_array.from_ndarray(outf)
    cclm.mus_file_to_frample(obj._ptr, loc, outf_ptr.data)
    return outf
    
cpdef is_file2frample(gen: mus_any):
    """Returns True if gen is an file2frample"""
    return cclm.mus_is_file_to_frample(gen._ptr)
    
    
# ---------------- frample-to-file ---------------- #
cpdef mus_any make_frample2file(filename, chans: Optional[int]=1, sample_type: Optional[Sample]=None, header_type: Optional[Header]=None, comment: Optional[str]=None):
    """Return an output generator writing the sound file 'filename' which is set up to have 'chans' channels of 'sample_type' 
        samples with a header of 'header_type'.  The latter should be sndlib identifiers."""
    sample_type = sample_type or CLM.sample_type
    header_type = header_type or CLM.header_type
    if comment:
        return mus_any.from_ptr(cclm.mus_make_frample_to_file_with_comment(filename, chans, sample_type, header_type, comment))
    else:
        return mus_any.from_ptr(cclm.mus_make_frample_to_file_with_comment(filename, chans, sample_type, header_type, NULL))

cpdef cython.double frample2file(obj: mus_any, samp: int, chan:int, val):
    """add frample 'val' to the output stream handled by the output generator 'obj' at frample 'samp'"""
    frample_ptr = mus_float_array.from_this(val)
    cclm.mus_frample_to_file(obj._ptr, samp, frample_ptr.data)
    return val
    
cpdef bint is_frample2file(obj: mus_any):
    """Returns True if gen is a frample2file"""
    return cclm.mus_is_frample_to_file(obj._ptr)
    

cpdef mus_any continue_frample2file(name: str):
    return mus_any.from_ptr(cclm.mus_continue_frample_to_file(name))


# ---------------- readin ---------------- #
cpdef mus_any make_readin(filename: str, chan: int=0, start: int=0, direction: Optional[int]=1, buffer_size: Optional[int]=None):
    """return a new readin (file input) generator reading the sound file 'file' starting at frample 'start' in channel 'channel' and reading forward if 'direction' is not -1"""
    buffer_size = buffer_size or CLM.buffer_size
    return mus_any.from_ptr(cclm.mus_make_readin_with_buffer_size(filename, chan, start, direction, buffer_size))
    
cpdef readin(rd: mus_any):
    """next sample from readin generator (a sound file reader)"""
    return cclm.mus_readin(rd._ptr)
    
cpdef is_readin(rd: mus_any):
    """Returns True if gen mus_any a readin"""
    return cclm.mus_is_readin(rd._ptr)
      
          
# ---------------- src ---------------- #
cpdef mus_any make_src(inp , srate: Optional[cython.double]=1.0, width: Optional[int]=10):
    """Return a new sampling-rate conversion generator (using 'warped sinc interpolation'). 'srate' is the ratio between the new rate and the old. 'width' is the sine 
        width (effectively the steepness of the low-pass filter), normally between 10 and 100. 'input' if given is an open file stream."""
        
    cdef cclm.input_cb cy_inp_f_ptr 
    
    if(isinstance(inp, mus_any)):
        res = mus_any.from_ptr(cclm.mus_make_src(<cclm.input_cb>input_callback_func, srate, width, <void*>((<mus_any>inp)._ptr)))
        res._inputcallback = <cclm.input_cb>input_callback_func
        return res
  
    if not callable(inp):
        raise TypeError(f"Input needs to be a clm gen or function not a {type(inp)}")

    inp_f = INPUTCALLBACK(inp)
    cy_inp_f_ptr = (<cclm.input_cb*><size_t>ctypes.addressof(inp_f))[0]
    res = mus_any.from_ptr(cclm.mus_make_src(cy_inp_f_ptr, srate, width, NULL))
    res._inputcallback = cy_inp_f_ptr

    return res
  
cpdef cython.double src(s: mus_any, sr_change: Optional[cython.double]=0.0):
    """next sampling rate conversion sample. 'pm' can be used to change the sampling rate on a sample-by-sample basis.  
        'input-function' is a function of one argument (the current input direction, normally ignored) that is called internally 
        whenever a new sample of input data is needed.  If the associated make_src included an 'input' argument, input-function is ignored."""
    if s._inputcallback:
        return cclm.mus_src(s._ptr, sr_change, <cclm.input_cb>s._inputcallback)
    else:
        return 0.0
    
cpdef bint is_src(s: mus_any):
    """Returns True if gen is a type of src"""
    return cclm.mus_is_src(s._ptr)
 
  
# ---------------- convolve ---------------- #
cpdef mus_any make_convolve(inp, filt, fft_size: Optional[int]=512, filter_size: Optional[int]=None ):
    """Return a new convolution generator which convolves its input with the impulse response 'filter'."""
    
    cdef cclm.input_cb cy_input_f_ptr 
    
    filter_ptr = mus_float_array.from_this(filt)

    if not filter_size:
        filter_size = clm_length(filt)
     
    fft_len = 0
    
    if fft_size < 0 or fft_size == 0:
        raise ValueError(f'fft_size must be a positive number greater than 0 not {fft_size}')
        
    if fft_size > csndlib.mus_max_malloc():
         raise ValueError(f'fft_size too large. cannot allocate {fft_size} size fft')
        
    if is_power_of_2(filter_size):
        fft_len = filter_size * 2
    else :
        fft_len = next_power_of_2(filter_size)
        
    if fft_size < fft_len:
        fft_size = fft_len
        
    if(isinstance(inp, mus_any)):
        res = mus_any.from_ptr(cclm.mus_make_convolve(<cclm.input_cb>input_callback_func, filter_ptr.data, fft_size, filter_size, <void*>(<mus_any>inp)._ptr))
        res._inputcallback = <cclm.input_cb>input_callback_func
        return res
        
    if not callable(inp):
        raise TypeError(f"Input needs to be a clm gen or function not a {type(inp)}")

    inp_f = INPUTCALLBACK(inp)
    cy_input_f_ptr = (<cclm.input_cb*><size_t>ctypes.addressof(inp_f))[0]
    res = mus_any.from_ptr(cclm.mus_make_convolve(cy_input_f_ptr, filter_ptr.data, fft_size, filter_size, NULL))
    res._inputcallback = cy_input_f_ptr

    return res

    
cpdef cython.double convolve(gen: mus_any):
    """next sample from convolution generator"""
    return cclm.mus_convolve(gen._ptr, gen._inputcallback)  
    
cpdef bint is_convolve(gen: mus_any):
    """Returns True if gen is a convolve"""
    return cclm.mus_is_convolve(gen._ptr)
    
 
 
# --------------- granulate ---------------- 
cpdef mus_any make_granulate(inp, 
                    expansion: Optional[cython.double]=1.0, 
                    length: Optional[cython.double]=.15,
                    scaler: Optional[cython.double]=.6,
                    hop: Optional[cython.double]=.05,
                    ramp: Optional[cython.double]=.4,
                    jitter: Optional[cython.double]=0.0,
                    max_size: Optional[int]=0,
                    edit=None):
    """Return a new granular synthesis generator.  'length' is the grain length (seconds), 'expansion' is the ratio in timing
         between the new and old (expansion > 1.0 slows things down), 'scaler' scales the grains
        to avoid overflows, 'hop' is the spacing (seconds) between successive grains upon output.
        'jitter' controls the randomness in that spacing, 'input' can be a file pointer. 'edit' can
        be a function of one arg, the current granulate generator.  It is called just before
        a grain is added into the output buffer. The current grain is accessible via mus_data.
        The edit function, if any, should return the length in samples of the grain, or 0."""
    
    
    cdef cclm.input_cb cy_input_f_ptr
    cdef cclm.edit_cb cy_edit_f_ptr 
            
    if(isinstance(inp, mus_any) and edit is None):
        res = mus_any.from_ptr(cclm.mus_make_granulate(<cclm.input_cb>input_callback_func, expansion, length, scaler, hop, ramp, jitter, max_size, NULL, <void*>(<mus_any>inp)._ptr))
        res._inputcallback = <cclm.input_cb>input_callback_func
        return res
     
    if not callable(inp):
        raise TypeError(f"Input needs to be a clm gen or function not a {type(inp)}")   
    
    if(callable(inp)):
        inp_f = INPUTCALLBACK(inp)
        cy_input_f_ptr = (<cclm.input_cb*><size_t>ctypes.addressof(inp_f))[0]
        res = mus_any.from_ptr(cclm.mus_make_granulate(cy_input_f_ptr, expansion, length, scaler, hop, ramp, jitter, max_size, NULL, <void*>(<mus_any>inp)._ptr))
        res._inputcallback = cy_input_f_ptr

    if not callable(edit):
        raise TypeError(f"Edit needs to be a callable not a {type(edit)}")   
        
    if(edit is not None):
        edit_f = EDITCALLBACK(edit)
        cy_edit_f_ptr = (<cclm.edit_cb*><size_t>ctypes.addressof(edit_f))[0]
        cclm.mus_granulate_set_edit_function(res._ptr, cy_edit_f_ptr)
        res._editcallback  = cy_edit_f_ptr

    return res
    
    
#TODO: mus_granulate_grain_max_length
cpdef cython.double granulate(e: mus_any):
    """next sample from granular synthesis generator"""
    if e._editcallback:
        return cclm.mus_granulate_with_editor(e._ptr, e._inputcallback, e._editcallback)
    else:
        return cclm.mus_granulate(e._ptr, e._inputcallback)

cpdef bint is_granulate(e: mus_any):
    """Returns True if gen is a granulate"""
    return cclm.mus_is_granulate(e._ptr)

#--------------- phase-vocoder ----------------#
cpdef mus_any make_phase_vocoder(inp, 
                        fft_size: Optional[int]=512, 
                        overlap: Optional[int]=4, 
                        interp: Optional[int]=128, 
                        pitch: Optional[cython.double]=1.0, 
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

    cdef cclm.input_cb cy_inp_f_ptr
    cdef cclm.edit_cb cy_edit_f_ptr 
    cdef cclm.analyze_cb cy_analyze_f_ptr
    cdef cclm.synthesize_cb cy_synthesize_f_ptr 
   
        
    if(isinstance(inp, mus_any)):
        res = mus_any.from_ptr(cclm.mus_make_phase_vocoder(<cclm.input_cb>input_callback_func, fft_size, overlap, interp, pitch, NULL, NULL, NULL, <void*>(<mus_any>inp)._ptr))
        res._inputcallback = <cclm.input_cb>input_callback_func
    elif callable(inp):
        inp_f = INPUTCALLBACK(inp)
        cy_input_f_ptr = (<cclm.input_cb*><size_t>ctypes.addressof(inp_f))[0]
        res = mus_any.from_ptr(cclm.mus_make_phase_vocoder(cy_input_f_ptr, fft_size, overlap, interp, pitch, NULL, NULL, NULL, <void*>(<mus_any>inp)._ptr))
        res._inputcallback = <cclm.input_cb>input_callback_func
    else:
        raise TypeError(f"Input needs to be a clm gen or function not a {type(inp)}")  
    
    if(edit is not None):
        edit_f = EDITCALLBACK(edit)
        cy_edit_f_ptr = (<cclm.edit_cb*><size_t>ctypes.addressof(edit_f))[0]
        res._editcallback  = cy_edit_f_ptr

    if (analyze is not None):
        analyze_f = ANALYZECALLBACK(analyze)
        cy_analyze_f_ptr = (<cclm.analyze_cb*><size_t>ctypes.addressof(analyze_f))[0]
        res._analyzecallback  = cy_analyze_f_ptr

    if (synthesize is not None):
        synthesize_f = SYNTHESIZECALLBACK(synthesize)
        cy_synthesize_f_ptr = (<cclm.synthesize_cb*><size_t>ctypes.addressof(synthesize_f))[0]
        res._synthesizecallback = cy_synthesize_f_ptr

    return res

    
cpdef cython.double phase_vocoder(pv: mus_any):
    """next phase vocoder value"""
    if pv._analyzecallback or pv._synthesizecallback or pv._editcallback :
        return cclm.mus_phase_vocoder_with_editors(pv._ptr, pv._inputcallback, pv._analyzecallback, pv._editcallback, pv._synthesizecallback)
    else:
        return cclm.mus_phase_vocoder(pv._ptr, pv._inputcallback)
    
cpdef bint is_phase_vocoder(pv: mus_any):
    """Returns True if gen is a phase_vocoder"""
    return cclm.mus_is_phase_vocoder(pv._ptr)
    
cpdef phase_vocoder_amps(gen: mus_any):
    """Returns a ndarray containing the current output sinusoid amplitudes"""
    size = cclm.mus_length(gen._ptr)
    cdef cclm.mus_float_t* ptr = cclm.mus_phase_vocoder_amps(gen._ptr)
    p = np.asarray(<np.float64_t [:size]> ptr)
    return p

cpdef phase_vocoder_amp_increments(gen: mus_any):
    """Returns a ndarray containing the current output sinusoid amplitude increments per sample"""    
    size = cclm.mus_length(gen._ptr)
    cdef cclm.mus_float_t* ptr = cclm.mus_phase_vocoder_amp_increments(gen._ptr)
    p = np.asarray(<np.float64_t [:size]> ptr)
    return p
    
cpdef phase_vocoder_freqs(gen: mus_any):
    """Returns a ndarray containing the current output sinusoid frequencies"""
    size = cclm.mus_length(gen._ptr)
    cdef cclm.mus_float_t* ptr = cclm.mus_phase_vocoder_freqs(gen._ptr)
    p = np.asarray(<np.float64_t [:size]> ptr)
    return p
    
cpdef phase_vocoder_phases(gen: mus_any):
    """Returns a ndarray containing the current output sinusoid phases"""
    size = cclm.mus_length(gen._ptr)
    cdef cclm.mus_float_t* ptr = cclm.mus_phase_vocoder_phases(gen._ptr)
    p = np.asarray(<np.float64_t [:size]> ptr)
    return p
    
cpdef phase_vocoder_phase_increments(gen: mus_any):
    """Returns a ndarray containing the current output sinusoid phase increments"""
    size = cclm.mus_length(gen._ptr)
    cdef cclm.mus_float_t* ptr = cclm.mus_phase_vocoder_phase_increments(gen._ptr)
    p = np.asarray(<np.float64_t [:size]> ptr)
    return p
    
# --------------- out-any ---------------- #
cpdef out_any(loc: int, data: float, channel, output):
        if isinstance(output, np.ndarray): 
            output[channel][loc] += data
        else:            
            out = <mus_any>output
            cclm.mus_out_any(loc, data, channel, out._ptr)
        
# --------------- outa ---------------- #
cpdef outa(loc: int, data: float, output=None):
    if output is not None:
        out_any(loc, data, 0, output)        
    else:
        out_any(loc, data, 0, CLM.output)    
# --------------- outb ---------------- #    
cpdef outb(loc: int, data: float, output=None):
    if output is not None:
        out_any(loc, data, 1, output)        
    else:
        out_any(loc, data, 1, CLM.output)
# --------------- outc ---------------- #    
cpdef outc(loc: int, data: float, output=None):
    if output is not None:
        out_any(loc, data, 2, output)        
    else:
        out_any(loc, data, 2, CLM.output)        
# --------------- outd ---------------- #    
cpdef outd(loc: int, data: float, output=None):
    if output is not None:
        out_any(loc, data, 3, output)        
    else:
        out_any(loc, data, 3, CLM.output)    
# --------------- out-bank ---------------- #    
cpdef out_bank(gens, loc, inp):
    for i in range(len(gens)):
        out_any(loc, cclm.mus_apply(<cclm.mus_any_ptr>gens[i]._ptr, inp, 0.), i, CLM.output)    


#--------------- in-any ----------------#
cpdef in_any(loc: int, channel: int, inp):
    """input stream sample at frample in channel chan"""
    if is_list_or_ndarray(input):
        return inp[channel][loc]
    elif isinstance(inp, types.GeneratorType):
        return next(inp)
    elif callable(inp):
        return inp(loc, channel)
    else:
        return cclm.mus_in_any(loc, channel, <cclm.mus_any_ptr>inp._ptr)

#--------------- ina ----------------#
cpdef ina(loc: int, inp):
    return in_any(loc, 0, inp)

#--------------- inb ----------------#    
cpdef inb(loc: int, inp):
    return in_any(loc, 1, inp)



# --------------- locsig ---------------- #
cpdef mus_any make_locsig(degree: Optional[float]=0.0, 
    distance: Optional[cython.double]=1., 
    reverb: Optional[cython.double]=0.0, 
    output: Optional[mus_any]=None, 
    revout: Optional[mus_any]=None, 
    channels: Optional[int]=None, 
    reverb_channels: Optional[int]=None,
    interp_type: Optional[Interp]=Interp.LINEAR):
    
    """Return a new generator for signal placement in n channels.  Channel 0 corresponds to 0 degrees."""
    
    cdef cclm.detour_cb cy_detour_f_ptr
    
    if not output:
        output = CLM.output  #TODO : check if this exists
    
    if not revout:
        if CLM.reverb is not None:
            revout = CLM.reverb
        else: 
            revout = None #this generates and error but still works

    if not channels:
        channels = clm_channels(output)
    
    if not reverb_channels:
        reverb_channels = clm_channels(revout)
    
    if isinstance(output, mus_any):
        res = mus_any.from_ptr(cclm.mus_make_locsig(degree, distance, reverb, channels, <cclm.mus_any*>(output._ptr), reverb_channels, <cclm.mus_any*>(revout._ptr),  interp_type))
        return res
        
    # TODO: What if revout is not an iterable? While possible not going to deal with it right now :)   
    elif is_list_or_ndarray(output):
        if not reverb_channels:
            reverb_channels = 0
            
        res = mus_any.from_ptr(cclm.mus_make_locsig(degree, distance, reverb, channels, NULL, reverb_channels, NULL, interp_type))
        cclm.mus_locsig_set_detour(res._ptr, <cclm.detour_cb>locsig_detour_callback_func)

        return res
            
    else:
        raise TypeError(f"Output needs to be a clm gen or np.array not a {type(output)}")  
        
cpdef void locsig(gen: mus_any, loc: int, val: cython.double):
    """locsig 'gen' channel 'chan' scaler"""
    cclm.mus_locsig(gen._ptr, loc, val)
    
cpdef is_locsig(gen: mus_any):
    """Returns True if gen is a locsig"""
    return cclm.mus_is_locsig(gen._ptr)
    
cpdef cython.double locsig_ref(gen: mus_any, chan: int):
    """locsig 'gen' channel 'chan' scaler"""
    return cclm.mus_locsig_ref(gen._ptr, chan)
    
cpdef cython.double locsig_set(gen: mus_any, chan: int, val:cython.double):
    """set the locsig generator's channel 'chan' scaler to 'val'"""
    return cclm.mus_locsig_set(gen._ptr, chan, val)
    
cpdef cython.double locsig_reverb_ref(gen: mus_any, chan: int):
    """ locsig reverb channel 'chan' scaler"""
    return cclm.mus_locsig_reverb_ref(gen._ptr, chan)

cpdef cython.double locsig_reverb_set(gen: mus_any, chan: int, val: cython.double):
    """set the locsig reverb channel 'chan' scaler to 'val"""
    return cclm.mus_locsig_reverb_set(gen._ptr, chan, val)
    
cpdef void move_locsig(gen: mus_any, degree: cython.double, distance: cython.double):
    """move locsig gen to reflect degree and distance"""
    cclm.mus_move_locsig(gen._ptr, degree, distance)
    
    
# Added some options . TODO: what about sample rate conversion    
cpdef convolve_files(file1: str, file2: str, maxamp: Optional[cython.double]=1., outputfile='test.aif', sample_type=CLM.sample_type, header_type=CLM.header_type):
    if header_type == Header.NEXT:
        cclm.mus_convolve_files(file1, file2, maxamp,outputfile)
    else:
        temp_file = tempfile.gettempdir() + '/' + 'temp-' + outputfile
        cclm.mus_convolve_files(file1, file2, maxamp, temp_file)
        with Sound(outputfile, header_type=header_type, sample_type=sample_type):
            length = seconds2samples(csndlib.mus_sound_duration(temp_file))
            reader = make_readin(temp_file)
            for i in range(0, length):
                outa(i, readin(reader))
    return outputfile


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


# cpdef sndinfo(filename):
#     """Returns a dictionary of info about a sound file including write date (data), sample rate (srate),
#     channels (chans), length in samples (samples), length in second (length), comment (comment), and loop information (loopinfo)"""
#     date = csndlib.mus_sound_write_date(filename)
#     srate = csndlib.mus_sound_srate(filename)
#     chans = csndlib.mus_sound_chans(filename)
#     samples = csndlib.mus_sound_samples(filename)
#     comment = csndlib.mus_sound_comment(filename) 
#     length = samples / (chans * srate)
# 
#     header_type = Header(mus_sound_header_type(filename))
#     sample_type = Sample(mus_sound_sample_type(filename))
#     
#     loop_info = mus_sound_loop_info(filename)
#     if loop_info:
#         loop_modes = [loop_info[6], loop_info[7]]
#         loop_starts = [loop_info[0], loop_info[2]]
#         loop_ends = [loop_info[1], loop_info[3]]
#         base_note = loop_info[4]
#         base_detune = loop_info[5]
#     
#         loop_info = {'sustain_start' : loop_starts[0], 'sustain_end' : loop_ends[0], 
#                     'release_start' : loop_starts[2], 'release_end' : loop_ends[1],
#                     'base_note' : base_note, 'base_detune' : base_detune, 
#                     'sustain_mode' : loop_modes[0], 'release_mode' : loop_modes[1]}
#     
#     info = {'date' : time.localtime(date), 'srate' : srate, 'chans' : chans, 'samples' : samples,
#             'comment' : comment, 'length' : length, 'header_type' : header_type, 'sample_type' : sample_type,
#             'loop_info' : loop_info}
#     return info

# def sound_loop_info(filename):
#     """Returns a dictionary of info about a sound file including write date (data), sample rate (srate),
#     channels (chans), length in samples (samples), length in second (length), comment (comment), and loop information (loopinfo)"""
#     
#     loop_info = sndlib.mus_sound_loop_info(filename)
#     if loop_info:
#         loop_modes = [loop_info[6], loop_info[7]]
#         loop_starts = [loop_info[0], loop_info[2]]
#         loop_ends = [loop_info[1], loop_info[3]]
#         base_note = loop_info[4]
#         base_detune = loop_info[5]
#     
#         loop_info = {'sustain_start' : loop_starts[0], 'sustain_end' : loop_ends[0], 
#                     'release_start' : loop_starts[2], 'release_end' : loop_ends[1],
#                     'base_note' : base_note, 'base_detune' : base_detune, 
#                     'sustain_mode' : loop_modes[0], 'release_mode' : loop_modes[1]}
#     return info
#     
# def file2array(filename: str, channel: Optional[int]=0, beg: Optional[int]=None, dur: Optional[int]=None):
#     """Return an ndarray with samples from file"""
#     length = dur or mus_sound_framples(filename)
#     chans = mus_sound_chans(filename)
#     srate = mus_sound_srate(filename)
#     bg = beg or 0
#     out = np.zeros(length, dtype=np.double)
#         
#     mus_file_to_array(filename,channel, bg, length, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
#     return out
#     
# def channel2array(filename: str, channel: Optional[int]=0, beg: Optional[int]=None, dur: Optional[int]=None): 
#     length = dur or mus_sound_framples(filename)
#     srate = mus_sound_srate(filename)
#     bg = beg or 0
#     out = np.zeros((1, length), dtype=np.double)
#     mus_file_to_array(filename,channel, bg, length, out[0].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
#     return out
# 
# def calc_length(start, dur):
#     st = seconds2samples(start)
#     nd = seconds2samples(start+dur)
#     return st, nd
# 
# def convert_frequency(gen):
#     gen.frequency = hz2radians(gen.frequency)
#     return gen
# 
# # TODO: maybe add an exception that have to use keyword args
# def make_generator(name, slots, wrapper=None, methods=None, docstring=None):
#     class mus_gen():
#         pass
#     def make_generator(**kwargs):
#         gen = mus_gen()
#         setattr(gen, 'name', name)
#         for k, v  in kwargs.items():
#             setattr(gen, k, v)
#         if methods:
#             for k, v  in methods.items():
#                 setattr(mus_gen, k, property(v[0], v[1], None) )
#         
#         return gen if not wrapper else wrapper(gen)
#     def is_a(gen):
#         return isinstance(gen, mus_gen) and gen.name == name
#     g =  functools.partial(make_generator, **slots)
#     if docstring:
#         g.__doc__ = docstring
#     
#     return g, is_a
# 
# 
# def array_reader(arr, chan, loop=0):
#     ind = 0
#     if chan > (clm_channels(arr)):
#         raise ValueError(f'array has {clm_channels(arr)} channels but {chan} asked for')
#     length = clm_length(arr)
#     if loop:
#         def reader(direction):
#             nonlocal ind
#             v = arr[chan][ind]
#             ind += direction
#             ind = wrap(ind, 0, length-1)
#             return v
#             
#     else: 
#         def reader(direction):
#             nonlocal ind
#             v = arr[chan][ind]
#             ind += direction
#             ind = clip(ind, 0, length-1)
#             return v
#     return reader    
    
