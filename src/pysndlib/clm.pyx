# cython: c_string_type=unicode
# cython: c_string_encoding=utf8

import ctypes
from functools import singledispatch, partial
import math
import random
import subprocess
import tempfile
import time
import types
import cython
from cython.cimports.cpython.mem import PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython cimport view
import numpy as np
cimport numpy as np
import numpy.typing as npt
cimport pysndlib.cclm as cclm
cimport pysndlib.csndlib as csndlib
from pysndlib.sndlib import sample, header

np.import_array()


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
INPUTCALLBACK = ctypes.cfunctype(ctypes.c_double, ctypes.c_void_p, ctypes.c_int)
EDITCALLBACK = ctypes.cfunctype(ctypes.c_int, ctypes.c_void_p)
ANALYZECALLBACK = ctypes.cfunctype(ctypes.c_bool, ctypes.c_void_p, ctypes.cfunctype(ctypes.c_double, ctypes.c_void_p, ctypes.c_int))
SYNTHESIZECALLBACK = ctypes.cfunctype(ctypes.c_double, ctypes.c_void_p)
LOCSIGDETOURCALLBACK = ctypes.cfunctype(None, ctypes.c_void_p, ctypes.c_longlong) 
ENVFUNCTION = ctypes.cfunctype(ctypes.c_double, ctypes.c_double) # for env_any    

#these may need to be set based on system type etc
DEFAULT_OUTPUT_SRATE = 44100
DEFAULT_OUTPUT_CHANS = 1
DEFAULT_OUTPUT_SAMPLE_TYPE = sample.bfloat
DEFAULT_OUTPUT_HEADER_TYPE = header.aifc

    
    

# --------------- main clm prefs ---------------- #

CLM  = types.simplenamespace(
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
    locsig_type = Interp.LINEAR,
    clipped = True,
    player = False,
    output = False,
    delete_reverb = False
)


CLM.player = 'afplay'


# --------------- initializations ---------------- #

cclm.mus_initialize()
cclm.mus_set_rand_seed(int(time.time()))


cdef void clm_error_handler(int error_type, char* msg):
    message =  msg + ". "  +  csndlib.mus_error_type_to_string(error_type)
    raise SNDLibError(message) 
    
class SNDLibError(Exception):
    """
    this is general class to raise an print errors as defined in sndlib. it is to be used internally by the defined error handler registered with sndlib
    
    :meta private:
    """
    
    def ___init___(self, message):
        super().__init__(self.message)
        
        
        
def check_ndim(arr: npt.NDArray[np.float64], dim=1):
    if arr.dim == 1:
        raise TypeError(f'expecting {dim} dimemsions but received {arr.dim}.')
        
def compare_shapes(arr1: npt.NDArray[np.float64], arr2: npt.NDArray[np.float64]):
    if arr1.shape != arr2.shape:
        raise RuntimeError(f'ndarrays of unequal shape {arr1.shape } vs. {arr2.shape }.')

csndlib.mus_error_set_handler(<csndlib.mus_error_handler_t *>clm_error_handler)
    
# todo add mus_set_xcoeff and mus_set_ycoeff

# --------------- extension types ---------------- #

cdef class mus_any:
    """
    a wrapper class for mus_any pointers in c
    
    instances of this class have the following properties, which may be 
    used if supported by the type of generator used to make the instance:
    
    
    """
    
    cdef cclm.mus_any *_ptr
    cdef bint ptr_owner
    cdef cclm.input_cb _inputcallback
    cdef cclm.edit_cb _editcallback
    cdef cclm.analyze_cb _analyzecallback
    cdef cclm.synthesize_cb _synthesizecallback
    cdef list _cache
    cdef cclm.mus_float_t* _data_ptr
    cdef np.ndarray _data
    cdef cclm.mus_float_t* _xcoeffs_ptr
    cdef np.ndarray _xcoeffs
    cdef cclm.mus_float_t* _ycoeffs_ptr
    cdef np.ndarray _ycoeffs
    cdef cclm.mus_float_t* _pv_amp_increments_ptr
    cdef np.ndarray _pv_amp_increments
    cdef cclm.mus_float_t* _pv_amps_ptr
    cdef np.ndarray _pv_amps
    cdef cclm.mus_float_t* _pv_freqs_ptr
    cdef np.ndarray _pv_freqs
    cdef cclm.mus_float_t* _pv_phases_ptr
    cdef np.ndarray _pv_phases
    cdef cclm.mus_float_t* _pv_phase_increments_ptr
    cdef np.ndarray _pv_phase_increments
    
    def __cinit__(self):
        self.ptr_owner = False
        self._cache = []
        self._inputcallback = NULL
        self._editcallback = NULL
        self._analyzecallback = NULL
        self._synthesizecallback = NULL
        self._cache = []

        
    def __delalloc__(self):
        if self._ptr is not NULL and self.ptr_owner is True:
            cclm.mus_free(self._ptr)
            self._ptr = NULL

    def __init__(self):
        # prevent accidental instantiation from normal python code
        # since we cannot pass a struct pointer into a python constructor.
        raise TypeError("this class cannot be instantiated directly.")

    @staticmethod
    cdef mus_any from_ptr(cclm.mus_any *_ptr, bint owner=True):
        """
        factory function to create mus_any objects from
        given mus_any pointer.

        setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated.

        :meta private:
        """
        # fast call to __new__() that bypasses the __init__() constructor.
        cdef mus_any wrapper = mus_any.__new__(mus_any)
        wrapper._ptr = _ptr
        wrapper.ptr_owner = owner
        if cclm.mus_data_exists(wrapper._ptr):  
            wrapper.set_up_data()
        if cclm.mus_xcoeffs_exists(wrapper._ptr):  
            wrapper.set_up_xcoeffs()
        if cclm.mus_ycoeffs_exists(wrapper._ptr):  
            wrapper.set_up_ycoeffs()
            
        if is_phase_vocoder(wrapper):
            wrapper.set_up_pv_data()
            
        return wrapper
       
    cpdef cache_append(self, obj):
        """
        :meta private:
        """
        self._cache.append(obj) 

    cpdef cache_extend(self, obj):
        """
        :meta private:
        """
        self._cache.extend(obj) 
        
    # this stuff with view.array is to work around an apparent bug when automatic string handling is turned on
    # https://github.com/cython/cython/issues/4521        
    cpdef set_up_data(self):
        """
        :meta private:
        """
        cdef cclm.mus_long_t size = cclm.mus_length(self._ptr)
        self._data_ptr = cclm.mus_data(self._ptr)
        cdef view.array arr = view.array(shape=(size,),itemsize=sizeof(double), format='d', allocate_buffer=True)
        arr.data = <char*>self._data_ptr
        self._data = np.asarray(arr)       
        
    cpdef set_up_xcoeffs(self):
        """
        :meta private:
        """
        cdef cclm.mus_long_t size = cclm.mus_length(self._ptr)
        self._xcoeffs_ptr = cclm.mus_xcoeffs(self._ptr)
        cdef view.array arr = view.array(shape=(size,),itemsize=sizeof(double), format='d', allocate_buffer=True)
        arr.data = <char*>self._xcoeffs_ptr
        self._xcoeffs = np.asarray(arr)  
        
    cpdef set_up_ycoeffs(self):
        """
        :meta private:
        """
        cdef cclm.mus_long_t size = cclm.mus_length(self._ptr)
        self._ycoeffs_ptr = cclm.mus_ycoeffs(self._ptr)
        cdef view.array arr = view.array(shape=(size,),itemsize=sizeof(double), format='d', allocate_buffer=True)
        arr.data = <char*>self._ycoeffs_ptr
        self._ycoeffs = np.asarray(arr)       
        
        
    cpdef set_up_pv_data(self):
        """
        :meta private:
        """
        cdef cclm.mus_long_t size = cclm.mus_length(self._ptr)
        
        self._pv_amp_increments_ptr = cclm.mus_phase_vocoder_amp_increments(self._ptr)
        cdef view.array pvai = view.array(shape=(size,),itemsize=sizeof(double), format='d', allocate_buffer=True)
        pvai.data = <char*>self._pv_amp_increments_ptr
        self._pv_amp_increments = np.asarray(pvai)
        self._pv_amp_increments.fill(0.0)
        
        self._pv_amps_ptr = cclm.mus_phase_vocoder_amps(self._ptr)
        cdef view.array pva = view.array(shape=(size // 2,),itemsize=sizeof(double), format='d', allocate_buffer=True)
        pva.data = <char*>self._pv_amps_ptr
        self._pv_amps = np.asarray(pva)    
        self._pv_amps.fill(0.0)
        
        self._pv_freqs_ptr = cclm.mus_phase_vocoder_freqs(self._ptr)
        cdef view.array pvf = view.array(shape=(size,),itemsize=sizeof(double), format='d', allocate_buffer=True)
        pvf.data = <char*>self._pv_freqs_ptr
        self._pv_freqs = np.asarray(pvf)    
        self._pv_freqs.fill(0.0)
        
        self._pv_phases_ptr = cclm.mus_phase_vocoder_phases(self._ptr)
        cdef view.array pvp = view.array(shape=(size // 2,),itemsize=sizeof(double), format='d', allocate_buffer=True)
        pvp.data = <char*>self._pv_phases_ptr
        self._pv_phases = np.asarray(pvp)
        self._pv_phases.fill(0.0)
        
        self._pv_phase_increments_ptr = cclm.mus_phase_vocoder_phase_increments(self._ptr)
        cdef view.array pvpi = view.array(shape=(size // 2,),itemsize=sizeof(double), format='d', allocate_buffer=True)
        pvpi.data = <char*>self._pv_phase_increments_ptr
        self._pv_phase_increments = np.asarray(pvpi)  
        self._pv_phase_increments.fill(0.0)
        
    def __call__(self, arg1=0.0, arg2=0.0):
        return cclm.mus_apply(self._ptr, arg1,arg2)
        
    def __str__(self):
        return f'{mus_any} {cclm.mus_describe(self._ptr)}'
          
        
    @property
    def mus_frequency(self):
        """
        frequency (hz), float
        """
        return cclm.mus_frequency(self._ptr)
    
    @mus_frequency.setter
    def mus_frequency(self, v):
        cclm.mus_set_frequency(self._ptr, v)
    
    @property
    def mus_phase(self):
        """
        phase (radians), float
        """
        return cclm.mus_phase(self._ptr)
    
    @mus_phase.setter
    def mus_phase(self, v):
        cclm.mus_set_phase(self._ptr, v)
        
    @property
    def mus_length(self):
        """
        data length, int
        """
        return cclm.mus_length(self._ptr)
    
    @mus_length.setter
    def mus_length(self, v):
        cclm.mus_set_length(self._ptr, v)  
        
    @property
    def mus_increment(self):
        """
        various increments, int
        """
        return cclm.mus_increment(self._ptr)
    
    @mus_increment.setter
    def mus_increment(self, v):
        cclm.mus_set_increment(self._ptr, v)  
        
    @property
    def mus_location(self):
        """
        sample location for reads/write, int
        """
        return cclm.mus_location(self._ptr)
    
    @mus_location.setter
    def mus_location(self, v):
        cclm.mus_set_location(self._ptr, v)  
    
    @property
    def mus_offset(self):
        return cclm.mus_offset(self._ptr)
    
    @mus_offset.setter
    def mus_offset(self, v):
        """
        envelope offset, int
        """
        cclm.mus_set_offset(self._ptr, v)  
        
    @property
    def mus_channels(self):
        """
        channels open, int
        """
        return cclm.mus_channels(self._ptr)
    
        
    @property
    def mus_interp_type(self):
        """
        interpolation type (inter.linear, etc), interp
        not setable
        """
        return cclm.mus_interp_type(self._ptr)
    
    @property
    def mus_width(self):
        """
        width of interpolation tables, etc, int
        """
        return cclm.mus_width(self._ptr)
    
    @mus_width.setter
    def mus_width(self, v):
        cclm.mus_set_width(self._ptr, v)  
        
    @property
    def mus_order(self):
        """
        filter order, int
        """
        return cclm.mus_order(self._ptr)

    @property
    def mus_scaler(self):
        """
        scaler, normally on an amplitude, float
        """
        return cclm.mus_scaler(self._ptr)
    
    @mus_scaler.setter
    def mus_scaler(self, v):
        cclm.mus_set_scaler(self._ptr, v)  
        
    @property
    def mus_feedback(self):
        """
        feedback coefficient, float
        """
        return cclm.mus_feedback(self._ptr)
    
    @mus_feedback.setter
    def mus_feedback(self, v):
        cclm.mus_set_feedback(self._ptr, v)  
        
    @property
    def mus_feedforward(self):
        """
        feedforward coefficient, float
        """
        return cclm.mus_feedforward(self._ptr)
    
    @mus_feedforward.setter
    def mus_feedforward(self, v):
        cclm.mus_set_feedforward(self._ptr, v) 

    @property
    def mus_hop(self):
        """
        hop size for block processing, int
        """
        return cclm.mus_hop(self._ptr)
    
    @mus_hop.setter
    def mus_hop(self, v):
        cclm.mus_set_hop(self._ptr, v) 
        
    @property
    def mus_ramp(self):
        """
        granulate grain envelope ramp setting, int
        """
        return cclm.mus_ramp(self._ptr)
    
    @mus_ramp.setter
    def mus_ramp(self, v):
        cclm.mus_set_ramp(self._ptr, v) 
    
    @property
    def mus_channel(self):
        """
        channel being read/written, int
        """
        return cclm.mus_channel(self._ptr)

    @property
    def mus_data(self):
        """
        array of data, np.ndarray
        """
        if not cclm.mus_data_exists(self._ptr):   #could do this on all properties but seems best with array 
            raise TypeError(f'mus_data can not be called on {cclm.mus_name(self._ptr)}')
        return self._data 

    @mus_data.setter
    def mus_data(self, data: npt.NDArray[np.float64]):
        if not cclm.mus_data_exists(self._ptr):   #could do this on all properties but seems best with array 
            raise TypeError(f'mus_data can not be called on {cclm.mus_name(self._ptr)}')
                
        np.copyto(self._data , data)

    @property
    def mus_xcoeffs(self):
        """
        x (input) coefficient, float
        not settable
        """
        if not cclm.mus_xcoeffs_exists(self._ptr):   #could do this on all properties but seems best with array 
            raise TypeError(f'mus_xcoeffs can not be called on {cclm.mus_name(self._ptr)}')
        return self._xcoeffs
    
    @property
    def mus_ycoeffs(self):
        """
        y (output, feedback) coefficient, float
        not settable
        """
        if not cclm.mus_xcoeffs_exists(self._ptr):   #could do this on all properties but seems best with array 
            raise TypeError(f'mus_ycoeffs can not be called on {cclm.mus_name(self._ptr)}')
        return self._ycoeffs
    
   
   
cdef class mus_any_array:
    """
    a wrapper class arrays of mus_any objects
    :meta private:
    """
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
        PyMem_Free(self.data)  # no-op if self.data is null
    
    def __len__(self):
        return self.item_count
        
    def __str__(self):
        return f"count {self.item_count}"

    @staticmethod
    cdef mus_any_array from_pylist(lst: list):
        """
        factory function to create a wrapper around mus_float_t (double) and fill it with values from the list
        """
        cdef int i
        
        cdef mus_any_array wrapper = mus_any_array(len(lst))
        for i in range(len(lst)):
            wrapper.data[i] = (<mus_any>lst[i])._ptr
        return wrapper
        



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
         CLM.output[i][val] += outf[i]
    for i in range(reverb_channels):
         CLM.reverb[i][val] += revf[i]
         
         
# --------------- file2ndarray, ndarray2file ---------------- #                 
cpdef file2ndarray(filename: str, channel: Optional[int]=None, beg: Optional[int]=None, dur: Optional[int]=None):
    """
    return an ndarray with samples from file and the sample rate of the data
    
    :param filename: filename
    :param channel: if None, will read all channels, otherwise just channel specified 
    :param beg: beginning positions to read in samples
    :param dur: duration in samples to read
    :return: tuple of np.ndarray and sample rate
    :rtype: tuple
    
    """
    length = dur or csndlib.mus_sound_framples(filename)
    chans = csndlib.mus_sound_chans(filename)
    srate = csndlib.mus_sound_srate(filename)
    bg = beg or 0
    out = np.zeros((1 if (channel != None) else chans, length), dtype=np.double)
    
    cdef double [: , :] arr_view = out
    
    if not channel:
        for i in range(chans):
            csndlib.mus_file_to_array(filename, i, bg, length, &arr_view[0][i])
    else:
        csndlib.mus_file_to_array(filename,0, bg, length, &arr_view[0][0])
    return out, srate
    
cpdef ndarray2file(filename: str, arr: npt.NDArray[np.float64], length=None, sr=None, sample_type: Optional[sndlib.sample]=CLM.sample_type, header_type: Optional[sndlib.header]=CLM.header_type, comment=None ):
    """
    write an ndarray of samples to file
    
    :param filename: name of file
    :param arr: np.ndarray of samples
    :param length: length of samples to write. if None write all
    :param sr: sample rate of file to write
    :param sample_type: type of sample type to use. defaults to clm.sample_type
    :param header_type: header of sample type to use. defaults to clm.header_type
    :return: length in samples of file
    :rtype: int
    
    
    """
    
    if not sr:
        sr = CLM.srate

    chans = np.shape(arr)[0]
    length = length or np.shape(arr)[1]
    fd = csndlib.mus_sound_open_output(filename, int(sr), chans, sample_type, header_type, NULL)

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
         
# --------------- with sound context manager ---------------- #      
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
            if (self.scaled_by or self.scaled_to) and (self.sample_type in [sample.bfloat, sample.lfloat, sample.bdouble, sample.ldouble]):
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
            raise TypeError(f"writing to  {type(self.output)} not supported")
            
        if self.reverb_to_file:
            if self.continue_old_file:
                CLM.reverb = continue_sample2file(self.revfile)
            else:
                CLM.reverb = make_sample2file(self.revfile,self.reverb_channels, sample_type=self.sample_type , header_type=self.header_type)
        
        if self.reverb and not self.reverb_to_file and is_list_or_ndarray(self.output):
            CLM.reverb = np.zeros((self.reverb_channels, np.shape(CLM.output)[1]), dtype=CLM.output.dtype)
    
        return self
        
    def __exit__(self, *args):
        cdef cclm.mus_float_t [:] vals_view = None
        cdef cclm.mus_long_t [:] times_view = None
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
                    times = np.zeros(chans, dtype=np.int64)
                    vals_view = vals
                    times_view = times
                    maxamp = csndlib.mus_sound_maxamps(self.output, chans, &vals_view[0], &times_view[0])
                    statstr += f": maxamp: {vals} {times} "
            else:
                chans = clm_channels(self.output)
                vals = np.zeros(chans, dtype=np.double)
                times = np.zeros(chans, dtype=np.int64)
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
                    vals_view = vals
                    times_view = times
                    maxamp = csndlib.mus_sound_maxamps(self.revfile, chans, &vals_view[0], &times_view[0])
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
                ndarray2file(self.output, arr)
            else:
                self.output *= (self.scaled_to / np.max(np.abs(self.output)))
        elif self.scaled_by:
            if self.output_to_file:
                arr, _ = file2ndarray(self.output)
                arr *= self.scaled_by
                ndarray2file(self.output, arr)
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
    
    

cpdef np.ndarray to_partials(harms):
    if isinstance(harms, list):
        p = harms[::2]
        maxpartial = max(p)
        partials = [0.0] * int(maxpartial+1)
        check_ndim(partials)
        for i in range(0, len(harms),2):
            partials[int(harms[i])] = harms[i+1]
        return partials
    elif isinstance(harms[0], np.double): 
        p = harms[::2]
        maxpartial = np.max(p)
        partials = np.zeros(int(maxpartial)+1, dtype=np.double)
        check_ndim(partials)
        for i in range(0, len(harms),2):
            partials[int(harms[i])] = harms[i+1]
        return partials
    else: 
        raise TypeError(f'{type(harms)} cannot be converted to a mus_float_array')
        
    
# --------------- generic functions ---------------- #

cpdef int mus_close(obj: mus_any):
    return cclm.mus_close_file(obj._ptr)
    
cpdef bint mus_is_output(obj: mus_any):
    """
    returns true if gen is a type of output
    """
    return cclm.mus_is_output(obj._ptr)
    
cpdef bint mus_is_input(obj: mus_any):
    """
    returns true if gen is a type of output
    """
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
# todo: fix as not working for negative numbers
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

# todo fix for negative nuymbers

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
    


# # --------------- clm utility functions ---------------- #
# 
cpdef cython.double radians2hz(radians: cython.double):
    """
    convert radians per sample to frequency:  `hz = rads * srate / (2 * pi)`
    
    :param radians: frequency \in radians
    :return: frequency \in hertz
    :rtype: float
    """
    
    return cclm.mus_radians_to_hz(radians)

cpdef cython.double hz2radians(hz: cython.double):
    """
    convert frequency in hz to radians per sample:  `hz * 2 * pi / srate`
    
    :param hz: frequency \in hertz
    :return: frequency \in radians
    :rtype: float
    """
    return cclm.mus_hz_to_radians(hz)

cpdef cython.double degrees2radians(degrees: cython.double):
    """
    convert degrees to radians:  `degrees * 2 * pi / 360`
    
    :param degrees: angle in degrees
    :return: angle in radians
    :rtype: float
    
    """
    return cclm.mus_degrees_to_radians(degrees)
    
cpdef cython.double radians2degrees(radians: cython.double):
    """
    convert radians to degrees: `rads * 360 / (2 * pi)`
    
    :param radians: angle in radians
    :return: degree
    :rtype: float
    
    """
    
    return cclm.mus_radians_to_degrees(radians)
    
cpdef cython.double db2linear(x: cython.double):
    """
    convert decibel value db to linear value: `pow(10, db / 20)`

    :param x: decibel
    :return: linear amplitude
    :rtype: float
    
    """
    
    return cclm.mus_db_to_linear(x)
    
cpdef cython.double linear2db(x: cython.double):
    """
    convert linear value to decibels 20 * log10(lin)

    :param x: linear amplitude
    :return: decibel
    :rtype: float
    
    """
    
    return cclm.mus_linear_to_db(x)
    
cpdef cython.double odd_multiple(x: cython.double , y: cython.double ):
    """
    return y times the nearest odd integer to x
    
    :param x:
    :param y:
    :return: nearest odd integer as a float
    :rtype: float
    
    """
    
    return cclm.mus_odd_multiple(x,y)
    
cpdef cython.double even_multiple(x: cython.double , y: cython.double ):
    """
    return y times the nearest even integer to x
    
    :param x:
    :param y:
    :return: nearest even integer as a float
    :rtype: float
    """
    
    return cclm.mus_even_multiple(x,y)
    
cpdef cython.double odd_weight(x: cython.double):
    """
    return a number between 0.0 (x is even) and 1.0 (x is odd)
    
    :param x:
    :return weight:
    :rtype: float
    
    """
    
    return cclm.mus_odd_weight(x)
    
cpdef cython.double even_weight(x: cython.double ):
    """
    return a number between 0.0 (x is odd) and 1.0 (x is even)
    
    :param x:
    :return weight:
    :rtype: float
    """
    
    return cclm.mus_even_weight(x)
    
cpdef cython.double get_srate():
    """
    return current sample rate
    
    :return samplerate:
    :rtype: float
    
    """
    return cclm.mus_srate()
    
cpdef cython.double set_srate(r: cython.double ):
    """
    set current sample rate
    """
    return cclm.mus_set_srate(r)
    
cpdef cclm.mus_long_t seconds2samples(secs: cython.double ):
    """
    use mus_srate to convert seconds to samples.
    
    :param secs: time in seconds
    :return: time in samples
    :rtype: int
    """
    return cclm.mus_seconds_to_samples(secs)

cpdef cython.double samples2seconds(samples: mus_long_t):
    """
    use mus_srate to convert samples to seconds.
    
    :param samples: number of samples
    :return: time in seconds
    :rtype: float
    
    """
    return cclm.mus_samples_to_seconds(samples)
    
cpdef cython.double ring_modulate(s1: cython.double, s2: cython.double ):
    """
    return s1 * s2 (sample by sample multiply)
    
    :param s1: input 1
    :param s2: input 2
    :return: result
    :rtype: float
    
    """
    
    return cclm.mus_ring_modulate(s1, s2)
    
cpdef cython.double amplitude_modulate(s1: cython.double, s2: cython.double , s3: cython.double):
    """
    carrier in1 in2): in1 * (carrier + in2)
        
    :param s1: input 1
    :param s2: input 2
    :param s3: input 3
    :return: result
    :rtype: float
    
    """
    return cclm.mus_amplitude_modulate(s1, s2, s3)
    
cpdef cython.double contrast_enhancement(insig: cython.double, fm_index: cython.double =1.0 ):
    """
    returns insig (index 1.0)): sin(sig * pi / 2 + fm_index * sin(sig * 2 * pi))
    contrast_enhancement passes its input to sin as a kind of phase modulation.
    
    :param insig: input
    :param fm_index: 
    :return: result
    :rtype: float
    
    """
    return cclm.mus_contrast_enhancement(insig, fm_index)
    

cpdef cython.double dot_product(data1: npt.NDArray[np.float64], data2: npt.NDArray[np.float64]):
    """
    returns v1 v2 (size)): sum of v1[i] * v2[i] (also named scalar product)
    
    :param data1: input 1
    :param data2: input 2
    :return: result
    :rtype: float
    
    """
    check_ndim(data1)
    check_ndim(data2)
    compare_shapes(data1, data2)
    
    cdef double [:] data1_view = data1
    cdef double [:] data2_view = data2
    return cclm.mus_dot_product(&data1_view[0], &data2_view[0], len(data1))

cpdef cython.double polynomial(coeffs: npt.NDArray[np.float64], x: cython.double ):
    """
    evaluate a polynomial at x.  coeffs are in order of degree, so coeff[0] is the constant term.
    
    :param coeffs: coefficients where coeffs[0] is the constant term, and so on.
    :param x: input
    :return: result
    :rtype: float
    
    """
    check_ndim(coeffs)
    cdef double [:] coeffs_view = coeffs
    return cclm.mus_polynomial(&coeffs_view[0], x, len(coeffs))
    
cpdef cython.double array_interp(fn: npt.NDArray[np.float64], x: cython.double):
    """
    taking into account wrap-around (size is size of data), with linear interpolation if phase is not an integer.
    
    :param fn: input array
    :param x: interp point
    :return: result
    :rtype: float
    
    """
    check_ndim(fn)
    cdef double [:] fn_view = fn
    return cclm.mus_array_interp(&fn_view[0], x, len(fn))

cpdef cython.double bessi0(x: cython.double):
    """
    bessel function of zeroth order
    """
    return cclm.mus_bessi0(x)
    
cpdef cython.double mus_interpolate(interp_type: Interp, x: cython.double, table: npt.NDArray[np.float64], y1: cython.double = 0.):
    """
    interpolate in data ('table' is a ndarray) using interpolation 'type', such as Interp.linear.
    
    :param interp_type: type of interpolation
    :param x: interpolation value
    :param table: table to interpolate in
    :return: result
    :rtype: float
    
    """
    check_ndim(table)
    cdef double [:] table_view = table
    return cclm.mus_interpolate(<cclm.mus_interp_t>interp_type, x, &table_view[0], len(table), y1)
   
cpdef np.ndarray fft(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64], fft_size: int, sign: int):
    """
    return the fft of rl and im which contain the real and imaginary parts of the data; len should be a power of 2, dir = 1 for fft, -1 for inverse-fft
    
    :param rdat: real data
    :param imaginary: imaginary data
    :param fft_size: must be power of two
    :param sign: 1 for fft, -1 for inverse-fft
    :return: result written into rdat
    :rtype: np.ndarray
    
    """
    check_ndim(rdat)
    check_ndim(idat)
    compare_shapes(rdat, idat)
    
    cdef double [:] rdat_view = rdat
    cdef double [:] idat_view = idat
    res = cclm.mus_fft(&rdat_view[0], &idat_view[0], fft_size, sign)
    return rdat

cpdef np.ndarray make_fft_window(window_type: Window, size: int, beta: Optional[float]=0.0, alpha: Optional[float]=0.0):
    """
    fft data window (a ndarray). type is one of the sndlib fft window identifiers such as window.kaiser, beta is the window family parameter, if any.
    
    :param window_type: type of window
    :param size: window size
    :param beta: beta parameter if needed
    :param alpha: alpha parameter if needed
    :return: window
    :rtype: np.ndarray
    
    """
    win = np.zeros(size, dtype=np.double)
    check_ndim(win)
    
    cdef double [:] win_view = win
    cclm.mus_make_fft_window_with_window(<cclm.mus_fft_window_t>window_type, size, beta, alpha, &win_view[0])
    return win

cpdef np.ndarray rectangular2polar(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64]):
    """
    convert real/imaginary data in s rl and im from rectangular form (fft output) to polar form (a spectrum)
    
    :param rdat: real data
    :param imaginary:  imaginary data
    :return: magnitude written into rdat, idat contains phases
    :rtype: np.ndarray
    
    """
    size = len(rdat)
    
    check_ndim(rdat)
    check_ndim(idat)
    compare_shapes(rdat, idat)
    
    cdef double [:] rdat_view = rdat
    cdef double [:] idat_view = idat
    res = cclm.mus_rectangular_to_polar(&rdat_view[0], &idat_view[0], size)
    return rdat

cpdef np.ndarray rectangular2magnitudes(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64]):
    """
    convert real/imaginary data in rl and im from rectangular form (fft output) to polar form, but ignore the phases
    
    :param rdat: real data
    :param imaginary:  imaginary data
    :return: magnitude written into rdat
    :rtype: np.ndarray
    
    """
    size = len(rdat)
    cdef double [:] rdat_view = rdat
    cdef double [:] idat_view = idat
    compare_shapes(rdat, idat)
    
    cclm.mus_rectangular_to_magnitudes(&rdat_view[0], &idat_view[0], size)
    return rdat

cpdef np.ndarray polar2rectangular(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64]):
    """
    convert real/imaginary data in rl and im from polar (spectrum) to rectangular (fft)
    
    :param rdat: magnitude data
    :param imaginary: phases data
    :return: real data written into rdat, idat contains imaginary
    :rtype: np.ndarray
    
    """
    size = len(rdat)
    cdef double [:] rdat_view = rdat
    cdef double [:] idat_view = idat
    compare_shapes(rdat, idat)
    
    cclm.mus_polar_to_rectangular(&rdat_view[0], &idat_view[0], size)
    return rdat

cpdef np.ndarray spectrum(rdat: npt.NDArray[np.float64], idat: npt.NDArray[np.float64], window: npt.NDArray[np.cython.double64], norm_type: spectrum):
    """real and imaginary data in ndarrays rl and im, returns (in rl) the spectrum thereof; window is the fft data window (a ndarray as returned by 
        make_fft_window  and type determines how the spectral data is scaled:
              spectrum.in_db= data in db,
              spectrum.normalized (default) = linear and normalized
              spectrum.raw = linear and un-normalized.
    
    :param rdat: real data
    :param imaginary: imaginary data
    :param window: fft window
    :param norm_type: normalization type
    :return: spectrum
    :rtype: np.ndarray
              
    """
    if isinstance(window, list):
        window = np.array(window, dtype=np.double)
    size = len(rdat)
    cdef double [:] rdat_view = rdat
    cdef double [:] idat_view = idat
    cdef double [:] window_view = window
    
    check_ndim(rdat)
    check_ndim(idat)
    check_ndim(window)
    compare_shapes(rdat, idat)
    
    cclm.mus_spectrum(&rdat_view[0], &idat_view[0], &window_view[0], size, <cclm.mus_spectrum_t>norm_type)
    return rdat

cpdef np.ndarray convolution(rl1: npt.NDArray[np.float64], rl2: npt.NDArray[np.float64], fft_size: int):
    """
    convolution of ndarrays v1 with v2, using fft of size len (a power of 2), result in v1

    :param rl1: input data 1
    :param rl2: input  data 2
    :param fft_size: fft size
    :return: convolved output. also written into rl1
    :rtype: np.ndarray
    
    """
    size = len(rl1)
    cdef double [:] rl1_view = rl1
    cdef double [:] rl2_view = rl2
    check_ndim(rl1)
    check_ndim(rl2)
    compare_shapes(rl1, rl2)
    
    cclm.mus_convolution(&rl1_view[0], &rl2_view[0], size)
    return rl1

cpdef np.ndarray autocorrelate(data: npt.NDArray[np.float64]):
    """
    in place autocorrelation of data (a ndarray)

    :param data: data
    :return: autocorrelation result
    :rtype: np.ndarray

    """
    size = len(data)
    check_ndim(data)
    
    cdef double [:] data_view = data
    cclm.mus_autocorrelate(&data_view[0], size)
    return data
    
cpdef np.ndarray correlate(data1: npt.NDArray[np.float64], data2: npt.NDArray[np.float64]):
    """
    in place cross-correlation of data1 and data2 (both ndarrays)
    
    :param data1: data 1
    :param data2: data 2
    :return: correlation result written into data1
    :rtype: np.ndarray
    
    
    """
    
    size = len(data1)
    check_ndim(data1)
    check_ndim(data2)
    compare_shapes(data1, data2)
    cdef double [:] data1_view = data1
    cdef double [:] data2_view = data2

    cclm.mus_correlate(&data1_view[0], &data2_view[0], size)
    return data1

cpdef np.ndarray cepstrum(data: npt.NDArray[np.float64]):
    """
    return cepstrum of signal
    
    :param data: samples to analyze
    :return: cepstrum. also written into data
    :rtype: np.ndarray
    """
    size = len(data)
    check_ndim(data)
    cdef double [:] data_view = data
    cclm.mus_cepstrum(&data_view[0], size)
    return data
    
cpdef np.ndarray partials2wave(partials, wave: npt.NDArray[np.float64]=None, table_size: Optional[int]=None, norm: Optional[bool]=True ):
    """
    take a list or np.ndarray of partials (harmonic number and associated amplitude) and produce a waveform for use in table_lookup
    
    :param partials: list or np.ndarray of partials (harm and amp)
    :param wave: array to write wave into. if not provided, one will be allocated
    :param table_size: size of table
    :param norm: whether to normalize partials
    :return: array provided in wave or new array.
    :rtype: np.ndarray
    
    """
    
    if isinstance(partials, list):
        partials = np.array(partials, dtype=np.double)
    
    if isinstance(wave, list):
        wave = np.array(wave, dtype=np.double)
                        
    if not wave:
        if table_size:
            wave = np.zeros(table_size)
        else:
            wave = np.zeros(CLM.table_size)
    else:
        table_size = len(wave)

    check_ndim(partials)
    check_ndim(wave)

    cdef double [:] wave_view = wave
    cdef double [:] partials_view = partials
    
    cclm.mus_partials_to_wave(&partials_view[0], len(partials) // 2, &wave_view[0], table_size, norm)
    return wave
    
cpdef np.ndarray phase_partials2wave(partials, wave: npt.NDArray[np.float64], table_size: Optional[int]=None, norm: Optional[bool]=True ):
    """
    take a list of partials (harmonic number, amplitude, initial phase) and produce a waveform for use in table_lookup
    
    :param partials: list or np.ndarray of partials (harm, amp, phase)
    :param wave: array to write wave into. if not provided, one will be allocated
    :param table_size: size of table
    :param norm: whether to normalize partials
    :return: array provided in wave or new array.
    :rtype: np.ndarray
    
    """

    if isinstance(partials, list):
        partials = np.array(partials, dtype=np.double)        
            
    if (not wave):
        if table_size:
            wave = np.zeros(table_size)
        else:
            wave = np.zeros(CLM.table_size)
    else:
        table_size = len(wave)
    
    check_ndim(wave)
    check_ndim(partials)
    
    cdef double [:] wave_view = wave
    cdef double [:] partials_view = partials
    
    cclm.mus_partials_to_wave(&partials_view[0], len(partials) // 3, &wave_view[0], table_size, norm)
    return wave
    
cpdef np.ndarray partials2polynomial(partials, kind: Optional[polynomial]=polynomial.first_kind):
    """
    returns a chebyshev polynomial suitable for use with the polynomial generator to create (via waveshaping) the harmonic spectrum described by the partials argument.
    
    :param partials: list or np.ndarray of partials (harm and amp)
    :param kind: polynomial.either_kind, polynomial.first_kind, polynomial.second_kind, polynomial.both_kinds
    :return: chebyshev polynomial
    :rtype: np.ndarray
    
    """
    if isinstance(partials, list):
        partials = np.array(partials, dtype=np.double)
   
    check_ndim(partials)     
    
    p = to_partials(partials)
    
    cdef double [:] p_view = p
    
    cclm.mus_partials_to_polynomial(len(p), &p_view[0], kind)
    return p

cpdef np.ndarray normalize_partials(partials):
    """
    scales the partial amplitudes in the list/array or list 'partials' by the inverse of their sum (so that they add to 1.0).

    :param partials: list or np.ndarray of partials (harm and amp)
    :return: normalized partials
    :rtype: np.ndarray      
    
    """
    if isinstance(partials, list):
        partials = np.array(partials, dtype=np.double)
    
    check_ndim(partials)     
    
    cdef double [:] partials_view = partials
        
    cclm.mus_normalize_partials(len(partials), &partials_view[0])
    
    return partials

cpdef cython.double chebyshev_tu_sum(x: cython.double, tcoeffs: npt.NDArray[np.float64], ucoeffs: npt.NDArray[np.float64]):
    """
    returns the sum of the weighted chebyshev polynomials tn and un, with phase x
    
    :param x: input
    :param tcoeffs: tn
    :param ucoeffs: un
    :return: output
    :rtype: float
    
    """
    
    check_ndim(tcoeffs)
    check_ndim(ucoeffs)
    compare_shapes(tcoeffs, ucoeffs)
    
    cdef double [:] tcoeffs_view = tcoeffs
    cdef double [:] ucoeffs_view = ucoeffs
    return cclm.mus_chebyshev_tu_sum(x, len(tcoeffs), &tcoeffs_view[0], &ucoeffs_view[0])
    

cpdef cython.double chebyshev_t_sum(x: cython.double, tcoeffs: npt.NDArray[np.float64]):
    """
    returns the sum of the weighted chebyshev polynomials tn
    
    :param x: nput
    :param tcoeffs: tn
    :return: output
    :rtype: float
    
    """
    
    check_ndim(tcoeffs)
    
    cdef double [:] tcoeffs_view = tcoeffs
    return cclm.mus_chebyshev_t_sum(x,len(tcoeffs), &tcoeffs_view[0])

cpdef cython.double chebyshev_u_sum(x: cython.double, ucoeffs: npt.NDArray[np.float64]):
    """
    returns the sum of the weighted chebyshev polynomials un
    
    :param x: input
    :param ucoeffs: un
    :return: output
    :rtype: float
    
    """
    
    check_ndim(ucoeffs)
    
    cdef double [:] ucoeffs_view = ucoeffs
    return cclm.mus_chebyshev_u_sum(x,len(ucoeffs), &ucoeffs_view[0])
    



# ---------------- oscil ---------------- #
cpdef mus_any make_oscil(frequency: Optional[float]=0., initial_phase: Optional[float] = 0.0):
    """
    return a new oscil (sinewave) generator
    
    :param frequency: frequency in hz
    :param initial_phase: initial phase in radians
    :return: oscil gen
    
    """
    return mus_any.from_ptr(cclm.mus_make_oscil(frequency, initial_phase))
    
cpdef cython.double oscil(gen: mus_any, fm: Optional[float]=None, pm: Optional[float]=None):
    """
    return next sample from oscil gen: val = sin(phase + pm); phase += (freq + fm)
    
    :param gen: oscil gen
    :param fm: fm input
    :param pm: pm input
    :return: output
    :rtype: float
    
    """    
    if not (fm or pm):
        return cclm.mus_oscil_unmodulated(gen._ptr)
    elif fm and not pm:
        return cclm.mus_oscil_fm(gen._ptr, fm)
    else:
        return cclm.mus_oscil(gen._ptr, fm, pm)
               
cpdef bint is_oscil(gen: mus_any):
    """
    returns true if gen is an oscil
    
    :param gen: oscil gen
    :return: result
    :rtype: bool
    
    """
    return cclm.mus_is_oscil(gen._ptr)
    
# ---------------- oscil-bank ---------------- #

cpdef mus_any make_oscil_bank(freqs, phases, amps=None, stable: Optional[bool]=False):
    """
    return a new oscil_bank generator. (freqs in radians)
    
    :param freqs: list or np.ndarray of frequencies in radians
    :param phases: list or np.ndarray of initial phases in radians
    :param stable:  if it is true, oscil_bank can assume that the frequency. this is not operative
    :return: oscil_bank gen
    :rtype: mus_any
    
    """
    cdef cclm.mus_float_t [:] freqs_view = None
    cdef cclm.mus_float_t [:] phases_view = None
    cdef cclm.mus_float_t [:] amps_view = None
    
    if isinstance(freqs, list):
        freqs = np.array(freqs, dtype=np.double)
    if isinstance(phases, list):
        phases = np.array(phases, dtype=np.double)        

    check_ndim(freqs)
    check_ndim(phases)
    compare_shapes(freqs, phases)

    freqs_view = freqs
    phases_view = phases
    
    if amps is not None:
        if isinstance(amps, list):
            amps = np.array(amps, dtype=np.double)    
        check_ndim(amps)
        amps_view = amps
        

    gen = mus_any.from_ptr(cclm.mus_make_oscil_bank(len(freqs), &freqs_view[0], &phases_view[0], &amps_view[0], stable))
    gen.cache_extend([freqs, phases, amps])
    return gen

cpdef cython.double oscil_bank(gen: mus_any):
    """
    sum an array of oscils
    
    :param gen: oscil_bank gen
    :return: output
    :rtype: float
    
    """
    return cclm.mus_oscil_bank(gen._ptr)
    
cpdef bint is_oscil_bank(gen: mus_any):
    """
    returns true if gen is an oscil_bank
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_oscil_bank(gen._ptr)


# ---------------- env ---------------- #
cpdef mus_any make_env(envelope, scaler: Optional[float]=1.0, duration: Optional[float]=1.0, offset: Optional[float]=0.0, base: Optional[float]=1.0, length: Optional[int]=0):
    """
    return a new envelope generator.  'envelope' is a list/array of break-point pairs. to create the envelope, these points are offset by 'offset', scaled by 'scaler', and mapped over the time interval defined by
    either 'duration' (seconds) or 'length' (samples).  if 'base' is 1.0, the connecting segments 
    are linear, if 0.0 you get a step function, and anything else produces an exponential connecting segment.

    :param envelope: list or np.ndarry of breakpoint pairs
    :param scaler: scaler on every y value (before offset is added)
    :param duration: duration in seconds
    :param offset: value added to every y value
    :param base: type of connecting line between break-points
    :param length:  duration in samples
    :result: env gen
    :rtype: mus_any

    """
    if length > 0:
        duration = samples2seconds(length)
    
    if isinstance(envelope, list):
        envelope = np.array(envelope, dtype=np.double)        

    check_ndim(envelope)

    cdef double [:] envelope_view = envelope
    
    gen =  mus_any.from_ptr(cclm.mus_make_env(&envelope_view[0], len(envelope) // 2, scaler, offset, base, duration, 0, NULL))
    gen.cache_append(envelope)
    return gen

cpdef cython.double env(gen: mus_any):
    """
    next sample from envelope generator

    :param gen: env gen
    :return: env output
    :rtype: float

    """
    return cclm.mus_env(gen._ptr)

cpdef bint is_env(gen: mus_any):
    """
    returns true if gen is an env
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_env(gen._ptr)

cpdef cython.double env_interp(x: cython.double, env: mus_any):
    """
    value of envelope env at x

    :param x: location in envelope
    :param env: env gen
    :return: env output
    :rtype: float

    """
    return cclm.mus_env_interp(x, env._ptr)

cpdef cython.double envelope_interp(x: cython.double, env: mus_any):
    """
    value of envelope env at x

    :param x: location in envelope
    :param env: env gen
    :return: env output
    :rtype: float

    """
    return cclm.mus_env_interp(x, env._ptr)

# this is slow because of needing ctypes to define function at runtime
# but no other way to do it with cython without changing sndlib code
cpdef cython.double env_any(gen: mus_any, connection_function):
    """
    generate env output using conncection_func to 'connect the dots'.
    the env-any connecting function takes one argument, the current envelope value treated as going between 0.0 and 1.0 between each two points. 

    :param env: env gen
    :param connection_function: function used to connect points
    :return: env output
    :rtype: float

    """

    f = ENVFUNCTION(connection_function)
    cdef cclm.connect_points_cb cy_f_ptr = (<cclm.connect_points_cb*><size_t>ctypes.addressof(f))[0]
    return cclm.mus_env_any(gen._ptr, cy_f_ptr)


# ---------------- pulsed-env ---------------- #    
cpdef mus_any make_pulsed_env(envelope, duration: cython.double, frequency: cython.double):
    """
    produces a repeating envelope. env sticks at its last value, but pulsed-env repeats it over and over. 
    :param env: env gen
    :param duration: duration of envelope
    :param frequency: repetition rate in hz
    :return: env output
    :rtype: float
    """
    if isinstance(envelope, list):
        envelope = np.array(envelope, dtype=np.double)        

    pl = mus_any.from_ptr(cclm.mus_make_pulse_train(frequency, 1.0, 0.0))
    
    check_ndim(envelope)
    
    cdef double [:] envelope_view = envelope
    
    ge =  mus_any.from_ptr(cclm.mus_make_env(&envelope_view[0], len(envelope) // 2, 1.0, 0, 1.0, duration, 0, NULL))
    gen = mus_any.from_ptr(cclm.mus_make_pulsed_env(ge._ptr, pl._ptr))
    gen.cache_extend([pl, ge, envelope])
    return gen
    
cpdef cython.double pulsed_env(gen: mus_any, fm: Optional[cython.double]=None):
    """
    next sample from envelope generator
    
    :param gen: env gen
    :param fm: change frequency of repetition
    :return: env output
    :rtype: float
    
    """
    if(fm):
        return cclm.mus_pulsed_env(gen._ptr, fm)
    else:
        return cclm.mus_pulsed_env_unmodulated(gen._ptr)
        
cpdef bint is_pulsedenv(gen: mus_any):
    """
    returns true if gen is a pulsed_env
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_pulsed_env(gen._ptr)
        
# todo envelope-interp different than env-interp

# ---------------- table-lookup ---------------- #
cpdef mus_any make_table_lookup(frequency: Optional[cython.double]=0.0, 
                        initial_phase: Optional[cython.double]=0.0, 
                        wave=None, 
                        size: Optional[int]=512, 
                        interp_type: Optional[int]=Interp.LINEAR):        
    """
    return a new table_lookup generator. the default table size is 512; use size to set some other size, or pass your own list/array as the 'wave'.

    :param frequency: frequency of gen in hz
    :param initial_phase: initial phase of gen in radians
    :param wave: np.ndarray if provided is waveform
    :param size: if no wave provided, this will allocate a table of size 
    :param interp_type: type of interpolation used
    :return: table_lookup gen
    :rtype: mus_any
    """                                               
    
    if wave is None:
        wave = np.zeros(size)
    
    if isinstance(wave, list):
        wave = np.array(wave, dtype=np.double)        
    
    check_ndim(wave)
    
    cdef double [:] wave_view = wave
    
    gen =  mus_any.from_ptr(cclm.mus_make_table_lookup(frequency, initial_phase, &wave_view[0], size, interp_type))
    gen.cache_append(wave)
    return gen
    
cpdef cython.double table_lookup(gen: mus_any, fm: Optional[float]=None):
    """
    return next sample from table_lookup generator
        
    :param gen: table_lookup gen
    :param fm: fm input
    :return: table_lookup output
    :rtype: float
    
    """
    if fm:
        return cclm.mus_table_lookup(gen._ptr, fm)
    else:
        return cclm.mus_table_lookup_unmodulated(gen._ptr)
        
cpdef bint is_table_lookup(gen: mus_any):
    """
    returns true if gen is a table_lookup
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_table_lookup(gen._ptr)

# todo make-table-lookup-with-env


         
# ---------------- polywave ---------------- #
cpdef mus_any make_polywave(frequency: cython.double, 
                    partials = [0.,1.], 
                    kind: Optional[int]=Polynomial.FIRST_KIND, 
                    xcoeffs = None, 
                    ycoeffs = None):
    """
    return a new polynomial-based waveshaping generator. make_polywave(440.0, partials=[1.0,1.0]) is the same in effect as make_oscil
    """
    cdef double [:] xcoeffs_view = None
    cdef double [:] ycoeffs_view = None
    cdef double [:] prtls_view = None
    
    if(isinstance(xcoeffs, np.ndarray | list) ) and (isinstance(ycoeffs, np.ndarray | list)): # should check they are same length
        xcoeffs = np.array(xcoeffs, dtype=np.double)        
        xcoeffs_view = xcoeffs
        ycoeffs = np.array(ycoeffs, dtype=np.double)        
        ycoeffs_view = ycoeffs
        check_ndim(xcoeffs)
        check_ndim(ycoeffs)
        compare_shapes(xcoeffs, ycoeffs)
    
        gen = mus_any.from_ptr(cclm.mus_make_polywave_tu(frequency,&xcoeffs_view[0],&ycoeffs_view[0], len(xcoeffs)))
        gen.cache_extend([xcoeffs, ycoeffs])
        return gen
    else:
        prtls = to_partials(partials)
        
        prtls = np.array(prtls, dtype=np.double)        
        prtls_view = prtls
        
        check_ndim(prtls)
        
        gen = mus_any.from_ptr(cclm.mus_make_polywave(frequency, &prtls_view[0], len(prtls), kind))
        gen.cache_extend([prtls, xcoeffs, ycoeffs])
        return gen
    
    
cpdef cython.double polywave(gen: mus_any, fm: Optional[float]=None):
    """
    next sample of polywave waveshaper
    """
    if fm:
        return cclm.mus_polywave(gen._ptr, fm)
    else:
        return cclm.mus_polywave_unmodulated(gen._ptr)
        
cpdef bint is_polywave(gen: mus_any):
    """
    returns true if gen is a polywave
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_polywave(gen._ptr)



# ---------------- polyshape ---------------- #

cpdef mus_any make_polyshape(frequency: cython.double,
                    initial_phase: Optional[cython.double]=0.0,
                    coeffs=None,
                    partials= [1.,1.], 
                    kind: Optional[int]=Polynomial.FIRST_KIND):
                    
    """
    return a new polynomial-based waveshaping generator.
    """
    
    
    if coeffs:
        if isinstance(coeffs, list):
            data = np.array(coeffs, dtype=np.double)     
        else:
            data = coeffs   

    else:
        data = partials2polynomial(partials, kind)

    check_ndim(data) 

    cdef double [:] data_view = data
    
    gen = mus_any.from_ptr(cclm.mus_make_polyshape(frequency, initial_phase, &data_view[0], len(data), kind))
    gen.cache_append(data)
    return gen
    
cpdef cython.double polyshape(gen: mus_any, index: Optional[float]=1.0, fm: Optional[float]=None):
    """
    next sample of polynomial-based waveshaper
    """
    if fm:
        return cclm.mus_polyshape(gen._ptr, index, fm)
    else:
        return cclm.mus_polyshape_unmodulated(gen._ptr, index)
        
cpdef bint is_polyshape(gen: mus_any):
    """
    returns true if gen is a polyshape
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_polyshape(gen._ptr)


# ---------------- triangle-wave ---------------- #    
cpdef mus_any make_triangle_wave(frequency: cython.double, amplitude: Optional[cython.double]=1.0, phase: Optional[cython.double]=0.0):
    """
    return a new triangle_wave generator.
    """
    return mus_any.from_ptr(cclm.mus_make_triangle_wave(frequency, amplitude, phase))
    
cpdef cython.double triangle_wave(gen: mus_any, fm: Optional[cython.double]=None):
    """
    next triangle wave sample from generator
    """
    if fm:
        return cclm.mus_triangle_wave(gen._ptr, fm)
    else:
        return cclm.mus_triangle_wave_unmodulated(gen._ptr)
    
cpdef bint is_triangle_wave(gen: mus_any):
    """
    returns true if gen is a triangle_wave
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_triangle_wave(gen._ptr)

# ---------------- square-wave ---------------- #    
cpdef mus_any make_square_wave(frequency: cython.double, amplitude: Optional[cython.double]=1.0, phase: Optional[cython.double]=0.0):
    """
    return a new square_wave generator.
    """
    return mus_any.from_ptr(cclm.mus_make_square_wave(frequency, amplitude, phase))
    
cpdef cython.double square_wave(gen: mus_any, fm: cython.double=0.0):
    """
    next square wave sample from generator
    """
    return cclm.mus_square_wave(gen._ptr, fm)
    
cpdef bint is_square_wave(gen: mus_any):
    """
    returns true if gen is a square_wave
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_square_wave(gen._ptr)
    
# ---------------- sawtooth-wave ---------------- #    
cpdef mus_any make_sawtooth_wave(frequency: cython.double, amplitude: Optional[cython.double]=1.0, phase: Optional[cython.double]=0.0):
    """
    return a new sawtooth_wave generator.
    """
    return mus_any.from_ptr(cclm.mus_make_sawtooth_wave(frequency, amplitude, phase))
    
cpdef cython.double sawtooth_wave(gen: mus_any, fm: cython.double=0.0):
    """
    next sawtooth wave sample from generator
    """
    return cclm.mus_sawtooth_wave(gen._ptr, fm)

    
cpdef bint is_sawtooth_wave(gen: mus_any):
    """
    returns true if gen is a sawtooth_wave
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_sawtooth_wave(gen._ptr)

# ---------------- pulse-train ---------------- #        
cpdef mus_any make_pulse_train(frequency: cython.double, amplitude: Optional[cython.double]=1.0, phase: Optional[cython.double]=0.0):
    """
    return a new pulse_train generator. this produces a sequence of impulses.
    """
    return mus_any.from_ptr(cclm.mus_make_pulse_train(frequency, amplitude, phase))
    
cpdef cython.double pulse_train(gen: mus_any, fm: Optional[cython.double]=None):
    """
    next pulse train sample from generator
    """
    if fm:
        return cclm.mus_pulse_train(gen._ptr, fm)
    else:
        return cclm.mus_pulse_train_unmodulated(gen._ptr)
     
cpdef bint is_pulse_train(gen: mus_any):
    """
    returns true if gen is a pulse_train
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_pulse_train(gen._ptr)
    
# ---------------- ncos ---------------- #

cpdef mus_any make_ncos(frequency: cython.double, n: Optional[int]=1):
    """
    return a new ncos generator, producing a sum of 'n' equal amplitude cosines.
    """
    return mus_any.from_ptr(cclm.mus_make_ncos(frequency, n))
    
cpdef cython.double ncos(gen: mus_any, fm: Optional[cython.double]=0.0):
    """
    get the next sample from 'gen', an ncos generator
    """
    return cclm.mus_ncos(gen._ptr, fm)

cpdef bint is_ncos(gen: mus_any):
    """
    returns true if gen is a ncos
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_ncos(gen._ptr)
    
    
# ---------------- nsin ---------------- #
cpdef mus_any make_nsin(frequency: cython.double, n: Optional[int]=1):
    """
    return a new nsin generator, producing a sum of 'n' equal amplitude sines
    """
    return mus_any.from_ptr(cclm.mus_make_nsin(frequency, n))
    
cpdef cython.double nsin(gen: mus_any, fm: Optional[cython.double]=0.0):
    """
    get the next sample from 'gen', an nsin generator
    """
    return cclm.mus_nsin(gen._ptr, fm)
    
cpdef bint is_nsin(gen: mus_any):
    """
    returns true if gen is a nsin
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_nsin(gen._ptr)
    
# ---------------- nrxysin and nrxycos ---------------- #

cpdef mus_any make_nrxysin(frequency: cython.double, ratio: Optional[float]=1., n: Optional[int]=1, r: Optional[float]=.5):
    """
    return a new nrxysin generator.
    """
    return mus_any.from_ptr(cclm.mus_make_nrxysin(frequency, ratio, n, r))
    
cpdef cython.double nrxysin(gen: mus_any, fm: Optional[cython.double]=0.):
    """
    next sample of nrxysin generator
    """
    return cclm.mus_nrxysin(gen._ptr, fm)
    
cpdef bint is_nrxysin(gen: mus_any):
    """
    returns true if gen is a nrxysin
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_nrxysin(gen._ptr)
    
    
cpdef mus_any make_nrxycos(frequency: cython.double, ratio: Optional[cython.double]=1., n: Optional[int]=1, r: Optional[cython.double]=.5):
    """
    return a new nrxycos generator.
    """
    return mus_any.from_ptr(cclm.mus_make_nrxycos(frequency, ratio, n, r))
    
cpdef cython.double nrxycos(gen: mus_any, fm: Optional[cython.double]=0.):
    """
    next sample of nrxycos generator
    """
    return cclm.mus_nrxycos(gen._ptr, fm)
    
cpdef bint is_nrxycos(gen: mus_any):
    """
    returns true if gen is a nrxycos
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_nrxycos(gen._ptr)
    
    
# ---------------- rxykcos and rxyksin ---------------- #    
cpdef mus_any make_rxykcos(frequency: cython.double, phase: cython.double, r: Optional[cython.double]=.5, ratio: Optional[cython.double]=1.):
    """
    return a new rxykcos generator.
    """
    return mus_any.from_ptr(cclm.mus_make_rxykcos(frequency, phase, r, ratio))
    
cpdef cython.double rxykcos(gen: mus_any, fm: Optional[float]=0.):
    """
    next sample of rxykcos generator
    """
    return cclm.mus_rxykcos(gen._ptr, fm)
    
cpdef bint is_rxykcos(gen: mus_any):
    """
    returns true if gen is a rxykcos
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_rxykcos(gen._ptr)

cpdef mus_any make_rxyksin(frequency: cython.double, phase: cython.double, r: Optional[cython.double]=.5, ratio: Optional[cython.double]=1.):
    """
    return a new rxyksin generator.
    """
    return mus_any.from_ptr(cclm.mus_make_rxyksin(frequency, phase, r, ratio))

cpdef cython.double rxyksin(gen:  mus_any, fm: Optional[cython.double]=0.):
    """
    next sample of rxyksin generator
    """
    return cclm.mus_rxyksin(gen._ptr, fm)
    
cpdef bint  is_rxyksin(gen: mus_any):
    """
    returns true if gen is a rxyksin
    :param gen: gen
    :return: result
    :rtype: bool
    """    
    return cclm.mus_is_rxyksin(gen._ptr)
        
# ---------------- ssb-am ---------------- #    
cpdef mus_any make_ssb_am(frequency: cython.double, n: Optional[int]=40):
    """
    return a new ssb_am generator.
    """
    return mus_any.from_ptr(cclm.mus_make_ssb_am(frequency, n))
    
cpdef cython.double ssb_am(gen: mus_any, insig: cython.double, fm: Optional[cython.double]=None):
    """
    get the next sample from ssb_am generator
    """
    if(fm):
        return cclm.mus_ssb_am(gen._ptr, insig, fm)
    else:
        return cclm.mus_ssb_am_unmodulated(gen._ptr, insig)
        
cpdef bint is_ssb_am(gen: mus_any):
    """
    returns true if gen is a ssb_am
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_ssb_am(gen._ptr)



# ---------------- wave-train ----------------#
cpdef mus_any make_wave_train(frequency: cython.double, wave: npt.NDArray[np.float64], phase: Optional[cython.double]=0., interp_type=Interp.LINEAR):
    """
    return a new wave-train generator (an extension of pulse-train). frequency is the repetition rate of the wave found in wave. successive waves can overlap.
    """
    check_ndim(wave)
    cdef double [:] wave_view = wave
    gen = mus_any.from_ptr(cclm.mus_make_wave_train(frequency, phase, &wave_view[0], len(wave), interp_type))
    gen.cache_append(wave)
    return gen
    
cpdef cython.double wave_train(gen: mus_any, fm: Optional[float]=None):
    """
    next sample of wave_train
    """
    if fm:
        return cclm.mus_wave_train(gen._ptr, fm)
    else:
        return cclm.mus_wave_train_unmodulated(gen._ptr)
    
cpdef bint is_wave_train(gen: mus_any):
    """
    returns true if gen is a wave_train
    :param gen: gen
    :return: result
    :rtype: bool
    """    
    return cclm.mus_is_wave_train(gen._ptr)


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
    """
    return a new rand generator, producing a sequence of random numbers (a step  function). frequency is the rate at which new numbers are chosen.
    """
    cdef double [:] distribution_view = None
    
    if distribution:
        check_ndim(distribution)
        distribution_view = distribution
        gen =  mus_any.from_ptr(cclm.mus_make_rand_with_distribution(frequency, amplitude, &distribution_view[0], len(distribution)))
        gen.cache_append(distribution)
        return gen
    else:
        return mus_any.from_ptr(cclm.mus_make_rand(frequency, amplitude))

cpdef cython.double rand(gen: mus_any, sweep: Optional[cython.double]=None):
    """
    gen's current random number. fm modulates the rate at which the current number is changed.
    """
    if(sweep):
        return cclm.mus_rand(gen._ptr, sweep)
    else:
        return cclm.mus_rand_unmodulated(gen._ptr)
    
cpdef bint is_rand(gen: mus_any):
    """
    returns true if gen is a rand
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_rand(gen._ptr)

cpdef mus_any make_rand_interp(frequency: cython.double, amplitude: cython.double, distribution=None):
    """
    return a new rand_interp generator, producing linearly interpolated random numbers. frequency is the rate at which new end-points are chosen.
    """
    
    cdef double [:] distribution_view = None
    
    if distribution:
        check_ndim(distribution)
        distribution_view = distribution
        gen = mus_any.from_ptr(cclm.mus_make_rand_interp_with_distribution(frequency, amplitude, &distribution_view[0], len(distribution)))
        gen.cache_append(distribution)
        return gen
    else:
        return mus_any.from_ptr(cclm.mus_make_rand_interp(frequency, amplitude))
    
cpdef rand_interp(gen: mus_any, sweep: Optional[cython.double]=0.):
    """
    gen's current (interpolating) random number. fm modulates the rate at which new segment end-points are chosen.
    """
    if(sweep):
        return cclm.mus_rand_interp(gen._ptr, sweep)
    else:
        return cclm.mus_rand_interp_unmodulated(gen._ptr)
    
cpdef bint is_rand_interp(gen: mus_any):
    """
    returns true if gen is a rand_interp
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_rand_interp(gen._ptr)    
    
    
# ---------------- simple filters ---------------- #
cpdef mus_any make_one_pole(a0: cython.double, b1: cython.double):
    """
    return a new one_pole filter; a0*x(n) - b1*y(n-1)
    """
    return mus_any.from_ptr(cclm.mus_make_one_pole(a0, b1))
    
cpdef cython.double one_pole(gen: mus_any, insig: cython.double):
    """
    one pole filter of input.
    """
    return cclm.mus_one_pole(gen._ptr, insig)
    
cpdef bint is_one_pole(gen: mus_any):
    """
    returns true if gen is a one_pole
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_one_pole(gen._ptr)
    
cpdef mus_any make_one_zero(a0: cython.double, a1: cython.double):
    """
    return a new one_zero filter; a0*x(n) + a1*x(n-1)
    """
    return mus_any.from_ptr(cclm.mus_make_one_zero(a0, a1))
    
cpdef cython.double one_zero(gen: mus_any, insig: cython.double):
    """
    one zero filter of input.
    """
    return cclm.mus_one_zero(gen._ptr, insig)
    
cpdef bint is_one_zero(gen: mus_any):
    """
    returns true if gen is a one_zero
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_one_zero(gen._ptr)    

# make def for *args. are there other ways
def make_two_pole(*args):
    """
    return a new two_pole filter; a0*x(n) - b1*y(n-1) - b2*y(n-2)
    """
    if(len(args) == 2):
        return mus_any.from_ptr(cclm.mus_make_two_pole_from_frequency_and_radius(args[0], args[1]))
    elif(len(args) == 3):
        return mus_any.from_ptr(cclm.mus_make_two_pole(args[0], args[1], args[2]))
    else:
        raise RuntimeError("requires either 2 or 3 args but received {len(args)}.")
       
cpdef cython.double two_pole(gen: mus_any, insig: cython.double):
    """
    return a new two_pole filter; a0*x(n) - b1*y(n-1) - b2*y(n-2)
    """
    return cclm.mus_two_pole(gen._ptr, insig)
    
cpdef bint is_two_pole(gen: mus_any):
    """
    returns true if gen is a two_pole
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_two_pole(gen._ptr)

# make def for *args. are there other ways
def make_two_zero(*args):
    """
    return a new two_zero filter; a0*x(n) + a1*x(n-1) + a2*x(n-2)
    """
    if(len(args) == 2):
        return mus_any.from_ptr(cclm.mus_make_two_zero_from_frequency_and_radius(args[0], args[1]))
    elif(len(args) == 3):
        return mus_any.from_ptr(cclm.mus_make_two_zero(args[0], args[1], args[2]))
    else:
        raise RuntimeError("requires either 2 or 3 args but received {len(args)}.")

cpdef cython.double two_zero(gen: mus_any, input: float):
    """
    two zero filter of input.
    """
    return cclm.mus_two_zero(gen._ptr, input)
    
cpdef bint is_two_zero(gen: mus_any):
    """
    returns true if gen is a two_zero
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_two_zero(gen._ptr)
    
    
# ---------------- one_pole_all_pass ---------------- #
cpdef mus_any make_one_pole_all_pass(size: int, coeff: float):
    """
    return a new one_pole all_pass filter size, coeff
    
    """
    return mus_any.from_ptr(cclm.mus_make_one_pole_all_pass(size, coeff))
    
cpdef cython.double one_pole_all_pass(gen: mus_any, insig: cython.double):
    """
    one pole all pass filter of input.
    """
    return cclm.mus_one_pole_all_pass(gen._ptr, insig)
    
cpdef bint is_one_pole_all_pass(gen: mus_any):
    """
    returns true if gen is a one_pole_all_pass
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_one_pole_all_pass(gen._ptr)


# ---------------- formant ---------------- #
cpdef mus_any make_formant(frequency: cython.double, radius: cython.double):
    """return a new formant generator (a resonator). radius sets the pole radius (in terms of the 'unit circle').
        frequency sets the resonance center frequency (hz)."""
    return mus_any.from_ptr(cclm.mus_make_formant(frequency, radius))

cpdef cython.double formant(gen: mus_any, insig: cython.double, radians: Optional[float]=None):
    """
    next sample from resonator generator.
    """
    if radians:
        return cclm.mus_formant_with_frequency(gen._ptr, insig, radians)
    else:
        return cclm.mus_formant(gen._ptr, insig)
    
cpdef bint is_formant(gen: mus_any):
    """
    returns true if gen is a formant
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_formant(gen._ptr)

# ---------------- formant-bank ---------------- #   
cpdef mus_any make_formant_bank(filters: list, amps=None):
    """
    return a new formant-bank generator.
    """
    
    p = list(map(is_formant, filters))
    
    if not all(p):
        raise TypeError(f'filter list contains at least one element that is not a formant.')
        
    cdef double [:] amps_view = None
    
    filt_array = mus_any_array.from_pylist(filters)
    
    if amps is not None:
        if isinstance(amps, list):
            amps = np.array(amps)
        amps_view = amps
        gen = mus_any.from_ptr(cclm.mus_make_formant_bank(len(filters),filt_array.data, &amps_view[0]))
    else: 
        gen = mus_any.from_ptr(cclm.mus_make_formant_bank(len(filters),filt_array.data, NULL))    
    gen.cache_extend([filt_array, amps, filters])
    return gen


cpdef cython.double formant_bank(gen: mus_any, inputs):
    """
    sum a bank of formant generators
    """
    
    cdef double [:] inputs_view = None
    
    if isinstance(inputs, np.ndarray): 
        inputs_view = inputs
        res = cclm.mus_formant_bank_with_inputs(gen._ptr, &inputs_view[0])
        return res
    else:
        res = cclm.mus_formant_bank(gen._ptr, inputs)
    return res
     
cpdef bint is_formant_bank(gen: mus_any):
    """
    returns true if gen is a formant_bank
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_formant_bank(gen._ptr)

# ---------------- firmant ---------------- #
cpdef mus_any make_firmant(frequency: cython.double, radius: cython.double):
    """return a new firmant generator (a resonator).  radius sets the pole radius (in terms of the 'unit circle').
        frequency sets the resonance center frequency (hz)."""
    return mus_any.from_ptr(cclm.mus_make_firmant(frequency, radius))

cpdef firmant(gen: mus_any, insig: cython.double, radians: Optional[cython.double]=None ):
    """
    next sample from resonator generator.
    """
    if radians:
        return cclm.mus_firmant_with_frequency(gen._ptr, insig, radians)
    else: 
        return cclm.mus_firmant(gen._ptr, insig)
            
cpdef is_firmant(gen: mus_any):
    """
    returns true if gen is a firmant
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_firmant(gen._ptr)

# ---------------- filter ---------------- #
cpdef mus_any make_filter(order: int, xcoeffs, ycoeffs):   
    """
    return a new direct form fir/iir filter, coeff args are list/ndarray
    """
    cdef double [:] xcoeffs_view = None
    cdef double [:] ycoeffs_view = None

    if isinstance(xcoeffs, list):
        xcoeffs = np.array(xcoeffs, dtype=np.double)        
        
    if isinstance(ycoeffs, list):
        ycoeffs = np.array(ycoeffs, dtype=np.double)        
       
    xcoeffs_view = xcoeffs
    ycoeffs_view = ycoeffs

    gen =  mus_any.from_ptr(cclm.mus_make_filter(order, &xcoeffs_view[0], &ycoeffs_view[0], NULL))
    
    gen.cache_extend([xcoeffs, ycoeffs])
    return gen
    
cpdef cython.double filter(gen: mus_any, insig: cython.double): # todo : conflicts with buitl in function
    """
    next sample from filter
    """
    return cclm.mus_filter(gen._ptr, insig)
    
cpdef bint is_filter(gen: mus_any):
    """
    returns true if gen is a filter
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_filter(gen._ptr)

# ---------------- fir-filter ---------------- #
cpdef mus_any make_fir_filter(order: int, xcoeffs):
    """
    return a new fir filter, xcoeffs a list/ndarray
    """
    cdef double [:] xcoeffs_view = None

    if isinstance(xcoeffs, list):
        xcoeffs = np.array(xcoeffs, dtype=np.double)        
    
    xcoeffs_view = xcoeffs   

    gen =  mus_any.from_ptr(cclm.mus_make_fir_filter(order, &xcoeffs_view[0], NULL))
    gen.cache_append([xcoeffs])
    
    return gen
    
cpdef cython.double fir_filter(gen: mus_any, insig: cython.double):
    """
    next sample from fir filter
    """
    return cclm.mus_fir_filter(gen._ptr, insig)
    
cpdef bint is_fir_filter(gen: mus_any):
    """
    returns true if gen is a fir_filter
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_fir_filter(gen._ptr)


# ---------------- iir-filter ---------------- #
cpdef mus_any make_iir_filter(order: int, ycoeffs):
    """
    return a new iir filter, ycoeffs a list/ndarray
    """
    cdef double [:] ycoeffs_view = None

    if isinstance(ycoeffs, list):
        ycoeffs = np.array(ycoeffs, dtype=np.double)        
    
    ycoeffs_view = ycoeffs   
    
        
    gen = mus_any.from_ptr(cclm.mus_make_iir_filter(order, &ycoeffs_view[0], NULL))
    gen.cache_append([ycoeffs])
    return gen
    
cpdef cython.double iir_filter(gen: mus_any, insig: cython.double ):
    """
    next sample from iir filter
    """
    return cclm.mus_iir_filter(gen._ptr, insig)
    
cpdef bint is_iir_filter(gen: mus_any):
    """
    returns true if gen is a iir_filter
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_iir_filter(gen._ptr)

cpdef np.ndarray make_fir_coeffs(order: int, envelope):
    cdef double [:] envelope_view = None
        
    if isinstance(envelope, list):
        envelope = np.ndarray(envelope)
    check_ndim(envelope)
    
    envelope_view = envelope
    
    coeffs = np.zeros(order+1, dtype=np.double)
    
    cdef double [:] coeffs_view = coeffs
    
    cclm.mus_make_fir_coeffs(order, &envelope_view[0], &coeffs_view[0])
    return coeffs


# ---------------- delay ---------------- #
cpdef mus_any make_delay(size: int, 
                initial_contents=None, 
                initial_element: Optional[cython.double]=None, 
                max_size:Optional[int]=None,
                interp_type=Interp.NONE):
    """
    return a new delay line of size elements. if the delay length will be changing at run-time, max-size sets its maximum length
    """
    
    cdef double [:] initial_contents_view = None
    
    if not max_size:
        max_size = size
        
    if max_size != size and interp_type == Interp.NONE:
        interp_type = Interp.LAGRANGE #think this is correct from clm2xen.c
        
    if initial_contents is not None:
        if isinstance(initial_contents, list):
            initial_contents = np.array(initial_contents)
        check_ndim(initial_contents)
        initial_contents_view = initial_contents

    elif initial_element:
        initial_contents = np.zeros(max_size, dtype=np.double)
        check_ndim(initial_contents)
        initial_contents.fill(initial_element)
        initial_contents_view = initial_contents

    if initial_contents_view is not None:
        gen = mus_any.from_ptr(cclm.mus_make_delay(size, &initial_contents_view[0], max_size, interp_type))
    else:   
        gen = mus_any.from_ptr(cclm.mus_make_delay(size, NULL, max_size, interp_type))
    
    gen.cache_extend(initial_contents)
    return gen
    
cpdef cython.double delay(gen: mus_any, insig: cython.double, pm: Optional[cython.double]=None):
    """
    delay val according to the delay line's length and pm ('phase-modulation').
    if pm is greater than 0.0, the max-size argument used to create gen should have accommodated its maximum value. 
    """
    if pm:
        return cclm.mus_delay(gen._ptr, insig, pm)
    else: 
        return cclm.mus_delay_unmodulated(gen._ptr, insig)
        
cpdef bint is_delay(gen: mus_any):
    """
    returns true if gen is a delay
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_delay(gen._ptr)

cpdef cython.double tap(gen: mus_any, offset: Optional[cython.double ]=None):
    """
    tap the delay mus_any offset by pm
    """
    if offset:
        return cclm.mus_tap(gen._ptr, offset)
    else:
        return cclm.mus_tap_unmodulated(gen._ptr)
    
cpdef bint is_tap(gen: mus_any):
    """
    returns true if gen is a tap
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_tap(gen._ptr)
    
cpdef cython.double delay_tick(gen: mus_any, insig: cython.double ):
    """
    delay val according to the delay line's length. this merely 'ticks' the delay line forward.
    the argument 'val' is returned.
    """
    return cclm.mus_delay_tick(gen._ptr, insig)


# ---------------- comb ---------------- #
cpdef mus_any make_comb(scaler: Optional[cython.double]=1.0,
                size: Optional[int]=None, 
                initial_contents=None, 
                initial_element: Optional[cython.double]=None, 
                max_size:Optional[int]=None,
                interp_type=Interp.NONE):
    """
    return a new comb filter (a delay line with a scaler on the feedback) of size elements. 
        if the comb length will be changing at run-time, max-size sets its maximum length.
    """                
                
    cdef double [:] initial_contents_view = None
    
    if not max_size:
        max_size = size
        
    if max_size != size and interp_type == Interp.NONE:
        interp_type = Interp.BEZIER #think this is correct from clm2xen.c
    
    if initial_contents is not None:
        if isinstance(initial_contents, list):
            initial_contents = np.array(initial_contents)
        check_ndim(initial_contents)
        initial_contents_view = initial_contents

    elif initial_element:
        initial_contents = np.zeros(max_size)
        check_ndim(initial_contents)
        initial_contents.fill(initial_element)
        initial_contents_view = initial_contents
        
    if initial_contents_view is not None:
        gen = mus_any.from_ptr(cclm.mus_make_comb(scaler, size, &initial_contents_view[0], max_size, interp_type))
    else:
        gen = mus_any.from_ptr(cclm.mus_make_comb(scaler, size, NULL, max_size, interp_type))
    gen.cache_append(initial_contents)
    return gen    
           
cpdef cython.double comb(gen: mus_any, insig: cython.double, pm: Optional[cython.double]=None):
    """
    comb filter val, pm changes the delay length.
    """
    if pm:
        return cclm.mus_comb(gen._ptr, insig, pm)
    else:
        return cclm.mus_comb_unmodulated(gen._ptr, insig)
    
cpdef bint is_comb(gen: mus_any):
    """
    returns true if gen is a comb
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_comb(gen._ptr)    

# ---------------- comb-bank ---------------- #
cpdef mus_any make_comb_bank(combs: list):
    """
    return a new comb-bank generator.
    """
    
    p = list(map(is_comb, combs))
    if not all(p):
        raise TypeError(f'filter list contains at least one element that is not a formant.')

    comb_array = mus_any_array.from_pylist(combs)
    
    gen = mus_any.from_ptr(cclm.mus_make_comb_bank(len(combs), comb_array.data))
    gen.cache_extend([comb_array, combs])
    return gen

cpdef cython.double comb_bank(gen: mus_any, insig: cython.double):
    """
    sum an array of comb filters.
    """
    return cclm.mus_comb_bank(gen._ptr, insig)
    
cpdef bint is_comb_bank(gen: mus_any):
    """
    returns true if gen is a comb_bank
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_comb_bank(gen._ptr)


# ---------------- filtered-comb ---------------- #
cpdef mus_any make_filtered_comb(scaler:  cython.double,
                size: int, 
                filter: mus_any, 
                initial_contents=None, 
                initial_element: Optional[ cython.double]=0.0, 
                max_size:Optional[int]=None,
                interp_type=Interp.NONE):
                
    """
    return a new filtered comb filter (a delay line with a scaler and a filter on the feedback) of size elements.
    if the comb length will be changing at run-time, max-size sets its maximum length.
    """

    cdef double [:] initial_contents_view = None
    
    if not max_size:
        max_size = size
        
    if max_size != size and interp_type == Interp.NONE:
        interp_type = Interp.BEZIER #think this is correct from clm2xen.c
    
    if initial_contents is not None:
        if isinstance(initial_contents, list):
            initial_contents = np.array(initial_contents)
        initial_contents_view = initial_contents

    elif initial_element:
        initial_contents = np.zeros(max_size)
        check_ndim(initial_contents)
        initial_contents.fill(initial_element)
        initial_contents_view = initial_contents
    
    if initial_contents_view is not None:
        gen = mus_any.from_ptr(cclm.mus_make_filtered_comb(scaler, int(size), &initial_contents_view[0], int(max_size), interp_type, filter._ptr))
    else:
        gen = mus_any.from_ptr(cclm.mus_make_filtered_comb(scaler, int(size), NULL, int(max_size), interp_type, filter._ptr))
    gen.cache_append(initial_contents)
    return gen    
        
cpdef cython.double filtered_comb(gen: mus_any, insig: cython.double, pm: Optional[ cython.double]=None):
    """
    filtered comb filter val, pm changes the delay length.
    """
    if pm:
        return cclm.mus_filtered_comb(gen._ptr, insig, pm)
    else:
        return cclm.mus_filtered_comb_unmodulated(gen._ptr, insig)
        
cpdef bint is_filtered_comb(gen: mus_any):
    """
    returns true if gen is a filtered_comb
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_filtered_comb(gen._ptr)
    
# ---------------- filtered-comb-bank ---------------- #      
cpdef mus_any make_filtered_comb_bank(fcombs: list):
    """
    return a new filtered_comb-bank generator.
    """
    p = list(map(is_formant, fcombs))
    if not all(p):
        raise TypeError(f'filter list contains at least one element that is not a filtered_comb.')

    fcomb_array = mus_any_array.from_pylist(fcombs)
    
    gen =  mus_any.from_ptr(cclm.mus_make_filtered_comb_bank(len(fcombs), fcomb_array.data))
    gen.cache_extend([fcomb_array, fcombs])
    return gen

cpdef cython.double filtered_comb_bank(gen: mus_any):
    """
    sum an array of filtered_comb filters.
    """
    return cclm.mus_filtered_comb_bank(gen._ptr, input)
    
cpdef bint is_filtered_comb_bank(gen: mus_any):
    """
    returns true if gen is a filtered_comb_bank
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_filtered_comb_bank(gen._ptr)

# ---------------- notch ---------------- #
cpdef mus_any make_notch(scaler: Optional[cython.double]=1.0,
                size: Optional[int]=None, 
                initial_contents=None, 
                initial_element: Optional[cython.double]=0.0, 
                max_size:Optional[int]=None,
                interp_type=Interp.NONE):

    """
    return a new notch filter (a delay line with a scaler on the feedforward) of size elements.
    if the notch length will be changing at run-time, max-size sets its maximum length
    """
    
    cdef double [:] initial_contents_view = None
    
    if not max_size:
        max_size = size
        
    if max_size != size and interp_type == Interp.NONE:
        interp_type = Interp.BEZIER #think this is correct from clm2xen.c
     
    
    if initial_contents is not None:
        if isinstance(initial_contents, list):
            initial_contents = np.array(initial_contents)
        check_ndim(initial_contents)
        initial_contents_view = initial_contents
        
    elif initial_element:
        initial_contents = np.zeros(max_size, dtype=np.double)
        check_ndim(initial_contents)
        initial_contents.fill(initial_element)
        initial_contents_view = initial_contents
    
    if initial_contents_view is not None:
        gen = mus_any.from_ptr(cclm.mus_make_notch(scaler, size, &initial_contents_view[0], max_size, interp_type))
    else:
        gen = mus_any.from_ptr(cclm.mus_make_notch(scaler, size, NULL, max_size, interp_type))
    gen.cache_append(initial_contents)
    return gen    

cpdef cython.double notch(gen: mus_any, insig: cython.double, pm: Optional[cython.double]=None):
    """
    notch filter val, pm changes the delay length.
    """
    if pm:
        return cclm.mus_notch(gen._ptr, input, pm)
    else:
        return cclm.mus_notch_unmodulated(gen._ptr, insig)
    
cpdef bint is_notch(gen: mus_any):
    """
    returns true if gen is a notch
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_notch(gen._ptr)

 
# ---------------- all-pass ---------------- #
cpdef mus_any make_all_pass(feedback: float, feedforward: float, size: int, initial_contents: Optional[np.ndarray] = None, initial_element: Optional[float] = 0.0,  max_size:Optional[int] = None, interp_type: Optional[Interp] = Interp.NONE):

    """
    return a new allpass filter (a delay line with a scalers on both the feedback and the feedforward). if length will be changing at run-time, max-size sets its maximum length.
    
    :param feedback: scaler on feedback

    :param feedforward: scaler on input

    :param size: length \in samples of the delay line

    :param initial_contents: delay line's initial values, defaults to None

    :param initial_element: delay line's initial element, defaults to 0.0

    :param max_size: maximum delay size in case the delay changes, defaults to None

    :param interp_type: interpolation type, defaults to nterp.None
    
    :return: a new allpass filter
    
    :rtype: mus_any
     
    """

    cdef double [:] initial_contents_view = None
    
    if not max_size:
        max_size = size
        
    if max_size != size and interp_type == Interp.NONE:
        interp_type = Interp.HERMITE #think this is correct from clm2xen.c
     
    
    if initial_contents is not None:
        if isinstance(initial_contents, list):
            initial_contents = np.array(initial_contents)
        check_ndim(initial_contents)
        initial_contents_view = initial_contents

    elif initial_element:
        initial_contents = np.zeros(max_size, dtype=np.double)
        check_ndim(initial_contents)
        initial_contents.fill(initial_element)
        initial_contents_view = initial_contents
            
    if initial_contents_view is not None:
        gen = mus_any.from_ptr(cclm.mus_make_all_pass(feedback,feedforward, size, &initial_contents_view[0], max_size, interp_type))
    else:
        gen = mus_any.from_ptr(cclm.mus_make_all_pass(feedback,feedforward, size,NULL, max_size, interp_type))
    gen.cache_append(initial_contents)
    return gen
    
cpdef cython.double all_pass(gen: mus_any, insig: float, pm: Optional[float]=None):
    """
    all-pass filter insig value, pm changes the delay length.
    """
    if pm:
        return cclm.mus_all_pass(gen._ptr, insig, pm)
    else:
        return cclm.mus_all_pass_unmodulated(gen._ptr, insig)
    
cpdef is_all_pass(gen: mus_any):
    """
    returns true if gen is a all_pass
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_all_pass(gen._ptr)
    

# ---------------- all-pass-bank ---------------- #
cpdef mus_any make_all_pass_bank(all_passes: list):
    """
    return a new all_pass-bank generator.
    """
    p = list(map(is_all_pass, all_passes))
    if not all(p):
        raise TypeError(f'allpass list contains at least one element that is not a all_pass.')
        
    all_passes_array = mus_any_array.from_pylist(all_passes)
    gen =  mus_any.from_ptr(cclm.mus_make_all_pass_bank(len(all_passes), all_passes_array.data))
    gen.cache_extend([all_passes_array, all_passes])
    return gen

cpdef cython.double all_pass_bank(gen: mus_any, insig: cython.double):
    """
    sum an array of all_pass filters.
    """
    return cclm.mus_all_pass_bank(gen._ptr, insig)
    
cpdef bint is_all_pass_bank(gen: mus_any):
    """
    returns true if gen is a all_pass_bank
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_all_pass_bank(gen._ptr)
        
# ---------------- moving-average ---------------- #
cpdef mus_any make_moving_average(size: int, initial_contents=None, initial_element: Optional[cython.double]=0.0):
    """
    return a new moving_average generator. 
    """

    
    cdef double [:] initial_contents_view = None
    
    if initial_contents is not None:
        if isinstance(initial_contents, list):
            initial_contents = np.array(initial_contents)
        check_ndim(initial_contents)
        initial_contents_view = initial_contents
    else:
        initial_contents = np.zeros(size, dtype=np.double)
        check_ndim(initial_contents)
        initial_contents.fill(initial_element)
        initial_contents_view = initial_contents
    
    if initial_contents_view is not None:    
        gen = mus_any.from_ptr(cclm.mus_make_moving_average(size, &initial_contents_view[0]))

    gen.cache_append(initial_contents)
    return gen
        
        
cpdef cython.double moving_average(gen: mus_any, input: cython.double):
    """
    moving window average.
    """
    return cclm.mus_moving_average(gen._ptr, input)
    
cpdef bint is_moving_average(gen: mus_any):
    """
    returns true if gen is a moving_average
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_moving_average(gen._ptr)

# ---------------- moving-max ---------------- #
cpdef mus_any make_moving_max(size: int, 
                initial_contents=None, 
                initial_element: Optional[cython.double]=0.0):
                
    """
    return a new moving-max generator.
    """                
    
    cdef double [:] initial_contents_view = None
    
    if initial_contents is not None:
        if isinstance(initial_contents, list):
            initial_contents = np.array(initial_contents)
        check_ndim(initial_contents)
        initial_contents_view = initial_contents

    elif initial_element:
        initial_contents = np.zeros(size, dtype=np.double)
        check_ndim(initial_contents)
        initial_contents.fill(initial_element)
        initial_contents_view = initial_contents
    if initial_contents_view is not None:
        gen = mus_any.from_ptr(cclm.mus_make_moving_max(size, &initial_contents_view[0]))
        gen.cache_append(initial_contents)
    else:
        gen = mus_any.from_ptr(cclm.mus_make_moving_max(size, NULL))
    return gen
    
cpdef cython.double moving_max(gen: mus_any, insig: float):
    """
    moving window max.
    """
    return cclm.mus_moving_max(gen._ptr, insig)
    
cpdef bint is_moving_max(gen: mus_any):
    """
    returns true if gen is a moving_max
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_moving_max(gen._ptr)
    
# ---------------- moving-norm ---------------- #
cpdef mus_any make_moving_norm(size: int, initial_contents=None, scaler: Optional[float]=1.):
    """
    return a new moving-norm generator.
    """
    cdef double [:] initial_contents_view = None
    
        
    if initial_contents is not None:
        if isinstance(initial_contents, list):
            initial_contents = np.array(initial_contents)
        check_ndim(initial_contents)
        initial_contents_view = initial_contents
        
    if initial_contents_view is not None:
        gen = mus_any.from_ptr(cclm.mus_make_moving_norm(size, &initial_contents_view[0], scaler))
        gen.cache_append(initial_contents)
    else:
        gen = mus_any.from_ptr(cclm.mus_make_moving_norm(size, NULL, scaler))
    
    return gen
    
cpdef cython.double moving_norm(gen: mus_any, insig: cython.double):
    """
    moving window norm.
    """
    return cclm.mus_moving_norm(gen._ptr, insig)
    
cpdef is_moving_norm(gen: mus_any):
    """
    returns true if gen is a moving_norm
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_moving_norm(gen._ptr)
    
    
# ---------------- asymmetric-fm ---------------- #
cpdef mus_any make_asymmetric_fm(frequency: cython.double, initial_phase: Optional[cython.double]=0.0, r: Optional[cython.double]=1.0, ratio: Optional[cython.double]=1.):
    """
    return a new asymmetric_fm generator.
    """
    return mus_any.from_ptr(cclm.mus_make_asymmetric_fm(frequency, initial_phase, r, ratio))
    
cpdef cython.double asymmetric_fm(gen: mus_any, insig: cython.double, fm: Optional[cython.double]=None):
    """
    next sample from asymmetric fm generator.
    """
    if fm:
        return cclm.mus_asymmetric_fm(gen._ptr, insig, fm)
    else:
        return cclm.mus_asymmetric_fm_unmodulated(gen._ptr, insig)
    
cpdef bint is_asymmetric_fm(gen: mus_any):
    """
    returns true if gen is a asymmetric_fm
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_asymmetric_fm(gen._ptr)
    
# ---------------- file-to-sample ---------------- #
cpdef mus_any make_file2sample(filename, buffer_size: Optional[int]=None):
    """
    return an input generator reading 'filename' (a sound file)
    """
    buffer_size = buffer_size or CLM.buffer_size
    return mus_any.from_ptr(cclm.mus_make_file_to_sample_with_buffer_size(filename, buffer_size))
    
cpdef cython.double file2sample(gen: mus_any, loc: int, chan: Optional[int]=0):
    """
    sample value in sound file read by 'obj' in channel chan at frample
    """
    return cclm.mus_file_to_sample(gen._ptr, loc, chan)
    
cpdef bint is_file2sample(gen: mus_any):
    """
    returns true if gen is a file2sample
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_file_to_sample(gen._ptr)
    
# ---------------- sample-to-file ---------------- #
cpdef mus_any make_sample2file(filename, chans: Optional[int]=1, sample_type: Optional[sample]=None, header_type: Optional[header]=None, comment: Optional[str]=None):
    """return an output generator writing the sound file 'filename' which is set up to have chans' channels of 'sample_type' 
        samples with a header of 'header_type'.  the latter should be sndlib identifiers:"""
    sample_type = sample_type or CLM.sample_type
    header_type = header_type or CLM.header_type
    if comment is None:
        return mus_any.from_ptr(cclm.mus_make_sample_to_file_with_comment(filename, chans, sample_type, header_type, NULL))
    else:
        return mus_any.from_ptr(cclm.mus_make_sample_to_file_with_comment(filename, chans, sample_type, header_type, comment))

cpdef cython.double sample2file(gen: mus_any, samp: int, chan:int , val: cython.double):
    """
    add val to the output stream handled by the output generator 'obj', in channel 'chan' at frample 'samp'
    """
    return cclm.mus_sample_to_file(gen._ptr, samp, chan, val)
    
cpdef bint is_sample2file(gen: mus_any):
    """
    returns true if gen is a sample2file
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_sample_to_file(gen._ptr)
    
cpdef mus_any continue_sample2file(name: str):
    return mus_any.from_ptr(cclm.mus_continue_sample_to_file(name))
    
    
# ---------------- file-to-frample ---------------- #
cpdef mus_any make_file2frample(filename, buffer_size: Optional[int]=None):
    """
    return an input generator reading 'filename' (a sound file)
    """
    buffer_size = buffer_size or CLM.buffer_size
    return  mus_any.from_ptr(cclm.mus_make_file_to_frample_with_buffer_size(filename, buffer_size))
    
cpdef file2frample(gen: mus_any, loc: int):
    """
    frample of samples at frample 'samp' in sound file read by 'obj'
    """
    outf = np.zeros(cclm.mus_channels(gen._ptr), dtype=np.double)
    cdef double [:] outf_view = outf
    cclm.mus_file_to_frample(gen._ptr, loc, &outf_view[0])
    return outf
    
cpdef is_file2frample(gen: mus_any):
    """
    returns true if gen is a file2frample
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_file_to_frample(gen._ptr)
    
    
# ---------------- frample-to-file ---------------- #
cpdef mus_any make_frample2file(filename, chans: Optional[int]=1, sample_type: Optional[sample]=None, header_type: Optional[header]=None, comment: Optional[str]=None):
    """return an output generator writing the sound file 'filename' which is set up to have 'chans' channels of 'sample_type' 
        samples with a header of 'header_type'.  the latter should be sndlib identifiers."""
    sample_type = sample_type or CLM.sample_type
    header_type = header_type or CLM.header_type
    if comment:
        return mus_any.from_ptr(cclm.mus_make_frample_to_file_with_comment(filename, chans, sample_type, header_type, comment))
    else:
        return mus_any.from_ptr(cclm.mus_make_frample_to_file_with_comment(filename, chans, sample_type, header_type, NULL))

cpdef cython.double frample2file(gen: mus_any, samp: int, chan:int, val):
    """
    add frample 'val' to the output stream handled by the output generator 'obj' at frample 'samp'
    """
    cdef double [:] val_view = None
    
    if isinstance(val, list):
        val = np.srray(val)
        
    val_view = val
    
    frample = val_view
    cclm.mus_frample_to_file(gen._ptr, samp, &frample[0])
    return val
    
cpdef bint is_frample2file(gen: mus_any):
    """
    returns true if gen is a frample2file
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_frample_to_file(gen._ptr)
    

cpdef mus_any continue_frample2file(name: str):
    return mus_any.from_ptr(cclm.mus_continue_frample_to_file(name))


# ---------------- readin ---------------- #
cpdef mus_any make_readin(filename: str, chan: int=0, start: int=0, direction: Optional[int]=1, buffer_size: Optional[int]=None):
    """
    return a new readin (file input) generator reading the sound file 'file' starting at frample 'start' in channel 'channel' and reading forward if 'direction' is not -1
    """
    buffer_size = buffer_size or CLM.buffer_size
    return mus_any.from_ptr(cclm.mus_make_readin_with_buffer_size(filename, chan, start, direction, buffer_size))
    
cpdef cython.double readin(gen: mus_any):
    """
    next sample from readin generator (a sound file reader)
    """
    return cclm.mus_readin(gen._ptr)
    
cpdef is_readin(gen: mus_any):
    """
    returns true if gen is a readin
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_readin(gen._ptr)
      
          
# ---------------- src ---------------- #
def make_src(inp , srate: Optional[cython.double]=1.0, width: Optional[int]=10):
    """return a new sampling-rate conversion generator (using 'warped sinc interpolation'). 'srate' is the ratio between the new rate and the old. 'width' is the sine 
        width (effectively the steepness of the low-pass filter), normally between 10 and 100. 'input' if given is an open file stream."""
        
    cdef cclm.input_cb cy_inp_f_ptr 
    
    if(isinstance(inp, mus_any)):
        res = mus_any.from_ptr(cclm.mus_make_src(<cclm.input_cb>input_callback_func, srate, width, <void*>((<mus_any>inp)._ptr)))
        res._inputcallback = <cclm.input_cb>input_callback_func
        return res
        
    if not callable(inp):
        raise TypeError(f"input needs to be a clm gen or function not a {type(inp)}")

    @INPUTCALLBACK
    def inp_f(gen, d):
        return inp(d)
        
    cy_inp_f_ptr = (<cclm.input_cb*><size_t>ctypes.addressof(inp_f))[0]
    res = mus_any.from_ptr(cclm.mus_make_src(cy_inp_f_ptr, srate, width, NULL))
    res._inputcallback = cy_inp_f_ptr
    res.cache_append(inp_f)
    
    return res
  
cpdef cython.double src(gen: mus_any, sr_change: Optional[cython.double]=0.0):
    """next sampling rate conversion sample. 'pm' can be used to change the sampling rate on a sample-by-sample basis.  
        'input-function' is a function of one argument (the current input direction, normally ignored) that is called internally 
        whenever a new sample of input data is needed.  if the associated make_src included an 'input' argument, input-function is ignored."""
    if gen._inputcallback:
        return cclm.mus_src(gen._ptr, sr_change, <cclm.input_cb>gen._inputcallback)
    else:
        return 0.0
    
cpdef bint is_src(gen: mus_any):
    """
    returns true if gen is a src
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_src(gen._ptr)
 

# ---------------- convolve ---------------- #

def make_convolve(inp, filt, fft_size: Optional[int]=512, filter_size: Optional[int]=None ):
    """
    return a new convolution generator which convolves its input with the impulse response 'filter'.
    """
    
    cdef cclm.input_cb cy_input_f_ptr 
    
    cdef double [:] filt_view
    

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
    
    if isinstance(filt, list):       
        filt = np.array(filt)
        
    check_ndim(filt)
    
    filt_view = filt
    
    if(isinstance(inp, mus_any)):
        res = mus_any.from_ptr(cclm.mus_make_convolve(<cclm.input_cb>input_callback_func, &filt_view[0], fft_size, filter_size, <void*>(<mus_any>inp)._ptr))
        res._inputcallback = <cclm.input_cb>input_callback_func
        res.cache_append(filt)     
        return res
    
    if not callable(inp):
        raise TypeError(f"input needs to be a clm gen or function not a {type(inp)}")

    @INPUTCALLBACK
    def inp_f(gen, d):
        return inp(d)
        
    inp_f = INPUTCALLBACK(inp)
    cy_input_f_ptr = (<cclm.input_cb*><size_t>ctypes.addressof(inp_f))[0]
    res = mus_any.from_ptr(cclm.mus_make_convolve(cy_input_f_ptr, &filt_view[0], fft_size, filter_size, NULL))
    res._inputcallback = cy_input_f_ptr
    res.cache_append(inp_f)     

    return res

    
cpdef cython.double convolve(gen: mus_any):
    """
    next sample from convolution generator
    """
    return cclm.mus_convolve(gen._ptr, gen._inputcallback)  
    
cpdef bint is_convolve(gen: mus_any):
    """
    returns true if gen is a convolve
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_convolve(gen._ptr)
    

# --------------- granulate ---------------- 
def make_granulate(inp, 
                    expansion: Optional[cython.double]=1.0, 
                    length: Optional[cython.double]=.15,
                    scaler: Optional[cython.double]=.6,
                    hop: Optional[cython.double]=.05,
                    ramp: Optional[cython.double]=.4,
                    jitter: Optional[cython.double]=0.0,
                    max_size: Optional[int]=0,
                    edit=None):
                    
    """return a new granular synthesis generator.  'length' is the grain length (seconds), 'expansion' is the ratio in timing
         between the new and old (expansion > 1.0 slows things down), 'scaler' scales the grains
        to avoid overflows, 'hop' is the spacing (seconds) between successive grains upon output.
        'jitter' controls the randomness in that spacing, 'input' can be a file pointer. 'edit' can
        be a function of one arg, the current granulate generator.  it is called just before
        a grain is added into the output buffer. the current grain is accessible via mus_data.
        the edit function, if any, should return the length in samples of the grain, or 0.
    """
    
    
    cdef cclm.input_cb cy_input_f_ptr
    cdef cclm.edit_cb cy_edit_f_ptr 

    if(isinstance(inp, mus_any) and edit is None):
        res = mus_any.from_ptr(cclm.mus_make_granulate(<cclm.input_cb>input_callback_func, expansion, length, scaler, hop, ramp, jitter, max_size, NULL, <void*>(<mus_any>inp)._ptr))
        res._inputcallback = <cclm.input_cb>input_callback_func
        return res
    
    if not callable(inp):
        raise TypeError(f"input needs to be a clm gen or function not a {type(inp)}")
        
    @INPUTCALLBACK
    def inp_f(gen, d):
        return inp(d)
     
   
    cy_inp_f_ptr = (<cclm.input_cb*><size_t>ctypes.addressof(inp_f))[0]
    res = mus_any.from_ptr(cclm.mus_make_granulate(<cclm.input_cb>cy_inp_f_ptr, expansion, length, scaler, hop, ramp, jitter, max_size, NULL, NULL))
    res._inputcallback = cy_inp_f_ptr
    res.cache_append(inp_f)     
    
    if(edit is None):
        return res

    @EDITCALLBACK
    def edit_f(gen):
        return edit(res)

    cy_edit_f_ptr = (<cclm.edit_cb*><size_t>ctypes.addressof(edit_f))[0]
    cclm.mus_granulate_set_edit_function(res._ptr, cy_edit_f_ptr)
    res._editcallback  = cy_edit_f_ptr
    res.cache_append(edit_f)

    return res
    
#todo: mus_granulate_grain_max_length
cpdef cython.double granulate(gen: mus_any):
    """
    next sample from granular synthesis generator
    """
    if gen._editcallback is not NULL:
        return cclm.mus_granulate_with_editor(gen._ptr, <cclm.input_cb>gen._inputcallback, <cclm.edit_cb>gen._editcallback)
    else:      
        return cclm.mus_granulate(gen._ptr, <cclm.input_cb>gen._inputcallback)
    

cpdef bint is_granulate(e: mus_any):
    """
    returns true if gen is a granulate
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_granulate(e._ptr)

#--------------- phase-vocoder ----------------#
def make_phase_vocoder(inp, 
                        fft_size: oOptionalptional[int]=512, 
                        overlap: Optional[int]=4, 
                        interp: Optional[int]=128, 
                        pitch: Optional[cython.double]=1.0, 
                        analyze=None, 
                        edit=None, 
                        synthesize=None):
                        
    """return a new phase-vocoder generator; input is the input function (it can be set at run-time), analyze, edit,
        and synthesize are either None or functions that replace the default innards of the generator, fft_size, overlap
        and interp set the fft_size, the amount of overlap between ffts, and the time between new analysis calls.
        'analyze', if given, takes 2 args, the generator and the input function; if it returns true, the default analysis 
        code is also called.  'edit', if given, takes 1 arg, the generator; if it returns true, the default edit code 
        is run.  'synthesize' is a function of 1 arg, the generator; it is called to get the current vocoder 
        output."""

    cdef cclm.input_cb cy_inp_f_ptr
    cdef cclm.edit_cb cy_edit_f_ptr 
    cdef cclm.analyze_cb cy_analyze_f_ptr
    cdef cclm.synthesize_cb cy_synthesize_f_ptr 
   
    if fft_size <= 1:
        raise ValueError(f'fft_size must be a positive number greater than 1 not {fft_size}')
        
    if fft_size > csndlib.mus_max_malloc():
         raise ValueError(f'fft_size too large. cannot allocate {fft_size} size fft')
        
    if not is_power_of_2(fft_size):
        raise ValueError(f'fft_size must be power of 2 not {fft_size}')
        
    if(isinstance(inp, mus_any)):
        res = mus_any.from_ptr(cclm.mus_make_phase_vocoder(<cclm.input_cb>input_callback_func, fft_size, overlap, interp, pitch, NULL, NULL, NULL, <void*>(<mus_any>inp)._ptr))
        res._inputcallback = <cclm.input_cb>input_callback_func
        res.cache_append(inp)
    elif callable(inp):
        @INPUTCALLBACK
        def inp_f(gen, d):
            return inp(d)
        cy_inp_f_ptr = (<cclm.input_cb*><size_t>ctypes.addressof(inp_f))[0]
        res = mus_any.from_ptr(cclm.mus_make_phase_vocoder(cy_inp_f_ptr, fft_size, overlap, interp, pitch, NULL, NULL, NULL, NULL))
        res._inputcallback = cy_inp_f_ptr
        res.cache_append(inp_f)    
        
    else:
        raise TypeError(f"input needs to be a clm gen or a callable not a {type(inp)}")  
    
    if(edit is not None):
        @EDITCALLBACK
        def edit_f(gen):
            return edit(res)
        cy_edit_f_ptr = (<cclm.edit_cb*><size_t>ctypes.addressof(edit_f))[0]
        res._editcallback  = cy_edit_f_ptr
        res.cache_append(edit_f)    

    if (analyze is not None):
        @ANALYZECALLBACK
        def analyze_f(gen, func):
            return analyze(res, inp)
        cy_analyze_f_ptr = (<cclm.analyze_cb*><size_t>ctypes.addressof(analyze_f))[0]
        res._analyzecallback  = cy_analyze_f_ptr
        res.cache_append(analyze_f)

    if (synthesize is not None):
        @SYNTHESIZECALLBACK
        def synthesize_f(gen):
            return synthesize(res)
        cy_synthesize_f_ptr = (<cclm.synthesize_cb*><size_t>ctypes.addressof(synthesize_f))[0]
        res._synthesizecallback = cy_synthesize_f_ptr
        res.cache_append(synthesize_f)
        
    return res


cpdef cython.double phase_vocoder(gen: mus_any):
    """
    next phase vocoder value
    """
    if gen._analyzecallback or gen._synthesizecallback or gen._editcallback :
        return cclm.mus_phase_vocoder_with_editors(gen._ptr, gen._inputcallback, gen._analyzecallback, gen._editcallback, gen._synthesizecallback)
    else:
        return cclm.mus_phase_vocoder(gen._ptr, gen._inputcallback)
    
cpdef bint is_phase_vocoder(gen: mus_any):
    """
    returns true if gen is a phase_vocoder
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_phase_vocoder(gen._ptr)


cpdef np.ndarray phase_vocoder_amp_increments(gen: mus_any):
    """
    returns a ndarray containing the current output sinusoid amplitude increments per sample
    """    
    return gen._pv_amp_increments
    
cpdef np.ndarray phase_vocoder_amps(gen: mus_any):
    """
    returns a ndarray containing the current output sinusoid amplitudes
    """
    return gen._pv_amps
    
cpdef  np.ndarray phase_vocoder_freqs(gen: mus_any):
    """
    returns a ndarray containing the current output sinusoid frequencies
    """
    return gen._pv_freqs
    
cpdef  np.ndarray phase_vocoder_phases(gen: mus_any):
    """
    returns a ndarray containing the current output sinusoid phases
    """
    return gen._pv_phases
    
cpdef  np.ndarray phase_vocoder_phase_increments(gen: mus_any):
    """
    returns a ndarray containing the current output sinusoid phase increments
    """
    return gen._pv_phase_increments
    
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
    """
    input stream sample at frample in channel chan
    """
    if is_list_or_ndarray(input):
        return inp[channel][loc]
    elif isinstance(inp, types.generatortype):
        return next(inp)
    elif callable(inp):
        return inp(loc, channel)
    else:
        ipt = <mus_any>inp
        return cclm.mus_in_any(loc, channel, ipt._ptr)

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
    
    """
    return a new generator for signal placement in n channels.  channel 0 corresponds to 0 degrees.
    """
    
    cdef cclm.detour_cb cy_detour_f_ptr
    
    if not output:
        output = CLM.output  #todo : check if this exists
    
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
        
    # todo: what if revout is not an iterable? while possible not going to deal with it right now :)   
    elif is_list_or_ndarray(output):
        if not reverb_channels:
            reverb_channels = 0
            
        res = mus_any.from_ptr(cclm.mus_make_locsig(degree, distance, reverb, channels, NULL, reverb_channels, NULL, interp_type))
        cclm.mus_locsig_set_detour(res._ptr, <cclm.detour_cb>locsig_detour_callback_func)

        return res
            
    else:
        raise TypeError(f"output needs to be a clm gen or np.array not a {type(output)}")  
        
cpdef void locsig(gen: mus_any, loc: int, val: cython.double):
    """
    locsig 'gen' channel 'chan' scaler
    """
    cclm.mus_locsig(gen._ptr, loc, val)
    
cpdef is_locsig(gen: mus_any):
    """
    returns true if gen is a locsig
    :param gen: gen
    :return: result
    :rtype: bool
    """
    return cclm.mus_is_locsig(gen._ptr)
    
cpdef cython.double locsig_ref(gen: mus_any, chan: int):
    """
    locsig 'gen' channel 'chan' scaler
    """
    return cclm.mus_locsig_ref(gen._ptr, chan)
    
cpdef cython.double locsig_set(gen: mus_any, chan: int, val:cython.double):
    """
    set the locsig generator's channel 'chan' scaler to 'val'
    """
    return cclm.mus_locsig_set(gen._ptr, chan, val)
    
cpdef cython.double locsig_reverb_ref(gen: mus_any, chan: int):
    """
     locsig reverb channel 'chan' scaler
    """
    return cclm.mus_locsig_reverb_ref(gen._ptr, chan)

cpdef cython.double locsig_reverb_set(gen: mus_any, chan: int, val: cython.double):
    """
    set the locsig reverb channel 'chan' scaler to 'val
    """
    return cclm.mus_locsig_reverb_set(gen._ptr, chan, val)
    
cpdef void move_locsig(gen: mus_any, degree: cython.double, distance: cython.double):
    """
    move locsig gen to reflect degree and distance
    """
    cclm.mus_move_locsig(gen._ptr, degree, distance)
    
     
# added some options . todo: what about sample rate conversion    
cpdef convolve_files(file1: str, file2: str, maxamp: Optional[cython.double]=1., outputfile='test.aif', sample_type=CLM.sample_type, header_type=CLM.header_type):
    if header_type == header.next:
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
#  todo : look at allowing something like np.zeroes((8000), dtype=np.double)
# to just be treated as mono sound buffer 
# the python soundfile library does this differently 
# and would use 
# arr = np.zeros((8000,1), dtype=np.double))
# this seems less intuitive to me 
# very issue to translate with simple np.transpose()


# cpdef sndinfo(filename):
#     """returns a dictionary of info about a sound file including write date (data), sample rate (srate),
#     channels (chans), length in samples (samples), length in second (length), comment (comment), and loop information (loopinfo)"""
#     date = csndlib.mus_sound_write_date(filename)
#     srate = csndlib.mus_sound_srate(filename)
#     chans = csndlib.mus_sound_chans(filename)
#     samples = csndlib.mus_sound_samples(filename)
#     comment = csndlib.mus_sound_comment(filename) 
#     length = samples / (chans * srate)
# 
#     header_type = header(mus_sound_header_type(filename))
#     sample_type = sample(mus_sound_sample_type(filename))
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
#     """returns a dictionary of info about a sound file including write date (data), sample rate (srate),
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
# cpdef np.ndarray file2array(filename: str, channel: Optional[int]=0, beg: Optional[int]=None, dur: Optional[int]=None):
#     """
#     return an ndarray with samples from file
#     """
#     length = dur or csndlib.mus_sound_framples(filename)
#     chans = csndlib.mus_sound_chans(filename)
#     srate = csndlib.mus_sound_srate(filename)
#     bg = beg or 0
#     out = np.zeros(length, dtype=np.double)
#     
#     cdef double [:] out_view = None
#     out_view = out
# 
#     csndlib.mus_file_to_array(filename,channel, bg, length, &out_view[0])
#     return out
#     
# def channel2array(filename: str, channel: Optional[int]=0, beg: Optional[int]=None, dur: Optional[int]=None): 
#     length = dur or mus_sound_framples(filename)
#     srate = mus_sound_srate(filename)
#     bg = beg or 0
#     out = np.zeros((1, length), dtype=np.double)
#     mus_file_to_array(filename,channel, bg, length, out[0].ctypes.data_as(ctypes.pointer(ctypes.c_double)))
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
# # todo: maybe add an exception that have to use keyword args
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
    g =  partial(make_generator, **slots)
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
    
def sndplay(file):
    subprocess.run([CLM.player,file])
