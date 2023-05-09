import subprocess
import time
from typing import Optional
from functools import singledispatch
import numpy as np
import numpy.typing as npt
from .sndlib import *
from .enums import *
from .mus_any_pointer import *
# NOTE from what I understand about numpy.ndarray.ctypes 
# it says that when data_as is used that it keeps a reference to 
# the original array. that should mean as long as I cache the
# result of data_as I should not need to cache the original 
# numpy array. right?

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
    
def fft(rdat, idat, fftsize: int, sign: int):
    """Return the fft of rl and im which contain the real and imaginary parts of the data; len should be a power of 2, dir = 1 for fft, -1 for inverse-fft"""    
    rdat_ptr = get_array_ptr(rdat)
    idat_ptr = get_array_ptr(idat)    
    return mus_fft(rdat_ptr, idat_ptr, fftsize, sign)

def make_fft_window(window_type: int, size: int, beta: Optional[float]=0.0, alpha: Optional[float]=0.0):
    """fft data window (a ndarray). type is one of the sndlib fft window identifiers such as Window.KAISER, beta is the window family parameter, if any."""
    win = np.zeros(size, dtype=np.double)
    mus_make_fft_window_with_window(window_type, size, beta, alpha, win.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
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
