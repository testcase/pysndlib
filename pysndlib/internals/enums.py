from enum import Enum, IntEnum
from .sndlib import *

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