cimport cython
import numpy as np
cimport numpy as np
cimport pysndlib.cclm as cclm
cimport pysndlib.csndlib as csndlib
cimport pysndlib.sndlib as sndlib

np.import_array()
DTYPE = np.double

# --------------- clm enums ---------------- #
cpdef enum Interp:
    """
    various interpolation types
    """
    NONE, LINEAR,SINUSOIDAL, ALL_PASS, LAGRANGE, BEZIER, HERMITE

cpdef enum Window:
    """
    many useful windows
    """
    RECTANGULAR, HANN, WELCH, PARZEN, BARTLETT, HAMMING, BLACKMAN2, BLACKMAN3, BLACKMAN4, EXPONENTIAL, RIEMANN, KAISER, CAUCHY, POISSON, GAUSSIAN, TUKEY, DOLPH_CHEBYSHEV, HANN_POISSON, CONNES, SAMARAKI, ULTRASPHERICAL, BARTLETT_HANN, BOHMAN, FLAT_TOP, BLACKMAN5, BLACKMAN6, BLACKMAN7, BLACKMAN8, BLACKMAN9, BLACKMAN10, RV2, RV3, RV4, MLT_SINE, PAPOULIS, DPSS, SINC,

cpdef enum Spectrum:
    """
    types of normalizations when using the spectrum function. The results are in dB if IN_DB, or linear and normalized to 1.0 NORMALIZED, or linear unnormalized RAW
    """
    IN_DB, NORMALIZED, RAW
    
cpdef enum Polynomial:
    """
    used for polynomial based gens 
    """
    EITHER_KIND, FIRST_KIND, SECOND_KIND, BOTH_KINDS



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


    # these should all be considered private
    @staticmethod
    cdef mus_any from_ptr(cclm.mus_any *_ptr, bint owner=*, cython.int length=*)
    

    cpdef cache_append(self, obj)
    cpdef cache_extend(self, obj)
    cpdef set_up_data(self , cython.int length=*)
    cpdef set_up_xcoeffs(self)
    cpdef set_up_ycoeffs(self)
    cpdef set_up_pv_data(self)





cpdef int mus_close(mus_any obj)
cpdef bint mus_is_output(mus_any obj)
cpdef bint mus_is_input(mus_any obj)
cpdef mus_reset(mus_any obj)
cpdef cython.long clm_length(obj)
cpdef cython.double clm_srate(obj)
cpdef cython.long clm_framples(obj)
cpdef cython.double clm_random(cython.double x=*)
cpdef bint is_zero( cython.numeric x)
cpdef cython.double radians2hz(cython.double radians )
cpdef cython.double hz2radians(cython.double hz)
cpdef cython.double degrees2radians(cython.double degrees)
cpdef cython.double radians2degrees(cython.double radians)
cpdef cython.double db2linear(cython.double x)
cpdef cython.double linear2db(cython.double x)
cpdef cython.double odd_multiple(cython.double x, cython.double y)
cpdef cython.double even_multiple(cython.double x, cython.double y)
cpdef cython.double odd_weight(cython.double x)
cpdef cython.double even_weight(cython.double x)
cpdef cython.double get_srate()
cpdef cython.double set_srate(cython.double r)
cpdef cython.long seconds2samples(cython.double secs)
cpdef cython.double samples2seconds(cython.long samples)
cpdef cython.double ring_modulate(cython.double s1, cython.double s2)
cpdef cython.double amplitude_modulate(cython.double s1, cython.double s2, cython.double s3)
cpdef cython.double contrast_enhancement(cython.double insig, cython.double fm_index=* )
cpdef cython.double dot_product(np.ndarray data1, np.ndarray  data2)
cpdef cython.double polynomial(np.ndarray coeffs, np.ndarray x )
cpdef cython.double array_interp(np.ndarray  fn, cython.double  x)
cpdef cython.double bessi0(cython.double x)
cpdef cython.double mus_interpolate(Interp interp_type, cython.double x, np.ndarray table, cython.double y1 =*)
cpdef np.ndarray make_fft_window(Window window_type, cython.int size, cython.double beta=* , cython.double alpha=*)


cpdef np.ndarray mus_fft(np.ndarray rdat, np.ndarray  idat, cython.int fft_size , cython.int sign)
cpdef np.ndarray fft(np.ndarray rdat, np.ndarray idat,  cython.int fft_size, cython.int sign )
cpdef np.ndarray mus_rectangular2polar(np.ndarray  rdat, np.ndarray  idat)
cpdef tuple rectangular2polar(np.ndarray  rdat, np.ndarray  idat)
cpdef np.ndarray mus_rectangular2magnitudes(np.ndarray rdat, np.ndarray idat)
cpdef tuple rectangular2magnitudes(np.ndarray rdat, np.ndarray idat)
cpdef np.ndarray mus_polar2rectangular( np.ndarray rdat,  np.ndarray idat)
cpdef tuple polar2rectangular(np.ndarray rdat, np.ndarray idat)
cpdef np.ndarray mus_spectrum(np.ndarray  rdat, np.ndarray  idat, np.ndarray  window, Spectrum norm_type=*)
cpdef tuple spectrum(np.ndarray rdat,np.ndarray idat,np.ndarray window, Spectrum norm_type=*)
cpdef np.ndarray mus_spectrum(np.ndarray  rdat, np.ndarray  idat, np.ndarray  window, Spectrum norm_type=*)
cpdef np.ndarray convolution(np.ndarray  rl1, np.ndarray  rl2, fft_size = *)
cpdef np.ndarray mus_convolution(np.ndarray  rl1, np.ndarray  rl2, fft_size=*)
cpdef np.ndarray autocorrelate(np.ndarray data)
cpdef np.ndarray mus_autocorrelate(np.ndarray data)
cpdef np.ndarray mus_correlate( np.ndarray  data1,  np.ndarray  data2)
cpdef np.ndarray correlate(np.ndarray  data1, np.ndarray  data2)
cpdef np.ndarray mus_cepstrum( np.ndarray  data)
cpdef np.ndarray cepstrum(np.ndarray data)
cpdef np.ndarray partials2wave(partials,  np.ndarray wave=*, table_size=*, bint norm=* )
cpdef np.ndarray phase_partials2wave(partials, np.ndarray wave=*, table_size=*, bint norm=* )
cpdef np.ndarray partials2polynomial(partials, Polynomial kind=*)
cpdef np.ndarray normalize_partials(partials)
cpdef cython.double chebyshev_tu_sum(cython.double x, np.ndarray tcoeffs, np.ndarray ucoeffs)
cpdef cython.double chebyshev_t_sum(cython.double x, np.ndarray tcoeffs)
cpdef cython.double chebyshev_u_sum(cython.double x, np.ndarray ucoeffs)

# ---------------- clm gen ---------------- #
cpdef mus_any make_oscil(cython.double frequency=*,cython.double  initial_phase = *)
cpdef cython.double oscil(mus_any gen, cython.double fm=*, cython.double pm=*)
cpdef cython.double oscil_unmodulated(mus_any gen)
cpdef bint is_oscil(mus_any gen)

cpdef mus_any make_oscil_bank(freqs, phases, amps=*, bint stable=*)
cpdef cython.double oscil_bank(mus_any gen)
cpdef bint is_oscil_bank(mus_any gen)

cpdef mus_any make_env(envelope, cython.double scaler=*, cython.double duration=*, cython.double offset=*, cython.double base=*, cython.int length=*)
cpdef cython.double env(mus_any gen )
cpdef bint is_env(mus_any gen)
cpdef cython.double env_interp(cython.double x , mus_any env)
cpdef cython.double envelope_interp(cython.double x, mus_any env)
cpdef np.ndarray env_rates( mus_any gen)
cpdef cython.double env_any(mus_any gen , connection_function)

cpdef mus_any make_pulsed_env(envelope, cython.double duration , cython.double frequency )
cpdef cython.double pulsed_env(mus_any gen, cython.double fm=*)
cpdef cython.double pulsed_env_unmodulated(mus_any gen)
cpdef bint is_pulsed_env(mus_any gen)

cpdef mus_any make_table_lookup(cython.double frequency=*, cython.double initial_phase=*, wave=*, cython.int size=*, Interp interp_type=*)
cpdef cython.double table_lookup(mus_any gen, cython.double fm=*)
cpdef cython.double table_lookup_unmodulated(mus_any gen)
cpdef bint is_table_lookup(mus_any gen)
cpdef mus_any make_table_lookup_with_env(cython.double frequency, envelope, size= *)


cpdef mus_any make_polywave(cython.double frequency, partials = *, Polynomial kind=*, xcoeffs=*, ycoeffs=*)
cpdef cython.double polywave(mus_any gen, cython.double fm=*)
cpdef cython.double polywave_unmodulated(mus_any gen)        
cpdef bint is_polywave(mus_any gen)

cpdef mus_any make_polyshape(cython.double frequency, cython.double initial_phase=*, coeffs = *, partials = *, Polynomial kind=*)
cpdef cython.double polyshape(mus_any gen, cython.double index=*, cython.double fm=*)
cpdef cython.double polyshape_unmodulated(mus_any gen, cython.double index=*)
cpdef bint is_polyshape(mus_any gen)

cpdef mus_any make_triangle_wave(cython.double frequency, cython.double amplitude=*, cython.double phase=*)
cpdef cython.double triangle_wave(mus_any gen, cython.double fm=*)
cpdef cython.double triangle_wave_unmodulated(mus_any gen)
cpdef bint is_triangle_wave(mus_any gen)


cpdef mus_any make_square_wave(cython.double frequency , cython.double amplitude=*, cython.double phase=*)
cpdef cython.double square_wave(mus_any gen, cython.double fm=*)
cpdef bint is_square_wave(mus_any gen)

cpdef mus_any make_sawtooth_wave(cython.double frequency, cython.double amplitude=*, cython.double phase=*)
cpdef cython.double sawtooth_wave(mus_any gen, cython.double fm =*)
cpdef bint is_sawtooth_wave(mus_any gen)

cpdef mus_any make_pulse_train(cython.double frequency=*, cython.double amplitude=*, cython.double phase=*)
cpdef cython.double pulse_train(mus_any gen, cython.double fm=*)
cpdef cython.double pulse_train_unmodulated(mus_any gen)
cpdef bint is_pulse_train(mus_any gen)

cpdef mus_any make_ncos(cython.double frequency, cython.int n=*)
cpdef cython.double ncos(mus_any gen, cython.double mus_any, fm=*)
cpdef bint is_ncos(mus_any gen)

cpdef mus_any make_nsin(cython.double frequency, cython.int n =*)
cpdef cython.double nsin(mus_any gen, cython.double fm=*)
cpdef bint is_nsin(mus_any gen )

cpdef mus_any make_nrxysin( cython.double frequency,  cython.double ratio=*, cython.int n=*, cython.double r=*)
cpdef cython.double nrxysin(mus_any gen, cython.double fm=*)
cpdef bint is_nrxysin(mus_any gen)

cpdef mus_any make_nrxycos(cython.double frequency , cython.double ratio=*, cython.int n=*, cython.double r=*)
cpdef cython.double nrxycos(mus_any gen, cython.double fm=*)
cpdef bint is_nrxycos(mus_any gen)

cpdef mus_any make_rxykcos(cython.double frequency, cython.double phase=*, cython.double r=*, cython.double ratio=*)
cpdef cython.double rxykcos(mus_any gen, cython.double fm=*)
cpdef bint is_rxykcos(mus_any gen)

cpdef mus_any make_rxyksin(cython.double frequency, cython.double phase, cython.double r=*, cython.double ratio=*)
cpdef cython.double rxyksin(mus_any gen, cython.double fm=*)
cpdef bint is_rxyksin(mus_any gen)

cpdef mus_any make_ssb_am(cython.double frequency, cython.int order=*)
cpdef cython.double ssb_am(mus_any gen, cython.double insig , cython.double  fm=*)
cpdef cython.double ssb_am_unmodulated(mus_any gen, cython.double insig)
cpdef bint is_ssb_am(mus_any gen)

cpdef mus_any make_wave_train(cython.double frequency, wave=*, cython.double initial_phase=*, Interp interp_type=*)
cpdef cython.double wave_train(mus_any gen, cython.double fm=*)
cpdef cython.double wave_train_unmodulated(mus_any gen)
cpdef bint is_wave_train(mus_any gen)
cpdef mus_any make_wave_train_with_env(cython.double frequency, envelope, size=*)

cpdef mus_any make_rand(cython.double frequency, cython.double amplitude=*, distribution=*)
cpdef cython.double rand(mus_any gen, cython.double sweep=*)
cpdef cython.double rand_unmodulated(mus_any gen)
cpdef bint is_rand(mus_any gen)

cpdef mus_any make_rand_interp(cython.double frequency, cython.double amplitude=*, distribution=*)
cpdef rand_interp(mus_any gen, cython.double sweep=*)
cpdef rand_interp_unmodulated(mus_any gen)
cpdef bint is_rand_interp(mus_any gen)

cpdef mus_any make_one_pole(cython.double a0, cython.double b1)
cpdef cython.double one_pole(mus_any gen, cython.double insig)
cpdef bint is_one_pole(mus_any gen)

cpdef mus_any make_one_zero(cython.double a0 , cython.double a1)
cpdef cython.double one_zero(mus_any gen, cython.double insig)
cpdef bint is_one_zero(mus_any gen)

cpdef make_two_pole(frequency=*, radius=*, a0=*,b1=*,b2=*)
cpdef cython.double two_pole(mus_any gen, cython.double insig)
cpdef bint is_two_pole(mus_any gen)

cpdef make_two_zero(frequency=*, radius=*, a0=*,a1=*,a2=*)
cpdef cython.double two_zero(mus_any gen, cython.double insig)
cpdef bint is_two_zero(mus_any gen)

cpdef mus_any make_formant(cython.double frequency, cython.double radius)
cpdef cython.double formant(mus_any gen,  cython.double insig, radians=*)
cpdef bint is_formant(mus_any gen)

cpdef mus_any make_formant_bank(list filters, amps=*)
cpdef cython.double formant_bank(mus_any gen, inputs)
cpdef bint is_formant_bank(mus_any gen)

cpdef mus_any make_firmant(cython.double frequency, cython.double radius)
cpdef firmant(mus_any gen, mus_any insig, radians=* )
cpdef is_firmant(mus_any gen)

cpdef mus_any make_filter(cython.int order, xcoeffs, ycoeffs)
cpdef cython.double filter(mus_any gen, cython.double insig)
cpdef bint is_filter(mus_any gen )

cpdef mus_any make_fir_filter(cython.int order, xcoeffs)
cpdef cython.double fir_filter(mus_any gen, cython.double insig)
cpdef bint is_fir_filter(mus_any gen)

cpdef mus_any make_iir_filter(cython.int order, ycoeffs)
cpdef cython.double iir_filter(mus_any gen, cython.double insig )
cpdef bint is_iir_filter(mus_any gen)

cpdef np.ndarray make_fir_coeffs(cython.int order, envelope)


cpdef mus_any make_delay(cython.long size, initial_contents=*, initial_element=*, max_size=*, Interp interp_type=*)
cpdef cython.double delay(mus_any gen, cython.double insig, cython.double pm=*)
cpdef cython.double delay_unmodulated(mus_any gen, cython.double insig)
cpdef bint is_delay(mus_any gen)
cpdef cython.double tap(mus_any gen, offset=*)
cpdef cython.double tap_unmodulated(mus_any gen)
cpdef bint is_tap(mus_any gen)
cpdef cython.double delay_tick(mus_any gen, cython.double insig )

cpdef mus_any make_comb(cython.double feedback=*,size=*, initial_contents=*, initial_element=*, max_size=*, Interp interp_type=*)
cpdef cython.double comb(mus_any gen, cython.double insig, pm=*)
cpdef bint is_comb(mus_any gen)

cpdef mus_any make_comb_bank(list combs)
cpdef cython.double comb_bank(mus_any gen, cython.double insig)
cpdef bint is_comb_bank(mus_any gen)

cpdef mus_any make_filtered_comb(cython.double feedback,cython.long size, mus_any filter, initial_contents=*,  cython.double initial_element=*, max_size=*, Interp interp_type=*)
cpdef cython.double filtered_comb(mus_any gen, cython.double insig, cython.double pm=*)
cpdef cython.double filtered_comb_unmodulated(mus_any gen, cython.double insig)
cpdef bint is_filtered_comb(mus_any gen)

cpdef mus_any make_filtered_comb_bank(list fcombs)
cpdef cython.double filtered_comb_bank(mus_any gen)
cpdef bint is_filtered_comb_bank(mus_any gen)

cpdef mus_any make_notch(cython.double feedforward=*,size=*, initial_contents=*, cython.double initial_element=*, max_size=*,Interp interp_type=*)
cpdef cython.double notch(mus_any gen, cython.double insig, cython.double pm=*)
cpdef cython.double notch_unmodulated(mus_any gen, cython.double insig)
cpdef bint is_notch(mus_any gen)


cpdef mus_any make_all_pass(cython.double feedback, cython.double  feedforward, cython.long  size, initial_contents= *, cython.double  initial_element = *,  max_size= *, Interp interp_type = *)
cpdef cython.double all_pass(mus_any gen, cython.double insig, pm=*)
cpdef is_all_pass(mus_any gen)

cpdef mus_any make_all_pass_bank(list all_passes)
cpdef cython.double all_pass_bank(mus_any gen, cython.double insig)
cpdef bint is_all_pass_bank(mus_any gen)

cpdef mus_any make_one_pole_all_pass(cython.long size, cython.double coeff)
cpdef cython.double one_pole_all_pass(mus_any gen, cython.double insig)
cpdef bint is_one_pole_all_pass(mus_any gen)

cpdef mus_any make_moving_average(cython.long size, initial_contents=*, cython.double initial_element=*)
cpdef cython.double moving_average(mus_any gen, cython.double  insig)
cpdef bint is_moving_average(mus_any gen)

cpdef mus_any make_moving_max(cython.long size, initial_contents=*, cython.double initial_element=*)
cpdef cython.double moving_max(mus_any gen, cython.double insig)
cpdef bint is_moving_max(mus_any gen)

cpdef mus_any make_moving_norm(cython.long size, initial_contents=*, cython.double scaler=*)
cpdef cython.double moving_norm(mus_any gen, cython.double insig)
cpdef is_moving_norm(mus_any gen)

cpdef mus_any make_asymmetric_fm(cython.double frequency, cython.double initial_phase=*, cython.double r=*, cython.double ratio=*)
cpdef cython.double asymmetric_fm(mus_any gen, cython.double index, cython.double fm=*)
cpdef cython.double asymmetric_fm_unmodulated(mus_any gen, cython.double index)
cpdef bint is_asymmetric_fm(mus_any gen)

cpdef mus_any make_file2sample(str filename, buffer_size=*)
cpdef cython.double file2sample(mus_any gen, cython.long loc, cython.int chan=*)
cpdef bint is_file2sample(mus_any gen)


cpdef mus_any make_sample2file(str filename, cython.int chans=*, sndlib.Sample sample_type=*, sndlib.Header header_type=*, comment=*)


cpdef mus_any make_locsig(cython.double degree=*, cython.double distance=*,  cython.double reverb=*, output=*, revout=*, channels=*, reverb_channels=*,Interp interp_type=*)
cpdef cython.double locsig(mus_any gen, cython.long loc, cython.double val)
cpdef bint is_locsig(mus_any gen)


cpdef make_two_pole(frequency=*, radius=*, a0=*,b1=*,b2=*)
