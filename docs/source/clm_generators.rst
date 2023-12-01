CLM Generators
================

.. module:: 
    pysndlib.clm
    :noindex:

**in pysndlib.clm**

all_pass 
---------

.. autofunction:: make_all_pass
.. autofunction:: all_pass
.. autofunction:: is_all_pass

.. table:: all_pass properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_length       length of delay
    mus_order        same as mus-length
    mus_data         delay line itself (not setable)
    mus_feedback     feedback scaler
    mus_feedforward  feedforward scaler
    mus_interp_type  interpolation choice (not setable)
    ===============  =========================================

Example usage

:: 
    
    with Sound(play=True):
        alp = make_all_pass(-0.4, 0.4, seconds2samples(0.4))
        osc = make_oscil(440.0)
        ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
        for i in range(88200):
            outa(i, 0.5 * (all_pass(alp, env(ampf) * oscil(osc))))


.. seealso::
    `sndlib all-pass <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#all-passdoc>`_




all_pass_bank
--------------
.. autofunction:: make_all_pass_bank
.. autofunction:: all_pass_bank
.. autofunction:: is_all_pass_bank

            
asymmetric_fm
--------------
.. autofunction:: make_asymmetric_fm
.. autofunction:: asymmetric_fm
.. autofunction:: is_asymmetric_fm


.. table:: asymmetric_fm properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_scaler       "r" parameter; sideband scaler
    mus_offset       "ratio" parameter
    mus_increment    frequency in radians per sample
    ===============  =========================================

Example usage

:: 
    
    with Sound(play=True):    
        fm = make_asymmetric_fm(440.0, 0.0, 0.9, 0.5)
        for i in range(44100):
            outa(i, 0.5 * asymmetric_fm(fm, 1.0))
        
        

comb
--------------
.. autofunction:: make_comb
.. autofunction:: comb
.. autofunction:: is_comb


.. table:: comb properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_length       length of delay
    mus_order        same as mus_length
    mus_data         delay line itself (not setable)
    mus_feedback     scaler (comb only)
    mus_feedforward  scaler (notch only)
    mus_interp_type  interpolation choice (not setable)
    ===============  =========================================



Example usage

:: 
    
    with Sound(play=True):
        cmb = make_comb(0.4, seconds2samples(0.4))
        osc = make_oscil(440.0)
        ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
    
        for i in range(88200):
            outa(i, 0.5 * (comb(cmb, env(ampf) * oscil(osc))))

comb_bank
--------------
.. autofunction:: make_comb_bank
.. autofunction:: comb_bank
.. autofunction:: is_comb_bank

convolve
--------------
.. autofunction:: make_convolve
.. autofunction:: convolve
.. autofunction:: is_convolve


.. table:: convolve properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_length       fft size used in the convolution
    ===============  =========================================

Example usage

:: 
    
    with Sound(play=True, statistics=True):    
        flt, _ = file2array('oboe.snd', channel=0, beg=0, dur=clm_length('pistol.snd'))
        cnv = make_convolve(make_readin('pistol.snd'), flt)
        for i in range(88200):
            outa(i, .25 * convolve(cnv))
            
convolve_files
----------------
.. autofunction::convolve_files

delay
--------------
.. autofunction:: make_delay
.. autofunction:: delay
.. autofunction:: is_delay
.. autofunction:: tap
.. autofunction:: is_tap
.. autofunction:: delay_tick

.. table:: delay properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_length       length of delay
    mus_order        same as mus_length
    mus_data         delay line itself (not setable)
    mus_interp_type  interpolation choice (not setable)
    mus_scaler       available for delay specializations
    mus_location     current delay line write position
    ===============  =========================================


Example usage

:: 
    
    with Sound(play=True):
        dly = make_delay(seconds2samples(0.5))
        osc1 = make_oscil(440.0)
        osc2 = make_oscil(660.0)
        for i in range(44100):
            outa(i, 0.5 * (oscil(osc1) + delay(dly, oscil(osc2))))  


env
-------

.. module:: 
    pysndlib.clm
    :noindex:

.. autofunction:: make_env
.. autofunction:: env
.. autofunction:: is_env
.. autofunction:: env_interp
.. autofunction:: envelope_interp
.. autofunction:: env_any

.. table:: env properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_location     number of calls so far on this env
    mus_increment    base
    mus_data         original breakpoint list
    mus_scaler       scaler
    mus_offset       offset
    mus_length       duration in samples
    mus_channels     current position in the break-point list
    ===============  =========================================

Example usage

:: 
    
    with Sound(play=True):
        gen = make_oscil(440.0)
        ampf = make_env([0., 0., 0.01, 1.0, 0.25, 0.1, 1, 0], scaler=.5, length=44100)
        for i in range(44100):
            outa(i,  env(ampf)*oscil(gen))

.. seealso::
    `sndlib env <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#envdoc>`_
    
file2frample
--------------
.. autofunction:: make_file2frample
.. autofunction:: file2frample
.. autofunction:: is_file2frample
    
file2sample
--------------
.. autofunction:: make_file2sample
.. autofunction:: file2sample
.. autofunction:: is_file2sample

filtered_comb
--------------
.. autofunction:: make_filtered_comb
.. autofunction:: filtered_comb
.. autofunction:: is_filtered_comb


.. table:: filtered_comb properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_length       length of delay
    mus_order        same as mus_length
    mus_data         delay line itself (not setable)
    mus_feedback     scaler (comb only)
    mus_feedforward  scaler (notch only)
    mus_interp_type  interpolation choice (not setable)
    ===============  =========================================



filtered_comb_bank
----------------------
.. autofunction:: make_filtered_comb_bank
.. autofunction:: filtered_comb_bank
.. autofunction:: is_filtered_comb_bank

    
firmant
--------------
.. autofunction:: make_firmant
.. autofunction:: firmant
.. autofunction:: is_firmant


.. table:: firmant properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_phase        formant radius
    mus_frequency    formant center frequency
    mus_order        2 (not setable)
    ===============  =========================================



Example usage

:: 
    
    with Sound(play=True):
        flt = make_firmant(1000.0, 0.999)
        ran1 = make_rand(10000.0, 5.0)
        for i in range(44100):
            outa(i, 0.5 * firmant(flt, rand(ran1)))

.. seealso::
        
    `sndlib firmant <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#formantdoc>`_ 
    
filter
--------------
.. autofunction:: make_filter
.. autofunction:: filter
.. autofunction:: is_filter



.. table:: filter properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_order        filter order
    mus_xcoeff       x (input) coeff
    mus_xcoeffs      x (input) coeffs
    mus_ycoeff       y (output) coeff
    mus_ycoeffs      y (output) coeffs
    mus_data         current state (input values)
    mus_length       same as mus_order
    ===============  =========================================



fir_filter
--------------
.. autofunction:: make_fir_filter
.. autofunction:: fir_filter
.. autofunction:: is_fir_filter
.. autofunction:: make_fir_coeffs

.. table:: fir_filter properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_order        filter order
    mus_xcoeff       x (input) coeff
    mus_xcoeffs      x (input) coeffs
    mus_ycoeff       y (output) coeff
    mus_ycoeffs      y (output) coeffs
    mus_data         current state (input values)
    mus_length       same as mus_order
    ===============  =========================================



formant
--------------
.. autofunction:: make_formant
.. autofunction:: formant
.. autofunction:: is_formant


.. table:: formant properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_phase        formant radius
    mus_frequency    formant center frequency
    mus_order        2 (not setable)
    ===============  =========================================

.. seealso::
        
    `sndlib formant <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#formantdoc>`_ 

formant_bank
--------------
.. autofunction:: make_formant_bank
.. autofunction:: formant_bank
.. autofunction:: is_formant_bank


.. seealso::
        
    `sndlib formant_bank <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#formantdoc>`_ 
    

frample2file
--------------
.. autofunction:: make_frample2file
.. autofunction:: frample2file
.. autofunction:: is_frample2file
.. autofunction:: continue_frample2file
    
granulate
--------------
.. autofunction:: make_granulate
.. autofunction:: granulate
.. autofunction:: is_granulate


.. table:: granulate properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    time (seconds) between output grains (hop)
    mus_ramp         length (samples) of grain envelope ramp segment
    mus_hop          time (samples) between output grains (hop)
    mus_scaler       grain amp (scaler)
    mus_increment    expansion
    mus_length       grain length (samples)
    mus_data         grain samples (a float_vector)
    mus_location     granulate's local random number seed
    ===============  =========================================


Example usage

:: 
    
    with Sound(play=True):    
        osc = make_oscil(440.0)
        sweep = make_env([0,0,1,1], scaler=hz2radians(440.0), length=44100)
        grn = make_granulate(lambda d : .2 * oscil(osc, env(sweep)), expansion=2.0, length=.5)
        for i in range(88200):
            outa(i, granulate(grn))


iir_filter
--------------
.. autofunction:: make_iir_filter
.. autofunction:: iir_filter
.. autofunction:: is_iir_filter


.. table:: iir_filter properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_order        filter order
    mus_xcoeff       x (input) coeff
    mus_xcoeffs      x (input) coeffs
    mus_ycoeff       y (output) coeff
    mus_ycoeffs      y (output) coeffs
    mus_data         current state (input values)
    mus_length       same as mus_order
    ===============  =========================================


Example usage

:: 
    
    with Sound(play=True):
        flt = make_iir_filter(3, [0.0, -1.978, 0.998])
        ran1 = make_rand(10000.0, 0.002)
        for i in range(44100):
            outa(i, 0.5 * iir_filter(flt, rand(ran1)))
            
    
in_any, out_any, etc
----------------------
.. autofunction:: out_any
.. autofunction:: outa
.. autofunction:: outb
.. autofunction:: outc
.. autofunction:: outd
.. autofunction:: out_bank
.. autofunction:: in_any
.. autofunction:: ina
.. autofunction:: inb

locsig
--------------
.. autofunction:: make_locsig
.. autofunction:: locsig
.. autofunction:: is_locsig


.. table:: locsig properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_data         output scalers (a float_vector)
    mus_xcoeff       reverb scaler
    mus_xcoeffs      reverb scalers (a float_vector)
    mus_channels     output channels
    mus_length       output channels
    ===============  =========================================


Example usage

:: 
        
    with Sound(channels=2, play=True):    
        loc = make_locsig(60.0)
        osc = make_oscil(440.0)
        for i in range(44100):
            locsig(loc, i, .5 * oscil(osc))

.. autofunction:: locsig_ref
.. autofunction:: locsig_set
.. autofunction:: locsig_reverb_ref
.. autofunction:: locsig_reverb_set
.. autofunction:: move_locsig


moving_average
-----------------
.. autofunction:: make_moving_average
.. autofunction:: moving_average
.. autofunction:: is_moving_average


.. table:: moving_average properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_length       length of table
    mus_order        same as mus_length
    mus_data         table of last 'size' values
    ===============  =========================================

Example usage

:: 
    
    with Sound(play=True):
        avg = make_moving_average(4410)
        osc = make_oscil(440.0)
        stop = 44100 - 4410
        #avg.print_cache()
        for i in range(stop):
            val = oscil(osc)
            outa(i, val * moving_average(avg, abs(val))) 
        for i in range(stop, 44100):
            outa(i, oscil(osc) * moving_average(avg, 0.0))


moving_max
--------------
.. autofunction:: make_moving_max
.. autofunction:: moving_max
.. autofunction:: is_moving_max


.. table:: moving_max properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_length       length of table
    mus_order        same as mus_length
    mus_data         table of last 'size' values
    ===============  =========================================


moving_norm
--------------
.. autofunction:: make_moving_norm
.. autofunction:: moving_norm
.. autofunction:: is_moving_norm

.. table:: moving_norm properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_length       length of table
    mus_order        same as mus_length
    mus_data         table of last 'size' values
    ===============  =========================================



ncos
--------------
.. autofunction:: make_ncos
.. autofunction:: ncos
.. autofunction:: is_ncos

.. table:: ncos properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_scaler       dependent on number of cosines
    mus_length       n or cosines arg used in make_ncos
    mus_increment    frequency in radians per sample
    ===============  =========================================

Example usage

:: 
    
    with Sound(play=True):
        gen = make_ncos(440.0, 10);
        for i in range(44100):
            outa(i, .5 * ncos(gen))

.. seealso::
        
    `sndlib ncos <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#ncosdoc>`_
    
notch
--------------
.. autofunction:: make_notch
.. autofunction:: notch
.. autofunction:: is_notch


.. table:: notch properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_length       length of delay
    mus_order        same as mus_length
    mus_data         delay line itself (not setable)
    mus_feedback     scaler (comb only)
    mus_feedforward  scaler (notch only)
    mus_interp_type  interpolation choice (not setable)
    ===============  =========================================


    
nrxycos
--------------
.. autofunction:: make_nrxycos
.. autofunction:: nrxycos
.. autofunction:: is_nrxycos

 .. table:: nrxycos properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_scaler       "r" parameter; sideband scaler
    mus_length       "n" parameter
    mus_increment    frequency in radians per sample
    mus_offset       "ratio" parameter
    ===============  =========================================       
        
Example usage

:: 
    
    with Sound(play=True):
        gen = make_nrxycos(440.0,n=10)
        for i in range(44100):
            outa(i, .5 * nrxycos(gen))
        

.. seealso::
        
    `sndlib nrxycos <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#nrxydoc>`_    


nrxysin
--------------
.. autofunction:: make_nrxysin
.. autofunction:: nrxysin
.. autofunction:: is_nrxysin

.. table:: nrxysin properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_scaler       "r" parameter; sideband scaler
    mus_length       "n" parameter
    mus_increment    frequency in radians per sample
    mus_offset       "ratio" parameter
    ===============  =========================================


Example usage

:: 
    
    with Sound(play=True):
    gen = make_nrxysin(440.0,n=10)
    for i in range(44100):
        outa(i, .5 * nrxysin(gen))
        
.. seealso::
        
    `sndlib nrxysin <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#nrxydoc>`_



nsin
--------------
.. autofunction:: make_nsin
.. autofunction:: nsin
.. autofunction:: is_nsin

.. table:: nsin properties
    :align: left
    :width: 60%
    
    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_scaler       dependent on number of sines
    mus_length       n or sines arg used in make_nsin
    mus_increment    frequency in radians per sample
    ===============  =========================================


Example usage

:: 
    
    with Sound(play=True):
    gen = make_nsin(440.0, 10);
    for i in range(44100):
        outa(i, .5 * ncos(gen))
        

.. seealso::
        
    `sndlib nsin <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#ncosdoc>`_
    
    
    
one_pole
--------------
.. autofunction:: make_one_pole
.. autofunction:: one_pole
.. autofunction:: is_one_pole

.. table:: one_pole properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_xcoeff       a0, a1, a2 in equations
    mus_ycoeff       b1, b2 in equations
    mus_order        1 or 2 (not setable)
    mus_scaler       two_pole and two_zero radius
    mus_frequency    two_pole and two_zero center frequency
    ===============  =========================================

Example usage

:: 
    
    with Sound(play=True):
    r = make_rand(44100)
    op = make_one_zero(.2, .3)
    for i in range(44100):
        #outa(i, one_pole(op, 0.0))
        outa(i, one_zero(op, rand(r)))
        
.. seealso::
        
    `sndlib one_pole <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#one-poledoc>`_ 
    
one_pole_all_pass
--------------------
.. autofunction:: make_one_pole_all_pass
.. autofunction:: one_pole_all_pass
.. autofunction:: is_one_pole_all_pass

.. table:: one_pole_all_pass properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_length       length of delay
    mus_order        same as mus-length
    mus_data         delay line itself (not setable)
    mus_feedback     feedback scaler
    mus_feedforward  feedforward scaler
    mus_interp-type  interpolation choice (not setable)
    ===============  =========================================


one_zero
--------------
.. autofunction:: make_one_zero
.. autofunction:: one_zero
.. autofunction:: is_one_zero


.. table:: one_zero properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_xcoeff       a0, a1, a2 in equations
    mus_ycoeff       b1, b2 in equations
    mus_order        1 or 2 (not setable)
    mus_scaler       two_pole and two_zero radius
    mus_frequency    two_pole and two_zero center frequency
    ===============  =========================================

.. seealso::
        
    `sndlib one_zero <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#one-poledoc>`_ 
    
   

oscil
-------

.. autofunction:: make_oscil
.. autofunction:: oscil
.. autofunction:: is_oscil

.. table:: oscil properties
    :align: left
    :width: 60%

    ===============  ==================================
    ===============  ==================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_length       1 (not setable)
    mus_increment    frequency in radians per sample
    ===============  ==================================

Example usage

:: 
    
    with Sound(play=True):
        gen = make_oscil(440.0)
        for i in range(44100):
            outa(i,  oscil(gen))


.. seealso::
    `sndlib oscil <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#oscildoc>`_



oscil_bank
-----------

.. autofunction:: make_oscil_bank
.. autofunction:: oscil_bank
.. autofunction:: is_oscil_bank

Example usage

:: 
    
    with Sound(play=True):
    ob = make_oscil_bank([hz2radians(i) for i in [400,430,490]], [.5, .2, math.pi+.1], [.5, .3, .2])
    for i in range(44100*2):
        outa(i, oscil_bank(ob))


.. seealso::
    `sndlib oscil_bank <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#oscildoc>`_

phase_vocoder
--------------
.. autofunction:: make_phase_vocoder
.. autofunction:: phase_vocoder
.. autofunction:: is_phase_vocoder

.. table:: phase_vocoder properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    pitch shift
    mus_length       fft_size
    mus_increment    interp
    mus_hop          fft_size / overlap
    mus_location     outctr (counter to next fft)
    ===============  =========================================

Example usage

:: 
    
    with Sound(play=True):    
        pv = make_phase_vocoder(make_readin("oboe.snd"), pitch=2.0)
        for i in range(44100):
            outa(i, phase_vocoder(pv))    

.. autofunction:: phase_vocoder_amp_increments
.. autofunction:: phase_vocoder_amps
.. autofunction:: phase_vocoder_freqs
.. autofunction:: phase_vocoder_phases
.. autofunction:: phase_vocoder_phase_increments



polyshape
--------------
.. autofunction:: make_polyshape
.. autofunction:: polyshape
.. autofunction:: is_polyshape

.. table:: polyshape properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_scaler       index
    mus_phase        phase in radians
    mus_data         polynomial coeffs
    mus_length       number of partials
    mus_increment    frequency in radians per sample
    ===============  =========================================

Example usage

:: 
    
    with Sound(play=True):
        wav = make_polyshape(frequency=500, partials=[1, .5, 2, .3, 3, .2])
        for i in range(40000):
            outa(i, 1. * polyshape(wav))
        
.. seealso::

    `sndlib polyshape <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#polywavedoc>`_

    polywave :py:func:`pysndlib.clm.make_polywave`
    
    partials2polynomial :py:func:`pysndlib.clm.partials2polynomial`
    
    normalize_partials :py:func:`pysndlib.clm.normalize_partials`
    
    normalize_partials :py:func:`pysndlib.clm.normalize_partials`
    
    chebyshev_tu_sum :py:func:`pysndlib.clm.chebyshev_tu_sum`
    
    chebyshev_t_sum :py:func:`pysndlib.clm.chebyshev_t_sum`
    
    chebyshev_t_sum :py:func:`pysndlib.clm.chebyshev_t_sum`
    
    polynomial :py:func:`pysndlib.clm.polynomial`
    
    



polywave
---------
.. autofunction:: make_polywave
.. autofunction:: polywave
.. autofunction:: is_polywave

.. table:: polywave properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_data         polynomial coeffs
    mus_length       number of partials
    mus_increment    frequency in radians per sample
    ===============  =========================================

Example usage

:: 
    
    with Sound(play=True):
        gen = make_polywave(440, partials=[1., .5, 2, .5])
        for i in range(44100):
            outa(i, .5 * polywave(gen))
            
.. seealso::

    `sndlib polywave <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#polywavedoc>`_

    polyshape :py:func:`pysndlib.clm.make_polyshape`
    
    partials2polynomial :py:func:`pysndlib.clm.partials2polynomial`
    
    normalize_partials :py:func:`pysndlib.clm.normalize_partials`
    
    normalize_partials :py:func:`pysndlib.clm.normalize_partials`
    
    chebyshev_tu_sum :py:func:`pysndlib.clm.chebyshev_tu_sum`
    
    chebyshev_t_sum :py:func:`pysndlib.clm.chebyshev_t_sum`
    
    chebyshev_t_sum :py:func:`pysndlib.clm.chebyshev_t_sum`
    
    polynomial :py:func:`pysndlib.clm.polynomial`        



    
pulsed_env
-----------
.. autofunction:: make_pulsed_env
.. autofunction:: pulsed_env
.. autofunction:: is_pulsed_env

.. table:: pulsed_env properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_location     number of calls so far on this env
    mus_increment    base
    mus_data         original breakpoint list
    mus_scaler       scaler
    mus_offset       offset
    mus_length       duration in samples
    mus_channels     current position in the break-point list
    ===============  =========================================

Example usage

:: 
    
    with Sound():
        e = make_pulsed_env([0,0,1,1,2,0], .01, 1)
        frq = make_env([0,0,1,1], duration=1.0, scaler=hz2radians(50))
        for i in range(44100):
            outa(i, .5 * pulsed_env(e, env(frq)))
            
.. seealso::
    `sndlib env <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#envdoc>`_
        
pulse_train
--------------
.. autofunction:: make_pulse_train
.. autofunction:: pulse_train
.. autofunction:: is_pulse_train

.. table:: pulse_train properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_scaler       amplitude arg used in make_pulse_train
    mus_increment    frequency in radians per sample
    ===============  =========================================
    
Example usage

:: 
    
    with Sound(play=True):
    gen = make_pulse_train(100.0)
    e = make_env([0.0, 0.0, .5, 1.0, 1.0, 0.0], scaler=400, length=44100*3)
    for i in range(44100*3):
        outa(i, pulse_train(gen, hz2radians(env(e))))

    
.. seealso::
        
    `sndlib pulse_train <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#sawtoothdoc>`_
    
rand
--------------
.. autofunction:: make_rand
.. autofunction:: rand
.. autofunction:: is_rand


.. table:: rand properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_scaler       amplitude arg used in make_<gen>
    mus_length       distribution table np.ndarray length
    mus_data         distribution table np.ndarray, if any
    mus_increment    frequency in radians per sample
    ===============  =========================================



Example usage

:: 
    
    with Sound(channels=2, play=True):
        ran1 = make_rand(5.0, hz2radians(220.0))
        ran2 = make_rand_interp(5.0, hz2radians(220.0))
        osc1 = make_oscil(440.0)
        osc2 = make_oscil(1320.0)
        for i in range(88200):
            outa(i, 0.5 * oscil(osc1, rand(ran1)))
            outb(i, 0.5 * oscil(osc2, rand_interp(ran2)))
            
       
.. seealso::
        
    `sndlib rand <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#randdoc>`_ 

rand_interp
-------------------
.. autofunction:: make_rand_interp
.. autofunction:: rand_interp
.. autofunction:: is_rand_interp


.. table:: rand_interp properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_scaler       amplitude arg used in make_<gen>
    mus_length       distribution table np.ndarray length
    mus_data         distribution table np.ndarray, if any
    mus_increment    frequency in radians per sample
    ===============  =========================================

Example usage

:: 
    
    with Sound(channels=2, play=True):
        ran1 = make_rand(5.0, hz2radians(220.0))
        ran2 = make_rand_interp(5.0, hz2radians(220.0))
        osc1 = make_oscil(440.0)
        osc2 = make_oscil(1320.0)
        for i in range(88200):
            outa(i, 0.5 * oscil(osc1, rand(ran1)))
            outb(i, 0.5 * oscil(osc2, rand_interp(ran2)))
        
.. seealso::
        
    `sndlib rand_interp <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#randdoc>`_ 
    
readin
--------------
.. autofunction:: make_readin
.. autofunction:: readin
.. autofunction:: is_readin


.. table:: readin properties
    :align: left
    :width: 60%

    ===============  ==================================================
    ===============  ==================================================
    mus_channel      channel arg to make_readin (not setable)
    mus_location     current location in file
    mus_increment    sample increment (direction arg to make_readin)
    mus_file_name    name of file associated with gen
    mus_length       number of framples in file associated with gen
    ===============  ==================================================

    
rxykcos
--------------
.. autofunction:: make_rxykcos
.. autofunction:: rxykcos
.. autofunction:: is_rxykcos

.. table:: rxykcos properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_scaler       "r" parameter; sideband scaler
    mus_increment    frequency in radians per sample
    ===============  =========================================

Example usage

:: 
    
    with Sound(play=True):
    gen = make_rxykcos(440.0, r=.5, ratio=1.2)
    e = make_env([0.0, .5, 1.0, .0], length=44100*3, base=32)
    for i in range(44100*3):
        gen.mus_scaler = env(e)
        outa(i, .5 * rxykcos(gen))

rxyksin
--------------
.. autofunction:: make_rxyksin
.. autofunction:: rxyksin
.. autofunction:: is_rxyksin


.. table:: rxyksin properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_scaler       "r" parameter; sideband scaler
    mus_increment    frequency in radians per sample
    ===============  =========================================


Example usage

:: 
    
    with Sound(play=True):
    gen = make_rxykcos(440.0, r=.5, ratio=.1)
    e = make_env([0.0, 1.2, 1.0, .1], length=44100*3)
    for i in range(44100*3):
        gen.mus_scaler = env(e)
        outa(i, .5 * rxykcos(gen))
        

sample2file
--------------
.. autofunction:: make_sample2file
.. autofunction:: sample2file
.. autofunction:: is_sample2file
.. autofunction:: continue_sample2file
        
src
--------------
.. autofunction:: make_src
.. autofunction:: src
.. autofunction:: is_src


.. table:: src properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_increment    srate arg to make_src
    ===============  =========================================

Example usage

:: 
    
    with Sound(play=True, srate=22050):    
        rd = make_readin("oboe.snd");
        length = 2 * mus_sound_framples("oboe.snd")
        sr = make_src(rd, .5);
    
        for i in range(length):
            outa(i, src(sr))

ssb_am
--------------
.. autofunction:: make_ssb_am
.. autofunction:: ssb_am
.. autofunction:: is_ssb_am

.. table:: ssb_am properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase (of embedded sin osc) in radians
    mus_order        embedded delay line size
    mus_length       same as mus_order
    mus_interp_type  mus_interp_none
    mus_xcoeff       FIR filter coeff
    mus_xcoeffs      embedded Hilbert transform FIR filter coeffs
    mus_data         embedded filter state
    mus_increment    frequency in radians per sample
    ===============  =========================================



Example usage

:: 
    
    with Sound(play=True):
    gen1 = make_ssb_am(750, 40)
    gen2 = make_ssb_am(700, 40)
    rd = make_readin('oboe.snd')
    for i in range(44100):
        outa(i, .5*(ssb_am(gen1, readin(rd)) + ssb_am(gen2, readin(rd))))
        

.. seealso::
        
    `sndlib ssb_am <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#ssb-amdoc>`_

    
table_lookup
-------------
.. autofunction:: make_table_lookup
.. autofunction:: table_lookup
.. autofunction:: is_table_lookup
.. autofunction:: make_table_lookup_with_env

.. table:: table_lookup properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_data         wave float-vector
    mus_length       wave size (not setable)
    mus_interp-type  interpolation choice (not setable)
    mus_increment    table increment per sample
    ===============  =========================================

Example usage

:: 
    
    with Sound(play=True):
        gen = make_table_lookup(440.0, wave=partials2wave([1., .5, 2, .5]))
        for i in range(44100):
            outa(i, .5 * table_lookup(gen))
        
        

sawtooth_wave
--------------
.. autofunction:: make_sawtooth_wave
.. autofunction:: sawtooth_wave
.. autofunction:: is_sawtooth_wave

.. table:: sawtooth_wave properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_scaler       amplitude arg used in make_sawtooth_wave
    mus_increment    frequency in radians per sample
    ===============  =========================================
    
Example usage

:: 
    
    with Sound(play=True):
    gen = make_sawtooth_wave(261.0)
    lfo = make_oscil(.5)
    for i in range(44100*3):
        outa(i, .5 * sawtooth_wave(gen, oscil(lfo) * hz2radians(20)))    
    
.. seealso::
        
    `sndlib sawtooth_wave <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#sawtoothdoc>`_    
    
    
    
square_wave
--------------
.. autofunction:: make_square_wave
.. autofunction:: square_wave
.. autofunction:: is_square_wave

.. table:: square_wave properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_scaler       amplitude arg used in make_square_wave
    mus_width        width of square-wave pulse (0.0 to 1.0)
    mus_increment    frequency in radians per sample
    ===============  =========================================
    
Example usage

:: 
    
    with Sound(play=True):
        gen = make_square_wave(261.0)
        e = make_env([0.0, .5, .5, .1, 1.0, .8], length=44100*2)
        for i in range(44100*2):
            gen.mus_width = env(e)
            outa(i, .5 * square_wave(gen))    

.. seealso::
        
    `sndlib square_wave <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#sawtoothdoc>`_       
        


triangle_wave
--------------
.. autofunction:: make_triangle_wave
.. autofunction:: triangle_wave
.. autofunction:: is_triangle_wave

.. table:: triangle_wave properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_scaler       amplitude arg used in make_triangle_wave
    mus_increment    frequency in radians per sample
    ===============  =========================================        

Example usage

:: 
    
    with Sound(play=True):
        gen = make_triangle_wave(440.0)
        for i in range(44100):
        outa(i, .5 * triangle_wave(gen))

.. seealso::
        
    `sndlib triangle_wave <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#sawtoothdoc>`_



 
two_pole
--------------
.. autofunction:: make_two_pole
.. autofunction:: two_pole
.. autofunction:: is_two_pole


.. table:: two_pole properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_xcoeff       a0, a1, a2 in equations
    mus_ycoeff       b1, b2 in equations
    mus_order        1 or 2 (not setable)
    mus_scaler       two_pole and two_zero radius
    mus_frequency    two_pole and two_zero center frequency
    ===============  =========================================


Example usage

:: 
    
    with Sound("out1.aif", 1):
        flt = make_two_pole(1000.0, 0.999)
        ran1 = make_rand(10000.0, 0.002)
        for i in range(44100):
            outa(i, 0.5 * two_pole(flt, rand(ran1)))

.. seealso::
        
    `sndlib two_pole <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#one-poledoc>`_        

two_zero
--------------
.. autofunction:: make_two_zero
.. autofunction:: two_zero
.. autofunction:: is_two_zero


.. table:: two_zero properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_xcoeff       a0, a1, a2 in equations
    mus_ycoeff       b1, b2 in equations
    mus_order        1 or 2 (not setable)
    mus_scaler       two_pole and two_zero radius
    mus_frequency    two_pole and two_zero center frequency
    ===============  =========================================

.. seealso::
        
    `sndlib two_zero <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#one-poledoc>`_ 


    
    


wave_train
--------------
.. autofunction:: make_wave_train
.. autofunction:: wave_train
.. autofunction:: is_wave_train
.. autofunction:: make_wave_train_with_env


.. table:: wave_train properties
    :align: left
    :width: 60%

    ===============  =========================================
    ===============  =========================================
    mus_frequency    frequency in Hz
    mus_phase        phase in radians
    mus_data         wave array (not setable)
    mus_length       length of wave array (not setable)
    mus_interp_type  interpolation choice (not setable)
    ===============  =========================================
    
    
    
Example usage

:: 
    
    with Sound(play=True):
    v = np.zeros(64, dtype=np.double)
    g = make_ncos(400,10)
    g.mus_phase = -.5 * math.pi
    for i in range(64):
        v[i] = ncos(g)
    gen = make_wave_train(440, wave=v)    
    for i in range(44100):
        outa(i, .5 * wave_train(gen))
        
       
.. seealso::
        
    `sndlib wave_train <https://ccrma.stanford.edu/software/snd/snd/sndclm.html#wave-traindoc>`_ 
    




* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


