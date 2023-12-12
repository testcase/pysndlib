import math
import cython
#import pysndlib.clm as clm
# import cython
# if cython.compiled: 
cimport pysndlib.clm as clm
import numpy as np
#import pysndlib.clm as clm
# else:


#not sure typing any of the args are worth it.
@cython.ccall
def fm_violin(beg, dur, frequency, amplitude, fm_index, 
            amp_env = [ 0, 0,  25, 1,  75, 1,  100, 0],
            periodic_vibrato_rate = 5.0,
            random_vibrato_rate = 16.0,
            periodic_vibrato_amplitude = 0.0025,
            random_vibrato_amplitude = 0.005,
            fm1_index=None,
            fm2_index=None,
            fm3_index=None,
            fm1_rat = 1.0,
            fm2_rat = 3.0,
            fm3_rat = 4.0,
            fm1_env = [0., 1., 25., .4, 75., .6, 100., 0.],
            fm2_env = [0., 1., 25., .4, 75., .6, 100., 0.],
            fm3_env = [0., 1., 25., .4, 75., .6, 100., 0.],
            gliss_env = [0,0,100,0],
            glissando_amount = 0.0,
            noise_amount = 0.0,
            noise_freq = 1000.,
            ind_noise_freq = 10.,
            ind_noise_amount = 0.0,
            amp_noise_freq = 20.0,
            amp_noise_amount = 0.0,
            degree=45,
            distance=1.0,
            reverb_amount = .01):   



    start: cython.int = clm.seconds2samples(beg)
    end: cython.int = start + clm.seconds2samples(dur)
    frq_scl: cython.double = clm.hz2radians(frequency)
    maxdev: cython.double = frq_scl * fm_index
    index1: cython.double = fm1_index if fm1_index else min(math.pi, maxdev * (5.0 / math.log(frequency)))
    index2: cython.double = fm2_index if fm2_index else min(math.pi, maxdev * 3.0 * ((8.5 - math.log(frequency)) / (3.0 + (frequency / 1000.))))
    index3: cython.double = fm3_index if fm3_index else min(math.pi, maxdev * (4.0 / math.sqrt(frequency)))
    carrier: clm.mus_any = clm.make_oscil(frequency)
    fmosc1: clm.mus_any = clm.make_oscil(frequency)
    fmosc2: clm.mus_any = clm.make_oscil(frequency * fm2_rat)
    fmosc3: clm.mus_any = clm.make_oscil(frequency * fm3_rat)
    ampf: clm.mus_any = clm.make_env(amp_env, scaler = amplitude, duration = dur)
    indf1: clm.mus_any = clm.make_env(fm1_env, scaler = index1, duration = dur)
    indf2: clm.mus_any = clm.make_env(fm2_env, scaler = index2, duration = dur)
    indf3: clm.mus_any = clm.make_env(fm3_env, scaler = index3, duration = dur)
    frqf: clm.mus_any = clm.make_env(gliss_env, (glissando_amount * frq_scl), duration=dur)
    pervib: clm.mus_any = clm.make_triangle_wave(periodic_vibrato_rate,periodic_vibrato_amplitude * frq_scl)
    ranvib: clm.mus_any = clm.make_rand_interp(random_vibrato_rate, random_vibrato_amplitude * frq_scl)
    fm_noi: clm.mus_any = clm.make_rand(noise_freq, math.pi * noise_amount)
    ind_noi: clm.mus_any = clm.make_rand_interp(ind_noise_freq, ind_noise_amount)
    amp_noi: clm.mus_any = clm.make_rand_interp(amp_noise_freq, amp_noise_amount)
    loc: clm.mus_any = clm.make_locsig(degree, distance, reverb_amount)
    
    i: cython.int
    
    for i in range(start, end):
        vib: cython.double = clm.triangle_wave(pervib) + clm.rand_interp(ranvib) + clm.env(frqf)
        fuzz: cython.double = clm.rand(fm_noi)
        inoi: cython.double = 1.0 + clm.rand_interp(ind_noi)
        anoi: cython.double  = clm.env(ampf) * (1.0 + clm.rand_interp(amp_noi))
        clm.locsig(loc, i, anoi * clm.oscil(carrier, 
                        vib + 
                        inoi * 
                        ((clm.env(indf1) * clm.oscil(fmosc1, (vib * fm1_rat) + fuzz)) +
                        (clm.env(indf2) * clm.oscil(fmosc2, ((vib * fm2_rat) + fuzz))) +
                        (clm.env(indf3) * clm.oscil(fmosc3, ((vib * fm3_rat) + fuzz))))))


cdef nbd = [
    [1.0,             0.0,               0.0425,             0.3,    1.0               ], # hum (-2 octave)
    [1.000710732054,  0.0,               0.14083333333333,   0.3,    1.0               ],
    [1.6958066808813, 0.0,               0.25416666666667,   0.3,    0.46153846153846  ], # prime (-1 octave)
    [1.6966595593461, 0.0,               0.70416666666667,   0.3,    0.46153846153846  ], # tierce
    [2.2359630419332, 0.0,               0.79166666666667,   0.0,    0.17142857142857  ], # quint
    [2.9168443496802, 0.083333333333333, 0.031666666666667,  0.25,   0.08              ], # nominal
    [3.7029140014215, 1.0,               0.25,               0.1,    0.34285714285714  ],
    [4.3918123667377, 0.0,               0.29166666666667,   0.15,   0.1               ],
    [4.3951670220327, 0.0,               0.15,               0.15,   0.066666666666667 ],
    [4.680881307747,  0.14166666666667,  0.041666666666667,  0.0,    0.06              ],
    [5.455579246624,  0.29166666666667,  0.10833333333333,   0.5,    0.12              ],
    [6.1961620469083, 0.083333333333333, 0.025,              0.3,    0.12              ],
    [7.0717839374556, 0.0,               0.10833333333333,   0.05,   0.12              ],
    [7.3546552949538, 0.0,               0.19166666666667,   0.0,    0.048             ],
    [7.407249466951,  0.15,              0.061666666666667,  0.11,   0.13333333333333  ],
    [7.4103766879886, 0.058333333333333, 0.025,              0.11,   0.13333333333333  ],
    [9.4015636105188, 0.0,               0.045833333333333,  0.0,    0.034285714285714 ],
    [9.498223169865,  0.18333333333333,  0.15,               0.25,   0.024             ],
    [11.674484719261, 0.05,              0.0083333333333333, 0.5,    0.08              ],
    [13.384506041222, 0.010416666666667, 0.00625,            0.05,   0.024             ],
    [13.890547263682, 0.021666666666667, 0.021666666666667,  0.1,    0.08              ],
    [16.136460554371, 0.0125,            0.0041666666666667, 0.07,   0.1               ]
]

cpdef bell(beg:cython.double, dur: cython.double=8, freq: cython.double=351.75, amp: cython.double=0.3, deg: cython.double=45., dist: cython.double=0.0, rev:cython.double=0.0):
    cdef cython.int start = clm.seconds2samples(beg)
    cdef cython.int end   = start + clm.seconds2samples(dur)
    cdef clm.mus_any location =clm.make_locsig(degree=deg, distance=dist, reverb=rev)

    oscils_freqs_radians, envarray = [], []

    for b in nbd:
        hertz = round(b[0] * freq, 3)
        splashamp = round(b[1] * amp, 3)
        tailamp = round(b[2] * amp, 3)
        attack = round( b[3] if b[3] else .001, 3)
        decay = round(b[4] * dur, 3)
        # convert sc's envelope format [yyy][xxx] to sndlib's [xyxyxy]
        # Env.new([splashamp, tailamp*(2/3), tailamp, 0], [attack*(1/4), attack*(3/4), decay], 'sin');
        #        X                               Y
        envl =  [0,                              splashamp,  
                attack*(1/4),                    tailamp*(2/3), 
                attack*(1/4)+attack*(3/4),       tailamp,
                attack*(1/4)+attack*(3/4)+decay, 0]
        #print("SC:", [[round(attack*(1/4), 3) , round(attack*(3/4), 3), round(decay, 3)],
        #                  [round(splashamp, 3), round(tailamp*(2/3), 3), round(tailamp, 3), 0]])
        #print("PY:", envl)
        #plotenv(envl)
        
        envarray.append(clm.make_env(envl, scaler=amp, duration=dur))  
        
        #freqs in radians
        oscils_freqs_radians.append(clm.hz2radians(hertz) )

    # use oscil bank instead of python array of oscils    
    # starting phase. i am going to change make_oscil_bank so if you want all the values to just start at a 
    # certain phase you can just pass a scaler. i meant it to work that way but screwed it up
    
    cdef cython.double [:] starting_phases_ndarray = np.zeros(len(oscils_freqs_radians))
    # if an numpy array is used here the oscil_bank holds on to the array and any changes to it will
    # impact the underlying processing. 
    cdef cython.double [:] amps_ndarray = np.zeros(len(oscils_freqs_radians))
    cdef clm.mus_any oscils = clm.make_oscil_bank(oscils_freqs_radians, starting_phases_ndarray, amps_ndarray)

    # don't know if these save much to compute this once
    osc_bank_len = len(oscils_freqs_radians)
    
    cdef int i = 0
    
    for i in range(start, end):
        for j in range(osc_bank_len):
            amps_ndarray[j] = clm.env(envarray[j])
        clm.locsig(location, i, clm.oscil_bank(oscils))
        
        
        
# @cython.ccall
# def bell(beg:cython.double, dur: cython.double=8, freq: cython.double=351.75, amp: cython.double=0.3, deg: cython.double=45., dist: cython.double=0.0, rev:cython.double=0.0):
#     start: cython.long  = clm.seconds2samples(beg)
#     end: cython.long    = start + clm.seconds2samples(dur)
#     location: clm.mus_any =clm.make_locsig(degree=deg, distance=dist, reverb=rev)
# 
#     oscils_freqs_radians, envarray = [], []
# 
#     for b in nbd:
#         hertz = round(b[0] * freq, 3)
#         splashamp = round(b[1] * amp, 3)
#         tailamp = round(b[2] * amp, 3)
#         attack = round( b[3] if b[3] else .001, 3)
#         decay = round(b[4] * dur, 3)
#         # convert sc's envelope format [yyy][xxx] to sndlib's [xyxyxy]
#         # Env.new([splashamp, tailamp*(2/3), tailamp, 0], [attack*(1/4), attack*(3/4), decay], 'sin');
#         #        X                               Y
#         envl =  [0,                              splashamp,  
#                 attack*(1/4),                    tailamp*(2/3), 
#                 attack*(1/4)+attack*(3/4),       tailamp,
#                 attack*(1/4)+attack*(3/4)+decay, 0]
#         #print("SC:", [[round(attack*(1/4), 3) , round(attack*(3/4), 3), round(decay, 3)],
#         #                  [round(splashamp, 3), round(tailamp*(2/3), 3), round(tailamp, 3), 0]])
#         #print("PY:", envl)
#         #plotenv(envl)
#         
#         envarray.append(clm.make_env(envl, scaler=amp, duration=dur))  
#         
#         #freqs in radians
#         oscils_freqs_radians.append(clm.hz2radians(hertz) )
# 
#     # use oscil bank instead of python array of oscils    
#     # starting phase. i am going to change make_oscil_bank so if you want all the values to just start at a 
#     # certain phase you can just pass a scaler. i meant it to work that way but screwed it up
#     
#     starting_phases_ndarray: np.ndarray = np.zeros(len(oscils_freqs_radians))
#     # if an numpy array is used here the oscil_bank holds on to the array and any changes to it will
#     # impact the underlying processing. 
#     amps_ndarray: np.ndarray = np.zeros(len(oscils_freqs_radians))
#     oscils: clm.mus_any  = clm.make_oscil_bank(oscils_freqs_radians, starting_phases_ndarray, amps_ndarray)
# 
#     # don't know if these save much to compute this once
#     osc_bank_len = len(oscils_freqs_radians)
#     
#     i: cython.int = 0
#     
#     for i in range(start, end):
#         for j in range(osc_bank_len):
#             amps_ndarray[j] = clm.env(envarray[j])
#         clm.locsig(location, i, clm.oscil_bank(oscils))
#         
        
