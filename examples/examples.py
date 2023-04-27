#! /Users/toddingalls/Developer/Python/venvs/pysndlib-venv/bin/python


from musx import *
from pysndlib import *
from pysndlib.generators import *
import math
# import random
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import numpy as np
# import functools
# import types
# import librosa
# from numpy.random import default_rng
from pysndlib.jcrev import jc_reverb
from pysndlib.v import fm_violin
from pysndlib.clm_ins import pluck
from pysndlib.clm_ins import drone
from pysndlib.clm_ins import jl_reverb
from pysndlib.clm_ins import cellon
from pysndlib.clm_ins import reson
from pysndlib.clm_ins import gran_synth
from pysndlib.clm_ins import touch_tone
from pysndlib.clm_ins import spectra
from pysndlib.env import *


# with Sound("out1.aif", 1):
#     gen = make_oscil(440.0)
#     for i in range(44100):
#         outa(i,  oscil(gen))


# with Sound("out1.aif", 1):
#     gen = make_oscil(440.0)
#     ampf = make_env([0., 0., 0.01, 1.0, 0.25, 0.1, 1, 0], scaler=.5, length=44100)
#     for i in range(44100):
#         outa(i,  env(ampf)*oscil(gen))
    
#     
# with Sound("out1.aif", 1) as out:
#     gen = make_table_lookup(440.0, wave=partials2wave([1.0, 0.5, 2.0, 0.5, 3.0, .2, 5.0, .1]))
#     for i in range(44100):
#         outa(i, .5* table_lookup(gen))

# TODO Not right
# with Sound("out1.aif", 1):
#     #gen = make_polywave(400, partials=np.array([0.0,.7, 2., .3], dtype=np.double))
#     gen = make_polywave(440, partials=[1., .5, 2, .5])
#     print(gen._cache)
#     for i in range(44100):
#         outa(i, .1 * polywave(gen))


# with Sound("out1.aif", 1):
#     gen = make_triangle_wave(440.0)
#     for i in range(44100):
#         outa(i, .5 * triangle_wave(gen))

# with Sound("out1.aif", 1):
#     gen = make_ncos(440.0, 10);
#     for i in range(44100):
#         outa(i, .5 * ncos(gen))


# with Sound("out1.aif", 1):
#     gen = make_nrxycos(440.0, 1.1, 10, 0.3)
#     for i in range(44100):
#         outa(i, .5 * nrxycos(gen))

# with Sound("out1.aif", 1):
#     shifter = make_ssb_am(440.0, 20)
#     osc = make_oscil(440.0)
#     for i in range(44100):
#         outa(i, .5 * ssb_am(shifter, oscil(osc)))

# with Sound("out1.aif", 1) as out:
#     v = np.zeros(64, dtype=np.double)
#     g = make_ncos(400,10)
#     g.mus_phase = -.5 * np.pi
#     for i in range(64):
#         v[i] = ncos(g)
#     gen = make_wave_train(440, wave=v)    
#     for i in range(44100):
#         outa(i, .5 * wave_train(gen))
        
        

# with Sound("out1.aif", 2):
#     ran1 = make_rand(5.0, hz2radians(220.0))
#     ran2 = make_rand_interp(5.0, hz2radians(220.0))
#     osc1 = make_oscil(440.0)
#     osc2 = make_oscil(1320.0)
#     for i in range(88200):
#         outa(i, 0.5 * oscil(osc1, rand(ran1)))
#         outb(i, 0.5 * oscil(osc2, rand_interp(ran2)))
        
        
# with Sound("out1.aif", 1):
#     flt = make_two_pole(1000.0, 0.999)
#     ran1 = make_rand(10000.0, 0.002)
#     for i in range(44100):
#         outa(i, 0.5 * two_pole(flt, rand(ran1)))

# with Sound("out1.aif", 1):
#     flt = make_firmant(1000.0, 0.999)
#     ran1 = make_rand(10000.0, 5.0)
#     for i in range(44100):
#         outa(i, 0.5 * firmant(flt, rand(ran1)))

    
# with Sound("out1.aif", 1):
#     flt = make_iir_filter(3, [0.0, -1.978, 0.998])
#     ran1 = make_rand(10000.0, 0.002)
#     print(flt)
#     for i in range(44100):
#         outa(i, 0.5 * iir_filter(flt, rand(ran1)))

# with Sound("out1.aif", 1):
#     dly = make_delay(seconds2samples(0.5))
#     osc1 = make_oscil(440.0)
#     osc2 = make_oscil(660.0)
#     for i in range(44100):
#         outa(i, 0.5 * (oscil(osc1) + delay(dly, oscil(osc2))))        

# with Sound("out1.aif", 1):
#     cmb = make_comb(0.4, seconds2samples(0.4))
#     osc = make_oscil(440.0)
#     ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
#     
#     for i in range(88200):
#         outa(i, 0.5 * (comb(cmb, env(ampf) * oscil(osc))))

# with Sound("out1.aif", 1):
#     alp = make_all_pass(-0.4, 0.4, seconds2samples(0.4))
#     osc = make_oscil(440.0)
#     ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
#     for i in range(88200):
#         outa(i, 0.5 * (all_pass(alp, env(ampf) * oscil(osc))))
        

# with Sound("out1.aif", 1):
#     avg = make_moving_average(4410)
#     osc = make_oscil(440.0)
#     stop = 44100 - 4410
#     
#     for i in range(stop):
#         val = oscil(osc)
#         outa(i, val * moving_average(avg, abs(val))) 
#     for i in range(stop, 44100):
#         outa(i, oscil(osc) * moving_average(avg, 0.0))
        
    
# with Sound("out1.aif", 1):    
#     rd = make_readin("oboe.wav");
#     length = 2 * mus_sound_framples("oboe.wav")
#     sr = make_src(rd, .5);
#     
#     for i in range(length):
#         outa(i, src(sr))
    
    
# with Sound("out1.aif", 1):    
#     rd = make_readin("oboe.wav");
#     grn = make_granulate(rd, 2., hop=.3, jitter=1.4)
#     length = 2 * mus_sound_framples("oboe.wav")
#     for i in range(length):
#         outa(i, granulate(grn))
    
        
# with Sound("out1.aif", 1):    
#     rd = make_readin("oboe.wav");
#     pv = make_phase_vocoder(rd, pitch=1.3333)
#     length = 2 * mus_sound_framples("oboe.wav")
# 
#     for i in range(length):
#         outa(i, phase_vocoder(pv))    

# with Sound("out1.aif", 1):    
#     fm = make_asymmetric_fm(440.0, 0.0, 0.9, 0.5)
#     for i in range(44100):
#         outa(i, 0.5 * asymmetric_fm(fm, 1.0))
        
# with Sound("out1.aif", 1):    
#     reader = make_readin("oboe.wav")
#     
#     for i in range(44100):
#         outa(i, readin(reader))

        
# with Sound("out1.aif", 1):    
#     rate = .5
#     rd = make_readin("oboe.wav")
#     sr = make_src(rd, rate);
#     length = mus_sound_framples("oboe.wav")
#     grn = make_granulate(lambda x: src(sr), rate)
#     for i in range(length):
#         outa(i, granulate(grn))



# def jc_reverb(lowpass=False, volume=1., amp_env = None, tail=0):
#     allpass1 = make_all_pass(-.7, .7, 1051)
#     allpass2 = make_all_pass(-.7, .7, 337)
#     allpass3 = make_all_pass(-.7, .7, 113)
#     comb1 = make_comb(.742, 4799)
#     comb2 = make_comb(.733, 4999)
#     comb3 = make_comb(.715, 5399)
#     comb4 = make_comb(.697, 5801)
#     chans = Sound.output.mus_channels
#     
#     length = Sound.reverb.mus_length + seconds2samples(1.)
#     filts = [make_delay(seconds2samples(.013))] if chans == 1 else [make_delay(seconds2samples(.013)),make_delay(seconds2samples(.011)) ]
#     combs = make_comb_bank([comb1, comb2, comb3, comb4])
#     allpasses = make_all_pass_bank([allpass1,allpass2,allpass3])
#     
#     if lowpass or amp_env:
#         flt = make_fir_filter(3, [.25, .5, .25]) if lowpass else None
#         envA = make_env(amp_env, scaler=volume, duration = length / mus_srate())
#         
#         if lowpass:
#             for i in range(length):
#                 out_bank(filts, i, (env(envA) * fir_filter(flt, comb_bank(combs, all_pass(allpasses, ina(i, Sound.reverb))))))
#         else:
#             for i in range(length):
#                 out_bank(filts, i, (env(envA) * comb_bank(combs, all_pass_bank(allpasses, ina(i, Sound.reverb)))))
#     else:
#         if chans == 1:
#             
#             gen = filts[0]
#             for i in range(length):
#                 outa(i, delay(gen, volume * comb_bank(combs, all_pass_bank(allpasses, ina(i, Sound.reverb)))))
#         else:    
#             gen1 = filts[0]
#             gen2 = filts[1]
#             for i in range(length):
#                 val = volume * comb_bank(combs, all_pass_bank(allpasses, ina(i, Sound.reverb))) 
#                 outa(i, delay(gen1, val))
#                 outb(i, delay(gen2, val))
#             
# 
# 
# def blip(beg, dur, freq):
#     start = seconds2samples(beg)
#     end = seconds2samples(dur) + start
#     
#     loc = make_locsig(degree=random.randrange(-90, 90), distance=1., reverb=.7)
#     cmb = make_comb(0.4, seconds2samples(0.4))
#     osc = make_oscil(freq)
#     ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
#     
#     for i in range(start, end):
#         locsig(loc, i, 0.5 * (comb(cmb, env(ampf) * oscil(osc))))

# 
# 
# with Sound("out1.aif", 2, reverb=jc_reverb, reverb_channels=2, play=True):    
#     blip(0, 2, 100)
#     blip(1, 2, 400)
#     blip(2, 2, 200)
#     blip(3, 2, 300)
#     blip(4, 2, 500)
# 
# dump_mem()    
# 
# with Sound("out1.aif", 2, reverb=jc_reverb, reverb_channels=2, play=False):    
#     for i in range(1):
#         blip(i, 2, 100)
#         dump_mem()    
#         
#         
# #     blip(1, 2, 400)
# #     blip(2, 2, 200)
# #     blip(3, 2, 300)
# #     blip(4, 2, 500)
# 
# dump_mem()    
# print(gc.garbage)
# gc.collect()
# print(gc.garbage)
# dump_mem()


# with Sound("out1.aif", 2, reverb=jc_reverb, reverb_channels=2, play=False):    
#     blip(0, 2, 100)
#     blip(1, 2, 400)
#     blip(2, 2, 200)
#     blip(3, 2, 300)
#     blip(4, 2, 500)


#MUS_INPUT





















# 
# 
# print("1")
# mus_any_pointer=POINTER(mus_any)
# 
# RELEASE = CFUNCTYPE(UNCHECKED(None), POINTER(mus_any))
# DESCRIBE = CFUNCTYPE(UNCHECKED(String), POINTER(mus_any))
# EQUALP = CFUNCTYPE(UNCHECKED(c_bool), POINTER(mus_any), POINTER(mus_any))
# 
# 
# 
# def makeit():
#     
#     
#     @RELEASE
#     def osc_release(gen):
#         return gen
# 
#     @DESCRIBE
#     def osc_describe(gen):
#         return String(b"this is an osc")
#     
#     @EQUALP
#     def osc_equalp(gen1, gen2):
#         return TRUE
# 
#     osc_type = mus_make_generator_type()
#     osc = mus_make_generator(osc_type, "osc", osc_release, osc_describe, osc_equalp)
#     mus_generator_set_extended_type(osc, MUS_INPUT)
#     gen = mus_any_pointer()
#     
#     print(gen)
#     gen.core = osc
#     print("1")
#     #genp = mus_any_pointer()
#     #genp.contents = gen
# #    genp.core = osc
#     #print(genp.contents.core)
#     print("1")
#     return gen
#     
# c = makeit()
# print(c)
# print(mus_free(c))


# 
# out1 = np.zeros((1,88200*2), dtype=np.double)
# 
# 
# with Sound("out1.aif", 1, play=False):
#     cmb = make_comb(0.7, seconds2samples(0.2))
#     osc = make_triangle_wave(440.0)
#     ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
#     
#     for i in range(88200*2):
#         outa(i, 0.5 * (comb(cmb, env(ampf) * triangle_wave(osc))), out1)
# 
# fig, ax = plt.subplots(nrows=2, sharex=True)
# librosa.display.waveshow(out1, sr=44100, ax=ax[0])
# ax[0].set(title='Envelope view, mono')
# ax[0].label_outer()
# plt.show()



# a = default_rng().random((2,100))
# def rr(arr, c):
#     i = 0
#     length = np.shape(arr)[1]
#     while True:
#         yield arr[c][i]
#         i += 1
#         i %= length
#         
# k = rr(a, 0)
# 
# for i in range(500):
#     print(next(k))



# def fm(beg, end, freq, amp, mc_ratio, index):
#     carrier_phase = 0.0
#     carrier_phase_incr = hz2radians(freq)
#     modulator_phase_incr = hz2radians(mc_ratio * freq)
#     modulator_phase = .5 * (modulator_phase_incr + np.pi)
#     fm_index = hz2radians(mc_ratio * freq * index)
#     for i in range(beg, end):
#         modulation = fm_index * math.sin(modulator_phase)
#         fm_val = amp * math.sin(carrier_phase)
#         carrier_phase +=  modulation + carrier_phase_incr
#         modulator_phase += modulator_phase_incr
#         outa(i, fm_val)
#         
# with Sound("out.aif"):
#     fm(0, 44100, 1000, .5, .25, 4)
#     fm(44100, 44100*2, 1000, .5, .47, 4.8)
#     fm(44100*2, 44100*3, 1000, .5, .2, 3)

# def fm(beg, dur, freq, amp, mc_ratio, index, index_env=[0,1,100,1]):
#     start = seconds2samples(beg)
#     end = start + seconds2samples(dur)
#     cr = make_oscil(freq)
#     md = make_oscil(freq * mc_ratio)
#     fm_index = hz2radians(index * mc_ratio * freq)
#     ampf = make_env(index_env, scaler=amp, duration=dur)
#     indf = make_env(index_env, scaler=fm_index, duration=dur)
#     
#     for i in range(start, end):
#         outa(i, env(ampf) * oscil(cr, env(indf) * oscil(md)))
# #         
# #         
# with Sound("out.aif"):
#     fm(0, 5, 1000, .5, .25, 4, [0, 1, 100, .1])
# 
# import matplotlib.pyplot as plt
# y, sr = librosa.load("out.aif")
# fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
# D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
# img = librosa.display.specshow(D, y_axis='linear', x_axis='time',sr=sr, ax=ax[0])
# librosa.display.waveshow(y, sr=sr, ax=ax[1])
# ax[0].set(title='Linear-frequency power spectrogram')
# ax[0].label_outer()
# ax[1].set(title='Waveform')
# ax[1].label_outer()
# plt.show()
#     
    



# 
# def machine1(beg, dur, cfreq, mfreq, amp, index, gliss):
#     gen = make_fmssb(frequency=cfreq, ratio= (mfreq/cfreq), index=1)
#     start = seconds2samples(beg)
#     stop = start + seconds2samples(dur)
#     ampf = make_env([0, 0, 1, .75, 2, 1, 3, .1, 4, .7, 5, 1, 6, .8, 100, 0], base=32, scaler=amp, duration=dur)
#     indf = make_env([0,0,1,1,3,0], duration=dur, base=32, scaler=index)
#     frqf = make_env( [0,0,1,1] if gliss > 0. else [0,1,1,0], duration=dur, scaler=hz2radians(math.fabs(gliss)))
#     for i in range(start, stop):
#         gen.index = env(indf)
#         outa(i, env(ampf)*fmssb(gen, env(frqf)))
#         
#         
#         
# 
#         
# # with Sound("out1.aif"):
# #     for i in np.arange(0, 2, .2):
# #         machine1(i, .3, 100, 540, .5, 4.0, 0.)
# #         machine1(i+.1, .3, 200, 540, .5, 3.0, 0.0)
# # 
# #     for i in np.arange(0., 2., .6):
# #         machine1(i, .3, 1000, 540, .5, 6., 0.)
# #         machine1(i+.1, .1, 2000, 540, .5, 1.0, 0)
# 
# 
# 
# with Sound('out1.aif'):
#     gen = make_rkoddssb(frequency=1000.0, ratio=2.0, r=.875)
#     noi = make_rand(1500, .04)
#     gen1 = make_rkoddssb(frequency=100.0, ratio=.1, r=.9)
#     ampf = make_env([0, 0, 1, 1, 11, 1, 12, 0],duration=11.0,scaler=.5)
#     frqf = make_env([0, 0, 1, 1, 2, 0, 10, 0, 11, 1, 12, 0, 20, 0], duration=11.0, scaler=hz2radians(1.0))
#     for i in range(0, 12*44100):
#         outa(i, .4 * env(ampf) * (rkoddssb(gen1, env(frqf)) + (2.0 * math.sin(rkoddssb(gen, rand(noi))))))
#         
#     for i in range(0, 10, 6):
#         machine1(i, 3, 1000, 540, .5, 6., 0.)
#         machine1(i+1, 1, 2000, 540, .5, 1., 0.)

#{'frequency' : 0.0, 'r' : .5, 'osc' : False, 'rr' : 0.0, 'norm' : 0.0, 'rrp1' : 0.0, 'rrm1' : 0.0, 'r1' : 0}

# # 
# def stringy(beg, dur, freq, amp):
#     n = math.floor(get_srate() / (3 * freq))
#     start = seconds2samples(beg)
#     stop = seconds2samples(beg + dur)
#     r = math.pow(.001, (1. / n))
#     carrier = make_rcos(frequency = freq, r = (r * .5))
#     clang = make_rkoddssb(frequency = (freq * 2), ratio = (1.618 / 2), r=r)
#     ampf = make_env([0, 0, 1,1, 2, .5, 4, .25, 10, 0], scaler = amp, duration = dur)
#     clangf = make_env([0,0, .1, 1, .2, .1, .3, 0], scaler = amp*.5, duration = .1)
#     rf = make_env([0,1,1,0], scaler = .5*r, duration = dur)
#     crf = make_env([0,1,1,0], scaler = r, duration = .1)
#     
#     for i in range(start, stop):
#         clang.mus_scaler = env(crf)
#         carrier.r = env(rf)
#         outa(i, (env(clangf) * rkoddssb(clang, 0.0)) + (env(ampf) * rcos(carrier, 0.0)))
#         
# #         
# #         
# with Sound("out1.aif"):
#     stringy(0, 1, 120, .5)     
# with Sound("out1.aif"):
#     for i in range(0,10):
#         stringy(i*.3, .5, 100 + (100 * i), .2)


# 

#     
# with Sound('out1.aif', reverb=jc_reverb, statistics=True):
# #     violin(0, 3., 440, .1, 1.0)
# #     violin(3, 3., 440*1.333333, .1, 1.0)
# #     violin(4, 3., 110, .1, 1.0)
#     fm_violin(0, 3., 110, .1, 1.111, fm1_env=[0., 0, 20., .8, 75., .9, 100., 0.], 
#                                 fm2_rat=3.7, 
#                                 reverb_amount=.2, 
#                                 random_vibrato_amplitude=.01,
#                                 glissando_amount=.05,
#                                 gliss_env=[0.0, 0.0, 50, .1, 100, .0],
#                                 noise_amount=.01, noise_freq=.1,
#                                 amp_noise_amount=.1,
#                                 ind_noise_amount=.1)
#     fm_violin(3, 3., 110, .1, 1.111, reverb_amount=.2, amp_noise_amount=.1, amp_noise_freq=2)
#     fm_violin(4.5, 3., 225, .1, 1.111, reverb_amount=.2, amp_noise_amount=.1, amp_noise_freq=2)
#     
# #     for i in np.arange(6,10,.1):
# #         violin(i, .1, 110, .1, 1,amp_env=[0,1,40,.1,100,0], reverb_amount=(i/10)*.1)
#     
#     
# 
#     
#     
    

    
# fbell =[0, 1, 2, 1.1000, 25, .7500, 75, .5000, 100, .2000 ]
# abell = [0, 0 ,.1000, 1 ,10, .6000, 25, .3000, 50, .1500, 90, .1000, 100, 0 ]
# with Sound('out1.aiff'):
#     fm_bell(0.0, 1.0, 220.0, .5, abell, fbell, 1.0)

#def pluck(start, dur, freq, amp, weighting=.5, lossfact=.9):
# with Sound('out1.aiff', statistics=True):
#     pluck(0.0, 5.0, 110.0, .5, weighting=.9, lossfact=.999)
#     pluck(0.0, 5.0, 110.5, .5, weighting=.9, lossfact=.999)
#     pluck(0.0, 5.0, 111.0, .5, weighting=.8, lossfact=.989)


# def tuneIt(f, s1):
#     def getOptimumC(S,o,p):
#         pa = (1.0 / o) * math.atan2((S * math.sin(o)), (1.0 + (S * math.cos(o))) - S)
#         tmpInt = math.floor(p - pa)
#         pc = p - pa - tmpInt
#         
#         if pc < .1:
#             while pc >= .1:
#                 tmpInt = tmpInt - 1
#                 pc += 1.0
#         return [tmpInt, (math.sin(o) - math.sin(o * pc)) / math.sin(o + (o * pc))]
#         
#     p = get_srate() / f
#     s = .5 if s1 == 0.0 else s1
#     o = hz2radians(f)
#     vals = getOptimumC(s,o,p)
#     vals1 = getOptimumC((1.0 - s), o, p)
#     
#     if s != 1/2 and math.fabs(vals[1]) < math.fabs(vals1[1]):
#         return [(1.0 - s), vals[1], vals[0]]
#     else:
#         return [s, vals1[1], vals1[0]]    
#         
#         
# print(tuneIt(440, .5))



# with Sound('out1.aif'):
#     fofins(0, 1, 270, 0.2, 0.001, 730, 0.6, 1090, 0.3, 2440, 0.1)
#     fofins(1, 1, 270, 0.2, 0.001, 260, 0.6, 3500, 0.3, 3800, 0.1)
#     fofins(2, 3, 270, 0.2, 0.001, 300, 0.6, 870, 0.3, 2440, 0.1)

# with Sound('out1.aif', statistics=True):
#     for i in np.arange(0,1, .1):
#         fm_trumpet(i, .2, 250, 1500)
# 
# 
# x = stretch_envelope([0,1,25,.1, 75, 0, 100, 0], 25, 45, 75, 40)
# 
# with Sound('out1.aif', srate=22050):
#     locust  = [0, 0, 40, 1, 95, 1, 100, .5]
#     bug_hi 	= [0, 1, 25, .7, 75, .78, 100, 1]
#     amp = [0, 0, 25, 1, 75, .7, 100, 0]
#     fm_insect( 0,      1.699,  4142.627,  .015, amp, 60, -16.707, locust, 500.866, bug_hi,  .346,  .500)
#     fm_insect( 0.195,   .233,  4126.284,  .030, amp, 60, -12.142, locust, 649.490, bug_hi,  .407,  .500)
#     fm_insect( 0.217,  2.057,  3930.258,  .045, amp, 60, -3.011,  locust, 562.087, bug_hi,  .591,  .500)
#     fm_insect( 2.100,  1.500,   900.627,  .06,  amp, 40, -16.707, locust, 300.866, bug_hi,  .346,  .500)
#     fm_insect( 3.000,  1.500,   900.627,  .06,  amp, 40, -16.707, locust, 300.866, bug_hi,  .046,  .500)
#     fm_insect( 3.450,  1.500,   900.627,  .09,  amp, 40, -16.707, locust, 300.866, bug_hi,  .006,  .500)
#     fm_insect( 3.950,  1.500,   900.627,  .12,  amp, 40, -10.707, locust, 300.866, bug_hi,  .346,  .500)
#     fm_insect( 4.300,  1.500,   900.627,  .09,  amp, 40, -20.707, locust, 300.866, bug_hi,  .246,  .500)
# 
# with Sound('out1.aif', statistics=True):
#     fm_drum(0 ,1.5, 55, .3, 5 ,False)
#     fm_drum(2, 1.5, 66, .3, 4, True)

# with Sound('out1.aif', statistics=True):
#     gong(0,3,261.61, .6)
# 
# with Sound('out1.aif', statistics=True):
#     attract(0,3,.4, 2)

# with Sound('out1.aif', statistics=True):
#     pqw(0, 2, 200, 943, .1, [0,0,25,1,100,0], [0,1,100,0], [2,.1,3,.3,6,.5])


# with Sound('out1.aif', statistics=True):
#     tubebell(0,3, 400, .2)
#     tubebell(0,3, 500, .2)

# with Sound('out1.aif', statistics=True):
#     rhodey(0, 3, 261, .3)
#     rhodey(0, 3, 440, .3)
    
# with Sound('out1.aif', statistics=True):
#     hammondoid(0, 3, 261, .3)
#     hammondoid(0, 3, 440, .3)

# with Sound('out1.aif', statistics=True):
#     metal(0, 3, 261, .3)
#     metal(0, 3, 440, .3)
    
    
# solid = [0,0,5,1,95,1,100,0]
# bassdr2 = [.5,.06,1,.62,1.5,.07,2.0,.6,2.5,.08,3.0,.56,4.0,.24,
# 		5,.98,6,.53,7,.16,8,.33,9,.62,10,.12,12,.14,14,.86,
# 		16,.12,23,.14,24,.17]
# 		
# tenordr = [.3,.04,1,.81,2,.27,3,.2,4,.21,5,.18,6,.35,7,.03,8,.07,9,.02,10,.025,11,.035]
# #     
# # 
# # 
# # 
# with Sound('out1.aif', statistics=True, reverb=jc_reverb, channels=2, reverb_channels=2):
#     drone(0, 4, 115, .125, solid, bassdr2, .1, .5, .30, 45, 2, .010, 10)
#     drone(0, 4, 229, .125, solid, tenordr, .1, .5, .30, 45, 2, .010, 10)
#     drone(0, 4, 229.5, .5, solid, tenordr, .1, .5, .30, 45, 2, .010, 10)
#     


    
# with Sound('out1.aif', statistics=True, reverb=jl_reverb):
#     reson(0,1.0,440,.1,2,[0,0,100,1],[0,0,100,1],.1,.1,.1,5,.01,5,.01,0,1.0,0.1,[[[0,0,100,1],1200,.5,.1,.1,0,1.0,.1,.1],[[0,1,100,0],2400,.5,.1,.1,0,1.0,.1,.1]])
#     reson(1,2.0,440,.1,2,[0,0,100,1],[0,0,100,1],.1,.1,.1,5,.01,5,.01,0,1.0,0.1,[[[0,0,100,1],1500,.5,.1,.1,0,1.0,.1,.1],[[0,1,100,0],700,.5,.1,.1,0,1.0,.1,.1]])
    
# cellon 3 1 220 .1 '(0 0 25 1 75 1 100 0) '(0 0 25 1 75 1 100 0) .75 1.0 0 0 0 0 1 0 0 220 
# 	    '(0 0 25 1 75 1 100 0) 0 0 0 0 '(0 0 100 0) 0 0 0 0 '(0 0 100 0))
    

# with Sound('out1.aif', statistics=True, reverb=jc_reverb):
#     cellon(0,1,220,.1,[0,0,25,1,75,1,100,0],[0,0,25,1,75,1,100,0],.75,1.0,0,0,0,0,1,0,0,220,[0,0,25,1,75,1,100,0],0,0,0,0,[0,0,100,0],0,0,0,0,[0,0,100,0])
#     cellon(1,1,220*2,.1,[0,0,25,1,75,1,100,0],[0,0,25,1,75,1,100,0],.75,1.0,0,0,0,0,1,0,0,220,[0,0,25,1,75,1,100,0],0,0,0,0,[0,0,100,0],0,0,0,0,[0,0,100,0])
#     cellon(2,1,220*1.5,.1,[0,0,25,1,75,1,100,0],[0,0,25,1,75,1,100,0],.75,1.0,0,0,0,0,1,0,0,220,[0,0,25,1,75,1,100,0],0,0,0,0,[0,0,100,0],0,0,0,0,[0,0,100,0])
#     cellon(3,1,220*1.3333,.1,[0,0,25,1,75,1,100,0],[0,0,25,1,75,1,100,0],.75,1.0,0,0,0,0,1,0,0,220,[0,0,25,1,75,1,100,0],0,0,0,0,[0,0,100,0],0,0,0,0,[0,0,100,0])
    
    
# with Sound('out1.aif', statistics=True, reverb=jc_reverb):
#     gran_synth(0,2,400,.0189,.02,.4)
 
# with Sound('out1.aif', statistics=True, reverb=jc_reverb):    
#     touch_tone( 0.0, [9,5,1,8,8,5,8])

# with Sound('out1.aif', statistics=True, reverb=jc_reverb):  
#     spectra(0, 1, 440.0, .1, [1.0,.4,2.0,.2,3.0,.2,4.0,.1,6.0,.1], 
#                [0.0,0.0,1.0,1.0,5.0,0.9,12.0,0.5,25.0,0.25,100.0,0.0])


# from types import SimpleNamespace
# 
# def make_oscil(freq, phase=0.0):
#     return SimpleNamespace(freq=hz2radians(freq), phase=phase)
#     
# def oscil(gen,fm=0):
#     result = gen.phase
#     gen.phase += (gen.freq + fm)
#     return math.sin(result)
# 
#     

# outarray = np.zeros((2,44100*10), dtype=np.double)
# #print(outarray)
# # 
# # def test(start, end, freq):
# #     osc = make_oscil(freq)
# #     oscs = [make_oscil(i) for i in range(10)]
# #     beg = seconds2samples(start)
# #     nd = seconds2samples(end)
# #     
# #     for i in range(beg, nd):
# #         outa(i, oscil(osc, .01*oscil(oscs[0], .01*oscil(oscs[1], .01*oscil(oscs[2], 
# #                             .01*oscil(oscs[3], .01*oscil(oscs[4], 
# #                             .01*oscil(oscs[5], .01*oscil(oscs[6], 
# #                             .01*oscil(oscs[7], .01*oscil(oscs[9])))))))))))
# #           
# # 
# CLM.play = True
# CLM.srate = 44100
# 
# def blip(beg, dur, freq):
#     start = seconds2samples(beg)
#     end = seconds2samples(dur) + start
#     
#     loc = make_locsig(degree=45, distance=1., reverb=.7)
#     osc = make_oscil(freq)
#     ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
#     
#     for i in range(start, end):
#         val = 0.1 * env(ampf) * oscil(osc)
#         locsig(loc, i, val)
# 
# 
# # 
# # # 
# #                             
# with Sound(outarray, channels=2, reverb=jc_reverb, statistics=True): 
#     blip(0,1, 440)
#     blip(0,1, 441)
#     blip(2,1, 880)
#     blip(2,1, 882)
#     blip(7,2, 1000)
#     blip(7,2, 1002)
#     



# class PEvent():
#     def __init__(self, func):
#         self.func = func
#     
#     def __call__(self, *args, **kwargs):
#         self.func(*args, **kwargs)
# 
# 
# @PEvent
# def mm(time):
#     printw("time is ", time)
# 
# a = PEvent(1.)
# 
# seq = Seq()
# sco = Score(out=seq)
# 
# def p(sco, num):
#     for i in range(num):
#         k = mm(time=sco.now)
# 
#         sco.add(k)
#         yield .125
#         
#         
# sco.compose(p(sco, 10))
# 
# def writep(seq):
#     for v in seq:
#         v()
#         
#         
# writep(seq)




    
# class PEvent():
#     def __init__(self, func):
#         self.func = func
#         self.time = 0
#     def __call__(self, *args, **kwargs):
#         self.func(*args, **kwargs)

#    
# def clm_instrument(func):
#     @functools.wraps(func)
#     def call(*args, **kwargs):
#         return PEvent(functools.partial(func, *args, **kwargs))
#         
#     return call
    
   

# def clm_instrument(func):
#     @functools.wraps(func)
#     def create(*args, **kwargs):
#         def call():
#             func(*args, **kwargs)
#         
#         return call
#     return create        
    

    

#
# def blip(beg, dur, freq):
#     start = seconds2samples(beg)
#     end = seconds2samples(dur) + start
#     
#     loc = make_locsig(degree=45, distance=1., reverb=.1)
#     osc = make_oscil(freq)
#     ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
#     
#     for i in range(start, end):
#         val = 0.1 * env(ampf) * oscil(osc)
#         locsig(loc, i, val)
        
# with Sound('out3.aif', channels=2, reverb=jc_reverb, statistics=True, play=True): 
#     blip(0,1, 440)
#     blip(0,1, 441)
#     blip(2,1, 880)
#     blip(2,1, 882)
#     blip(7,2, 1000)
#     blip(7,2, 1002)

# seq = Seq()
# sco = Score(out=seq)
# 
# def p(sco, num):
#     for i in range(num):
#         k = blip(sco.now, 2, hertz(60+i))
#         sco.add(k)
#         yield .5
#   
# sco.compose(p(sco, 10))
# 
# print(sco)
# seq.print()

# def writep(seq):
#     with Sound("out2.aif",channels=2, reverb=jc_reverb, statistics=True, play=True):
#         for v in seq:
#             v()  
#         
#         
# writep(seq)
    
# @clm_instrument
# def dda(a,b=2):
#     print(a+b)
# 
# seq = Seq()
# sco = Score(out=seq)
# 
# def p(sco, num):
#     for i in range(num):
#         k = dda(sco.now)
#         sco.add(k)
#         yield .125
#         
#         
# sco.compose(p(sco, 10))
# 
# def writep(seq):
#     for v in seq:
#         v()
#         
#         
# writep(seq)
        


    
# blipx = clm_instrument(blip)
# 
# 
# seq = Seq()
# sco = Score(out=seq)
# 
# def p(sco, num):
#     for i in range(num):
#         k = blipx(sco.now, 1, 200*i)
#         sco.add(k)
#         yield .125
#         
#         
# sco.compose([[0,p(sco, 10)], [.2, p(sco, 10)]])
# # 
# #seq.print()
# #def perf(seq):
#    
# #         
# #         
# # perf(seq)
# 
# render_clm(seq, "out1.aiff", reverb=jc_reverb, play=True, statistics=True)
# 
# write_clm(seq)



#         
#         
# write(seq)


def blip(beg, dur, freq):
    start = seconds2samples(beg)
    end = seconds2samples(dur) + start
    
    loc = make_locsig(degree=45, distance=1., reverb=.1)
    osc = make_oscil(freq)
    ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
    
    for i in range(start, end):
        val = 0.1 * env(ampf) * oscil(osc)
        locsig(loc, i, val)


    
blip = clm_instrument(blip)


seq = Seq()
sco = Score(out=seq)

def p(sco, num, t):
    for i in range(num):
        k = blip(sco.now, 1, 200*i)
        sco.add(k)
        yield t
        
        
sco.compose([[0,p(sco, 10, .125)], [.1, p(sco, 10, .2)]])
# 
#seq.print()
#def perf(seq):
   
#         
#         
# perf(seq)

render_clm(seq, 'out1.aif', reverb=None, play=True, statistics=True)
