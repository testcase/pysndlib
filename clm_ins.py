#! /Users/toddingalls/Developer/Python/venvs/pysndlib-venv/bin/python
import math
import numpy as np
from pysndlib import *
from .env import stretch_envelope
from .env import interleave
# from pysndlib.env import stretch_envelope
# --------------- pluck ---------------- #

# could just use multiple value return
def tuneIt(f, s1):
    def getOptimumC(S,o,p):
        pa = (1.0 / o) * math.atan2((S * math.sin(o)), (1.0 + (S * math.cos(o))) - S)
        tmpInt = math.floor(p - pa)
        pc = p - pa - tmpInt
        
        if pc < .1:
            while pc >= .1:
                tmpInt = tmpInt - 1
                pc += 1.0
        return [tmpInt, (math.sin(o) - math.sin(o * pc)) / math.sin(o + (o * pc))]
        
    p = get_srate() / f
    s = .5 if s1 == 0.0 else s1
    o = hz2radians(f)
    vals = getOptimumC(s,o,p)
    vals1 = getOptimumC((1.0 - s), o, p)
    
    if s != 1/2 and math.fabs(vals[1]) < math.fabs(vals1[1]):
        return [(1.0 - s), vals[1], vals[0]]
    else:
        return [s, vals1[1], vals1[0]]



def pluck(start, dur, freq, amp, weighting=.5, lossfact=.9):
    vals = tuneIt(freq, weighting)
    wt0 = vals[0]
    c = vals[1]
    dlen = vals[2]
    beg = seconds2samples(start)
    end = seconds2samples(start + dur)
    lf = 1.0 if (lossfact == 0.0) else min(1.0, lossfact)
    wt = .5 if (wt0 == 0.0) else min(1.0, wt0)
    tab = np.zeros(dlen, dtype=np.double)
    allp = make_one_zero(lf * (1.0 - wt), (lf * wt))
    feedb = make_one_zero(c, 1.0)
    c1 = 1.0 - c
    
    for i in range(0,dlen):
        tab[i] = mus_random(1.0) # could replace with python random
        
    ctr = 0
    for i in range(beg, end):
        ctr = (ctr + 1) % dlen
        val = (c1 * (one_zero(feedb, one_zero(allp, tab[ctr]))))
        tab[ctr] = val
        outa(i, amp * val)


# TODO: --------------- mlbvoi ---------------- #




vox_formants = {
    'I' : ( 390, 1990, 2550),
    'E' : ( 530, 1840, 2480),
    'AE' : ( 660, 1720, 2410),
    'UH' : ( 520, 1190, 2390),
    'A' : ( 730, 1090, 2440),
    'OW' : ( 570, 840, 2410),
    'U' : ( 440, 1020, 2240),
    'OO' : ( 300, 870, 2240),
    'ER' : ( 490, 1350, 1690),
    'W' : ( 300, 610, 2200),
    'LL' : ( 380, 880, 2575),
    'R' : ( 420, 1300, 1600),
    'Y' : ( 300, 2200, 3065),
    'EE' : ( 260, 3500, 3800),
    'LH' : ( 280, 1450, 1600),
    'L' : ( 300, 1300, 3000),
    'I2' : ( 350, 2300, 3340),
    'B' : ( 200, 800, 1750),
    'D' : ( 300, 1700, 2600),
    'G' : ( 250, 1350, 2000),
    'M' : ( 280, 900, 2200),
    'N' : ( 280, 1700, 2600),
    'NG' : ( 280, 2300, 2750),
    'P' : ( 300, 800, 1750),
    'T' : ( 200, 1700, 2600),
    'K' : ( 350, 1350, 2000),
    'F' : ( 175, 900, 4400),
    'TH' : ( 200, 1400, 2200),
    'S' : ( 200, 1300, 2500),
    'SH' : ( 200, 1800, 2000),
    'V' : ( 175, 1100, 2400),
    'THE' : ( 200, 1600, 2200),
    'Z' : ( 200, 1300, 2500),
    'ZH' : ( 175, 1800, 2000),
    'ZZ' : ( 900, 2400, 3800),
    'VV' : ( 565, 1045, 2400)
}

# caar (car (car x))  [0][0]
# cdar (cdr (car x)) [1:][0]
# cadr (car (cdr x)) [1]
# caddr (car (cdr (cdr x))) [2]


# 
# def vox_fun(phons, which):
#     
#     def find_phoneme(phoneme, forms):
#                 
#     
#     pass
# 
# 
# 
# def vox(beg, dur, freq, amp, ampfun, freqfun, freqscl, phonemes, formant_amps, formant_indices,vibscl=.1, deg=0, pcrev=0):
#     start = seconds2samples(beg)
#     end = seconds2samples(beg + dur)
#     car_os = make_oscil(0)
#     fs = len(formant_amps)
#     per_vib = make_triangle_wave(6, amplitude=hz2radians(freq * vibscl))
#     ran_vib = make_rand(20, amplitude=hz2radians(freq * .5 * vibscl))
#     freqf = make_env(freqfun, duration=dur, scaler=hz2radians(freqscl * freq), offset=hz2radians(freq))
#     
#     # simple case
#     a0 = make_env(ampfun, scaler=amp*formant_amps[0], duration=dur)
#     a1 = make_env(ampfun, scaler=amp*formant_amps[1], duration=dur)
#     a2 = make_env(ampfun, scaler=amp*formant_amps[2], duration=dur)
#     o0 = make_oscil(0.0)
#     o1 = make_oscil(0.0)
#     o2 = make_oscil(0.0)
#     e0 = make_oscil(0.0)
#     e1 = make_oscil(0.0)
#     e2 = make_oscil(0.0)
#     ind0 = formant_indices[0]
#     ind1 = formant_indices[1]
#     ind2 = formant_indices[2]
#     f0 = make_env
    


# TODO: --------------- pqwvox ---------------- #

# --------------- fof ---------------- #


def fofins(beg, dur, frq, amp, vib, f0, a0, f1, a1, f2, a2, ae=[0, 0, 25, 1, 75, 1, 100,0], ve=[0,1,100,1]):
    foflen = math.floor(get_srate() / 220.5)
    start = seconds2samples(beg)
    end = seconds2samples(beg + dur)
    ampf = make_env(ae, scaler=amp, duration=dur)
    vibf = make_env(ve, scaler=vib, duration=dur)
    frq0 = hz2radians(f0)
    frq1 = hz2radians(f1)
    frq2 = hz2radians(f2)
    vibr = make_oscil(0)
    foftab = np.zeros(foflen)
    win_freq = ((2.0 * math.pi) / foflen)
    for i in range(foflen):
        v = (a0 * math.sin(i * frq0)) + (a1 * math.sin(i * frq1)) + (a2 * math.sin(i * frq2))
        foftab[i] = v * .5 * (1.0 - math.cos(i * win_freq))
    wt0 = make_wave_train(frq, foftab)
    
    for i in range(start, end):
        outa(i, env(ampf) * wave_train(wt0, (env(vibf) * oscil(vibr))))


# --------------- fm_trumpet ---------------- #

def fm_trumpet(startime, dur, frq1=250, frq2=1500, amp1=.5, amp2=.1, 
    ampatt1=.03, ampdec1=.35,
    ampatt2=.03, ampdec2=.3,
    modfrq1=250,
    modind11=0.0,
    modind12=2.66,
    modfrq2=250.,
    modind21=0.0,
    modind22=1.8,
    rvibamp=.007,
    rvibfrq=125.0,
    vibamp=.007,
    vibfrq=7.0,
    vibatt=.6,
    vibdec=.2,
    frqskw=.03,
    frqatt=.06,
    ampenv1=[0,0,25,1,75,.9,100,0],
    ampenv2=[0,0,25,1,75,.9,100,0],
    indenv1=[0,0,25,1,75,.9,100,0],
    indenv2=[0,0,25,1,75,.9,100,0],
    degree=0.0,
    distance=1.0,
    reverb_amount=.005):
    
    dec_01 = max(75, (100 * (1.0 - (.01 / dur))))
    beg = seconds2samples(startime)
    end = seconds2samples(startime + dur)
    loc = make_locsig(degree, distance, reverb_amount)
    vibe = stretch_envelope([0,1,25,.1, 75, 0, 100, 0], 25, min((100 * (vibatt / dur)), 45), 75, max((100 * (1.0 - (vibdec / dur)), 55)))

    per_vib_f = make_env(vibe, scaler=vibamp, duration=dur)
    ran_vib = make_rand_interp(rvibfrq, rvibamp)
    per_vib = make_oscil(vibfrq)
    frqe = stretch_envelope([0,0,25,1,75,1,100,0], 25, min(25, (100 * (frqatt / dur))), 75, dec_01)
    frq_f = make_env(frqe, scaler=frqskw, offset=1.0, duration=dur)
    ampattpt1 = min(25, (100 * (ampatt1 / dur)))
    ampdecpt1 = max(75, (100 * (1.0 - (ampdec1 / dur))))
    ampattpt2 = min(25, (100 * (ampatt2 / dur)))
    ampdecpt2 = max(75, (100 * (1.0 - (ampdec2 / dur))))
    
    mod1_f = make_env(stretch_envelope(indenv1, 25, ampattpt1, 75, dec_01), scaler=(modfrq1 * (modind12 - modind11)), duration=dur)
    mod1 = make_oscil(0.0)
    car1 = make_oscil(0.0)
    car1_f = make_env(stretch_envelope(ampenv1, 25, ampattpt1, 75, ampdecpt1), scaler=amp1, duration=dur)
    
    mod2_f = make_env(stretch_envelope(indenv2, 25, ampattpt2, 75, dec_01), scaler=(modfrq2 * (modind22 - modind21)), duration=dur)

    mod2 = make_oscil(0.0)
    car2 = make_oscil(0.0)
    car2_f = make_env(stretch_envelope(ampenv2, 25, ampattpt2, 75, ampdecpt2), scaler=amp2, duration=dur)
    
    for i in range(beg, end):
        frq_change = hz2radians((1.0 + rand_interp(ran_vib)) * 
                            (1.0 + (env(per_vib_f) * oscil(per_vib))) *
                            env(frq_f))
                            
                            
        locsig(loc, i, env(car1_f) * oscil(car1, frq_change * (frq1 + env(mod1_f) * oscil(mod1, modfrq1 * frq_change))) +
                        env(car2_f) * oscil(car2, frq_change * (frq2 + env(mod2_f) * oscil(mod2, modfrq2 * frq_change))))
#     

# TODO: --------------- stereo_flute ---------------- #

# --------------- fm_bell ---------------- #

def fm_bell(startime, dur, frequency, amplitude, 
        amp_env=[0, 0, .1, 1, 10, .6, 25, .3, 50, .15, 90, .1, 100, 0], 
        index_env=[0, 1, 2, 1.1, 25, .75, 75, .5, 100, .2], index=1.):
    fmInd2 = hz2radians( (8.0 - (frequency / 50.)) * 4)
    beg = seconds2samples(startime)
    end = seconds2samples(startime + dur)
    fmInd1 = hz2radians(32.0 * frequency)
    fmInd3 = fmInd2 * .705 * (1.4 - (frequency / 250.))
    fmInd4 = hz2radians(32.0 * (20  - (frequency / 20)))
    mod1 = make_oscil(frequency * 2)
    mod2 = make_oscil(frequency * 1.41)
    mod3 = make_oscil(frequency * 2.82)
    mod4 = make_oscil(frequency * 2.4)
    car1 = make_oscil(frequency)
    car2 = make_oscil(frequency)
    car3 = make_oscil(frequency * 2.4)
    indf = make_env(index_env, scaler=index, duration=dur)
    ampf = make_env(amp_env, scaler=amplitude, duration=dur)
    
    for i in range(beg, end):
        fmenv = env(indf)
        outa(i, env(ampf) * (oscil(car1, fm=(fmenv * fmInd1 * oscil(mod1))) +
                             (.15 * oscil(car2, fm=(fmenv * ( (fmInd2 * oscil(mod2)) + (fmInd3 * oscil(mod3)) )))) +
                             (.15 * oscil(car3, fm=(fmenv * fmInd4 * oscil(mod4)))) ))
                             

        
        
# --------------- fm_insect ---------------- #

def fm_insect(startime, dur, frequency, 
                amplitude, 
                amp_env, 
                mod_freq, 
                mod_skew, 
                mod_freq_env, 
                mod_index, 
                mod_index_env, 
                fm_index, 
                fm_ratio, 
                degree=0.0, 
                distance=1.0,
                reverb_amount=.005):
                
    beg = seconds2samples(startime)
    end = seconds2samples(startime + dur)
    loc = make_locsig(degree, distance, reverb_amount)
    carrier = make_oscil(frequency)
    fm1_osc = make_oscil(mod_freq)
    fm2_osc = make_oscil((fm_ratio * frequency))
    ampf = make_env(amp_env, scaler=amplitude, duration=dur)
    indf = make_env(mod_index_env, scaler=hz2radians(mod_index), duration=dur)
    modfrqf = make_env(mod_freq_env, scaler=hz2radians(mod_skew), duration=dur)
    fm2_amp = hz2radians(fm_index * fm_ratio * frequency)
    
    for i in range(beg, end):
        garble_in = (env(modfrqf) * oscil(fm1_osc, env(modfrqf)))
        locsig(loc, i, (env(ampf) * oscil(carrier, (fm2_amp * oscil(fm2_osc, garble_in)) + garble_in)))
        
        
# --------------- fm_drum ---------------- #

def fm_drum(start_time, duration, frequency, amplitude, index, high=True, degree=0.0, distance=1.0, reverb_amount=.01):
    indxfun = [0,0,5,.014,10,.033,15,.061,20,.099,
            	25,.153,30,.228,35,.332,40,.477,
		        45,.681,50,.964,55,.681,60,.478,65,.332,
		        70,.228,75,.153,80,.099,85,.061,
		        90,.033,95,.0141,100,0]

    indxpt = 100 - (100 * ((duration - .1) / duration))
    atdrpt = 100 * ((.01 if high else .015) / duration)
    divindxf = stretch_envelope(indxfun, 50, atdrpt, 65, indxpt)
    ampfun = [0,0,3,.05,5,.2,7,.8,8,.95,10,1.0,12,.95,20,.3,30,.1,100,0]
    casrat = 8.525 if high else 3.515    
    fmrat = 3.414 if high else 1.414
    glsfun = [0,0,25,0,75,1,100,1]
    beg = seconds2samples(start_time)
    end = seconds2samples(start_time + duration)
    glsf = make_env(glsfun, scaler=(hz2radians(66) if high else 0.0), duration=duration)
    ampe = stretch_envelope(ampfun, 10, atdrpt, 15, max((atdrpt + 1), 100 - (100 * ((duration - .05) / duration))))
    ampf = make_env(ampe, scaler=amplitude, duration=duration)
    indxf = make_env(divindxf, scaler=min(hz2radians(index * fmrat * frequency), math.pi), duration=duration)
    mindxf = make_env(divindxf, scaler=min(hz2radians(index * casrat * frequency), math.pi), duration=duration)
    deve = stretch_envelope(ampfun, 10, atdrpt, 90, max((atdrpt + 1), (100 - (100 * ((duration - .05) / duration)))))
    devf = make_env(deve, scaler=min(math.pi, hz2radians(7000)), duration=duration)
    loc = make_locsig(degree, distance, reverb_amount)
    rn = make_rand(7000., 1.0)
    carrier = make_oscil(frequency)
    fmosc = make_oscil(frequency*fmrat)
    cascade = make_oscil(frequency*casrat)
    
    for i in range(beg, end):
        gls = env(glsf)
        locsig(loc, i, (env(ampf) * oscil(carrier, (gls + (env(indxf) * 
                                        oscil(fmosc, ((gls*fmrat)+(env(mindxf)*
                                            oscil(cascade, ((gls*casrat)+(env(devf)*rand(rn))))))))))))
    
    



# --------------- fm_gong ---------------- #

def gong(start_time, duration, frequency, amplitude, degree=0.0, distance=1.0, reverb_amount=.005):
    mfq1 = frequency * 1.16
    mfq2 = frequency * 3.14
    mfq3 = frequency * 1.005    
    indx01 = hz2radians(.01 * mfq1)
    indx11 = hz2radians(.30 * mfq1)
    indx02 = hz2radians(.01 * mfq2)
    indx12 = hz2radians(.38 * mfq2)
    indx03 = hz2radians(.01 * mfq3)
    indx13 = hz2radians(.50 * mfq3)
    atpt = 5
    atdur = 100 * (.002 / duration)
    expf = [0,0,3,1,15,.5,27,.25,50,.1,100,0]
    rise = [0,0,15,.3,30,1.0,75,.5,100,0]
    fmup = [0,0,75,1.0,98,1.0,100,0]
    fmdwn = [0,0,2,1.0,100,0]
    ampfun = make_env(stretch_envelope(expf, atpt, atdur, None, None), scaler=amplitude, duration=duration)
    indxfun1 = make_env(fmup, duration=duration, scaler=indx11-indx01, offset=indx01)
    indxfun2 = make_env(fmup, duration=duration, scaler=indx12-indx02, offset=indx02)
    indxfun3 = make_env(fmup, duration=duration, scaler=indx13-indx03, offset=indx03)
    loc = make_locsig(degree, distance, reverb_amount)
    carrier = make_oscil(frequency)
    mod1 = make_oscil(mfq1)
    mod2 = make_oscil(mfq2)
    mod3 = make_oscil(mfq3)
    beg = seconds2samples(start_time)
    end = seconds2samples(start_time + duration)
    
    for i in range(beg, end):
        locsig(loc, i, env(ampfun) * oscil(carrier, (env(indxfun1) * oscil(mod1)) + (env(indxfun2) * oscil(mod2)) + (env(indxfun3) * oscil(mod3))))
    
# TODO: --------------- attract ---------------- #

# below not working
# def attract(beg, dur, amp, c_1):
#     st = seconds2samples(beg)
#     nd = seconds2samples(beg + dur)
#     c = c_1
#     a = .2
#     b = .2
#     dt = .04
#     scale = (.5 * amp) / c_1
#     x = -1.
#     y = 0.0
#     z = 0.0
#     
#     for i in range(st,nd):
#         x1 = x - (dt * (y + x))
#         y  = y + (dt * (x + (a * y)))
#         z  = z + (dt * ((b + (x * z)) - (c * z)))
#         x = x1
#         outa(i, scale * x)
    
# TODO: --------------- pqw ---------------- #


    
def clip_env(e):
    xs = e[0::2]
    ys = [min(x,1.0) for x in e[1::2]]
    return interleave(xs, ys)
    
    

def pqw(start, dur, spacing_freq, carrier_freq, amplitude, 
        ampfun, indexfun, partials, degree=0.0, distance=1.0, reverb_amount=.005):
    normalized_partials = normalize_partials(partials)
    spacing_cos = make_oscil(spacing_freq, initial_phase=math.pi/2.0)
    spacing_sin = make_oscil(spacing_freq)
    carrier_cos = make_oscil(carrier_freq, initial_phase=math.pi/2.0)
    carrier_sin = make_oscil(carrier_freq)
    sin_coeffs = partials2polynomial(normalized_partials, Polynomial.SECOND_KIND)
    cos_coeffs = partials2polynomial(normalized_partials, Polynomial.FIRST_KIND)
    amp_env = make_env(ampfun, scaler=amplitude, duration=dur)
    ind_env = make_env(clip_env(indexfun), duration=dur)
    loc = make_locsig(degree, distance, reverb_amount)
    r = carrier_freq / spacing_freq
    tr = make_triangle_wave(5, hz2radians(.005 * spacing_freq))
    rn = make_rand_interp(12, hz2radians(.005 * spacing_freq))
    beg = seconds2samples(start)
    end = seconds2samples(start + dur)
    
    for i in range(beg, end):
        vib = triangle_wave(tr) + rand_interp(rn)
        ax = env(ind_env) * oscil(spacing_cos, vib)
        locsig(loc, i, env(amp_env) * 
            ((oscil(carrier_sin, (vib * r)) * oscil(spacing_sin, vib) * polynomial(sin_coeffs, ax)) -
                (oscil(carrier_cos, (vib * r)) * polynomial(cos_coeffs, ax))))





# TODO: --------------- tubebell ---------------- #


def tubebell(beg, dur, freq, amp, base=32.0):
    osc0 = make_oscil(freq * .995)
    osc1 = make_oscil(freq * 1.414 * 0.995)
    osc2 = make_oscil(freq * 1.005)
    osc3 = make_oscil(freq * 1.414)
    ampenv1 = make_env([0,0,.005, 1, dur, 0], base=base, duration=dur, scaler=(amp * .5 * .707))
    ampenv2 = make_env([0,0,.001, 1, dur, 0], base=(2 * base), duration=dur, scaler=(amp * .5))
    ampmod = make_oscil(2.0)
    st = seconds2samples(beg)
    nd = seconds2samples(beg + dur)
    
    for i in range(st, nd):
        outa(i, ((.007 * oscil(ampmod)) + .993) *
                ((env(ampenv1) * oscil(osc0, (.203 * oscil(osc1)))) +
                 (env(ampenv2) * oscil(osc2, (.144 * oscil(osc3))))))
    
    
    
    

# TODO: --------------- wurley ---------------- #

# TODO: --------------- rhodey ---------------- #

# TODO: --------------- hammondoid ---------------- #

# TODO: --------------- metal ---------------- #

# TODO: --------------- drone ---------------- #

# TODO: --------------- canter ---------------- #

# TODO: --------------- nrev ---------------- #

# TODO: --------------- reson ---------------- #

# TODO: --------------- cellon ---------------- #

# TODO: --------------- jl_reverb ---------------- #

# TODO: --------------- gran_synth ---------------- #

# TODO: --------------- touch_tone ---------------- #

# TODO: --------------- spectra ---------------- #

# TODO: --------------- two_tab ---------------- #

# TODO: --------------- lbj_piano ---------------- #

# TODO: --------------- resflt ---------------- #

# TODO: --------------- scratch ---------------- #

# TODO: --------------- pins ---------------- #

# TODO: --------------- zc ---------------- #

# TODO: --------------- zn ---------------- #

# TODO: --------------- za ---------------- #

# TODO: --------------- clm_expsrc ---------------- #

# TODO: --------------- exp_snd ---------------- #

# TODO: --------------- grm ---------------- #

# TODO: --------------- expfil ---------------- #

# TODO: --------------- graphEq ---------------- #

# TODO: --------------- anoi ---------------- #

# TODO: --------------- bes_fm ---------------- #

# TODO: --------------- sbfm ---------------- #

# TODO: --------------- fm2 ---------------- #

# TODO: --------------- rmsg ---------------- #

# TODO: --------------- rms ---------------- #

# TODO: --------------- cnvrev ---------------- #


        
        
        
        
    
    