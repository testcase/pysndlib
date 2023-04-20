#! /Users/toddingalls/Developer/Python/venvs/pysndlib-venv/bin/python
import math
import numpy as np
from pysndlib import *
from .env import stretch_envelope
from .env import interleave


def is_zero(n):
    return n == 0
    
def is_number(n):
    return type(n) == int or type(n) == float

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
    
# --------------- pqw ---------------- #


    
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





# --------------- tubebell ---------------- #


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
    
    

# --------------- wurley ---------------- #

def wurley(beg, dur, freq, amp):
    osc0 = make_oscil(freq)
    osc1 = make_oscil(freq * 4.0)
    osc2 = make_oscil(510.0)
    osc3 = make_oscil(510.0)
    ampmod = make_oscil(8.0)
    g0 = .5 * amp
    ampenv = make_env([0,0,1,1,9,1,10,0], duration=dur)
    indenv = make_env([0,0,.001,1,.15,0,max(dur, .16),0], duration=dur, scaler=.117)
    resenv = make_env([0,0,.001,1,.25,0,max(dur, .26),0], duration=dur, scaler=(.5 * .307 * amp))
    st = seconds2samples(beg)
    nd = seconds2samples(beg + dur)
    for i in range(st, nd):
        outa(i, env(ampenv) * 
            (1.0 + (.007 * oscil(ampmod))) * 
            ((g0 * oscil(osc0, (.307 * oscil(osc1)))) + (env(resenv) * oscil(osc2, (env(indenv) * oscil(osc3))))))
            


# --------------- rhodey ---------------- #

def rhodey(beg, dur, freq, amp, base=.5):
    osc0 = make_oscil(freq)
    osc1 = make_oscil(freq * .5)
    osc2 = make_oscil(freq)
    osc3 = make_oscil(freq * 15.0)
    ampenv1 = make_env([0,0,.005, 1, dur, 0], base=base, duration=dur, scaler=(amp * .5))
    ampenv2 = make_env([0,0,.001, 1, dur, 0], base=(1.5 * base), duration=dur, scaler=(amp * .5))
    ampenv3 = make_env([0,0,.001, 1, .25, 0, max(dur, .26), 0], base=(4. * base), duration=dur, scaler= .109)
    st = seconds2samples(beg)
    nd = seconds2samples(beg + dur)
    for i in range(st, nd):
        outa(i, ( env(ampenv1)*oscil(osc0, (.535 * oscil(osc1)))) + (env (ampenv2) * oscil(osc2, env(ampenv3) * oscil(osc3))))

# --------------- hammondoid ---------------- #

def hammondoid(beg, dur, freq, amp):
    osc0 = make_oscil(freq * .999)
    osc1 = make_oscil(freq * 1.997)
    osc2 = make_oscil(freq * 3.006)
    osc3 = make_oscil(freq * 6.009)
    ampenv1 = make_env([0,0,.005, 1, dur-.008, 1, dur, 0], duration=dur)
    ampenv2 = make_env([0,0,.005, 1, dur, 0], duration=dur, scaler=(.5 * .75 * amp))
    g0 = .25 * .75 * amp
    g1 = .25 * .75 * amp
    g2 = .5 * amp
    st = seconds2samples(beg)
    nd = seconds2samples(beg + dur)
    
    for i in range(st, nd):
        outa(i, (env(ampenv1) * ((g0 * oscil(osc0)) + (g1 * oscil(osc1)) + (g2 * oscil(osc2)))) +
                    (env(ampenv2)*oscil(osc3)))

# --------------- metal ---------------- #

def metal(beg, dur, freq, amp):
    osc0 = make_oscil(freq)
    osc1 = make_oscil(freq * 4.0 * 0.999)
    osc2 = make_oscil(freq * 3.0 * 1.001)
    osc3 = make_oscil(freq * 0.50 * 1.002)
    ampenv0 = make_env([0,0,.001,1, (dur-.002),1,dur,0], duration=dur, scaler=(amp * .615))
    ampenv1 = make_env([0,0,.001,1,(dur-.011),1,dur,0], duration=dur, scaler=.202)
    ampenv2 = make_env([0,0,.01,1,(dur-.015),1,dur,0], duration=dur, scaler=.574)
    ampenv3 = make_env([0,0,.03,1,(dur-.040),1,dur,0], duration=dur, scaler=.116)
    st = seconds2samples(beg)
    nd = seconds2samples(beg + dur)
    
    for i in range(st, nd):
        outa(i, (env(ampenv0) * (oscil(osc0, ((env(ampenv1) * oscil(osc1, env(ampenv2) * oscil(osc2))))) +
                    (env(ampenv3) * oscil(osc3)))))
                    
                    
# --------------- drone ---------------- #

def drone(startime, dur, frequency, amp, ampfun, synth, ampat, ampdc, amtrev, deg, dis, rvibamt, rvibfreq):
    beg = seconds2samples(startime)
    end = seconds2samples(startime + dur)
    waveform = partials2wave(synth)
    amplitude = amp * .25
    freq = hz2radians(frequency)
    s = make_table_lookup(frequency, wave=waveform)
    ampe = stretch_envelope(ampfun, 25, (100 * (ampat / dur)), 75, (100 - (100 * (ampdc / dur))))
    amp_env = make_env(ampe, scaler=amplitude, duration=dur)
    ran_vib = make_rand(rvibfreq, rvibamt*freq)
    loc = make_locsig(deg, dis, amtrev)
    
    for i in range(beg, end):
        locsig(loc, i, env(amp_env) * table_lookup(s, rand(ran_vib)))


# TODO: --------------- canter ---------------- #

# --------------- nrev ---------------- #

def is_even(n):
    return bool(n%2==0)
    
def next_prime(n):
    np=[]
    isprime=[]
    for i in range (n+1,n+200):
        np.append(i)
    for j in np:
        val_is_prime = True
        for x in range(2,j-1):
            if j % x == 0:
                val_is_prime = False
                break
        if val_is_prime:
            isprime.append(j)
    return min(isprime)
    
def nrev(reverb_factor=1.09, lp_coeff=.7, volume=1.0):
    srscale = get_srate() / 25641
    dly_len = [1433,1601,1867,2053,2251,2399,347,113,37,59,53,43,37,29,19]
    chans = Sound.output.mus_channels
    chan2 = chans > 1
    chan4 = chans == 4
    
    for i in range(0,15):
        val = math.floor(srscale * dly_len[i])
        if is_even(val):
            val += 1
        dly_len[i] = next_prime(val)
        
    length = math.floor(get_srate()) + Sound.reverb.mus_length
    comb1 = make_comb(.822 * reverb_factor, dly_len[0])
    comb2 = make_comb(.802 * reverb_factor, dly_len[1])
    comb3 = make_comb(.733 * reverb_factor, dly_len[2])
    comb4 = make_comb(.753 * reverb_factor, dly_len[3])
    comb5 = make_comb(.753 * reverb_factor, dly_len[4])
    comb6 = make_comb(.733 * reverb_factor, dly_len[5])
    low = make_one_pole(lp_coeff, lp_coeff - 1.0)
    allpass1 = make_all_pass(-.7000, .700, dly_len[6])
    allpass2 = make_all_pass(-.7000, .700, dly_len[7])
    allpass3 = make_all_pass(-.7000, .700, dly_len[8])
    allpass4 = make_all_pass(-.7000, .700, dly_len[9])
    allpass5 = make_all_pass(-.7000, .700, dly_len[11])
    allpass6 = make_all_pass(-.7000, .700, dly_len[12]) if chan2 else None
    allpass7 = make_all_pass(-.7000, .700, dly_len[13]) if chan4 else None
    allpass8 = make_all_pass(-.7000, .700, dly_len[14]) if chan4 else None
    
    filts = []
    
    if not chan2:
        filts.append(allpass5)
        if not chan4:
            filts.extend([allpass5, allpass6])
        else:
            filts.extend([allpass5, allpass6, allpass7, allpass8])
    
    combs = make_comb_bank([comb1, comb2, comb3, comb4, comb5, comb6])
    allpasses = make_all_pass_bank([allpass1, allpass2, allpass3])
    for i in range(length):
        out_bank(filts, i, all_pass(allpass4, one_pole(low, all_pass_bank(allpasses, comb_bank(combs, volume * ina(i, Sound.reverb))))))
    
    

# --------------- reson ---------------- #

def reson(startime, dur, pitch, amp, numformants, indxfun, skewfun, pcskew, skewat, skewdc,
		      vibfreq, vibpc, ranvibfreq, ranvibpc, degree, distance, reverb_amount, data):
		      
    beg = seconds2samples(startime)
    end = seconds2samples(startime + dur)
    carriers = [None] * numformants
    modulator = make_oscil(pitch)
    ampfs = [None] * numformants
    indfs = [None] * numformants
    c_rats = [None] * numformants
    totalamp = 0.0
    loc = make_locsig(degree, distance, reverb_amount)
    pervib = make_triangle_wave(vibfreq, hz2radians(vibpc * pitch))
    ranvib = make_rand_interp(ranvibfreq, hz2radians(ranvibpc * pitch))
    frqe = stretch_envelope(skewfun, 25, 100 * (skewat / dur), 75, 100 - (100 * (skewdc / dur)))
    frqf = make_env(frqe, scaler=hz2radians(pcskew * pitch), duration=dur)
    
    for i in range(0,numformants):
        totalamp += data[i][2]
        
    for i in range(0, numformants):
        frmdat = data[i]
        freq = frmdat[1]
        harm = round(freq / pitch)
        rfamp = frmdat[2]
        ampat = 100 * (frmdat[3] / dur)
        ampdc = 100 - (100 * (frmdat[4] / dur))
        dev0 = hz2radians(frmdat[5] * freq)
        dev1 = hz2radians(frmdat[6] * freq)
        indxat = 100 * (frmdat[7] / dur)
        indxdc = 100 - (100 * (frmdat[8] / dur))
        ampf = frmdat[0]
        rsamp = 1.0 - math.fabs(harm - (freq / pitch))
        cfq = pitch * harm
        if ampat == 0:
            ampat = 25
        if ampdc == 0:
            ampdc = 75
        if indxat == 0:
            indxat = 25
        if indxdc == 0:
            indxdc = 75
        indfs[i] = make_env(stretch_envelope(indxfun, 25, indxat, 75, indxdc), duration=dur, scaler=(dev1 - dev0), offset=dev0)
        ampfs[i] = make_env(stretch_envelope(ampf, 25, ampat, 75, ampdc), duration=dur, scaler=(rsamp * amp * rfamp) / totalamp)
        c_rats[i] = harm
        carriers[i] = make_oscil(cfq)
        
    if numformants == 2:
        e1 = ampfs[0]
        e2 = ampfs[1]
        c1 = carriers[0]
        c2 = carriers[1]
        i1 = indfs[0]
        i2 = indfs[1]
        r1 = c_rats[0]
        r2 = c_rats[1]
        
        for i in range(beg, end):
            vib = env(frqf) + triangle_wave(pervib) + rand_interp(ranvib)
            modsig = oscil(modulator, vib)
            locsig(loc, i, (env(e1) * oscil(c1, (vib * r1) + (env(i1) * modsig))) + (env(e2) * oscil(c2, (vib * r2) + (env(i2) * modsig))))
        
    else:
        for i in range(beg, end):
            outsum = 0.0
            vib = env(frqf) + triangle_wave(pervib) + rand_interp(ranvib)
            modsig = oscil(modulator, vib)
            
            for k in range(0,numformants):
                outsum += env(ampfs[k]) * oscil(carriers[k], (vib * c_rats[k]) + (env(indfs[k]) * modsig))
            
            locsig(loc, i, outsum)       
        

# --------------- cellon ---------------- #


def cellon(beg, dur, pitch0, amp, ampfun, betafun, beta0, beta1, betaat, betadc, ampat, ampdc, dis, pcrev, deg,
            pitch1, glissfun, glissat, glissdc, pvibfreq, pvibpc, pvibfun=[0,1,100,1], pvibat=0, pvibdc=0, rvibfreq=0, rvibpc=0, rvibfun=[0,1,100,1]):
	
    st = seconds2samples(beg)
    nd = seconds2samples(beg + dur)
    pit1 = pitch0 if pitch1 == 0.0 else pitch1
    loc = make_locsig(deg, dis, pcrev)
    carrier = make_oscil(pitch0)
    low = make_one_zero(.5, -.5)
    fm = 0.0
    fmosc = make_oscil(pitch0)
    pvib = make_triangle_wave(pvibfreq, 1.0)
    rvib = make_rand_interp(rvibfreq, 1.0)
    ampap = (100 * (ampat / dur)) if ampat > 0.0 else 25
    ampdp = (100 * (1.0 - (ampdc / dur))) if ampdc > 0.0 else 75
    glsap = (100 * (glissat / dur)) if glissat > 0.0 else 25
    glsdp = (100 * (1.0 - (glissdc / dur))) if glissdc > 0.0 else 75
    betap = (100 * (betaat / dur)) if betaat > 0.0 else 25
    betdp = (100 * (1.0 - (betadc / dur))) if betadc > 0.0 else 75
    pvbap = (100 * (pvbap / dur)) if pvibat > 0.0 else 25
    pvbdp = (100 * (1.0 - (pvibdc / dur))) if pvibdc > 0.0 else 75
    
    pvibenv = make_env(stretch_envelope(pvibfun, 25, pvbap, 75, pvbdp), duration=dur, scaler=hz2radians(pvibpc * pitch0))
    rvibenv = make_env(rvibfun, duration=dur, scaler=hz2radians(rvibpc * pitch0))
    glisenv = make_env(stretch_envelope(pvibfun, 25, pvbap, 75, pvbdp), duration=dur, scaler=hz2radians(pvibpc * pitch0))
    amplenv = make_env(stretch_envelope(ampfun, 25, ampap, 75, ampdp), scaler=amp, duration=dur)
    betaenv = make_env(stretch_envelope(betafun, 25, betap, 75, betdp), duration=dur, scaler=(beta1 - beta0), offset=beta0)
    
    if (pitch0 == pitch1) and (is_zero(pvibfreq) or is_zero(pvibpc)) and (is_zero(rvibfreq) or is_zero(rvibpc)):
        for i in range(st, nd):
            fm = one_zero(low, env(betaenv) * oscil(fmosc, fm))
            locsig(loc, i, env(amplenv)*oscil(carrier,fm))
    else:
        for i in range(st, nd):
            vib = (env(pvibenv)*triangle_wave(pvib)) + (env(rvibenv)*rand_interp(rvib)) + env(glisenv)
            fm = one_zero(low, env(betaenv) * oscil(fmosc, (fm+vib)))
            locsig(loc, i, env(amplenv)*oscil(carrier,fm+vib))
    
    
# --------------- jl_reverb ---------------- #

def jl_reverb(decay=3.0, volume=1.0):
    allpass1 = make_all_pass(-.700, .700, 2111)
    allpass2 = make_all_pass(-.700, .700, 673)
    allpass3 = make_all_pass(-.700, .700, 223)
    comb1 = make_comb(.742, 9601)
    comb2 = make_comb(.733, 10007)
    comb3 = make_comb(.715, 10799)
    comb4 = make_comb(.697, 11597)
    outdel1 = make_delay(seconds2samples(.013))
    outdel2 = make_delay(seconds2samples(.011))
    length = math.floor((decay * get_srate()) + Sound.reverb.mus_length)
    filts = [outdel1, outdel2]
    combs = make_comb_bank([comb1, comb2, comb3, comb4])
    allpasses = make_all_pass_bank([allpass1, allpass2, allpass3])
    for i in range(0, length):
        out_bank(filts, i, volume * comb_bank(combs, all_pass_bank(allpasses, ina(i, Sound.reverb))))

# --------------- gran_synth ---------------- #

def gran_synth(startime, duration, audio_freq, grain_dur, grain_interval, amp):
    grain_size = math.ceil(max(grain_dur, grain_interval) * get_srate())
    beg = seconds2samples(startime)
    end = seconds2samples(startime + duration)
    grain_env = make_env([0,0,25,1,75,1,100,0], duration=grain_dur)
    carrier = make_oscil(audio_freq)
    grain = [(env(grain_env) * oscil(carrier)) for i in range(grain_size)]
    grains = make_wave_train(1.0 / grain_interval, grain)
    
    for i in range(beg, end):
        outa(i, amp * wave_train(grains))

# --------------- touch_tone ---------------- #

def touch_tone(start, telephone_number):
    touch_tab_1 = [0,697,697,697,770,770,770,852,852,852,941,941,941]
    touch_tab_2 = [0,1209,1336,1477,1209,1336,1477,1209,1336,1477,1209,1336,1477]
    
    for i in range(len(telephone_number)):
        k = telephone_number[i]
        beg = seconds2samples(start + (i * .4))
        end = beg + seconds2samples(.3)
        if is_number(k):
            i = k if not (0 == k) else 11
        else:
            i = 10 if k == '*' else 12
        
        frq1 = make_oscil(touch_tab_1[i])
        frq2 = make_oscil(touch_tab_2[i])
        
        for j in range(beg, end):
            outa(j, (.1 * (oscil(frq1) + oscil(frq2))))
        
# --------------- spectra ---------------- #

def spectra(startime, duration, frequency, amplitude, partials=[1,1,2,0.5], amp_envelope=[0,0,50,1,100,0], 
                vibrato_amplitude=0.005, vibrato_speed=5.0,degree=0.0, distance=1.0, reverb_amount=0.005):

    beg = seconds2samples(startime)
    end = seconds2samples(startime + duration)
    waveform = partials2wave(partials)
    freq = hz2radians(frequency)
    s = make_table_lookup(frequency=frequency, wave=waveform)
    amp_env = make_env(amp_envelope, scaler=amplitude, duration=duration)
    per_vib = make_triangle_wave(vibrato_speed, vibrato_amplitude * freq)
    loc = make_locsig(degree, distance, reverb_amount)
    ran_vib = make_rand_interp(vibrato_speed + 1.0, vibrato_amplitude * freq)
    for i in range(beg, end):
        locsig(loc, i, env(amp_env) * table_lookup(s, triangle_wave(per_vib) + rand_interp(ran_vib)))




    
    

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


        
        
        
        
    
    