import math
import numpy as np
from pysndlib import *

# TODO: --------------- pluck ---------------- #

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

# cadr (car (cdr x)) [1]
# caddr (car (cdr (cdr x))) [2]

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

# TODO: --------------- pqwvox ---------------- #

# TODO: --------------- fof ---------------- #

# TODO: --------------- fm_trumpet ---------------- #

# TODO: --------------- stereo_flute ---------------- #

# TODO: --------------- fm_bell ---------------- #

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
                             

        
        
# TODO: --------------- fm_insect ---------------- #

# TODO: --------------- fm_drum ---------------- #

# TODO: --------------- fm_gong ---------------- #

# TODO: --------------- pqw ---------------- #

# TODO: --------------- tubebell ---------------- #

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


        
        
        
        
    
    