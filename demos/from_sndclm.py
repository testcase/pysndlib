#! /Users/toddingalls/Developer/Python/venvs/pysndlib-venv/bin/python
# Examples from https://ccrma.stanford.edu/software/snd/snd/sndclm.html#oscildoc
# showing basic examples of built-in generators
import math
from random import uniform
import numpy as np
from pysndlib import *


with Sound(play=True):
    gen = make_oscil(440.0)
    for i in range(44100):
        outa(i,  oscil(gen))


with Sound(play=True):
    gen = make_oscil(440.0)
    ampf = make_env([0., 0., 0.01, 1.0, 0.25, 0.1, 1, 0], scaler=.5, length=44100)
    for i in range(44100):
        outa(i,  env(ampf)*oscil(gen))
    
     
with Sound(play=True):
    gen = make_table_lookup(440.0, wave=partials2wave([1., .5, 2, .5]))
    for i in range(44100):
        outa(i, .5 * table_lookup(gen))

with Sound(play=True):
    gen = make_polywave(440, partials=[1., .5, 2, .5])
    for i in range(44100):
        outa(i, .5 * polywave(gen))


with Sound(play=True):
    gen = make_triangle_wave(440.0)
    for i in range(44100):
        outa(i, .5 * triangle_wave(gen))

with Sound(play=True):
    gen = make_ncos(440.0, 10);
    for i in range(44100):
        outa(i, .5 * ncos(gen))


with Sound(play=True):
    gen = make_nrxycos(440.0,n=10)
    for i in range(44100):
        outa(i, .5 * nrxycos(gen))


with Sound(play=True):
    v = np.zeros(64, dtype=np.double)
    g = make_ncos(400,10)
    g.mus_phase = -.5 * math.pi
    for i in range(64):
        v[i] = ncos(g)
    gen = make_wave_train(440, wave=v)    
    for i in range(44100):
        outa(i, .5 * wave_train(gen))
        
        

with Sound(channels=2, play=True):
    ran1 = make_rand(5.0, hz2radians(220.0))
    ran2 = make_rand_interp(5.0, hz2radians(220.0))
    osc1 = make_oscil(440.0)
    osc2 = make_oscil(1320.0)
    for i in range(88200):
        outa(i, 0.5 * oscil(osc1, rand(ran1)))
        outb(i, 0.5 * oscil(osc2, rand_interp(ran2)))
        
        
with Sound("out1.aif", 1):
    flt = make_two_pole(1000.0, 0.999)
    ran1 = make_rand(10000.0, 0.002)
    for i in range(44100):
        outa(i, 0.5 * two_pole(flt, rand(ran1)))
        
        
with Sound(play=True):
    flt = make_two_pole(1000.0, .999)
    ran1 = make_rand(10000.0, .002)
    for i in range(44100):
        outa(i, .5 * two_pole(flt, rand(ran1)))

with Sound(play=True):
    flt = make_firmant(1000.0, 0.999)
    ran1 = make_rand(10000.0, 5.0)
    for i in range(44100):
        outa(i, 0.5 * firmant(flt, rand(ran1)))

    
with Sound(play=True):
    flt = make_iir_filter(3, [0.0, -1.978, 0.998])
    ran1 = make_rand(10000.0, 0.002)
    for i in range(44100):
        outa(i, 0.5 * iir_filter(flt, rand(ran1)))

with Sound(play=True):
    dly = make_delay(seconds2samples(0.5))
    osc1 = make_oscil(440.0)
    osc2 = make_oscil(660.0)
    for i in range(44100):
        outa(i, 0.5 * (oscil(osc1) + delay(dly, oscil(osc2))))        

with Sound(play=True):
    cmb = make_comb(0.4, seconds2samples(0.4))
    osc = make_oscil(440.0)
    ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
    
    for i in range(88200):
        outa(i, 0.5 * (comb(cmb, env(ampf) * oscil(osc))))

with Sound(play=True):
    alp = make_all_pass(-0.4, 0.4, seconds2samples(0.4))
    osc = make_oscil(440.0)
    ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
    for i in range(88200):
        outa(i, 0.5 * (all_pass(alp, env(ampf) * oscil(osc))))
        

with Sound(play=True):
    avg = make_moving_average(4410)
    osc = make_oscil(440.0)
    stop = 44100 - 4410
    for i in range(stop):
        val = oscil(osc)
        outa(i, val * moving_average(avg, abs(val))) 
    for i in range(stop, 44100):
        outa(i, oscil(osc) * moving_average(avg, 0.0))
        
    
with Sound(play=True, srate=22050):    
    rd = make_readin("oboe.snd");
    length = 2 * mus_sound_framples("oboe.snd")
    sr = make_src(rd, .5);
    
    for i in range(length):
        outa(i, src(sr))
    
# slightly different as don't have samples  function. 
with Sound(play=True, statistics=True):    
    flt, _ = file2array('oboe.snd', channel=0, beg=0, dur=clm_length('pistol.snd'))
    cnv = make_convolve(make_readin('pistol.snd'), flt)
    for i in range(88200):
        outa(i, .25 * convolve(cnv))
    
with Sound(play=True):    
    grn = make_granulate(make_readin('oboe.snd'), 2.)
    for i in range(44100):
        outa(i, granulate(grn))
        
        
with Sound(play=True):    
    osc = make_oscil(440.0)
    sweep = make_env([0,0,1,1], scaler=hz2radians(440.0), length=44100)
    grn = make_granulate(lambda d : .2 * oscil(osc, env(sweep)), expansion=2.0, length=.5)
    for i in range(88200):
        outa(i, granulate(grn))
    
        
with Sound(play=True):    
    pv = make_phase_vocoder(make_readin("oboe.snd"), pitch=2.0)
    for i in range(44100):
        outa(i, phase_vocoder(pv))    

with Sound(play=True):    
    fm = make_asymmetric_fm(440.0, 0.0, 0.9, 0.5)
    for i in range(44100):
        outa(i, 0.5 * asymmetric_fm(fm, 1.0))
        
with Sound(play=True):    
    reader = make_readin("oboe.snd")
    for i in range(44100):
        outa(i, readin(reader))

       
with Sound(play=True):    
    infile = make_file2sample('oboe.snd')
    for i in range(44100):
        out_any(i, in_any(i, 0, infile), 0)


with Sound(channels=2, play=True):    
    loc = make_locsig(60.0)
    osc = make_oscil(440.0)
    for i in range(44100):
        locsig(loc, i, .5 * oscil(osc))
        

# -----------------------------
# some other examples from sndclm

def simp(beg, dur, freq, amp, envelope):
    os = make_oscil(freq)
    amp_env = make_env(envelope, duration=dur, scaler=amp)
    start = seconds2samples(beg)
    end = seconds2samples(beg + dur)
    for i in range(start, end):
        outa(i, env(amp_env) * oscil(os))
        
with Sound(play=True):
    simp(0,2,440,.1,[0,0,.1,1.0,1.0,0.0])
    
    
def simp(start, end, freq, amp, frq_env):
    os = make_oscil(freq)
    frqe = make_env(frq_env, length= (end+1) - start, scaler=hz2radians(freq))
    for i in range(start, end):
        outa(i, amp * oscil(os, env(frqe)))
        
with Sound(play=True):
    simp(0,10000,440,.1,[0, 0, 1, 1])
    
    
def simple_fm(beg, dur, freq, amp, mc_ratio, idx, amp_env=None, index_env=None):
    start = seconds2samples(beg)
    end = start + seconds2samples(dur)
    cr = make_oscil(freq)
    md = make_oscil(freq * mc_ratio)
    fm_index = hz2radians(idx * mc_ratio * freq)
    ampf = make_env(amp_env or [0,0,.5,1,1,0],scaler=amp, duration=dur)
    indf = make_env(index_env or [0,0,.5,1,1,0], scaler=fm_index, duration=dur)
    for i in range(start, end):
        outa(i, env(ampf) * oscil(cr, env(indf) * oscil(md)))  
    
with Sound(play=True):
    simple_fm(0, 1, 440, .1, 2, 1.0)
    
    
    
with Sound(play=True, statistics=True):
    peaks = [23,0.0051914,32,0.0090310,63,0.0623477,123,0.1210755,185,0.1971876,
            209,0.0033631,247,0.5797809,309,1.0000000,370,0.1713255,432,0.9351965,
            481,0.0369873,495,0.1335089,518,0.0148626,558,0.1178001,617,0.6353443,
            629,0.1462804,661,0.0208941,680,0.1739281,701,0.0260423,742,0.1203807,
            760,0.0070301,803,0.0272111,865,0.0418878,926,0.0090197,992,0.0098687,
            1174,0.00444,1298,0.0039722,2223,0.0033486,2409,0.0083675,2472,0.0100995,
            2508,0.004262,2533,0.0216248,2580,0.0047732,2596,0.0088663,2612,0.0040592,
            2657,0.005971,2679,0.0032541,2712,0.0048836,2761,0.0050938,2780,0.0098877,
            2824,0.003421,2842,0.0134356,2857,0.0050194,2904,0.0147466,2966,0.0338878,
            3015,0.004832,3027,0.0095497,3040,0.0041434,3092,0.0044802,3151,0.0038269,
            3460,0.003633,3585,0.0050849,4880,0.0042301,5121,0.0037906,5136,0.0048349,
            5158,0.004336,5192,0.0037841,5200,0.0038025,5229,0.0035555,5356,0.0045781,
            5430,0.003687,5450,0.0055170,5462,0.0057821,5660,0.0041789,5673,0.0044932,
            5695,0.007370,5748,0.0031716,5776,0.0037921,5800,0.0062308,5838,0.0034629,
            5865,0.005942,5917,0.0032254,6237,0.0046164,6360,0.0034708,6420,0.0044593,
            6552,0.005939,6569,0.0034665,6752,0.0041965,7211,0.0039695,7446,0.0031611,
            7468,0.003330,7482,0.0046322,8013,0.0034398,8102,0.0031590,8121,0.0031972,
            8169,0.003345,8186,0.0037020,8476,0.0035857,8796,0.0036703,8927,0.0042374,
            9388,0.003173,9443,0.0035844,9469,0.0053484,9527,0.0049137,9739,0.0032365,
            9853,0.004297,10481,0.0036424,10490,0.0033786,10606,0.0031366]
    
    length = len(peaks) // 2
    dur = 10
    oscs = [None] * length
    amps = [None] * length
    ramps = [None] * length
    freqs = [None] * length
    vib = make_rand_interp(50, hz2radians(.01))
    ampf = make_env([0,0,1,1,10,1,11,0], duration=dur, scaler=.1)
    samps = seconds2samples(dur)
    for i in range(length):
        freqs[i] = peaks[i*2]
        oscs[i] = make_oscil(freqs[i], uniform(0, math.pi))
        amps[i] = peaks[1+(2*i)]
        ramps[i] = make_rand_interp(1.0 + (i * (20.0 / length)), 
                                    (.1 + (i * (3.0 / length)))*amps[i])
    
    for i in range(samps):
        sm = 0.0
        fm = rand_interp(vib)
        for k in range(length):
            sm += (amps[k] + rand_interp(ramps[k])) * oscil(oscs[k], freqs[k]*fm)
        
        outa(i, env(ampf) * sm)
    
    
with Sound(channels=2, play=True):
    dur = 2.0
    samps = seconds2samples(dur)
    pitch = 1000
    modpitch = 100
    pm_index = 4.0
    fm_index = hz2radians(4.0 * modpitch)
    car1 = make_oscil(pitch)
    mod1 = make_oscil(modpitch)
    car2 = make_oscil(pitch)
    mod2 = make_oscil(modpitch)
    frqf = make_env([0,0,1,1], duration=dur)
    ampf = make_env([0,0,1,1,20,1,21,0], duration=dur, scaler=.5)
    for i in range(samps):
        frq = env(frqf)
        rfrq = hz2radians(frq)
        amp = env(ampf)
        outa(i, amp * (oscil(car1, (rfrq*pitch) + (fm_index * (frq + 1) * oscil(mod1, (rfrq * modpitch))))))
        outb(i, amp * (oscil(car2, (rfrq*pitch) , pm_index * oscil(mod2, rfrq * modpitch))))
        
        
    car1 = make_oscil(pitch)
    mod1 = make_oscil(modpitch)
    car2 = make_oscil(pitch)
    mod2 = make_oscil(modpitch)
    frqf = make_env([0,0,1,1], duration=dur)
    ampf = make_env([0,0,1,1,20,1,21,0], duration=dur, scaler=.5)
    for i in range(samps):
        frq = env(frqf)
        rfrq = hz2radians(frq)
        amp = env(ampf)
        outa(i, amp * (oscil(car1, (rfrq*pitch) + (fm_index * oscil(mod1, (rfrq * modpitch))))))
        outb(i, amp * (oscil(car2, (rfrq*pitch) , (pm_index / (frq + 1)) * oscil(mod2, rfrq * modpitch))))



      
    
    
with Sound(play=True, statistics=False):
    phases98 = [0.000000, -0.183194, 0.674802, 1.163820, -0.147489, 1.666302, 0.367236, 0.494059, 0.191339,
                0.714980, 1.719816, 0.382307, 1.017937, 0.548019, 0.342322, 1.541035, 0.966484, 0.936993,
                -0.115147, 1.638513, 1.644277, 0.036575, 1.852586, 1.211701, 1.300475, 1.231282, 0.026079,
                0.393108, 1.208123, 1.645585, -0.152499, 0.274978, 1.281084, 1.674451, 1.147440,0.906901,
                1.137155, 1.467770, 0.851985, 0.437992, 0.762219, -0.417594, 1.884062, 1.725160, -0.230688,
		        0.764342, 0.565472, 0.612443, 0.222826, -0.016453, 1.527577, -0.045196, 0.585089, 0.031829,
		        0.486579, 0.557276, -0.040985, 1.257633, 1.345950, 0.061737, 0.281650, -0.231535, 0.620583,
		        0.504202, 0.817304, -0.010580, 0.584809, 1.234045, 0.840674, 1.222939, 0.685333, 1.651765,
		        0.299738, 1.890117, 0.740013, 0.044764, 1.547307, 0.169892, 1.452239, 0.352220, 0.122254,
		        1.524772, 1.183705, 0.507801, 1.419950, 0.851259, 0.008092, 1.483245, 0.608598, 0.212267,
		        0.545906, 0.255277, 1.784889, 0.270552, 1.164997, -0.083981, 0.200818, 1.204088]
		        
    freq = 10.0
    dur = 5.0
    n = 98
	
    samps = math.floor(dur * 44100)
    onedivn = 1.0 / n
    freqs = np.zeros(n)
    phases = np.zeros(n)
    phases.fill(math.pi*.5)
    for i in range(n):
        off = (math.pi * (.5 - phases98[i])) / dur / 44100
        h = hz2radians(freq*(i+1))
        freqs[i] = h + off
    
    ob = make_oscil_bank(freqs, phases)
    for i in range(1000): #get rid of the distracting initial click
        nada = oscil_bank(ob) # added in this assignment because otherwise was printing output
    for k in range(samps):
        outa(k, onedivn * oscil_bank(ob))
    




def mapenv(beg, dur, frq, amp, en):
    start = seconds2samples(beg)
    end = start + seconds2samples(dur)
    osc = make_oscil(frq)
    zv = make_env(en, 1.0, dur)
    for i in range(start, end):
        zval = env(zv)
        outa(i, amp * math.sin(.5*math.pi*zval*zval*zval)*oscil(osc))
    
with Sound(play=True):
    mapenv(0, 1, 440, .5, [0,0,50,1,75,0,86,.5,100,0])   
    
    

with Sound():
    e = make_env([0,0,1,1,2,.25,3,1,4,0], duration=.5)
    for i in range(44100):
          outa(i, env_any(e, lambda y : y * y))
    

    
def sine_env(e):
    return env_any(e, lambda y : .5 * (1.0 + math.sin((-.5*math.pi) + (math.pi*y))))
    
with Sound():
    e = make_env([0,0,1,1,2,.25,3,1,4,0], duration=.5)
    for i in range(44100):
        outa(i, sine_env(e))


with Sound():
    e = make_pulsed_env([0,0,1,1,2,0], .01, 1)
    frq = make_env([0,0,1,1], duration=1.0, scaler=hz2radians(50))
    for i in range(44100):
        outa(i, .5 * pulsed_env(e, env(frq)))
    

with Sound(play=True):
    wav = make_polyshape(frequency=500, partials=[1, .5, 2, .3, 3, .2])
    for i in range(40000):
        outa(i, 1. * polyshape(wav))
    
    

with Sound(play=True):
    harms = np.zeros(10)
    k = 1
    for i in range(0,10,2):
        harms[i] = k
        harms[i+1] = (1.0 / math.sqrt(k))
        k+= 3 
    gen = make_polywave(200, harms)
    ampf = make_env([0,0,1,1,10,1,11,0], duration=1.0, scaler=.5)
    for i in range(44100):
        outa(i, env(ampf) * polywave(gen) * .5)
    
    
def pqw(start, dur, spacing, carrier, partials):
    spacing_cos = make_oscil(spacing, math.pi / 2.0)
    spacing_sin = make_oscil(spacing)
    carrier_cos = make_oscil(carrier, math.pi / 2.0)
    carrier_sin = make_oscil(carrier)
    sin_coeffs = partials2polynomial(partials, Polynomial.SECOND_KIND)
    cos_coeffs = partials2polynomial(partials, Polynomial.FIRST_KIND)
    beg = seconds2samples(start)
    end = beg + seconds2samples(dur)
    for i in range(beg, end):
        ax = oscil(spacing_cos)
        outa(i, (oscil(carrier_sin) * oscil(spacing_sin) * polynomial(sin_coeffs, ax)) -
                (oscil(carrier_cos) * polynomial(cos_coeffs, ax)))
    
    
with Sound(play=True):
    pqw(0,1,200.0,1000.0, [2,.2,3,.3,6,.5])
    
    
with Sound(play=True):
    modulator = make_polyshape(100, partials=[0,.4,1,.4,2,.1,3,.05,4,.05])    
    carrier = make_oscil(1000.)
    for i in range(20000):
        outa(i, .5 * oscil(carrier) * polyshape(modulator))
    

with Sound(play=True):
    dur = 1
    samps = seconds2samples(dur)
    coeffs = [0.0, 0.5, .25, .125, .125]
    x = 0.0
    incr = hz2radians(100)
    ampf = make_env([0,0,1,1,10,1,11,0], duration=dur, scaler=.5)
    harmf = make_env([0, .125, 1, .25], duration=dur)
    for i in range(samps):
        harm = env(harmf)
        coeffs[3] = harm
        coeffs[4] = .25 - harm
        outa(i, env(ampf) * chebyshev_t_sum( x, coeffs))
        x += incr
    
    
with Sound(play=True):
    gen = make_polyshape(100, partials=[11,1,20,1])
    ampf = make_env([0,0,1,1,20,1,21,0], scaler=.4, length=88200)
    indf = make_env([0, 0, 1, 1, 1.1, 1], length=88200)
    for i in range(88200):
        outa(i, env(ampf)*polyshape(gen, env(indf)))


with Sound(play=True):
    gen = make_polyshape(1000.0, partials=[1, .25, 2, .25, 3, .125, 4, .125, 5, .25])
    indf = make_env([0,0,1,1,2,0], duration = 2.0)
    ampf = make_env([0,0,1,1,2,1,3,0], duration=2.0)
    mx = make_moving_max(256)
    samps = seconds2samples(2.0)
    for i in range(samps):
        val = polyshape(gen, env(indf))
        outa(i, (env(ampf)*val) / max(.001, moving_max(mx, val)))
    

with Sound(play=True, reverb=jc_reverb):
    pcoeffs = partials2polynomial([5,1])
    gen1 = make_oscil(100.)
    gen2 = make_oscil(2000.0)
    for i in range(44100):
        outa(i, polynomial(pcoeffs, .5 * (oscil(gen1) + oscil(gen2))))
    
with Sound(play=True, channels=2):
    dur = 2.0
    samps = seconds2samples(dur)
    p1 = make_polywave(800, [1,.1,2,.3,3,.4,5,.2])
    p2 = make_polywave(400, [1,.1,2,.3,3,.4,5,.2])
    interpf = make_env([0,0,1,1], duration=dur)
    p3 = partials2polynomial([1,.1,2,.3,3,.4,5,.2])
    g1 = make_oscil(800)
    g2 = make_oscil(400)
    ampf = make_env([0,0,1,1,10,1,11,0], duration=dur)
    
    for i in range(samps):
        interp = env(interpf)
        amp = env(ampf)
        #chan A: interpolate from one spectrum to the next directly
        outa(i, amp * ((interp * polywave(p1)) + ((1.0 - interp) * polywave(p2))))
        #chan B: interpolate inside the sum of Tns!
        outb(i, amp*(polynomial(p3, (interp * oscil(g1)) + ((1.0-interp) * oscil(g2)))))



with Sound(play=True):
    duty_factor = .25
    p_on = make_pulse_train(100, .5)
    p_off = make_pulse_train(100, -.5, (2 * math.pi * (1 - duty_factor)))
    sm = 0.0
    for i in range(44100):
        sm += pulse_train(p_on) + pulse_train(p_off)
        outa(i, sm)


def simple_soc(beg, dur, freq, amp):
    os = make_ncos(freq, 10)
    start = seconds2samples(beg)
    end = start + seconds2samples(dur)
    for i in range(start, end):
        outa(i, amp*ncos(os))


with Sound(play=True):
    simple_soc(0,1,100,.9)
    
    
    
    
with Sound(play=True):
    gen1 = make_nrxycos(400, 1, 15, .95)
    indr = make_env([0,-1,1,1], length=80000, scaler=.9999)
    for i in range(80000):
        gen1.mus_scaler = env(indr)
        outa(i, .5 * nrxycos(gen1, 0.0))



def shift_pitch(beg, dur, file, freq, order=40):
    st = seconds2samples(beg)
    nd = st + seconds2samples(dur)
    gen = make_ssb_am(freq, order)
    rd = make_readin(file)
    for i in range(st, nd):
        outa(i, ssb_am(gen, readin(rd)))

with Sound(play=True):
    shift_pitch(0, 3, 'oboe.snd', 1108.0)




def fofins(beg, dur, frq, amp, vib, f0, a0, f1, a1, f2, a2, ve=[0,1,100,1], ae=[0,0,25,1,75,1,100,0]):
    start = seconds2samples(beg)
    end = start + seconds2samples(dur)
    ampf = make_env(ae, scaler=amp, duration=dur)
    frq0 = hz2radians(f0)
    frq1 = hz2radians(f1)
    frq2 = hz2radians(f2)
    foflen = 100 if CLM.srate == 22050 else 200
    vibr = make_oscil(6)
    vibenv = make_env(ve , scaler=vib, duration=dur)
    win_freq = (2 * math.pi) / foflen
    foftab = np.zeros(foflen)
    wt0 = make_wave_train(frq, foftab)
    for i in range(foflen):
        #this is not the pulse shape used by B&R
        foftab[i] = ((a0 * math.sin(i*frq0)) + (a1 * math.sin(i*frq1)) +
                    (a2 * math.sin(i*frq2))) * .5 * (1.0 - math.cos(i * win_freq))
              
    for i in range(start, end):
        outa(i, env(ampf) * wave_train(wt0, env(vibenv) * oscil(vibr)))  

with Sound(play=True):
    fofins(0, 1, 270, .2, .001, 730, .6, 1090, .3, 2440, .1) # "Ahh"


with Sound( play = True ): #one of JC's favorite demos
    fofins(0,4,270,.2,0.005,730,.6,1090,.3,2440,.1,[0,0,40,0,75,.2,100,1],
            [0,0,.5,1,3,.5,10,.2,20,.1,50,.1,60,.2,85,1,100,0])
    fofins(0,4,(6/5 * 540),.2,0.005,730,.6,1090,.3,2440,.1,[0,0,40,0,75,.2,100,1],
            [0,0,.5,.5,3,.25,6,.1,10,.1,50,.1,60,.2,85,1,100,0])
    fofins(0,4,135,.2,0.005,730,.6,1090,.3,2440,.1,[0,0,40,0,75,.2,100,1],
            [0,0,1,3,3,1,6,.2,10,.1,50,.1,60,.2,85,1,100,0])
            
            
            
            
def when(start, duration, start_freq, end_freq, grain_file):
    beg = seconds2samples(start)
    length = seconds2samples(duration)
    end = beg + length
    grain_dur = mus_sound_duration(grain_file)
    frqf = make_env([0,0,1,1], scaler = hz2radians(end_freq - start_freq), duration=duration)
    click_track = make_pulse_train(start_freq)
    grain_size = seconds2samples(grain_dur)
    grain_arr = file2array(grain_file, 0, 0, grain_size) #different than scheme version
    grains = make_wave_train(start_freq, grain_arr) # TODO: make_wave_train just take size arg
    original_grain = np.copy(grain_arr)
    ampf = make_env([0,1,1,0], scaler=.7, offset=.3, duration=duration, base=3.0)
     #grain = grains.mus_data this needs work TODO:
    grain = mus_data(grains)
#     
    for i in range(beg, end):
        gliss = env(frqf)
        outa(i, env(ampf) * wave_train(grains, gliss))
        click = pulse_train(click_track, gliss)
        if click > 0.0:
            scaler = max(.1, (1.0 * ((i - beg) / length)))
            comb_len = 32
            c1 = make_comb(scaler, comb_len)
            c2 = make_comb(scaler, math.floor(comb_len * .75))
            c3 = make_comb(scaler, math.floor(comb_len * 1.25))
            for k in range(0, grain_size):
                x = original_grain[k]
                #print(x)
                grain[k] = .25 * (comb(c1, x) + comb(c2, x) + comb(c3, x))


with Sound(play=True):
    when(0, 4, 2.0, 8.0, 'flute_trill_1.wav')



def fractal(start, duration, m, x, amp):
    # use formula of M J Feigenbaum
    beg = seconds2samples(start)
    end = beg + seconds2samples(duration)
    for i in range(beg, end):
        outa(i, amp * x)
        x = (1.0 - (m * x * x))

with Sound():
    fractal(0, 1, .5, 0, .5)
    
with Sound():
    fractal(0, 1, 1.5, .20, .2)


def attract(beg, dur, amp, c):
    # by James McCartney, from CMJ vol 21 no 3 p 6
    st = seconds2samples(beg)
    nd = st + seconds2samples(dur)
    a = .2
    b = .2
    dt = .04
    scale = (.5 * amp) / c
    x1 = 0.0
    x = -1.
    y = 0.0
    z = 0.0
    for i in range(st,nd):
        x1 = x - (dt * (y+z))
        y += dt * (x + (a * y))
        z += dt * ((b+(x*z)) - (c * z))
        x = x1
        outa(i, (scale*x))
        
            
with Sound(play=True):
    attract(0, 2, .5, 7.4)
    attract(2, 2, .5, 4.19)
    attract(4, 2, .5, 6.37)
    
        
        
def test_filter(flt):
    osc = make_oscil()
    samps = seconds2samples(.5)
    ramp = make_env([0,0,1,1], scaler=hz2radians(samps), length=samps)
    with Sound():
        for i in range(0, samps):
            outa(i, flt(oscil(osc, env(ramp))))    
            
test_filter(make_one_zero(.5,.5))  
test_filter(make_one_pole(.1,-.9))      
test_filter(make_two_pole(.1,.1,.9))  
test_filter(make_two_zero(.5,.2,.3))  



def echo(beg, dur, scaler, secs, file):
    dly = make_delay(seconds2samples(secs))
    rd = make_readin(file)
    for i in range(beg,dur):
        inval = rd()
        outa(i, inval + delay(dly, scaler * (tap(dly) + inval)))
        
with Sound(play=True):
    echo(0, 60000, .5, 1.0, 'pistol.snd')
    



# another version in clm_ins.py
def zc(time, dur, freq, amp, length1, length2, feedback):
    beg = seconds2samples(time)
    end = seconds2samples(time + dur)
    s = make_pulse_train(freq)
    d0 = make_comb(feedback, size=length1, max_size=(max(length1, length2)))
    aenv = make_env([0,0,.1,1,.9,1,1,0], scaler=amp, duration=dur)
    zenv = make_env([0,0,1,1], scaler=(length2-length1), base=12, duration=dur)
    for i in range(beg, end):
        outa(i, env(aenv) * comb(d0, pulse_train(s), env(zenv)))
        
        
with Sound(play=True):
    zc(0,3,100,.1,20,100,.5)
    zc(3.5,3,100,.1,90,100,.95)


