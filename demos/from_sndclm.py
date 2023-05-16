# Examples from https://ccrma.stanford.edu/software/snd/snd/sndclm.html#oscildoc
# showing basic examples of built-in generators
from pysndlib import *
import math


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
        outa(i, .1 * polywave(gen))


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
