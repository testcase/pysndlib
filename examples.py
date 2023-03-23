#! /Users/toddingalls/Developer/Python/pysndlib/venv/bin/python



from musx import *
from pysndlib import *
import math
import random

# with Sound("out1.aif", 1):
# 	gen = make_oscil(440.0)
# 	for i in range(44100):
# 		outa(i,  oscil(gen))


# with Sound("out1.aif", 1):
# 	gen = make_oscil(440.0)
# 	ampf = make_env([0., 0., 0.01, 1.0, 0.25, 0.1, 1, 0], scaler=.5, length=44100)
# 	for i in range(44100):
# 		outa(i,  env(ampf)*oscil(gen))
	
# 	
# with Sound("out1.aif", 1) as out:
# 	gen = make_table_lookup(440.0, wave=partials2wave([1.0, 0.5, 2.0, 0.5, 3.0, .2, 5.0, .1]))
# 	for i in range(44100):
# 		outa(i, .5* table_lookup(gen))

# TODO Not right
# with Sound("out1.aif", 1) as out:
# 	gen = make_polywave(100., partials=[1.0, 0.5, 2.0, 0.3, 3.0, .1, 5.0, .1])
# 	for i in range(44100):
# 		outa(i, .4 * polywave(gen), out.output)


# with Sound("out1.aif", 1):
# 	gen = make_triangle_wave(440.0)
# 	for i in range(44100):
# 		outa(i, .5 * triangle_wave(gen))

# with Sound("out1.aif", 1):
# 	gen = make_ncos(440.0, 10);
# 	for i in range(44100):
# 		outa(i, .5 * ncos(gen))


# with Sound("out1.aif", 1):
# 	gen = make_nrxycos(440.0, 1.1, 10, 0.3)
# 	for i in range(44100):
# 		outa(i, .5 * nrxycos(gen))

# with Sound("out1.aif", 1):
# 	shifter = make_ssb_am(440.0, 20)
# 	osc = make_oscil(440.0)
# 	for i in range(44100):
# 		outa(i, .5 * ssb_am(shifter, oscil(osc)))

# with Sound("out1.aif", 1) as out:
# 	v = np.zeros(64, dtype=np.double)
# 	g = make_ncos(400,10)
# 	g.mus_phase = -.5 * np.pi
# 	for i in range(64):
# 		v[i] = ncos(g)
# 	gen = make_wave_train(440, wave=v)	
# 	for i in range(44100):
# 		outa(i, .5 * wave_train(gen))
		
		

# with Sound("out1.aif", 2):
# 	ran1 = make_rand(5.0, hz2radians(220.0))
# 	ran2 = make_rand_interp(5.0, hz2radians(220.0))
# 	osc1 = make_oscil(440.0)
# 	osc2 = make_oscil(1320.0)
# 	for i in range(88200):
# 		outa(i, 0.5 * oscil(osc1, rand(ran1)))
# 		outb(i, 0.5 * oscil(osc2, rand_interp(ran2)))
		
		
# with Sound("out1.aif", 1):
# 	flt = make_two_pole(1000.0, 0.999)
# 	ran1 = make_rand(10000.0, 0.002)
# 	for i in range(44100):
# 		outa(i, 0.5 * two_pole(flt, rand(ran1)))

# with Sound("out1.aif", 1):
# 	flt = make_firmant(1000.0, 0.999)
# 	ran1 = make_rand(10000.0, 5.0)
# 	for i in range(44100):
# 		outa(i, 0.5 * firmant(flt, rand(ran1)))

# TODO not working		
# with Sound("out1.aif", 1):
# 	flt = make_iir_filter(3, [0.0, -1.978, 0.998])
# 	ran1 = make_rand(10000.0, 0.02)
# 	print(flt)
# 	for i in range(44100):
# 		outa(i, 0.5 * iir_filter(flt, rand(ran1)))

# with Sound("out1.aif", 1):
# 	dly = make_delay(seconds2samples(0.5))
# 	osc1 = make_oscil(440.0)
# 	osc2 = make_oscil(660.0)
# 	for i in range(44100):
# 		outa(i, 0.5 * (oscil(osc1) + delay(dly, oscil(osc2))))		

# with Sound("out1.aif", 1):
# 	cmb = make_comb(0.4, seconds2samples(0.4))
# 	osc = make_oscil(440.0)
# 	ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
# 	
# 	for i in range(88200):
# 		outa(i, 0.5 * (comb(cmb, env(ampf) * oscil(osc))))

# with Sound("out1.aif", 1):
# 	alp = make_all_pass(-0.4, 0.4, seconds2samples(0.4))
# 	osc = make_oscil(440.0)
# 	ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
# 	for i in range(88200):
# 		outa(i, 0.5 * (all_pass(alp, env(ampf) * oscil(osc))))
		

# with Sound("out1.aif", 1):
# 	avg = make_moving_average(4410)
# 	osc = make_oscil(440.0)
# 	stop = 44100 - 4410
# 	
# 	for i in range(stop):
# 		val = oscil(osc)
# 		outa(i, val * moving_average(avg, abs(val))) 
# 	for i in range(stop, 44100):
# 		outa(i, oscil(osc) * moving_average(avg, 0.0))
		
	
# with Sound("out1.aif", 1):	
# 	rd = make_readin("oboe.wav");
# 	length = 2 * mus_sound_framples("oboe.wav")
# 	print(length)
# 	sr = make_src(rd, .5);
# 	
# 	for i in range(length):
# 		outa(i, src(sr))
	
	
# with Sound("out1.aif", 1):	
# 	rd = make_readin("oboe.wav");
# 	grn = make_granulate(rd, 2., hop=.3, jitter=1.4)
# 	length = 2 * mus_sound_framples("oboe.wav")
# 	for i in range(length):
# 		outa(i, granulate(grn))
	
		
# with Sound("out1.aif", 1):	
# 	rd = make_readin("oboe.wav");
# 	pv = make_phase_vocoder(rd, pitch=2.)
# 	length = 2 * mus_sound_framples("oboe.wav")
# 
# 	for i in range(length):
# 		outa(i, phase_vocoder(pv))	

# with Sound("out1.aif", 1):	
# 	fm = make_asymmetric_fm(440.0, 0.0, 0.9, 0.5)
# 	for i in range(44100):
# 		outa(i, 0.5 * asymmetric_fm(fm, 1.0))
		
# with Sound("out1.aif", 1):	
# 	reader = make_readin("oboe.wav")
# 	
# 	for i in range(44100):
# 		outa(i, readin(reader))

		
# with Sound("out1.aif", 1):	
# 	rate = .5
# 	rd = make_readin("oboe.wav")
# 	sr = make_src(rd, rate);
# 	length = mus_sound_framples("oboe.wav")
# 	grn = make_granulate(lambda x: src(sr), rate)
# 	for i in range(length):
# 		outa(i, granulate(grn))




def jc_reverb(lowpass=False, volume=1., amp_env = None, tail=0):
	allpass1 = make_all_pass(-.7, .7, 1051)
	allpass2 = make_all_pass(-.7, .7, 337)
	allpass3 = make_all_pass(-.7, .7, 113)
	comb1 = make_comb(.742, 4799)
	comb2 = make_comb(.733, 4999)
	comb3 = make_comb(.715, 5399)
	comb4 = make_comb(.697, 5801)
	chans = Sound.output.mus_channels
	
	length = Sound.reverb.mus_length + seconds2samples(1.)
	filts = [make_delay(seconds2samples(.013))] if chans == 1 else [make_delay(seconds2samples(.013)),make_delay(seconds2samples(.011)) ]
	combs = make_comb_bank([comb1, comb2, comb3, comb4])
	allpasses = make_all_pass_bank([allpass1,allpass2,allpass3])
	
	if lowpass or amp_env:
		flt = make_fir_filter(3, [.25, .5, .25]) if lowpass else None
		envA = make_env(amp_env, scaler=volume, duration = length / mus_srate())
		
		if lowpass:
			for i in range(length):
				out_bank(filts, i, (env(envA) * fir_filter(flt, comb_bank(combs, all_pass(allpasses, ina(i, Sound.reverb))))))
		else:
			for i in range(length):
				out_bank(filts, i, (env(envA) * comb_bank(combs, all_pass_bank(allpasses, ina(i, Sound.reverb)))))
	else:
		if chans == 1:
			
			gen = filts[0]
			for i in range(length):
				outa(i, delay(gen), volume * comb_bank(combs, all_pass_bank(allpasses, ina(i, Sound.reverb))))
		else:	
			gen1 = filts[0]
			gen2 = filts[1]
			for i in range(length):
				val = volume * comb_bank(combs, all_pass_bank(allpasses, ina(i, Sound.reverb))) 
				outa(i, delay(gen1, val))
				outb(i, delay(gen2, val))
			


def blip(beg, dur, freq):
	start = seconds2samples(beg)
	end = seconds2samples(dur) + start
	
	loc = make_locsig(degree=random.randrange(-90, 90), distance=1., reverb=.7)
	cmb = make_comb(0.4, seconds2samples(0.4))
	osc = make_oscil(freq)
	ampf = make_env([0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0], length=4410)
	
	for i in range(start, end):
		locsig(loc, i, 0.5 * (comb(cmb, env(ampf) * oscil(osc))))

	
with Sound("out1.aif", 2, reverb=jc_reverb, reverb_channels=2, play=True):	
	blip(0, 2, 100)
	blip(1, 2, 400)
	blip(2, 2, 200)
	blip(3, 2, 300)
	blip(4, 2, 500)
