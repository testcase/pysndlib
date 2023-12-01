#! /Users/toddingalls/Developer/Python/venvs/pysndlib-venv/bin/python
import pysndlib.clm as CLM

import pysndlib.sndlib as SND

import pysndlib.clm_ins as INS

import pysndlib.jcrev as jcrev




def gimp(start=0,dur=1,freq=440,amp=.25,ampenv=[0,0, .15, 1, .85, 1, 1, 0],
    degree=45,
    distance=0,
    reverb_amount=0):

    beg = CLM.seconds2samples(start)

    end = CLM.seconds2samples(start + dur)

    osc = CLM.make_oscil(freq)

    ampenv = CLM.make_env(ampenv, scaler=amp, duration=dur)

    loc = CLM.make_locsig(degree=degree,distance=distance, reverb=reverb_amount)

    for i in range(beg, end):
        samp = CLM.env(ampenv) * CLM.oscil(osc)
        CLM.locsig(loc,i, samp)

#--------------------------------------------------------------------------------

ampenv = [0.0, 1.0, 0.1, 0.5, 0.2, 0.25, 0.3, 0.125, 0.4, 0.0625, 0.5, 0.03125,0.6, 0.015625, 0.7, 0.0078125, 0.8, 0.00390625, 0.9, 0.001953125, 1.0, 0.0]


    
with CLM.Sound("test.wav", play=True, channels=2, reverb=INS.nrev, reverb_data={'volume' : .2}, reverb_channels=1):
    gimp(0,.15, 440, .7, ampenv, degree=45,distance=1,reverb_amount=.15)
    
with CLM.Sound("test.wav", play=True, channels=2, reverb=INS.nrev,reverb_channels=1):
    gimp(0,.15, 440, .7, ampenv, degree=45,distance=1,reverb_amount=.15)
    
with CLM.Sound("test.wav", play=True, channels=2, reverb=INS.nrev(),reverb_channels=1):
    gimp(0,.15, 440, .7, ampenv, degree=45,distance=1,reverb_amount=.15)
    
with CLM.Sound("test.wav", play=True, channels=2, reverb=INS.nrev(1.1, .5, .5, 2.),reverb_channels=1):
    gimp(0,.15, 440, .7, ampenv, degree=45,distance=1,reverb_amount=.15)

with CLM.Sound("test.wav", play=True, channels=2, reverb=INS.nrev(decay_time=3),reverb_channels=1):
    gimp(0,.15, 440, .7, ampenv, degree=45,distance=1,reverb_amount=.15)

with CLM.Sound("test.wav", play=True, channels=2, reverb=jcrev.jc_reverb,reverb_channels=1):
    gimp(0,.15, 440, .7, ampenv, degree=45,distance=1,reverb_amount=.15)
    
with CLM.Sound("test.wav", play=True, channels=2, reverb=jcrev.jc_reverb,reverb_data={'volume' : .4}, reverb_channels=1):
    gimp(0,.15, 440, .7, ampenv, degree=45,distance=1,reverb_amount=.15)
    
with CLM.Sound("test.wav", play=True, channels=2, reverb=jcrev.jc_reverb(decay_time=3),reverb_channels=1):
    gimp(0,.15, 440, .7, ampenv, degree=45,distance=1,reverb_amount=.15)
    
with CLM.Sound("test.wav", play=True, channels=2, reverb=jcrev.jc_reverb(True, .2, [0.0, 1., .2, .1], 2),reverb_channels=1):
    gimp(0,.15, 440, .7, ampenv, degree=45,distance=1,reverb_amount=.15)
    
