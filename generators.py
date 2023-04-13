#! /Users/toddingalls/Developer/Python/venvs/pysndlib-venv/bin/python

from pysndlib import *
import math


def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)


# --------------- fmssb ---------------- #

make_fmssb, is_fmssb = make_generator('fmssb', {'frequency' : 0.0, 'ratio' : 1.0, 'index' : 1.0, 'angle' : 0.0}, wrapper=convert_frequency)

def fmssb(gen, fm=0.):
    cx = gen.angle
    mx = cx * gen.ratio
    gen.angle += fm + gen.frequency
    
    return (math.cos(cx) * (math.sin(gen.index * math.cos(mx)))) - (math.sin(cx) * (math.sin(gen.index * math.sin(mx))))




# --------------- rkoddssb ---------------- #

def rkoddssb_wrapper(g):
    g = convert_frequency(g)
    g.r = clamp(g.r, -.999999, .999999)
    g.rr1 = 1.0 + (g.r * g.r)
    g.norm = 1.0 / (math.log(1.0 + g.r) - math.log(1.0 - g.r))
    return g
    
def rkoddssb_setter(g, val):
    g.r = clamp(val, -.999999, .9999999)
    g.rr1 = 1.0 + (g.r * g.r)
    g.norm = 1.0 / (math.log(1.0 + g.r) - math.log(1.0 - g.r))

make_rkoddssb, is_rkoddssb = make_generator('rkoddssb', {'frequency' : 0.0, 'ratio' : 1.0, 'r' : .5, 'angle' : 0.0, 'fm' : 0.0, 'rr1' : 0.0, 'norm' : 0.0},
                            wrapper=rkoddssb_wrapper, 
                            methods={'mus_scaler' : [lambda gen : gen.r, rkoddssb_setter]})


def rkoddssb(gen, fm=0.):
    cx = gen.angle
    mx = cx * gen.ratio
    cxx = cx - mx
    cmx = 2.0 * gen.r * math.cos(mx)
    gen.angle += fm + gen.frequency
    return (((math.cos(cxx) * .5 * math.log((gen.rr1+cmx) / (gen.rr1-cmx))) - (math.sin(cxx) * (math.atan2((2.0 * gen.r * math.sin(mx)), (1.0 - (gen.r*gen.r)))))) * gen.norm)


# --------------- rcos ---------------- #

def rcos_wrapper(g):
    g.osc = make_oscil(g.frequency, .5 * math.pi)
    g.r = clamp(g.r, -.999999, .999999)
    g.rr = g.r * g.r
    g.rrp1 = 1.0 + g.rr
    g.rrm1 = 1.0 - g.rr
    g.r2 = 2. * g.r
    absr = math.fabs(g.r)
    g.norm = 0.0 if absr < 1.0e-10 else (1.0 - absr) /  ( 2.0 * absr)
    return g
    
def rcos_set_mus_frequency(g,vl):
    g.osc.frequency = val
    
def rcos_set_mus_scaler(g,val):
    g.r = clamp(val, -.999999, .999999)
    g.rr = g.r*g.r
    g.rrp1 = 1.0 + g.rr
    g.rrm1 = 1.0 - g.rr
    g.r2 = 2.0 * g.r
    absr = math.fabs(g.r)
    g.norm = 0.0 if absr < 1.0e-10 else (1.0 - absr) /  ( 2.0 * absr)
    return val
    
make_rcos, is_rcos = make_generator('rcos', {'frequency' : 0.0, 'r' : .5, 'fm' : 0.0, 'osc' : False, 'rr' : 0.0, 'norm' : 0.0, 'rrp1' : 0.0, 'rrm1' : 0.0, 'r1' : 0},
                        wrapper = rcos_wrapper,
                        methods = {'mus_frequency' : [lambda g : g.osc.mus_frequency, rcos_set_mus_frequency], 
                                    'mus_scale' : [lambda g: g.r, rcos_set_mus_scaler]})
    
def rcos(gen, fm=0.):
    return ((gen.rrm1 / (gen.rrp1 - (gen.r2 * oscil(gen.osc, fm)))) - 1.0) * gen.norm
    