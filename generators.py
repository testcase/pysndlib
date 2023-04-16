#! /Users/toddingalls/Developer/Python/venvs/pysndlib-venv/bin/python

import math

from pysndlib import *

NEARLY_ZERO = 1.0e-10

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)


# --------------- nssb ---------------- #
make_nssb, is_nssb = make_generator('nssb', {'frequency' : 0.0, 'ratio' : 1.0, 'n' : 1, 'angle' : 0.0})

def nssb(gen, fm=0.0):
    cx = gen.angle
    mx = cx * ratio
    den = math.sin(.5 * mx)
    gen.angle += fm + gen.frequency
    if math.fabs(den) <  NEARLY_ZERO:
        return -1.0
    else:
       return ((math.sin(cx) * math.sin(mx * ((n + 1) / 2)) * math.sin((n * mx) / 2)) - (math.cos(cx) * .5 * (den * math.sin(mx * (n + .5))))) /  ((n + 1) * den)


# TODO: --------------- nxysin ---------------- #



# --------------- nxycos ---------------- #

make_nxycos, is_nxycos = make_generator('nxycos', {'frequency' : 0.0, 'ratio' : 1.0, 'n' : 1, 'angle' : 0.0})

def nxycos(gen, fm=0.0):
    x = gen.angle
    y = x * gen.ratio
    den = math.sin(y * .5)
    gen.angle += frequency + fm
    if math.fabs(den) < NEARLY_ZERO:
        return 1.0
    else:
        return (math.cos(x + (.5 * (n - 1) * y)) * math.sin(.5 * n * y)) / (n * den)
    
# --------------- nxy1cos ---------------- #

make_nxy1cos, is_nxy1cos = make_generator('nxy1cos', {'frequency' : 0.0, 'ratio' : 1.0, 'n' : 1, 'angle' : 0.0})

def nxy1cos(gen, fm=0.0):
    x = gen.angle
    y = x * gen.ratio
    den = math.cos(y * .5)
    gen.angle += frequency + fm
    if math.fabs(den) < NEARLY_ZERO:
        return -1.0
    else:
        return max(-1., min(1.0, (math.sin(n * y) * math.sin(x + ((n - .5) * y))) / (2 * n * den)))
        
        
# --------------- nxy1sin ---------------- #

make_nxy1sin, is_nxy1sin = make_generator('nxy1sin', {'frequency' : 0.0, 'ratio' : 1.0, 'n' : 1, 'angle' : 0.0})

def nxy1sin(gen, fm=0.0):
    x = gen.angle
    y = x * gen.ratio
    den = math.cos(y * .5)
    gen.angle += frequency + fm
    return (math.sin(x + (.5 * (n - 1) * (y + math.pi))) * math.sin(.5 * n * (y + math.pi))) / (n * den)
    
    


# TODO: --------------- noddsin ---------------- #



# --------------- noddcos ---------------- #

make_noddcos, is_noddcos = make_generator('noddcos', {'frequency' : 0.0, 'n' : 1, 'angle' : 0.0}, wrapper = convert_frequency)

def noddcos(gen, fm=0.0):
    cx = gen.angle
    den = 2 * n * math.sin(gen.angle)
    angle += fm + frequency
    
    if math.fabs(den) < NEARLY_ZERO:
        fang = cx % (2 * math.pi)
        if (fang < .001) or (math.fabs(fang - (2 * math.pi)) < .001):
            return 1.0
        else: 
            return -1.0
    else:
        return math.sin(2 * n * cx) / den
        

# --------------- noddssb ---------------- #


make_noddssb, is_noddssb = make_generator('noddssb', {'frequency' : 0.0, 'ratio' : 1.0, 'n' : 1, 'angle' : 0.0}, wrapper = convert_frequency)


def noddssb(gen, fm=0.0):
    cx = gen.angle
    mx = cx * gen.ratio
    x = cx - mx
    sinnx = math.sin(n * mx)
    den = n * math.sin(mx)
    gen.angle += fm + frequency
    
    if math.fabs(den) < NEARLY_ZERO:
        if (mx % (2 * math.pi)) < .1:
            return -1
        else:
            return 1
    else:
        return (math.sin(x) * ((sinnx*sinnx) / den)) - ((math.cos(x) * (math.sin(2 * n * mx)) / (2 * den)))


# --------------- ncos2 ---------------- #

make_ncos2, is_ncos2 = make_generator('ncos2', {'frequency' : 0.0, 'n' : 1, 'angle' : 0.0}, wrapper = convert_frequency)

def ncos2(gen, fm=0.0):
    x = gen.angle
    den = math.sin(.5 * x)
    gen.angle += fm + frequency
    if math.fabs(den) < NEARLY_ZERO:
        return 1.0
    else:
        val = math.sin(0.5 * (n + 1) * x) / ((n + 1) * den)
        return val * val
        
        
# --------------- ncos4 ---------------- #

make_ncos4, is_ncos4 = make_generator('ncos4', {'frequency' : 0.0, 'n' : 1, 'angle' : 0.0}, wrapper = convert_frequency)

def ncos4(gen, fm=0.0):
    val = ncos2(gen, fm)
    return val * val
  
# TODO: --------------- npcos ---------------- #      



# TODO: --------------- ncos5 ---------------- #  

# TODO: --------------- nsin5 ---------------- #  


# TODO: --------------- nrsin ---------------- #  

# TODO: --------------- nrcos ---------------- #  


# TODO: --------------- nrssb ---------------- #  

# TODO: --------------- nkssb ---------------- #  

# TODO: --------------- nsincos ---------------- #  


# TODO: --------------- n1cos ---------------- #  


# TODO: --------------- npos1cos ---------------- #  


# TODO: --------------- npos3cos ---------------- #  

# --------------- rcos ---------------- #

def rcos_wrapper(g):
    g.osc = make_oscil(g.frequency, .5 * math.pi)
    g.r = clamp(g.r, -.999999, .999999)
    g.rr = g.r * g.r
    g.rrp1 = 1.0 + g.rr
    g.rrm1 = 1.0 - g.rr
    g.r2 = 2. * g.r
    absr = math.fabs(g.r)
    g.norm = 0.0 if absr < NEARLY_ZERO else (1.0 - absr) /  ( 2.0 * absr)
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


# TODO: --------------- rssb ---------------- #  

# TODO: --------------- rxysin ---------------- #  


# TODO: --------------- rxycos ---------------- #  


# TODO: --------------- safe-rxycos ---------------- #  

# TODO: --------------- ercos ---------------- #  

# TODO: --------------- erssb ---------------- #  

# TODO: --------------- r2sin ---------------- #  

# TODO: --------------- r2cos ---------------- #  

# TODO: --------------- r2ssb ---------------- #  

# TODO: --------------- eoddcos ---------------- #  

# TODO: --------------- rkcos ---------------- #  

# TODO: --------------- rksin ---------------- #  

# TODO: --------------- rkssb ---------------- #  

# TODO: --------------- rk!cos ---------------- #  

# TODO: --------------- rk!ssb ---------------- #  

# TODO: --------------- rxyk!sin ---------------- # 

# TODO: --------------- rxyk!cos ---------------- # 

# TODO: --------------- r2k!cos ---------------- # 

# TODO: --------------- k2sin ---------------- # 

# TODO: --------------- k2cos ---------------- # 

# TODO: --------------- k2ssb ---------------- # 

# TODO: --------------- dblsum ---------------- # 



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


# TODO: --------------- krksin ---------------- # 


# TODO: --------------- abcos ---------------- # 

# TODO: --------------- absin ---------------- # 

# TODO: --------------- r2k2cos ---------------- # 

# TODO: --------------- blsaw ---------------- # 

# TODO: --------------- asyfm ---------------- # 

# TODO: --------------- bess ---------------- # 

# TODO: --------------- jjcos ---------------- # 

# TODO: --------------- j0evencos ---------------- # 

# TODO: --------------- j2cos ---------------- # 

# --------------- fmssb ---------------- #

make_fmssb, is_fmssb = make_generator('fmssb', {'frequency' : 0.0, 'ratio' : 1.0, 'index' : 1.0, 'angle' : 0.0}, wrapper=convert_frequency)

def fmssb(gen, fm=0.):
    cx = gen.angle
    mx = cx * gen.ratio
    gen.angle += fm + gen.frequency
    
    return (math.cos(cx) * (math.sin(gen.index * math.cos(mx)))) - (math.sin(cx) * (math.sin(gen.index * math.sin(mx))))



    