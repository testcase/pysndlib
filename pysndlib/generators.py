import math
import random
from pysndlib import *
# from .internals.utils import clamp
NEARLY_ZERO = 1.0e-10
TWO_PI = math.pi * 2


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

# TODO: --------------- jpcos ---------------- # 

# TODO: --------------- jncos ---------------- # 

# TODO: --------------- j0j1cos  ---------------- # 

# TODO: --------------- jycos  ---------------- # 

# TODO: --------------- jcos  ---------------- # 

# TODO: --------------- blackman  ---------------- # 

# --------------- fmssb ---------------- #

make_fmssb, is_fmssb = make_generator('fmssb', {'frequency' : 0.0, 'ratio' : 1.0, 'index' : 1.0, 'angle' : 0.0}, wrapper=convert_frequency)

def fmssb(gen, fm=0.):
    cx = gen.angle
    mx = cx * gen.ratio
    gen.angle += fm + gen.frequency
    
    return (math.cos(cx) * (math.sin(gen.index * math.cos(mx)))) - (math.sin(cx) * (math.sin(gen.index * math.sin(mx))))



# TODO: --------------- k3sin  ---------------- # 

# TODO: --------------- izcos  ---------------- # 

# TODO: --------------- adjustable-square-wave ---------------- # 

# TODO: --------------- adjustable-triangle-wave ---------------- # 

# TODO: --------------- adjustable-sawtooth-wave ---------------- # 

# TODO: --------------- adjustable-oscil-wave ---------------- # 

# TODO: --------------- make-table-lookup-with-env ---------------- # 

# TODO: --------------- make-wave-train-with-env ---------------- # 

# TODO: --------------- round-interp ---------------- # 


# TODO: --------------- env-any functions ---------------- # 

# TODO: --------------- run-with-fm-and-pm ---------------- # 

# TODO: --------------- nchoosekcos ---------------- # 

# TODO: --------------- sinc-train ---------------- # 

# --------------- pink-noise ---------------- # 

def pink_noise_wrapper(g):
    g.data = np.zeros(g.n*2, dtype=np.double)
    amp = 1.0 / (2.5 * math.sqrt(g.n))
    g.data[0] = amp
    for i in range(2,2*g.n,2):
        g.data[i] =  mus_random(amp)
        g.data[i+1] = random.random()
    return g
    
make_pink_noise, is_pink_noise = make_generator('pink_noise', {'n' : 1}, wrapper=pink_noise_wrapper,
    docstring="""Creates a pink-noise generator with n octaves of rand (12 is recommended).""")

def pink_noise(gen):
    """Returns the next random value in the 1/f stream produced by gen."""
    x = 0.5 
    s = 0.0
    amp = gen.data[0]
    size = gen.n*2

    for i in range(2,size,2):
        s += gen.data[i]
        gen.data[i+1] -= x
        
        if gen.data[i+1] < 0.0:
            gen.data[i] = mus_random(amp)
            gen.data[i+1] += 1.0
        x *= .5
    return s + mus_random(amp)

# --------------- brown-noise ---------------- # 

def brown_noise_wrapper(g):
    g.prev = 0.0 
    g.sum = 0.0
    g.gr = make_rand(g.frequency, g.amplitude)
    return g
    
make_brown_noise, is_brown_noise = make_generator('brown_noise', {'frequency' : 0, 'amplitude' : 1.0}, 
                                    wrapper=brown_noise_wrapper,
                                    docstring="""Returns a generator that produces brownian noise.""")

def brown_noise(gen, fm=0.0):
    """returns the next brownian noise sample"""
    val = rand(gen.gr, fm)
    if not val == gen.prev:
        gen.prev = val
        gen.sum += val
    return gen.sum




# --------------- green-noise ---------------- # 

def green_noise_wrapper(g):
    g.gr = make_rand(g.frequency, g.amplitude)
    g.sum = .5 * (g.low + g.high)
    g.prev = 0.0 
    return g
    
make_green_noise, is_green_noise = make_generator('green_noise', {'frequency' : 0.0, 'amplitude' : 1.0, 'low' : -1.0, 'high' : 1.0},    
                                wrapper=green_noise_wrapper, docstring="""returns a new green-noise (bounded brownian noise) generator.""")
                                
                                
def green_noise(gen, fm=0.0):
    """Returns the next sample in a sequence of bounded brownian noise samples."""
    val = rand(gen.gr, fm)
    if not val == gen.prev:
        gen.prev = val
        gen.sum += val
        if not (gen.low <= gen.sum <= gen.high):
            gen.sum -= 2*val
    return gen.sum        
    
    

# --------------- green-noise-interp ---------------- # 

def green_noise_interp_wrapper(g):
    g.sum = .5 * (g.low + g.high)
    g.dv = 1.0 / math.ceil(CLM.srate / max(1.0, g.frequency))
    convert_frequency(g)
    g.incr = mus_random(g.amplitude) * g.dv
    g.angle = 0.0
    return g
    
make_green_noise_interp, is_green_noise_interp = make_generator('green_noise_interp', {'frequency' : 0.0, 'amplitude' : 1.0, 'low' : -1.0, 'high' : 1.0},
                            wrapper = green_noise_interp_wrapper, docstring="""Returns a new interpolating green noise (bounded brownian noise) generator.""")

def green_noise_interp(gen, fm=0.0):
    """Returns the next sample in a sequence of interpolated bounded brownian noise samples."""
    if not (0.0 <= gen.angle and gen.angle <= TWO_PI):
        val = mus_random(gen.amplitude)
        gen.angle %= TWO_PI # in scheme version modulo used which should be same as %
        if gen.angle < 1.0:
            gen.angle += TWO_PI
        if not (gen.low <= (gen.sum+val) <= gen.high):
            val = min(gen.high-gen.sum, max(gen.low-gen.sum, -val))
        gen.incr = gen.dv * val
    gen.angle += fm + gen.frequency
    gen.sum += gen.incr
    return gen.sum
    
        

# TODO: --------------- moving-sum ---------------- # 

def moving_sum_wrapper(g):
    g.gen = make_moving_average(g.n)
    g.gen.mus_increment = 1.0
    return g

make_moving_sum, is_moving_sum = make_generator('moving_sum', {'n' : 128}, wrapper=moving_sum_wrapper,docstring="""Returns a moving-sum generator""")

def moving_sum(gen, inpt):
    """Returns the sum of the absolute values in a moving window over the last n inputs."""
    return moving_average(gen.gen, abs(inpt))

# TODO: --------------- moving-variance ---------------- # 

# TODO: --------------- moving-rms ---------------- # 

# TODO: --------------- moving-length ---------------- # 

# TODO: --------------- weighted-moving-average ---------------- # 

# TODO: --------------- exponentially-weighted-moving-average ---------------- # 

# --------------- polyoid ---------------- # 

def make_polyoid(frequency, partial_amps_and_phases):
    length = len(partial_amps_and_phases)
    n = 0
    for i in range(0, length, 3):
        n = max(n, math.floor(partial_amps_and_phases[i]))   
    topk = n + 1
    sin_amps = np.zeros(topk, dtype=np.double)
    cos_amps = np.zeros(topk, dtype=np.double)
    for j in range(0,length,3):
        n = math.floor((partial_amps_and_phases[j]))
        amp = partial_amps_and_phases[j+1]
        phase = partial_amps_and_phases[j+2]
        if n > 0:
            sin_amps[n] = amp * math.cos(phase)
        cos_amps[n] = amp * math.sin(phase)
    return make_polywave(frequency, xcoeffs=cos_amps, ycoeffs=sin_amps)

def is_polyoid(g):
    return is_polywave(g) and g.mus_channel == Polynomial.BOTH_KINDS
    
polyoid = polywave

#def polyoid_env

# --------------- noid ---------------- # 
# def make_noid(frequency=0.0, n=1, phases=None, choice='all'):  
# the full version of this requires potentially loading a file so
# wondering good way to do this. using maybe binary numpy files
       
    
    
    
#         
        


# TODO: --------------- knoid ---------------- # 

# TODO: --------------- roid ---------------- # 

# --------------- tanhsin ---------------- # 

# is_waveshape = is_polyshape


# TODO: --------------- tanhsin ---------------- # 

def tanhsin_wrapper(g):
    g.osc = make_oscil(g.frequency, g.initial_phase)
    g.frequency = hz2radians(g.frequency)
    return g
    
make_tanhsin, is_tanhsin = make_generator('tanhsin', {'frequency' : 0.0, 'r' : 1.0, 
            'initial_phase' : 0.0}, wrapper=tanhsin_wrapper,
            docstring="""Returns a tanhsin generator.""")
                            

def tanhsin(gen, fm=0.0):
    """Produces tanh(r*sin) which approaches a square wave as r increases."""
    return math.tanh(gen.r * oscil(gen.osc,fm))



# TODO: --------------- moving-fft ---------------- # 

def moving_fft_wrapper(g):
    g.rl = np.zeros(g.n, dtype=np.double)
    g.im = np.zeros(g.n, dtype=np.double)
    g.data = np.zeros(g.n, dtype=np.double)
    g.window = make_fft_window(g.window, g.n)
    s = np.sum(g.window)
   # g.window = g.window * (2./s)
    g.outctr = g.n+1
   # print(g.window)
    return g
    
moving_fft_methods = {'mus_data' : [lambda g : g.data, None],
                        'mus_xcoeffs' : [lambda g : g.rl, None],
                        'mus_ycoeffs' : [lambda g : g.im, None],
                        'mus_run' : [lambda g,arg1,arg2 : moving_fft(g), None]}

make_moving_fft, is_moving_fft = make_generator('moving_fft', 
                {'input' : False, 
                'n' : 512, 
                'hop' : 128, 
                'window' : Window.HAMMING},
                wrapper=moving_fft_wrapper, 
                methods=moving_fft_methods,
                docstring="""returns a moving-fft generator.""")

def moving_fft(gen):
    new_data = False
    if gen.outctr >= gen.hop:
        if gen.outctr > gen.n:
            for i in range(gen.n):
                gen.data[i] = readin(gen.input)
            mid = gen.n - gen.hop
            np.roll(gen.data, gen.hop)
            for i in range(mid,gen.n):
                gen.data[i] =readin(gen.input)   
        gen.outctr = 0
        new_data = True
        gen.im.fill(0.0)
        np.copyto(gen.rl, gen.data)
        np.multiply(gen.rl,gen.window, gen.rl)

        fft(gen.rl,gen.im, gen.n,1)
        rectangular2polar(gen.rl,gen.im)
    gen.outctr += 1
    return new_data


# TODO: --------------- moving-spectrum ---------------- # 

# TODO: --------------- moving-scentroid ---------------- # 

# TODO: --------------- moving-autocorrelation ---------------- # 

# TODO: --------------- moving-pitch ---------------- # 

# TODO: --------------- flocsig ---------------- # 