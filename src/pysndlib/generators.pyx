import math
import cython
import random
import pysndlib.clm as clm
cimport pysndlib.clm as clm
import numpy as np
cimport numpy as np



cdef cython.double NEARLY_ZERO = 1.0e-10
cdef cython.double TWO_PI = math.pi * 2
cdef cython.double PI = math.pi


cdef class mus_any_user:
    cpdef cython.double call(self, cython.double fm =0.0):
        pass
#     
# cdef class mus_any_tmp(mus_any_user):
#     cdef cython.double frequency 
#     cdef cython.double n
#     
#     
#     def __init__(self, frequency, n):
#         self.frequency = clm.hz2radians(frequency)
#         self.n = n
#         self.angle = 0.0
#     
# 
#     cpdef cython.double call(self, cython.double = fm=0.0):
# 
#                 
#     def __call__(self, cython.double fm=0.) -> cython.double:
#         return self.call(fm)
#         
# 
# cpdef mus_any_tmp make_tmp(frequency=0, n=1):
#     return mus_any_npcos(frequency, n)
#     
# cpdef cython.double tmp(gen, cython.double fm=0.):
#     return gen.call(fm)
#     
# cpdef cython.bint is_tmp(gen):
#     return isinstance(gen, mus_any_tmp)    
     
    
# TODO: Some nice printed descriptions

# --------------- nssb ---------------- #
cdef class Nssb(mus_any_user):

    def __init__(self, frequency, ratio=1., n=1):
        self.frequency = clm.hz2radians(frequency)
        self.ratio = ratio
        self.n = n
        self.angle = 0.0
           
    cpdef cython.double call(self, cython.double fm =0.0) :
        cdef cython.double cx  = self.angle
        cdef cython.double mx  = cx * self.ratio
        cdef cython.double den  = math.sin(.5 * mx)
        cdef cython.double n  = self.n
        self.angle += fm + self.frequency
        if math.fabs(den) <  NEARLY_ZERO:
            # return -1.0 in original but seems to cause weird inconsistencies
            den = NEARLY_ZERO
        
        return ((math.sin(cx) * math.sin(mx * ((n + 1.) / 2.)) * math.sin((n * mx) / 2.)) - 
                (math.cos(cx) * .5 * (den + math.sin(mx * (n + .5))))) /  ((n + 1.) * den)
    
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)

 
cpdef Nssb make_nssb(frequency=0, ratio=1., n=1):
    """Creates an nssb generator.
        nssb is the single side-band version of ncos and nsin. It is very similar to nxysin and nxycos.
    """
    return Nssb(frequency, ratio, n)
    
     
cpdef cython.double nssb(gen: Nssb, fm: cython.double = 0.):
    return gen.call(fm)
    
     
cpdef cython.bint is_nssb(gen):
    return isinstance(gen, Nssb)

# --------------- nxysin ---------------- #
 
def nodds(x, n):
    den = math.sin(x)
    num = math.sin(n * x)
    if den == 0.0:
        return 0.0
    else:
        return (num*num) / den
        
def find_nxysin_max(n, ratio):
    def ns(x, n):
        a2 = x / 2
        den = math.sin(a2)
        if den == 0.0:
            return 0.0
        else:
            return (math.sin(n*a2) * math.sin((1 + n) * a2)) / den
            
    def find_mid_max(n, lo, hi):
        mid = (lo + hi) / 2
        ylo = ns(lo, n)
        yhi = ns(hi, n)
        if math.fabs(ylo - yhi) < NEARLY_ZERO:
            return ns(mid,n)
        else:
            if ylo > yhi:
                return find_mid_max(n, lo, mid)   
            else:
                return find_mid_max(n, mid, hi)    
    
    def find_nodds_mid_max(n, lo, hi):
        mid = (lo + hi) / 2
        ylo = nodds(lo, n)
        yhi = nodds(hi, n)
        if math.fabs(ylo - yhi) < NEARLY_ZERO:
            return nodds(mid,n)
        else:
            if ylo > yhi:
                return find_nodds_mid_max(n, lo, mid)   
            else:
                return find_nodds_mid_max(n, mid, hi)    
            
    if ratio == 1:
        return find_mid_max(n, 0.0, PI / (n + .5))
    elif ratio == 2:
        return find_nodds_mid_max(n, 0.0, PI / ((2 * n) + .5))
    else:
        return n   


cdef class Nxysin(mus_any_user):
    
    def __init__(self, frequency, ratio=1.0, n=1):
        self.frequency = clm.hz2radians(frequency)
        self.ratio = ratio
        self.n = n
        self.angle = 0.0
        self.norm = 1.0 / find_nxysin_max(self.n, self.ratio)


    cpdef cython.double call(self, cython.double fm =0.0):
        cdef cython.double x  = self.angle
        cdef cython.double y  = x * self.ratio
        cdef cython.double den  = math.sin(y * .5)
        cdef cython.double n  = self.n
        cdef cython.double norm  = self.norm
        self.angle += fm + self.frequency
        if math.fabs(den) <  NEARLY_ZERO:
            return 0.0
        return ((math.sin(x + (0.5 * (n - 1) * y)) * math.sin(0.5 * n * y) * norm) / den)
                
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)
 

cpdef Nxysin make_nxysin(frequency=0, ratio=1., n=1):
    """Creates an nxysin generator."""
    return Nxysin(frequency, ratio, n)

cpdef cython.double nxysin(Nxysin gen, cython.double fm=0.):
    """returns n sines from frequency spaced by frequency * ratio."""
    return gen.call(fm)

cpdef cython.bint is_nxysin(gen):
    return isinstance(gen, Nxysin)        

# --------------- nxycos ---------------- #

cdef class Nxycos(mus_any_user):
    
    def __init__(self, frequency, ratio=1., n=1):
        self.frequency = clm.hz2radians(frequency)
        self.ratio = ratio
        self.n = n
        self.angle = 0.0
    
  
    cpdef cython.double call(self, cython.double fm =0.0):
        cdef cython.double x  = self.angle
        cdef cython.double y  = x * self.ratio
        cdef cython.double den  = math.sin(y * .5)
        cdef cython.double n = self.n
        self.angle += fm + self.frequency
        if math.fabs(den) <  NEARLY_ZERO:
            return 1.
        return (math.cos(x + (0.5 * (n - 1) * y)) * math.sin(.5 * n * y)) / (n * den)
                
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)
        

cpdef Nxycos make_nxycos(frequency=0, ratio=1., n=1):
    """Creates an nxycos generator."""
    return Nxycos(frequency, ratio, n)

cpdef cython.double nxycos(Nxycos gen, cython.double fm =0.):
    """returns n cosines from frequency spaced by frequency * ratio."""
    return gen.call(fm)

cpdef cython.bint is_nxycos(gen):
    return isinstance(gen, Nxycos)        

      
# --------------- nxy1cos ---------------- #

cdef class Nxy1cos(mus_any_user):
    
    def __init__(self, frequency, ratio=1.0, n=1):
        self.frequency = clm.hz2radians(frequency)
        self.ratio = ratio
        self.n = n
        self.angle = 0.0

    cpdef cython.double call(self, cython.double fm =0.0):
        cdef cython.double x = self.angle
        cdef cython.double y = x * self.ratio
        cdef cython.double den = math.cos(y * .5)
        cdef cython.double n = self.n
        self.angle += self.frequency + fm
        if math.fabs(den) < NEARLY_ZERO:
            return -1.0
        else:
            return max(-1., min(1.0, (math.sin(n * y) * math.sin(x + ((n - .5) * y))) / (2 * n * den)))
                
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)
        

cpdef cython.double make_nxy1cos(frequency=0, ratio=1., n=1):
    """Creates an nxy1co generator."""
    return Nxy1cos(frequency, ratio, n)
    

cpdef cython.double nxy1cos(Nxy1cos gen, cython.double fm =0.):
    return gen.call(fm)
    

cpdef cython.bint is_nxy1cos(gen):
    return isinstance(gen, Nxy1cos)  


# --------------- nxy1sin ---------------- #

cdef class Nxy1sin(mus_any_user):

    def __init__(self, frequency, ratio=1., n=1):
        self.frequency = clm.hz2radians(frequency)
        self.ratio = ratio
        self.n = n
        self.angle = 0.0
    

    cpdef cython.double call(self, cython.double fm =0.0):
        cdef cython.double x  = self.angle
        cdef cython.double y  = x * self.ratio
        cdef cython.double den  = math.cos(y * .5)
        cdef cython.double n = self.n
        self.angle += fm + self.frequency
        cdef cython.double res = (math.sin(x + (.5 * (n - 1) * (y + PI))) * math.sin(.5 * n * (y + PI))) / (n * den)
        return res
        
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)
        

cpdef Nxy1sin make_nxy1sin(frequency=0, ratio=1., n=1):
    """Creates an nxy1sin generator."""
    return Nxy1sin(frequency, ratio, n)


cpdef cython.double nxy1sin(Nxy1sin gen, cython.double fm =0.):
    return gen.call(fm)
    

cpdef cython.bint is_nxy1sin(gen):
    return isinstance(gen, Nxy1sin)        


# --------------- noddsin ---------------- #

def find_noddsin_max(n):    
    def find_mid_max(n=n, lo=0.0, hi= PI / ((2 * n) + 0.5)):
        mid = (lo + hi) / 2
        ylo = nodds(lo, n)
        yhi = nodds(hi, n)
        if math.fabs(ylo - yhi) < 1.0e-9:
            return nodds(mid,n)
        else:
            if ylo > yhi:
                return find_mid_max(n, lo, mid)   
            else:
                return find_mid_max(n, mid, hi)
    return find_mid_max(n)
    

@cython.cclass
cdef class Noddsin(mus_any_user):
    
    def __init__(self, frequency, ratio=1., n=1):
        noddsin_maxes = np.zeros(100, dtype=np.double) # temporary
        self.frequency = clm.hz2radians(frequency)
        self.ratio = ratio
        self.n = max(n, 1)
        self.norm = 1.
        if not (self.n < 100 and noddsin_maxes[self.n] > 0.0) :
            noddsin_maxes[self.n] = find_noddsin_max(self.n)
        else:
            self.norm = 1.0 / noddsin_maxes[self.n]
        self.angle = 0.0


    cpdef cython.double call(self, cython.double fm =0.0):
        cdef cython.double n = <cython.double>self.n
        cdef cython.double snx = math.sin(n * self.angle)
        cdef cython.double den = math.sin(self.angle)
        self.angle += fm + self.frequency
        if math.fabs(den) < NEARLY_ZERO:
            return 0.0
        else:
            return (self.norm * snx * snx) / den
                
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)
        

cpdef Noddsin make_noddsin(frequency=0, ratio=1., n=1):
    """Creates an noddsin generator."""
    return Noddsin(frequency, ratio, n)


cpdef cython.double noddsin(Noddsin gen, cython.double fm =0.):
    """returns n odd-numbered sines spaced by frequency."""
    return gen.call(fm)


cpdef cython.bint is_noddsin(gen):
    return isinstance(gen, Noddsin) 


# --------------- noddcos ---------------- #

cdef class Noddcos(mus_any_user):
    
    def __init__(self, frequency, ratio=1., n=1):
        self.frequency = clm.hz2radians(frequency)
        self.ratio = ratio
        self.n = n
        self.angle = 0.0


    cpdef cython.double call(self, cython.double fm =0.0):
        cdef cython.double cx = self.angle
        cdef cython.double n = self.n
        cdef cython.double den = 2 * self.n * math.sin(self.angle)
        cdef cython.double fang = 0.
        self.angle += fm + self.frequency
        
        if math.fabs(den) < NEARLY_ZERO:
            fang = cx % TWO_PI
            if (fang < .001) or (math.fabs(fang - (2 * PI)) < .001):
                return 1.0
            else: 
                return -1.0
        else:
            return math.sin(2 * n * cx) / den
                
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)


cpdef Noddcos make_noddcos(frequency=0, ratio=1., n=1):
    return Noddcos(frequency, ratio, n)

cpdef cython.double noddcos(Noddcos gen, cython.double fm =0.):
    return gen.call(fm)
   
cpdef cython.bint is_noddcos(gen):
    return isinstance(gen, Noddcos)        
    

# --------------- noddssb ---------------- #

cdef class Noddssb(mus_any_user):
    
    def __init__(self, frequency, ratio, n=1):
        self.frequency = clm.hz2radians(frequency)
        self.ratio = ratio
        self.n = n
        self.angle = 0.0

  
    cpdef cython.double call(self, cython.double fm =0.0):
        cdef cython.double cx = self.angle
        cdef cython.double mx = cx * self.ratio
        cdef cython.double x = cx - mx
        cdef cython.double n = self.n
        cdef cython.double sinnx = math.sin(n * mx)
        cdef cython.double den = n * math.sin(mx)
        self.angle += fm + self.frequency
        if math.fabs(den) <  NEARLY_ZERO:
            if (mx % TWO_PI) < .1: #TODO something is wrong here
                return -1
#             else: # removed seemed like it causing large spikes
#                 return 1
        return (math.sin(x) * ((sinnx*sinnx) / den)) - ((math.cos(x) * (math.sin(2 * n * mx) / (2 * den))))
                
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)
        

cpdef Noddssb make_noddssb(frequency=0, ratio=1., n=1):
    """Creates an noddssb generator."""
    return Noddssb(frequency, ratio, n)
    

cpdef cython.double noddssb(Noddssb gen, cython.double fm =0.):
    """Returns n sinusoids from frequency spaced by 2 * ratio * frequency."""
    return gen.call(fm)
    

cpdef cython.bint is_noddssb(gen):
    return isinstance(gen, Noddssb)   

# --------------- ncos2 ---------------- #


cdef class Ncos2(mus_any_user):
    
    def __init__(self, frequency, n=1):
        self.frequency = clm.hz2radians(frequency)
        self.n = n
        self.angle = 0.0
    

    cpdef cython.double call(self, cython.double fm =0.0):
        cdef cython.double x = self.angle
        cdef cython.double den = math.sin(.5 * x)
        cdef cython.double n = self.n
        self.angle += fm + self.frequency
        if math.fabs(den) < NEARLY_ZERO:
            return 1.0
        else:
            val: cython.double = math.sin(0.5 * (n + 1) * x) / ((n + 1) * den)
            return val * val
                
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)
        

cpdef Ncos2 make_ncos2(frequency=0, n=1):
    """Creates an ncos2 (Fejer kernel) generator"""
    return Ncos2(frequency, n)
    

cpdef cython.double ncos2(Ncos2 gen, cython.double fm =0.):
    """returns n cosines from frequency spaced by frequency * ratio."""
    return gen.call(fm)


cpdef cython.bint is_ncos2(gen):
    return isinstance(gen, Ncos2)        
       
# --------------- ncos4 ---------------- #

cdef class Ncos4(Ncos2):

    cpdef cython.double call(self, cython.double fm =0.0):
        cdef cython.double val = super(Ncos4, self).call(fm)
        return val * val
                

cpdef Ncos4 make_ncos4(frequency=0, n=1):
    """Creates an ncos4 (Jackson kernel) generator."""
    return Ncos4(frequency, n)
    

cpdef cython.double ncos4(Ncos4 gen, cython.double fm =0.):
    """returns n cosines from frequency spaced by frequency * ratio."""
    return gen.call(fm)


cpdef cython.bint is_ncos4(gen):
    return isinstance(gen, Ncos4)      
 
# --------------- npcos ---------------- #      

cdef class Npcos(mus_any_user):
    
    def __init__(self, frequency, n):
        self.frequency = clm.hz2radians(frequency)
        self.n = n
        self.angle = 0.0

    cpdef cython.double call(self, cython.double fm =0.0):
        cdef cython.double result  = 0.0
        cdef cython.double result1  = 0.0
        cdef cython.double result2  = 0.0
        cdef cython.double den  = math.sin(.5 * self.angle)
        cdef cython.double n = self.n
        if math.fabs(den) < NEARLY_ZERO:
            result = 1.0
        else:
            n1 = n + 1
            val = math.sin(.5 * n1 * self.angle) / (n1 * den)
            result1 = val * val
            p2n2 = (2 * n) + 2
            val = math.sin(.5 * p2n2 * self.angle) / (p2n2 * den)
            result2 = val * val
            result = (2 * result2) - result1
        self.angle += fm + self.frequency
        return result
                
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)
        

cpdef Npcos make_npcos(frequency=0, n=1):
    """Creates an npcos (Poussin kernel) generator."""
    return Npcos(frequency, n)


cpdef cython.double npcos(Npcos gen, cython.double fm =0.):
    """returns n*2+1 sinusoids spaced by frequency with amplitudes in a sort of tent shape."""
    return gen.call(fm)

cpdef cython.bint is_npcos(gen):
    return isinstance(gen, Npcos)        


# --------------- ncos5 ---------------- #  


cdef class Ncos5(mus_any_user):
    
    def __init__(self, frequency, n):
        self.frequency = clm.hz2radians(frequency)
        self.n = n
        self.angle = 0.0
    

    cpdef cython.double call(self, cython.double fm =0.0):
        x: cython.double = self.angle
        den: cython.double = math.tan(.5 * x)
        n: cython.double = self.n
        self.angle += fm + self.frequency
        if math.fabs(den) < NEARLY_ZERO:
            return 1.0
        else:
            return ((math.sin(n*x) / (2 * den)) - .5) / (n - .5)
               
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)
        

cpdef Ncos5 make_ncos5(frequency=0, n=1):
    """Creates an ncos5 generator"""
    return Ncos5(frequency, n)


cpdef cython.double ncos5(gen: Ncos5, cython.double fm =0.):
    """returns n cosines spaced by frequency. All are equal amplitude except the first and last at half amp"""
    return gen.call(fm)
  
cpdef cython.bint is_ncos5(gen):
    return isinstance(gen, Ncos5) 
    
                  
# --------------- nsin5 ---------------- #  

def find_nsin5_max(n):
    def ns(x, n):
            den = math.tan(.5 * x)
            if math.fabs(den) < NEARLY_ZERO:
                return 0.0
            else:
                return (1.0 - math.cos(n * x)) / den
                
    def find_mid_max(n, lo, hi):
        mid = (lo + hi) / 2
        ylo = ns(lo, n)
        yhi = ns(hi, n)
        if math.fabs(ylo - yhi) < 1e-9:
            return ns(mid,n)
        else:
            if ylo > yhi:
                return find_mid_max(n, lo, mid)   
            else:
                return find_mid_max(n, mid, hi)
                
    return find_mid_max(n, 0.0, math.pi / (n + .5))


cdef class Nsin5(mus_any_user):
    
    def __init__(self, frequency, n):
        self.frequency = clm.hz2radians(frequency)
        self.n = max(2, n)
        self.angle = 0.0
        self.norm = find_nsin5_max(self.n)
    

    cpdef cython.double call(self, cython.double fm =0.0):
        cdef cython.double x = self.angle
        cdef cython.double den = math.tan(.5 * x)
        cdef cython.double n = self.n
        self.angle += fm + self.frequency
        if math.fabs(den) < NEARLY_ZERO:
            return 0.0
        else:
            return (1.0 - math.cos(n*x)) / (den * self.norm)

                
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)
        

cpdef Nsin5 make_nsin5(frequency=0, n=1):
    """Creates an nsin5 generator."""
    return Nsin5(frequency, n)
    
cpdef cython.double nsin5(Nsin5 gen, cython.double fm =0.):
    return gen.call(fm)
    
cpdef cython.bint is_nsin5(gen):
    return isinstance(gen, Nsin5)  


# --------------- nrsin ---------------- #  
    
cpdef clm.mus_any make_nrsin(frequency=0, n=1, r=.5):
    return clm.make_nrxysin(frequency, 1., n, r)
    
cpdef cython.double nrsin(clm.mus_any gen, cython.double fm=0.):
    return clm.nrxysin(gen, fm)
    
cpdef cython.bint is_nrsin(gen):
    return clm.is_nrxysin(gen)   
# --------------- nrcos ---------------- #  

cdef class Nrcos(mus_any_user):
    
    def __init__(self, frequency, n=1, r=.5):
        self.frequency = clm.hz2radians(frequency)
        self.n = 1 + n
        self.r = clm.clamp(r, -.999999, .999999)
        self.angle = 0.0
        self.rr = self.r * self.r
        self.r1 = 1.0 + self.rr
        self.e1 = math.pow(self.r, self.n)
        self.e2 = math.pow(self.r, self.n + 1)
        self.norm = ((math.pow(math.fabs(self.r), self.n) - 1.0) / (math.fabs(self.r) - 1.)) - 1.0
        self.trouble = (self.n == 1) or (math.fabs(self.r) < NEARLY_ZERO)

    

    cpdef cython.double call(self, cython.double fm =0.0):
        cdef cython.double x = self.angle
        cdef cython.double rcos = self.r * math.cos(self.angle)
        self.angle += fm + self.frequency
        if self.trouble:
            return 0.0
        else: 
            return ((rcos + (self.e2 * math.cos((self.n - 1) * x))) - (self.e1 * math.cos(self.n * x)) - self.rr) / (self.norm * (self.r1 + (-2.0 * rcos)))
                
    def __call__(self, fm: cython.double=0.):
        return self.call(fm)
        
        
    @property
    def mus_order(self) -> cython.double:
        return self.n -1

    @mus_order.setter
    def mus_order(self, val: cython.double):
        self.n = 1 + val
        self.e1 = math.pow(self.r, self.n)
        self.e2 = math.pow(self.r, (self.n + 1))
        self.norm = ((math.pow(math.fabs(self.r), self.n) - 1.0) / (math.fabs(self.r) - 1.)) - 1.0
        self.trouble = (self.n == 1) or (math.abs(self.r) < NEARLY_ZERO)
 
    @property
    def mus_frequency(self) -> cython.double:
        return clm.radians2hz(self.frequency)
    
    @mus_frequency.setter
    def mus_frequency(self, val: cython.double):
        self.frequency = clm.hz2radians(val)
    
    @property
    def mus_scaler(self) -> cython.double:
        return self.r

    @mus_scaler.setter
    def mus_scaler(self, val: cython.double):
        self.r = min(.999999, max(-.999999, val))
        absr: cython.double = math.fabs(self.r)
        self.rr = self.r * self.r
        self.r1 = 1.0 + self.rr
        self.norm = ((math.pow(absr, self.n) - 1) / (absr - 1)) - 1.0
        self.trouble = ((self.n == 1) or (absr < NEARLY_ZERO))

        

cpdef Nrcos make_nrcos(frequency=0, n=1, r=.5):
    """Creates an nxycos generator."""
    return Nrcos(frequency, n, r)
    
cpdef cython.double nrcos(Nrcos gen, cython.double fm =0.):
    return gen.call(fm)
    
cpdef cython.bint is_nrcos(gen):
    return isinstance(gen, Nrcos)         
 
# --------------- nrssb ---------------- #

cdef class Nrssb(mus_any_user):

    def __init__(self, frequency, ratio=1.0, n=1, r=.5):
        self.frequency = clm.hz2radians(frequency)
        self.ratio = ratio
        self.n = n
        self.r = clm.clamp(r, -.999999, .999999)
        self.r = max(self.r, 0.0)
        self.angle = 0.0
        self.rn = -math.pow(self.r, self.n)
        self.rn1 = math.pow(self.r, (self.n + 1))
        self.norm = (self.rn - 1) / (self.r - 1)
    

    cpdef cython.double call(self, fm: cython.double =0.0):
        cdef cython.double cx = self.angle
        cdef cython.double mx = cx * self.ratio
        cdef cython.double n = self.n
        cdef cython.double r = self.r
        cdef cython.double rn1 = self.rn1
        cdef cython.double rn = self.rn
        cdef cython.double nmx = n * mx
        cdef cython.double n1mx = (n - 1) * mx
        cdef cython.double den = self.norm * (1.0 + (-2.0 * r * math.cos(mx)) + (r * r))
        self.angle += fm + self.frequency
        return (((math.sin(cx) * ((r * math.sin(mx)) + (rn * math.sin(nmx)) + (self.rn * math.sin(n1mx)))) -
                (math.cos(cx) * (1.0 + (-1. * r * math.cos(mx)) + (rn * math.cos(nmx)) + (rn1 * math.cos(n1mx)))))) / den

    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)
        

    cpdef call_interp(self, cython.double fm =0.0, cython.double interp=0.0):
        self.interp = interp 
        cdef cython.double cx = self.angle
        cdef cython.double mx = cx * self.ratio
        cdef cython.double n = self.n
        cdef cython.double r = self.r
        cdef cython.double rn1 = self.rn1
        cdef cython.double rn = self.rn
        cdef cython.double nmx = n * mx
        cdef cython.double n1mx = (n - 1) * mx
        cdef cython.double den = self.norm * (1.0 + (-2.0 * r * math.cos(mx)) + (r * r))
        self.angle += fm + self.frequency
    
        return (((self.interp * math.sin(cx) * ((r * math.sin(mx)) + (rn * math.sin(nmx)) + (rn1 * math.sin(n1mx)))) -
                (math.cos(cx) * (1.0 + (-1. * r * math.cos(mx)) + (rn * math.cos(nmx)) + (rn1 * math.cos(n1mx)))))) / den    
        
        
cpdef Nrssb make_nrssb(frequency=0, ratio=1.0, n=1, r=.5) :
    """Creates an nrssb generator."""
    return Nrssb(frequency, ratio, n, r)
    
cpdef cython.double nrssb(Nrssb gen, cython.double fm =0.):
    """returns n sinusoids from frequency spaced by frequency * ratio with amplitudes scaled by r^k."""
    return gen.call(fm)
    
cpdef cython.double nrssb_interp(Nrssb gen, cython.double fm=0.,cython.double interp =0. ):
    """returns n sinusoids from frequency spaced by frequency * ratio with amplitudes scaled by r^k."""
    return gen.call_interp(fm)
    
cpdef cython.bint is_nrssb(gen):
    return isinstance(gen, Nrssb)                   

# --------------- nkssb ---------------- #  

cdef class Nkssb(mus_any_user):

    
    def __init__(self, frequency, ratio=1.0, n=1):
        self.frequency = clm.hz2radians(frequency)
        self.ratio = ratio
        self.n = n + 1
        self.norm = 1.0 / (.5 * self.n * (self.n - 1.))
        self.angle = 0.0
        self.interp = 0.0

    cpdef cython.double call(self, cython.double fm=0.0):
        cdef cython.double x = self.angle * self.ratio
        cdef cython.double cxx = self.angle - x
        cdef cython.double sx2 = math.sin(.5 * x)
        cdef cython.double nx = self.n * x
        cdef cython.double nx2 = .5 * ((2*self.n) - 1) * x
        cdef cython.double sx22 = 2.0 * sx2
        cdef cython.double sxsx = 4 * sx2 * sx2
        cdef cython.double s1
        cdef cython.double c1
        self.angle += fm + self.frequency
        if math.fabs(sx2) < NEARLY_ZERO:
             return -1. # TODO original has -1 but that seems wrong
        
        s1 = (math.sin(nx) / sxsx) - ((self.n * math.cos(nx2)) / sx22)
        c1 = (math.sin(nx2) / sx22) - ((1.0 - math.cos(nx)) / sxsx)
        return ((s1 * math.sin(cxx)) - (c1 * math.cos(cxx))) * self.norm
    
    cpdef cython.double call_interp(self, cython.double fm=0.0, cython.double interp=0.0):
        cdef cython.double x = self.angle * self.ratio
        cdef cython.double cxx = self.angle - x
        cdef cython.double sx2 = math.sin(.5 * x)
        cdef cython.double nx = self.n * x
        cdef cython.double nx2 = .5 * ((2*self.n) - 1) * x
        cdef cython.double sx22 = 2.0 * sx2
        cdef cython.double sxsx = 4 * sx2 * sx2
        cdef cython.double s1
        cdef cython.double c1
        self.angle += fm + self.frequency
        if math.fabs(sx2) < NEARLY_ZERO:
            return 1. 
        else:
            s1 = (math.sin(nx) / sxsx) - ((self.n * math.cos(nx2)) / sx22)
            c1 = (math.sin(nx2) / sx22) - ((1.0 - math.cos(nx))/ sxsx)
            return ((c1 * math.cos(cxx)) - (interp * math.sin(cxx) * s1)) * self.norm
    
    def __call__(self, cython.double fm=0.) -> cython.double:
        return self.call(fm)
        
    @property
    def mus_order(self) -> cython.double:
        return self.n -1

    @mus_order.setter
    def mus_order(self, val: cython.double):
        self.n = 1 + val
        self.norm = 1.0 / (.5 * val * (val - 1.))

cpdef Nkssb make_nkssb(frequency=0, ratio=1.0, n=1):
    return Nkssb(frequency, ratio, n)
    
cpdef cython.double nkssb(Nkssb gen, cython.double fm=0.):
    return gen.call(fm)
    
cpdef cython.double nkssb_interp(Nkssb gen, cython.double fm=0., cython.double interp=0.0):
    return gen.call_interp(fm, interp)
    
cpdef cython.bint is_nkssb(gen):
    return isinstance(gen, Nkssb)    



# # TODO: --------------- nsincos ---------------- #  
# 
# 
# # TODO: --------------- n1cos ---------------- #  
# 
# 
# # TODO: --------------- npos1cos ---------------- #  
# 
# 
# # TODO: --------------- npos3cos ---------------- #  


# --------------- rcos ---------------- #

cdef class Rcos(mus_any_user):
    
    def __init__(self, frequency, r):
        self.osc = clm.make_oscil(frequency, .5 * PI)
        self.r = clm.clamp(r, -.999999, .999999)
        self.rr = self.r * self.r
        self.rrp1 = 1.0 + self.rr
        self.rrm1 = 1.0 - self.rr
        self.r2 = 2. * self.r
        cdef cython.double absr = math.fabs(self.r)
        self.norm = 0.0 if absr < NEARLY_ZERO else (1.0 - absr) /  ( 2.0 * absr)
    

    cpdef cython.double call(self, cython.double fm =0.0):
        return ((self.rrm1 / (self.rrp1 - (self.r2 * clm.oscil(self.osc, fm)))) - 1.0) * self.norm

                
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)
        
    @property
    def mus_frequency(self) -> cython.double:
        return clm.osc.mus_frequency
    
    @mus_frequency.setter
    def mus_frequency(self, val: cython.double):
        clm.osc.mus_frequency = val
        
    @property
    def mus_scale(self) -> cython.double:
        return self.r
    
    @mus_scale.setter
    def mus_scale(self, val: cython.double):
        self.r = clm.clamp(val, -.999999, .999999)
        self.rr = self.r*self.r
        self.rrp1 = 1.0 + self.rr
        self.rrm1 = 1.0 - self.rr
        self.r2 = 2.0 * self.r
        absr: cython.double = math.fabs(self.r)
        self.norm = 0.0 if absr < NEARLY_ZERO else (1.0 - absr) /  ( 2.0 * absr)


cpdef Rcos make_rcos(frequency=0, r=1):
    return Rcos(frequency, r)
    
cpdef cython.double rcos(Rcos gen, cython.double fm =0.):
    """returns n cosines from frequency spaced by frequency * ratio."""
    return gen.call(fm)
    
cpdef cython.bint is_rcos(gen):
    return isinstance(gen, Rcos)    


# # TODO: --------------- rssb ---------------- #  
# 
# # TODO: --------------- rxysin ---------------- #  
# 
# 
# # TODO: --------------- rxycos ---------------- #  
# 
# 
# # TODO: --------------- safe-rxycos ---------------- #  
# 
# # TODO: --------------- ercos ---------------- #  
# 
# # TODO: --------------- erssb ---------------- #  
# 
# # TODO: --------------- r2sin ---------------- #  
# 
# # TODO: --------------- r2cos ---------------- #  
# 
# # TODO: --------------- r2ssb ---------------- #  
# 
# # TODO: --------------- eoddcos ---------------- #  
# 
# # TODO: --------------- rkcos ---------------- #  
# 
# # TODO: --------------- rksin ---------------- #  
# 
# # TODO: --------------- rkssb ---------------- #  
# 
# # TODO: --------------- rk!cos ---------------- #  
# 
# # TODO: --------------- rk!ssb ---------------- #  
# 
# # TODO: --------------- rxyk!sin ---------------- # 
# 
# # TODO: --------------- rxyk!cos ---------------- # 
# 
# # TODO: --------------- r2k!cos ---------------- # 
# 
# # TODO: --------------- k2sin ---------------- # 
# 
# # TODO: --------------- k2cos ---------------- # 
# 
# # TODO: --------------- k2ssb ---------------- # 
# 
# # TODO: --------------- dblsum ---------------- # 
# # # # 
# # # # 
# # # # 
# # --------------- rkoddssb ---------------- #

cdef class Rkoddssb(mus_any_user):
    
    def __init__(self, frequency, ratio=1., r=.5):
        self.frequency = clm.hz2radians(frequency)
        self.ratio = ratio
        self.r = clm.clamp(r, -.999999, .999999)
        self.rr1 = 1.0 + (self.r*self.r)
        self.norm = 1.0 / (math.log(1.0 + self.r) - math.log(1.0 - self.r))
        self.angle = 0.0
    

    cpdef cython.double call(self, cython.double fm =0.0):
        cx: cython.double = self.angle
        mx: cython.double = cx * self.ratio
        cxx: cython.double = cx - mx
        cmx: cython.double = 2.0 * self.r * math.cos(mx)
        self.angle += fm + self.frequency
        return (((math.cos(cxx) * .5 * math.log((self.rr1+cmx) / (self.rr1-cmx))) - (math.sin(cxx) * (math.atan2((2.0 * self.r * math.sin(mx)), (1.0 - (self.r*self.r)))))) * self.norm)

                
    def __call__(self, fm=0.) -> cython.double:
        return self.call(fm)
        
    @property
    def mus_scale(self) -> cython.bint:
        return self.r
    
    @mus_scale.setter
    def mus_scale(self, val: cython.double):
        self.r = clm.clamp(val, -.999999, .9999999)
        self.rr1 = 1.0 + (self.r * self.r)
        self.norm = 1.0 / (math.log(1.0 + self.r) - math.log(1.0 - self.r))

cpdef Rkoddssb make_rkoddssb(frequency=0, ratio=1., r=.5):
    """Creates an nxycos generator."""
    return Rkoddssb(frequency, ratio, r)
    
cpdef cython.double rkoddssb(Rkoddssb gen, cython.double fm =0.):
    """returns n cosines from frequency spaced by frequency * ratio."""
    return gen.call(fm)
    
cpdef cython.bint is_rkoddssb(gen):
    return isinstance(gen, Rkoddssb)           
    
# # # # 
# # TODO: --------------- krksin ---------------- # 
# 
# 
# # TODO: --------------- abcos ---------------- # 
# 
# # TODO: --------------- absin ---------------- # 
# 
# # TODO: --------------- r2k2cos ---------------- # 
# 
# # TODO: --------------- blsaw ---------------- # 
# 
# # TODO: --------------- asyfm ---------------- # 
# 
# # TODO: --------------- bess ---------------- # 
# 
# # TODO: --------------- jjcos ---------------- # 
# 
# # TODO: --------------- j0evencos ---------------- # 
# 
# # TODO: --------------- j2cos ---------------- # 
# 
# # TODO: --------------- jpcos ---------------- # 
# 
# # TODO: --------------- jncos ---------------- # 
# 
# # TODO: --------------- j0j1cos  ---------------- # 
# 
# # TODO: --------------- jycos  ---------------- # 
# 
# # TODO: --------------- jcos  ---------------- # 
# 
# # TODO: --------------- blackman  ---------------- # 
# # # # 

# --------------- fmssb ---------------- #

@cython.cclass
cdef class Fmssb(mus_any_user):

    def __init__(self, frequency, ratio=1.0, index=1.):
        self.frequency = clm.hz2radians(frequency)# added because this did not seem righ. was generating octave higher
        self.ratio = ratio
        self.idx = index
    

    cpdef cython.double call(self, fm: cython.double =0.0):
        cdef cython.double cx = self.angle
        cdef cython.double mx = cx * self.ratio
        self.angle += fm + self.frequency
        return (math.cos(cx)*math.sin(self.idx * math.cos(mx))) - (math.sin(cx)*math.sin(self.idx * math.sin(mx)))
                
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)
       
    @property
    def index(self) -> cython.double:
        return self.idx
    
    @index.setter
    def index(self, val: cython.double): 
        self.idx = val
        

cpdef Fmssb make_fmssb(frequency=0, ratio=1.0, index=1.):
    return Fmssb(frequency, ratio, index)
    
cpdef cython.double fmssb(Fmssb gen, cython.double fm =0.):
    return gen.call(fm)
    
cpdef cython.bint is_fmssb(gen):
    return isinstance(gen, Fmssb)           
         
# --------------- k3sin  ---------------- # 
cdef class K3sin(mus_any_user):

    def __init__(self, frequency):
        self.frequency = clm.hz2radians(frequency)
        self.coeffs = np.array([0.0, (PI*PI) / 6.0, PI / -4.0, .08333])
        self.angle = 0.0
 
    
    cpdef cython.double call(self, cython.double fm =0.0):
        cdef cython.double x = self.angle
        if not 0.0 <= x <= TWO_PI:
            x = x % TWO_PI*2
        self.angle = x + fm + self.frequency
        return clm.polynomial(self.coeffs, x)
                
    def __call__(self, fm: cython.double =0.) -> cython.double:
        return self.call(fm)
        
    def mus_reset(self):
        self.frequency = 0.
        self.angle = 0.
        

cpdef K3sin make_k3sin(frequency=0, n=1):
    """creates a k3sin generator."""
    return K3sin(frequency)
    
cpdef cython.double k3sin(K3sin gen, cython.double fm =0.):
    return gen.call(fm)
    
cpdef cython.bint is_k3sin(gen):
    return isinstance(gen, K3sin)       
    
# 
# # TODO: --------------- izcos  ---------------- # 
# 
# # TODO: --------------- adjustable-square-wave ---------------- # 
# 
# # TODO: --------------- adjustable-triangle-wave ---------------- # 
# 
# # TODO: --------------- adjustable-sawtooth-wave ---------------- # 
# 
# # TODO: --------------- adjustable-oscil-wave ---------------- # 
# 
# # TODO: --------------- make-table-lookup-with-env ---------------- # 
# 
# # TODO: --------------- make-wave-train-with-env ---------------- # 
# 
# # TODO: --------------- round-interp ---------------- # 
# 
# 
# # TODO: --------------- env-any functions ---------------- # 
# 
# # TODO: --------------- run-with-fm-and-pm ---------------- # 
# 
# # TODO: --------------- nchoosekcos ---------------- # 
# 
# # TODO: --------------- sinc-train ---------------- # 
# # # # 
# # --------------- pink-noise ---------------- # 
# 
cdef class PinkNoise(mus_any_user):

    def __init__(self, n):
        self.n = n
        self.data = np.zeros(self.n*2, dtype=np.double)
        amp: cython.double = 1.0 / (2.5 * math.sqrt(self.n))
        self.data[0] = amp
        for i in range(2,2*self.n,2):
            self.data[i] =  clm.random(amp)
            self.data[i+1] = random.random()
    
    cpdef cython.double call(self, fm: cython.double =0.0):
        cdef cython.double x = 0.5 
        cdef cython.double s = 0.0
        cdef cython.double amp = self.data[0]
        cdef cython.int size = self.n*2

        cdef cython.long i = 0

        for i in range(2,size,2):
            s += self.data[i]
            self.data[i+1] -= x
        
            if self.data[i+1] < 0.0:
                self.data[i] = clm.random(amp)
                self.data[i+1] += 1.0
            x *= .5
        return s + clm.random(amp)
                
    def __call__(self, fm: cython.double =0.) -> cython.double:
        return self.call(fm)

cpdef PinkNoise make_pink_noise(n=1):
    """Creates a pink-noise generator with n octaves of rand (12 is recommended)."""
    return PinkNoise( n)
    
cpdef cython.double pink_noise(PinkNoise gen, cython.double fm=0.):
    """Returns the next random value in the 1/f stream produced by gen."""
    return gen.call(fm)
    
cpdef cython.bint is_pink_noise(gen):
    return isinstance(gen, PinkNoise)           

# --------------- brown-noise ---------------- # 
cdef class BrownNoise(mus_any_user):
    
    def __init__(self, frequency, amplitude=1.0):
        self.gr = clm.make_rand(frequency, amplitude)
        self.prev = 0
        self.sum = 0
    
    cpdef cython.double call(self, fm: cython.double =0.0):
        val: cython.double = clm.rand(self.gr, fm)
        if val != self.prev:
            self.prev = val
            self.sum += val
        return self.sum
                
    def __call__(self, fm: cython.double=0.) ->  cython.double:
        return self.call(fm)
        

cpdef BrownNoise make_brown_noise(frequency, amplitude=1.0):
    """Returns a generator that produces brownian noise."""
    return BrownNoise(frequency, amplitude=1.0)
    
cpdef cython.double brown_noise(BrownNoise gen, cython.double fm=0.):
    """returns the next brownian noise sample"""
    return gen.call(fm)
    
cpdef cython.bint is_brown_noise(gen):
    return isinstance(gen, BrownNoise)           
# 
# --------------- green-noise ---------------- # 

cdef class GreenNoise(mus_any_user):
    
    def __init__(self, frequency, amplitude=1.0, low=-1, high=1):
        self.gr = clm.make_rand(frequency, amplitude)
        self.low = low
        self.high = high
        self.sum = .5 * (self.low + self.high)
        self.prev = 0.
        self.sum = 0.
    

    cpdef cython.double call(self, cython.double fm =0.0) :
        val: cython.double = clm.rand(self.gr, fm)
        if val != self.prev:
            self.prev = val
            self.sum += val
            if not (self.low <= self.sum <= self.high):
                self.sum -= 2*val
        return self.sum
                
    def __call__(self, fm: cython.double=0.) -> cython.double:
        return self.call(fm)
        
cpdef GreenNoise make_green_noise(frequency, amplitude=1.0, low=-1.0, high=1.0):
    """Returns a new green-noise (bounded brownian noise) generator."""
    return GreenNoise(frequency, amplitude=1.0)
    
cpdef cython.double  green_noise(GreenNoise gen, cython.double fm=0.):
    """returns the next sample in a sequence of bounded brownian noise samples."""
    return gen.call(fm)
    
cpdef cython.bint is_green_noise(gen):
    return isinstance(gen, GreenNoise)           

# --------------- green-noise-interp ---------------- # 
cdef class GreenNoiseInterp(mus_any_user):
    
    def __init__(self, frequency, amplitude=1.0, low=-1, high=1):
        self.low = low
        self.high = high
        self.amplitude = amplitude
        self.sum = .5 * (self.low + self.high)
        self.dv = 1.0 / (math.ceil(clm.get_srate() / max(1.0, self.frequency)))
        self.frequency = clm.hz2radians(frequency)
        self.incr = clm.random(self.amplitude) * self.dv
        self.sum = 0.
    

    cpdef cython.double call(self, cython.double fm = 0.0):
        cdef cython.double val 
        if not (0.0 <= self.angle <= TWO_PI):
            val = clm.random(self.amplitude)
            self.angle %= TWO_PI # in scheme version modulo used which should be same as %\
            if self.angle < 0.0:
                self.angle += TWO_PI
            if not (self.low <= (self.sum+val) <= self.high):
                val = min(self.high-self.sum, max(self.low-self.sum, -val))
            self.incr = self.dv * val
        print(self.angle, self.dv, self.incr, self.sum, self.low, self.high)
        self.angle += fm + self.frequency
        self.sum += self.incr
        return self.sum
                
    def __call__(self, fm: cython.double =0.) -> cython.double:
        return self.call(fm)
        

cpdef GreenNoiseInterp make_green_noise_interp(frequency, amplitude=1.0, low=-1.0, high=1.0):
    """Returns a new green-noise (bounded brownian noise) generator."""
    return GreenNoiseInterp(frequency, amplitude=1.0)
    
cpdef green_noise_interp(GreenNoiseInterp gen, cython.double fm=0.):
    """Returns the next sample in a sequence of interpolated bounded brownian noise samples."""
    return gen.call(fm)
    
cpdef cython.bint is_green_noise_interp(gen):
    return isinstance(gen, GreenNoiseInterp)          


# --------------- moving-sum ---------------- # 

cdef class MovingSum(mus_any_user):

    def __init__(self, n=128):
        self.n = n
        self.gen = clm.make_moving_average(self.n)
        
    cpdef cython.double call(self, cython.double insig=0.0):
        return clm.moving_average(self.gen, math.fabs(insig))

    def __call__(self, cython.double insig=0.) -> cython.double:
        return self.call(insig)
        

cpdef MovingSum make_moving_sum(n=128):
    return MovingSum(n)
    
cpdef cython.double moving_sum(MovingSum gen, cython.double insig=0.):
    return gen.call(insig)
    
cpdef cython.bint is_moving_sum(gen):
    return isinstance(gen, MovingSum)    
     
# # TODO: --------------- moving-variance ---------------- # 
# 
# # TODO: --------------- moving-rms ---------------- # 
# 
# # TODO: --------------- moving-length ---------------- # 
# 
# # TODO: --------------- weighted-moving-average ---------------- # 
# 
# # TODO: --------------- exponentially-weighted-moving-average ---------------- # 
# # # # 
# # TODO: --------------- polyoid ---------------- # 
# # # # 
# # # # def make_polyoid(frequency, partial_amps_and_phases):
# # # #     length = len(partial_amps_and_phases)
# # # #     n = 0
# # # #     for i in range(0, length, 3):
# # # #         n = max(n, math.floor(partial_amps_and_phases[i]))   
# # # #     topk = n + 1
# # # #     sin_amps = np.zeros(topk, dtype=np.double)
# # # #     cos_amps = np.zeros(topk, dtype=np.double)
# # # #     for j in range(0,length,3):
# # # #         n = math.floor((partial_amps_and_phases[j]))
# # # #         amp = partial_amps_and_phases[j+1]
# # # #         phase = partial_amps_and_phases[j+2]
# # # #         if n > 0:
# # # #             sin_amps[n] = amp * math.cos(phase)
# # # #         cos_amps[n] = amp * math.sin(phase)
# # # #     return make_polywave(frequency, xcoeffs=cos_amps, ycoeffs=sin_amps)
# # # # 
# # # # def is_polyoid(g):
# # # #     return is_polywave(g) and g.mus_channel == Polynomial.BOTH_KINDS
# # # #     
# # # # polyoid = polywave
# # # # 
# # # # #def polyoid_env
# # # # 
# # TODO: --------------- noid ---------------- # 
# # # # # def make_noid(frequency=0.0, n=1, phases=None, choice='all'):  
# # # # # the full version of this requires potentially loading a file so
# # # # # wondering good way to do this. using maybe binary numpy files
# # # #        
# # # #     
# # # #     
# # # #     
# # # # #         
# # # #         
# # # # 
# # # # 
# # TODO: --------------- knoid ---------------- # 
# 
# # TODO: --------------- roid ---------------- # 
# # # # 
# # TODO: --------------- tanhsin ---------------- # 
# # # # 
# # # # # is_waveshape = is_polyshape
# # # # 
# # # # 
# --------------- tanhsin ---------------- # 

cdef class Tanhsin(mus_any_user):
    
    def __init__(self, frequency, r=1.0,initial_phase=0.0):
        self.frequency = frequency
        self.r = r
        self.osc = clm.make_oscil(frequency, initial_phase)
        self.frequency = clm.hz2radians(frequency)

    cpdef cython.double call(self, cython.double fm =0.0):
        return math.tanh(self.r * clm.oscil(self.osc, fm))
   
    def __call__(self, cython.double fm=0.) -> cython.double:
        return self.call(fm)
        

cpdef Tanhsin make_tanhsin(frequency=0, r=1.):
    return Tanhsin(frequency, r)
    
cpdef cython.double tanhsin(Tanhsin gen, cython.double fm=0.):
    return gen.call(fm)
    
cpdef cython.bint is_tanhsin(gen):
    return isinstance(gen, Tanhsin)    

# # # # 
# # # # # --------------- moving-fft ---------------- # 
# # # # 
# # # # def moving_fft_wrapper(g):
# # # #     g.rl = np.zeros(g.n)
# # # #     g.im = np.zeros(g.n)
# # # #     g.data = np.zeros(g.n)
# # # #     g.window = make_fft_window(g.window, g.n)
# # # #     s = np.sum(g.window)
# # # #     g.window = g.window * (2./s)
# # # #     g.outctr = g.n+1
# # # #     return g
# # # #     
# # # # moving_fft_methods = {'mus_data' : [lambda g : g.data, None],
# # # #                         'mus_xcoeffs' : [lambda g : g.rl, None],
# # # #                         'mus_ycoeffs' : [lambda g : g.im, None],
# # # #                         'mus_run' : [lambda g,arg1,arg2 : moving_fft(g), None]}
# # # # 
# # # # make_moving_fft, is_moving_fft = make_generator('moving_fft', 
# # # #                 {'input' : False, 
# # # #                 'n' : 512, 
# # # #                 'hop' : 128, 
# # # #                 'window' : Window.HAMMING},
# # # #                 wrapper=moving_fft_wrapper, 
# # # #                 methods=moving_fft_methods,
# # # #                 docstring="""returns a moving-fft generator.""")
# # # # 
# # # # 
# # # # def moving_fft(gen):
# # # #     new_data = False
# # # #     if gen.outctr >= gen.hop:
# # # #         if gen.outctr > gen.n:
# # # #             for i in range(gen.n):
# # # #                 gen.data[i] = readin(gen.input)
# # # #         else:
# # # #             mid = gen.n - gen.hop
# # # #             gen.data = np.roll(gen.data, gen.hop)
# # # #             for i in range(mid,gen.n):
# # # #                 gen.data[i] = readin(gen.input)
# # # #         
# # # #         gen.outctr = 0
# # # #         new_data = True
# # # #         gen.im.fill(0.0)
# # # #         np.copyto(gen.rl, gen.data)        
# # # #         gen.rl *= gen.window
# # # #         mus_fft(gen.rl,gen.im, gen.n,1)
# # # #         gen.rl = rectangular2polar(gen.rl,gen.im)
# # # #     gen.outctr += 1
# # # #     return new_data
# # # # 
# # # # 
# # TODO --------------- moving-spectrum ---------------- # 
# # # TODO: I am not convinced this is working properly
# # 
# # def phasewrap(x):
# #     return x - 2.0 * np.pi * np.round(x / (2.0 * np.pi))
# # 
# # 
# # def moving_spectrum_wrapper(g):
# #     g.amps = np.zeros(g.n)
# #     g.phases = np.zeros(g.n)
# #     g.amp_incs = np.zeros(g.n)
# #     g.freqs = np.zeros(g.n)
# #     g.freq_incs = np.zeros(g.n)
# #     g.new_freq_incs = np.zeros(g.n)
# #     g.data = np.zeros(g.n)
# #     g.dataloc = 0
# #     g.window = make_fft_window(g.window, g.n)
# #     s = np.sum(g.window)
# #     g.window = g.window * (2./s)
# #     g.outctr = g.n+1
# #     return g
# #     
# # moving_spectrum_methods = {'mus_xcoeffs' : [lambda g : g.phases, None],
# #                         'mus_ycoeffs' : [lambda g : g.amps, None],
# #                         'mus_run' : [lambda g,arg1,arg2 : moving_spectrum(g), None]}
# # 
# # make_moving_spectrum, is_moving_spectrum = make_generator('moving_spectrum', 
# #                 {'input' : False, 
# #                 'n' : 512, 
# #                 'hop' : 128, 
# #                 'window' : Window.HAMMING},
# #                 wrapper=moving_spectrum_wrapper, 
# #                 methods=moving_spectrum_methods,
# #                 docstring="""returns a moving-spectrum generator.""")    
# # 
# # 
# # 
# # def moving_spectrum(gen):
# #     if gen.outctr >= gen.hop:
# #         # first time through fill data array with n samples
# #         if gen.outctr > gen.n:
# #             for i in range(gen.n):
# #                 gen.data[i] = readin(gen.input)
# #         #
# #         else:
# #             mid = gen.n - gen.hop
# #             gen.data = np.roll(gen.data, gen.hop)
# #             for i in range(mid,gen.n):
# #                 gen.data[i] = readin(gen.input)
# #         
# #         gen.outctr = 0
# #         gen.dataloc = gen.dataloc % gen.n
# # 
# #         gen.new_freq_incs.fill(0.0)
# #        
# #         data_start = 0
# #         data_end = gen.n - gen.dataloc
# #         
# # 
# #         gen.amp_incs[gen.dataloc:gen.n] = gen.window[data_start:data_end] * gen.data[data_start:data_end]
# # 
# #         if gen.dataloc > 0:
# #             data_start = gen.n - gen.dataloc
# #             data_end = data_start + gen.dataloc
# #             gen.amp_incs[0:gen.dataloc] = gen.window[data_start:data_end] * gen.data[data_start:data_end]  
# #         
# #         gen.dataloc += gen.hop
# #         
# #         mus_fft(gen.amp_incs, gen.new_freq_incs, gen.n, 1)
# # 
# #         gen.amp_incs = rectangular2polar(gen.amp_incs, gen.new_freq_incs)
# #                 
# #         scl = 1.0 / gen.hop
# #         kscl = (np.pi*2) / gen.n
# #         gen.amp_incs -= gen.amps
# # 
# #         gen.amp_incs *= scl
# # 
# #         
# #         
# #         n2 = gen.n // 2
# #         ks = 0.0
# #         for i in range(n2):
# #             diff = (gen.new_freq_incs[i] - gen.freq_incs[i]) #% (np.pi*2)
# #             gen.freq_incs[i] = gen.new_freq_incs[i]
# #            # diff = phasewrap(diff)
# #             if diff > np.pi:
# #                 diff = diff - (2*np.pi)
# #             if diff < -np.pi:
# #                 diff = diff + (2*np.pi)
# #             gen.new_freq_incs[i] = diff*scl + ks
# #             ks += kscl
# #         gen.new_freq_incs -= gen.freqs
# #         gen.new_freq_incs *= scl
# # 
# #         
# # 
# #     gen.outctr += 1
# #     
# #     gen.amps += gen.amp_incs
# #     gen.freqs += gen.new_freq_incs
# #     gen.phases += gen.freqs
# # 
# # # #  
# # # # 
# # TODO: --------------- moving-scentroid ---------------- # 
# 
# # TODO: --------------- moving-autocorrelation ---------------- # 
# 
# # TODO: --------------- moving-pitch ---------------- # 
# 
# # TODO: --------------- flocsig ---------------- # 
