# cython: c_string_type=unicode
# cython: c_string_encoding=utf8

cimport cython
import numpy as np
cimport numpy as np
#cimport pysndlib.cclm as cclm
cimport pysndlib.clm as clm
from pysndlib.sndlib cimport Sample, Header

np.import_array()

cdef class mus_any_user:
    cpdef cython.double call(self, cython.double  fm=*)

# --------------- nssb ---------------- #
cdef class Nssb(mus_any_user):
    cdef cython.double frequency
    cdef cython.double  ratio
    cdef cython.int n
    cdef cython.double angle

    
cpdef cython.double nssb (Nssb gen, cython.double fm=*)
cpdef cython.bint is_nssb(gen)
    
# --------------- nxysin ---------------- #
cdef class Nxysin(mus_any_user):
    cdef cython.double frequency
    cdef cython.double  ratio
    cdef cython.int n
    cdef cython.double angle
    cdef cython.double norm

cpdef cython.double nxysin(Nxysin gen, cython.double fm=*)
cpdef cython.bint is_nxysin(gen)

# --------------- nxycos ---------------- #
cdef class Nxycos(mus_any_user):
    cdef cython.double frequency 
    cdef cython.double ratio 
    cdef cython.int n 
    cdef cython.double angle 

cpdef cython.double nxycos(Nxycos gen, cython.double fm =*)
cpdef cython.bint is_nxycos(gen)
   
# --------------- nxy1cos ---------------- #
cdef class Nxy1cos(mus_any_user):
    cdef cython.double frequency
    cdef cython.double ratio
    cdef cython.int n
    cdef cython.double angle 

cpdef cython.double nxy1cos(Nxy1cos gen, cython.double fm =*)
cpdef cython.bint is_nxy1cos(gen)

# --------------- nxy1sin ---------------- #
cdef class Nxy1sin(mus_any_user):
    cdef cython.double frequency
    cdef cython.double ratio
    cdef cython.int n
    cdef cython.double angle

cpdef cython.double nxy1sin(Nxy1sin gen, cython.double fm =*)
cpdef cython.bint is_nxy1sin(gen)

# --------------- Noddcos ---------------- #
cdef class Noddcos(mus_any_user):
    cdef cython.double frequency 
    cdef cython.double ratio 
    cdef cython.int n 
    cdef cython.double angle 


cpdef cython.double noddcos(Noddcos gen, cython.double fm=*)
cpdef cython.bint is_noddcos(gen)

# --------------- noddssb ---------------- #
cdef class Noddssb(mus_any_user):
    cdef cython.double frequency 
    cdef cython.double ratio 
    cdef cython.double n
    cdef cython.double angle 

cpdef cython.double noddssb(Noddssb gen, cython.double fm=*)
cpdef cython.bint is_noddssb(gen)

# --------------- ncos2 ---------------- #
cdef class Ncos2(mus_any_user):
    cdef cython.double frequency 
    cdef cython.int n
    cdef cython.double angle 
    
cpdef cython.double ncos2(Ncos2 gen, cython.double fm=*)
cpdef cython.bint is_ncos2(gen)

# --------------- ncos4 ---------------- #
cdef class Ncos4(Ncos2):
    pass

cpdef cython.double ncos4(Ncos4 gen, cython.double fm=*)
cpdef cython.bint is_ncos4(gen)

# --------------- npcos ---------------- #      
cdef class Npcos(mus_any_user):
    cdef cython.double frequency 
    cdef cython.double n
    cdef cython.double angle 

cpdef cython.double npcos(Npcos gen, cython.double fm=*)
cpdef cython.bint is_npcos(gen)

# --------------- ncos5 ---------------- #  
cdef class Ncos5(mus_any_user):
    cdef cython.double frequency 
    cdef cython.double n
    cdef cython.double angle 

cpdef cython.double ncos5(Ncos5 gen, cython.double fm=*)
cpdef cython.bint is_ncos5(gen)


# --------------- nsin5 ---------------- #  

cdef class Nsin5(mus_any_user):
    cdef cython.double frequency 
    cdef cython.double n
    cdef cython.double angle 
    cdef cython.double norm 

cpdef cython.double nsin5(Nsin5 gen, cython.double fm=*)
cpdef cython.bint is_nsin5(gen)

# --------------- nrsin ---------------- #  
    
cpdef cython.double nrsin(clm.mus_any gen, cython.double fm=*)
cpdef cython.bint is_nrsin(gen)

# 
# --------------- nrcos ---------------- #  
cdef class Nrcos(mus_any_user):
    cdef cython.double frequency 
    cdef cython.double n
    cdef cython.double r
    cdef cython.double angle 
    cdef cython.double rr
    cdef cython.double r1 
    cdef cython.double e1 
    cdef cython.double e2 
    cdef cython.double norm 
    cdef cython.bint trouble

cpdef cython.double nrcos(Nrcos gen, cython.double fm=*)
cpdef cython.bint is_nrcos(gen)

# --------------- nrssb ---------------- #
cdef class Nrssb(mus_any_user):
    cdef cython.double frequency
    cdef cython.double ratio 
    cdef cython.double n
    cdef cython.double r 
    cdef cython.double angle
    cdef cython.double interp
    cdef cython.double rn
    cdef cython.double rn1
    cdef cython.double norm
    cpdef call_interp(self, cython.double fm =*, cython.double interp=*)

cpdef cython.double nrssb(Nrssb gen, cython.double fm=*)
cpdef cython.double nrssb_interp(Nrssb gen, cython.double fm=*,cython.double interp =* )
cpdef cython.bint is_nrssb(gen)

# --------------- nkssb ---------------- #  
cdef class Nkssb(mus_any_user):
    cdef cython.double frequency 
    cdef cython.double ratio
    cdef cython.double n
    cdef cython.double angle
    cdef cython.double norm
    cdef cython.double interp
    cpdef cython.double call_interp(self, cython.double fm=*, cython.double interp=*)
    
cpdef cython.double nkssb(Nkssb gen, cython.double fm=*)
cpdef cython.double nkssb_interp(Nkssb gen, cython.double fm=*, cython.double interp=*)
cpdef cython.bint is_nkssb(gen)

# TODO: --------------- nsincos ---------------- #  


# TODO: --------------- n1cos ---------------- #  


# TODO: --------------- npos1cos ---------------- #  


# TODO: --------------- npos3cos ---------------- #  

# --------------- rcos ---------------- #

cdef class Rcos(mus_any_user):
    cdef clm.mus_any osc
    cdef cython.double r
    cdef cython.double rr
    cdef cython.double rrp1
    cdef cython.double rrm1
    cdef cython.double r2
    cdef cython.double norm 

cpdef cython.double rcos(Rcos gen, cython.double fm=*)
cpdef cython.bint is_rcos(gen)


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
cdef class Rkoddssb(mus_any_user):
    cdef cython.double frequency 
    cdef cython.double ratio 
    cdef cython.double r
    cdef cython.double rr1
    cdef cython.double angle 
    cdef cython.double norm 
    
cpdef cython.double rkoddssb(Rkoddssb gen, cython.double fm=*)
cpdef cython.bint is_rkoddssb(gen)

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

cdef class Fmssb(mus_any_user):
    cdef cython.double frequency 
    cdef cython.double ratio
    cdef cython.double idx
    cdef cython.double angle 

cpdef cython.double fmssb(Fmssb gen, cython.double fm=*)
cpdef cython.bint is_fmssb(gen)

# --------------- k3sin  ---------------- # 

cdef class K3sin(mus_any_user):
    cdef cython.double frequency 
    cdef np.ndarray coeffs
    cdef cython.double angle 
    
cpdef cython.double k3sin(K3sin gen, cython.double fm=*)
cpdef cython.bint is_k3sin(gen)
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
cdef class PinkNoise(mus_any_user):
    cdef cython.int n
    cdef np.ndarray data 

cpdef cython.double pink_noise(PinkNoise gen, cython.double fm=*)
cpdef cython.bint is_pink_noise(gen)

# --------------- brown-noise ---------------- # 
cdef class BrownNoise(mus_any_user):
    cdef cython.double frequency
    cdef cython.double amplitude
    cdef cython.double prev
    cdef cython.double sum
    cdef clm.mus_any gr
    
cpdef cython.double brown_noise(BrownNoise gen, cython.double fm=*)
cpdef cython.bint is_brown_noise(gen)


# --------------- green-noise ---------------- # 
cdef class GreenNoise(mus_any_user):
    cdef cython.double frequency
    cdef cython.double amplitude
    cdef cython.double prev
    cdef cython.double sum
    cdef clm.mus_any gr
    cdef cython.double low
    cdef cython.double high
    
# --------------- green-noise-interp ---------------- # 
cdef class GreenNoiseInterp(mus_any_user):
    cdef cython.double frequency
    cdef cython.double amplitude
    cdef cython.double sum
    cdef cython.double low
    cdef cython.double high
    cdef cython.double dv
    cdef cython.double incr
    cdef cython.double angle


# --------------- tanhsin ---------------- # 
cdef class Tanhsin(mus_any_user):
    cdef cython.double frequency 
    cdef cython.double r
    cdef clm.mus_any osc 

# --------------- moving-sum ---------------- # 

cdef class MovingSum(mus_any_user):
    cdef cython.int n
    cdef clm.mus_any gen
