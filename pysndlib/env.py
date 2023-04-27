#! /Users/toddingalls/Developer/Python/venvs/pysndlib-venv/bin/python

# from functools import 

#from pysndlib import *
from functools import singledispatch
import operator
import types

import numpy as np
import numpy.typing as npt

from pysndlib import make_generator

# def window_envelop(beg, end, e):

@singledispatch
def mapper(e1,e2,op):
    pass
    
@mapper.register   
def _(e1: list, e2, op):
    return list(map(op, e1, e2))

@mapper.register
def _(e1: np.ndarray, e2, op):
    return op(e1,e2)
    
    
@singledispatch
def interleave(a,b):
    pass
 
@interleave.register  
def _(a: list, b):
    return [x for y in zip(a, b) for x in y]
     
@interleave.register  
def _(a: np.ndarray, b):
    return np.vstack((a,b)).reshape((-1,),order='F')
    
  
@singledispatch
def scale_and_offset(e, scl, offset=0.):
    pass

@scale_and_offset.register
def _(e: list, scl, offset=0.):
    return [x * scl + offset for x in e]

@scale_and_offset.register
def _(e: np.ndarray, scl, offset=0.):
    return e * scl + offset
    


    


# --------------- map_envelopes ---------------- #

def map_envelopes(op, e1, e2):
    return mapper(e1,e2,op)

# --------------- multiply_envelopes ---------------- #

def multiply_envelopes(e1,e2):
    return map_envelopes(operator.mul, e1,e2)
    
# --------------- add_envelopes ---------------- #

def add_envelopes(e1,e2):
    return map_envelopes(operator.add, e1,e2)

# --------------- max-envelope ---------------- #

def max_envelope(e):
    return max(e[1::1])
  
# --------------- min-envelope ---------------- #  

def min_envelope(e):
    return min(e[1::1])
    
 
# --------------- integrate-envelope  ---------------- #    
# 
def integrate_envelope(e):
    sum = 0.0
    for i in range(0,len(e)-2,2):
        sum += (e[i+1] + e[i+3]) * .5 * (e[i+2] - e[i])
    return sum

# --------------- envelope_last_x  ---------------- #  

def envelope_last_x(e):
    return e[-2]


# ---------------  stretch_envelope   ---------------- #  


#TODO: need to make sure envelop is valid on return - that each x greater than previ
def stretch_envelope(fn, old_att, new_att, old_dec, new_dec):
    #test args
   # print(old_att, new_att, old_dec, new_dec)
    if not new_att and old_att:
        print("error") # TODO raise exception
        
    else:
        if not new_att:
            return fn
        else:
            if old_dec and not new_dec:
                print('error') # TODO raise exception
            
            else:
                new_x = x0 = fn[0]
                last_x = fn[-2]
                y0 = fn[1]
                new_fn = [x0,y0]
                scl = (new_att-x0) / max(.0001, old_att-x0)
                
                if old_dec and (old_dec == old_att):
                    old_dec = old_dec + (.000001 * last_x)
                
                for i in range(2,len(fn)-1, 2):
                    x1 = fn[i]
                    y1 = fn[i+1]
                    
                    if x0 < old_att and x1 >= old_att:
                        y0 = y1 if x1==old_att else y0 + (y1 - y0) * ((old_att - x0) / (x1 - x0))
                        
                        x0 = old_att

                        new_x = new_att
                        new_fn.extend((new_x,y0))   
                        
                        scl = (new_dec - new_att) / (old_dec - old_att) if old_dec else (last_x - new_att) / (last_x - old_att)
                
                if old_dec and x0 < old_dec and x1 >= old_dec:
                    y0 = y1 if x1 == old_dec else y0 + (y1 - y0) * ((old_dec - x0) / (x1 - x0))
                    x0 = old_dec
                    new_x = new_dec
                   
                    new_fn.extend((new_x,y0))
                    scl = (last_x - new_dec) / (last_x - old_dec)
                    #print(new_fn)
                if x0 != x1:
                    new_x = new_x + scl * (x1 - x0)
                    new_fn.extend((new_x,y1))
                    x0 = x1
                    y0 = y1
                #print(new_fn)
                return new_fn

# --------------- scale_envelope  ---------------- # 

def scale_envelope(e, scl, offset=0.0):
    ys = scale_and_offset(e[1::2], scl, offset)
    xs = e[0::2]
    return interleave(xs,ys)
    
# --------------- reverse_envelope  ---------------- # 

def reverse_envelope(e):
    xs = e[0::2]
    ys = e[:0:-2]
    return interleave(xs, ys)

# TODO: --------------- concatenate_envelope  ---------------- # 


# --------------- normalize_envelope  ---------------- # 

def normalize_envelope(e, new_max=1.0):
    maxy = max_envelope(e)
    scl = new_max / maxy
    return scale_envelope(e, scl)

# invert-envelope seens in mix.scm seeme like general good idea
#print(normalize_envelope([0.,1., .5, 2., .8, 1.7, 1.0, 0.0], .2))

# a = [0,1, .5, .8, 1.0, 0.]
# print(reverse_envelope(a))
  
#print(scale_envelope(np.array([1,2,3,4]), 2.5, 1.))
#print([1,2,3] + 2)
# print(interleave([1,2,3], [5,6,7]))   
# print(interleave(np.array([1,2,3]), np.array([5,6,7])) )
# print(integrate_envelope([0, 0, 1, 1]))
# print(integrate_envelope([0, 1, 1, 1]))
# print(integrate_envelope([0, 0, 1, 1, 2, .5]))
# print(max_envelope([1,2,3, 10, 2, 1, 9, 20]))
# print(max_envelope(np.array([1,2,3, 10, 2, 1, 9, 20])))



# print(mapper([1,2,3],[5,6,7], operator.add))
# print(mapper(np.array([1.00,2.00,3.00], dtype=np.double),np.array([5,6,7]), operator.add))
# print(np.array([6.00,8.00,10.00], dtype=np.double))
#def map_envelopes(op, e1, e2):

    


# print(list(map(operator.add, [1,2,3],[4,5,6])))
# 
# print(np.array(map(operator.add, np.array([1,2,3]), np.array([1,2,3]))))