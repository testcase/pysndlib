#! /Users/toddingalls/Developer/Python/venvs/pysndlib-venv/bin/python

import math
from random import uniform
from musx import *
from pysndlib import *
from pysndlib.v import fm_violin
from pysndlib.clm_ins import nrev
from pysndlib.musx import *



# can use clm_instrument to as decorator to make an instrument. but
# then it will not work correctly with just plain with Sound.
# this examples shows how to use the decorator function to wrap 
# an already defined function and use it with musx


# 
fm_violin = clm_instrument(fm_violin)

seq = Seq()
sco = Score(out=seq)

def p(sco, num):
    a = jumble([48,53,55,57])
    b = choose([0,7,17], weights=[.4,.4,.2])
    for i in range(num):
        k = fm_violin(sco.now, 3, hertz(next(a)+next(b)), .1, uniform(4.0, 5.0), distance=uniform(2.0, 4.0), reverb_amount=.1, degree=random.uniform(20, 70))
        sco.add(k)
        yield .65
    yield 1.
    sco.add(fm_violin(sco.now, 3, hertz(53), .1, 7., distance=uniform(2.0, 4.0), reverb_amount=.1, degree=random.uniform(20, 70)))
    sco.add(fm_violin(sco.now, 3, hertz(60), .1, 7., distance=uniform(2.0, 4.0), reverb_amount=.1, degree=random.uniform(20, 70)))
    
    yield 2.
    sco.add(fm_violin(sco.now, 5, hertz(48), .1, 6.2, distance=uniform(2.0, 4.0), reverb_amount=.1, degree=random.uniform(20, 70),  amp_env = [ 0, 0,  25, .5,  75, 1,  100, 0],))
    sco.add(fm_violin(sco.now, 5, hertz(64), .1, 5.4, distance=uniform(2.0, 4.0), reverb_amount=.1, degree=random.uniform(20, 70),  amp_env = [ 0, 0,  25, .5,  75, 1,  100, 0],))
    
 
  
sco.compose(p(sco, 30))

render_clm(seq, "out1.aiff", play=True, reverb=nrev, channels=2, statistics=True)


