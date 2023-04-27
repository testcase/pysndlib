#! /Users/toddingalls/Developer/Python/venvs/pysndlib-venv/bin/python


from pysndlib import *
from pysndlib.generators import *
import math
import functools
from musx import *

x = None

if x == None:
    print('ff')

#print(sndinfo('Bassoon sus forte  C3.aif'))
#assert False, f'no'
#print(111)
#print(sndinfo('out1.aif'))
#raise ValueError(f"not a rhythmic expression: {expr}.")
#  TypeError
# RuntimeError
# ModuleNotFoundError
# IndexError
# SyntaxError
# NotImplementedError
# assert 
# with Sound("out1.aiff", play=True):
#     osc = make_oscil(440.0)
#     
#     for i in range(88200):
#         outa(i, 0.5 * oscil(osc))



#     
#     
# def bar(func):
#     def wrapper(*args):
#         func(*args)
#         print(func.__name__)
#     return wrapper
#     
#     
#     
# k = bar(foo)
# k(10,9,8)
# 
# 

# 
# #    
# # def clm_instrument(func):
# #     @functools.wraps(func)
# #     def call(*args, **kwargs):
# #         return PEvent(functools.partial(func, *args, **kwargs))
# #         
# #     return call
#     
#    
# def clm_instrument(func):
#     @functools.wraps(func)
#     def call(*args, **kwargs):
#         return PEvent(functools.partial(func, *args, **kwargs))
#     return call    
# 



# def clm_wrapper(func)

# # 
# def clm1(func):
#     @functools.wraps(func)
#     def call(*args, **kwargs):
#         return functools.partial(func, *args, **kwargs)
#     return call
    
# 
# def clm1(func):
#     @functools.wraps(func)
#     def call(*args, **kwargs):
#         def wrapper():
#              func(*args, **kwargs)
#         return wrapper
#     return call
#     
#     
# a = 10
# b = 11
# c = 12
# 
# k = clm1(foo)
# 
# j = k(a,b,c)
# 
# c = 10
# 
# j()
# 
# def foo(t,a, b, c):
#     print(a,b,c)
#     
    
    
# a = partial(foo, 1, b=2)
# print(a.func.__name__, a.args, a.keywords )
# class CLMWrapper(Event):
#     def __init__(self, time, *args, **kwargs):
#         super().__init__(time)
#         self.func = func
#         self.args  = None
#         self.kwargs = None
#     def __call__(self, *args, **kwargs):
#         print(self, args)
#         p = functools.partial(self.func,self.time, *args, **kwargs)
#         self.args = args
#         self.kwargs = kwargs
# #        print(*args)
#         return p
#         
#     def __str__(self):
#         return f"<{self.name} {self.time}"#" {self.args}, {self.kwargs}>"
#         

# def clm_instrument(func):
#     @functools.wraps(func)
#     def call(time, *args, **kwargs):
#         name = func.__name__
#         obj = CLMWrapper(time, func)
#         obj.name = name
#         return obj
#     return call    \














