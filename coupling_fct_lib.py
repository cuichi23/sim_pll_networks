#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import sys, gc
import numpy as np
import random
#cimport numpy as np
#cimport cython
import networkx as nx
from scipy.signal import sawtooth
from scipy.signal import square
from scipy.stats import cauchy

def neg_cosine(x):
	return -np.cos(x)

def cosine(x):
	return np.cos(x)

def neg_sine(x):
	return -np.sin(x)

def sine(x):
	return np.sin(x)

def triangular(x):
	return sawtooth(x,width=0.5)

def deriv_triangular(x):
	return (2/np.pi)*square(x,duty=0.5)	

def square_wave(x):
	return 0.5*(1.0+square(x,duty=0.5))

def pfd(x):
	return 0.5*(np.sign(x)*(1+sawtooth(1*x*np.sign(x), width=1)))
