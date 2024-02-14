#!/usr/bin/python
from numpy import pi, sin
import numpy as np
from sympy import *
# from sympy import solve, nroots, I
# from sympy import simplify, Symbol, pprint, collect_const
from sympy.abc import q
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.signal import sawtooth
from scipy.signal import square
import scipy.optimize as optimize
from scipy.optimize import fsolve
from scipy.optimize import root
import cmath
from scipy import signal


mu = np.arange(-10,10,0.01);
plot(mu, np.cos(mu)/np.sin(mu), 'b');
plot(mu, 1/mu+mu, 'r');
plt.ylim([-10, 10]);
