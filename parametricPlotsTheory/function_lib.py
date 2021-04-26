#Linear Stability of Global Frequency (LiStaGoF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)

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

def initial_guess(discr=75):
	return [np.linspace( -1e-3, 1e-3, discr), np.linspace(-0.28*2.0*np.pi, 0.20*2.0*np.pi, discr)];

def linStabEq(l_vec, params):

	x = np.zeros(2)

	l = l_vec[0] + 1J * l_vec[1]
	f = l*(1.0+l*(1.0/params['wc'])) + params['a']*(1.0-params['zeta']*np.exp(-l*params['tau']))

	x[0] = np.real(f)
	x[1] = np.imag(f)

	return x

def solveLinStab(params):

	init = initial_guess();

	tempRe = []; tempIm = [];

	for i in range(len(init[0])):

		temp =  optimize.root(linStabEq, (init[0][i],init[1][i]), args=(params), tol=1e-14, method='hybr')

		if ( temp.success == True and np.round(temp.x[0], 16) != 0.0 and np.round(temp.x[1], 16) != 0.0 ):
			tempRe.append(temp.x[0])
			tempIm.append(temp.x[1])

	if len(tempRe) != 0:
		lambsolReMax = np.real(np.max(tempRe));
		if np.array(tempRe).argmax() < len(tempIm):
			lambsolImMax = tempIm[np.array(tempRe).argmax()]
	else:
		lambsolReMax = 0.0;
		lambsolImMax = 0.0;

	solArray = np.array([lambsolReMax, np.abs(lambsolImMax)])

	return solArray
