#Linear Stability of Global Frequency (LiStaGoF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
from numpy import pi, sin, cos, arccos, arcsin
import numpy as np
import sympy
from sympy import solve, nroots, I
from sympy.abc import q
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.signal import sawtooth
from scipy.signal import square
import scipy.optimize as optimize
from scipy.optimize import fsolve
from scipy.optimize import root
# from mpmath import findroot
# from sympy import I, limit
# from sympy.solvers import solveset
# from sympy.solvers import solve
# from sympy import Symbol, Function, nsolve
# from sympy import sin, cos, exp
# import mpmath
# mpmath.mp.dps = 25

# conditions due to structure of the equations determine values of wR depending on value of K
def xrange(maxp, tauf):
	dp		= 0.025;
	return np.linspace(-tauf, maxp, (tauf+maxp)/dp)

def Krange():
	return (np.linspace(0.01, 3.0, 40));
# set initial guess
def initial_guess():
	return (1e8, 1e8);

# digital case
cfDig        = lambda x: sawtooth(x,width=0.5)
def cfDigInverse(x):
	if np.abs(x)>1:
		print('Error! Inverse fct. of triangular wave called with argument out of bounds.'); exit();
	# 	return -np.pi/2.0*x-np.pi/2.0;
	# else:
	return +np.pi/2.0*x+np.pi/2.0;
cfDigDeriv   = lambda x: (2.0/np.pi)*square(x,duty=0.5)
# analog case
cfAna        = lambda x: np.cos(x);
def cfAnaInverse(x):
	if np.abs(x)>1:
		print('Error! Inverse fct. of arccos wave called with argument out of bounds.'); exit();
	return -np.arccos(x);
cfAnaDeriv   = lambda x: -1.0*np.sin(x);

def globalFreq(w, K, tauf, v, digital, maxp):
	# K = Krange()

	p = xrange(maxp, tauf)
	# print(K)
	if digital:
		Omega = w + K * cfDig( -p )
		tau	  = ( Omega * tauf + v * p ) / Omega
	else:
		Omega = w + K * cfAna( -p )
		tau	  = ( Omega * tauf + v * p ) / Omega
		# print('sfsa', Omega, tau)
	return {'Omeg': Omega, 'tau': tau}

def linStabEq(l_vec, Omega, tau, tauf, K, tauc, v, zeta, digital):
	x = np.zeros(2)
	if digital:
		alpha = (K/v)*cfDigDeriv(-Omega*(tau-tauf)/v)
	else:
		alpha = (K/v)*cfAnaDeriv(-Omega*(tau-tauf)/v)

	l = l_vec[0] + 1j * l_vec[1]
	f = l*(1.0+l*tauc)*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)))
	#print(f)
	x[0] = np.real(f)
	x[1] = np.imag(f)
	#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x

def linStabEq_expansion(l_vec, Omega, tau, tauf, K, tauc, v, zeta, digital):
	x = np.zeros(2)
	if digital:
		alpha = (K/v)*cfDigDeriv(-Omega*(tau-tauf)/v)
	else:
		alpha = (K/v)*cfAnaDeriv(-Omega*(tau-tauf)/v)

	l = l_vec[0] + 1j * l_vec[1]
	f = ( (alpha - alpha*zeta)+(1.0+(tau-tauf)*zeta*alpha)*l+(tauc + tauf-0.5*(-tau+tauf)*(-tau+tauf)*zeta*alpha)*l**2+(tauc*tauf+0.5*tauf*tauf-(1.0/6.0)*(-tau+tauf)*(-tau+tauf)*(-tau+tauf)*zeta*alpha)*l**3 )
	#print(f)
	x[0] = np.real(f)
	x[1] = np.imag(f)
	#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x
#
def solveLinStab(Omega, tau, tauf, K, tauc, v, digital, maxp, expansion):
	krange = Krange();
	lambsolRe = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	lambsolIm = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	lambsolReKra = np.zeros([len(xrange(maxp, tauf)),len(krange)],dtype=np.float64)
	lambsolImKra = np.zeros([len(xrange(maxp, tauf)),len(krange)],dtype=np.float64)
	lambsolReKraMax = np.zeros(len(krange),dtype=np.float64)
	lambsolImKraMax = np.zeros(len(krange),dtype=np.float64)

	init   = initial_guess();
	zeta   = -1;
	c 	   = 3E8;
	OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];
	#print(type(init), init)
	#print(len(xrange(K)))

	if expansion:
		for index2 in range(len(krange)):
			for index in range(len(xrange(maxp, tauf))):
			# print(index, wR[index])
				temp =  optimize.root(linStabEq_expansion, init, args=( Omega[index], tau[index], tauf, K[index2], tauc, v, zeta, digital),  tol=1.364e-8, method='hybr')
			# print('temp =',temp)
			# print('temp =',temp)
				lambsolReKra[index2,index]  = temp.x[0]
				lambsolImKra[index2,index]  = temp.x[1]

			tempRe = lambsolReKra[np.round(lambsolReKra[:,index],12)!=0,index];
			tempIm = lambsolImKra[np.round(lambsolImKra[:,index],32)!=0,index];
		if len(tempRe) != 0:
			#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
			lambsolRebKra[index] = np.real(np.max(tempRe));
			lambsolImbeKra[index] = tempIm[tempRe[:].argmax()]
		else:
			lambsolReKra[index] = 0.0;
			lambsolImKra[index] = 0.0;
	# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
		init = initial_guess();
	else:

		for index2 in range(len(krange)):
			for index in range(len(xrange(maxp, tauf))):
				temp = optimize.root(linStabEq, init, args=( Omega[index], tau[index], tauf, K[index2], tauc, v, zeta, digital),  tol=1.364e-8, method='hybr')
				lambsolReKra[index2,index]  = temp.x[0]
				lambsolImKra[index2,index]  = temp.x[1]

			tempRe = lambsolReKra[np.round(lambsolReKra[:,index],12)!=0,index];
			tempIm = lambsolImKra[np.round(lambsolImKra[:,index],32)!=0,index];
			if len(tempRe) != 0:
				#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
				lambsolRebKraMax[index] = np.real(np.max(tempRe));
				lambsolImbeKraMax[index] = tempIm[tempRe[:].argmax()]
			else:
				lambsolReKraMax[index] = 0.0;
				lambsolImKraMax[index] = 0.0;
			init = initial_guess();
			print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])

			# if lambsolRe[index] >= 0:
			# 	OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
			# else:
			# 	OmegStab.append(Omega[index]); tauStab.append(tau[index]);

		print('\n\nlen(lambsolImKra):', len(lambsolReKra), '     len(lambsolImKra):', len(lambsolImKra),'\n')
	# return {'Re': lambsolRe, 'Im': lambsolIm, 'OmegUnst': OmegUnst, 'tauUnst': tauUnst, 'OmegStab': OmegStab, 'tauStab': tauStab}
	return {'ReKra': lambsolReKra, 'ImKra': lambsolImKra,'ReKraMax': lambsolReKraMax,'ImKraMax':lambsolImKraMax}

	#
