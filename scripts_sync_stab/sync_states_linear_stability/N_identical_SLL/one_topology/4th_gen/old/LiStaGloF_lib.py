#Linear Stability of Global Frequency (LiStaGoF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
from numpy import pi, sin
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

# set initial guess
def initial_guess():
	return (1e9, 1e9);
def K(Kvco, AkPD, Ga1):
	return (Kvco*AkPD*Ga1/2.0)

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

def globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp):
	# K = Krange()
	p = xrange(maxp, tauf)
	# print( K(Kvco, AkPD, Ga1))
	if digital:
		Omega = w + K(Kvco, AkPD, Ga1) * cfDig( -p )
		tau	  = ( Omega * tauf + v * p ) / Omega
	else:
		Omega = w + K(Kvco, AkPD, Ga1) * cfAna( -p )
		tau	  = ( Omega * tauf + v * p ) / Omega
		# print('tau=', tau)

	return {'Omeg': Omega, 'tau': tau}



# def linStabEq(l_vec, Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, digital):
# 	x = np.zeros(2)
# 	if digital:
# 		alpha = ( K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v)
# 	else:
# 		alpha = ( K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v)
#
# 	l = l_vec[0] + 1j * l_vec[1]
# 	f = l*(1.0+l*tauc)*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)))
# 	#print(f)
# 	x[0] = np.real(f)
# 	x[1] = np.imag(f)
# 	#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
# 	return x

def linStabEq(l_vec, Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital):
	x = np.zeros(2)
	if digital:
		alpha = ( K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v)
	else:
		alpha = ( K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v)

	l = l_vec[0] + 1j * l_vec[1]
	f = l*((1.0+l*tauc/order)**order)*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)))
	#print(f)
	x[0] = np.real(f)
	x[1] = np.imag(f)
	#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x

def linStabEq_expansion(l_vec, Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta,order, digital):
	x = np.zeros(2)
	if digital:
		alpha = ( K(Kvco, AkPD, Ga1)/v )*cfDigDeriv(-Omega*(tau-tauf)/v)
	else:
		alpha = ( K(Kvco, AkPD, Ga1)/v )*cfAnaDeriv(-Omega*(tau-tauf)/v)

	l = l_vec[0] + 1j * l_vec[1]

	f = alpha*(1.0 - zeta) + (1.0 - alpha*(-tau + tauf)*zeta)*l + (tauc +tauf - 0.5*alpha*zeta*(-tau + tauf)**2)*l**2 + ((0.5*(-1.0 + order)*tauc**2)/order + tauc*tauf + 0.5*tauf**2-(1.0/6.0)*alpha*zeta*(-tau + tauf)**3)*l**3

	x[0] = np.real(f)
	x[1] = np.imag(f)
	#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x
#
# lambsolReExpan = np.zeros(len(globalFreq(w, K, tauf, v, digital, maxp)['tau']),dtype=np.complex)
# lambsolImExpan = np.zeros(len(globalFreq(w, K, tauf, v, digital, maxp)['tau']),dtype=np.complex)

def solveLinStab(Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, expansion):
	lambsolRe = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	lambsolIm = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	init = initial_guess();
	zeta = -1;
	c 	 = 3E8;
	OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];
	#print(type(init), init)
	#print(len(xrange(K)))

	if expansion:
		for index in range(len(xrange(maxp, tauf))):

			# print(index, wR[index])
			temp =  optimize.root(linStabEq_expansion, init, args=(Omega[index], tau[index], tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order,digital), tol=1e-14, method='hybr')
		# print('temp =',temp)
		# print('temp =',temp)
			lambsolRe[index] = temp.x[0]
			lambsolIm[index] = abs(temp.x[1])
		# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
			# init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))
			init = initial_guess()
		#print(type(init), init)
		#print('Re[lambda] =',lambsol)
			if lambsolRe[index] >= 0:
				OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
			else:
				OmegStab.append(Omega[index]); tauStab.append(tau[index]);
	else:
		for index in range(len(xrange(maxp, tauf))):

			temp = optimize.root(linStabEq, init, args=(Omega[index], tau[index], tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital), tol=1e-14, method='hybr')
		# print('temp =',temp)
			lambsolRe[index] = temp.x[0]
			lambsolIm[index] = temp.x[1]
		# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
			# init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))
			init = initial_guess()
			if lambsolRe[index] >= 0:
				OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
			else:
				OmegStab.append(Omega[index]); tauStab.append(tau[index]);

	# print('\n\nlen(OmegStab):', len(OmegStab), '     len(OmegUnst):', len(OmegUnst),'\n')
	return {'Re': lambsolRe, 'Im': np.abs(lambsolIm), 'OmegUnst': OmegUnst, 'tauUnst': tauUnst, 'OmegStab': OmegStab, 'tauStab': tauStab}

# def solveLinStab(Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion):
# 	lambsolRe = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
# 	lambsolIm = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
# 	init = initial_guess();
# 	zeta = -1;
# 	c 	 = 3E8;
# 	OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];
# 	#print(type(init), init)
# 	#print(len(xrange(K)))
#
# 	if expansion:
# 		for index in range(len(xrange(maxp, tauf))):
#
# 			# print(index, wR[index])
# 			temp =  optimize.root(linStabEq_expansion, init, args=(Omega[index], tau[index], tauf, Kvco, AkPD, Ga1, tauc, v, zeta, digital), tol=1e-14, method='hybr')
# 		# print('temp =',temp)
# 		# print('temp =',temp)
# 			lambsolRe[index] = temp.x[0]
# 			lambsolIm[index] = temp.x[1]
# 		# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
# 			init = initial_guess();
# 		#print(type(init), init)
# 		#print('Re[lambda] =',lambsol)
# 			if lambsolRe[index] >= 0:
# 				OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
# 			else:
# 				OmegStab.append(Omega[index]); tauStab.append(tau[index]);
# 	else:
# 		for index in range(len(xrange(maxp, tauf))):
#
# 			temp = optimize.root(linStabEq, init, args=(Omega[index], tau[index], tauf, Kvco, AkPD, Ga1, tauc, v, zeta, digital), tol=1e-14, method='hybr')
# 		# print('temp =',temp)
# 			lambsolRe[index] = temp.x[0]
# 			lambsolIm[index] = temp.x[1]
# 		# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
# 			init = initial_guess();
# 			if lambsolRe[index] >= 0:
# 				OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
# 			else:
# 				OmegStab.append(Omega[index]); tauStab.append(tau[index]);
#
# 		print('distance',np.asarray(3.0E8)*tauStab)
# 	return {'Re': lambsolRe, 'Im': np.abs(lambsolIm), 'OmegUnst': OmegUnst, 'tauUnst': tauUnst, 'OmegStab': OmegStab, 'tauStab': tauStab}
#
# 	#
