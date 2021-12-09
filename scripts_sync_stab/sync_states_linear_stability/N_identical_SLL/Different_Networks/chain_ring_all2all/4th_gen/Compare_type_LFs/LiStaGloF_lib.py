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
	return (8e6, 8e6);
def K(Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga):
	return (Kvco*GkPD*Gk*Ak*Gl*Al*GkLF*Gvga)/2.0

def filterRC(l, tauc, order):
	return (1.0+l*tauc)**order

def filterChe(l, tauc, order):
	return (1.0+l*tauc/order)**order

def filterLeadLag(l, tauc, order):
	tauc2=1.0/(2.0*np.pi*700E6)
	return (1.0+l*(tauc+tauc2))/(1.0+l*tauc)

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
cfAna        = lambda x: -np.cos(x);
def cfAnaInverse(x):
	if np.abs(x)>1:
		print('Error! Inverse fct. of arccos wave called with argument out of bounds.'); exit();
	return -np.arccos(x);
cfAnaDeriv   = lambda x: 1.0*np.sin(x);

def globalFreq(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase,INV):
	#Here we calculate analyticaly the Global Frequency of the synchronised states of the network of two SLLs.
	#First of all, there are two synchronised states, the in-phase and anti-phase. So, the inphase refers to that.
	#we distiguish the analog and the digital case of the SLLs. So, the digital refers to that.
	#The choice of digital and inphase mode is made in the main programs.
	print('coupling strength K=', K(Kvco, GkPD, Gk, Ak, Gl, Al, GkLF, Gvga),' radHz and K_Hz=', K(Kvco, GkPD, Gk, Ak, Gl, Al, GkLF, Gvga)/(2.0*np.pi),' Hz')
	p = xrange(maxp, tauf)				 #p=Omega*(tau-tauf)
	# print( K(Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga))
	if inphase:
		if digital:
			Omega = w + K(Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga)* cfDig( p + INV )
			tau	  = ( Omega * tauf + v * p ) / Omega
			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tauIn=', tau,'OmIn=', Omega))

		else:
			Omega = w + K(Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga) * cfAna( -p + INV )
			tau	  = ( Omega * tauf + v * p ) / Omega
			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tau=',tau,'Om=', Omega))

	else:
		if digital:
			Omega = w + (K(Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga)/2.0)*(cfDig( p + np.pi+INV)+cfDig(p-np.pi+INV) )
			tau	  = ( Omega * tauf + v * p ) / Omega
			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tauAn=', tau,'OmAn=', Omega))
			# if (p>=0.15 and p<=0.22):
			# 	print('tau=',tau,'Omega=',Omega)
		else:
			Omega = w - K(Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga) * cfAna( -p + INV)
			tau	  = ( Omega * tauf + v * p ) / Omega

	return {'Omeg': Omega, 'tau': tau}

def linStabEq(l_vec, Omega, tau, tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, zeta, order, digital, inphase, INV, filter):
	#This function gives the characteristic equation of the model
	#alpha = K/v*h'(Omega*(tau-tauf)/v)
	x = np.zeros(2)
	if inphase:
		if digital:
			alpha = (K(Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+INV)
		else:
			alpha = (K(Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v+INV)

	else:
		if digital:
			alpha = (K(Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+np.pi+INV)
		else:
			alpha = (K(Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v-np.pi+INV)

	l = l_vec[0] + 1j * l_vec[1]
	if filter == 1:
		f = l*(filterChe(l, tauc, order) )*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)))
	elif filter ==2:
		f = l*(filterRC(l, tauc, order) )*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)))
	elif filter ==3:
		f = l*(filterLeadLag(l, tauc, order) )*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)))
	# f = l*((1.0+l*tauc/order)**order)*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)))
	#print(f)
	x[0] = np.real(f)
	x[1] = np.imag(f)

	#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x

def linStabEq_expansion(l_vec, Omega, tau, tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, zeta, order, digital):
	x = np.zeros(2)
	if digital:
		alpha = ( K(Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga)/v ) * cfDigDeriv( -Omega*(tau-tauf)/v )
	else:
		alpha = ( K(Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga)/v ) * cfAnaDeriv( -Omega*(tau-tauf)/v )

	l = l_vec[0] + 1j * l_vec[1]

	f = alpha*(1.0 - zeta) + (1.0 - alpha*(-tau + tauf)*zeta)*l + (tauc +tauf - 0.5*alpha*zeta*(-tau + tauf)**2)*l**2 + ((0.5*(-1.0 + order)*tauc**2)/order + tauc*tauf + 0.5*tauf**2-(1.0/6.0)*alpha*zeta*(-tau + tauf)**3)*l**3

	x[0] = np.real(f)
	x[1] = np.imag(f)
	#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x
#
# lambsolReExpan = np.zeros(len(globalFreq(w, K, tauf, v, digital, maxp)['tau']),dtype=np.complex)
# lambsolImExpan = np.zeros(len(globalFreq(w, K, tauf, v, digital, maxp)['tau']),dtype=np.complex)

def solveLinStab(Omega, tau, tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, order, digital, maxp, inphase, expansion, INV, filter):

	# Here we solve the characteristic equation and we get the real and imaginary parts of the solutionsself.
	# We calculate for different inital conditions, so we calculate the largest value of the real part of the solution in order to study the stability of the system.
	##########################################################
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
			temp =  optimize.root(linStabEq_expansion, init, args=(Omega[index], tau[index], tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, zeta, order, digital, inphase, INV), tol=1e-14, method='hybr')
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

			temp = optimize.root(linStabEq, init, args=(Omega[index], tau[index], tauf, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauc, v, zeta, order, digital, inphase, INV, filter), tol=1e-14, method='hybr')
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
