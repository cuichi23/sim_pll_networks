#Linear Stability of Global Frequency (LiStaGoF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
from numpy import pi, sin
import topology
from topology import eigenvalzeta
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
	return (0.7, 0.90);

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
	p = xrange(maxp, tauf)
	if digital:
		Omega = w + K * cfDig( -p )
		tau	  = ( Omega * tauf + v * p ) / Omega
	else:
		Omega = w + K * cfAna( -p )
		tau	  = ( Omega * tauf + v * p ) / Omega
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
# lambsolReExpan = np.zeros(len(globalFreq(w, K, tauf, v, digital, maxp)['tau']),dtype=np.complex)
# lambsolImExpan = np.zeros(len(globalFreq(w, K, tauf, v, digital, maxp)['tau']),dtype=np.complex)


def solveLinStab(Omega, tau, tauf, K, tauc, v, digital, maxp, expansion):
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
			temp = optimize.root(linStabEq_expansion, init, args=(Omega[index], tau[index], tauf, K, tauc, v, zeta, digital), tol=1e-14, method='hybr')
		# print('temp =',temp)
			lambsolRe[index] = temp.x[0]
			lambsolIm[index] = abs(temp.x[1])
		# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
			init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))
		#print(type(init), init)
		#print('Re[lambda] =',lambsol)
			if lambsolRe[index] >= 0:
				OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
			else:
				OmegStab.append(Omega[index]); tauStab.append(tau[index]);
	else:
		for index in range(len(xrange(maxp, tauf))):

			temp = optimize.root(linStabEq, init, args=(Omega[index], tau[index], tauf, K, tauc, v, zeta, digital), tol=1e-14, method='hybr')
		# print('temp =',temp)
			lambsolRe[index] = temp.x[0]
			lambsolIm[index] = abs(temp.x[1])
		# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
			init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))
			if lambsolRe[index] >= 0:
				OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
			else:
				OmegStab.append(Omega[index]); tauStab.append(tau[index]);

	return {'Re': lambsolRe, 'Im': lambsolIm, 'OmegUnst': OmegUnst, 'tauUnst': tauUnst, 'OmegStab': OmegStab, 'tauStab': tauStab}




def solveLinStabKrange(Omega, tau, tauf, K, tauc, v, digital, maxp, expansion):
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
			temp = optimize.root(linStabEq_expansion, init, args=(Omega[index], tau, tauf, K[index], tauc, v, zeta, digital), tol=1e-14, method='hybr')
		# print('temp =',temp)
			lambsolRe[index] = temp.x[0]
			lambsolIm[index] = abs(temp.x[1])
		# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
			init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))
		#print(type(init), init)
		#print('Re[lambda] =',lambsol)
			if lambsolRe[index] >= 0:
				OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
			else:
				OmegStab.append(Omega[index]); tauStab.append(tau[index]);
	else:
		for index in range(len(xrange(maxp, tauf))):

			temp = optimize.root(linStabEq, init, args=(Omega[index], tau, tauf, K[index], tauc, v, zeta, digital), tol=1e-14, method='hybr')
		# print('temp =',temp)
			lambsolRe[index] = temp.x[0]
			lambsolIm[index] = abs(temp.x[1])
		# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
			init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))
			if lambsolRe[index] >= 0:
				OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
			else:
				OmegStab.append(Omega[index]); tauStab.append(tau[index]);

	return {'Re': lambsolRe, 'Im': lambsolIm, 'OmegUnst': OmegUnst, 'tauUnst': tauUnst, 'OmegStab': OmegStab, 'tauStab': tauStab}








def solveLinStab_topology_comparison(Omega, tau, tauf, K, tauc, v, digital, maxp, expansion, zetas):
	# global x;
	# x = np.zeros(len(zetas))
	lambsolRe 	 = np.zeros([len(zetas),len(xrange(maxp, tauf))],dtype=np.float64)
	lambsolIm 	 = np.zeros([len(zetas),len(xrange(maxp, tauf))],dtype=np.float64)
	lambsolReMax = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	lambsolImMax = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	init = initial_guess();
	#print(type(init), init)
	#print(len(xrange(K)))
	# print('\nSolve stability for set of eigenvalues zeta:', zetas, ' and store it in lambsolRe with shape:', lambsolRe.shape)
	if expansion:
		for index1 in range(len(xrange(maxp, tauf))):
			for index2 in range(len(zetas)):
				# print(index, wR[index])
				temp = optimize.root(linStabEq_expansion, init, args=(Omega[index1], tau[index1], tauf, K, tauc, v, zetas[index2], digital), tol=1e-14, method='hybr')
				# print('temp =',temp)
				# lambsolRe[index2][index1] = temp.append(x[index2])
				# lambsolIm[index2][index1] = abs(temp.append(x[index2+1]))
				lambsolRe[index2,index1] = temp.x[0]
				lambsolIm[index2,index1] = abs(temp.x[1])
				# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
			#print('TEST:', lambsolRe[np.round(lambsolRe[:,index1],12)!=0,index1], type(lambsolRe[:,index1]))
			#time.sleep(1.5)
			tempRe = lambsolRe[np.round(lambsolRe[:,index1],12)!=0,index1];
			tempIm = lambsolIm[np.round(lambsolRe[:,index1],12)!=0,index1];
			#print('\ntype(temp1):', type(temp1), 'temp1:', temp1, 'len(temp1):', len(temp1)); #time.sleep(1)
			if len(tempRe) != 0:
				#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
				lambsolReMax[index1] = np.real(np.max(tempRe));
				lambsolImMax[index1] = tempIm[tempRe[:].argmax()]
			else:
				lambsolReMax[index1] = 0.0;
				lambsolImMax[index1] = 0.0;
			#lambsolReMax[index1] = np.max(lambsolRe[:,index1])
			#lambsolImMax[index1] = lambsolIm[lambsolRe[:,index1].argmax(),index1]
			init = initial_guess();
			#print(type(init), init)
			#print('Re[lambda] =',lambsol)
	else:
		for index1 in range(len(xrange(maxp, tauf))):
			for index2 in range(len(zetas)):
				# print(index, wR[index])
				temp = optimize.root(linStabEq, init, args=(Omega[index1], tau[index1], tauf, K, tauc, v, zetas[index2], digital), tol=1e-14, method='hybr')
				#print('\n\ntemp[',zetas[index2],'][',tau[index1],'] =',temp)
				lambsolRe[index2,index1] = temp.x[0]
				lambsolIm[index2,index1] = abs(temp.x[1])
				# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
			#print('TEST:', lambsolRe[np.round(lambsolRe[:,index1],12)!=0,index1], type(lambsolRe[:,index1]))
			#time.sleep(1.5)
			tempRe = lambsolRe[np.round(lambsolRe[:,index1],12)!=0,index1];
			tempIm = lambsolIm[np.round(lambsolRe[:,index1],12)!=0,index1];
			#print('\ntype(temp1):', type(temp1), 'temp1:', temp1, 'len(temp1):', len(temp1)); #time.sleep(1)
			if len(tempRe) != 0:
				#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
				lambsolReMax[index1] = np.real(np.max(tempRe));
				lambsolImMax[index1] = tempIm[tempRe[:].argmax()]
			else:
				lambsolReMax[index1] = 0.0;
				lambsolImMax[index1] = 0.0;
			#lambsolReMax[index1] = np.max(lambsolRe[:,index1])
			#lambsolImMax[index1] = lambsolIm[lambsolRe[:,index1].argmax(),index1]
			init = initial_guess();
			#print(type(init), init)
			#print('Re[lambda] =',lambsol)

	return {'Re': lambsolRe, 'Im': lambsolIm, 'ReMax': lambsolReMax, 'ImMax': lambsolImMax, 'Omeg': Omega, 'tau': tau}
