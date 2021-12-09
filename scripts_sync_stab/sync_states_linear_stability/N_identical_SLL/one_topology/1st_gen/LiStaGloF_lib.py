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
	return np.linspace(-tauf, maxp, int( ( tauf+maxp )/dp ) )

# set initial guess
def initial_guess():
	return (0.5, 0.01);
def K(Kvco, AkPD, Ga1):
	return (Kvco*Ga1*AkPD)/2.0
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

# phase difference
def thetam(mx, Nx):
	return 2.0*np.pi*m/Nx

def globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, sync_state, INV, mx, Nx):
	#Here we calculate analyticaly the Global Frequency of the synchronised states of the network of two SLLs.
	#First of all, there are two synchronised states, the in-phase and anti-phase. So, the inphase refers to that.
	#we distiguish the analog and the digital case of the SLLs. So, the digital refers to that.
	#The choice of digital and inphase mode is made in the main programs.

	p = xrange(maxp, tauf)				 #p=Omega*(tau-tauf)
	# print( K(Kvco, AkPD, Ga1))
	if sync_state==1:
		if digital:
			Omega = w + K(Kvco, AkPD, Ga1)* cfDig( p +INV)
			tau	  = ( Omega * tauf + v * p ) / Omega
			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tauIn=', tau,'OmIn=', Omega))

		else:
			Omega = w + K(Kvco, AkPD, Ga1) * cfAna( p +INV)
			tau	  = ( Omega * tauf + v * p ) / Omega
			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tau=',tau,'Om=', Omega))

	elif sync_state==2:
		if digital:
			Omega = w + (K(Kvco, AkPD, Ga1)/2.0)*(cfDig( p + np.pi + INV)+cfDig(p-np.pi+INV) )
			tau	  = ( Omega * tauf + v * p ) / Omega
			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tauAn=', tau,'OmAn=', Omega))
			# if (p>=0.15 and p<=0.22):
			# 	print('tau=',tau,'Omega=',Omega)
		else:
			Omega = w - K(Kvco, AkPD, Ga1) * cfAna( p + INV)
			tau	  = ( Omega * tauf + v * p ) / Omega
#################################################################################################################################
# m-twist state for 1-dimensional topology and with boundary conditions

	elif sync_state==3:
		if digital:
			Omega = w + (K(Kvco, AkPD, Ga1)/2.0)*(cfDig( p + thetam(mx, Nx) + INV)+cfDig(p- thetam(mx, Nx) + INV) )
			tau	  = ( Omega * tauf + v * p ) / Omega
			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tauAn=', tau,'OmAn=', Omega))
			# if (p>=0.15 and p<=0.22):
			# 	print('tau=',tau,'Omega=',Omega)
		else:
			Omega = w - K(Kvco, AkPD, Ga1) * cfAna( p + thetam(mx, Nx) )
			tau	  = ( Omega * tauf + v * p ) / Omega

	return {'Omeg': Omega, 'tau': tau}

# class Polynomial:
#
#     def __init__(self, *coefficients):
#         """ input: coefficients are in the form a_n, ...a_1, a_0
#         """
#         # for reasons of efficiency we save the coefficients in reverse order,
#         # i.e. a_0, a_1, ... a_n
#         self.coefficients = coefficients[::-1] # tuple is also turned into list
#
#     def __repr__(self):
#         """
#         method to return the canonical string representation
#         of a polynomial.
#
#         """
#         # The internal representation is in reverse order,
#         # so we have to reverse the list
#         return "Polynomial" + str(self.coefficients[::-1])
#
#     def __call__(self, l1):
#         res = 0
# 	l1 = l1_vec[0] + 1j * l1_vec[1]
#         for index, coeff in enumerate(self.coefficients):
#             res += coeff * l1** index
#         return res
def linStabEq(l_vec, Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, sync_state, INV, mx, Nx):
	#This function gives the characteristic equation of the model
	#alpha = K/v*h'(Omega*(tau-tauf)/v)
	x = np.zeros(2)
	if sync_state==1:
		if digital:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+INV)
		else:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(Omega*(tau-tauf)/v+INV)

	elif sync_state==2:
		if digital:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+np.pi+INV)
		else:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(Omega*(tau-tauf)/v-np.pi+INV)
#################################################################################################################################
# m-twist state for 1-dimensional topology and with boundary conditions

	elif sync_state==3:
		if digital:
			alpha = (1.0/2.0)*(K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v + thetam(mx, Nx)+INV )
		else:
			alpha = (1.0/2.0)*(K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv( Omega*(tau-tauf)/v - thetam(mx, Nx)+INV )

	# print(alpha)
	l = l_vec[0] + 1j * l_vec[1]
	f = l*((1.0+l*tauc/order)**order)*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)))
	#print(f)
	x[0] = np.real(f)
	x[1] = np.imag(f)

	#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x



def linStabEq_expansion(l_vec, Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, sync_state, INV, mx, Nx):
	#This function gives the expansion of 3rd order of characteristic equation of the model
	x = np.zeros(2)
	if sync_state==1:
		if digital:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+INV)
		else:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(Omega*(tau-tauf)/v+INV)

	elif sync_state==2:
		if digital:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+np.pi+INV)
		else:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(Omega*(tau-tauf)/v-np.pi+INV)
#################################################################################################################################
# m-twist state for 1-dimensional topology and with boundary conditions

	elif sync_state==3:
		if digital:
			alpha = (1.0/2.0)*(K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v + thetam(mx, Nx)+INV )
		else:
			alpha = (1.0/2.0)*(K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv( Omega*(tau-tauf)/v - thetam(mx, Nx)+INV )

	l = l_vec[0] + 1j * l_vec[1]

 # (tauc^2/2 - tauc^2/(
 #    2 order) + tauc tauf + tauf^2/2 -
 #    1/6 alpha (-tau + tauf)^3 zeta) l^3

	f = ((alpha - alpha*zeta)+(1.0+(tau-tauf)*zeta*alpha)*l+(tauc + tauf-0.5*(-tau+tauf)*(-tau+tauf)*zeta*alpha)*l**2+(0.5*(1.0/(2.0*order))*tauc**2-tauc*tauf+0.5*tauf*tauf-(1.0/6.0)*(-tau+tauf)*(-tau+tauf)*(-tau+tauf)*zeta*alpha)*l**3 )
	#print(f)
	x[0] = np.real(f)
	x[1] = np.imag(f)
	#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x
# lambsolReExpan = np.zeros(len(globalFreq(w, K, tauf, v, digital, maxp)['tau']),dtype=np.complex)
# lambsolImExpan = np.zeros(len(globalFreq(w, K, tauf, v, digital, maxp)['tau']),dtype=np.complex)


def solveLinStab(Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase, INV, mx, Nx, maxp, expansion):
	lambsolRe = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	lambsolIm = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	init = initial_guess();
	zeta = -1;
	c 	 = 3E8;
	OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];
	#print(type(init), init)
	#print(len(xrange(K)))
	#
	if expansion:
		for index in range(len(xrange(maxp, tauf))):

			# print(index, wR[index])
			temp = optimize.root(linStabEq_expansion, init, args=(Omega[index], tau[index], tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, sync_state, INV, mx, Nx), tol=1e-14, method='hybr')
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

			temp = optimize.root(linStabEq, init, args=(Omega[index], tau[index], tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, sync_state, INV,  mx, Nx), tol=1e-14, method='hybr')
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

def solveLinStab_topology_comparison(Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v,  order, digital, sync_state, INV, mx, Nx, maxp, expansion, zetas):
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
				temp = optimize.root(linStabEq_expansion, init, args=(Omega[index1], tau[index1], tauf, Kvco, AkPD, Ga1, tauc, v, zetas[index2], order, digital, sync_state, INV, mx, Nx), tol=1e-14, method='hybr')
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
			init = (np.real(lambsolReMax[index1]), np.real(lambsolImMax[index1]))
			#print(type(init), init)
			#print('Re[lambda] =',lambsol)
	else:
		for index1 in range(len(xrange(maxp, tauf))):
			for index2 in range(len(zetas)):
				# print(index, wR[index])
				temp = optimize.root(linStabEq, init, args=(Omega[index1], tau[index1], tauf, Kvco, AkPD, Ga1, tauc, v, zetas[index2], order, digital, sync_state, INV, mx, Nx), tol=1e-14, method='hybr')
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
			init = (np.real(lambsolReMax[index1]), np.real(lambsolImMax[index1]))
			#print(type(init), init)
			#print('Re[lambda] =',lambsol)

	return {'Re': lambsolRe, 'Im': lambsolIm, 'ReMax': lambsolReMax, 'ImMax': lambsolImMax, 'Omeg': Omega, 'tau': tau}
