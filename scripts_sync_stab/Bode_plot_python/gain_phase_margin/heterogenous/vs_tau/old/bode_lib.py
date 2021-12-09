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
# from mpmath import findroot
# from sympy import I, limit
# from sympy.solvers import solveset
# from sympy.solvers import solve
# from sympy import Symbol, Function, nsolve
# from sympy import sin, cos, exp
# import mpmath
# mpmath.mp.dps = 25

coupfun='cos'
def coupfunction(coupfun):
	return coupfun

# conditions due to structure of the equations determine values of wR depending on value of K
def xrange(maxp, tauf, dp=0.0005): #dp=0.0005
	return np.linspace(1E-9, maxp, int((tauf+maxp)/dp))

def K(Kvco, AkPD, GkLF, Gvga):
	return (Kvco*AkPD*GkLF*Gvga)/2.0
# set initial guess
# def initial_guess():
# 	return (3.101, 1.01);

def initial_guess():
	return (np.linspace(-1e-5, 1e-5, 6));

# digital case
cfDig = lambda x: sawtooth(x,width=0.5)
def cfDigInverse(x):
	if np.abs(x)>1:
		print('Error! Inverse fct. of triangular wave called with argument out of bounds.'); exit();
	# 	return -np.pi/2.0*x-np.pi/2.0;
	# else:
	return +np.pi/2.0*x+np.pi/2.0;
cfDigDeriv   = lambda x: (2.0/np.pi)*square(x,duty=0.5)

# analog case
if coupfun=='sin':
	cfAna        = lambda x: np.sin(x);
	# def cfAnaInverse(x):
	# 	if np.abs(x)>1:
	# 		print('Error! Inverse fct. of arccos wave called with argument out of bounds.'); exit();
	# 	return -np.arccos(x);
	cfAnaDeriv   = lambda x: 1.0*np.cos(x);

if coupfun=='cos':
	cfAna        = lambda x: np.cos(x);
	# def cfAnaInverse(x):
	# 	if np.abs(x)>1:
	# 		print('Error! Inverse fct. of arccos wave called with argument out of bounds.'); exit();
	# 	return -np.arccos(x);
	cfAnaDeriv   = lambda x: -1.0*np.sin(x);

def globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state, INV):
	p = xrange(maxp, tauf)
	if sync_state == 'inphase':
		if digital:
			beta  = 0.0;
			Omega = w + K(Kvco, AkPD,  GkLF, Gvga) * cfDig( -p + INV )
			tau	  = ( Omega * tauf + v * p ) / Omega
			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tauIn=', tau,'OmIn=', Omega))

		else:
			beta  = 0.0;
			Omega = w + K(Kvco, AkPD,  GkLF, Gvga) * cfAna( -p + INV )
			tau	  = ( Omega * tauf + v * p ) / Omega
			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tau=',tau,'Om=', Omega))

	elif sync_state == 'antiphase':
		if digital:
			beta  = np.pi;
			Omega = w + (K(Kvco, AkPD, GkLF, Gvga)/2.0) * (cfDig( p + np.pi + INV )+cfDig(p- np.pi + INV ) )
			tau	  = ( Omega * tauf + v * p ) / Omega
		else:
			beta  = np.pi;
			Omega = w - K(Kvco, AkPD, GkLF, Gvga) * cfAna( -p + INV )
			tau	  = ( Omega * tauf + v * p ) / Omega

	elif sync_state == 'entrainment':
		if digital:

			tau   = xrange(maxp, tauf)
			Omega = wref
			beta  =v*cfDigInverse((wR-w)/K) + wR*(tau-tauf)
		else:

			tau   = xrange(maxp, tauf)
			Omega = wref
			beta  = v*np.arcsin((w-wref)/K) + wref*(tau-tauf)
			# print(tau)
	return {'Omeg': Omega, 'tau': tau, 'beta': beta}

def linStabEq(l_vec, wref, w, Omega, tau, tauf, Kvco, AkPD,  GkLF,Gvga, tauc, v, zeta, digital, sync_state, INV ):
	x = np.zeros(2)
	if sync_state == 'inphase':
		if digital:
			alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*cfDigDeriv(-Omega*(tau-tauf)/v)
		else:
			alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v)

		l = l_vec[0] + 1j * l_vec[1]
		f = l*(1.0+l*tauc)*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)))
		#print(f)
	elif sync_state == 'antiphase':
		if digital:
			alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+np.pi)
		else:
			alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v-np.pi)

		l = l_vec[0] + 1j * l_vec[1]
		f = l*(1.0+l*tauc)*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)))
		#print(f)
	elif sync_state == 'entrainment':
		beta	= v*np.arccos((wref-w)/K)-wref*(tau-tauf)
		# print('For the entrainment case, the constant phase-shift will be beta=', beta)
		if digital:
			print('Fix it')
			# alpha = K*cfDigDeriv((-wref*(tau-tauf)-beta
		else:
			alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*np.sin((wref*(tau-tauf))/v+beta/v)

		l = l_vec[0] + 1j * l_vec[1]
		#f = l*(1.0+l*tauc) + alpha * np.exp(l*tauf)

		f = (l**2.0) + (1.0/tauc)*l + alpha/tauc #* np.exp(l*tauf)


	x[0] = np.real(f)
	x[1] = np.imag(f)

		#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x

# def linStabEq_expansion(l_vec, wref, w, Omega, tau, tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, zeta, digital, sync_state):
#
#
# 	print('Fix it')
# 	# x = np.zeros(2)
# 	# if digital:
# 	# 	alpha = (K/v)*cfDigDeriv(-Omega*(tau-tauf)/v)
# 	# else:
# 	# 	alpha = (K/v)*cfAnaDeriv(-Omega*(tau-tauf)/v)
# 	#
# 	# l = l_vec[0] + 1j * l_vec[1]
# 	# f = ( (alpha - alpha*zeta)+(1.0+(tau-tauf)*zeta*alpha)*l+(tauc + tauf-0.5*(-tau+tauf)*(-tau+tauf)*zeta*alpha)*l**2+(tauc*tauf+0.5*tauf*tauf-(1.0/6.0)*(-tau+tauf)*(-tau+tauf)*(-tau+tauf)*zeta*alpha)*l**3 )
# 	# #print(f)
# 	# x[0] = np.real(f)
# 	# x[1] = np.imag(f)
# 	#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
# 	return x
#
# lambsolReExpan = np.zeros(len(globalFreq(w, K, tauf, v, digital, maxp)['tau']),dtype=np.complex)
# lambsolImExpan = np.zeros(len(globalFreq(w, K, tauf, v, digital, maxp)['tau']),dtype=np.complex)

def solveLinStab(wref, w, Omega, tau, tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, digital, maxp, sync_state, expansion, INV ):
	# lambsolRe = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	# lambsolIm = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	init = initial_guess();
	lambsolRe = np.zeros([len(init),len(xrange(maxp, tauf))],dtype=np.float64)
	lambsolIm = np.zeros([len(init),len(xrange(maxp, tauf))],dtype=np.float64)
	lambsolReMax = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	lambsolImMax = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)

	zeta = -1.0;
	c 	 = 3E8;
	OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];
	#print(type(init), init)
	#print(len(xrange(K)))
	if sync_state == 'inphase':
		if expansion:
			for index in range(len(xrange(maxp, tauf))):
				for index2 in range(len(init)):
				# print(index, wR[index])
					temp =  optimize.root(linStabEq_expansion, (init[index2],init[index2]), args=(wref, w, Omega[index], tau[index], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, zeta, digital, sync_state, INV ), tol=1e-14, method='hybr')
				# print('temp =',temp)
				# print('temp =',temp)
					lambsolRe[index2,index]  = temp.x[0]
					lambsolIm[index2,index]  = temp.x[1]
				# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
					# init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))
					tempRe = lambsolRe[np.round(lambsolRe[:,index],16)!=0.0,index];
					tempIm = lambsolIm[np.round(lambsolIm[:,index],16)!=0.0,index];
					if len(tempRe) != 0:
						#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
						lambsolReMax[index] = np.real(np.max(tempRe));
						if tempRe[:].argmax() < len(tempIm):
							lambsolImMax[index] = tempIm[tempRe[:].argmax()]
					else:
						lambsolReMax[index] = 0.0;
						lambsolImMax[index] = 0.0;

					init = initial_guess()

		else:
			for index in range(len(xrange(maxp, tauf))):
				for index2 in range(len(init)):
					# print(init[1])
					temp = optimize.root(linStabEq, (init[index2],init[index2]), args=(wref, w, Omega[index], tau[index], tauf, Kvco, AkPD, GkLF,Gvga,tauc, v, zeta, digital, sync_state, INV ),  tol=1.0e-14, method='hybr')
					lambsolRe[index2,index]  = temp.x[0]
					lambsolIm[index2,index]  = temp.x[1]
					# print(lambsolRe)
					# lambsolRebetain[index]  = temp.x[0]
					# lambsolImbetain[index]  = temp.x[1]
					# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
					# init = (np.real(lambsolRebetain[index]), np.real(lambsolImbetain[index]))

				tempRe = lambsolRe[np.round(lambsolRe[:,index],16)!=0.0,index];
				tempIm = lambsolIm[np.round(lambsolIm[:,index],16)!=0.0,index];
				if len(tempRe) != 0:
					#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
					lambsolReMax[index] = np.real(np.max(tempRe));
					if tempRe[:].argmax() < len(tempIm):
						lambsolImMax[index] = tempIm[tempRe[:].argmax()]
				else:
					lambsolReMax[index] = 0.0;
					lambsolImMax[index] = 0.0;

				init = initial_guess()

	elif sync_state == 'antiphase':
		if expansion:
			for index in range(len(xrange(maxp, tauf))):

				# print(index, wR[index])
				temp =  optimize.root(linStabEq_expansion, (init[index2],init[index2]), args=(wref, w, Omega[index], tau[index], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, zeta, digital, sync_state, INV ), tol=1e-14, method='hybr')
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
				for index2 in range(len(init)):
					# print(init[1])
					temp = optimize.root(linStabEq, (init[index2],init[index2]), args=(wref, w, Omega[index], tau[index], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, zeta, digital, sync_state, INV ),  tol=1.0e-14, method='hybr')
					lambsolRe[index2,index]  = temp.x[0]
					lambsolIm[index2,index]  = temp.x[1]
					# print(lambsolRe)
					# lambsolRebetain[index]  = temp.x[0]
					# lambsolImbetain[index]  = temp.x[1]
					# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
					# init = (np.real(lambsolRebetain[index]), np.real(lambsolImbetain[index]))

				tempRe = lambsolRe[np.round(lambsolRe[:,index],16)!=0.0,index];
				tempIm = lambsolIm[np.round(lambsolIm[:,index],16)!=0.0,index];
				if len(tempRe) != 0:
					#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
					lambsolReMax[index] = np.real(np.max(tempRe));
					if tempRe[:].argmax() < len(tempIm):
						lambsolImMax[index] = tempIm[tempRe[:].argmax()]
				else:
					lambsolReMax[index] = 0.0;
					lambsolImMax[index] = 0.0;

			init = initial_guess()

	elif sync_state == 'entrainment':
		if expansion:
			print('Fix it')
		# 	for index in range(len(xrange(maxp, tauf))):
		#
		# 		# print(index, wR[index])
		# 		temp =  optimize.root(linStabEq_expansion, (init[index2],init[index2]), args=(wref, w, Omega, tau[index], tauf, K, tauc, v, zeta, digital, sync_state), tol=1e-14, method='hybr')
		# 	# print('temp =',temp)
		# 	# print('temp =',temp)
		# 		lambsolRe[index] = temp.x[0]
		# 		lambsolIm[index] = abs(temp.x[1])
		# 	# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
		# 		# init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))
		# 		init = initial_guess()
		# 	#print(type(init), init)
		# 	#print('Re[lambda] =',lambsol)
		# 		if lambsolRe[index] >= 0:
		# 			OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
		# 		else:
		# 			OmegStab.append(Omega[index]); tauStab.append(tau[index]);
		# # else:

			#
		else:
			for index in range(len(globalFreq(wref, Omega, K, tauf, v, digital, maxp, sync_state)['tau'])):
				for index2 in range(len(init)):
					# print(init[1])
					temp = optimize.root(linStabEq, (init[index2],init[index2]), args=(wref, w, Omega, tau[index], tauf, Kvco, AkPD, GkLF,Gvga,tauc, v, zeta, digital, sync_state),  tol=1.0e-15, method='hybr')
					lambsolRe[index2,index]  = temp.x[0]
					lambsolIm[index2,index]  = temp.x[1]
					# print(lambsolRe)
					# lambsolRebetain[index]  = temp.x[0]
					# lambsolImbetain[index]  = temp.x[1]
					# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
					# init = (np.real(lambsolRebetain[index]), np.real(lambsolImbetain[index]))
					# print(globalFreq(wref, Omega, K, tauf, v, digital, maxp, sync_state)['tau'])
				tempRe = lambsolRe[np.round(lambsolRe[:,index],23)!=0.0,index];
				tempIm = lambsolIm[np.round(lambsolIm[:,index],36)!=0.0,index];
				if len(tempRe) != 0:
					# print('type(np.max(tempRe)):',np.real(np.max(tempRe)));
					lambsolReMax[index] = np.real(np.max(tempRe));
					if tempRe[:].argmax() < len(tempIm):
						lambsolImMax[index] = tempIm[tempRe[:].argmax()]
				else:
					lambsolReMax[index] = 0.0;
					lambsolImMax[index] = 0.0;

			init = initial_guess()

	return {'Re': lambsolRe, 'Im': np.abs(lambsolIm), 'OmegUnst': OmegUnst, 'tauUnst': tauUnst, 'OmegStab': OmegStab, 'tauStab': tauStab,'ReMax': lambsolReMax,'ImMax':np.abs(lambsolImMax)}

def analytical(wref, w, Omega, tau, tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, digital, maxp, sync_state):
	if globalFreq(wref, Omega, K, tauf, v, digital, maxp, sync_state)['tau'][0] == 0 and tauf == 0:
		beta	= v*np.arccos((wref-w)/K)
		alpha = (K(Kvco, AkPD, GkLF,Gvga,)/v)*cfAnaDeriv(beta/v)
		lRe1, lRe2, lIm1, lIm2 = symbols("lRe1 lRe12 lIm1 lIm2", real=True)
		l1 = lRe1 + I*lIm1
		l2 = lRe2 + I*lIm2
		sol1 = solve(l1-0.5*(-1.0/tauc+cmath.sqrt(1.0+4.0*alpha*tauc)/tauc))
		sol2 = solve(l2-0.5*(-1.0/tauc-cmath.sqrt(1.0+4.0*alpha*tauc)/tauc))
		# print(l1.subs(sol1[0]), l1.subs(sol1[1]))
		# print(l2.subs(sol2[0]), l2.subs(sol2[0]))
		# l1 =0.5*(-1.0/tauc+np.sqrt(1.0+4.0*alpha*tauc)/tauc)
		# l2 =0.5*(-1.0/tauc-np.sqrt(1.0+4.0*alpha*tauc)/tauc)
		# print('l1=',l1,'l2=',l2)
		# print('l1=',l1,'l2=',l2)
		print('The solution are {0} and {1}'.format(sol1,sol2))
		return (sol1,sol2)

def LoopGainSteadyState(Omega, tau, tauf, v, Kvco, AkPD, GkLF, Gvga, digital, model):
	if model=='Nonlinear':
		if digital:
			alpha = (K(Kvco, AkPD, GkLF, Gvga)/v)*cfDigDeriv(-Omega*(tau-tauf)/v)
		else:
			alpha = (K(Kvco, AkPD, GkLF, Gvga)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v)
	elif model=='Linear':
		alpha = (K(Kvco, AkPD, GkLF, Gvga)/v)
	return alpha;

def HopenLoopMutuallyCoupledOnePLL(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital, model):
	return ( LoopGainSteadyState(Omega, tau, tauf, v, Kvco, AkPD, GkLF, Gvga, digital, model)/( 1j*gamma*( 1.0 + 1j*gamma*tauc ) ) )

def PhaseopenloopMutuallyCoupledOnePLL(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital, model):
	phase =	 cmath.phase(HopenLoopMutuallyCoupledOnePLL(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))
	return 360.0*phase/(2.0*np.pi)

def HclosedLoopMutuallyCoupledOnePLL(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital, model):
	# return HopenLoopMutuallyCoupledOnePLL(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital, model)/( 1.0 + HopenLoopMutuallyCoupledOnePLL(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital, model)*np.exp(1j*gamma*( -tauf ) ) )
	return 1.0 / ( 1.0/HopenLoopMutuallyCoupledOnePLL(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital, model) + np.exp(1j*gamma*( -tauf ) ) )
	# return ((alpha*np.exp(1j*w*(-tauf))/(1j*w*(1+1j*w*tauc)))/(1.0+(alpha/(1j*w*(1+1j*w*tauc)))))

def PhaseclosedloopMutuallyCoupledOnePLL(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model):
	phase =	 cmath.phase(HclosedLoopMutuallyCoupledOnePLL(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))
	return 360.0*phase/(2.0*np.pi)

def HopenLoopMutuallyCoupledNet(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model):
	return (np.exp(-2.0*1j*gamma*tau)*(HclosedLoopMutuallyCoupledOnePLL(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))**2 )

	# return (np.exp(-2*1j*w*tau)*((alpha*np.exp(1j*w*(-tauf))/(1j*w*((1+1j*w*tauc)) ))/(1.0+(alpha*np.exp(1j*w*(-tauf))/(1j*w*((1+1j*w*tauc)) ))))**2)

def HopenLoopMutuallyCoupledNet2(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model):

	return (np.exp(-1j*gamma*tau )*( 1.0/HopenLoopMutuallyCoupledOnePLL(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital )+1.0) )

def PhaseopenloopMutuallyCoupledNet(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model):
	phase =	 cmath.phase(HopenLoopMutuallyCoupledNet(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))
	return 360.0*phase/(2.0*np.pi)

def HclosedloopMutuallyCoupledNet(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model):
	# return ( (np.exp(2*1j*w*(-tau))*((alpha*np.exp(1j*w*(-tauf))/(1j*w*((1+1j*w*tauc)**2)))/(1.0+(alpha/(1j*w*((1+1j*w*tauc)**2) ))))**2)/(1.0- (np.exp(2*1j*w*(-tauf))*((alpha*np.exp(1j*w*(-tauf))/(1j*w*(1+1j*w*tauc)))/(1.0+(alpha/(1j*w*(1+1j*w*tauc)))))**2) )   )
	return 1.0 / ( 1.0/HopenLoopMutuallyCoupledNet(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model) - 1.0)

def PhaseclosedloopMutuallyCoupledNet(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model):
	phase =	 cmath.phase(HopenLoopMutuallyCoupledNet(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))
	return 360.0*phase/(2.0*np.pi)

def GainMarginMutuallyCoupledNet(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model):

	temp = HopenLoopMutuallyCoupledNet(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model)
	realpart = temp.real
	imagpart = temp.imag
	return 1.0-np.abs(np.sqrt(realpart**2+imagpart**2))

def PhaseMarginMutuallyCoupledNet(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model):

	temp = HopenLoopMutuallyCoupledNet(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model)
	realpart = temp.real
	imagpart = temp.imag
	if realpart>0.0:
		return np.arctan(imagpart/realpart)
	if realpart<0.0:
		return np.pi+np.arctan(imagpart/realpart)

def GainMarginMutuallyCoupledOne(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model):

	temp = HopenLoopMutuallyCoupledOnePLL(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model)
	realpart = temp.real
	imagpart = temp.imag
	# print(realpart,imagpart)
	return 1.0-np.abs(np.sqrt(realpart**2+imagpart**2))

def PhaseMarginMutuallyCoupledOne(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model):

	temp = HopenLoopMutuallyCoupledOnePLL(gamma, Omega, tau, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model)
	realpart = temp.real
	imagpart = temp.imag

	if realpart>0.0:
		return np.pi-np.arctan(imagpart/realpart)
	if realpart<0.0:
		return np.arctan(imagpart/realpart)
