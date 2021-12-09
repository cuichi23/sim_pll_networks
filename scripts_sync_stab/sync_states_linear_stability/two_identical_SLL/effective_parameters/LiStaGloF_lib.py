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
from scipy.special import lambertw
# from mpmath import findroot
# from sympy import I, limit
# from sympy.solvers import solveset
# from sympy.solvers import solve
# from sympy import Symbol, Function, nsolve
# from sympy import sin, cos, exp
# import mpmath
# mpmath.mp.dps = 25


###########################################################################################

# select coupling Function

coupfun='cos'

def coupfunction(coupfun):
	return coupfun

def xrange(maxp, tauf):

	dp		= 0.1;
	initial = 0.0 #3.8001E5
	return np.linspace(initial, maxp, int((-initial+maxp)/dp))


def xrangedifv(v, maxp, tauf):

	dp		= 0.01;

	initial = 0.0 #3.8001E5
	return np.linspace(initial, maxp, int((-initial+maxp)/dp))

def K(Kvco, AkPD, Ga1):
	return (Kvco*Ga1*AkPD)/2.0
# set initial guess
# def initial_guess():
# 	return (3.101, 1.01);
def initial_guess():
	return (np.linspace(-4e-1, 4e-1, 10));
def initial_guess2():
	return (np.linspace(-4e-2, 4e-2, 10));

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
if coupfun=='sin':
	cfAna        = lambda x: np.sin(x);
	# def cfAnaInverse(x):
	# 	if np.abs(x)>1:
	# 		print('Error! Inverse fct. of arccos wave called with argument out of bounds.'); exit();
	# 	return -np.arccos(x);
	cfAnaDeriv   = lambda x: 1.0*np.cos(x);

if coupfun=='negcos':
	cfAna        = lambda x: -np.cos(x);
	# def cfAnaInverse(x):
	# 	if np.abs(x)>1:
	# 		print('Error! Inverse fct. of arccos wave called with argument out of bounds.'); exit();
	# 	return -np.arccos(x);
	cfAnaDeriv   = lambda x: 1.0*np.sin(x);

if coupfun=='cos':
	cfAna        = lambda x: np.cos(x);
	# def cfAnaInverse(x):
	# 	if np.abs(x)>1:
	# 		print('Error! Inverse fct. of arccos wave called with argument out of bounds.'); exit();
	# 	return -np.arccos(x);
	cfAnaDeriv   = lambda x: -1.0*np.sin(x);

def globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase, INV):
	#Here we calculate analyticaly the Global Frequency of the synchronised states of the network of two SLLs.
	#First of all, there are two synchronised states, the in-phase and anti-phase. So, the inphase refers to that.
	#we distiguish the analog and the digital case of the SLLs. So, the digital refers to that.
	#The choice of digital and inphase mode is made in the main programs.

	# p = xrange(maxp, tauf)				 #p=Omega*(tau-tauf)
	p = xrangedifv(v, maxp, tauf)
	# print( K(Kvco, AkPD, Ga1))
	if inphase:
		if digital:
			Omega = w + K(Kvco, AkPD, Ga1)* cfDig( p +INV)
			tau	  = ( Omega * tauf + v * p ) / Omega
			beta=0.0
			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tauIn=', tau,'OmIn=', Omega))

		else:
			beta=0.0
			Omega = w + K(Kvco, AkPD, Ga1) * cfAna( -p +INV)
			tau	  = ( Omega * tauf + v * p + beta) / Omega

			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tau=',tau,'Om=', Omega))

	else:
		if digital:
			Omega = w + (K(Kvco, AkPD, Ga1)/2.0)*( cfDig( p + np.pi+INV)+cfDig(p-np.pi+INV) )
			tau	  = ( Omega * tauf + v * p ) / Omega
			beta = np.pi
			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tauAn=', tau,'OmAn=', Omega))
			# if (p>=0.15 and p<=0.22):
			# 	print('tau=',tau,'Omega=',Omega)
		else:
			beta = np.pi
			Omega = w - K(Kvco, AkPD, Ga1) * cfAna( -p + INV)
			tau	  = ( Omega * tauf + v * p ) / Omega




	return {'Omeg': Omega, 'tau': tau, 'beta': beta}


def alpha(Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase, INV):
	if inphase:
		if digital:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+INV)
		else:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v+INV)

	else:
		if digital:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v-np.pi/v+INV)
		else:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v-np.pi/v+INV)
	return alpha

def linStabEq(l_vec, Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase, INV):
	#This function gives the characteristic equation of the model
	#alpha = K/v*h'(Omega*(tau-tauf)/v)
	x = np.zeros(2)
	alphaparam=alpha(Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase, INV)
	# if inphase:
	# 	if digital:
	# 		alpha = (K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+INV)
	# 	else:
	# 		alpha = (K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(Omega*(tau-tauf)/v+INV)
	#
	# else:
	# 	if digital:
	# 		alpha = (K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+np.pi+INV)
	# 	else:
	# 		alpha = (K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(Omega*(tau-tauf)/v-np.pi+INV)

	l = l_vec[0] + 1j * l_vec[1]
	f = l*((1.0+l*tauc/order)**order)*np.exp(l*tauf) + alphaparam*(1.0-zeta*np.exp(-l*(tau-tauf)))
	#print(f)
	x[0] = np.real(f)
	x[1] = np.imag(f)

	#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x


def linStabEq_expansion(l_vec, Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase, INV):
	#This function gives the expansion of 3rd order of characteristic equation of the model
	x = np.zeros(2)
	if inphase:
		if digital:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+INV)
		else:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v+INV)

	else:
		if digital:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+np.pi+INV)
		else:
			alpha = (K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v-np.pi+INV)

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
#
# lambsolReExpan = np.zeros(len(globalFreq(w, K, tauf, v, digital, maxp)['tau']),dtype=np.complex)
# lambsolImExpan = np.zeros(len(globalFreq(w, K, tauf, v, digital, maxp)['tau']),dtype=np.complex)

def solveLinStab(Omega, tau, w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase, expansion, INV, zeta):

# Here we solve the characteristic equation and we get the real and imaginary parts of the solutionsself.
# We calculate for different inital conditions, so we calculate the largest value of the real part of the solution in order to study the stability of the system.
##########################################################
	# lambsolRe = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	# lambsolIm = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)

	lambsolRe = np.zeros([len(initial_guess()),len(xrangedifv(v, maxp, tauf))],dtype=np.float64)
	lambsolIm = np.zeros([len(initial_guess()),len(xrangedifv(v, maxp, tauf))],dtype=np.float64)

##########################################################

	#
	lambsolReMax = np.zeros(len(xrangedifv(v, maxp, tauf)),dtype=np.float64)
	lambsolImMax = np.zeros(len(xrangedifv(v, maxp, tauf)),dtype=np.float64)
	init = initial_guess();
	init2 = initial_guess2();
	# zeta = -1.0#*1j;
	c 	 = 3E8;
	OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];betaUnst=[];betaStab=[];
	#print(type(init), init)
	#print(len(xrange(K)))
	if expansion:
		for index in range(len(xrangedifv(v, maxp, tauf))):
			for index2 in range(len(init)):
				temp =  optimize.root(linStabEq_expansion, (init[index2],init2[index2]), args=(Omega[index], tau[index], tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase, INV), tol=1e-14, method='hybr')
		# print('temp =',temp)
		# print('temp =',temp)
				if temp.success == True:
					lambsolRe[index2,index]  = temp.x[0]
					lambsolIm[index2,index]  = temp.x[1]

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
		#print(type(init), init)
		#print('Re[lambda] =',lambsol)
			if lambsolRe[index2,index] >= 0:
				OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
			else:
				OmegStab.append(Omega[index]); tauStab.append(tau[index]);

	else:
		for index in range(len(xrangedifv(v, maxp, tauf))):

			for index2 in range(len(init)):


				# print(init[1])
				temp = optimize.root(linStabEq, (init[index2], init2[index2]), args=( Omega[index], tau[index], tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase, INV),  tol=1.0e-12, method='hybr')
				if temp.success == True:
					lambsolRe[index2,index]  = temp.x[0]
					lambsolIm[index2,index]  = temp.x[1]
			tempRe = lambsolRe[np.round(lambsolRe[:,index],16)!=0.0,index];
			tempIm = lambsolIm[np.round(lambsolIm[:,index],16)!=0.0,index];
			if len(tempRe) != 0:

				lambsolReMax[index] = np.real(np.max(tempRe));
				if tempRe[:].argmax() < len(tempIm):
					lambsolImMax[index] = tempIm[tempRe[:].argmax()]
			else:
				lambsolReMax[index] = 0.0;
				lambsolImMax[index] = 0.0;
			# print(alpha(Omega[index], tau[index], tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase, INV), tau[index], lambsolReMax[index])
			# print('sol=',linStabEq(np.array([lambsolReMax[index], lambsolImMax[index]]),Omega[index], tau[index], tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase, INV))
		################################################################################

		#Here we distiguish the stable from the unstable synchronised states
			init = initial_guess()
			if lambsolReMax[index] >= 0:
				OmegUnst.append(Omega[index]); tauUnst.append(tau[index]); betaUnst.append(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase, INV)['beta'])

			else:
				OmegStab.append(Omega[index]); tauStab.append(tau[index]); betaStab.append(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase, INV)['beta'])


	# print('\n\nlen(OmegStab):', len(OmegStab), '     len(OmegUnst):', len(OmegUnst),'\n')


	return {'Re': lambsolRe, 'Im': abs(lambsolIm),'betaUnst':betaUnst, 'betaStab': betaStab, 'OmegUnst': OmegUnst, 'tauUnst': tauUnst, 'OmegStab': OmegStab, 'tauStab': tauStab,'ReMax': lambsolReMax,'ImMax': abs(lambsolImMax)}








def solveLinStabSingle(Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase, expansion, INV, zeta):

# Here we solve the characteristic equation and we get the real and imaginary parts of the solutionsself.
# We calculate for different inital conditions, so we calculate the largest value of the real part of the solution in order to study the stability of the system.
##########################################################
	# lambsolRe = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	# lambsolIm = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)

	lambsolRe = np.zeros(len(initial_guess()),dtype=np.float64)
	lambsolIm = np.zeros(len(initial_guess()),dtype=np.float64)

##########################################################

	#
	lambsolReMax=[];
	lambsolImMax=[];

	# lambsolReMax = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	# lambsolImMax = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	init = initial_guess();
	init2 = initial_guess2();
	# zeta = -1.0#*1j;
	c 	 = 3E8;
	OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];betaUnst=[];betaStab=[];
	#print(type(init), init)
	#print(len(xrange(K)))
	if expansion:
		for index2 in range(len(init)):
			temp =  optimize.root(linStabEq_expansion, (init[index2],init2[index2]), args=(Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase, INV), tol=1e-14, method='hybr')
		# print('temp =',temp)
		# print('temp =',temp)
			if temp.success == True:
				lambsolRe[index2]  = temp.x[0]
				lambsolIm[index2]  = temp.x[1]

		tempRe = lambsolRe[np.round(lambsolRe[:],16)!=0.0];
		tempIm = lambsolIm[np.round(lambsolIm[:],16)!=0.0];

		if len(tempRe) != 0:
			#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
			# lambsolReMax = np.real(np.max(tempRe));
			lambsolReMax.append(np.real(np.max(tempRe)));
			if tempRe[:].argmax() < len(tempIm):
				# lambsolImMax = tempIm[tempRe[:].argmax()]
				lambsolImMax.append(tempIm[tempRe[:].argmax()])
		else:
			lambsolReMax.append(0.0);
			lambsolImMax.append(0.0);
		init = initial_guess()
	#print(type(init), init)
	#print('Re[lambda] =',lambsol)
		if lambsolRe[index2] >= 0:
			OmegUnst.append(Omega); tauUnst.append(tau); #betaUnst.append(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase, INV)['beta'])
		else:
			OmegStab.append(Omega); tauStab.append(tau);# betaStab.append(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase, INV)['beta'])

	else:
		# for index in range(len(xrange(maxp, tauf))):

		for index2 in range(len(init)):
			# print(init[1])
			temp = optimize.root(linStabEq, (init[index2],init2[index2]), args=( Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase, INV),  tol=1.0e-14, method='hybr')
			if temp.success == True:
				lambsolRe[index2]  = temp.x[0]
				lambsolIm[index2]  = temp.x[1]
		tempRe = lambsolRe[np.round(lambsolRe[:],16)!=0.0];
		tempIm = lambsolIm[np.round(lambsolIm[:],16)!=0.0];
		if len(tempRe) != 0:

			lambsolReMax.append(np.real(np.max(tempRe)));
			if tempRe[:].argmax() < len(tempIm):
				# lambsolImMax = tempIm[tempRe[:].argmax()]
				lambsolImMax.append(tempIm[tempRe[:].argmax()])
			else:
				# lambsolReMax.append(0.0);
				lambsolImMax.append(0.0);
		else:
			lambsolReMax.append(0.0);
			lambsolImMax.append(0.0);
		# print('sol=',linStabEq(np.array([lambsolReMax[index], lambsolImMax[index]]),Omega[index], tau[index], tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase, INV))
	################################################################################

	#Here we distiguish the stable from the unstable synchronised states
		init = initial_guess()
		if lambsolRe[index2] >= 0:
			OmegUnst.append(Omega); tauUnst.append(tau);#betaUnst.append(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase, INV)['beta'])
		else:
			OmegStab.append(Omega); tauStab.append(tau);# betaStab.append(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase, INV)['beta'])


	# print('\n\nlen(OmegStab):', len(OmegStab), '     len(OmegUnst):', len(OmegUnst),'\n')
	return {'Re': lambsolRe, 'Im': abs(lambsolIm),'betaUnst':betaUnst, 'betaStab': betaStab, 'OmegUnst': OmegUnst, 'tauUnst': tauUnst, 'OmegStab': OmegStab, 'tauStab': tauStab,'ReMax': lambsolReMax,'ImMax': np.abs(lambsolImMax)}














def xrangetau(maxtau):
	min_tau = 0.00000001;
	return np.linspace(min_tau, maxtau, 200)


def initial_guessbeta():
	# return (1.0*np.pi,1.00*np.pi);
	num_initial_guess=2
	return (np.linspace(0.0*np.pi,np.pi, num_initial_guess));

def initial_guessOmeg(w,Kvco):
	# return (1.0*np.pi,1.00*np.pi);
	num_initial_guess=2
	return (np.linspace(w-Kvco,w-Kvco, num_initial_guess));





def solverglobalfrequenc(x, w, tau, Kvco, AkPD, Ga1, tauf, v, digital, INV):
	if digital:
		return ( x[0] - w - K(Kvco, AkPD, Ga1)* cfDig( (-x[0] *tau -x[1])/v +INV), x[0]  - w - K(Kvco, AkPD, Ga1)* cfDig( (-x[0] *tau +x[1])/v +INV))
		# np.where((tau.any()>0.2 and tau.any()<0.25), print('tauIn=', tau,'OmIn=', Omega))
	else:
		return ( x[0] - w - K(Kvco, AkPD, Ga1)* cfAna( (-x[0] *tau -x[1])/v +INV), x[0]  - w - K(Kvco, AkPD, Ga1)* cfAna( (-x[0] *tau +x[1])/v +INV) )








def phase_diffsFullEquations(w, maxtau, Kvco, AkPD, Ga1, tauf, v, digital, INV):

	tau=xrangetau(maxtau);
	Omega = np.zeros(len(tau),dtype=np.float64);
	beta = np.zeros(len(tau),dtype=np.float64);

	# betaNum12vec=[];
	initbeta = initial_guessbeta();
	initOmeg= initial_guessOmeg(w,Kvco)
	for index in range(len(tau)):

		# for index2 in range(len(initbeta)):
		temp = ( optimize.root(solverglobalfrequenc, ( w-Kvco, np.pi), args=(w, tau[index], Kvco, AkPD, Ga1, tauf, v, digital, INV), tol=1.0e-10,method='hybr') )
		# print(root.x[0])
		# print(root.x[1])
		#
		if temp.success == True:
			# print(temp)
			# betaNumR1vec[index]=temp.x[0]
			# betaNumR2vec[index]=temp.x[1]
			Omega[index] = temp.x[0]
			beta[index]  = temp.x[1]
		# betaNumR1vec[index], betaNumR2vec[index] = ( fsolve(equationsDig, init, args=(tau, tauf, wR[index], w1, w2, Kvco, AkPD, Ga1, INV), xtol=1.0e-10, maxfev=5000) );

		# betaNumR1vec[index], betaNumR2vec[index] = ( fsolve(equationsDig, init,args=(tau, tauf, wR[index],w1, w2, Kvco, AkPD, Ga1, INV), xtol=1.0e-10, maxfev=5000) );
		#init = (betaNumR1vec[index], betaNumR2vec[index]);
		# init = initial_guessbeta1(INV, num_initial_guess2);
		# calculate phase-difference between PLL1 and PLL2

	# return {'betaR1vec': np.fmod(betaNumR1vec,2.0*np.pi), 'betaR2vec': np.fmod(betaNumR2vec,2.0*np.pi), 'beta12vec': np.fmod(betaNum12vec,2.0*np.pi)}


	return {'Omega': Omega, 'beta': beta}


def phase_diffsFullEquationsOne(w, tau, Kvco, AkPD, Ga1, tauf, v, digital, INV, initbeta, initOmeg):

	# tau=xrangetau(maxtau);
	# Omega = np.zeros(len(tau),dtype=np.float64);
	# beta = np.zeros(len(tau),dtype=np.float64);

	# betaNum12vec=[];
	# initbeta = initial_guessbeta();
	# initOmeg= initial_guessOmeg(w,Kvco)
	# for index in range(len(initial_guessbeta)):
	Omega=[]; beta=[];
		# for index2 in range(len(initbeta)):
	temp = ( optimize.root(solverglobalfrequenc, (initOmeg,initbeta), args=(w, tau, Kvco, AkPD, Ga1, tauf, v, digital, INV), tol=1.0e-10,method='hybr') )
	# print(root.x[0])
	# print(root.x[1])
	#
	if temp.success == True:
		# print(temp)
		# betaNumR1vec[index]=temp.x[0]
		# betaNumR2vec[index]=temp.x[1]
		Omega.append(temp.x[0])
		beta.append(temp.x[1])
	# betaNumR1vec[index], betaNumR2vec[index] = ( fsolve(equationsDig, init, args=(tau, tauf, wR[index], w1, w2, Kvco, AkPD, Ga1, INV), xtol=1.0e-10, maxfev=5000) );

		# betaNumR1vec[index], betaNumR2vec[index] = ( fsolve(equationsDig, init,args=(tau, tauf, wR[index],w1, w2, Kvco, AkPD, Ga1, INV), xtol=1.0e-10, maxfev=5000) );
		#init = (betaNumR1vec[index], betaNumR2vec[index]);
		# init = initial_guessbeta1(INV, num_initial_guess2);
		# calculate phase-difference between PLL1 and PLL2

	# return {'betaR1vec': np.fmod(betaNumR1vec,2.0*np.pi), 'betaR2vec': np.fmod(betaNumR2vec,2.0*np.pi), 'beta12vec': np.fmod(betaNum12vec,2.0*np.pi)}


	return {'Omega': Omega, 'beta': np.mod(beta,2.0*np.pi)}



def alpha_gen(Omega, tau, beta, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, INV):
	# if inphase:
	if digital:
		alpha = (K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-(Omega*(tau-tauf)-beta)/v+INV)
	else:
		alpha = (K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(-(Omega*(tau-tauf)-beta)/v+INV)
#
	# else:
	# 	if digital:
	# 		alpha = (K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v-np.pi/v+INV)
	# 	else:
	# 		alpha = (K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v-np.pi/v+INV)
	return alpha

def linStabEq_General(l_vec, Omega, tau, beta, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, INV):
	#This function gives the characteristic equation of the model
	#alpha = K/v*h'(Omega*(tau-tauf)/v)
	x = np.zeros(2)
	alphaparam=alpha_gen(Omega, tau, beta, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, INV)
	# if inphase:
	# 	if digital:
	# 		alpha = (K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+INV)
	# 	else:
	# 		alpha = (K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(Omega*(tau-tauf)/v+INV)
	#
	# else:
	# 	if digital:
	# 		alpha = (K(Kvco, AkPD, Ga1)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+np.pi+INV)
	# 	else:
	# 		alpha = (K(Kvco, AkPD, Ga1)/v)*cfAnaDeriv(Omega*(tau-tauf)/v-np.pi+INV)
	# print(alpha)
	l = l_vec[0] + 1j * l_vec[1]
	f = l*((1.0+l*tauc/order)**order)*np.exp(l*tauf) + alphaparam*(1.0-zeta*np.exp(-l*(tau-tauf)))
	#print(f)
	x[0] = np.real(f)
	x[1] = np.imag(f)

	#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x




def solveLinStabSingleGen(Omega, tau, beta, w, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, expansion, INV, zeta, initbeta, initOmeg):

# Here we solve the characteristic equation and we get the real and imaginary parts of the solutionsself.
# We calculate for different inital conditions, so we calculate the largest value of the real part of the solution in order to study the stability of the system.
##########################################################
	# lambsolRe = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	# lambsolIm = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)

	lambsolRe = np.zeros(len(initial_guess()),dtype=np.float64)
	lambsolIm = np.zeros(len(initial_guess()),dtype=np.float64)

##########################################################

	#
	lambsolReMax=[];
	lambsolImMax=[];

	# lambsolReMax = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	# lambsolImMax = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	init = initial_guess();
	init2 = initial_guess2();
	# zeta = -1.0#*1j;
	c 	 = 3E8;
	OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];betaUnst=[];betaStab=[];
	#print(type(init), init)
	#print(len(xrange(K)))
	if expansion:
		for index2 in range(len(init)):
			temp =  optimize.root(linStabEq_expansion, (init[index2],init2[index2]), args=(Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase, INV), tol=1e-14, method='hybr')
		# print('temp =',temp)
		# print('temp =',temp)
			if temp.success == True:
				lambsolRe[index2]  = temp.x[0]
				lambsolIm[index2]  = temp.x[1]

		tempRe = lambsolRe[np.round(lambsolRe[:],16)!=0.0];
		tempIm = lambsolIm[np.round(lambsolIm[:],16)!=0.0];

		if len(tempRe) != 0:
			#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
			# lambsolReMax = np.real(np.max(tempRe));
			lambsolReMax.append(np.real(np.max(tempRe)));
			if tempRe[:].argmax() < len(tempIm):
				# lambsolImMax = tempIm[tempRe[:].argmax()]
				lambsolImMax.append(tempIm[tempRe[:].argmax()])
		else:
			lambsolReMax.append(0.0);
			lambsolImMax.append(0.0);
		init = initial_guess()
	#print(type(init), init)
	#print('Re[lambda] =',lambsol)
		if lambsolRe[index2] >= 0:
			OmegUnst.append(Omega); tauUnst.append(tau); betaUnst.append(phase_diffsFullEquationsOne(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase, INV)['beta'])
		else:
			OmegStab.append(Omega); tauStab.append(tau); betaStab.append(phase_diffsFullEquationsOne(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase, INV)['beta'])

	else:
		# for index in range(len(xrange(maxp, tauf))):

		for index2 in range(len(init)):
			# print(init[1])
			temp = optimize.root(linStabEq_General, (init[index2],init2[index2]), args=( Omega, tau, beta, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital,  INV),  tol=1.0e-14, method='hybr')
			if temp.success == True:
				lambsolRe[index2]  = temp.x[0]
				lambsolIm[index2]  = temp.x[1]
		tempRe = lambsolRe[np.round(lambsolRe[:],16)!=0.0];
		tempIm = lambsolIm[np.round(lambsolIm[:],16)!=0.0];
		if len(tempRe) != 0:

			lambsolReMax.append(np.real(np.max(tempRe)));
			if tempRe[:].argmax() < len(tempIm):
				# lambsolImMax = tempIm[tempRe[:].argmax()]
				lambsolImMax.append(tempIm[tempRe[:].argmax()])
			else:
				# lambsolReMax.append(0.0);
				lambsolImMax.append(0.0);
		else:
			lambsolReMax.append(0.0);
			lambsolImMax.append(0.0);
		# print('sol=',linStabEq(np.array([lambsolReMax[index], lambsolImMax[index]]),Omega[index], tau[index], tauf, Kvco, AkPD, Ga1, tauc, v, zeta, order, digital, inphase, INV))
	################################################################################

	#Here we distiguish the stable from the unstable synchronised states
		init = initial_guess()
		if lambsolRe[index2] >= 0:
			OmegUnst.append(Omega); tauUnst.append(tau); betaUnst.append(phase_diffsFullEquationsOne(w, tau, Kvco, AkPD, Ga1, tauf, v, digital, INV, initbeta, initOmeg)['beta'])
		else:
			OmegStab.append(Omega); tauStab.append(tau); betaStab.append(phase_diffsFullEquationsOne(w, tau, Kvco, AkPD, Ga1, tauf, v, digital, INV, initbeta, initOmeg)['beta'])


	# print('\n\nlen(OmegStab):', len(OmegStab), '     len(OmegUnst):', len(OmegUnst),'\n')
	return {'Re': lambsolRe, 'Im': abs(lambsolIm),'betaUnst':betaUnst, 'betaStab': betaStab, 'OmegUnst': OmegUnst, 'tauUnst': tauUnst, 'OmegStab': OmegStab, 'tauStab': tauStab,'ReMax': lambsolReMax,'ImMax': np.abs(lambsolImMax)}





def equationSigma(tau, a, wc, zeta, gamma=0):
	''' USE for gammas without the condition
		np.abs( a*zeta*np.sin(gamma[i]*tau) ) <= gamma[i] and np.abs(gamma[i]) > zero_treshold '''
	# real zeta, approximation for small gamma*tau
	#sigma = -wc/2.0 + 1.0/tau * lambertw( -0.5*np.exp( 0.5*wc*tau )*wc*a*zeta*tau**2 )
	# real zeta, no approximation, need to calculate gamma assuming sigma=0, hence only valid for sigma << 1
	sigma = -wc/2.0 + 1.0/tau * lambertw( -(0.5/gamma)*np.exp( 0.5*wc*tau )*np.sin(gamma*tau)*wc*a*zeta*tau )
	# real zeta, approximation of setting sigma**2 to zero in real part of char. equation,
	# need to calculate gamma assuming sigma=0, hence only valid for sigma << 1
	#sigma = gamma**2/wc - a + 1.0/tau * lambertw( np.exp( -(gamma**2/wc - a)*tau )*zeta*np.cos(gamma*tau)*a*tau )


	return sigma


def equationGamma(tau, a, wc, zeta):

	zero_treshold = 1E-14

	#A = wc**2.0 - 2.0*np.abs(a)*wc
	#B = (1.0-zeta**2.0)*(np.abs(a)*wc)**2.0
	A = wc**2.0 - 2.0*a*wc
	B = (1.0-zeta**2.0)*(a*wc)**2.0
	# print('A:', A); print('B:', B);
	gamma = np.array([ +np.sqrt(0.5*(-A+A*np.sqrt(1.0-4.0*B/(A**2.0)))), +np.sqrt(0.5*(-A-A*np.sqrt(1.0-4.0*B/(A**2.0)))),
					   -np.sqrt(0.5*(-A+A*np.sqrt(1.0-4.0*B/(A**2.0)))), -np.sqrt(0.5*(-A-A*np.sqrt(1.0-4.0*B/(A**2.0)))) ])

	#if wc > 5:
	#	print('wc>5; gamma:', gamma)

	#if np.all(gamma[np.isfinite(gamma)] < -zero_treshold):
	#	print('WATCH OUT! :)')

	#gamma[np.isnan(gamma)] 		= 0.0
	#gamma[gamma < zero_treshold]= 0.0	#-.25

	if not ( wc/(2.0*np.abs(a))-1+np.sqrt(1.0-zeta**2.0) < 0 or wc/(2.0*np.abs(a))-1-np.sqrt(1.0-zeta**2.0) > 0 ):
		gamma[:] =None;

	for i in range(len(gamma)):													# this condition can also be checked in prepare2D
		if ( np.abs( a*zeta*np.sin(gamma[i]*tau) ) <= gamma[i] and np.abs(gamma[i]) > zero_treshold  ): #params[params['discrP']][i]*
			gamma[i] = None

	if a <= zero_treshold:
		gamma[:] = 0;	#-0.1;

	# print(gamma)
	return gamma
