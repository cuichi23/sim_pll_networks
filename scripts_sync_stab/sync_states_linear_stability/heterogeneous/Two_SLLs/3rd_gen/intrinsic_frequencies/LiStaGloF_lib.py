#Linear Stability of Global Frequency (LiStaGloF.py)


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
from scipy.optimize import newton
import time

def xrange(maxp, tauf):
	dp = 20.5E-3;
	return np.linspace(-tauf, maxp, (tauf+maxp)/dp)

# set initial guess
def initial_guess():
	return (1.0e9,1e8);


def initial_guess2():
	return (np.linspace(-1.0e5, 2.0e5,5));

#
# def initial_guess():
# 	return (np.linspace(-1.0, 1.0, 10));
# def initial_guess():
# 	return (np.linspace(-1.0, 1.0, 10));
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

def globalFreqdigital(sol_vec, wmean, Kvco, AkPD, Ga1, tauf, v,  digital, maxp, Dw, inphase, tau):
	# p = xrange(maxp, tauf);
	Omega = sol_vec[0];
	beta  = sol_vec[1];
	x = np.zeros(2)
	if inphase:
		if digital:# K = Krange()

			f1 = - Omega + wmean+Dw/2.0 + K(Kvco, AkPD, Ga1)* cfDig( (-(Omega*(tau-tauf))-beta)/v)
			f2 = - Omega + wmean-Dw/2.0 + K(Kvco, AkPD, Ga1)* cfDig( (-(Omega*(tau-tauf))+beta)/v)
		else:
			f1 = - Omega + wmean+Dw/2.0 + K(Kvco, AkPD, Ga1)* cfAna( (-(Omega*(tau-tauf))-beta)/v)
			f2 = - Omega + wmean-Dw/2.0 + K(Kvco, AkPD, Ga1)* cfAna( (-(Omega*(tau-tauf))+beta)/v)

	else:
		if digital:
			f1 = - Omega + wmean+Dw/2.0 + (K(Kvco, AkPD, Ga1)/2.0)*(cfDig( (-(Omega*(tau-tauf))-beta)/v )+cfDig((-(Omega*(tau-tauf))-beta)/v) )
			f2 = - Omega + wmean-Dw/2.0 + (K(Kvco, AkPD, Ga1)/2.0)*(cfDig( (-(Omega*(tau-tauf))+beta)/v )+cfDig((-(Omega*(tau-tauf))+beta)/v) )

	x[0] = f1
	x[1] = f2

	return x

def solveglobalFreqdigital(wmean,  Kvco, AkPD, Ga1, tauf, v,  digital, maxp, Dw, inphase):

	init = initial_guess()
	print(init)
	Omega 	= np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	beta 	= np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	tau		= xrange(maxp, tauf)
	# print(tau)
	#print(type(init), init)
	#print(len(xrange(K)))
	# print('len(Oxrange(maxp, tauf)))',len(xrange(maxp, tauf)))
	for index in range(len(xrange(maxp, tauf))):
		# print('tau=',tau[index])
		temp = optimize.root(globalFreqdigital, init, args=(wmean, Kvco, AkPD, Ga1, tauf, v,  digital, maxp, Dw, inphase, tau[index]),  tol=1.364e-8, method='hybr')

		Omega[index]  	= temp.x[0]
		beta[index]		= temp.x[1]
		# lambsolRebetain[index]  = temp.x[0]
		# lambsolImbetain[index]  = temp.x[1]
		# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
		init = initial_guess()

		# print('\n\nlen(OmegStab):', len(OmegStab), '     len(OmegUnst):', len(OmegUnst),'\n')
	return {'Omega': Omega, 'Beta': beta}
# taubetain		= xrange(maxp, tauf)
# taubetain=0.01
# print('Omega=',solveglobalFreqdigitalin(wmean, tauf, Kvco, AkPD, Ga1, Dw,  v, digital, maxp, inphase1)['Omega'])
# print('Beta=',solveglobalFreqdigitalin(wmean, tauf, Kvco, AkPD, Ga1, Dw,  v, digital, maxp, inphase1)['Beta'])
#
#
# 	#*************************************************************************************************************************************************************************************
#

def globalFreq(wmean,  Kvco, AkPD, Ga1, tauf, v,  digital, maxp, Dw, inphase):
	p = xrange(maxp, tauf);
	Omegabetain=[]; Omegabetaanti=[]; tau=[]; taubetain=[]; taubetaanti=[]; betain=[]; betaanti=[]; pp=[];
	if inphase:
		if digital:
			print('Fix it.')
			# for value in temp:
				# # print('\n\n', value, type(value), '\n\n')
				# if value>0.0:
				# 	beta=??????( Dw/( 2.0*K*value) )
				# 	betain.append(beta)
				# elif value<0.0:
				# 	beta=2.0*np.pi-?????( Dw/( 2.0*K*value) )
				# 	betain.append(beta)
				# #Omegabetain= wmean + K * cfDig( -p )*np.cos( beta )
				# taubetain=( Omegabetain * tauf + p ) / Omegabetain
		else:
			for value in p:
				if   ((2.0*K(Kvco, AkPD, Ga1)*np.sin(value))**2 >= Dw**2) and np.sin(value) > 0:
					beta=v*np.arcsin(Dw/(2.0*K(Kvco, AkPD, Ga1)*np.sin(value)))
					betain.append(beta)
					pp.append(value)
					#print('YES_01')
				elif ((2.0*K(Kvco, AkPD, Ga1)*np.sin(value))**2 >= Dw**2) and np.sin(value) < 0:
					beta=v*2.0*np.pi+v*np.arcsin(Dw/(2.0*K(Kvco, AkPD, Ga1)*np.sin(value)))
					betain.append(beta)
					pp.append(value)
					#print('YES_02')

				#print('SHAPES:', np.shape(pp), type(pp), np.shape(betain))
				Omegabetain = wmean + K(Kvco, AkPD, Ga1) * cfAna( -np.array(pp) ) * cfAna( np.array(betain)/v )
				taubetain   = ( np.array(Omegabetain) * tauf + v * np.array(pp) ) / np.array(Omegabetain)


	else:
		if digital:
			for value in temp:
				if value>0.0:
					print('Fix it.')
				# 	beta=np.pi-np.arcsin??????(Dw/(2.0*K*value))
				# 	betaanti.append(beta)
				# elif value<0.0:
				# 	betaanti=np.pi+np.arcsin??????(Dw/(2.0*K*value))
				# 	betaanti.append(beta)
				# Omegabetaanti = wmean + K * cfAna( -p )*np.cos(beta)
				# taubetaanti	  = ( Omegabetaanti * tauf + p ) / Omegabetaanti
		else:
			for value in p:
				if   ((2.0*K(Kvco, AkPD, Ga1)*np.sin(value))**2 >= Dw**2) and np.sin(value) > 0:
					beta=v*np.pi-v*np.arcsin(Dw/(2.0*K(Kvco, AkPD, Ga1)*np.sin(value)))
					betaanti.append(beta)
					pp.append(value)
					#print('YES_01')
				elif ((2.0*K(Kvco, AkPD, Ga1)*np.sin(value))**2 >= Dw**2) and np.sin(value) < 0:
					beta=v*np.pi-v*np.arcsin(Dw/(2.0*K(Kvco, AkPD, Ga1)*np.sin(value)))
					betaanti.append(beta)
					pp.append(value)
					#print('YES_02')

				#print('SHAPES:', np.shape(pp), type(pp), np.shape(betaanti))
				Omegabetaanti = wmean + K(Kvco, AkPD, Ga1) * cfAna( -np.array(pp) ) * cfAna( np.array(betaanti)/v )
				taubetaanti   = ( np.array(Omegabetaanti) * tauf + v * np.array(pp) ) / np.array(Omegabetaanti)

	return {'Omegabetaanti': Omegabetaanti, 'Omegabetain': Omegabetain, 'betain': betain,'betaanti': betaanti, 'taubetain': taubetain,'taubetaanti': taubetaanti}


def linStabEq(l_vec, Omega, tau, tauf,  Kvco, AkPD, Ga1, tauc, Dw, beta, order,v, digital):
	x = np.zeros(2)
	# print(Dw)
	# print('beta=',beta,'inside arcsin',Dw/(2.0*K*np.sin(Omega*tau)),'condition=',(np.sin(Omega*tau))*(np.sin(Omega*tau)),'condition2=',(Dw/(2.0*K))*(Dw/(2.0*K)),'condition3=',np.sin(Omega*tau))
	if digital:
		alpha12 = (K/v)*cfDigDeriv((-Omega*tau+beta)/v)
		alpha21 = (K/v)*cfDigDeriv((-Omega*tau-beta)/v)
	else:
		alpha12 = ( K(Kvco, AkPD, Ga1)/v ) * cfAnaDeriv( (-Omega*tau+beta)/v )
		alpha21 = ( K(Kvco, AkPD, Ga1)/v ) * cfAnaDeriv( (-Omega*tau-beta)/v )
	# print('alpha12=',alpha12,'alpha21=', alpha21)
	l = l_vec[0] + 1j * l_vec[1]
	f = l*l*((1.0+l*tauc/order)**order)*((1.0+l*tauc/order)**order)+(alpha12+alpha21)*l*((1.0+l*tauc/order)**order)-alpha12*alpha21*(np.exp(-2.0*l*tau)-1.0)#*(1.0+l*tauc)+(alpha12+alpha21)*l*(1.0+l*tauc)#
	#print(f)
	x[0] = np.real(f)
	x[1] = np.imag(f)

	# print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x


def linStabEq_expansion(l_vec, Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, Dw, beta,v, digital):
	x = np.zeros(2)
	if digital:
		alpha = (K/v)*cfDigDeriv((-Omega*(tau-tauf))/v)
	else:
		alpha = (K/v)*cfAnaDeriv((-Omega*(tau-tauf))/v)

	l = l_vec[0] + 1j * l_vec[1]
	f = l*l*(1.0+l*tauc)-alpha12*alpha21*(np.exp(-2.0*l*tau)-1.0)
	#print(f)
	x[0] = np.real(f)
	x[1] = np.imag(f)
	# print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x


def solveLinStabbetain(Omegabetain, taubetain, tauf, Kvco, AkPD, Ga1, tauc, Dw, betain,v, order, digital, maxp, expansion):

	init = initial_guess()
	init2 = initial_guess2()
	print(init)
	lambsolRebetain = np.zeros([len(betain),len(betain)],dtype=np.float64)
	lambsolImbetain = np.zeros([len(betain),len(betain)],dtype=np.float64)
	lambsolRebetainMax = np.zeros(len(betain),dtype=np.float64)
	lambsolImbetainMax = np.zeros(len(betain),dtype=np.float64)
	# print(len(lambsolImbetainMax))
	c    = 3E8;
	OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];

	#print(type(init), init)
	#print(len(xrange(K)))
	# print('len(Oxrange(maxp, tauf)))',len(xrange(maxp, tauf)))
	if expansion:
		for index in range(len(betain)):

			print(index, betain[index])
		# 	temp =  optimize.root(linStabEq_expansion, init, args=(Omega[index], tau[index], tauf, K, tauc, Dw, beta, digital),  tol=1.364e-8, method='hybr')
		# # print('temp =',temp)
		# # print('temp =',temp)
		# 	lambsolRe[index] = np.real(temp.x[0])
		# 	lambsolIm[index] = np.real(temp.x[1])
		# # print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
		# 	#init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))
		# 	init = initial_guess();
		# #print(type(init), init)
		# #print('Re[lambda] =',lambsol)
		# 	if lambsolRe[index] >= 0:
		# 		OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
		# 	else:
		# 		OmegStab.append(Omega[index]); tauStab.append(tau[index]);
	else:
		for index in range(len(betain)):
			for index2 in range(len(init)):
				# print(init[1])
				temp = optimize.root(linStabEq, (init2[index2],init[index2]), args=( Omegabetain[index], taubetain[index], tauf, Kvco, AkPD, Ga1, tauc, Dw, betain[index], v, order, digital),  tol=1.364e-8, method='hybr')
				lambsolRebetain[index2,index]  = temp.x[0]
				lambsolImbetain[index2,index]  = temp.x[1]
				# lambsolRebetain[index]  = temp.x[0]
				# lambsolImbetain[index]  = temp.x[1]
				# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
				# init = (np.real(lambsolRebetain[index]), np.real(lambsolImbetain[index]))

			tempRe = lambsolRebetain[np.round(lambsolRebetain[:,index],12)!=0,index];
			tempIm = lambsolImbetain[np.round(lambsolImbetain[:,index],32)!=0,index];
			# print(len(tempRe),'\n\n',tempIm)
			if len(tempRe) != 0:
				#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
				lambsolRebetainMax[index] = np.real(np.max(tempRe));
				if tempRe[:].argmax() < len(tempIm):
					lambsolImbetainMax[index] = tempIm[tempRe[:].argmax()]
			else:
				lambsolRebetainMax[index] = 0.0;
				lambsolImbetainMax[index] = 0.0;

	# print('\n\nlen(OmegStab):', len(OmegStab), '     len(OmegUnst):', len(OmegUnst),'\n')
	return {'Rebetain': lambsolRebetain, 'Imbetain': lambsolImbetain,'RebetainMax': lambsolRebetainMax,'ImbetainMax':lambsolImbetainMax}

def solveLinStabbetaanti(Omegabetaanti, taubetaanti, tauf, Kvco, AkPD, Ga1, tauc, Dw, betaanti, v, order, digital, maxp, expansion):
	init = initial_guess()
	init2 = initial_guess2()
	print(init)
	lambsolRebetaanti = np.zeros([len(betaanti),len(betaanti)],dtype=np.float64)
	lambsolImbetaanti = np.zeros([len(betaanti),len(betaanti)],dtype=np.float64)
	lambsolRebetaantiMax = np.zeros(len(betaanti),dtype=np.float64)
	lambsolImbetaantiMax = np.zeros(len(betaanti),dtype=np.float64)

	c    = 3E8;
	OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];

	#print(type(init), init)
	#print(len(xrange(K)))
	# print('len(Oxrange(maxp, tauf)))',len(xrange(maxp, tauf)))
	if expansion:
		for index in range(len(betaanti)):

			print(index, betaanti[index])
		# 	temp =  optimize.root(linStabEq_expansion, init, args=(Omega[index], tau[index], tauf, K, tauc, Dw, beta, digital),  tol=1.364e-8, method='hybr')
		# # print('temp =',temp)
		# # print('temp =',temp)
		# 	lambsolRe[index] = np.real(temp.x[0])
		# 	lambsolIm[index] = np.real(temp.x[1])
		# # print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
		# 	#init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))
		# 	init = initial_guess();
		# #print(type(init), init)
		# #print('Re[lambda] =',lambsol)
		# 	if lambsolRe[index] >= 0:
		# 		OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
		# 	else:
		# 		OmegStab.append(Omega[index]); tauStab.append(tau[index]);
	else:
		for index in range(len(betaanti)):
			for index2 in range(len(init)):
				# print(init[1])
				temp = optimize.root(linStabEq, (init2[index2],init[index2]), args=( Omegabetaanti[index], taubetaanti[index], tauf, Kvco, AkPD, Ga1, tauc, Dw, betaanti[index], v, order, digital),  tol=1.364e-8, method='hybr')
				lambsolRebetaanti[index2,index]  = temp.x[0]
				lambsolImbetaanti[index2,index]  = temp.x[1]
				# lambsolRebetain[index]  = temp.x[0]
				# lambsolImbetain[index]  = temp.x[1]
				# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
				# init = (np.real(lambsolRebetain[index]), np.real(lambsolImbetain[index]))

			tempRe = lambsolRebetaanti[np.round(lambsolRebetaanti[:,index],12)!=0,index];
			tempIm = lambsolImbetaanti[np.round(lambsolImbetaanti[:,index],32)!=0,index];
			if len(tempRe) != 0:

				lambsolRebetaantiMax[index] = np.real(np.max(tempRe));
				if tempRe[:].argmax() < len(tempIm):
					lambsolImbetaantiMax[index] = tempIm[tempRe[:].argmax()]
			else:
				lambsolRebetaantiMax[index] = 0.0;
				lambsolImbetaantiMax[index] = 0.0;


			# if lambsolRe[index] >= 0:
			# 	OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
			# else:
			# 	OmegStab.append(Omega[index]); tauStab.append(tau[index]);
			# print('len(lambsolRebetain))',len(lambsolRebetain))
			# print(type(lambsolRebetain),type(lambsolImbetain))
	# print('\n\nlen(OmegStab):', len(OmegStab), '     len(OmegUnst):', len(OmegUnst),'\n')
	return {'Rebetaanti': lambsolRebetaanti, 'Imbetaanti': lambsolImbetaanti,'RebetaantiMax': lambsolRebetaantiMax,'ImbetaantiMax':lambsolImbetaantiMax}
	#
