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


coupfun='cos'

def coupfununction(coupfun):
	return coupfun

def xrange(maxp, tauf):
	dp = 0.125;
	return np.linspace(tauf, maxp, int((tauf+maxp)/dp))

# set initial guess
def initial_guess():
	return (np.linspace(-4e-2, 4e-2, 10));
def initial_guess():
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

if coupfun=='cos':
	cfAna        = lambda x: np.cos(x);
	# def cfAnaInverse(x):
	# 	if np.abs(x)>1:
	# 		print('Error! Inverse fct. of arccos wave called with argument out of bounds.'); exit();
	# 	return -np.arccos(x);
	cfAnaDeriv   = lambda x: -1.0*np.sin(x);



def globalFreq(wmean, K, tauf,  digital, maxp, Dw, inphase):
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
				if   ((2.0*K*np.sin(value))**2 >= Dw**2) and np.sin(value) > 0:
					beta=np.arcsin(Dw/(2.0*K*np.sin(value)))
					betain.append(beta)
					pp.append(value)
					#print('YES_01')
				elif ((2.0*K*np.sin(value))**2 >= Dw**2) and np.sin(value) < 0:
					beta=2.0*np.pi+np.arcsin(Dw/(2.0*K*np.sin(value)))
					betain.append(beta)
					pp.append(value)
					#print('YES_02')

				#print('SHAPES:', np.shape(pp), type(pp), np.shape(betain))
				Omegabetain = wmean + K * cfAna( -np.array(pp) ) * cfAna( np.array(betain) )
				taubetain = ( np.array(Omegabetain) * tauf + np.array(pp) ) / np.array(Omegabetain)


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
				if   ((2.0*K*np.sin(value))**2 >= Dw**2) and np.sin(value) > 0:
					beta=np.pi-np.arcsin(Dw/(2.0*K*np.sin(value)))
					betaanti.append(beta)
					pp.append(value)
					#print('YES_01')
				elif ((2.0*K*np.sin(value))**2 >= Dw**2) and np.sin(value) < 0:
					beta=np.pi-np.arcsin(Dw/(2.0*K*np.sin(value)))
					betaanti.append(beta)
					pp.append(value)
					#print('YES_02')

				#print('SHAPES:', np.shape(pp), type(pp), np.shape(betaanti))
				Omegabetaanti = wmean + K * cfAna( -np.array(pp) ) * cfAna( np.array(betaanti) )
				taubetaanti = ( np.array(Omegabetaanti) * tauf + np.array(pp) ) / np.array(Omegabetaanti)

	return {'Omegabetaanti': Omegabetaanti, 'Omegabetain': Omegabetain, 'betain': betain,'betaanti': betaanti, 'taubetain': taubetain,'taubetaanti': taubetaanti}


def linStabEq(l_vec, Omega, tau, tauf, K, tauc, Dw, beta, digital):
	x = np.zeros(2)
	# print(Dw)
	# print('beta=',beta,'inside arcsin',Dw/(2.0*K*np.sin(Omega*tau)),'condition=',(np.sin(Omega*tau))*(np.sin(Omega*tau)),'condition2=',(Dw/(2.0*K))*(Dw/(2.0*K)),'condition3=',np.sin(Omega*tau))
	if digital:
		alpha12 = K*cfDigDeriv(-Omega*tau+beta)
		alpha21 = K*cfDigDeriv(-Omega*tau-beta)
	else:
		alpha12 = K*cfAnaDeriv(-Omega*tau+beta)
		alpha21 = K*cfAnaDeriv(-Omega*tau-beta)
	# print('alpha12=',alpha12,'alpha21=', alpha21)
	l = l_vec[0] + 1j * l_vec[1]
	f = l*l*(1.0+l*tauc)*(1.0+l*tauc)+(alpha12+alpha21)*l*(1.0+l*tauc)-alpha12*alpha21*(np.exp(-2.0*l*tau)-1.0)#*(1.0+l*tauc)+(alpha12+alpha21)*l*(1.0+l*tauc)#
	#print(f)
	x[0] = np.real(f)
	x[1] = np.imag(f)

	# print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x


def solveLinStabbetain(Omegabetain, taubetain, tauf, K, tauc, Dw, betain, digital, maxp):

	init = initial_guess()
	# print(init)
	lambsolRebetain = np.zeros([len(init),len(betain)],dtype=np.float64)
	lambsolImbetain = np.zeros([len(init),len(betain)],dtype=np.float64)
	lambsolRebetainMax = np.zeros(len(betain),dtype=np.float64)
	lambsolImbetainMax = np.zeros(len(betain),dtype=np.float64)
	# print(len(lambsolImbetainMax))
	c    = 3E8;
	OmegUnst = []; OmegStab = []; tauinUnst = []; tauinStab = []; betainUnst=[]; betainStab=[];
	# lambsolRebetainMax =[];
	# lambsolImbetainMax =[];
	#print(type(init), init)
	#print(len(xrange(K)))
	# print('len(Oxrange(maxp, tauf)))',len(xrange(maxp, tauf)))

	for index in range(len(betain)):
		for index2 in range(len(init)):
			# print(init[1])
			temp = optimize.root(linStabEq, (init[index2],init[index2]), args=( Omegabetain[index], taubetain[index], tauf, K, tauc, Dw, betain[index], digital),  tol=1.0e-12, method='hybr')
			if temp.success==True:
				lambsolRebetain[index2,index]  = temp.x[0]
				lambsolImbetain[index2,index]  = temp.x[1]
			# lambsolRebetain[index]  = temp.x[0]
			# lambsolImbetain[index]  = temp.x[1]
			# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
			# init = (np.real(lambsolRebetain[index]), np.real(lambsolImbetain[index]))
		tempRe = lambsolRebetain[np.round(lambsolRebetain[:,index],16)!=0,index];
		tempIm = lambsolImbetain[np.round(lambsolImbetain[:,index],16)!=0,index];
		# print(len(tempRe),'\n\n',tempIm)
		if len(tempRe) != 0:
			# print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
			lambsolRebetainMax[index] = np.real(np.max(tempRe));
			# lambsolRebetainMax.append(np.real(np.max(tempRe)));
			if tempRe[:].argmax() < len(tempIm):
				# lambsolImMax.append(tempIm[tempRe[:].argmax()])
				lambsolImbetainMax[index] = tempIm[tempRe[:].argmax()]
				# lambsolImbetainMax.append(tempIm[tempRe[:].argmax()])
		else:
			lambsolRebetainMax[index] = 0.0;
			lambsolImbetainMax[index] = 0.0;
			# lambsolRebetainMax.append(0.0);
			# lambsolImbetainMax.append(0.0);

		if lambsolRebetainMax[index] >= 0:
			betainUnst.append(betain[index]); tauinUnst.append(taubetain[index]);
		else:
			betainStab.append(betain[index]); tauinStab.append(taubetain[index]);


	# print('\n\nlen(OmegStab):', len(OmegStab), '     len(OmegUnst):', len(OmegUnst),'\n')
	return {'Rebetain': lambsolRebetain, 'Imbetain': lambsolImbetain,'RebetainMax': lambsolRebetainMax,'ImbetainMax':lambsolImbetainMax, 'betainUnst': betainUnst, 'betainStab': betainStab, 'tauinUnst': tauinUnst, 'tauinStab': tauinStab}

def solveLinStabbetaanti(Omegabetaanti, taubetaanti, tauf, K, tauc, Dw, betaanti, digital, maxp, expansion):
	init = initial_guess()
	# print(init)
	lambsolRebetaanti = np.zeros([len(betaanti),len(betaanti)],dtype=np.float64)
	lambsolImbetaanti = np.zeros([len(betaanti),len(betaanti)],dtype=np.float64)
	lambsolRebetaantiMax = np.zeros(len(betaanti),dtype=np.float64)
	lambsolImbetaantiMax = np.zeros(len(betaanti),dtype=np.float64)

	c    = 3E8;
	OmegUnst = []; OmegStab = []; tauantiUnst = []; tauantiStab = [];betaantiStab=[];betaantiUnst=[];

	#print(type(init), init)
	#print(len(xrange(K)))

	for index in range(len(betaanti)):
		for index2 in range(len(init)):
			# print(init[1])
			temp = optimize.root(linStabEq, (init[index2],init[index2]), args=( Omegabetaanti[index], taubetaanti[index], tauf, K, tauc, Dw, betaanti[index], digital),  tol=1.0e-12, method='hybr')
			if temp.success==True:
				lambsolRebetaanti[index2,index]  = temp.x[0]
				lambsolImbetaanti[index2,index]  = temp.x[1]
			# lambsolRebetain[index]  = temp.x[0]
			# lambsolImbetain[index]  = temp.x[1]
			# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
			# init = (np.real(lambsolRebetain[index]), np.real(lambsolImbetain[index]))

		tempRe = lambsolRebetaanti[np.round(lambsolRebetaanti[:,index],16)!=0,index];
		tempIm = lambsolImbetaanti[np.round(lambsolImbetaanti[:,index],16)!=0,index];
		if len(tempRe) != 0:
			#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
			lambsolRebetaantiMax[index] = np.real(np.max(tempRe));
			if tempRe[:].argmax() < len(tempIm):
				# lambsolImMax.append(tempIm[tempRe[:].argmax()])
				lambsolImbetaantiMax[index] = tempIm[tempRe[:].argmax()]
			# lambsolImbetaantiMax[index] = tempIm[tempRe[:].argmax()]
		else:
			lambsolRebetaantiMax[index] = 0.0;
			lambsolImbetaantiMax[index] = 0.0;

		if lambsolRebetaantiMax[index] >= 0:
			betaantiUnst.append(betaanti[index]); tauantiUnst.append(taubetaanti[index]);
		else:
			betaantiStab.append(betaanti[index]); tauantiStab.append(taubetaanti[index]);

		# if lambsolRe[index] >= 0:
			# 	OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
			# else:
			# 	OmegStab.append(Omega[index]); tauStab.append(tau[index]);
			# print('len(lambsolRebetain))',len(lambsolRebetain))
			# print(type(lambsolRebetain),type(lambsolImbetain))
	# print('\n\nlen(OmegStab):', len(OmegStab), '     len(OmegUnst):', len(OmegUnst),'\n')
	return {'Rebetaanti': lambsolRebetaanti, 'Imbetaanti': lambsolImbetaanti,'RebetaantiMax': lambsolRebetaantiMax,'ImbetaantiMax':lambsolImbetaantiMax, 'betaantiUnst': betaantiUnst, 'betaantiStab': betaantiStab, 'tauantiUnst': tauantiUnst, 'tauantiStab': tauantiStab  }
	#
