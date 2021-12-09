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
from scipy.special import lambertw
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
def xrange(maxp, tauf, dp=0.05):
	return np.linspace(1E-9, maxp, int((tauf+maxp)/dp))

def K(Kvco, AkPD, GkLF, Gvga):
	return (Kvco*AkPD*GkLF*Gvga)/2.0
# set initial guess
# def initial_guess():
# 	return (3.101, 1.01);
def initial_guess():
	return (np.linspace(-1e-3, 1e2, 55));

def initial_guess2():
	return (np.linspace(-0.28*2.0*np.pi, 0.70*2.0*np.pi, 55));
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



# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## # # ## ## ## ## ## ## ## ## ## ## ## #  # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
#													Linear Stability Analysis
#
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## # # ## ## ## ## ## ## ## ## ## ## ## # # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #






def globalFreq(wmean, Dw, Kmean, DK, tauf,  digital, maxp, sync_state):
	p = xrange(maxp, tauf);
	Omegabetain=[]; Omegabetaanti=[]; tau=[]; taubetain=[]; taubetaanti=[]; betain=[]; betaanti=[]; pp=[];

	if sync_state == 'inphase':
		if digital:
			print('Fix it.')
		else:
			for value in p:
				H=np.sqrt( ( 2.0*Kmean*np.sin(value) )**2 + ( DK*np.cos(value) )**2   )
				if abs(Dw /( H ))<=1.0 and abs( DK*np.cos(value)/H )<=1.0:

					B= np.arcsin( Dw/H ) + np.arcsin( DK*np.cos(value)/H  )

					if   ((2.0*Kmean*np.sin(value))**2 >= Dw**2) and np.sin(value) > 0:

						beta= np.arcsin( Dw /( H ) )+ np.arcsin(DK*np.cos( value )/H )
						betain.append(beta)
						pp.append(value)

						#print('YES_01')
					elif ((2.0*Kmean*np.sin(value))**2 >= Dw**2) and np.sin(value) < 0:

						beta=2.0*np.pi - ( np.arcsin( Dw /( H ) ) + np.arcsin(DK*np.cos( value )/H ) )
						betain.append(beta)
						pp.append(value)
						#print('YES_02')

					Omegabetain = wmean + Kmean * np.cos( np.array(pp) ) * np.cos( np.array(betain) )- 0.5*DK*np.sin(pp)*np.sin(B)
					taubetain = ( np.array(Omegabetain) * tauf + np.array(pp) ) / np.array(Omegabetain)

	elif sync_state == 'antiphase':
		if digital:
			print('Fix it.')

		else:
			for value in p:
				H=np.sqrt( ( 2.0*Kmean*np.sin(value) )**2 + ( DK*np.cos(value) )**2   )

				if abs(Dw /( H )) <= 1.0 and abs( DK*np.cos(value)/H ) <=1.0:
					B=np.arcsin( Dw/H ) + np.arcsin( DK*np.cos(value)/H  )

					if   ((2.0*Kmean*np.sin(value))**2 >= Dw**2) and np.sin(value) > 0:
						# print(B)
						beta= np.pi - ( np.arcsin( Dw /( H ) )- np.arcsin(DK*np.cos( value )/H ) )
						betaanti.append(beta)
						pp.append(value)
						#print('YES_01')
					elif ((2.0*Kmean*np.sin(value))**2 >= Dw**2) and np.sin(value) < 0:

						beta= np.pi + (np.arcsin( Dw /( H ) )- np.arcsin(DK*np.cos( value )/H ) )
						betaanti.append(beta)
						pp.append(value)
						#print('YES_02')

					#print('SHAPES:', np.shape(pp), type(pp), np.shape(betaanti))
					Omegabetaanti = wmean + Kmean * np.cos( np.array(pp) ) * np.cos( np.array(betaanti) ) - 0.5*DK*np.sin(pp)*np.sin(B)
					taubetaanti = ( np.array(Omegabetaanti) * tauf + np.array(pp) ) / np.array(Omegabetaanti)

	return {'Omegabetaanti': Omegabetaanti, 'Omegabetain': Omegabetain, 'betain': betain,'betaanti': betaanti, 'taubetain': taubetain,'taubetaanti': taubetaanti}


def linStabEq(l_vec, Omega, tau, tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, beta, digital):
	x = np.zeros(2)
	# print(Dw)
	# print('beta=',beta,'inside arcsin',Dw/(2.0*K*np.sin(Omega*tau)),'condition=',(np.sin(Omega*tau))*(np.sin(Omega*tau)),'condition2=',(Dw/(2.0*K))*(Dw/(2.0*K)),'condition3=',np.sin(Omega*tau))

	if digital:
		alpha12 = K(Kvco1, AkPD, GkLF, Gvga)*cfDigDeriv(-Omega*tau+beta)
		alpha21 = K(Kvco2, AkPD, GkLF, Gvga)*cfDigDeriv(-Omega*tau-beta)
	else:
		alpha12 = K(Kvco1, AkPD, GkLF, Gvga)*cfAnaDeriv(-Omega*tau+beta)
		alpha21 = K(Kvco2, AkPD, GkLF, Gvga)*cfAnaDeriv(-Omega*tau-beta)
	# print('alpha12=',alpha12,'alpha21=', alpha21)
	l = l_vec[0] + 1j * l_vec[1]
	f = l*l*(1.0+l*tauc1)*(1.0+l*tauc2)*np.exp(2.0*l*tauf)+alpha21*l*np.exp(l*tauf)*(1.0+l*tauc1)+alpha12*l*(1.0+l*tauc2)*np.exp(l*tauf)-alpha12*alpha21*(np.exp(-2.0*l*(tau-tauf) )-1.0 )
	x[0] = np.real(f)
	x[1] = np.imag(f)

	# print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x


def solveLinStabbetain(Omegabetain, taubetain, tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, betain, digital, maxp):

	init = initial_guess()
	init2 = initial_guess2()
	# print(init)
	lambsolRebetain = np.zeros([len(init),len(betain)],dtype=np.float64)
	lambsolImbetain = np.zeros([len(init),len(betain)],dtype=np.float64)
	lambsolRebetainMax = np.zeros(len(betain),dtype=np.float64)
	lambsolImbetainMax = np.zeros(len(betain),dtype=np.float64)
	# print(len(lambsolImbetainMax))
	# lambsolRebetainMax = [];
	# lambsolImbetainMax = [];
	c    = 3E8;
	OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];

	#print(type(init), init)
	#print(len(xrange(K)))
	# print('len(Oxrange(maxp, tauf)))',len(xrange(maxp, tauf)))

	for index in range(len(betain)):
		for index2 in range(len(init)):
			# print(init[1])
			temp = optimize.root(linStabEq, (init[index2],init2[index2]), args=( Omegabetain[index], taubetain[index], tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, betain[index], digital),  tol=1.364e-8, method='hybr')
			if temp.success == True:
				lambsolRebetain[index2,index]  = temp.x[0]
				lambsolImbetain[index2,index]  = temp.x[1]
		tempRe = lambsolRebetain[np.round(lambsolRebetain[:,index],16)!=0,index];
		tempIm = lambsolImbetain[np.round(lambsolImbetain[:,index],16)!=0,index];
		if len(tempRe) != 0:
			# print(index)
			# print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
			lambsolRebetainMax[index] = np.real(np.max(tempRe));
			if tempRe[:].argmax() < len(tempIm):
				lambsolImbetainMax[index] = tempIm[tempRe[:].argmax()]
			# if tempRe[:].argmax() < len(tempIm):
			# lambsolImbetainMax[index] = tempIm[tempRe[:].argmax()]
		else:
			lambsolRebetainMax[index] = 0.0;
			lambsolImbetainMax[index] = 0.0;


	# print('\n\nlen(OmegStab):', len(OmegStab), '     len(OmegUnst):', len(OmegUnst),'\n')
	return {'Rebetain': lambsolRebetain, 'Imbetain': lambsolImbetain,'RebetainMax': lambsolRebetainMax,'ImbetainMax':lambsolImbetainMax}


def solveLinStabbetaanti(Omegabetaanti, taubetaanti, tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga,  tauc1, tauc2, Dw, betaanti, digital, maxp):
	init = initial_guess()
	init2 = initial_guess2()
	# print(init)
	lambsolRebetaanti = np.zeros([len(init),len(betaanti)],dtype=np.float64)
	lambsolImbetaanti = np.zeros([len(init),len(betaanti)],dtype=np.float64)
	lambsolRebetaantiMax = np.zeros(len(betaanti),dtype=np.float64)
	lambsolImbetaantiMax = np.zeros(len(betaanti),dtype=np.float64)

	c    = 3E8;
	OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];

	#print(type(init), init)
	#print(len(xrange(K)))
	# print('len(Oxrange(maxp, tauf)))',len(xrange(maxp, tauf)))

	for index in range(len(betaanti)):
		for index2 in range(len(init)):
			# print(init[1])
			temp = optimize.root(linStabEq, (init[index2],init2[index2]), args=( Omegabetaanti[index], taubetaanti[index], tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, betaanti[index], digital),  tol=1.364e-8, method='hybr')
			if temp.success == True:
				lambsolRebetaanti[index2,index]  = temp.x[0]
				lambsolImbetaanti[index2,index]  = temp.x[1]

		tempRe = lambsolRebetaanti[np.round(lambsolRebetaanti[:,index],12)!=0,index];
		tempIm = lambsolImbetaanti[np.round(lambsolImbetaanti[:,index],32)!=0,index];
		if len(tempRe) != 0:
			#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
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

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## # # ## ## ## ## ## ## ## ## ## ## ## #  # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
#													Transfer functions
#
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## # # ## ## ## ## ## ## ## ## ## ## ## # # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #






# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #      PLL1        # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #




def LoopGainSteadyState1(Omega, tau, beta, tauf, v, Kvco1, AkPD, GkLF, Gvga, digital, model):
	if model=='Nonlinear':
		if digital:
			alpha = (K(Kvco1, AkPD, GkLF, Gvga)/v)*cfDigDeriv(-( Omega*(tau-tauf) + beta) / v)
		else:
			alpha = (K(Kvco1, AkPD, GkLF, Gvga)/v)*cfAnaDeriv(-( Omega*(tau-tauf) + beta) /v)
	elif model=='Linear':
		alpha = (K(Kvco1, AkPD, GkLF, Gvga)/v)
	return alpha;

def HopenLoopMutuallyCoupledOnePLL1(gamma, Omega, tau, beta, tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga, digital, model):
	return ( LoopGainSteadyState1(Omega, tau, beta, tauf, v, Kvco1, AkPD, GkLF, Gvga, digital, model)/( 1j*gamma*( 1.0 + 1j*gamma*tauc1 ) ) )

def PhaseopenloopMutuallyCoupledOnePLL1(gamma, Omega, tau, beta, tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga, digital, model):
	phase =	 cmath.phase(HopenLoopMutuallyCoupledOnePLL1(gamma, Omega, tau, beta, tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga,  digital, model))
	return 360.0*phase/(2.0*np.pi)

def HclosedLoopMutuallyCoupledOnePLL1(gamma, Omega, tau, beta, tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga, digital, model):
	# return HopenLoopMutuallyCoupledOnePLL(gamma, Omega, tau, beta, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital, model)/( 1.0 + HopenLoopMutuallyCoupledOnePLL(gamma, Omega, tau, beta, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital, model)*np.exp(1j*gamma*( -tauf ) ) )
	return 1.0 / ( 1.0/HopenLoopMutuallyCoupledOnePLL1(gamma, Omega, tau, beta, tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga, digital, model) + np.exp(1j*gamma*( -tauf ) ) )
	# return ((alpha*np.exp(1j*w*(-tauf))/(1j*w*(1+1j*w*tauc)))/(1.0+(alpha/(1j*w*(1+1j*w*tauc)))))

def PhaseclosedloopMutuallyCoupledOnePLL1(gamma, Omega, tau, beta, tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga,  digital, model):
	phase =	 cmath.phase(HclosedLoopMutuallyCoupledOnePLL1(gamma, Omega, tau, beta, tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga,  digital, model))
	return 360.0*phase/(2.0*np.pi)


def GainMarginMutuallyCoupledOne1(gamma, Omega, tau, beta, tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga, digital, model):

	temp = HopenLoopMutuallyCoupledOnePLL1(gamma, Omega, tau, beta, tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga, digital, model)
	realpart = temp.real
	imagpart = temp.imag
	# print(realpart,imagpart)
	return 1.0-np.abs(np.sqrt(realpart**2+imagpart**2))

def PhaseMarginMutuallyCoupledOne1(gamma, Omega, tau, beta, tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga, digital, model):

	temp = HopenLoopMutuallyCoupledOnePLL1(gamma, Omega, tau, beta, tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga, digital, model)
	realpart = temp.real
	imagpart = temp.imag

	if realpart>0.0:
		return np.pi-np.arctan(imagpart/realpart)
	if realpart<0.0:
		return np.arctan(imagpart/realpart)

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #      PLL2        # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
def LoopGainSteadyState2(Omega, tau, beta, tauf, v, Kvco2, AkPD, GkLF, Gvga, digital, model):
	if model=='Nonlinear':
		if digital:
			alpha = (K(Kvco2, AkPD, GkLF, Gvga)/v )*cfDigDeriv(-( Omega*(tau-tauf) - beta) / v)
		else:
			alpha = (K(Kvco2, AkPD, GkLF, Gvga)/v)*cfAnaDeriv(-(Omega*(tau-tauf) - beta) /v)
	elif model=='Linear':

		alpha = (K(Kvco2, AkPD, GkLF, Gvga)/v)
	return alpha;



def HopenLoopMutuallyCoupledOnePLL2(gamma, Omega, tau, beta, tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga, digital, model):
	return ( LoopGainSteadyState2(Omega, tau, beta, tauf, v, Kvco2, AkPD, GkLF, Gvga, digital, model)/( 1j*gamma*( 1.0 + 1j*gamma*tauc2 ) ) )

def PhaseopenloopMutuallyCoupledOnePLL2(gamma, Omega, tau, beta, tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga, digital, model):
	phase =	 cmath.phase(HopenLoopMutuallyCoupledOnePLL2(gamma, Omega, tau, beta, tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga,  digital, model))
	return 360.0*phase/(2.0*np.pi)

def HclosedLoopMutuallyCoupledOnePLL2(gamma, Omega, tau, beta, tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga, digital, model):
	# return HopenLoopMutuallyCoupledOnePLL(gamma, Omega, tau, beta, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital, model)/( 1.0 + HopenLoopMutuallyCoupledOnePLL(gamma, Omega, tau, beta, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga, digital, model)*np.exp(1j*gamma*( -tauf ) ) )
	return 1.0 / ( 1.0/HopenLoopMutuallyCoupledOnePLL2(gamma, Omega, tau, beta, tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga, digital, model) + np.exp(1j*gamma*( -tauf ) ) )
	# return ((alpha*np.exp(1j*w*(-tauf))/(1j*w*(1+1j*w*tauc)))/(1.0+(alpha/(1j*w*(1+1j*w*tauc)))))

def PhaseclosedloopMutuallyCoupledOnePLL2(gamma, Omega, tau, beta, tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga,  digital, model):
	phase =	 cmath.phase(HclosedLoopMutuallyCoupledOnePLL2(gamma, Omega, tau, beta, tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga,  digital, model))
	return 360.0*phase/(2.0*np.pi)

def GainMarginMutuallyCoupledOne2(gamma, Omega, tau, beta, tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga, digital, model):

	temp = HopenLoopMutuallyCoupledOnePLL2(gamma, Omega, tau, beta, tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga, digital, model)
	realpart = temp.real
	imagpart = temp.imag
	# print(realpart,imagpart)
	return 1.0-np.abs(np.sqrt(realpart**2+imagpart**2))

def PhaseMarginMutuallyCoupledOne2(gamma, Omega, tau, beta, tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga, digital, model):

	temp = HopenLoopMutuallyCoupledOnePLL2(gamma, Omega, tau, beta, tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga, digital, model)
	realpart = temp.real
	imagpart = temp.imag

	if realpart>0.0:
		return np.pi-np.arctan(imagpart/realpart)
	if realpart<0.0:
		return np.arctan(imagpart/realpart)



# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #      Network        # ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #

def HopenLoopMutuallyCoupledNet(gamma, Omega, tau, beta, tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, model):
	return (np.exp(-2.0*1j*gamma*tau)*(HclosedLoopMutuallyCoupledOnePLL1(gamma, Omega, tau, beta, tauf1, tauc1, v, Kvco1, AkPD, GkLF, Gvga,  digital, model)*HclosedLoopMutuallyCoupledOnePLL2(gamma, Omega, tau, beta, tauf2, tauc2, v, Kvco2, AkPD, GkLF, Gvga,  digital, model)) )

	# return (np.exp(-2*1j*w*tau)*((alpha*np.exp(1j*w*(-tauf))/(1j*w*((1+1j*w*tauc)) ))/(1.0+(alpha*np.exp(1j*w*(-tauf))/(1j*w*((1+1j*w*tauc)) ))))**2)

def PhaseopenloopMutuallyCoupledNet(gamma, Omega, tau, beta, tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, model):
	phase =	 cmath.phase(HopenLoopMutuallyCoupledNet(gamma, Omega, tau, beta, tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, model))
	return 360.0*phase/(2.0*np.pi)

def HclosedloopMutuallyCoupledNet(gamma, Omega, tau, beta, tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, model):
	# return ( (np.exp(2*1j*w*(-tau))*((alpha*np.exp(1j*w*(-tauf))/(1j*w*((1+1j*w*tauc)**2)))/(1.0+(alpha/(1j*w*((1+1j*w*tauc)**2) ))))**2)/(1.0- (np.exp(2*1j*w*(-tauf))*((alpha*np.exp(1j*w*(-tauf))/(1j*w*(1+1j*w*tauc)))/(1.0+(alpha/(1j*w*(1+1j*w*tauc)))))**2) )   )
	return 1.0 / ( 1.0/HopenLoopMutuallyCoupledNet(gamma, Omega, tau, beta, tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, model) - 1.0)

def PhaseclosedloopMutuallyCoupledNet(gamma, Omega, tau, beta, tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, model):
	phase =	 cmath.phase(HopenLoopMutuallyCoupledNet(gamma, Omega, tau, beta, tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, model))
	return 360.0*phase/(2.0*np.pi)

def GainMarginMutuallyCoupledNet(gamma, Omega, tau, beta, tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2, AkPD, GkLF, Gvga,  digital, model):

	temp = HopenLoopMutuallyCoupledNet(gamma, Omega, tau, beta, tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, model)
	realpart = temp.real
	imagpart = temp.imag
	return 1.0-np.abs(np.sqrt(realpart**2+imagpart**2))

def PhaseMarginMutuallyCoupledNet(gamma, Omega, tau, beta, tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, model):

	temp = HopenLoopMutuallyCoupledNet(gamma, Omega, tau, beta, tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, model)
	realpart = temp.real
	imagpart = temp.imag
	if realpart>0.0:
		return np.arctan(imagpart/realpart)
	if realpart<0.0:
		return np.pi+np.arctan(imagpart/realpart)












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

	# if not ( wc/(2.0*np.abs(a))-1+np.sqrt(1.0-zeta**2.0) < 0 or wc/(2.0*np.abs(a))-1-np.sqrt(1.0-zeta**2.0) > 0 ):
	# 	gamma[:] =None;
	#
	# for i in range(len(gamma)):													# this condition can also be checked in prepare2D
	# 	if ( np.abs( a*zeta*np.sin(gamma[i]*tau) ) <= gamma[i] and np.abs(gamma[i]) > zero_treshold  ): #params[params['discrP']][i]*
	# 		gamma[i] = None

	if a <= zero_treshold:
		gamma[:] = 0;	#-0.1;

	# print(gamma)
	return gamma
