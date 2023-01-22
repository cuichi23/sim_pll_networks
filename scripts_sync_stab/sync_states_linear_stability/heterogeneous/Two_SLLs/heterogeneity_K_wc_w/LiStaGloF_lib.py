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

def xrange(maxp, tauf, dp=0.01):
	return np.linspace(1E-9, maxp, int((tauf+maxp)/dp))

# set initial guess
def initial_guess():
	return (np.linspace(-1e-3, 1e4, 15));

def initial_guess2():
	return (np.linspace(-0.28*2.0*np.pi, 0.70*2.0*np.pi, 15));

# def initial_guess():
# 	return (np.linspace(-1e-1,1e0, 55));
#
# def initial_guess2(tauc):
# 	wc=1.0/tauc
# 	return (np.linspace(-2.0*wc, 2.0*wc, 55));

def K(Kvco, AkPD, GkLF, Gvga):
	return (Kvco*AkPD*GkLF*Gvga)/2.0

# digital case
cfDig        = lambda x: sawtooth(x,width=0.5)
cfDigDeriv   = lambda x: (2.0/np.pi)*square(x,duty=0.5)
# analog case
cfAna        = lambda x: np.cos(x);
cfAnaDeriv   = lambda x: -1.0*np.sin(x);

def cfDigInverse(x):
	if np.abs(x)>1:
		print('Error! Inverse fct. of triangular wave called with argument out of bounds.'); exit();
	# 	return -np.pi/2.0*x-np.pi/2.0;
	# else:
	return +np.pi/2.0*x+np.pi/2.0;

def cfAnaInverse(x, slope):
	if slope==1:
		return -np.arccos(x);
	if slope==2:
		return np.arccos(x);

def solNumericalOmegTau(x, ....):
	x[0] =

	return

def globalFreq(wmean, Dw, Kmean, DK, tauf,  digital, maxp, inphase):
	p = xrange(maxp, tauf);
	Omegabetain=[]; Omegabetaanti=[]; tau=[]; taubetain=[]; taubetaanti=[]; betain=[]; betaanti=[]; pp=[];

	if digital:
		print('Digital case: using max(tau) = maxp parameter.')
		for value in p:
			for index in range(len(init)):

			temp = optimize.root(solNumericalOmegTau, (init[0,index], init[1,index]), args=( Omegabetain[index], taubetain[index], tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, betain[index], digital),  tol=1.0e-11, method='hybr')


	else:
		if inphase:
			for value in p:
				H=np.sqrt( ( 2.0*Kmean*np.sin(value) )**2 + ( DK*np.cos(value) )**2   )
				if abs(Dw /( H ))<=1.0 and abs( DK*np.cos(value)/H )<=1.0:

					B= np.arcsin( Dw/H ) + np.arcsin( DK*np.cos(value)/H  )

					if   ((2.0*Kmean*np.sin(value))**2 >= Dw**2) and np.sin(value) > 0:

						beta= np.arcsin( Dw /( H ) )+ np.arcsin(DK*np.cos( value )/H )
						betain.append(beta)
						pp.append(value)

					elif ((2.0*Kmean*np.sin(value))**2 >= Dw**2) and np.sin(value) < 0:

						beta=2.0*np.pi - ( np.arcsin( Dw /( H ) ) + np.arcsin(DK*np.cos( value )/H ) )
						betain.append(beta)
						pp.append(value)
						#print('YES_02')

					Omegabetain = wmean + Kmean * np.cos( np.array(pp) ) * np.cos( np.array(betain) )- 0.5*DK*np.sin(pp)*np.sin(B)
					taubetain = ( np.array(Omegabetain) * tauf + np.array(pp) ) / np.array(Omegabetain)

		elif not inphase:
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


def linStabEq_expansion(l_vec, Omega, tau, tauf, K, tauc, Dw, beta, digital):
	x = np.zeros(2)
	if digital:
		alpha = (K)*cfDigDeriv(-Omega*(tau-tauf))
	else:
		alpha = (K)*cfAnaDeriv(-Omega*(tau-tauf))

	l = l_vec[0] + 1j * l_vec[1]
	f = l*l*(1.0+l*tauc)-alpha12*alpha21*(np.exp(-2.0*l*tau)-1.0)
	#print(f)
	x[0] = np.real(f)
	x[1] = np.imag(f)
	# print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x


def solveLinStabbetain(Omegabetain, taubetain, tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, betain, digital, maxp, expansion):
	init = initial_guess()
	# print(init)
	lambsolRebetain = np.zeros([len(init),len(betain)],dtype=np.float64)
	lambsolImbetain = np.zeros([len(init),len(betain)],dtype=np.float64)
	# lambsolRebetainMax = np.zeros(len(betain),dtype=np.float64)
	# lambsolImbetainMax = np.zeros(len(betain),dtype=np.float64)
	lambsolRebetainMax =[];
	lambsolImbetainMax =[];
	# print(len(lambsolImbetainMax))
	c    = 3E8;
	Omega = []; OmegStab = []; tau = []; tauStab = [];

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

				temp = optimize.root(linStabEq, (init[index2],init[index2]), args=( Omegabetain[index], taubetain[index], tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, betain[index], digital),  tol=1.0e-11, method='hybr')
				if temp.success == True:
					lambsolRebetain[index2,index]  = temp.x[0]
					lambsolImbetain[index2,index]  = temp.x[1]
				# lambsolRebetain[index]  = temp.x[0]
				# lambsolImbetain[index]  = temp.x[1]
				# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
				# init = (np.real(lambsolRebetain[index]), np.real(lambsolImbetain[index]))

			tempRe = lambsolRebetain[np.round(lambsolRebetain[:,index],8)!=0,index];
			tempIm = lambsolImbetain[np.round(lambsolImbetain[:,index],32)!=0,index];
			# print(len(tempRe),'\n\n',tempIm)
			if len(tempRe) != 0:
				# print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
				lambsolRebetainMax.append( np.real(np.max(tempRe)) );

				# else:
				if tempRe[:].argmax() < len(tempIm):
					lambsolImbetainMax.append( tempIm[tempRe[:].argmax()])
			else:
				lambsolRebetainMax.append(990.0);
				lambsolImbetainMax.append(990.0);


	# print('\n\nlen(OmegStab):', len(OmegStab), '     len(OmegUnst):', len(OmegUnst),'\n')
	return {'Rebetain': lambsolRebetain, 'Imbetain': lambsolImbetain,'RebetainMax': lambsolRebetainMax,'ImbetainMax':lambsolImbetainMax, 'tau':tau, 'Omega':Omega }


def solveLinStabbetaanti(Omegabetaanti, taubetaanti, tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga,  tauc1, tauc2, Dw, betaanti, digital, maxp, expansion):
	init = initial_guess()
	# print(init)
	lambsolRebetaanti = np.zeros([len(init),len(betaanti)],dtype=np.float64)
	lambsolImbetaanti = np.zeros([len(init),len(betaanti)],dtype=np.float64)
	lambsolRebetaantiMax = np.zeros(len(betaanti),dtype=np.float64)
	lambsolImbetaantiMax = np.zeros(len(betaanti),dtype=np.float64)

	c    = 3E8;
	Omega = []; OmegStab = []; tau = []; tauStab = [];

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

				temp = optimize.root(linStabEq, (init[index2],init[index2]), args=( Omegabetaanti[index], taubetaanti[index], tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, betaanti[index], digital),  tol=1.364e-8, method='hybr')
				if temp.success == True:
					lambsolRebetaanti[index2,index]  = temp.x[0]
					lambsolImbetaanti[index2,index]  = temp.x[1]
				# lambsolRebetain[index]  = temp.x[0]
				# lambsolImbetain[index]  = temp.x[1]
				# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
				# init = (np.real(lambsolRebetain[index]), np.real(lambsolImbetain[index]))

			tempRe = lambsolRebetaanti[np.round(lambsolRebetaanti[:,index],12)!=0,index];
			tempIm = lambsolImbetaanti[np.round(lambsolImbetaanti[:,index],32)!=0,index];
			if len(tempRe) != 0:
				#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
				lambsolRebetaantiMax[index] = np.real(np.max(tempRe));
				if tempRe[:].argmax() < len(tempIm):
					lambsolImbetainMax[index] = tempIm[tempRe[:].argmax()]
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

def solveLinStabSingle(Omega, tau, tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, beta, digital, maxp, expansion):
	# Here we solve the characteristic equation and we get the real and imaginary parts of the solutionsself.
	# We calculate for different inital conditions, so we calculate the largest value of the real part of the solution in order to study the stability of the system.
	##########################################################
	init  = initial_guess()
	init2 = initial_guess2()
	# print(init)
	# lambsolRebetain = np.zeros([len(betain),len(betain)],dtype=np.float64)
	# lambsolImbetain = np.zeros([len(betain),len(betain)],dtype=np.float64)
	# lambsolRebetainMax = np.zeros(len(betain),dtype=np.float64)
	# lambsolImbetainMax = np.zeros(len(betain),dtype=np.float64)

	lambsolRe = np.zeros(len(initial_guess()),dtype=np.float64)
	lambsolIm = np.zeros(len(initial_guess()),dtype=np.float64)

	##########################################################



	lambsolReMax=[];
	lambsolImMax=[];
	OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];betaUnst=[];betaStab=[];

	#print(type(init), init)
	#print(len(xrange(K)))
	# print('len(Oxrange(maxp, tauf)))',len(xrange(maxp, tauf)))
	if expansion:
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

		for index2 in range(len(init)):
			# print(init[1])
			temp = optimize.root(linStabEq, (init[index2], init2[index2]), args=( Omega, tau, tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, beta, digital),  tol=1.364e-12, method='hybr')
			if temp.success == True:
				lambsolRe[index2]  = temp.x[0]
				lambsolIm[index2]  = temp.x[1]
			# else:
			# 	lambsolRe[index2]=0.0;
			# 	lambsolIm[index2]=0.0

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
				OmegUnst.append(Omega); tauUnst.append(tau); betaUnst.append(beta)
			else:
				OmegStab.append(Omega); tauStab.append(tau); betaStab.append(beta)


	# 	# lambsolRebetain[index]  = temp.x[0]
	# 	# lambsolImbetain[index]  = temp.x[1]
	# 	# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
	# 	# init = (np.real(lambsolRebetain[index]), np.real(lambsolImbetain[index]))
	#
	# tempRe = lambsolRebetain[np.round(lambsolRebetain[:,index],12)!=0,index];
	# tempIm = lambsolImbetain[np.round(lambsolImbetain[:,index],32)!=0,index];
	# # print(len(tempRe),'\n\n',tempIm)
	# if len(tempRe) != 0:
	# 	# print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
	# 	lambsolRebetainMax[index] = np.real(np.max(tempRe));
	# 	lambsolImbetainMax[index] = tempIm[tempRe[:].argmax()]
	# else:
	# 	lambsolRebetainMax[index] = 0.0;
	# 	lambsolImbetainMax[index] = 0.0;
	# print('\n\nlen(OmegStab):', len(OmegStab), '     len(OmegUnst):', len(OmegUnst),'\n')
	return {'Re': lambsolRe, 'Im': lambsolIm, 'ReMax': lambsolReMax, 'ImMax':lambsolImMax, 'OmegStab': OmegStab, 'OmegUnst': OmegUnst, 'tauUnst': tauUnst, 'tauStab': tauStab, 'betaUnst': betaUnst, 'betaStab': betaStab}
