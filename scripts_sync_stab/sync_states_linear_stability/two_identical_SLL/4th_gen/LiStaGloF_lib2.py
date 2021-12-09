from numpy import pi, sin
import numpy as np
import sympy
from sympy import solve, nroots, I
from sympy.abc import q
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.signal import sawtooth
from scipy.signal import square
from scipy import optimize
import scipy.optimize as optimize
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import root_scalar
import time
test_solution_charEq 		= True;
num_sol_treshold	 		= 1e-7;
global num_initial_guess;
num_initial_guess = 2.0;

def xrange(maxp, tauf):
	dp		= 0.0125;
	initial = 3.99E3
	return np.linspace(initial, maxp, int((-initial+maxp)/dp))

def initial_guess():
	return 1e5;

def K(Kvco, AkPD, GkLF, Gvga):
	return (Kvco*AkPD*GkLF*Gvga)/2.0

def filterRC(l, tauc, order):
	return (1.0+l*tauc)**order

def filterChe(l, tauc, order):
	return (1.0+l*tauc/order)**order

def filterLeadLag(l, tauc, order):
	# tauc=
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

def globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, inphase,INV):
	#Here we calculate analyticaly the Global Frequency of the synchronised states of the network of two SLLs.
	#First of all, there are two synchronised states, the in-phase and anti-phase. So, the inphase refers to that.
	#we distiguish the analog and the digital case of the SLLs. So, the digital refers to that.
	#The choice of digital and inphase mode is made in the main programs.
	p = xrange(maxp, tauf)				 #p=Omega*(tau-tauf)
	# print( K(Kvco, AkPD, Gk, Ak, Gl,Al, GkLF,Gvga))
	if inphase:
		if digital:
			Omega = w + K(Kvco, AkPD,  GkLF,Gvga)* cfDig( p + INV )
			tau	  = ( Omega * tauf + v * p ) / Omega
			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tauIn=', tau,'OmIn=', Omega))

		else:
			Omega = w + K(Kvco, AkPD,  GkLF,Gvga) * cfAna( -p + INV )
			tau	  = ( Omega * tauf + v * p ) / Omega
			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tau=',tau,'Om=', Omega))

	else:
		if digital:
			Omega = w + (K(Kvco, AkPD,  GkLF,Gvga)/2.0)*(cfDig( p + np.pi+INV)+cfDig(p-np.pi+INV) )
			tau	  = ( Omega * tauf + v * p ) / Omega
			# np.where((tau.any()>0.2 and tau.any()<0.25), print('tauAn=', tau,'OmAn=', Omega))
			# if (p>=0.15 and p<=0.22):
			# 	print('tau=',tau,'Omega=',Omega)
		else:
			Omega = w - K(Kvco, AkPD,GkLF,Gvga) * cfAna( -p + INV)
			tau	  = ( Omega * tauf + v * p ) / Omega

	return {'Omeg': Omega, 'tau': tau}


def LoopGain(Omega, tau, tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase, INV):
	#This function gives the characteristic equation of the model
	#alpha = K/v*h'(Omega*(tau-tauf)/v)
	if inphase:
		if digital:
			alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+INV)
		else:
			alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v+INV)

	else:
		if digital:
			alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+np.pi+INV)
		else:
			alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v-np.pi+INV)

	#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return {'LPGain':alpha}



def LoopBDWidth(Omega, tau, tauf, alpha, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase, INV):
	#This function gives the characteristic equation of the model
	#alpha = K/v*h'(Omega*(tau-tauf)/v)
	LoopBand1= np.sqrt( ( -1.0 + np.sqrt( 1.0 + 4.0 *( alpha**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )
	LoopBand2=-np.sqrt( ( -1.0 + np.sqrt( 1.0 + 4.0 *( alpha**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )
	LoopBand3= np.sqrt( ( -1.0 - np.sqrt( 1.0 + 4.0 *( alpha**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )
	LoopBand4=-np.sqrt( ( -1.0 - np.sqrt( 1.0 + 4.0 *( alpha**2 )*( tauc**2 ) ) ) /( 2.0*tauc**2 ) )

	return {'BDWidth1':LoopBand1, 'BDWidth2':LoopBand2, 'BDWidth3':LoopBand3, 'BDWidth4':LoopBand4}


def test_solution(LoopBand,sigma, Omega, tau, tauf, alpha, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase, INV):
	results = [];
	for i in range(len(LoopBand)):
		diff = LoopBDWidthGeneral(LoopBand[i],sigma, Omega, tau, tauf, alpha, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase, INV)
		# print(diff)
		results.append(np.sqrt(diff**2))
	return results


def LoopBDWidthGeneral(x, sigma, Omega, tau, tauf, alpha, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase, INV):
	#This function gives the characteristic equation of the model
	#alpha = K/v*h'(Omega*(tau-tauf)/v)
	gamma=x
	# sigma=0.0
	Bhta= sigma + sigma**2.0 - (gamma**2.0)*tauc
	Gam = gamma*(sigma*tauc + 1.0)
	denominator = Bhta**2.0 + Gam**2.0
	Alp=alpha*np.exp(-sigma*(tau+tauf) )

	# f= ( (Alp*Bhta*np.cos(tau*gamma) + Alp*Gam*np.sin(tauc*gamma) )/ denominator)**2 + ( ( Alp*Gam*np.cos(tau*gamma) + Alp*Bhta*np.sin(tau*gamma))/denominator)**2 -1.0
	f=( (Alp**2)*(Bhta**2.0+Gam**2.0+2.0*Bhta*Gam*np.sin(2.0*(tau+tauf)*gamma) )/denominator**2.0  )-1.0
	return f




def solveLoopBandwidth(sigma, Omega, tau, tauf, alpha, Kvco, AkPD,  GkLF, Gvga, tauc, v, maxp, digital, inphase, INV):
	# print(LoopBDWidthGeneral(sigma, Omega, tau, tauf, alpha, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase, INV))
	# Here we solve the characteristic equation and we get the real and imaginary parts of the solutionsself.
	# We calculate for different inital conditions, so we calculate the largest value of the real part of the solution in order to study the stability of the system.
	##########################################################
	LoopBand = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	init = initial_guess();
	# sigma=0.0
	c 	 = 3E8;
	# LoopBand=[];
	for index in range(len(xrange(maxp, tauf))):

		temp = optimize.fsolve(LoopBDWidthGeneral, init, args=(sigma[index], Omega[index], tau[index], tauf, alpha[index], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase, INV), xtol=1e-10)#, method='newton')
		# optim =optimize.OptimizeResult(temp)
		# print('fR=',wR[index]/(2*np.pi))
		print(temp)
		LoopBand[index]=temp
		# print(np.isclose(LoopBDWidthGeneral(temp, sigma[index], Omega[index], tau[index], tauf, alpha[index], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase, INV), 0.0))
		if test_solution_charEq == True:
			temp = test_solution(LoopBand,  sigma[index], Omega[index], tau[index], tauf, alpha[index], Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, inphase, INV)
			if any(temp[:]) > num_sol_treshold:
				print('Problem with numerical solution, results vector:', temp[index], r' Ωτ/2π',  Omega[index]*tau[index]/(2.0*np.pi) )#, ',', w, ',', tau, ',', tauf,  ',', tauc, ',', digital, ',', slope)

	# print('temp =',temp)
		#  = temp.gamma
		# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
		# init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))
		init = initial_guess()
		# time.sleep(0.01)

	# print('\n\nlen(OmegStab):', len(OmegStab), '     len(OmegUnst):', len(OmegUnst),'\n')
	return {'LoopBand': LoopBand}


def test_solution2(lambsolIm, Omega, tau, tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, zeta, order, digital, inphase, INV, filter):
	results = [];
	for i in range(len(lambsolIm)):
		diff = linStaBandw(lambsolIm[i],Omega, tau, tauf, Kvco, AkPD,  GkLF,Gvga, tauc, v, zeta, order, digital, inphase, INV, filter)
		# print(diff)
		results.append(np.sqrt(diff**2))
	return results


def linStaBandw(l_vec, Omega, tau, tauf, Kvco, AkPD,  GkLF,Gvga, tauc, v, zeta, order, digital, inphase, INV, filter):
	#This function gives the characteristic equation of the model
	#alpha = K/v*h'(Omega*(tau-tauf)/v)
	# x = np.zeros(2)
	if inphase:
		if digital:
			alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+INV)
		else:
			alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v+INV)

	else:
		if digital:
			alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+np.pi+INV)
		else:
			alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v-np.pi+INV)

	l = 0.0+1j * l_vec[0]
	# l_vec[0] = 0.0
	if filter == 1:
		f = l*(filterChe(l, tauc, order) )*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)) )
	elif filter ==2:
		f = l*(filterRC(l, tauc, order) )*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)) )
	elif filter ==3:
		f = l*(filterLeadLag(l, tauc, order) )*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)) )
	# f = l*((1.0+l*tauc/order)**order)*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)))
	# print(f)
	# print(l)
	# x[0] = np.real(f)
	# x[1] = np.imag(f)
	g=np.sqrt(abs(f**2))
	# print(g)
	#print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return g


def solveLinStabBandw(Omega, tau, tauf, Kvco, AkPD,  GkLF,Gvga, tauc, v, order, digital, maxp, inphase, expansion, INV, filter):

	# Here we solve the characteristic equation and we get the real and imaginary parts of the solutionsself.
	# We calculate for different inital conditions, so we calculate the largest value of the real part of the solution in order to study the stability of the system.
	##########################################################
	lambsolIm =	np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)

	init = initial_guess();
	zeta = -1;
	c 	 = 3E8;
	# OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];
	#print(type(init), init)
	#print(len(xrange(K)))
	# linStaBandw(lambsolIm, Omega[:], tau[:], tauf, Kvco, AkPD,  GkLF,Gvga, tauc, v, zeta, order, digital, inphase, INV, filter).shape == lambsolIm.shape
	# func(x0,y[:,newaxis],z[:,newaxis]).shape == x0.shape

	for index in range(len(xrange(maxp, tauf))):

		temp = optimize.root(linStaBandw, init, args=(Omega[index], tau[index], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, zeta, order, digital, inphase, INV, filter), tol=1e-12, method='hybr')
		# print('temp =',temp)

		lambsolIm[index] = temp.x
		# if test_solution_charEq == True:
		# 	temp1 = test_solution2	(lambsolIm, Omega[index], tau[index], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, zeta, order, digital, inphase, INV, filter)
		# 	# print(temp1)
		# 	if any(temp1[:]) > num_sol_treshold:
		# 		print('Problem with numerical solution, results vector:', temp1[index], r' Ωτ/2π',  Omega[index]*tau[index]/(2.0*np.pi) )
		# optim =optimize.OptimizeResult(temp)
		# print('fR=',wR[index]/(2*np.pi))
		# print(optim)
	# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
		# init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))
		init = initial_guess()
		# if lambsolRe[index] >= 0:
		# 	OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
		# else:
		# 	OmegStab.append(Omega[index]); tauStab.append(tau[index]);

	# print('\n\nlen(OmegStab):', len(OmegStab), '     len(OmegUnst):', len(OmegUnst),'\n')
	return { 'Im': np.abs(lambsolIm)}
