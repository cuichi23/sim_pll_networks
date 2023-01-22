
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
from scipy.optimize import minimize
import time
digital =False
w    	  	= 2.0*np.pi*60E9; 			# intrinsic	frequency
Kvco      	= 2.0*np.pi*(100E6); 		# Sensitivity of VCO
AkPD	  	= 0.162*2.0					# Amplitude of the output of the PD --
GkLF		= 1.0
Gvga	  	= 0.5						# Gain of the first adder
tauf 	  	= 0.0						# tauf = sum of all processing delays in the feedback
order 	  	= 1.0						# the order of the Loop Filter
tauc	  	= 1.0/(2.0*np.pi*800E6); 	# the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
v	 	  	= 8.0;						# the division
c		  	= 3E8						# speed of light
min_delay 	= 0.1E-9
INV		  	= 0.0*np.pi					# Inverter
#			# Inverter
#
maxp 	  	= 10;
wnormy 		= 2.0*np.pi;				# this is the frequency with which we rescale the y-axis, choose either 1, 2pi or w (y-axis: radHz, Hz, dimless)
wnormx 		= 2.0*np.pi;				# this is the frequency with which we rescale the x-axis, choose either w/2pi-rescaling (2.0*np.pi) or no rescaling of tau (w)

filter		= 1
figwidth  	= 6;
figheight 	= 6;

Nx=3;
Ny=1;
start = time.time()
def xrange(maxp, tauf):
	dp		= 0.0015;
	return np.linspace(-tauf, maxp, (tauf+maxp)/dp)

# set initial guess
def initial_guess():
	return (1e7, 1e7);

def K(Kvco, AkPD, GkLF, Gvga):
	return (Kvco*AkPD*GkLF*Gvga)/2.0
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
def globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, INV):
	p = xrange(maxp, tauf)
	if digital:
		Omega = w + K(Kvco, AkPD, GkLF, Gvga) * cfDig( p+INV )
		tau	  = ( Omega * tauf + v * p ) / Omega
	else:
		Omega = w + K(Kvco, AkPD, GkLF, Gvga) * cfAna( -p+INV )
		tau	  = ( Omega * tauf + v * p ) / Omega
	return {'Omeg': Omega, 'tau': tau}

def linStabEq(l_vec, Omega, tau, tauf, Kvco, AkPD,  GkLF,Gvga, tauc, v, zeta, digital, INV):
	x = np.zeros(2)
	if digital:
		alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*cfDigDeriv(-Omega*(tau-tauf)/v+INV)
	else:
		alpha = (K(Kvco, AkPD, GkLF,Gvga)/v)*cfAnaDeriv(-Omega*(tau-tauf)/v+INV)

	l = l_vec[0] + 1j * l_vec[1]
	f = l*(1.0+l*tauc)*np.exp(l*tauf) + alpha*(1.0-zeta*np.exp(-l*(tau-tauf)))
	#print(f)
	x[0] = np.real(f)
	x[1] = np.imag(f)
	# #print('stability for tau=',tau,'s and Omega=',Omega,'Hz: ', x[0] ,'+ i', x[1])
	return x

def solveLinStab_topology_comparison(Omega, tau, tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, maxp, zetas, INV):
	# global x;
	# x = np.zeros(len(zetas))
	start = time.time()
	lambsolRe 	 = np.zeros([len(zetas),len(xrange(maxp, tauf))],dtype=np.float64)
	lambsolIm 	 = np.zeros([len(zetas),len(xrange(maxp, tauf))],dtype=np.float64)
	lambsolReMax = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	lambsolImMax = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	init = initial_guess();
	#print(type(init), init)
	#print(len(xrange(K)))
	for index1 in range(len(xrange(maxp, tauf))):
		for index2 in range(len(zetas)):
		# print(index, wR[index])
			# print(index, wR[index])
			temp = optimize.root(linStabEq, init, args=(Omega[index1], tau[index1], tauf, Kvco, AkPD,  GkLF,Gvga, tauc, v, zetas[index2], digital, INV), tol=1.49012e-08,  method='lm')
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
		# init = (np.real(lambsolReMax[index1]), np.real(lambsolImMax[index1]))
		init = initial_guess()
		#print(type(init), init)
		#print('Re[lambda] =',lambsol)
	end = time.time()
	print('calc2',end - start)
	return {'Re': lambsolRe, 'Im': lambsolIm, 'ReMax': lambsolReMax, 'ImMax': lambsolImMax, 'Omeg': Omega, 'tau': tau}
solveLinStab_topology_comparison(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, INV)['Omeg'], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, INV)['tau'], tauf, Kvco, AkPD,  GkLF, Gvga, tauc, v, digital, maxp, eigenvalzeta('chain',Nx,Ny)['zeta'], INV)['Re']
# init = initial_guess()
# nsolve = minimize(linStabEq, init, args=(globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, INV)['Omeg'][0], globalFreq(w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, INV)['tau'][0], tauf, Kvco, AkPD,  GkLF,Gvga, tauc, v, eigenvalzeta('chain',Nx,Ny)['zeta'], digital, INV), method='Nelder-Mead')
# print(nsolve)
