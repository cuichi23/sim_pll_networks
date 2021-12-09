#Linear Stability of Global Frequency (LiStaGoF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
from numpy import pi, sin, cos, arccos, arcsin
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
import time
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
	dp		= 1.482978723;
	return np.linspace(-tauf, maxp, (tauf+maxp)/dp)
# def Krange():
# 	Ga1=np.linspace(0.01, 16.0, 40)
# 	return Ga1;

def Krange():
	dk		= 0.018297872;
	return (np.linspace(0.01, 0.850, (0.01+0.850)/dk))


def taucrange():
	dg		= 0.000000008/47;
	taucmin=0.005*1.0/(2.0*np.pi*120E6);
	taucmax=1.0/(2.0*np.pi*120E6);
	return (np.linspace(taucmin,taucmax, (taucmin+taucmax)/dg))


# set initial guess
def initial_guess():
	return (1e8, 1e8);

# def K(Kvco, AkPD, Ga1):
# 	return (Kvco)*AkPD*Ga1/2.0

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
#
# def globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp):
# 	# K = Krange()
# 	p = xrange(maxp, tauf)
# 	# print( K(Kvco, AkPD, Ga1))
# 	if digital:
# 		Omega = w + K(Kvco, AkPD, Ga1) * cfDig( -p )
# 		tau	  = ( Omega * tauf + v * p ) / Omega
# 	else:
# 		Omega = w + K(Kvco, AkPD, Ga1) * cfAna( -p )
# 		tau	  = ( Omega * tauf + v * p ) / Omega
# 		# print('tau=', tau)
# 	return {'Omeg': Omega, 'tau': tau}



def globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp):
	# K = Krange()
	p = xrange(maxp, tauf)
	K=(Kvco)*AkPD*Ga1/2.0
	# print( K(Kvco, AkPD, Ga1))
	if digital:
		Omega = w + K * cfDig( -p )
		tau	  = ( Omega * tauf + v * p ) / Omega
	else:
		Omega = w + K* cfAna( -p )
		tau	  = ( Omega * tauf + v * p ) / Omega
		# print('tau=', tau)

	return {'Omeg': Omega, 'tau': tau}






def linStabEq(l_vec, Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, digital):
	x = np.zeros(2)
	K=(Kvco)*AkPD*Ga1/2.0
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

def linStabEq_expansion(l_vec, Omega, tau, tauf, Kvco, AkPD, Ga1, tauc, v, zeta, digital):
	x = np.zeros(2)
	K=(Kvco)*AkPD*Ga1/2.0
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
def solveLinStab(Omega, tau, tauf,w, Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion):
	Ga1 = Krange();
	lambsolRe = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	lambsolIm = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	lambsolReKra = np.zeros([len(Ga1),len(xrange(maxp, tauf))],dtype=np.float64)
	lambsolImKra = np.zeros([len(Ga1),len(xrange(maxp, tauf))],dtype=np.float64)
	lambsolReKraMax = np.zeros(len(Ga1),dtype=np.float64)
	lambsolImKraMax = np.zeros(len(Ga1),dtype=np.float64)
	tauGa1= [];
	results = [];
	tauga1=np.zeros(len(Ga1),dtype=np.float64)
	init   = initial_guess();
	zeta   = -1;
	c 	   = 3E8;
	OmegUnst = []; OmegStab = []; tauUnst = []; tauStab = [];
	#print(type(init), init)
	#print(len(xrange(K)))

	if expansion:
		for index in range(len(Ga1)):
			tauGa1[index]=globalFreq(w, Kvco, AkPD, Ga1[index], tauf, v, digital, maxp)['tau']

			for index2 in range(len(xrange(maxp, tauf))):
			# print(index, wR[index])
				temp =  optimize.root(linStabEq_expansion, init, args=( Omega[index2], tau[index2], tauf, Kvco, AkPD, Ga1[index], tauc, v, zeta, digital),  tol=1.364e-8, method='hybr')
			# print('temp =',temp)
			# print('temp =',temp)
				lambsolReKra[index2,index]  = temp.x[0]
				lambsolImKra[index2,index]  = temp.x[1]

			tempRe = lambsolReKra[np.round(lambsolReKra[:,index],12)!=0,index];
			tempIm = lambsolImKra[np.round(lambsolImKra[:,index],32)!=0,index];
		if len(tempRe) != 0:
			#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
			lambsolRebKra[index] = np.real(np.max(tempRe));
			lambsolImbeKra[index] = tempIm[tempRe[:].argmax()]
		else:
			lambsolReKra[index] = 0.0;
			lambsolImKra[index] = 0.0;
	# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
		init = initial_guess();
	else:

		for index in range(len(Ga1)):
			# print(len(tauGa1))
			tauga1=np.asarray(globalFreq(w, Kvco, AkPD, Ga1[index], tauf, v, digital, maxp)['tau'])
			# print(tauga1)
			# time.sleep(4)
			tauGa1=np.append(tauGa1,tauga1)
			# print('index',index)
			# print('lenG=',len(tauGa1))
			# print('len=',len(tauga1))
			# # time.sleep(0.0000001)

			for index2 in range(len(xrange(maxp, tauf))):
				# print('Omega[index2]=',Omega[index2])
				temp = optimize.root(linStabEq, init, args=(Omega[index2], tau[index2], tauf, Kvco, AkPD, Ga1[index], tauc, v, zeta, digital), tol=1e-14, method='hybr')
				lambsolReKra[index,index2]  = temp.x[0]
				lambsolImKra[index,index2]  = temp.x[1]
			results.append([lambsolReKra[index,index2], lambsolImKra[index,index2],tauGa1[index], Ga1[index]]);

			tempRe = lambsolReKra[np.round(lambsolReKra[:,index2],18)!=0,index];
			tempIm = lambsolImKra[np.round(lambsolImKra[:,index2],32)!=0,index];
			if len(tempRe) != 0:
				#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
				lambsolReKraMax[index] = np.real(np.max(tempRe));
				lambsolImKraMax[index] = tempIm[tempRe[:].argmax()]
				# results.append([lambsolReKraMax[index], lambsolImKraMax[index], lambsolReKra[index,index2], lambsolImKra[index,index2],
				# 				tauGa1,Ga1[index]]);
			else:
				lambsolReKraMax[index] = 0.0;
				lambsolImKraMax[index] = 0.0;
			init = initial_guess();

				# results.append([TR_vec[index1], fR_vec[index1], K_vec[index2], orderP['order'][-1],
			# 			np.mean(orderP['order'][s_average_index:]), np.var(orderP['order'][s_average_index:]),
			# 			phase_diff[-1], np.mean(phase_diff[s_average_index:]), np.var(phase_diff[s_average_index:]),
			# 			boolean_stability ,averaging_time, freq_diff[-1], np.mean(freq_diff[-f_average_index:]),
			# 			np.var(freq_diff[-f_average_index:])]);

		# print(len(Ga1))
		# print('lenlambsolReKraMax=',len(lambsolReKraMax))
		# print('lenlambsolReKra=',len(lambsolReKra))
		# print('lenlambsolImKra=',len(lambsolImKra))
		# print('lentauGa1=',len(tauGa1))



		# print('len(tauGa1)=',tauGa1[0],len(Ga1))
			# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])

			# if lambsolRe[index] >= 0:
			# 	OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
			# else:
			# 	OmegStab.append(Omega[index]); tauStab.append(tau[index]);

		# print('\n\nlen(lambsolImKra):', lambsolReKra, '     len(lambsolImKra):', len(lambsolImKra),'\n')
		# print('\n\nlen(lambsolImKra):', len(lambsolReKra), '     len(lambsolImKra):', len(lambsolImKra),'\n')
	# return {'Re': lambsolRe, 'Im': lambsolIm, 'OmegUnst': OmegUnst, 'tauUnst': tauUnst, 'OmegStab': OmegStab, 'tauStab': tauStab}

	return {'ReKra': lambsolReKra, 'ImKra': lambsolImKra,'ReKraMax': lambsolReKraMax,'ImKraMax':lambsolImKraMax,'tauGa1':tauGa1,'results': np.array(results)}

def solveLinStab2(Omega, tau, tauf,w, Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion):
	Ga1 = Krange();
	taucran=taucrange()
	results = [];
	lambsolRe = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	lambsolIm = np.zeros(len(xrange(maxp, tauf)),dtype=np.float64)
	lambsolReKra = np.zeros([len(Ga1),len(taucran),len(xrange(maxp, tauf))],dtype=np.float64)
	lambsolImKra = np.zeros([len(Ga1),len(taucran),len(xrange(maxp, tauf))],dtype=np.float64)
	lambsolReKraMax = np.zeros([len(Ga1),len(taucran)],dtype=np.float64)
	lambsolImKraMax = np.zeros([len(Ga1),len(taucran)],dtype=np.float64)
	tauGa1= np.zeros([len(Ga1),len(taucran)],dtype=np.float64)
	taucRan=np.zeros(len(taucran),dtype=np.float64)
	init   = initial_guess();
	zeta   = -1;
	c 	   = 3E8;

	#print(type(init), init)
	#print(len(xrange(K)))

	if expansion:
		for index in range(len(taucran)):
			for index1 in range(len(Ga1)):

				for index2 in range(len(xrange(maxp, tauf))):
				# print(index, wR[index])
					temp =  optimize.root(linStabEq_expansion, init, args=( Omega[index2], tau[index2], tauf, Kvco, AkPD, Ga1[index], tauc, v, zeta, digital),  tol=1.364e-8, method='hybr')
				# print('temp =',temp)
				# print('temp =',temp)
					lambsolReKra[index2,index1]  = temp.x[0]
					lambsolImKra[index2,index1]  = temp.x[1]

				tempRe = lambsolReKra[np.round(lambsolReKra[:,index],32)!=0,index];
				tempIm = lambsolImKra[np.round(lambsolImKra[:,index],32)!=0,index];
		if len(tempRe) != 0:
			#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
			lambsolRebKra[index1] = np.real(np.max(tempRe));
			lambsolImbeKra[index1] = tempIm[tempRe[:].argmax()]
		else:
			lambsolReKra[index1] = 0.0;
			lambsolImKra[index1] = 0.0;
	# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])
		init = initial_guess();
	else:
		for index in range(len(taucran)):

			for index1 in range(len(Ga1)):
				# print('Ga1=',Ga1[index])
				tauGa1[index1,index]=globalFreq(w, Kvco, AkPD, Ga1[index1], tauf[index], v, digital, maxp)
				print('raga1',tauGa1)
				for index2 in range(len(xrange(maxp, tauf))):
					# print('Omega[index2]=',Omega[index2])
					temp = optimize.root(linStabEq, init, args=(Omega[index2], tau[index2],tauf[index], Kvco, AkPD, Ga1[index1], tauc, v, zeta, digital), tol=1e-14, method='hybr')
					lambsolReKra[index1,index,index2]  = temp.x[0]
					lambsolImKra[index1,index,index2]  = temp.x[1]

				tempRe = lambsolReKra[np.round(lambsolReKra[:,:,index2],12)!=0,index];
				tempIm = lambsolImKra[np.round(lambsolImKra[:,:,index2],32)!=0,index];
			if len(tempRe) != 0:
				#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
				lambsolReKraMax[index1,index] = np.real(np.max(tempRe));
				lambsolImKraMax[index1,index] = tempIm[tempRe[:].argmax()]
				results.append([lambsolReKraMax[index1,index], lambsolImKraMax[index1,index], lambsolReKra[index1,index,index2], lambsolImKra[index1,index,index2],
								tauGa1[index1,index]]);
			else:
				lambsolReKraMax[index1,index] = 0.0;
				lambsolImKraMax[index1,index] = 0.0;
				results.append([lambsolReKraMax[index1,index], lambsolImKraMax[index1,index], lambsolReKra[index1,index,index2], lambsolImKra[index1,index,index2],
								tauGa1[index1,index]]);
		init = initial_guess();
			# print('temp.x[0] =', temp.x[0], 'temp.x[1] =', temp.x[1])




			# if lambsolRe[index] >= 0:
			# 	OmegUnst.append(Omega[index]); tauUnst.append(tau[index]);
			# else:
			# 	OmegStab.append(Omega[index]); tauStab.append(tau[index]);

		# print('\n\nlen(lambsolImKra):', lambsolReKra, '     len(lambsolImKra):', len(lambsolImKra),'\n')
			# print('\n\nlen(lambsolImKra):', len(lambsolReKra), '     len(lambsolImKra):', len(lambsolImKra),'\n')
		# return {'Re': lambsolRe, 'Im': lambsolIm, 'OmegUnst': OmegUnst, 'tauUnst': tauUnst, 'OmegStab': OmegStab, 'tauStab': tauStab}
	return {'ReKra': lambsolReKra, 'ImKra': lambsolImKra,'ReKraMax': lambsolReKraMax,'ImKraMax':lambsolImKraMax,'tauGa1':tauGa1,'results': np.array(results)}



# choose digital vs analog
digital = False;

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider
w    	= 2.0*np.pi*24E9;
Kvco    = 2*np.pi*(754.64E6);
AkPD	= 1.6
# Ga_1	= Ga1
Ga1=0.01

tauf 	= 0.0
tauc	= 1.0/(2.0*np.pi*120E6);
v	 	= 32.0;
c		= 3E8
maxp 	= 70;
start = time.time()
# print('a=',len(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf,w,  Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion)['tauGa1']))
# print('b=',len(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf,w,  Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion)['ReKraMax']))
X= Krange();
Y=np.asarray(w/2.0*np.pi)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf,w,  Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion)['tauGa1'];
Z=2.0*np.pi*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf,w,  Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion)['ReKra']/np.asarray(w); print(len(X),len(Y),len(Z))

print('c=',solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf,w,  Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion)['results'])

end = time.time()
print(end - start)
# print(len(solveLinStab2(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf,w,  Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion)['tauGa1']))
# print(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp)['Omeg'],globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp)['tau'], tauf,w,  Kvco, AkPD, Ga1, tauc, v, digital, maxp, expansion)['tauGa1'])
