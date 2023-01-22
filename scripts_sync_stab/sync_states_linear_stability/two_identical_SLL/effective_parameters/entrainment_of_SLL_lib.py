from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.signal import sawtooth
from scipy.signal import square
import scipy.optimize as optimize
from scipy.optimize import fsolve
from scipy.optimize import root
# maxtau = 5.0;


def xrangetau(maxtau):
	min_tau = 0.0;
	return np.linspace(min_tau, maxtau, 200)


def xrange1(w, Kval):
	dp		= 0.001;
	min_wR = -1.0*Kval+ 1.0*w;
	max_wR =  1.0*Kval+ 1.0*w;
	return np.linspace(min_wR, max_wR,200)
def K(Kvco, AkPD, Ga1):
	return (Kvco*Ga1*AkPD)/2.0
# set initial guess
# def initial_guess():
# 	return (0.71, -2.1);
def initial_guess1():
	return (np.linspace(-0.1, 0.1, 8));
def initial_guess2():
	return (np.linspace(-0.1, 0.1, 8));

# digital case
cfDig        = lambda x: sawtooth(x,width=0.5)
def cfDigInverse(x,slope):
	if slope==1:
		return -np.pi/2.0*x-np.pi/2.0;
	if slope==2:
		return +np.pi/2.0*x+np.pi/2.0;

cfDigDeriv   = lambda x: (2.0/np.pi)*square(x,duty=0.5)
# analog case
cfAna        = lambda x: np.cos(x);
def cfAnaInverse(x,slope):
	if slope==1:
		return -np.arccos(x);
	if slope==2:
		return np.arccos(x);
cfAnaDeriv   = lambda x: -1.0*np.sin(x);

def beta(wR, w, tau, tauf, Kvco, AkPD, Ga1, INV, v, digital, slope):
	if digital:
		return  v*cfDigInverse((wR-w)/K(Kvco, AkPD, Ga1),slope) - wR*(tau-tauf) - v*INV ;
	else:
		return v*cfAnaInverse((wR-w)/K(Kvco, AkPD, Ga1),slope) - wR*(tau-tauf) - v*INV ;

def alphaent(wR, w, tau, tauf, Kvco, AkPD, Ga1, INV, v, tauc, digital, slope):
	if digital:
		alpha = K(Kvco, AkPD, Ga1)*cfDigDeriv((-wR*(tau-tauf)-beta(wR, w, tau, tauf, Kvco, AkPD, Ga1, INV, v, digital, slope))/v+INV)
	else:
		alpha =  K(Kvco, AkPD, Ga1)*cfAnaDeriv((-wR*(tau-tauf)-beta(wR, w, tau, tauf, Kvco, AkPD, Ga1, INV, v, digital, slope))/v+INV)

	return alpha

def linStabEq(l_vec, wR, w, tau, tauf, Kvco, AkPD, Ga1, INV, v, tauc, digital, slope):
	x = np.zeros(2)
	# if digital:
	# 	alpha = K(Kvco, AkPD, Ga1)*cfDigDeriv((-wR*(tau-tauf)-beta(wR, w, tau, tauf, Kvco, AkPD, Ga1, INV, v, digital,slope))/v+INV)
	# else:
	# 	alpha = K(Kvco, AkPD, Ga1)*cfAnaDeriv((-wR*(tau-tauf)-beta(wR, w, tau, tauf, Kvco, AkPD, Ga1, INV, v, digital,slope))/v+INV)
	# print(alpha)
	l = l_vec[0] + 1j * l_vec[1]
	# f = l*(1.0+l*tauc) + alpha * np.exp(l*tauf)
	# f = l+tauc*l**2 + alpha * np.exp(l*tauf)
	f = l+tauc*l**2 + alphaent(wR, w, tau, tauf, Kvco, AkPD, Ga1, INV, v, tauc, digital, slope) * np.exp(l*tauf)

	x[0] = np.real(f)
	x[1] = np.imag(f)
	return x

def solveLinStabEnt(wR, w, tauf, Kvco, AkPD, Ga1, INV, v, tauc, maxtau, digital,slope):
	lambsolRe = np.zeros([len(initial_guess1()),len(xrangetau(maxtau))],dtype=np.float64)
	lambsolIm = np.zeros([len(initial_guess1()),len(xrangetau(maxtau))],dtype=np.float64)
	lambsolReMax = np.zeros(len(xrangetau(maxtau)),dtype=np.float64)
	lambsolImMax = np.zeros(len(xrangetau(maxtau)),dtype=np.float64)

	init1 = initial_guess1();
	init2 = initial_guess2();
	tau= xrangetau(maxtau);
	# f= open("11_02_2020b.txt","w+")
	#print(type(init), init)
	#print(len(xrange(K)))
	for index in range(len(tau)):
		for index2 in range(len(init1)):
			# print(init[1])													 wR, tau, tauf, K, INV, v, tauc, digital
			temp = optimize.root(linStabEq, (init1[index2],init1[index2]), args=(wR, w, tau[index], tauf, Kvco, AkPD, Ga1, INV, v, tauc, digital,slope), tol=1.364e-12, method='hybr')
			if temp.success==True:
				lambsolRe[index2,index]  = temp.x[0]
				lambsolIm[index2,index]  = temp.x[1]

		tempRe = lambsolRe[np.round(lambsolRe[:,index],122)!=0,index];
		tempIm = lambsolIm[np.round(lambsolIm[:,index],442)!=0,index];

		if len(tempRe) != 0:
			#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
			lambsolReMax[index] = np.real(np.max(tempRe));
			lambsolImMax[index] = tempIm[tempRe[:].argmax()]
		else:
			lambsolReMax[index] = 0.0;
			lambsolImMax[index] = 0.0;

		# np.savetxt(f,np.column_stack(((2.0*np.pi*xrange(maxtau)/wR),(2.0*np.pi*lambsolReMax)/wR, abs(lambsolImMax)/wR)), fmt='%s %s %s');
		# #init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))
		init1 = initial_guess1();
		init2 = initial_guess2();
		#print(type(init), init)
		#print('Re[lambda] =',lambsol)
	# f.close()
	return {'Re':  np.array(lambsolRe), 'Im': np.array(lambsolIm), 'ReMax': lambsolReMax,'ImMax':abs(lambsolImMax)}

# def solveLinStabwR(wR, w, tau, tauf, K, INV, v, tauc, digital, slope):
def solveLinStabwR(wR, w, tau, tauf, K, INV, v, tauc, digital,slope):
	lambsolRe = np.zeros([len(xrange1(w,K)),len(xrange1(w,K))],dtype=np.float64)
	lambsolIm = np.zeros([len(xrange1(w,K)),len(xrange1(w,K))],dtype=np.float64)
	lambsolReMax = np.zeros(len(xrange1(w,K)),dtype=np.float64)
	lambsolImMax = np.zeros(len(xrange1(w,K)),dtype=np.float64)

	init1 = initial_guess1();
	init2 = initial_guess1();

	# f= open("11_02_2020b.txt","w+")
	#print(type(init), init)
	#print(len(xrange(K)))
	if digital:
		for index in range(len(wR)):
			for index2 in range(len(init2)):
				# print(init[1])													 wR, tau, tauf,Kvco, AkPD, Ga1, INV, v, tauc, digital
				temp = optimize.root(linStabEq, (init2[index2],init2[index2]), args=(wR[index],w, tau, tauf, K, INV, v, tauc, digital,slope), tol=1.0e-14, method='hybr')
				lambsolRe[index2,index]  = temp.x[0]
				lambsolIm[index2,index]  = temp.x[1]

			tempRe = lambsolRe[np.round(lambsolRe[:,index],16)!=0,index];
			tempIm = lambsolIm[np.round(lambsolIm[:,index],16)!=0,index];
			if len(tempRe) != 0:
				#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
				lambsolReMax[index] = np.real(np.max(tempRe));
				if tempRe[:].argmax() < len(tempIm):
					lambsolImMax[index] = tempIm[tempRe[:].argmax()]
			else:
				lambsolReMax[index] = 0.0;
				lambsolImMax[index] = 0.0;
			# if len(tempRe) != 0:
			# 	#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
			# 	lambsolReMax[index] = np.real(np.max(tempRe));
			# 	lambsolImMax[index] = tempIm[tempRe[:].argmax()]
			# else:
			# 	lambsolReMax[index] = 0.0;
			# 	lambsolImMax[index] = 0.0;

			# np.savetxt(f,np.column_stack(((2.0*np.pi*xrange(maxtau)/wR),(2.0*np.pi*lambsolReMax)/wR, abs(lambsolImMax)/wR)), fmt='%s %s %s');
			# #init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))

			init2 = initial_guess1();
	else:
			for index in range(len(wR)):
				for index2 in range(len(init1)):
					# print(init[1])													 wR, tau, tauf,Kvco, AkPD, Ga1, INV, v, tauc, digital
					temp = optimize.root(linStabEq, (init1[index2],init1[index2]), args=(wR[index], w, tau, tauf, K, INV, v, tauc, digital,slope), tol=1.0e-14, method='hybr')
					lambsolRe[index2,index]  = temp.x[0]
					lambsolIm[index2,index]  = temp.x[1]

				tempRe = lambsolRe[np.round(lambsolRe[:,index],16)!=0,index];
				tempIm = lambsolIm[np.round(lambsolIm[:,index],16)!=0,index];
				if len(tempRe) != 0:
					#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
					lambsolReMax[index] = np.real(np.max(tempRe));
					if tempRe[:].argmax() < len(tempIm):
						lambsolImMax[index] = tempIm[tempRe[:].argmax()]
				else:
					lambsolReMax[index] = 0.0;
					lambsolImMax[index] = 0.0;
				# if len(tempRe) != 0:
				# 	#print('type(np.max(tempRe)):', type(np.real(np.max(tempRe))));
				# 	lambsolReMax[index] = np.real(np.max(tempRe));
				# 	lambsolImMax[index] = tempIm[tempRe[:].argmax()]
				# else:
				# 	lambsolReMax[index] = 0.0;
				# 	lambsolImMax[index] = 0.0;

				# np.savetxt(f,np.column_stack(((2.0*np.pi*xrange(maxtau)/wR),(2.0*np.pi*lambsolReMax)/wR, abs(lambsolImMax)/wR)), fmt='%s %s %s');
				# #init = (np.real(lambsolRe[index]), np.real(lambsolIm[index]))
				init1 = initial_guess1();

		#print(type(init), init)
		#print('Re[lambda] =',lambsol)
	# f.close()
	return {'Re':  np.array(lambsolRe), 'Im': np.array(lambsolIm), 'ReMax': lambsolReMax,'ImMax':abs(lambsolImMax)}
