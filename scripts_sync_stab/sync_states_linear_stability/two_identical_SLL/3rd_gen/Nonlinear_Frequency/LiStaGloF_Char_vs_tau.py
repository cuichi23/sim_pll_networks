#Linear Stability of Global Frequency (LiStaGloF.py)


#This program solves the condition for Global Frequency for the in phase stability of N=2 mutually coupled identical PLLs for analog and digital case.
# This program solves numerically the characteristic equation of the system in order to see the stability of the in Phase synchroniazed solutions
#There is a slider for the parameters of the system in order to see how the the stability and the system changes
# There is also a plot of the Im[l]vs Re[l](Nyquist Stability Creterion)



#!/usr/bin/python
import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq, linStabEq,globalFreqNonlinear, solveLinStabNonliner,NonlinearFreqResponse, globalFreqComparstor, distances,NonlinearKvco #,globalFreqKrange
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
# from mpmath import findroot
# from sympy import I, limit
# from sympy.solvers import solveset
# from sympy.solvers import solve
# from sympy import Symbol, Function, nsolve
# from sympy import sin, cos, exp
# import mpmath
# mpmath.mp.dps = 25

# choose digital vs analog
digital = True;

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;

inphase1= True;
inphase2= False;
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider

w    	= 2.0*np.pi*24.29E9;	# intrinsic	frequency
Kvco    = 2.0*np.pi*(757.64E6);	# Sensitivity of VCO
AkPD	= 0.8					# amplitude of the PD -- the factor of 2 here is related to the IMS 2020 paper, AkPD as defined in that paper already includes the 0.5 that we later account for when calculating K
Ga1     = 1.0					 	# Gain of the first adder
order	= 2.0					# the order of the Loop Filter
tauf    = 0.0					# tauf = sum of all processing delays in the feedback
v		= 512;					# the division

Vbias  = 2.55				# DC voltage offset old: 2.49
if v == 512:
	tauc	= 1.0/(2.0*np.pi*0.965E6);  # the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
if v == 128:
	tauc	= 1.0/(2.0*np.pi*4.019E6);  # the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
if v == 32:
	tauc	= 1.0/(2.0*np.pi*40.19E6);  # the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter

c		= 0.63*3E8				# speed of light
if v==32:
	maxp=35
if v==128:
	maxp 	= 17;
if v==512:
	maxp=25
INV		= 1.0*np.pi				# Inverter
mindelay = 4.236E-9
threshold =1.0e8
if v==32:
	threshold2 =9.0e6
if v==128:
	threshold2 =1.0e6
if v==512:
	threshold2 =0.5e6
wnormy = 2.0*np.pi;				# this is the frequency with which we rescale the y-axis, choose either 1, 2pi or w (y-axis: radHz, Hz, dimless)
wnormx = 2.0*np.pi;				# this is the frequency with which we rescale the x-axis, choose either 2pi or w

zeta	= -1
figwidth  =	6;
figheight = 6;
#
####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'

DatasetN032 = np.load('Delay_vs_NetworkFreq_N032b.npy', allow_pickle=True)
DatasetN032 = DatasetN032.item()

DatasetN128 = np.load('Delay_vs_NetworkFreq_N128.npy', allow_pickle=True)
DatasetN128 = DatasetN128.item()

DatasetN512 = np.load('Delay_vs_NetworkFreq_N512.npy', allow_pickle=True)
DatasetN512 = DatasetN512.item()

# array_dataset=np.array(Dataset)
t32     = np.linspace(1,len(DatasetN032['Net_Freq_NodeA']),len(DatasetN032['Net_Freq_NodeA']))
t128    = np.linspace(1,len(DatasetN128['Net_Freq_NodeA']),len(DatasetN128['Net_Freq_NodeA']))
t512    = np.linspace(1,len(DatasetN512['Net_Freq_NodeA']),len(DatasetN512['Net_Freq_NodeA']))

meanFreqN032=[];    meanFreqN128=[];    meanFreqN512=[];
stdN032=[];    stdN128=[];    stdN512=[];
meanFreqVCON032=[];    meanFreqVCON128=[];    meanFreqVCON512=[];
stdVCON032=[];    stdVCON128=[];    stdVCON512=[];

delay128=[]; delay512=[]; delay32=[];
distanceLinear512=[]; distanceNonLinear512=[]; distanceLinear128=[];
distanceNonLinear128=[]; distanceLinear32=[];distanceNonLinear32=[];
distanceLinear32Net=[]; distanceLinear128Net=[]; distanceLinear512Net=[];
distanceNonLinear32Net=[]; distanceNonLinear128Net=[];distanceNonLinear512Net=[];

distance512LinearNetMean=[];distance512NonLinearNetMean=[];
distance128LinearNetMean=[]; distance128NonLinearNetMean=[];
distance32LinearNetMean=[]; distance32NonLinearNetMean=[];

distance512LinearNetSTD=[];distance512NonLinearNetSTD=[];
distance128LinearNetSTD=[];distance128NonLinearNetSTD=[];
distance32LinearNetSTD=[];distance32NonLinearNetSTD=[];
inphase=False
k=0
# print(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'][np.where( globalFreq(w, Kvco, AkPD,
# 	Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= 4.236e-9)[-1][-1] ])
# print(solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD,
# 	Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= 4.236e-9)[-1][-1] ], globalFreq(w, Kvco, AkPD,
# 		Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'][np.where( globalFreq(w, Kvco, AkPD,
# 		Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= 4.236e-9)[-1][-1] ], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegStab']/np.asarray(2.0*np.pi*v) )
#
#
# print(solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
#  				   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
#  			   	tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegStab'][np.where( solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 				 				   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 				 			   	tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['tauStab']<= np.asarray(4.236e-9))[-1][-1] ]/np.asarray(2.0*np.pi*v) )


# print(DatasetN128['Net_Freq_NodeA'])


if v==32:


	for i in range(len(DatasetN032['Net_Freq_NodeA'])):
		meanFreqN032.append(np.mean(DatasetN032['Net_Freq_NodeA'][i])*np.array(1.0e6))
		stdN032.append(np.std(DatasetN032['Net_Freq_NodeA'][i]))
		delay32.append(np.float(DatasetN032['delay'][i])*(0.01E-9)+mindelay)
		# print(delay32[i])
		# print(np.float(DatasetN032['delay'][i]))

		if abs(np.mean(DatasetN032['Net_Freq_NodeA'][i])*np.array(1.0e6)-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD,
			Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v) ) < threshold2 and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
							   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
							tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay32[i]))[-1][-1] ] < 0.0:
			inphase=True;
			# print('Inphase')
			distanceLinear32Net.append(abs(DatasetN032['Net_Freq_NodeA'][i]*np.array(1.0e6)-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v) ) )
			distanceNonLinear32Net.append(abs(DatasetN032['Net_Freq_NodeA'][i]*np.array(1.0e6)-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v) ) )
		else:
			inphase=False;
			# print('Antiphase')
			distanceLinear32Net.append(abs(DatasetN032['Net_Freq_NodeA'][i]*np.array(1.0e6)-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v) ) )
			distanceNonLinear32Net.append(abs(DatasetN032['Net_Freq_NodeA'][i]*np.array(1.0e6)-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v) ) )



		distance32LinearNetMean.append(np.mean(distanceLinear32Net[i]))
		distance32NonLinearNetMean.append(np.mean(distanceNonLinear32Net[i]))
		distance32LinearNetSTD.append( np.std(distanceLinear32Net[i]) )
		distance32NonLinearNetSTD.append( np.std(distanceNonLinear32Net[i]) )
		print(delay32[i],distance32LinearNetMean[i],distance32NonLinearNetMean[i],distance32LinearNetMean[i]-distance32NonLinearNetMean[i],distance32LinearNetSTD[i],distance32NonLinearNetSTD[i])

# print('32')
# print(np.mean(distance32LinearNetMean))
	# data32.write(delay32[i])#,',', distance32LinearNetMean[i],distance32NonLinearNetMean[i],distance32LinearNetMean[i]-distance32NonLinearNetMean[i],distance32LinearNetSTD[i],distance32NonLinearNetSTD[i])
	# # print(distances(DatasetN032['Net_Freq_NodeA'][i],
		# globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v),
		# globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v) )  ) )
# print(np.mean(distance32NonLinearNetMean) )
# print(np.mean(distance32LinearNetMean)-np.mean(distance32NonLinearNetMean))


# print(DatasetN032['Net_Freq_NodeA'][0]*np.array(1.0e6))
if v==128:


	for i in range(len(DatasetN128['Net_Freq_NodeA'])):

		# print(DatasetN128['Net_Freq_NodeA'][0])
		meanFreqN128.append(np.mean(DatasetN128['Net_Freq_NodeA'][i]))
		stdN128.append(np.std(DatasetN128['Net_Freq_NodeA'][i]))
		# print(DatasetN128["delay"][i])
		# print( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV )['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi*v) )
		# print(type(DatasetN128["delay"][i]))
		delay128.append(np.float(DatasetN128['delay'][i])*(0.01E-9)+mindelay)
		# print(delay128[i])

		if (abs(np.mean(DatasetN128['Net_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi*v) ) )<= threshold2 and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
						   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
						tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay128[i]))[-1][-1] ] < 0.0:
			inphase=True;
			# print('Inphase')
			distanceLinear128Net.append(abs(DatasetN128['Net_Freq_NodeA'][i]-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi*v) ) )
			distanceNonLinear128Net.append(abs(DatasetN128['Net_Freq_NodeA'][i]-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi*v) ) )
		else:
			inphase=False;
			# print('Antiphase')
			distanceLinear128Net.append(abs(DatasetN128['Net_Freq_NodeA'][i]-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi*v) ) )
			distanceNonLinear128Net.append(abs(DatasetN128['Net_Freq_NodeA'][i]-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi*v) ) )

		distance128LinearNetMean.append(np.mean(distanceLinear128Net[i]))
		distance128NonLinearNetMean.append(np.mean(distanceNonLinear128Net[i]))
		distance128LinearNetSTD.append( np.std(distanceLinear128Net[i]) )
		distance128NonLinearNetSTD.append( np.std(distanceNonLinear128Net[i]) )
		print(delay128[i],distance128LinearNetMean[i],distance128NonLinearNetMean[i],distance128LinearNetMean[i]-distance128NonLinearNetMean[i],distance128LinearNetSTD[i],distance128NonLinearNetSTD[i])

# print('128')
# print(np.mean(distance128LinearNetMean))
# print(np.mean(distance128NonLinearNetMean))
# print(np.mean(distance128LinearNetMean)-np.mean(distance128NonLinearNetMean))

# print(distance128LinearNetSTD/np.mean(distance128LinearNetMean), distance128NonLinearNetSTD/np.mean(distance128NonLinearNetMean))

	# print(distanceLinear128Net[i])


	# print(np.float(DatasetN128['delay'][i]))
if v==512:

	for i in range(len(DatasetN512['Net_Freq_NodeA'])):
		# print()
		meanFreqN512.append(np.mean(DatasetN512['Net_Freq_NodeA'][i]))
		stdN512.append(np.std(DatasetN512['Net_Freq_NodeA'][i]))
		delay512.append(np.float(DatasetN512['delay'][i])*(0.01E-9)+mindelay)

		if (abs(np.mean(DatasetN512['Net_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi*v) ) )<= threshold2 and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
						   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
						tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay512[i]))[-1][-1] ] < 0.0:
			inphase=True;
			# print('Inphase')
			distanceLinear512Net.append(abs(DatasetN512['Net_Freq_NodeA'][i]-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi*v) ) )
			distanceNonLinear512Net.append(abs(DatasetN512['Net_Freq_NodeA'][i]-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi*v) ) )
		else:
			inphase=False;
			# print('Antiphase')
			distanceLinear512Net.append(abs(DatasetN512['Net_Freq_NodeA'][i]-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi*v) ) )
			distanceNonLinear512Net.append(abs(DatasetN512['Net_Freq_NodeA'][i]-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi*v) ) )

		distance512LinearNetMean.append(np.mean(distanceLinear512Net[i]))
		distance512NonLinearNetMean.append(np.mean(distanceNonLinear512Net[i]))
		distance512LinearNetSTD.append( np.std(distanceLinear512Net[i]) )
		distance512NonLinearNetSTD.append( np.std(distanceNonLinear512Net[i]) )
		print(delay512[i],distance512LinearNetMean[i],distance512NonLinearNetMean[i],distance512LinearNetMean[i]-distance512NonLinearNetMean[i],distance512LinearNetSTD[i],distance512NonLinearNetSTD[i])

# print('512')
# print(np.mean(distance512LinearNetMean))
# print(np.mean(distance512NonLinearNetMean))
#
# print(np.mean(distance512LinearNetMean)-np.mean(distance512NonLinearNetMean))
# # print(distance512LinearNetSTD, distance512NonLinearNetSTD)
# print(np.mean(distanceLinear512Net)-np.mean(distanceNonLinear512Net))



if v==32:


	for i in range(len(DatasetN032['VCO_Freq_NodeA'])):
		meanFreqVCON032.append(np.mean(DatasetN032['VCO_Freq_NodeA'][i]))
		stdVCON032.append(np.std(DatasetN032['VCO_Freq_NodeA'][i]))
		# print(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'])
		# print(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'])

		# globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear( w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV )['tau']>= delay32[i] )[0][0] ] /(2.0*np.pi)
		# globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay32[i])[0][0] ] /(2.0*np.pi)
		# print(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay512[i])[0][0] ] /(2.0*np.pi))
		# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay512[i])[0][0] ] /(2.0*np.pi) )
		if (abs(np.mean(DatasetN032['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi) ) )<= threshold and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
						   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
						tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay32[i]))[-1][-1] ] < 0.0:
			inphase=True;
			# print('Inphase')
			distanceLinear32.append(np.mean(DatasetN032['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi) )
			distanceNonLinear32.append(np.mean(DatasetN032['VCO_Freq_NodeA'][i])-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi) )
		else:
			inphase=False;
			# print('Antiphase')
			distanceLinear32.append(np.mean(DatasetN032['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi) )
			distanceNonLinear32.append(np.mean(DatasetN032['VCO_Freq_NodeA'][i])-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi) )

		# distanceLinear32.append(np.mean(DatasetN032['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay32[i])[0][0] ] /(2.0*np.pi) )
		# distanceNonLinear32.append(np.mean(DatasetN032['VCO_Freq_NodeA'][i])-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']>= delay32[i])[0][0] ] /(2.0*np.pi) )
		# print(distanceLinear[i],distanceNonLinear[i])
		if v==32 and abs(distanceLinear32[i])>abs(distanceNonLinear32[i]):
			if inphase==True:
				print('Inphase')
			else:
				print('Antiphase')

			k=k+1
			print('Noninear is closer')
		elif v==32 and abs(distanceLinear32[i])<abs(distanceNonLinear32[i]):
			if inphase==True:
				print('Inphase')
			else:
				print('Antiphase')

			print('Linear is closer')
		inphase=False

	inphase=False
if v==128:

	for i in range(len(DatasetN128['VCO_Freq_NodeA'])):
		meanFreqVCON128.append(np.mean(DatasetN128['VCO_Freq_NodeA'][i]))
		stdVCON128.append(np.std(DatasetN128['VCO_Freq_NodeA'][i]))
		# globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay128[i])[0][0] ] /(2.0*np.pi)
		# # print(np.float(DatasetN032["VCO_Freq_NodeA"][i]))
		# globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay128[i])[0][0] ] /(2.0*np.pi)
		# # print(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay512[i])[0][0] ] /(2.0*np.pi))
		# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay512[i])[0][0] ] /(2.0*np.pi) )
		#
		if (abs(np.mean(DatasetN128['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi) ))<= threshold and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
						   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
						tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay128[i]))[-1][-1] ] < 0.0:
			inphase=True;
			# print('Inphase')
			distanceLinear128.append(np.mean(DatasetN128['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi) )
			distanceNonLinear128.append(np.mean(DatasetN128['VCO_Freq_NodeA'][i])-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi) )
		else:
			inphase=False
			# print('Antiphase')
			distanceLinear128.append(np.mean(DatasetN128['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi) )
			distanceNonLinear128.append(np.mean(DatasetN128['VCO_Freq_NodeA'][i])-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi) )


		# distanceLinear128.append(np.mean(DatasetN128['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay128[i])[0][0] ] /(2.0*np.pi) )
		# distanceNonLinear128.append(np.mean(DatasetN128['VCO_Freq_NodeA'][i])-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']>= delay128[i])[0][0] ] /(2.0*np.pi) )
		# print(distanceLinear[i],distanceNonLinear[i])
		if v==128 and abs(distanceLinear128[i])>abs(distanceNonLinear128[i]):
			if inphase==True:
				print('Inphase')

			else:
				print('Antiphase')

			k=k+1
			print('Noninear is closer')

		elif v==128 and abs(distanceLinear128[i])<abs(distanceNonLinear128[i]):
			if inphase==True:
				print('Inphase')

			else:
				print('Antiphase')

			print('Linear is closer')
		inphase=False

	inphase=False

if v==512:

	for i in range(len(DatasetN512['VCO_Freq_NodeA'])):
		meanFreqVCON512.append(np.mean(DatasetN512['VCO_Freq_NodeA'][i]))
		stdVCON512.append(np.std(DatasetN512['VCO_Freq_NodeA'][i]))
		# globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']>= delay512[i])[0][0] ] /(2.0*np.pi)
		# # print(np.float(DatasetN032["VCO_Freq_NodeA"][i]))
		# globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay512[i])[0][0] ] /(2.0*np.pi)
		# print(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] )
		# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ]  )
		# print(delay512[i])
		# print(abs(np.mean(DatasetN512['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi) ) )

		if (abs(np.mean(DatasetN512['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi) ) )<= threshold and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
						   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
						tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay512[i]))[-1][-1] ] < 0.0:
			inphase=True

			distanceLinear512.append(np.mean(DatasetN512['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi) )
			distanceNonLinear512.append(np.mean(DatasetN512['VCO_Freq_NodeA'][i])-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi) )
		else:
			inphase=False
			distanceLinear512.append(np.mean(DatasetN512['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi) )
			distanceNonLinear512.append(np.mean(DatasetN512['VCO_Freq_NodeA'][i])-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi) )

		# print(abs(distanceLinear512[i]),distanceNonLinear512[i])
		# print(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi))
		if v==512 and np.abs(distanceLinear512[i])>np.abs(distanceNonLinear512[i]):
			if inphase==True:
				print('Inphase')
			else:
				print('Antiphase')

			k=k+1
			print('Noninear is closer')
		elif v==512 and np.abs(distanceLinear512[i])<np.abs(distanceNonLinear512[i]):
			if inphase==True:
				print('Inphase')
			else:
				print('Antiphase')

			print('Linear is closer')
		inphase=False
# print(distanceNonLinear512[9], distanceLinear512[9])
# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay512)[0][0] ] /(2.0*np.pi))
#
# print(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']>= delay512)[0][0] ] /(2.0*np.pi) )# print(k)
# print(delay512)

if v==32:
	print(np.float(k)/np.float(len(DatasetN032['VCO_Freq_NodeA'])))
elif v==128:
	print(np.float(k)/np.float(len(DatasetN128['VCO_Freq_NodeA'])))
elif v==512:
	print(np.float(k)/np.float(len(DatasetN512['VCO_Freq_NodeA'])))



# print(meanFreqVCON032)
####################################################################################################################################################################################

		# globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase, INV)
# print( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg']==w)]/(2.0*np.pi))
print(NonlinearFreqResponse(Vbias, AkPD, Ga1, 0, np.pi/2, INV, digital, inphase1)/(2*np.pi) 	)
# print(NonlinearKvco(NonlinearFreqResponse(Vbias, AkPD, Ga1, 0, np.pi/2, INV, digital, inphase1), 0.0, Vbias, AkPD, Ga1, tauf, v, INV, digital, inphase2)/(2*np.pi)	)
wfreePLLNonLinear=NonlinearFreqResponse(Vbias, AkPD, Ga1, 0, np.pi/2, INV, digital, inphase1)
# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>=3.84E-9) ]/(2*np.pi))
# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'][np.where(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<=3.84E-9) ][-1])
# print(globalFreqINV(w, Kvco, GkPD, Gk, Ak, Gl,Al, GkLF,Gvga, tauf, v, digital, maxp, inphase1, INV)['Omeg'])
# fig0    = plt.figure(num=0, figsize=(figwidth, figheight), facecolor='w', edgecolor='k')
# print(globalFreqComparstor(w, delay512[0], Kvco,Vbias,  AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['OmegLinear']/(2*np.pi) , globalFreqComparstor(w, delay512[0], Kvco,Vbias,  AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['OmegNon'][0]/(2*np.pi) )
# print(np.around(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], decimals=12) )
# print(delay512[0])
# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'][-1])
# print(np.around(delay512, decimals=11))
# print(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay512[0]-0.03e-9)[0][0] ] /(2.0*np.pi))
# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay512[0]-0.03e-9)[0][0] ] /(2.0*np.pi) )


#
# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay512[0])[0][0] ] -0.03e-9)
#
#
#
# distanceLinear		= meanFreqVCON512[0]-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay512[0])[0][0] ] /(2.0*np.pi)
# distanceNonLinear	= meanFreqVCON512[0]-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']>= delay512[0]-0.03e-9)[0][0] ] /(2.0*np.pi)
# print(meanFreqVCON512[0])
# print(distanceLinear)
# print(distanceNonLinear)
# print(solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']==delay32)][0], delay32, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegStab']/np.asarray(wnormy))
# print(np.asarray(2.0*np.pi/w)*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, digital, maxp, inphase1,INV)['Omeg'],
# 	globalFreqNonlinear(w, tauf, v, digital, maxp, inphase1,INV)['tau'], tauf,  tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ReMax'])

# print(meanFreqN032-solveLinStab(globalFreq(w, Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase1, INV)['Omeg'], delay32, tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegStab']/np.asarray(wnormy))





#*******************************************************************************
fig         = plt.figure(figsize=(figwidth,figheight))
ax          = fig.add_subplot(111)
if digital == True:
	plt.title(r'digital case for $\omega=2\pi\cdot$%2.4E Hz and division v=%3.3E' %(w/(2.0*np.pi),v));
	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
else:
	plt.title(r'analog case for $\omega=2\pi\cdot$%2.4E Hz' %(w/(2.0*np.pi)));# adjust the subplots region to leave some space for the sliders and buttons
# fig.subplots_adjust(left=0.12, bottom=0.25)
# plot grid, labels, define intial values
plt.grid()
if wnormx == w:
	plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
else:
	plt.xlabel(r'$\tau$', fontsize=20)
if wnormy == 1:
	plt.ylabel(r'$\Omega$', fontsize=20)
elif wnormy == 2.0*np.pi:
	plt.ylabel(r'$f_{\Omega}$', fontsize=20)
elif wnormy == w:
	plt.ylabel(r'$\Omega/\omega$', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=10)
# # draw the initial plot
# # # the 'lineXXX' variables are used for modifying the lines later
# # #*******************************************************************************
if v==512:
	plt.errorbar(delay512,meanFreqVCON512, xerr=0*0.75E-9, yerr=stdN512, fmt='o')
if v==128:
	plt.errorbar(delay128, meanFreqVCON128,xerr=0*0.75E-9, yerr=stdN128, fmt='o')
if v==32:
	plt.errorbar(delay32, meanFreqVCON032, xerr=0*0.75E-9, yerr=stdN032, fmt='o')

[lineOmegStabIn] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['tauStab'],
						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegStab']/np.asarray(wnormy),
						'o',ms=7, color='blue',  label=r'Inphase linear Response Stable')
# #
# [lineOmegUnstIn] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['tauUnst'],
# 						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegUnst']/np.asarray(wnormy),
# 						  'o',ms=1, color='blue', label=r'Inphase linear Response Unstable')
#
# # #
# # [lineOmegStabAprox] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['OmegaAproximation']/np.asarray(wnormy),
# # 						'o',ms=1, color='purple', alpha=0.3,  label=r'Aproximation(-cos) linear Response Stable')
# #

[lineOmegStabAnti] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['tauStab'],
						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['OmegStab']/np.asarray(wnormy),
						'+',ms=7, color='blue',  label=r'Anti-phase linear Response Stable')
#
# [lineOmegUnstAnti] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['tauUnst'],
# 						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['OmegUnst']/np.asarray(wnormy),
# 						  '+',ms=1, color='blue', label=r'Anti-phase linear Response Unstable')

#
# #
# # # print(solveLinStabNonliner(globalFreqNonlinear(w,tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# # # globalFreqNonlinear(w,  tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ReMax'])
#
[lineOmegStabInNon] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
						globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauStab'],
						 solveLinStabNonliner(globalFreqNonlinear(w,tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
						 globalFreqNonlinear(w,  tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegStab']/np.asarray(wnormy),
						 'o',ms=7, color='red',  label=r'Inphase Stable Nonlinear Response')
#
# [lineOmegUnstInNon] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# 								globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauUnst'],
# 						solveLinStabNonliner(globalFreqNonlinear(w,  tauf, v,Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# 						 globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegUnst']/np.asarray(wnormy),
# 						 'o',ms=1, color='red', label=r'Inphase Unstable Nonlinear Response')
# # #
# # #
[lineOmegStabAntiNon] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
						globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauStab'],
						 solveLinStabNonliner(globalFreqNonlinear(w,tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
						 globalFreqNonlinear(w,  tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegStab']/np.asarray(wnormy),
						 '+',ms=7, color='red',  label=r'Anti-phase Stable Nonlinear Response')
#
# [lineOmegUnstAntiNon] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
# 								globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauUnst'],
# 						 solveLinStabNonliner(globalFreqNonlinear(w,  tauf, v,Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
# 						 globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegUnst']/np.asarray(wnormy),
# 						 '+',ms=1, color='red', label=r'Anti-phase Unstable Nonlinear Response')
#

#
#
fig.set_size_inches(17,8.5)

plt.savefig('plots/3rdGenOmega_tau_Nonlinear_v_%.2f.svg' %(v), dpi=150, bbox_inches=0)
plt.savefig('plots/3rdGenOmega_tau_Nonlinear_v_%.2f.png' %(v), dpi=150, bbox_inches=0)



# fig0         = plt.figure(figsize=(figwidth,figheight))
# ax0          = fig0.add_subplot(111)
# if digital == True:
# 	plt.title(r'digital case for $\omega=2\pi\cdot$%2.4E Hz and division v=%3.3E' %(w/(2.0*np.pi),v));
# 	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# else:
# 	plt.title(r'analog case for $\omega=2\pi\cdot$%2.4E Hz' %(w/(2.0*np.pi)));# adjust the subplots region to leave some space for the sliders and buttons
# # fig.subplots_adjust(left=0.12, bottom=0.25)
# # plot grid, labels, define intial values
# plt.grid()
# if wnormx == w:
# 	plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
# # # # the 'lineXXX' variables are used for modifying the lines later
# else:
# 	plt.xlabel(r'$\tau$', fontsize=20)
# if wnormy == 1:
# 	plt.ylabel(r'$\Omega$', fontsize=20)
# elif wnormy == 2.0*np.pi:
# 	plt.ylabel(r'$f_{\Omega}$', fontsize=20)
# elif wnormy == w:
# 	plt.ylabel(r'$\Omega/\omega$', fontsize=20)
# ax.tick_params(axis='both', which='major', labelsize=10)
# # # draw the initial plot
# # # #*******************************************************************************
# if v==512:
# 	plt.errorbar(delay512,meanFreqN512, xerr=0*0.75E-9, yerr=stdN512, fmt='o')
# if v==128:
# 	plt.errorbar(delay128, meanFreqN128,xerr=0*0.75E-9, yerr=stdN128, fmt='o')
# if v==32:
# 	plt.errorbar(delay32, meanFreqN032, xerr=0*0.75E-9, yerr=stdN032, fmt='o')
#
# [lineOmegStabIn] = ax0.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['tauStab'],
# 						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegStab']/np.asarray(wnormy*v),
# 						'o',ms=7, color='blue',  label=r'Inphase linear Response Stable')
# # #
# # [lineOmegUnstIn] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['tauUnst'],
# # 						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegUnst']/np.asarray(wnormy*v),
# # 						  'o',ms=1, color='blue', label=r'Inphase linear Response Unstable')
#
# # #
# # # [lineOmegStabAprox] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['OmegaAproximation']/np.asarray(wnormy),
# # # 						'o',ms=1, color='purple', alpha=0.3,  label=r'Aproximation(-cos) linear Response Stable')
# # #
#
# [lineOmegStabAnti] = ax0.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['tauStab'],
# 						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['OmegStab']/np.asarray(wnormy*v),
# 						'+',ms=7, color='blue',  label=r'Anti-phase linear Response Stable')
# #
# # [lineOmegUnstAnti] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['tauUnst'],
# # 						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['OmegUnst']/np.asarray(wnormy),
# # 						  '+',ms=1, color='blue', label=r'Anti-phase linear Response Unstable')
#
# #
# # #
# # # # print(solveLinStabNonliner(globalFreqNonlinear(w,tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# # # # globalFreqNonlinear(w,  tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ReMax'])
# #
# [lineOmegStabInNon] = ax0.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# 						globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauStab'],
# 						 solveLinStabNonliner(globalFreqNonlinear(w,tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# 						 globalFreqNonlinear(w,  tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegStab']/np.asarray(wnormy*v),
# 						 'o',ms=7, color='red',  label=r'Inphase Stable Nonlinear Response')
# #
# # [lineOmegUnstInNon] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# # 								globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauUnst'],
# # 						solveLinStabNonliner(globalFreqNonlinear(w,  tauf, v,Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# # 						 globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegUnst']/np.asarray(wnormy),
# # 						 'o',ms=1, color='red', label=r'Inphase Unstable Nonlinear Response')
# # # #
# # # #
# [lineOmegStabAntiNon] = ax0.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
# 						globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauStab'],
# 						 solveLinStabNonliner(globalFreqNonlinear(w,tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
# 						 globalFreqNonlinear(w,  tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegStab']/np.asarray(wnormy*v),
# 						 '+',ms=7, color='red',  label=r'Anti-phase Stable Nonlinear Response')
# #
# [lineOmegUnstAntiNon] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
# 								globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauUnst'],
# 						 solveLinStabNonliner(globalFreqNonlinear(w,  tauf, v,Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
# 						 globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegUnst']/np.asarray(wnormy),
# 						 '+',ms=1, color='red', label=r'Anti-phase Unstable Nonlinear Response')
#



#
# fig0         = plt.figure(figsize=(figwidth,figheight))
# ax0          = fig0.add_subplot(111)
# if digital == True:
# 	plt.title(r'digital case for $\omega=2\pi\cdot$%2.4E Hz and division v=%3.3E' %(w/(2.0*np.pi),v));
# 	#plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/
# else:
# 	plt.title(r'analog case for $\omega=2\pi\cdot$%2.4E Hz' %(w/(2.0*np.pi)));# adjust the subplots region to leave some space for the sliders and buttons
# # fig.subplots_adjust(left=0.12, bottom=0.25)
# # plot grid, labels, define intial values
# plt.grid()
# if wnormx == w:
# 	plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
# else:
# 	plt.xlabel(r'$\tau$', fontsize=18)
# if wnormy == 1:
# 	plt.ylabel(r'$\Omega$', fontsize=18)
# elif wnormy == 2.0*np.pi:
# 	plt.ylabel(r'$f_{\Omega}$', fontsize=18)
# elif wnormy == w:
# 	plt.ylabel(r'$\Omega/\omega$', fontsize=18)
#
# # # draw the initial plot
# # # the 'lineXXX' variables are used for modifying the lines later
# # #*******************************************************************************
#
# [lineOmegStabIn] = ax0.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
#  					tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 					globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
# 					 					tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegStab'],
# 						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 						  globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegStab']/np.asarray(wnormy),
# 						'o',ms=7, color='blue',  label=r'Inphase linear Response Stable')
# #
# # [lineOmegUnstIn] = ax0.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
# #  					tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# # 					globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
# # 					 					tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegUnst'],
# # 						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegUnst']/np.asarray(wnormy),
# # 						  'o',ms=1, color='blue', label=r'Inphase linear Response Unstable')
# # #
# #
# # [lineOmegStabAprox] = ax.plot(np.asarray(1.0/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['OmegaAproximation']/np.asarray(wnormy),
# # 						'o',ms=1, color='purple', alpha=0.3,  label=r'Aproximation(-cos) linear Response Stable')
# #
# [lineOmegStabAnti] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['tauStab']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# 		globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['OmegStab'],
# 						 solveLinStab(globalFreq(w,Kvco, AkPD, Ga1,tauf, v, digital, maxp, inphase2, INV)['Omeg'], globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['OmegStab']/np.asarray(wnormy),
# 						'+',ms=7, color='blue',  label=r'Anti-phase linear Response Stable')
#
# # [lineOmegUnstAnti] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],
# # 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['tauUnst']*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# # 	globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
# # 						tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['OmegUnst'],
# # 						 solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'],globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['OmegUnst']/np.asarray(wnormy),
# # 						  '+',ms=1, color='blue', label=r'Anti-phase linear Response Unstable')
# #
# #
# # print(solveLinStabNonliner(globalFreqNonlinear(w,tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# # globalFreqNonlinear(w,  tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ReMax'])
#
# [lineOmegStabInNon] = ax0.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# 						globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauStab']*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# 												globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegStab'],
# 						 solveLinStabNonliner(globalFreqNonlinear(w,tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# 						 globalFreqNonlinear(w,  tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegStab']/np.asarray(wnormy),
# 						 'o',ms=7, color='red',  label=r'Inphase Stable Nonlinear Response')
# #
# # [lineOmegUnstInNon] = ax0.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# # 								globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['tauUnst']*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# # 																globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegUnst'],
# # 						solveLinStabNonliner(globalFreqNonlinear(w,  tauf, v,Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# # 						 globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['OmegUnst']/np.asarray(wnormy),
# # 						 'o',ms=1, color='red', label=r'Inphase Unstable Nonlinear Response')
#
# #
# [lineOmegStabAntiNon] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
# 						globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauStab']*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
# 												globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegStab'],
# 						 solveLinStabNonliner(globalFreqNonlinear(w,tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
# 						 globalFreqNonlinear(w,  tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegStab']/np.asarray(wnormy),
# 						 '+',ms=7, color='red',  label=r'Anti-phase Stable Nonlinear Response')
# #
# [lineOmegUnstAntiNon] = ax.plot(np.asarray(1.0/(2.0*np.pi))*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
# 								globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['tauUnst']*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
# 																globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegUnst'],
# 						 solveLinStabNonliner(globalFreqNonlinear(w,  tauf, v,Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
# 						 globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['OmegUnst']/np.asarray(wnormy),
# 						 '+',ms=1, color='red', label=r'Anti-phase Unstable Nonlinear Response')
#
#


#
#
# [lineOmegStabAproxNon] = ax.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['OmegaAproximation']/np.asarray(wnormy),
# 						'+',ms=1, color='grey', alpha=0.3,  label=r'Aproximation(-cos) Nonlinear Response Stable')



#
# #
# #
# #
#
#
#
fig1         = plt.figure(figsize=(figwidth,figheight))
ax1          = fig1.add_subplot(111)
# plot grid, labels, define intial values
plt.title(r'Analysis of full expression of characteristic equation', fontsize=20)
plt.grid()
if wnormx == w:
	plt.xlabel(r'$\omega\tau/2\pi$', fontsize=20)
else:
	plt.xlabel(r'$\tau$', fontsize=20)
plt.ylabel(r'$2\pi\sigma/\omega$', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=10)
# draw the initial plot
[lineSigmaIn] = ax1.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['Omeg'],
	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'],
	 linewidth=4, color='blue', label=r' (linear freq response, Inphase)')


[lineSigmaAnti] = ax1.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2,INV)['Omeg'],
	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase2, expansion, INV)['Re'],
	 linewidth=1, color='blue', label=r' (linear freq response, Antiphase)')
#
# [lineGammaIn] = ax1.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Im'],
# 	'.', ms=4, color='blue', label=r'$\gamma$=Im$(\lambda)$ (in-phase)')

#																																										solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'], globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)
[lineSigmaInNon] = ax1.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],np.asarray(2.0*np.pi/w)*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
	globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf,  tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ReMax'],
	linewidth=4, color='red', label=r'(Nonlinear freq response, Inphase)')
[lineSigmaAntiNon] = ax1.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],np.asarray(2.0*np.pi/w)*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'],
	globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau'],Vbias, AkPD, Ga1, tauf,  tauc, v, order, digital, maxp, inphase2, expansion, INV, zeta)['ReMax'],
	linewidth=1, color='red', label=r'(Nonlinear freq response, Antiphase)')
#
# plt.hlines(0,0 , max(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']), colors='k', linestyles='dashed')
# # #
#
#
#
# fig1.set_size_inches(17,8.5)
#
# plt.savefig('plots/3rdGenSigma_tau_Nonlinear_v_%.2f.svg' %(v), dpi=150, bbox_inches=0)
# plt.savefig('plots/3rdGenSigma_tau_Nonlinear_v_%.2f.png' %(v), dpi=150, bbox_inches=0)
# #
# # fig2         = plt.figure(figsize=(figwidth,figheight))
# ax2          = fig2.add_subplot(111)
# # plot grid, labels, define intial values
# plt.title(r'Analysis of full expression of characteristic equation')
# plt.grid()
# if wnormx == w:
# 	plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=18)
# else:
# 	plt.xlabel(r'$\tau$', fontsize=18)
# plt.ylabel(r'$2\pi\sigma/\omega$', fontsize=18)
#
# # draw the initial plot
# [lineSigmaIn] = ax2.plot(np.asarray(1/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'], np.asarray(2.0*np.pi/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['Omeg'],
# 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'],
# 	 linewidth=4, color='blue', label=r' (linear freq response, Inphase)')
#
# #
# # # [lineGammaIn] = ax1.plot(np.asarray(wnormx/(2.0*np.pi))*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], np.asarray(1.0/w)*solveLinStab(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
# # # 	globalFreq(w,Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1,INV)['tau'], tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Im'],
# # # 	'.', ms=4, color='blue', label=r'$\gamma$=Im$(\lambda)$ (in-phase)')
# #
# #																																										solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'], globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf, tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)
# [lineSigmaInNon] = ax2.plot(np.asarray(1/(2.0*np.pi))*globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']*globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],np.asarray(2.0*np.pi/w)*solveLinStabNonliner(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'],
# 	globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'],Vbias, AkPD, Ga1, tauf,  tauc, v, order, digital, maxp, inphase1, expansion, INV, zeta)['ReMax'],
# 	linewidth=4, color='red', label=r'(Nonlinear freq response, Inphase)')
#

# ax0.legend()
ax.legend()
# ax1.legend()
# ax2.legend()
#ax5.legend()
plt.draw()
plt.show()
