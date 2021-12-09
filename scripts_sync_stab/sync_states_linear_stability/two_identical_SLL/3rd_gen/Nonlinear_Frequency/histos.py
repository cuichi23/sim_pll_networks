import LiStaGloF_lib
from LiStaGloF_lib import solveLinStab, globalFreq, linStabEq,globalFreqNonlinear, solveLinStabNonliner,NonlinearFreqResponse, globalFreqComparstor, distances #,globalFreqKrange
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
import math
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
v		= 32;					# the division

Vbias  = 2.55		# DC voltage offset old: 2.49
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
	maxp=5
INV		= 1.0*np.pi				# Inverter
mindelay = 4.236E-9
threshold =0.72e8
if v==32:
	threshold2 =9.0e6
if v==128:
	threshold2 =0.80e6
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
linearModel=[];
NonlinearModel=[];
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


Normalised_distance512LinearNetMean=[]; Normalised_distance512NonLinearNetMean=[];
Normalised_distance128LinearNetMean=[]; Normalised_distance128NonLinearNetMean=[];
Normalised_distance32LinearNetMean=[]; Normalised_distance32NonLinearNetMean=[];
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

		OmegaTau=math.ceil(2.0*np.pi*meanFreqN032[i]*delay32[i]/(v*np.pi))
		# print(meanFreqVCON512[i])
		# print(delay32[i])
		# print(OmegaTau)
		# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ])
		# print(delay32[i])
		# print(OmegaTau)

		if (OmegaTau % 2) != 0:
		# if abs(np.mean(DatasetN032['VCO_Freq_NodeA'][i])*np.array(1.0e6)-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD,
		# 	Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi) ) < threshold and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
		# 					   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
		# 					tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay32[i]))[-1][-1] ] < 0.0:
			inphase=True;
			linearModel.append(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v))
			NonlinearModel.append( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v) )

			# print('Inphase')
			distanceLinear32Net.append(abs(DatasetN032['Net_Freq_NodeA'][i]*np.array(1.0e6)-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v) ) )
			distanceNonLinear32Net.append(abs(DatasetN032['Net_Freq_NodeA'][i]*np.array(1.0e6)-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v) ) )
		else:
			inphase=False;
			# print('Antiphase')
			linearModel.append(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v))
			NonlinearModel.append( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v) )

			distanceLinear32Net.append(abs(DatasetN032['Net_Freq_NodeA'][i]*np.array(1.0e6)-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v) ) )
			distanceNonLinear32Net.append(abs(DatasetN032['Net_Freq_NodeA'][i]*np.array(1.0e6)-globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v) ) )



		distance32LinearNetMean.append(np.mean(distanceLinear32Net[i]))
		distance32NonLinearNetMean.append(np.mean(distanceNonLinear32Net[i]))
		distance32LinearNetSTD.append( np.std(distanceLinear32Net[i]) )
		distance32NonLinearNetSTD.append( np.std(distanceNonLinear32Net[i]) )
		Normalised_distance32LinearNetMean.append(np.mean(distanceLinear32Net[i])/linearModel[i])
		Normalised_distance32NonLinearNetMean.append(np.mean(distanceNonLinear32Net[i])/NonlinearModel[i])
		print(delay32[i], distance32LinearNetMean[i],distance32NonLinearNetMean[i],distance32LinearNetMean[i]-distance32NonLinearNetMean[i],distance32LinearNetSTD[i],distance32NonLinearNetSTD[i],linearModel[i], NonlinearModel[i],Normalised_distance32LinearNetMean[i], Normalised_distance32NonLinearNetMean[i] )
# 		# print(delay32[i],distance32LinearNetMean[i],distance32NonLinearNetMean[i],distance32LinearNetMean[i]-distance32NonLinearNetMean[i],distance32LinearNetSTD[i],distance32NonLinearNetSTD[i])
#
# # print('32')
	print(np.mean(distance32LinearNetMean))
	print(np.mean(distance32NonLinearNetMean))
	print(np.mean(distance32LinearNetMean)-np.mean(distance32NonLinearNetMean))
	print(np.mean(Normalised_distance32LinearNetMean))
	print(np.mean(Normalised_distance32NonLinearNetMean))
	# #
	print(np.mean(Normalised_distance32LinearNetMean)-np.mean(Normalised_distance32NonLinearNetMean) )
	print(np.mean(Normalised_distance32LinearNetMean)/np.mean(Normalised_distance32NonLinearNetMean) )

		# print(delay32[i],distance32LinearNetMean[i],distance32NonLinearNetMean[i],distance32LinearNetMean[i]-distance32NonLinearNetMean[i],distance32LinearNetSTD[i],distance32NonLinearNetSTD[i])

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

		OmegaTau=math.ceil(2.0*np.pi*meanFreqN128[i]*delay128[i]/(v*np.pi))
		# print(meanFreqVCON128[i])
		# print(delay128[i])
		# print(OmegaTau)
		# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ])
		# print(delay32[i])
		# print(OmegaTau)

		if (OmegaTau % 2) != 0:
		# if (abs(np.mean(DatasetN128['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi*v) ) )<= threshold2 and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
		# 				   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
		# 				tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay128[i]))[-1][-1] ] < 0.0:
			inphase=True;
			linearModel.append(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi*v))
			NonlinearModel.append( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi*v) )

			# print('Inphase')
			distanceLinear128Net.append(abs(DatasetN128['Net_Freq_NodeA'][i]-linearModel[i] ) )
			distanceNonLinear128Net.append(abs(DatasetN128['Net_Freq_NodeA'][i]-NonlinearModel[i] ) )
		else:
			inphase=False;
			linearModel.append(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi*v))
			NonlinearModel.append( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi*v) )

			# print('Antiphase')
			distanceLinear128Net.append(abs(DatasetN128['Net_Freq_NodeA'][i]-linearModel[i] ) )
			distanceNonLinear128Net.append(abs(DatasetN128['Net_Freq_NodeA'][i]-NonlinearModel[i] ) )

		distance128LinearNetMean.append(np.mean(distanceLinear128Net[i]))
		distance128NonLinearNetMean.append(np.mean(distanceNonLinear128Net[i]))
		distance128LinearNetSTD.append( np.std(distanceLinear128Net[i]) )
		distance128NonLinearNetSTD.append( np.std(distanceNonLinear128Net[i]) )
		Normalised_distance128LinearNetMean.append(np.mean(distanceLinear128Net[i])/linearModel[i])
		Normalised_distance128NonLinearNetMean.append(np.mean(distanceNonLinear128Net[i])/NonlinearModel[i])
		print(delay128[i], distance128LinearNetMean[i],distance128NonLinearNetMean[i],distance128LinearNetMean[i]-distance128NonLinearNetMean[i],distance128LinearNetSTD[i],distance128NonLinearNetSTD[i],linearModel[i], NonlinearModel[i],Normalised_distance128LinearNetMean[i], Normalised_distance128NonLinearNetMean[i] )
# 		# print(delay128[i],distance128LinearNetMean[i],distance128NonLinearNetMean[i],distance128LinearNetMean[i]-distance128NonLinearNetMean[i],distance128LinearNetSTD[i],distance128NonLinearNetSTD[i])
#
# # print('128')
	print(np.mean(distance128LinearNetMean))
	print(np.mean(distance128NonLinearNetMean))
	print(np.mean(distance128LinearNetMean)-np.mean(distance128NonLinearNetMean))
	print(np.mean(Normalised_distance128LinearNetMean))
	print(np.mean(Normalised_distance128NonLinearNetMean))
	# #
	print(np.mean(Normalised_distance128LinearNetMean)-np.mean(Normalised_distance128NonLinearNetMean) )
	print(np.mean(Normalised_distance128LinearNetMean)/np.mean(Normalised_distance128NonLinearNetMean) )

		# print(delay128[i],distance128LinearNetMean[i],distance128NonLinearNetMean[i],distance128LinearNetMean[i]-distance128NonLinearNetMean[i],distance128LinearNetSTD[i],distance128NonLinearNetSTD[i])

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
		OmegaTau=math.ceil(2.0*np.pi*meanFreqN512[i]*delay512[i]/(v*np.pi))
		# print(meanFreqN128[i])
		# print(delay512[i])
		# print(OmegaTau)
		# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ])
		# print(delay32[i])
		# print(OmegaTau)

		if (OmegaTau % 2) != 0:

		# if (abs(np.mean(DatasetN512['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi*v) ) )<= threshold and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
		# 				   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
		# 				tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay512[i]))[-1][-1] ] < 0.0:
			inphase=True;
			linearModel.append(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi*v))
			NonlinearModel.append( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi*v) )

			# print('Inphase')
			distanceLinear512Net.append(abs(DatasetN512['Net_Freq_NodeA'][i]-linearModel[i] ) )
			distanceNonLinear512Net.append(abs(DatasetN512['Net_Freq_NodeA'][i]-NonlinearModel[i] ) )
		else:
			inphase=False;
			linearModel.append(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi*v))
			NonlinearModel.append( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi*v) )

			# print('Antiphase')
			distanceLinear512Net.append(abs(DatasetN512['Net_Freq_NodeA'][i]-linearModel[i] ) )
			distanceNonLinear512Net.append(abs(DatasetN512['Net_Freq_NodeA'][i]-NonlinearModel[i] ) )

		distance512LinearNetMean.append(np.mean(distanceLinear512Net[i]))
		distance512NonLinearNetMean.append(np.mean(distanceNonLinear512Net[i]))
		Normalised_distance512LinearNetMean.append(np.mean(distanceLinear512Net[i])/linearModel[i])
		Normalised_distance512NonLinearNetMean.append(np.mean(distanceNonLinear512Net[i])/NonlinearModel[i])
		distance512LinearNetSTD.append( np.std(distanceLinear512Net[i]) )
		distance512NonLinearNetSTD.append( np.std(distanceNonLinear512Net[i]) )
		print(delay512[i], distance512LinearNetMean[i],distance512NonLinearNetMean[i],distance512LinearNetMean[i]-distance512NonLinearNetMean[i],distance512LinearNetSTD[i],distance512NonLinearNetSTD[i],linearModel[i], NonlinearModel[i],Normalised_distance512LinearNetMean[i], Normalised_distance512NonLinearNetMean[i] )
# 		# print(delay512[i],distance512LinearNetMean[i],distance512NonLinearNetMean[i],distance512LinearNetMean[i]-distance512NonLinearNetMean[i],distance512LinearNetSTD[i],distance512NonLinearNetSTD[i])
#
# # print('512')
	print(np.mean(distance512LinearNetMean))
	print(np.mean(distance512NonLinearNetMean))
	print(np.mean(distance512LinearNetMean)-np.mean(distance512NonLinearNetMean) )
	print(np.mean(Normalised_distance512LinearNetMean))
	print(np.mean(Normalised_distance512NonLinearNetMean))
	# #
	print(np.mean(Normalised_distance512LinearNetMean)-np.mean(Normalised_distance512NonLinearNetMean) )
	print(np.mean(Normalised_distance512LinearNetMean)/np.mean(Normalised_distance512NonLinearNetMean) )

# # print(distance512LinearNetSTD, distance512NonLinearNetSTD)
# print(np.mean(distanceLinear512Net)-np.mean(distanceNonLinear512Net))



if v==32:


	for i in range(len(DatasetN032['VCO_Freq_NodeA'])):
		meanFreqVCON032.append(np.mean(DatasetN032['VCO_Freq_NodeA'][i]))
		stdVCON032.append(np.std(DatasetN032['VCO_Freq_NodeA'][i]))
		# print(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau'])
		# print(globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'])
		OmegaTau=math.ceil(2.0*np.pi*meanFreqVCON032[i]*delay32[i]/(v*np.pi))
		# print(meanFreqVCON512[i])
		# print(delay32[i])
		# print(OmegaTau)
		# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ])
		# print(delay32[i])
		# print(OmegaTau)

		if (OmegaTau % 2) != 0:
		# if (abs(np.mean(DatasetN032['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi) ) )<= threshold and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
		# 				   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
		# 				tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay32[i]))[-1][-1] ] < 0.0:
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

		OmegaTau=math.ceil(2.0*np.pi*meanFreqVCON128[i]*delay128[i]/(v*np.pi))
		# print(meanFreqVCON512[i])
		# print(delay128[i])
		# print(OmegaTau)
		# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ])
		# print(delay32[i])
		# print(OmegaTau)

		if (OmegaTau % 2) != 0:
		# if (abs(np.mean(DatasetN128['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi) ))<= threshold and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
		# 				   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
		# 				tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay128[i]))[-1][-1] ] < 0.0:
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
# print(DatasetN032['VCO_Freq_NodeA'])
if v==512:

	for i in range(len(DatasetN512['VCO_Freq_NodeA'])):
		meanFreqVCON512.append(np.mean(DatasetN512['VCO_Freq_NodeA'][i]))
		stdVCON512.append(np.std(DatasetN512['VCO_Freq_NodeA'][i]))

		OmegaTau=math.ceil(2.0*np.pi*meanFreqVCON512[i]*delay512[i]/(v*np.pi))
		# print(meanFreqVCON512[i])
		# print(delay512[i])
		# print(OmegaTau)
		# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ])
		# print(delay32[i])
		# print(OmegaTau)

		if (OmegaTau % 2) != 0:
				# if (abs(np.mean(DatasetN512['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi) ) )<= threshold and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
				# 		   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
				# 		tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay512[i]))[-1][-1] ] < 0.0:
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
print(NonlinearFreqResponse(Vbias, AkPD, Ga1, 0, np.pi/2, INV, digital, inphase2)/(2*np.pi) 	)
wfreePLLNonLinear=NonlinearFreqResponse(Vbias, AkPD, Ga1, 0, np.pi/2, INV, digital, inphase2)

#
# fig0         = plt.figure(figsize=(figwidth,figheight))
# ax0          = fig0.add_subplot(111)
# for i in range(len(DatasetN128['Net_Freq_NodeA'])):
# 	[lineOmegStabIn] = ax0.plot(delay128[i], DatasetN128['Net_Freq_NodeA'],
# 						'o',ms=7, color='blue',  label=r'Inphase linear Response Stable')
# #
if v==32:
	for i in range(len(DatasetN512['Net_Freq_NodeA'])):
		fig       = plt.figure(figsize=(figwidth,figheight))
	# ax2          = fig2.add_subplot(111)
	#

		plt.style.use('ggplot')

		plt.hist(DatasetN032['Net_Freq_NodeA'][i]*np.array(1.0e6), bins=35,  label=r'Measurements')
		plt.axvline(x=np.mean(DatasetN032['Net_Freq_NodeA'][i])*np.array(1.0e6), color='green', linestyle='--',  label=r'Measurements (mean value)')
# (abs(np.mean(DatasetN032['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi) ) )<= threshold
		n=0
		OmegaTau=math.ceil(2.0*np.pi*meanFreqVCON032[i]*delay32[i]/(v*np.pi))
		# print(meanFreqVCON032[i])
		# print(delay32[i])
		# print(OmegaTau)
		if (OmegaTau % 2) != 0:


		# if (
		#
		#  and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
		# 			   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
		# 			tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay32[i]))[-1][-1] ] < 0.0:
			plt.title(r'$\tau=$%2.2E s, Inphase=%s ' %(delay32[i], 'True') );
			plt.axvline(x=globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v ), color='k', linestyle='--',  label=r' Linear Model')
			plt.axvline(x=globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ]/(2.0*np.pi*v ), color='blue', linestyle='-',  label=r'Nonlinear Model')

		else:
			plt.title(r'$\tau=$%2.2E s, Inphase=%s ' %(delay32[i], 'False') );
			plt.axvline(x=globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau']<= delay32[i])[-1][-1] ]/(2.0*np.pi*v ), color='blue', linestyle='-',  label=r'Nonlinear Model')
			plt.axvline(x=globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ] /(2.0*np.pi*v ), color='k', linestyle='--',  label=r' Linear Model')

		plt.legend()
		plt.savefig('histos/division_%.7f_delay_%.12f.png' %(v,delay32[i]), dpi=150, bbox_inches=0)


if v==128:
	for i in range(len(DatasetN128['Net_Freq_NodeA'])):
		fig       = plt.figure(figsize=(figwidth,figheight))
	# ax2          = fig2.add_subplot(111)
	#

		plt.style.use('ggplot')

		plt.hist(DatasetN128['Net_Freq_NodeA'][i], bins=35,  label=r'Measurements')
		plt.axvline(x=np.mean(DatasetN128['Net_Freq_NodeA'][i]), color='green', linestyle='--',  label=r'Measurements (mean value)')
		OmegaTau=math.ceil(2.0*np.pi*meanFreqVCON128[i]*delay128[i]/(v*np.pi))
		# print(meanFreqVCON128[i])
		# print(delay128[i])
		# print(OmegaTau)
		if (OmegaTau % 2) != 0:


		# if abs(np.mean(DatasetN128['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD,
		# 			Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi) ) < threshold and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
		# 							   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
		# 							tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay128[i]))[-1][-1] ] < 0.0:
			plt.title(r'$\tau=$%2.2E s, Inphase=%s ' %(delay128[i], 'True') );
			plt.axvline(x=globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi*v ), color='k', linestyle='--',  label=r' Linear Model')
			plt.axvline(x=globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ]/(2.0*np.pi*v ), color='blue', linestyle='-',  label=r'Nonlinear Model')

		else:
			plt.title(r'$\tau=$%2.2E s, Inphase=%s ' %(delay128[i], 'False') );
			plt.axvline(x=globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau']<= delay128[i])[-1][-1] ]/(2.0*np.pi*v ), color='blue', linestyle='-',  label=r'Nonlinear Model')
			plt.axvline(x=globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay128[i])[-1][-1] ] /(2.0*np.pi*v ), color='k', linestyle='--',  label=r' Linear Model')

		plt.legend()
		plt.savefig('histos/division_%.7f_delay_%.12f.png' %(v,delay128[i]), dpi=150, bbox_inches=0)

if v==512:
	for i in range(len(DatasetN512['Net_Freq_NodeA'])):
		fig       = plt.figure(figsize=(figwidth,figheight))
	# ax2          = fig2.add_subplot(111)
	#

		plt.style.use('ggplot')

		plt.hist(DatasetN512['Net_Freq_NodeA'][i], bins=35,  label=r'Measurements')
		plt.axvline(x=np.mean(DatasetN512['Net_Freq_NodeA'][i]), color='green', linestyle='--',  label=r'Measurements (mean value)')
		OmegaTau=math.ceil(2.0*np.pi*meanFreqVCON512[i]*delay512[i]/(v*np.pi))
		# print(meanFreqVCON512[i])
		# print(delay512[i])
		# print(OmegaTau)
		# print(globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay32[i])[-1][-1] ])
		# print(delay32[i])
		# print(OmegaTau)

		if (OmegaTau % 2) != 0:
		# if abs(np.mean(DatasetN512['VCO_Freq_NodeA'][i])-globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD,
		# 			Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi) ) < threshold and solveLinStab(globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'],
		# 							   globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau'],
		# 							tauf, Kvco, AkPD, Ga1, tauc, v, order, digital, maxp, inphase1, expansion, INV)['Re'][np.where( globalFreq(w, Kvco, AkPD,Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= np.asarray(delay512[i]))[-1][-1] ] < 0.0:
		# 	# inphase=True;
			plt.title(r'$\tau=$%2.2E s, Inphase=%s ' %(delay512[i], 'True') );
			plt.axvline(x=globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi*v ), color='k', linestyle='--',  label=r' Linear Model')
			plt.axvline(x=globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ]/(2.0*np.pi*v ), color='blue', linestyle='-',  label=r'Nonlinear Model')
		else:
			# inphase=False;
			plt.title(r'$\tau=$%2.2E s, Inphase=%s ' %(delay512[i], 'False') );
			plt.axvline(x=globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreqNonlinear(w, tauf, v, Vbias, AkPD, Ga1, digital, maxp, inphase2, INV)['tau']<= delay512[i])[-1][-1] ]/(2.0*np.pi*v ), color='blue', linestyle='-',  label=r'Nonlinear Model')
			plt.axvline(x=globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase2, INV)['Omeg'][np.where( globalFreq(w, Kvco, AkPD, Ga1, tauf, v, digital, maxp, inphase1, INV)['tau']<= delay512[i])[-1][-1] ] /(2.0*np.pi*v ), color='k', linestyle='--',  label=r' Linear Model')

		plt.legend()
		plt.savefig('histos/division_%.7f_delay_%.12f.png' %(v,delay512[i]), dpi=150, bbox_inches=0)


plt.draw()
plt.show()
