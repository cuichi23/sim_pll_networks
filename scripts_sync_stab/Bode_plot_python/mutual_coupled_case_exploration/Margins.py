from pylab import *
import sympy
from numpy import pi, sin
import numpy as np
import sympy
from scipy import signal
import matplotlib.pyplot as plt
import datetime
now = datetime.datetime.now()
from itertools import count
import Bode_lib
# from Bode_lib import Holnodel,Hcl,Holwithdel,fMolnodel,fMcl, fMolwithdel
from Bode_lib import solveLinStab, globalFreq, linStabEq, analytical,HopenloopMutuallyCoupledOnePLL,K,LoopGainSteadyState, coupfununction, K
from Bode_lib import PhaseopenloopMutuallyCoupledOnePLL,PhaseclosedloopMutuallyCoupledOnePLL,HclosedloopMutuallyCoupledOnePLL, GainMarginMutuallyCoupledOne, PhaseMarginMutuallyCoupledOne
from Bode_lib import HopenloopMutuallyCoupledNet, PhaseopenloopMutuallyCoupledNet,HclosedloopMutuallyCoupledNet, PhaseclosedloopMutuallyCoupledNet, HopenloopMutuallyCoupledNet2
from Bode_lib import GainMarginMutuallyCoupledNet,PhaseMarginMutuallyCoupledNet
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# plot parameter
''' All plots in latex mode '''
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.rcParams['agg.path.chunksize'] = 10000

''' STYLEPACKS '''
titlefont = {
  'family' : 'serif',
  'color'  : 'black',
  'weight' : 'normal',
  'size'   : 9,
  }

labelfont = {
  'family' : 'sans-serif',
  # 'color'  : 'black',
  'weight' : 'normal',
  'style'  : 'normal',
  'size'   : 36,
  }

annotationfont = {
  'family' : 'monospace',
  'color'  : (0, 0.27, 0.08),
  'weight' : 'normal',
  'size'   : 14,
  }

# plot parameter
axisLabel = 50;
titleLabel= 10;
dpi_val   = 150;
figwidth  = 10;
figheight = 5;

####################################################################################################################################################################################
coupfun='cos'

####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
#
w       = 2.0*np.pi;                                                            # intrinsic    frequency
Kvco    = 2.0*np.pi*(0.8);                                                      # Sensitivity of VCO
AkPD    = 1.0                                                                   # amplitude of the PD -- the factor of 2 here is related to the IMS 2020 paper, AkPD as defined in that paper already includes the 0.5 that we later account for when calculating K
GkLF    = 1.0
Gvga    = 1.0
Ga1     = 1.0                                                                   # Gain of the first adder
order   = 1.0                                                                   # the order of the Loop Filter
tauf    = 0.0                                                                   # tauf = sum of all processing delays in the feedback
tauc    = 1.0/(2.0*np.pi*0.0146);                                               # the integration time of the Loop Filter tauc=1/wc=1/(2πfc), fc the cutoff frequency of the Loop Filter
v       = 1                                                                     # the division
c       = 0.63*3E8                                                              # speed of light
maxp    = 10.5
INV     = 0.0*np.pi
model   = 'Nonlinear'
#
#
wref=w
w_cutoff      = 1.0/tauc



# choose digital vs analog
digital = False;

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;
# choose phase or anti-phase synchroniazed states,

sync_state1='inphase';                                                            # choose between inphase, antiphase, entrainment
sync_state2='antiphase';
sync_state3='entrainment';
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, visor of vider

				#2.0*np.pi*24E9;
# K    = 0.15*2.0*np.pi            #2.0*np.pi*250E6;
# tauf = tau_f
# tauc = tau_c;
c     = 3E8*(2/3);
# maxp = 1e7;
###### python library for bode plots (lti)
# list_of_coefficients
# system        = signal.lti([alpha], [v*tau_c, v, alpha])
df        = 0.0001;
f         = np.logspace(-4.0, 1, num=int((tauf+maxp)/df), endpoint=True, base=10.0)

####################################################################################################################################################################################

tau1=    0.9
sec=[];
gamma_gain_margin_One=[]; gamma_phase_margin_One=[];PM_One=[];GM_One=[];tauOne=[];
gamma_phase_margin_Net=[];PM_Net=[];GM_Net=[];tauNet=[];OmegaNet=[];OmegaOne=[];
gain_cross_freq_One=[];gain_cross_freq_Net=[]; phase_cross_freq_One=[];
GM_One_Lin=[];  PM_One_Lin=[];tauOne_Lin=[];OmegaOne_Lin=[];

PhaseOnePLL        = np.vectorize(PhaseclosedloopMutuallyCoupledOnePLL)
PhaseNet           = np.vectorize(PhaseopenloopMutuallyCoupledNet)
# PhaseclosedloopMutuallyCoupledNet

#Here we calculate and save to lists the gain and phase crossover frequencies and gain and phase margin as well as functions of the delay

for j in range(len(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'])):
	# print(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j])
	# for i in range(len(f)):
	gain_cross_freq_One.append(f[np.where((abs(HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))) <=0.0001)])
	PM=f[np.where(20.0*log10(abs(HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))) <=0.0001)][0]

	PM_One.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*PM, globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))



	gain_cross_freq_Net.append(f[np.where((abs(HopenloopMutuallyCoupledNet(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))) <=0.0001)][-1] )
	PMNet=f[np.where(20.0*log10(abs(HopenloopMutuallyCoupledNet(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))) <=0.0001)][-1]

	PM_Net.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledNet(2.0*np.pi*PMNet, globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))


	phase_cross_freq_One.append(f[np.where((abs(180.0+PhaseOnePLL(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))) <0.7)] )

	if (abs(180.0+PhaseOnePLL(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model)) <=0.7).any():

		GM=f[np.where(abs(180.0+PhaseOnePLL(2.0*np.pi*f,
			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
			tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model)) <=0.7)][0]
		# print(GM)
		GM_One.append(GainMarginMutuallyCoupledOne(2.0*np.pi*GM, globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
			tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))
		tauOne.append(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j])
		OmegaOne.append(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j])

	if (abs(PhaseNet(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model)) <=0.01).any():

		GMNet=f[np.where(abs(PhaseNet(2.0*np.pi*f,
			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
			tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model)) <=0.01)][-1]
		# print(GM)
		GM_Net.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNet, globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
			tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))
		tauNet.append(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j])
		OmegaNet.append(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j])


######################################################################################################## LINEAR

	PMLin=f[np.where(20.0*log10(abs(HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear'))) <=0.0001)][0]

	PM_One_Lin.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*PMLin, globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear'))


	if (abs(180.0+PhaseOnePLL(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear')) <=0.7).any():

		GMLin=f[np.where(abs(180.0+PhaseOnePLL(2.0*np.pi*f,
			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
			tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear')) <=0.7)][0]
		# print(GM)
		GM_One_Lin.append(GainMarginMutuallyCoupledOne(2.0*np.pi*GMLin, globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j],
			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j],
			tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear'))

		tauOne_Lin.append(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j])
		OmegaOne_Lin.append(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j])












print('Hallo 1')
#


tauvalues2plot = [0*w/(2*np.pi), 0.25*w/(2*np.pi), 0.5*w/(2*np.pi), 0.75*w/(2*np.pi), 1.0*w/(2*np.pi), 1.25*w/(2*np.pi)]
bar = [];
for i in range(len(tauvalues2plot)):
	tempo = np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][:]>tauvalues2plot[i])[0][0]
	# print('tempo:', tempo)
	bar.append( tempo )
	# print('bar[i]', bar[i])




##Here we calculate and save to lists the gain and phase crossover frequencies and gain and phase margin as well as functions of the coupling strength
###########

# #
# Kvcovalues=np.arange(0.0001*np.pi, 0.8*np.pi, 0.01)
# PM_One_DifK=[];GM_One_DifK=[]; KKvalue=[];
# for j in range(len(Kvcovalues)):
# 	tempo1 = np.where(globalFreq(wref, w, Kvcovalues[j], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][:]>tauvalues2plot[1])[0][0]
#
# 	# print(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j])
# 	# for i in range(len(f)):
#
#
# 	PMDifK=f[np.where(20.0*log10(abs(HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
# 		globalFreq(wref, w, Kvcovalues[j], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
# 		globalFreq(wref, w, Kvcovalues[j], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
# 		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))) <=0.0001)][0]
#
# 	PM_One_DifK.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*PMDifK,
# 		globalFreq(wref, w, Kvcovalues[j], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
# 		globalFreq(wref, w, Kvcovalues[j], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
# 		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))
#
# 	if (abs(180.0+PhaseOnePLL(2.0*np.pi*f,
# 		globalFreq(wref, w, Kvcovalues[j], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
# 		globalFreq(wref, w, Kvcovalues[j], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
# 		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model)) <=0.7).any():
#
# 		GMDifK=f[np.where(abs(180.0+PhaseOnePLL(2.0*np.pi*f,
# 			globalFreq(wref, w, Kvcovalues[j], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
# 			globalFreq(wref, w, Kvcovalues[j], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
# 			tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model)) <=0.7)][0]
# 		# print(GM)
# 		GM_One_DifK.append(GainMarginMutuallyCoupledOne(2.0*np.pi*GMDifK,
# 			globalFreq(wref, w, Kvcovalues[j], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
# 			globalFreq(wref, w, Kvcovalues[j], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
# 			tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))
# 		KKvalue.append(Kvcovalues[i])
#
# 		# tauOne_Lin.append(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j])
# 		# OmegaOne_Lin.append(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j])
#
# print('Hallo 2')
#



##Here we calculate and save to lists the gain and phase crossover frequencies and gain and phase margin as well as functions of the cutoff frequency
###########

# wcvalues=np.arange(0.0001*2.0*np.pi, 0.8*2.0*np.pi, 0.01)
# PM_One_Difwc=[];GM_One_Difwc=[]; wcvalue=[];
# tempo1 = np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][:]>tauvalues2plot[1])[0][0]
#
# for j in range(len(wcvalues)):
#
# 	# print(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j])
# 	# for i in range(len(f)):
#
# 	PMDifwc=f[np.where(20.0*log10(abs(HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
# 		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
# 		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
# 		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))) <=0.0001)][0]
#
# 	PM_One_Difwc.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*PMDifwc,
# 		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
# 		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
# 		tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model))
#
# 	if (abs(180.0+PhaseOnePLL(2.0*np.pi*f,
# 		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
# 		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
# 		tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model)) <=0.7).any():
#
# 		GMDifwc=f[np.where(abs(180.0+PhaseOnePLL(2.0*np.pi*f,
# 			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
# 			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
# 			tauf, tauc, v, 1.0/wcvalues[j], AkPD, GkLF, Gvga,  digital, model)) <=0.7)][0]
# 		# print(GM)
# 		GM_One_Difwc.append(GainMarginMutuallyCoupledOne(2.0*np.pi*GMDifwc,
# 			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
# 			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
# 			tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model))
# 		wcvalue.append(wcvalues[j]/(2.0*np.pi))
#
#
#
#


# ADD PLOTS
####################################################################################################################################################################################


# tau1=    0.001

#bar= [1, 8, 30, 60]

colorbar=['blue','green','orange','purple', 'cyan', 'black']
# print(globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1]/(2*np.pi))
# print(globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1])
##################################################################################################################################################################
fig1= plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig2, (, ) = plt.subplots(2)

# print(180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))
ax1= plt.subplot(211)
plt.axhline(y=0, color='r', linestyle='-.')
# fig1.suptitle(r'magnitude and phase-response of a open loop $H_{ol}$ of One PLL in a network two mutually coupled PLLs')

# plt.title(r' $f_{\Omega}=%0.2f$Hz, $\tau=%.2f$s'%(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w1, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1]/(2.0*np.pi),tau1), fontsize=8)
# print(PhaseclosedloopMutuallyCoupledOnePLL(2.0*np.pi*f,
				# globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
				# tau1, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))
# plt.ylabel('dB', fontsize=axisLabel)
plt.xscale('log');
for i in range(len(bar)):
	[lineol00]     =    ax1.plot(f, 20.0*log10(abs(HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[i]],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))),'-',ms=1, color= colorbar[i], label=r'$\tau=$%.5f' %(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]]))
# [lineol00]     =    ax1.plot(f, 20.0*log10(abs(HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
#     globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
#     tau1,
#     tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))),'-', label=r'magnitude,',color='red')

#
# PhaseOnePLL(2.0*np.pi*f,
#                 globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
#                 tau1, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model),'-', label=r'magnitude,',color='red')

# [lineol00]     =    ax1.plot(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'], np.array(PM_One),'-', label=r'phase margin,',color='red');
plt.grid();
plt.ylabel(r'$20log\left|H_k^\textrm{OL}(j\gamma)\right|$', rotation=90,fontsize=45, labelpad=30)
ax1.legend(fontsize=15);
ax1.tick_params(axis='both', which='major', labelsize=25, pad=1)
ax2= plt.subplot(212)
plt.xscale('log')
					# PhaseclosedloopMutuallyCoupledOnePLL
# [lineol00]     =    ax1.plot(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'], np.array(gamma_phase_margin_One),'-', label=r'magnitude,',color='red');
# GM_One
plt.axhline(y=-180, color='r', linestyle='-.')
for i in range(len(bar)):
	[lineol00]     =    ax2.plot(f, PhaseOnePLL(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[i]],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model),'.',ms=1, color= colorbar[i], label=r'$\tau=$%.5f' %(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]]))

# [lineol00]     =    ax2.plot(f, PhaseOnePLL(2.0*np.pi*f,
#                 globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
#                 tau1, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model),'-', label=r'magnitude,',color='red')
# [lineol00]     =    ax1.plot((1.0/(2.0*np.pi))*globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg']*globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'], np.array(PM_One),':', label=r'GM,',color='red');
# [lineol01]     =    ax2.plot(np.array(tauOne), np.array(GM_One),'-', label=r'gain margin,',color='blue');
plt.grid();
plt.ylabel(r'phase', rotation=90,fontsize=45, labelpad=30)
plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=40,labelpad=-5)
ax2.tick_params(axis='both', which='major', labelsize=25, pad=1)

ax2.legend(fontsize=15)
fig1.set_size_inches(20,10)
if digital == True:
	plt.savefig('Plots/digital_bode_plot_One%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('Plots/digital_bode_plot_One%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfununction(coupfun)=='sin':
		plt.savefig('Plots/analog_sin_bode_plot_One%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('Plots/analog_sin_bode_plot_One%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfununction(coupfun)=='cos':
		plt.savefig('Plots/analog_cos_bode_plot_One%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('Plots/analog_cos_bode_plot_One%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)



##################################################################################################################################################################


#
fig2= plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig2, (, ) = plt.subplots(2)
#
# # print(180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))
ax3= plt.subplot(211)

plt.xscale('log');
#
# #
# # #
#
for i in range(len(bar)):
	[lineol00]     =    ax3.plot(f, 20.0*log10(abs(HopenloopMutuallyCoupledNet(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[i]],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))),'-',ms=1, color= colorbar[i], label=r'$\tau=$%.5f' %(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]]))

# [lineol00]     =    ax3.plot(f, 20.0*log10(abs(HopenloopMutuallyCoupledNet(2.0*np.pi*f,
#                 globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
#                 tau1, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))),'-', label=r'magnitude,',color='red')


plt.grid();
plt.ylabel(r'$20log \left|H_\textrm{net}^\textrm{OL}(j\gamma)\right|$', rotation=90,fontsize=45, labelpad=30)
ax3.tick_params(axis='both', which='major', labelsize=25, pad=1)
ax3.legend(fontsize=15);
ax4= plt.subplot(212)
plt.xscale('log');

# [lineoNet]     =    ax4.plot(np.array(tauNet), np.array(GM_Net),'-', label=r'gain margin,',color='blue');
for i in range(len(bar)):
	[lineol00]     =    ax4.plot(f, PhaseNet(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[i]],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model),'.',ms=1, color= colorbar[i], label=r'$\tau=$%.5f' %(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[i]]))

#
# [lineol00]     =    ax4.plot(f, PhaseNet(2.0*np.pi*f,
#                 globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']<=tau1)][-1],
#                 tau1, tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model),'.', label=r'phase,',color='red')

plt.grid();
plt.ylabel(r'phase', rotation=90,fontsize=45, labelpad=30)
plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=40,labelpad=-5)
ax4.tick_params(axis='both', which='major', labelsize=25, pad=1)

ax4.legend(fontsize=15)

fig2.set_size_inches(20,10)
if digital == True:
	plt.savefig('Plots/digital_bode_plot_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('Plots/digital_bode_plot_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfununction(coupfun)=='sin':
		plt.savefig('Plots/analog_sin_bode_plot_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('Plots/analog_sin_bode_plot_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfununction(coupfun)=='cos':
		plt.savefig('Plots/analog_cos_bode_plot_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('Plots/analog_cos_bode_plot_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)



##################################################################################################################################################################


fig3= plt.figure(num=3, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig2, (, ) = plt.subplots(2)
#
# # print(180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))
ax5= plt.subplot(211)

[line4]     =    ax5.plot(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']*globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg']/(2.0*np.pi), np.array(PM_One),'.',color='red', label=r'nonlinear');
[line44]     =    ax5.plot(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']*globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg']/(2.0*np.pi), np.array(PM_One_Lin),'.',color='black', label=r'linear');
plt.ylabel(r'phase margin', rotation=90,fontsize=45, labelpad=30)
ax5.tick_params(axis='both', which='major', labelsize=25, pad=1)
plt.grid();
ax5.legend()
ax6= plt.subplot(212)
#plt.xscale('log');

# [lineoNet]     =    ax4.plot(np.array(tauNet), np.array(GM_Net),'-', label=r'gain margin Net',color='blue');

[lineol06]     =    ax6.plot(np.array(tauOne)*np.array(OmegaOne)/(2.0*np.pi), np.array(GM_One),'.',color='blue', label=r'nonlinear');
[lineol66]     =    ax6.plot(np.array(tauOne_Lin)*np.array(OmegaOne_Lin)/(2.0*np.pi), np.array(GM_One_Lin),'-',color='black', label=r'nonlinear');
plt.ylabel(r'gain margin', rotation=90,fontsize=45, labelpad=30)
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=40,labelpad=-5)
ax6.tick_params(axis='both', which='major', labelsize=25, pad=1)
ax6.legend()
plt.grid();
if digital == True:
	plt.savefig('Plots/digital_Margins_One%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('Plots/digital_Margins_One%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfununction(coupfun)=='sin':
		plt.savefig('Plots/analog_sin_Margins_One%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('Plots/analog_sin_Margins_One%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfununction(coupfun)=='cos':
		plt.savefig('Plots/analog_cos_Margins_One%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('Plots/analog_cos_Margins_One%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)



##################################################################################################################################################################




fig4= plt.figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig2, (, ) = plt.subplots(2)
#
# # print(180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))
ax7= plt.subplot(211)

# [line7]     =    ax7.plot(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'], np.array(PM_One),'-', label=r'phase margin,',color='red');
[lineNet7]     =    ax7.plot(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg']*globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau']/(2.0*np.pi), np.array(PM_Net),'.', label=r'phase margin Net',color='red');
[lineSigmaIn] = ax7.plot(np.asarray(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg']/(2.0*np.pi))*globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'],
	np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'],
	globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, digital, maxp, sync_state1, expansion, INV)['ReMax'],
	 '-',linewidth=1, color='green', label=r'in-phase')
plt.grid();
plt.ylabel(r'phase margin')
# ax7.legend();
ax8= plt.subplot(212)
#plt.xscale('log');

[lineoNet]     =    ax8.plot(np.array(tauNet)*np.array(OmegaNet)/(2.0*np.pi), np.array(GM_Net),'.', label=r'gain margin Net',color='blue');
plt.ylabel(r'gain margin')
plt.xlabel(r'$\Omega\tau/2 \pi$')
# [lineol08]     =    ax6.plot(np.array(tauOne), np.array(GM_One),'-', label=r'gain margin,',color='blue');
[lineSigmaIn] = ax8.plot(np.asarray(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg']/(2.0*np.pi))*globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'],
	np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'],
	globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'], tauf, Kvco, AkPD, GkLF,Gvga, tauc, v, digital, maxp, sync_state1, expansion, INV)['ReMax'],
	 '-',linewidth=1, color='green', label=r'in-phase')
plt.grid();
# ax8.legend();
#
if digital == True:
	plt.savefig('Plots/digital_Margins_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('Plots/digital_Margins_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfununction(coupfun)=='sin':
		plt.savefig('Plots/analog_sin_Margins_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('Plots/analog_sin_Margins_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfununction(coupfun)=='cos':
		plt.savefig('Plots/analog_cos_Margins_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('Plots/analog_cos_Margins_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

#
#
#
#
# #########################################################################################################################################
#
# Kvcobar=[0.002, 0.02, 0.05,  0.1, 0.5, 0.8]
# fig5= plt.figure(num=5, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
#
# # print(180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))
# ax9= plt.subplot(211)
# plt.axhline(y=0, color='r', linestyle='-.')
#
# plt.xscale('log');
# for i in range(len(Kvcobar)):
# 	[lineol00]     =    ax9.plot(f, 20.0*log10(abs(HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
# 		globalFreq(wref, w, Kvcobar[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[1]],
# 		globalFreq(wref, w, Kvcobar[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[1]],
# 		tauf, tauc, v, Kvcobar[i], AkPD, GkLF, Gvga,  digital, model))),'-',ms=1, color= colorbar[i], label=r' K=%.5f' %(K(Kvcobar[i], AkPD, GkLF, Gvga)))
#
# plt.grid();
# plt.ylabel(r'$20log\left|H_k^\textrm{ OL}(j\gamma)\right|$', rotation=90,fontsize=45, labelpad=30)
# ax9.legend(fontsize=15);
# ax9.tick_params(axis='both', which='major', labelsize=25, pad=1)
#
#
# ######## Subplot
#
#
# ax10= plt.subplot(212)
# plt.xscale('log')
# plt.axhline(y=-180, color='r', linestyle='-.')
# 					# PhaseclosedloopMutuallyCoupledOnePLL
#
#
# for i in range(len(Kvcobar)):
# 	[lineol00]     =    ax10.plot(f, PhaseOnePLL(2.0*np.pi*f,
# 		globalFreq(wref, w, Kvcobar[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[1]],
# 		globalFreq(wref, w, Kvcobar[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[1]],
# 		tauf, tauc, v, Kvcobar[i], AkPD, GkLF, Gvga,  digital, model),'.',ms=1, color= colorbar[i], label=r'K=%.5f' %(K(Kvcobar[i], AkPD, GkLF, Gvga)))
#
# plt.grid();
# plt.ylabel(r'phase', rotation=90,fontsize=45, labelpad=30)
# plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=40,labelpad=-5)
# ax10.tick_params(axis='both', which='major', labelsize=25, pad=1)
#
# ax10.legend(fontsize=15)
# fig5.set_size_inches(20,10)
# if digital == True:
# 	plt.savefig('Plots/digital_bode_plot_One_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('Plots/digital_bode_plot_One_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# else:
# 	if coupfununction(coupfun)=='sin':
# 		plt.savefig('Plots/analog_sin_bode_plot_One_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('Plots/analog_sin_bode_plot_One_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# 	if coupfununction(coupfun)=='cos':
# 		plt.savefig('Plots/analog_cos_bode_plot_One_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('Plots/analog_cos_bode_plot_One_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
#
#
#
#
#
#
#
#
#
# #########################################################################################################################################
#
# wcbar=[0.002*2.0*np.pi, 0.02*np.pi, 0.05*np.pi,  0.1*np.pi, 0.5*np.pi, 0.8*np.pi]
# fig6= plt.figure(num=6, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
#
# # print(180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))
# ax11= plt.subplot(211)
# plt.axhline(y=0, color='r', linestyle='-.')
#
# plt.xscale('log');
# for i in range(len(wcbar)):
# 	[lineol00]     =    ax11.plot(f, 20.0*log10(abs(HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
# 		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[1]],
# 		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[1]],
# 		tauf, (1.0/wcbar[i]), v, Kvco, AkPD, GkLF, Gvga,  digital, model))),'-',ms=1, color= colorbar[i], label=r' fc=%.5f' %( wcbar[i]/(2.0*np.pi)))
#
# plt.grid();
# plt.ylabel(r'$20\log\left|H_k^\textrm{ OL}(j\gamma)\right|$', rotation=90,fontsize=45, labelpad=30)
# ax11.legend(fontsize=15);
# ax11.tick_params(axis='both', which='major', labelsize=25, pad=1)
#
#
# ######## Subplot
#
#
# ax12= plt.subplot(212)
# plt.xscale('log')
# plt.axhline(y=-180, color='r', linestyle='-.')
# 					# PhaseclosedloopMutuallyCoupledOnePLL
#
#
# for i in range(len(Kvcobar)):
# 	[lineol00]     =    ax12.plot(f, PhaseOnePLL(2.0*np.pi*f,
# 		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[1]],
# 		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[1]],
# 		tauf, (1.0/wcbar[i]), v, Kvco, AkPD, GkLF, Gvga,  digital, model),'.',ms=1, color= colorbar[i], label=r' fc=%.5f' %(wcbar[i]/(2.0*np.pi)))
#
# plt.grid();
# plt.ylabel(r'phase', rotation=90,fontsize=45, labelpad=30)
# plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=40,labelpad=-5)
# ax12.tick_params(axis='both', which='major', labelsize=25, pad=1)
#
# ax12.legend(fontsize=15)
# fig6.set_size_inches(20,10)
# if digital == True:
# 	plt.savefig('Plots/digital_bode_plot_One_different_wc_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 	plt.savefig('Plots/digital_bode_plot_One_different_wc_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# else:
# 	if coupfununction(coupfun)=='sin':
# 		plt.savefig('Plots/analog_sin_bode_plot_One_different_wc_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('Plots/analog_sin_bode_plot_One_different_wc_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# 	if coupfununction(coupfun)=='cos':
# 		plt.savefig('Plots/analog_cos_bode_plot_One_different_wc_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('Plots/analog_cos_bode_plot_One_different_wc_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
#
#
#
#





#
# fig7= plt.figure(num=7, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
#
# # print(180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))
# ax13= plt.subplot(211)
# 					# PhaseclosedloopMutuallyCoupledOnePLL
#
# [line4]     =    ax13.plot(K(Kvcovalues, AkPD, GkLF, Gvga)/(2.0*np.pi), np.array(PM_One_DifK),'.',color='red');
#
# plt.ylabel(r'phase margin', rotation=90,fontsize=45, labelpad=30)
# ax13.tick_params(axis='both', which='major', labelsize=25, pad=1)
# plt.grid();
# ax13.legend()
# ax14= plt.subplot(212)
# #plt.xscale('log');
#
# # [lineoNet]     =    ax4.plot(np.array(tauNet), np.array(GM_Net),'-', label=r'gain margin Net',color='blue');
#
# [line66]     =    ax14.plot(np.array(KKvalue)/(2.0*np.pi), np.array(GM_One_DifK),'.',color='blue');
#
# plt.ylabel(r'gain margin', rotation=90,fontsize=45, labelpad=30)
# plt.xlabel(r'$K$', fontsize=40,labelpad=-5)
# ax14.tick_params(axis='both', which='major', labelsize=25, pad=1)
# ax14.legend()
# plt.grid();
# if digital == True:
# 	plt.savefig('Plots/digital_Margins_OneVsK%d_%d_%d.pdf' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
# 	plt.savefig('Plots/digital_Margins_OneVsK%d_%d_%d.png' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
# else:
# 	if coupfununction(coupfun)=='sin':
# 		plt.savefig('Plots/analog_sin_Margins_OneVsK%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('Plots/analog_sin_Margins_OneVsK%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# 	if coupfununction(coupfun)=='cos':
# 		plt.savefig('Plots/analog_cos_Margins_OneVsK%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('Plots/analog_cos_Margins_OneVsK%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# #
# #
# #
# #
# fig8= plt.figure(num=8, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# # fig2, (, ) = plt.subplots(2)
#
# # print(180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))
# ax15= plt.subplot(211)
# 					# PhaseclosedloopMutuallyCoupledOnePLL
#
# [line4]     =    ax15.plot(wcvalues/(2.0*np.pi), np.array(PM_One_Difwc),'.',color='red');
#
# plt.ylabel(r'phase margin', rotation=90,fontsize=45, labelpad=30)
# ax15.tick_params(axis='both', which='major', labelsize=25, pad=1)
# plt.grid();
# ax15.legend()
# ax16= plt.subplot(212)
# #plt.xscale('log');
#
# # [lineoNet]     =    ax4.plot(np.array(tauNet), np.array(GM_Net),'-', label=r'gain margin Net',color='blue');
#
# [line66]     =    ax16.plot(np.array(wcvalue)/(2.0*np.pi), np.array(GM_One_Difwc),'.',color='blue');
#
# plt.ylabel(r'gain margin', rotation=90,fontsize=45, labelpad=30)
# plt.xlabel(r'$\omega_c$', fontsize=40,labelpad=-5)
# ax16.tick_params(axis='both', which='major', labelsize=25, pad=1)
# ax16.legend()
# plt.grid();
# if digital == True:
# 	plt.savefig('Plots/digital_Margins_OneVswc%d_%d_%d.pdf' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
# 	plt.savefig('Plots/digital_Margins_OneVswc%d_%d_%d.png' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
# else:
# 	if coupfununction(coupfun)=='sin':
# 		plt.savefig('Plots/analog_sin_Margins_OneVswc%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('Plots/analog_sin_Margins_OneVswc%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# 	if coupfununction(coupfun)=='cos':
# 		plt.savefig('Plots/analog_cos_Margins_OneVswc%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# 		plt.savefig('Plots/analog_cos_Margins_OneVswc%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
# #
# #
#
plt.show();
