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
import bode_lib
# from Bode_lib import Holnodel,Hcl,Holwithdel,fMolnodel,fMcl, fMolwithdel
from bode_lib import solveLinStabbetaanti, solveLinStabbetain, globalFreq, linStabEq, K, coupfunction

from bode_lib import HopenLoopMutuallyCoupledOnePLL1, LoopGainSteadyState1, PhaseopenloopMutuallyCoupledOnePLL1,  PhaseclosedloopMutuallyCoupledOnePLL1, HclosedLoopMutuallyCoupledOnePLL1, GainMarginMutuallyCoupledOne1, PhaseMarginMutuallyCoupledOne1
from bode_lib import HopenLoopMutuallyCoupledOnePLL2, LoopGainSteadyState2, PhaseopenloopMutuallyCoupledOnePLL2,  PhaseclosedloopMutuallyCoupledOnePLL2, HclosedLoopMutuallyCoupledOnePLL2, GainMarginMutuallyCoupledOne2, PhaseMarginMutuallyCoupledOne2
from bode_lib import HopenLoopMutuallyCoupledNet, PhaseopenloopMutuallyCoupledNet, HclosedloopMutuallyCoupledNet, PhaseclosedloopMutuallyCoupledNet
from bode_lib import GainMarginMutuallyCoupledNet, PhaseMarginMutuallyCoupledNet
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
  'size'   : 16,
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
figwidth  = 20;
figheight = 10;

####################################################################################################################################################################################
coupfun='cos'

####################################################################################################################################################################################
axis_color  = 'lightgoldenrodyellow'
#
w1   	 	= (1.0 - 0.0*0.02)*2.0*np.pi                                          # intrinsic frequency of PLL1
w2	 	    = (1.0 + 0.0*0.02)*2.0*np.pi                                          # intrinsic frequency of PLL2
wmean	 	= (w1+w2)/2.0
Dw	 	    = w2-w1
Kvco1    	= (0.470-0.0*0.02)*2.0*np.pi                                            # Sensitivity of VCO of PLL1
Kvco2    	= (0.470+0.0*0.02)*2.0*np.pi                                            # Sensitivity of VCO of PLL1
AkPD     	= 1.0
GkLF     	= 1.0
Gvga     	= 1.0		                                                          # Gain of the first adder
K1 		    = K(Kvco1, AkPD, GkLF, Gvga)
K2 		    = K(Kvco2, AkPD, GkLF, Gvga)
Kmean	 	= (K1+K2)/2.0
DK	 	    = K2-K1
tauf 		= 0.0
tauf1       = 0.0
tauf2       = 0.0
tauchomogen	= 1.0/((0.055 - 0.0*0.04)*2.0*np.pi);
tauc1 		= 1.0/((0.055 - 1.0*0.04)*2.0*np.pi);
tauc2 		= 1.0/((0.055 + 1.0*0.04)*2.0*np.pi);
wc1			= 1.0/tauc1
wc2			= 1.0/tauc2
Dwc			= wc2 -wc1
v           = 1                                                                    # the division
c           = 0.63*3E8                                                             # speed of light
maxp        = 9.90
INV         = 0.0*np.pi

# w_cutoff = 1.0/tauc

# choose digital vs analog
digital = False;

# choose full expression of the characteristic equation vs the expansion of 3d Order, False is the Full, True is the expansion
expansion=False;

# choose phase or anti-phase synchroniazed states,
sync_state1='inphase';                                                          # choose between inphase, antiphase, entrainment
sync_state2='antiphase';
sync_state3='entrainment';




df        = 0.00025 #0.00025;
fmin      = -4
fmax      =  11
f         = np.logspace(fmin, fmax, num=int((tauf+maxp)/df), endpoint=True, base=10.0)

tauvalues2plot = [0*wmean/(2*np.pi), 0.25*wmean/(2*np.pi), 0.5*wmean/(2*np.pi), 0.75*wmean/(2*np.pi), 1.0*wmean/(2*np.pi), 1.25*wmean/(2*np.pi)]
####################################################################################################################################################################################

gcf_vs_tau_singlePLL1=[];
gcf_vs_tau_singlePLL2=[];
gamma_phase_margin_One1=[];
gamma_phase_margin_One2=[];
pm_vs_tau_singlePLL1=[];
pm_vs_tau_singlePLL2=[];
GM_Single1=[];
GM_Single2=[];
tau_Single1=[];
tau_Single2=[];
Omega_Single1=[];
Omega_Single2=[];
gamma_phase_margin_Net=[];

pm_vs_tau_Net_Homog=[]; GM_NetHomog=[];
pm_vs_tau_Net_Homog_AP=[];
tauNetHomog=[]; OmegaNetHomog=[];
GM_NetHomog_AP=[];
tauNetHomog_AP=[]; OmegaNetHomog_AP=[];
pm_vs_tau_Net_Hetero=[];

GM_NetHetero=[];
tauNetHetero=[]; OmegaNetHetero=[];
GM_NetHetero_AP=[];
tauNetHetero_AP=[]; OmegaNetHetero_AP=[];
gcf_vs_tau_Net_Hetero=[]
gcf_vs_tau_Net_Hetero_AP=[]
gcf_vs_tau_Net_Homog_AP =[]
gain_cross_freq_One=[];
gcf_vs_tau_Net_Homog=[];
phase_cross_freq_One1=[];
phase_cross_freq_One2=[];
GM_One_Lin=[];
pm_vs_tau_singlePLL_Lin=[];
tauOne_Lin=[];OmegaOne_Lin=[];
phaseCrossOverHomog=[]
phaseCrossOverHomog_AP=[]
phaseCrossOverHetero=[]
phaseCrossOverHetero_AP=[]
pm_vs_tau_Net_Hetero_AP =[]

PhaseOnePLL1        = np.vectorize(PhaseopenloopMutuallyCoupledOnePLL1)
PhaseOnePLL2        = np.vectorize(PhaseopenloopMutuallyCoupledOnePLL2)
PhaseNet           = np.vectorize(PhaseopenloopMutuallyCoupledNet)
# globalFreq(wmean, Dw, Kmean, DK, tauf,  digital, maxp, sync_state)
syncStateInphase = globalFreq(wmean, Dw, Kmean, DK, tauf, digital, maxp, sync_state1);
syncStateAnphase = globalFreq(wmean, Dw, Kmean, DK, tauf, digital, maxp, sync_state2);
#Here we calculate and save to lists the gain and phase crossover frequencies and gain and phase margin as functions of the delay.
for j in range(len(syncStateInphase['taubetain'])):

	print('inphase progress:',j,'/',len(syncStateInphase['taubetain']))

	######################################################################################################## INPHASE

# ## ## ## # ## ## ## # ## ## ## # ## ## ## PLL1 # ## ## ## # ## ## ## # ## ## ## # ## ## ## # ## ## ## # ## ## ##
	gcf_vs_tau_singlePLL1.append(f[np.where(20.0*log10((abs(HopenLoopMutuallyCoupledOnePLL1(2.0*np.pi*f, syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j],
		tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga,  digital, 'Nonlinear')))) <= 0.0001)][0])

	pm_vs_tau_singlePLL1.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne1(2.0*np.pi*gcf_vs_tau_singlePLL1[-1], syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j],
		tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga,  digital, 'Nonlinear'))



	phase_cross_freq_One1.append(f[np.where((abs(180.0+PhaseOnePLL1(2.0*np.pi*f, syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j],
		tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga,  digital, 'Nonlinear'))) < 0.7 )] )

	phase1 = PhaseOnePLL1(2.0*np.pi*f, syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j], tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga,  digital, 'Nonlinear')
	if (abs(180.0+phase1) <= 0.7 ).any():
		GM1 = f[np.where(abs(180.0+phase1) <= 0.7)][0]

		GM_Single1.append(GainMarginMutuallyCoupledOne1(2.0*np.pi*GM1, syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j], tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga,  digital, 'Nonlinear'))
		tau_Single1.append( syncStateInphase['Omegabetain'][j],)
		Omega_Single1.append(syncStateInphase['taubetain'][j])




# ## ## ## # ## ## ## # ## ## ## # ## ## ## PLL2 # ## ## ## # ## ## ## # ## ## ## # ## ## ## # ## ## ## # ## ## ##

	gcf_vs_tau_singlePLL2.append(f[np.where(20.0*log10((abs(HopenLoopMutuallyCoupledOnePLL2(2.0*np.pi*f, syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j],
		tauf, tauc2, v, Kvco1, AkPD, GkLF, Gvga,  digital, 'Nonlinear')))) <= 0.0001)][0])

	pm_vs_tau_singlePLL2.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne2(2.0*np.pi*gcf_vs_tau_singlePLL2[-1], syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j],
		tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga,  digital, 'Nonlinear'))


	phase_cross_freq_One2.append(f[np.where((abs(180.0+PhaseOnePLL2(2.0*np.pi*f, syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j],
		tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga,  digital, 'Nonlinear'))) < 0.7 )] )

	phase2 = PhaseOnePLL2(2.0*np.pi*f, syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j], tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga, digital, 'Nonlinear')

	if (abs(180.0+phase2) <= 0.7 ).any():
		GM2 = f[np.where(abs(180.0+phase2) <= 0.7)][0]

		GM_Single2.append(GainMarginMutuallyCoupledOne2(2.0*np.pi*GM2, syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j], tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga,  digital, 'Nonlinear'))
		tau_Single2.append( syncStateInphase['Omegabetain'][j],)
		Omega_Single2.append(syncStateInphase['taubetain'][j])




# ## ## ## # ## ## ## # ## ## ## # ## ## ## Network # ## ## ## # ## ## ## # ## ## ## # ## ## ## # ## ## ## # ## ## ##




	gcf_vs_tau_Net_Homog.append(f[np.where(20.0*log10((abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j],
		tauf1, tauf2, tauchomogen, tauchomogen, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear')))) <= 0.0001)][-1] )

	pm_vs_tau_Net_Homog.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledNet(2.0*np.pi*gcf_vs_tau_Net_Homog[-1], syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j],
		tauf1, tauf2, tauchomogen, tauchomogen, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear'))


	phaseNetHomog = PhaseNet(2.0*np.pi*f, syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j],
							  tauf1, tauf2, tauchomogen, tauchomogen, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear')


	index1   = np.where(np.abs(phaseNetHomog) > 1E-10)[0][0]
	# print('index1: ', index1)
	# print('phaseNet[index1:]:', phaseNet[index1:])
	if np.any(phaseNetHomog[index1:]<0):												# catch the zero delay case where the phase never goes below zero
		if phaseNetHomog[index1] > 0:

			index2 = np.where(phaseNetHomog[index1:] < 0)[0][0]
			GMNetHomog  = f[index1+index2]
		elif phaseNetHomog[index1] < 0:

			index2 = np.where(phaseNetHomog[index1:] > 0)[0][0]
			index3 = np.where(phaseNetHomog[(index1+index2):] < 0)[0][0]
			GMNetHomog  = f[index1+index2+index3]
		else:
			print('Error! Debug.'); sys.exit();
	elif np.min(phaseNetHomog[index1:]) < treshold_detect_zero_delay:
		index2 = np.where(phaseNetHomog[index1:] < treshold_detect_zero_delay)[0][0]
		GMNetHomog  = f[index1+index2]
	else:
		print('Phase associated to zero delay does not go below threshold, increase frequency range of Bode plot or change treshold!');
		print('Smallest value for phase of zero delay case:', np.min(phaseNetHomog[index1:]))
		sys.exit();

	#

	# index1   = np.where(phaseNet >=120)[0][0]
	# #print('index1: ', index1)
	# #print('phaseNet[index1:]:', phaseNet[index1:])
	# index2 = np.where(phaseNet[index1:] <= 0.7)[0][0]
	# # if syncStateInphase['tau'][j] > 0:
	# # 	index2 = np.where(phaseNet[index1:] <= 0.0)[0][0]
	# # else:
	# # 	index2 = np.where(phaseNet[index1:] <= 0.0)[0]
	# # print('index2: ', index2)
	# GMNet  = f[index1+index2]
	# #print('found GMNet:', GMNet, ' at index:', index1+index2)
	# #Kvcobar			= [0.002*2.0*np.pi, 0.02*2.0*np.pi, 0.05*2.0*np.pi, 0.1*2.0*np.pi, 0.5*2.0*np.pi, 0.8*2.0*np.pi]
	if ( np.abs(syncStateInphase['taubetain'][j]-tauvalues2plot[3]) <= 0.01 and len(phaseCrossOverHomog)==0 ):
		print('Save phaseCrossOver for one value of the delay! PCF:', GMNetHomog)
		phaseCrossOverHomog.append(GMNetHomog);

	GM_NetHomog.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNetHomog, syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j],
							  tauf1, tauf2, tauchomogen, tauchomogen, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear'))


	tauNetHomog.append(syncStateInphase['taubetain'][j])
	OmegaNetHomog.append(syncStateInphase['Omegabetain'][j])


######################################################################################################## ANTIPHASE





#############################################################################################################
#Heterogeneous

	gcf_vs_tau_Net_Hetero.append(f[np.where(20.0*log10((abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j],
		tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear')))) <= 0.0001)][-1] )

	pm_vs_tau_Net_Hetero.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledNet(2.0*np.pi*gcf_vs_tau_Net_Hetero[-1], syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j],
		tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear'))




	phaseNetHetero = PhaseNet(2.0*np.pi*f, syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j],
							  tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear')
	index1Hetero   = np.where(np.abs(phaseNetHetero) > 1E-10)[0][0]
	# print('index1: ', index1)
	# print('phaseNet[index1:]:', phaseNet[index1:])
	if np.any(phaseNetHetero[index1Hetero:]<0):												# catch the zero delay case where the phase never goes below zero
		if phaseNetHetero[index1Hetero] > 0:

			index2Hetero = np.where(phaseNetHetero[index1Hetero:] < 0)[0][0]
			GMNetHetero  = f[index1Hetero+index2Hetero]
		elif phaseNetHetero[index1Hetero] < 0:

			index2Hetero = np.where(phaseNetHetero[index1Hetero:] > 0)[0][0]
			index3Hetero = np.where(phaseNetHetero[(index1Hetero+index2Hetero):] < 0)[0][0]
			GMNetHetero  = f[index1Hetero+index2Hetero+index3Hetero]
		else:
			print('Error! Debug.'); sys.exit();
	elif np.min(phaseNetHetero[index1:]) < treshold_detect_zero_delay:
		index2Hetero = np.where(phaseNetHetero[index1Hetero:] < treshold_detect_zero_delay)[0][0]
		GMNetHetero  = f[index1Hetero+index2Hetero]
	else:
		print('Phase associated to zero delay does not go below threshold, increase frequency range of Bode plot or change treshold!');
		print('Smallest value for phase of zero delay case:', np.min(phaseNetHetero[index1Hetero:]))
		sys.exit();

	#

	# index1   = np.where(phaseNet >=120)[0][0]
	# #print('index1: ', index1)
	# #print('phaseNet[index1:]:', phaseNet[index1:])
	# index2 = np.where(phaseNet[index1:] <= 0.7)[0][0]
	# # if syncStateInphase['tau'][j] > 0:
	# # 	index2 = np.where(phaseNet[index1:] <= 0.0)[0][0]
	# # else:
	# # 	index2 = np.where(phaseNet[index1:] <= 0.0)[0]
	# # print('index2: ', index2)
	# GMNet  = f[index1+index2]
	# #print('found GMNet:', GMNet, ' at index:', index1+index2)
	# #Kvcobar			= [0.002*2.0*np.pi, 0.02*2.0*np.pi, 0.05*2.0*np.pi, 0.1*2.0*np.pi, 0.5*2.0*np.pi, 0.8*2.0*np.pi]
	if ( np.abs(syncStateInphase['taubetain'][j]-tauvalues2plot[3]) <= 0.01 and len(phaseCrossOverHetero)==0 ):
		print('Save phaseCrossOver for one value of the delay! PCF:', GMNetHetero)
		phaseCrossOverHetero.append(GMNetHetero);

	GM_NetHetero.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNetHetero, syncStateInphase['Omegabetain'][j], syncStateInphase['taubetain'][j], syncStateInphase['betain'][j],
							  tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear'))


	tauNetHetero.append(syncStateInphase['taubetain'][j])
	OmegaNetHetero.append(syncStateInphase['Omegabetain'][j])







##################################################################################################################################
#ANTIPHASE

for j in range(len(syncStateAnphase['taubetaanti'])):

	print('antiphase progress:',j,'/',len(syncStateAnphase['taubetaanti']))

	gcf_vs_tau_Net_Hetero_AP.append(f[np.where(20.0*log10((abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncStateAnphase['Omegabetaanti'][j], syncStateAnphase['taubetaanti'][j], syncStateAnphase['betaanti'][j],
		tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear')))) <= 0.0001)][-1] )

	pm_vs_tau_Net_Hetero_AP.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledNet(2.0*np.pi*gcf_vs_tau_Net_Hetero_AP[-1], syncStateAnphase['Omegabetaanti'][j], syncStateAnphase['taubetaanti'][j], syncStateAnphase['betaanti'][j],
		tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear'))




	phaseNetHetero_AP = PhaseNet(2.0*np.pi*f, syncStateAnphase['Omegabetaanti'][j], syncStateAnphase['taubetaanti'][j], syncStateAnphase['betaanti'][j],
							  tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear')
	index1Hetero_AP   = np.where(np.abs(phaseNetHetero_AP) > 1E-10)[0][0]
	# print('index1: ', index1)
	# print('phaseNet[index1:]:', phaseNet[index1:])
	if np.any(phaseNetHetero_AP[index1Hetero_AP:]<0):												# catch the zero delay case where the phase never goes below zero
		if phaseNetHetero_AP[index1Hetero_AP] > 0:

			index2Hetero_AP = np.where(phaseNetHetero_AP[index1Hetero_AP:] < 0)[0][0]
			GMNetHetero_AP  = f[index1Hetero_AP+index2Hetero_AP]
		elif phaseNetHetero[index1Hetero] < 0:

			index2Hetero_AP = np.where(phaseNetHetero_AP[index1Hetero_AP:] > 0)[0][0]
			index3Hetero_AP = np.where(phaseNetHetero_AP[(index1Hetero_AP+index2Hetero_AP):] < 0)[0][0]
			GMNetHetero_AP  = f[index1Hetero_AP+index2Hetero_AP+index3Hetero_AP]
		else:
			print('Error! Debug.'); sys.exit();
	elif np.min(phaseNetHetero_AP[index1_AP:]) < treshold_detect_zero_delay:
		index2Hetero_AP = np.where(phaseNetHetero_AP[index1Hetero_AP:] < treshold_detect_zero_delay)[0][0]
		GMNetHetero_AP  = f[index1Hetero_AP+index2Hetero_AP]
	else:
		print('Phase associated to zero delay does not go below threshold, increase frequency range of Bode plot or change treshold!');
		print('Smallest value for phase of zero delay case:', np.min(phaseNetHetero_AP[index1Hetero_AP:]))
		sys.exit();


	if ( np.abs(syncStateAnphase['taubetaanti'][j]-tauvalues2plot[3]) <= 0.01 and len(phaseCrossOverHetero_AP)==0 ):
		print('Save phaseCrossOver for one value of the delay! PCF:', GMNetHetero_AP)
		phaseCrossOverHetero_AP.append(GMNetHetero_AP);

	GM_NetHetero_AP.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNetHetero_AP, syncStateAnphase['Omegabetaanti'][j], syncStateAnphase['taubetaanti'][j], syncStateAnphase['betaanti'][j],
							  tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear'))


	tauNetHetero_AP.append(syncStateAnphase['taubetaanti'][j])
	OmegaNetHetero_AP.append(syncStateAnphase['Omegabetaanti'][j])




	gcf_vs_tau_Net_Homog_AP.append(f[np.where(20.0*log10((abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncStateAnphase['Omegabetaanti'][j], syncStateAnphase['taubetaanti'][j], syncStateAnphase['betaanti'][j],
		tauf1, tauf2, tauchomogen,tauchomogen, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear')))) <= 0.0001)][-1] )

	pm_vs_tau_Net_Homog_AP.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledNet(2.0*np.pi*gcf_vs_tau_Net_Homog_AP[-1], syncStateAnphase['Omegabetaanti'][j], syncStateAnphase['taubetaanti'][j], syncStateAnphase['betaanti'][j],
		tauf1, tauf2, tauchomogen, tauchomogen, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear'))


	phaseNetHomog_AP = PhaseNet(2.0*np.pi*f, syncStateAnphase['Omegabetaanti'][j], syncStateAnphase['taubetaanti'][j], syncStateAnphase['betaanti'][j],
							  tauf1, tauf2, tauchomogen, tauchomogen, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear')


	index1_AP   = np.where(np.abs(phaseNetHomog_AP) > 1E-10)[0][0]
	# print('index1: ', index1)
	# print('phaseNet[index1:]:', phaseNet[index1:])
	if np.any(phaseNetHomog_AP[index1_AP:]<0):												# catch the zero delay case where the phase never goes below zero
		if phaseNetHomog_AP[index1_AP] > 0:

			index2_AP = np.where(phaseNetHomog_AP[index1_AP:] < 0)[0][0]
			GMNetHomog_AP  = f[index1_AP+index2_AP]
		elif phaseNetHomog_AP[index1_AP] < 0:

			index2_AP = np.where(phaseNetHomog_AP[index1_AP:] > 0)[0][0]
			index3_AP = np.where(phaseNetHomog_AP[(index1_AP+index2_AP):] < 0)[0][0]
			GMNetHomog_AP  = f[index1_AP+index2_AP+index3_AP]
		else:
			print('Error! Debug.'); sys.exit();
	elif np.min(phaseNetHomog_AP[index1_AP:]) < treshold_detect_zero_delay:
		index2_AP = np.where(phaseNetHomog_AP[index1:] < treshold_detect_zero_delay)[0][0]
		GMNetHomog_AP  = f[index1_AP+index2_AP]
	else:
		print('Phase associated to zero delay does not go below threshold, increase frequency range of Bode plot or change treshold!');
		print('Smallest value for phase of zero delay case:', np.min(phaseNetHomog_AP[index1_AP:]))
		sys.exit();


	if ( np.abs(syncStateAnphase['taubetaanti'][j]-tauvalues2plot[3]) <= 0.01 and len(phaseCrossOverHomog_AP)==0 ):
		print('Save phaseCrossOver for one value of the delay! PCF:', GMNetHomog_AP)
		phaseCrossOverHomog_AP.append(GMNetHomog_AP);

	GM_NetHomog_AP.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNetHomog_AP, syncStateAnphase['Omegabetaanti'][j], syncStateAnphase['taubetaanti'][j], syncStateAnphase['betaanti'][j],
							  tauf1, tauf2, tauchomogen, tauchomogen, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear'))


	tauNetHomog_AP.append(syncStateAnphase['taubetaanti'][j])
	OmegaNetHomog_AP.append(syncStateAnphase['Omegabetaanti'][j])




	#
	# phase_cross_freq_One.append(f[np.where((abs(180.0+PhaseOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
	# 	tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear'))) < 0.7 )] )
	#
	# phase = PhaseOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear')
	# if (abs(180.0+phase) <= 0.7 ).any():
	# 	GM = f[np.where(abs(180.0+phase) <= 0.7)][0]
	#
	# 	GM_One.append(GainMarginMutuallyCoupledOne(2.0*np.pi*GM, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear'))
	# 	tauOne.append(syncStateInphase['tau'][j])
	# 	OmegaOne.append(syncStateInphase['Omeg'][j])
	#
	# phaseNet = PhaseNet(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear')
	# index1   = np.where(phaseNet >=120)[0][0]
	# #print('index1: ', index1)
	# #print('phaseNet[index1:]:', phaseNet[index1:])
	# index2 = np.where(phaseNet[index1:] <= 0.7)[0][0]
	# # if syncStateInphase['tau'][j] > 0:
	# # 	index2 = np.where(phaseNet[index1:] <= 0.0)[0][0]
	# # else:
	# # 	index2 = np.where(phaseNet[index1:] <= 0.0)[0]
	# # print('index2: ', index2)
	# GMNet  = f[index1+index2]
	# #print('found GMNet:', GMNet, ' at index:', index1+index2)
	# #Kvcobar			= [0.002*2.0*np.pi, 0.02*2.0*np.pi, 0.05*2.0*np.pi, 0.1*2.0*np.pi, 0.5*2.0*np.pi, 0.8*2.0*np.pi]
	# if ( np.abs(syncStateInphase['tau'][j]-tauvalues2plot[3]) <= 0.01 and len(phaseCrossOver)==0 ):
	# 	print('Save phaseCrossOver for one value of the delay! PCF:', GMNet)
	# 	phaseCrossOver.append(GMNet);
	#
	# GM_Net.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNet, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j], tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Nonlinear'))
	# tauNet.append(syncStateInphase['tau'][j])
	# OmegaNet.append(syncStateInphase['Omeg'][j])


	#
	# ADD THE CURVES TO THE PLOTS
	#
	# ######################################################################################################## LINEAR
	#
	# PMLin = f[np.where(20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
	# 	tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear'))) <= 0.0001)][0]
	#
	# pm_vs_tau_singlePLL_Lin.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*PMLin, syncStateInphase['Omeg'][j],
	# 	syncStateInphase['tau'][j],
	# 	tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear'))
	#
	#
	# if (abs(180.0+PhaseOnePLL(2.0*np.pi*f, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
	# 			tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear')) <=0.7).any():
	#
	# 	GMLin=f[np.where(abs(180.0+PhaseOnePLL(2.0*np.pi*f,	syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
	# 			tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear')) <=0.7)][0]
	# 	# print(GM)
	# 	GM_One_Lin.append(GainMarginMutuallyCoupledOne(2.0*np.pi*GMLin, syncStateInphase['Omeg'][j], syncStateInphase['tau'][j],
	# 		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, 'Linear'))
	#
	# 	tauOne_Lin.append(syncStateInphase['tau'][j])
	# 	OmegaOne_Lin.append(syncStateInphase['Omeg'][j])

bar = [];
for i in range(len(tauvalues2plot)):
	tempo = np.where(syncStateInphase['taubetain'][:]>tauvalues2plot[i])[0][0]
	# print('tempo:', tempo)
	bar.append( tempo )

# ADD PLOTS
####################################################################################################################################################################################
colorbar=['blue','green','orange','purple', 'cyan', 'black']
# print(globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1]/(2*np.pi))
# print(globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1])
##################################################################################################################################################################
fig0 = plt.figure(num=0, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig0.set_size_inches(20,10)

ax01 = plt.subplot(211)
plt.title(r'PLL1 $\omega_1=$ %2.2f , K1=%2.2f , $f^c_1=$ %2.2f '%( w1/(2.0*np.pi), K1/(2.0*np.pi), 1/(2.0*np.pi*tauc1) ) )
ax01.set_xlim(f[0], f[-1])
plt.axhline(y=0, color='r', linestyle='-.')
plt.xscale('log');
for i in range(len(bar)):
	[lineol01] = ax01.plot(f, 20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL1(2.0*np.pi*f, syncStateInphase['Omegabetain'][bar[i]], syncStateInphase['taubetain'][bar[i]], syncStateInphase['betain'][bar[i]],
								tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga,  digital, 'Nonlinear'))), '-', ms=1.5, color= colorbar[i], label=r'$\tau=$%.5f' %(syncStateInphase['taubetain'][bar[i]]))

plt.grid();
plt.ylabel(r'$20\textrm{log}\left|H_k^\textrm{\LARGE OL}(j\gamma)\right|$', rotation=90, fontsize=36, labelpad=30)
ax01.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont);
ax01.tick_params(axis='both', which='major', labelsize=25, pad=1)

ax02 = plt.subplot(212)
ax02.set_xlim(f[0], f[-1])
plt.xscale('log')
plt.axhline(y=-180, color='r', linestyle='-.')
for i in range(len(bar)):
	[lineol02] = ax02.plot(f, PhaseOnePLL1(2.0*np.pi*f, syncStateInphase['Omegabetain'][bar[i]], syncStateInphase['taubetain'][bar[i]], syncStateInphase['betain'][bar[i]],
								tauf, tauc1, v, Kvco1, AkPD, GkLF, Gvga,  digital, 'Nonlinear'), '.', ms=1.5, color= colorbar[i], label=r'$\tau=$%.5f' %(syncStateInphase['taubetain'][bar[i]]))
plt.grid();
plt.ylabel(r'phase', rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=40,labelpad=-5)
ax02.tick_params(axis='both', which='major', labelsize=25, pad=1)
# ax2.legend(fontsize=15)
if digital == True:
	plt.savefig('plots/digital_bode_plot_One_PLL1_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_One_PLL1_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_One_PLL1_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_bode_plot_One_PLL1_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_One_PLL1_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_One_PLL1_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_bode_plot_One_PLL1_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_One_PLL1_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_One_PLL1_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)






fig1 = plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig1.set_size_inches(20,10)

ax1 = plt.subplot(211)
plt.title(r'PLL2 $\omega_2=$ %2.2f , K2=%2.2f , $f^c_2=$ %2.2f '%( w2/(2.0*np.pi), K2/(2.0*np.pi), 1/(2.0*np.pi*tauc2) ) )
ax1.set_xlim(f[0], f[-1])
plt.axhline(y=0, color='r', linestyle='-.')
plt.xscale('log');
for i in range(len(bar)):
	[lineol] = ax1.plot(f, 20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL2(2.0*np.pi*f, syncStateInphase['Omegabetain'][bar[i]], syncStateInphase['taubetain'][bar[i]], syncStateInphase['betain'][bar[i]],
								tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga,  digital, 'Nonlinear'))), '-', ms=1.5, color= colorbar[i], label=r'$\tau=$%.5f' %(syncStateInphase['taubetain'][bar[i]]))

plt.grid();
plt.ylabel(r'$20\textrm{log}\left|H_k^\textrm{\LARGE OL}(j\gamma)\right|$', rotation=90, fontsize=36, labelpad=30)
ax1.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont);
ax1.tick_params(axis='both', which='major', labelsize=25, pad=1)

ax2 = plt.subplot(212)
ax2.set_xlim(f[0], f[-1])
plt.xscale('log')
plt.axhline(y=-180, color='r', linestyle='-.')
for i in range(len(bar)):
	[lineol2] = ax2.plot(f, PhaseOnePLL2(2.0*np.pi*f, syncStateInphase['Omegabetain'][bar[i]], syncStateInphase['taubetain'][bar[i]], syncStateInphase['betain'][bar[i]],
								tauf, tauc2, v, Kvco2, AkPD, GkLF, Gvga,  digital, 'Nonlinear'), '.', ms=1.5, color= colorbar[i], label=r'$\tau=$%.5f' %(syncStateInphase['taubetain'][bar[i]]))
plt.grid();
plt.ylabel(r'phase', rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=40,labelpad=-5)
ax2.tick_params(axis='both', which='major', labelsize=25, pad=1)
# ax2.legend(fontsize=15)
if digital == True:
	plt.savefig('plots/digital_bode_plot_One_PLL2_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_One_PLL2_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_One_PLL2_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_bode_plot_One_PLL2_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_One_PLL2_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_One_PLL2_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_bode_plot_One_PLL2_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_One_PLL2_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_One_PLL2_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

##################################################################################################################################################################

fig2 = plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig2.set_size_inches(20,10)

ax3 = plt.subplot(211)
ax3.set_xlim(f[0], f[-1])
plt.xscale('log');
plt.axhline(y=0, color='r', linestyle='-.')
if phaseCrossOverHetero:
	print('Plot PCF for first tau value! PCF:', phaseCrossOverHetero)
	plt.axvline(x=phaseCrossOverHetero, color='y', linestyle='-', linewidth=0.5, alpha=0.5)
for i in range(len(bar)):
	[lineol00] = ax3.plot(f, 20.0*log10(abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f, syncStateInphase['Omegabetain'][bar[i]], syncStateInphase['taubetain'][bar[i]], syncStateInphase['betain'][bar[i]],
								tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear'))), '-', ms=1.5, color= colorbar[i], label=r'$\tau=$%.5f' %(syncStateInphase['taubetain'][bar[i]]))


plt.grid();
plt.ylabel(r'$20\textrm{log} \left|H_\textrm{net}^\textrm{\LARGE OL}(j\gamma)\right|$', rotation=90, fontsize=36, labelpad=30)
ax3.tick_params(axis='both', which='major', labelsize=25, pad=1)
ax3.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont);

ax4 = plt.subplot(212)
plt.xscale('log');
ax4.set_xlim(f[0], f[-1])
plt.axhline(y=0, color='r', linestyle='-.')
if phaseCrossOverHetero:
	print('Plot PCF for first tau value!')
	plt.axvline(x=phaseCrossOverHetero, color='y', linestyle='-', linewidth=0.5, alpha=0.5)
for i in range(len(bar)):
	[lineol00] = ax4.plot(f, PhaseNet(2.0*np.pi*f, syncStateInphase['Omegabetain'][bar[i]], syncStateInphase['taubetain'][bar[i]], syncStateInphase['betain'][bar[i]],
								tauf1, tauf2, tauc1, tauc2, v, Kvco1, Kvco2,  AkPD, GkLF, Gvga,  digital, 'Nonlinear'), '.', ms=1.5, color= colorbar[i], label=r'$\tau=$%.5f' %(syncStateInphase['taubetain'][bar[i]]))
plt.grid();
plt.ylabel(r'phase', rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=40,labelpad=-5)
ax4.tick_params(axis='both', which='major', labelsize=25, pad=1)
# ax4.legend(fontsize=15)
if digital == True:
	plt.savefig('plots/digital_bode_plot_Net_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_Net_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_Net_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_bode_plot_Net_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_Net_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_Net_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_bode_plot_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

##################################################################################################################################################################

fig3 = plt.figure(num=3, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig3.set_size_inches(20,10)

ax5 = plt.subplot(211)
plt.title(r'PLL1 $\omega_1=$ %2.2f , K1=%2.2f , $f^c_1=$ %2.2f '%( w1/(2.0*np.pi), K1/(2.0*np.pi), 1/(2.0*np.pi*tauc1) ) )
OmegTau = syncStateInphase['taubetain']*syncStateInphase['Omegabetain']/(2.0*np.pi)
ax5.set_xlim(OmegTau[0], OmegTau[-1])
[line4]  = ax5.plot(OmegTau, np.array(pm_vs_tau_singlePLL1), '.', ms=1.5, color='blue', label=r'nonlinear');
# [line44] = ax5.plot(OmegTau, np.array(pm_vs_tau_singlePLL_Lin), '.', ms=1.5, color='black', label=r'linear');
plt.ylabel(r'phase margin', rotation=90, fontsize=36, labelpad=30)
ax5.tick_params(axis='both', which='major', labelsize=25, pad=1)
plt.grid();
ax5.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)

ax6 = plt.subplot(212)
ax6.set_xlim(OmegTau[0], OmegTau[-1])
[lineol06] = ax6.plot(np.array(tau_Single1)*np.array(Omega_Single1)/(2.0*np.pi), np.array(GM_Single1), '.', ms=1.5, color='blue', label=r'nonlinear');
# [lineol66] = ax6.plot(np.array(tauOne_Lin)*np.array(OmegaOne_Lin)/(2.0*np.pi), np.array(GM_One_Lin), '-', color='black', label=r'nonlinear');
plt.ylabel(r'gain margin', rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=40, labelpad=5)
ax6.tick_params(axis='both', which='major', labelsize=25, pad=1)
# ax6.legend()
plt.grid();
if digital == True:
	plt.savefig('plots/digital_Margins_One_PLL1_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_One_PLL1_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_One_PLL1_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_One_PLL1_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_One_PLL1_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_One_PLL1_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_One_PLL1_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_One_PLL1_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_One_PLL1_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

#############################################################################################################################################


fig03 = plt.figure(num=13, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig03.set_size_inches(20,10)

ax05 = plt.subplot(211)
plt.title(r'PLL2 $\omega_2=$ %2.2f , K2=%2.2f , $f^c_2=$ %2.2f '%( w2/(2.0*np.pi), K2/(2.0*np.pi), 1/(2.0*np.pi*tauc2) ) )
ax05.set_xlim(OmegTau[0], OmegTau[-1])
[line04]  = ax05.plot(OmegTau, np.array(pm_vs_tau_singlePLL2), '.', ms=1.5, color='blue', label=r'nonlinear');
# [line44] = ax5.plot(OmegTau, np.array(pm_vs_tau_singlePLL_Lin), '.', ms=1.5, color='black', label=r'linear');
plt.ylabel(r'phase margin', rotation=90, fontsize=36, labelpad=30)
ax05.tick_params(axis='both', which='major', labelsize=25, pad=1)
plt.grid();
ax05.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)

ax06 = plt.subplot(212)
ax06.set_xlim(OmegTau[0], OmegTau[-1])
[lineol06] = ax06.plot(np.array(tau_Single2)*np.array(Omega_Single2)/(2.0*np.pi), np.array(GM_Single2), '.', ms=1.5, color='blue', label=r'nonlinear');
# [lineol66] = ax6.plot(np.array(tauOne_Lin)*np.array(OmegaOne_Lin)/(2.0*np.pi), np.array(GM_One_Lin), '-', color='black', label=r'nonlinear');
plt.ylabel(r'gain margin', rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$\Omega\tau/2\pi$', fontsize=40, labelpad=5)
ax06.tick_params(axis='both', which='major', labelsize=25, pad=1)
# ax6.legend()
plt.grid();
if digital == True:
	plt.savefig('plots/digital_Margins_One_PLL2_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_One_PLL2_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_One_PLL2_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_One_PLL2_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_One_PLL2_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_One_PLL2_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_One_PLL2_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_One_PLL2_%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_One_PLL2_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)












##################################################################################################################################################################

fig4 = plt.figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig4.set_size_inches(20,10)

ax7 = plt.subplot(211)
ax7.set_xlim(OmegTau[0], OmegTau[-1])
# [lineNet7]    = ax7.plot(syncStateInphase['Omegabetain']*syncStateInphase['taubetain']/(2.0*np.pi), np.array(pm_vs_tau_Net), '.', ms=1.5, label=r'phase margin Net', color='blue');
#[lineSigmaIn] = ax7.plot(np.asarray(syncStateInphase['Omeg']/(2.0*np.pi))*syncStateInphase['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, syncStateInphase['Omeg'], syncStateInphase['tau'],
#								tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state1, expansion, INV)['ReMax'], '-', linewidth=1, color='green', label=r'in-phase')
[lineSigmaIn] = ax7.plot(np.asarray(syncStateInphase['Omegabetain'])*np.asarray(syncStateInphase['taubetain'])/(2.0*np.pi), np.asarray(2.0*np.pi/wmean)*solveLinStabbetain(syncStateInphase['Omegabetain'],
							syncStateInphase['taubetain'], tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauchomogen, tauchomogen, Dw, syncStateInphase['betain'], digital, maxp)['RebetainMax'],
					'o' ,mfc='none', ms=4, label=r'in-phase $\Delta \omega_c=0$', color='blue');

[lineSigmaInHetero] = ax7.plot(np.asarray(syncStateInphase['Omegabetain'])*np.asarray(syncStateInphase['taubetain'])/(2.0*np.pi), np.asarray(2.0*np.pi/wmean)*solveLinStabbetain(syncStateInphase['Omegabetain'],
							syncStateInphase['taubetain'], tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, syncStateInphase['betain'], digital, maxp)['RebetainMax'],
					 'o' ,mfc='none', ms=4, label=r'in-phase $\Delta \omega_c=%0.2f\cdot2\pi \textrm{Hz}$' %(Dwc/(2.0*np.pi)), color='orange');


[lineSigmaAn] = ax7.plot(np.asarray(syncStateAnphase['Omegabetaanti'])*np.asarray(syncStateAnphase['taubetaanti'])/(2.0*np.pi), np.asarray(2.0*np.pi/wmean)*solveLinStabbetaanti(syncStateAnphase['Omegabetaanti'],
							syncStateAnphase['taubetaanti'], tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauchomogen, tauchomogen, Dw, syncStateAnphase['betaanti'], digital, maxp)['RebetaantiMax'],
					'o' ,mfc='none', ms=4, label=r'anti-phase $\Delta \omega_c=0 $', color='green');

[lineSigmaAnHetero] = ax7.plot(np.asarray(syncStateAnphase['Omegabetaanti'])*np.asarray(syncStateAnphase['taubetaanti'])/(2.0*np.pi), np.asarray(2.0*np.pi/wmean)*solveLinStabbetaanti(syncStateAnphase['Omegabetaanti'],
							syncStateAnphase['taubetaanti'], tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, syncStateAnphase['betaanti'], digital, maxp)['RebetaantiMax'],
					'o' ,mfc='none', ms=4, label=r'anti-phase $\Delta \omega_c=%0.2f\cdot2\pi \textrm{Hz}$' %(Dwc/(2.0*np.pi)), color='red');




plt.grid();
plt.ylabel(r' $\frac{2\pi\sigma}{\omega}$', rotation=90, fontsize=36, labelpad=30)
ax7.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
ax7.tick_params(axis='both', which='major', labelsize=35, pad=1)

ax8 = plt.subplot(212)
ax8.set_xlim(OmegTau[0], OmegTau[-1])
ax8.tick_params(axis='both', which='major', labelsize=25, pad=1)
plt.ylabel(r'gain margin', rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$\Omega\tau/2 \pi$', fontsize=60, labelpad=5)
[lineoNetHomog] = ax8.plot(np.array(tauNetHomog)*np.array(OmegaNetHomog)/(2.0*np.pi), np.array(GM_NetHomog), 'o' ,mfc='none', ms=4, label=r'in-phase $\Delta \omega_c=0$', color='blue');
[lineoNetHetero] = ax8.plot(np.array(tauNetHetero)*np.array(OmegaNetHetero)/(2.0*np.pi), np.array(GM_NetHetero), 'o' ,mfc='none', ms=4, label=r'in-phase $\Delta \omega_c=%0.2f\cdot2\pi \textrm{Hz}Hz$' %(Dwc/(2.0*np.pi)), color='orange');
[lineoNetHomog_AP] = ax8.plot(np.array(tauNetHomog)*np.array(OmegaNetHomog)/(2.0*np.pi), np.array(GM_NetHomog), 'o' ,mfc='none', ms=4, label=r'anti-phase $\Delta \omega_c=0 $', color='green');
[lineoNetHetero_AP] = ax8.plot(np.array(tauNetHetero_AP)*np.array(OmegaNetHetero_AP)/(2.0*np.pi), np.array(GM_NetHetero_AP), 'o' ,mfc='none', ms=4, label=r'anti-phase $\Delta \omega_c=%0.2f\cdot2\pi \textrm{Hz}$' %(Dwc/(2.0*np.pi)), color='red');

# solveLinStabbetain(syncStateInphase['Omegabetain'], syncStateInphase['taubetain'], tauf, K1, K2, tauc1, tauc2, Dw, syncStateInphase['betain'], digital, maxp, expansion)['RebetainMax']
plt.grid();
if digital == True:
	plt.savefig('plots/digital_Margins_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

##################################################################################################################################################################
#
fig5 = plt.figure(num=5, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig5.set_size_inches(20,10)

ax9 = plt.subplot(211)
ax9.set_xlim(0.0, 1.6)
# [lineNet7]    = ax9.plot(syncStateInphase['Omeg']*syncStateInphase['tau']/(2.0*np.pi), np.array(pm_vs_tau_Net), '.', ms=1.5, label=r'phase margin Net', color='blue');
#[lineSigmaIn] = ax9.plot(np.asarray(syncStateInphase['Omeg']/(2.0*np.pi))*syncStateInphase['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, syncStateInphase['Omeg'], syncStateInphase['tau'],
#								tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state1, expansion, INV)['ReMax'], '-', linewidth=1, color='green', label=r'in-phase')
ax9.legend(bbox_to_anchor=(0.7415,0.75), prop=labelfont)
ax9.tick_params(axis='both', which='major', labelsize=25, pad=1)
plt.grid();
# plt.ylabel(r'phase margin', rotation=90, fontsize=36, labelpad=30)

ax10 = plt.subplot(212)
ax10.set_xlim(0.0, 1.6)
ax10.tick_params(axis='both', which='major', labelsize=25, pad=1)
# [lineoNet] = ax10.plot(np.array(tauNet)*np.array(OmegaNet)/(2.0*np.pi), np.array(GM_Net), '.', ms=1.5, label=r'gain margin Net', color='blue');
plt.ylabel(r'$\gamma$', rotation=90, fontsize=36, labelpad=30 )
plt.xlabel(r'$\Omega\tau/2 \pi$', fontsize=36, labelpad=5)
# [lineSigmaIn] = ax8.plot(np.asarray(syncStateInphase['Omeg']/(2.0*np.pi))*syncStateInphase['tau'], np.asarray(2.0*np.pi/w)*solveLinStab(wref, w, syncStateInphase['Omeg'],
# 					syncStateInphase['tau'], tauf, Kvco, AkPD, GkLF, Gvga, tauc, v, digital, maxp, sync_state1, expansion, INV)['ReMax'],
# 	 				'-', linewidth=1, color='green', label=r'in-phase')

[lineGammaIn] = ax10.plot(np.asarray(syncStateInphase['Omegabetain'])*np.asarray(syncStateInphase['taubetain'])/(2.0*np.pi),
						np.asarray(1.0/wmean)*solveLinStabbetain(syncStateInphase['Omegabetain'],
								syncStateInphase['taubetain'], tauf, Kvco1, Kvco2, AkPD, GkLF, Gvga, tauc1, tauc2, Dw, syncStateInphase['betain'], digital, maxp)['ImbetainMax'],
								'.', ms=2, color='green', label=r'$\gamma$ in-phase') #ms=1.5
plt.grid();
if digital == True:
	plt.savefig('plots/digital_gamma_freqMaxG_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_gamma_freqMaxG_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_gamma_freqMaxG_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_gamma_freqMaxG_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_gamma_freqMaxG_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_gamma_freqMaxG_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_gamma_freqMaxG_Net%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_gamma_freqMaxG_Net%d_%d_%d.svg' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_gamma_freqMaxG_Net%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

plt.show();
