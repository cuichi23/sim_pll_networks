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

PhaseOnePLL        = np.vectorize(PhaseclosedloopMutuallyCoupledOnePLL)
PhaseNet           = np.vectorize(PhaseopenloopMutuallyCoupledNet)
# PhaseclosedloopMutuallyCoupledNet





print('Hallo 1')
#


tauvalues2plot = [0*w/(2*np.pi), 0.25*w/(2*np.pi), 0.5*w/(2*np.pi), 0.75*w/(2*np.pi), 1.0*w/(2*np.pi), 1.25*w/(2*np.pi)]
bar = [];
for i in range(len(tauvalues2plot)):
	tempo = np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][:]>tauvalues2plot[i])[0][0]
	# print('tempo:', tempo)
	bar.append( tempo )
	# print('bar[i]', bar[i])





#Here we calculate and save to lists the gain and phase crossover frequencies and gain and phase margin as well as functions of the cutoff frequency
##########

wcvalues=np.arange(0.0001*2.0*np.pi, 0.8*2.0*np.pi, 0.01)
PM_One_Difwc=[];GM_One_Difwc=[]; wcvalue=[];
tempo1 = np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][:]>tauvalues2plot[1])[0][0]

for j in range(len(wcvalues)):

	# print(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j])
	# for i in range(len(f)):

	PMDifwc=f[np.where(20.0*log10(abs(HopenloopMutuallyCoupledOnePLL(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
		tauf, tauc, v, Kvco, AkPD, GkLF, Gvga,  digital, model))) <=0.0001)][0]

	PM_One_Difwc.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*PMDifwc,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
		tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model))

	if (abs(180.0+PhaseOnePLL(2.0*np.pi*f,
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
		globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
		tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model)) <=0.7).any():

		GMDifwc=f[np.where(abs(180.0+PhaseOnePLL(2.0*np.pi*f,
			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
			tauf, tauc, v, 1.0/wcvalues[j], AkPD, GkLF, Gvga,  digital, model)) <=0.7)][0]
		# print(GM)
		GM_One_Difwc.append(GainMarginMutuallyCoupledOne(2.0*np.pi*GMDifwc,
			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][tempo1],
			globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][tempo1],
			tauf, 1.0/wcvalues[j], v, Kvco, AkPD, GkLF, Gvga,  digital, model))
		wcvalue.append(wcvalues[j]/(2.0*np.pi))






# ADD PLOTS
####################################################################################################################################################################################


# tau1=    0.001

#bar= [1, 8, 30, 60]

colorbar=['blue','green','orange','purple', 'cyan', 'black']
# print(globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1]/(2*np.pi))
# print(globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1])
##################################################################################################################################################################
# #
#
#
fig1= plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig2, (, ) = plt.subplots(2)

# print(180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, v, tau_c, K))
ax1= plt.subplot(211)
					# PhaseclosedloopMutuallyCoupledOnePLL

[line1]     =    ax1.plot(wcvalues/(2.0*np.pi), np.array(PM_One_Difwc),'.',color='red');

plt.ylabel(r'phase margin', rotation=90,fontsize=45, labelpad=30)
ax11.tick_params(axis='both', which='major', labelsize=25, pad=1)
plt.grid();
ax11.legend()
ax2= plt.subplot(212)
#plt.xscale('log');

# [lineoNet]     =    ax4.plot(np.array(tauNet), np.array(GM_Net),'-', label=r'gain margin Net',color='blue');

[line66]     =    ax2.plot(np.array(wcvalue)/(2.0*np.pi), np.array(GM_One_Difwc),'.',color='blue');

plt.ylabel(r'gain margin', rotation=90,fontsize=45, labelpad=30)
plt.xlabel(r'$\omega_c$', fontsize=40,labelpad=-5)
ax2.tick_params(axis='both', which='major', labelsize=25, pad=1)
ax2.legend()
plt.grid();
if digital == True:
	plt.savefig('Plots/digital_Margins_OneVswc%d_%d_%d.pdf' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('Plots/digital_Margins_OneVswc%d_%d_%d.png' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
else:
	if coupfununction(coupfun)=='sin':
		plt.savefig('Plots/analog_sin_Margins_OneVswc%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('Plots/analog_sin_Margins_OneVswc%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfununction(coupfun)=='cos':
		plt.savefig('Plots/analog_cos_Margins_OneVswc%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('Plots/analog_cos_Margins_OneVswc%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#
# #
#
plt.show();
