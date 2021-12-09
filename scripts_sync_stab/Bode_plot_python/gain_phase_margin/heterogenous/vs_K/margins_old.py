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
from bode_lib import solveLinStab, globalFreq, linStabEq, analytical,HopenLoopMutuallyCoupledOnePLL,K,LoopGainSteadyState, coupfunction, K
from bode_lib import PhaseopenloopMutuallyCoupledOnePLL,PhaseclosedloopMutuallyCoupledOnePLL,HclosedLoopMutuallyCoupledOnePLL, GainMarginMutuallyCoupledOne, PhaseMarginMutuallyCoupledOne
from bode_lib import HopenLoopMutuallyCoupledNet, PhaseopenloopMutuallyCoupledNet,HclosedloopMutuallyCoupledNet, PhaseclosedloopMutuallyCoupledNet, HopenLoopMutuallyCoupledNet2
from bode_lib import GainMarginMutuallyCoupledNet,PhaseMarginMutuallyCoupledNet
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

wref=w
w_cutoff      = 1.0/tauc

# choose digital vs analog
digital = False;

# choose full expression of the characteristic equation vs the expansion of 3d Order, False is the Full, True is the expansion
expansion=False;

# choose phase or anti-phase synchroniazed states,
sync_state1='inphase';                                                          # choose between inphase, antiphase, entrainment
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
df 		   = 0.01;
f  		   = np.logspace(-8, 6, num=int((tauf+maxp)/df), endpoint=True, base=10.0)
dK 		   = 0.5;
Kvcovalues = np.arange(0.0001*2.0*np.pi, 0.8*2.0*np.pi, dK)

tauvalues2plot = [0*w/(2*np.pi), 0.25*w/(2*np.pi), 0.5*w/(2*np.pi), 0.75*w/(2*np.pi), 1.0*w/(2*np.pi), 1.25*w/(2*np.pi)]
bar = [];
for i in range(len(tauvalues2plot)):
	tempo = np.where(globalFreq(wref, w, Kvco, AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][:]>tauvalues2plot[i])[0][0]
	# print('tempo:', tempo)
	bar.append( tempo )

####################################################################################################################################################################################

PhaseOnePLL        = np.vectorize(PhaseopenloopMutuallyCoupledOnePLL)
PhaseNet           = np.vectorize(PhaseopenloopMutuallyCoupledNet)

# syncState = globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV);

##Here we calculate and save to lists the gain and phase crossover frequencies and gain and phase margin as well as functions of the coupling strength
###########

#
pm_vs_wc_singlePLL=[];gm_vs_wc_singlePLL=[]; Kvalue=[];

KNetvalue=[]; gm_vs_wc_Net=[]; gcf_vs_wc_Net=[]; pm_vs_wc_Net=[]; GM_Net=[];

#synchState=[]
#for j in range(len(Kvcovalues)):
#    syncState.append(globalFreq(wref, w, Kvcovalues[j], AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV));


for j in range(len(Kvcovalues)):

	print('Progress:',j,'/',len(Kvcovalues))

	syncState = globalFreq(wref, w, Kvcovalues[j], AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV);

	tempo1 = np.where(syncState['tau'][:]>tauvalues2plot[1])[0][0]


	gfc_vs_K_singlePLL=f[np.where(20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f,
		syncState['Omeg'][tempo1], syncState['tau'][tempo1], tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, model))) <=0.0001)][0]

	pm_vs_wc_singlePLL.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledOne(2.0*np.pi*gfc_vs_K_singlePLL,
		syncState['Omeg'][tempo1], syncState['tau'][tempo1],
		tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, model))

	gcf_vs_wc_Net.append(f[np.where(20.0*log10(abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f,
		syncState['Omeg'][tempo1], syncState['tau'][tempo1], tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, model))) <=0.0001)][-1] )

	pm_vs_wc_Net.append((360.0/(2.0*np.pi))*PhaseMarginMutuallyCoupledNet(2.0*np.pi*gcf_vs_wc_Net[-1],
	syncState['Omeg'][tempo1], syncState['tau'][tempo1],
	tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, model))


	if (abs(180.0+PhaseOnePLL(2.0*np.pi*f,
		syncState['Omeg'][tempo1],
		syncState['tau'][tempo1],
		tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, model)) <=0.7).any():

		pfc_vs_K_singlePLL=f[np.where(abs(180.0+PhaseOnePLL(2.0*np.pi*f,
			syncState['Omeg'][tempo1],
			syncState['tau'][tempo1],
			tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, model)) <=0.7)][0]
		# print(GM)
		gm_vs_wc_singlePLL.append(GainMarginMutuallyCoupledOne(2.0*np.pi*pfc_vs_K_singlePLL,
			syncState['Omeg'][tempo1],
			syncState['tau'][tempo1],
			tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, model))
		Kvalue.append(K(Kvcovalues[j], AkPD, GkLF, Gvga))



	phaseNet = PhaseNet(2.0*np.pi*f,
		syncState['Omeg'][tempo1],
		syncState['tau'][tempo1],
		tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, model)
	if (abs(phaseNet) <=0.01).any():
		index = np.where(abs(phaseNet) >=0.001)[0]
		print(index)
		GMNet = f[np.where(abs(phaseNet) <=0.001)][-1]
		# print(GM)
		GM_Net.append(GainMarginMutuallyCoupledNet(2.0*np.pi*GMNet,
			syncState['Omeg'][tempo1], syncState['tau'][tempo1], tauf, tauc, v, Kvcovalues[j], AkPD, GkLF, Gvga,  digital, model) )
		KNetvalue.append(K(Kvcovalues[j], AkPD, GkLF, Gvga))



		# tauOne_Lin.append(globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][j])
		# OmegaOne_Lin.append(globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][j])




tauvalues2plot = [0*w/(2*np.pi), 0.25*w/(2*np.pi), 0.5*w/(2*np.pi), 0.75*w/(2*np.pi), 1.0*w/(2*np.pi), 1.25*w/(2*np.pi)]
bar = [];
for i in range(len(tauvalues2plot)):
	tempo = np.where(syncState['tau'][:]>tauvalues2plot[i])[0][0]
	# print('tempo:', tempo)
	bar.append( tempo )

# ADD PLOTS
####################################################################################################################################################################################
colorbar=['blue','green','orange','purple', 'cyan', 'black']
# print(globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1]/(2*np.pi))
# print(globalFreq(wref, w,  Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][np.where(globalFreq(wref, w, Kvco, AkPD, GkLF, Gvga, tauf, v, digital, maxp, sync_state1,INV)['tau']<tau1)][-1])
##################################################################################################################################################################




fig1= plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig1.set_size_inches(20,10)
ax1= plt.subplot(211)
ax1.set_xlim(0.0, K(Kvcovalues[-1], AkPD, GkLF, Gvga)/(2.0*np.pi))


[line1]     =    ax1.plot(K(Kvcovalues, AkPD, GkLF, Gvga)/(2.0*np.pi), np.array(pm_vs_wc_singlePLL),'.',color='red');

plt.ylabel(r'phase margin', rotation=90, fontsize=36, labelpad=30)
ax1.tick_params(axis='both', which='major', labelsize=25, pad=1)
plt.grid();
ax1.legend()


ax2= plt.subplot(212)

[line2]     =    ax2.plot(np.array(Kvalue)/(2.0*np.pi), np.array(gm_vs_wc_singlePLL),'.',color='blue');

plt.ylabel(r'gain margin', rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$K$', fontsize=40,labelpad=-5)
ax2.tick_params(axis='both', which='major', labelsize=25, pad=1)
ax2.legend()
plt.grid();
if digital == True:
	plt.savefig('plots/digital_Margins_OneVsK%d_%d_%d.pdf' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_OneVsK%d_%d_%d.png' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_OneVsK%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_OneVsK%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_OneVsK%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_OneVsK%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#




Kvcobar=[0.002*2.0*np.pi, 0.02*2.0*np.pi, 0.05*2.0*np.pi,  0.1*2.0*np.pi, 0.5*2.0*np.pi, 0.8*2.0*np.pi]
fig2= plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig2.set_size_inches(20,10)
ax3= plt.subplot(211)
plt.axhline(y=0, color='r', linestyle='-.')

plt.xscale('log');
for i in range(len(Kvcobar)):
	[lineol00]     =    ax3.plot(f, 20.0*log10(abs(HopenLoopMutuallyCoupledOnePLL(2.0*np.pi*f,
		globalFreq(wref, w, Kvcobar[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[1]],
		globalFreq(wref, w, Kvcobar[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[1]],
		tauf, tauc, v, Kvcobar[i], AkPD, GkLF, Gvga,  digital, model))),'-',ms=1, color= colorbar[i], label=r' K=%.5f' %(K(Kvcobar[i], AkPD, GkLF, Gvga)/(2.0*np.pi)))

plt.grid();
plt.ylabel(r'$20log\left|H_k^\textrm{ OL}(j\gamma)\right|$',  rotation=90, fontsize=36, labelpad=30)
ax3.legend(fontsize=15);
ax3.tick_params(axis='both', which='major', labelsize=25, pad=1)


######## Subplot


ax4= plt.subplot(212)
plt.xscale('log')
plt.axhline(y=-180, color='r', linestyle='-.')
					# PhaseclosedloopMutuallyCoupledOnePLL


for i in range(len(Kvcobar)):
	[lineol00]     =    ax4.plot(f, PhaseOnePLL(2.0*np.pi*f,
		globalFreq(wref, w, Kvcobar[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[1]],
		globalFreq(wref, w, Kvcobar[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[1]],
		tauf, tauc, v, Kvcobar[i], AkPD, GkLF, Gvga,  digital, model),'.',ms=1, color= colorbar[i], label=r'K=%.5f' %(K(Kvcobar[i], AkPD, GkLF, Gvga)/(2.0*np.pi)))

plt.grid();
plt.ylabel(r'phase',  rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=40,labelpad=-5)
ax4.tick_params(axis='both', which='major', labelsize=25, pad=1)

ax4.legend(fontsize=15)
fig2.set_size_inches(20,10)
if digital == True:
	plt.savefig('plots/digital_bode_plot_One_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_One_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_bode_plot_One_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_One_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_bode_plot_One_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_One_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)





fig3= plt.figure(num=3, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig3.set_size_inches(20,10)
ax5= plt.subplot(211)
					# PhaseclosedloopMutuallyCoupledOnePLL

[line1]     =    ax5.plot(K(Kvcovalues, AkPD, GkLF, Gvga)/(2.0*np.pi), np.array(pm_vs_wc_Net),'.',color='red');

plt.ylabel(r'phase margin', rotation=90, fontsize=36, labelpad=30)
ax5.tick_params(axis='both', which='major', labelsize=25, pad=1)
plt.grid();
ax1.legend()


ax6= plt.subplot(212)

[line2]     =    ax6.plot(np.array(KNetvalue)/(2.0*np.pi), np.array(GM_Net),'.',color='blue');

plt.ylabel(r'gain margin',  rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$K$', fontsize=40,labelpad=-5)
ax6.tick_params(axis='both', which='major', labelsize=25, pad=1)
ax6.legend()
plt.grid();
if digital == True:
	plt.savefig('plots/digital_Margins_NetVsK%d_%d_%d.pdf' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_Margins_NetVsK%d_%d_%d.png' %( now.year, now.month, now.day ), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_Margins_NetVsK%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_Margins_NetVsK%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_Margins_NetVsK%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_Margins_NetVsK%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
#







Kvcobar=[0.002*2.0*np.pi, 0.02*2.0*np.pi, 0.05*2.0*np.pi,  0.1*2.0*np.pi, 0.5*2.0*np.pi, 0.8*2.0*np.pi]
fig4= plt.figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig4.set_size_inches(20,10)
ax7= plt.subplot(211)
plt.axhline(y=0, color='r', linestyle='-.')

plt.xscale('log');
for i in range(len(Kvcobar)):
	[lineol00]     =    ax7.plot(f, 20.0*log10(abs(HopenLoopMutuallyCoupledNet(2.0*np.pi*f,
		globalFreq(wref, w, Kvcobar[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[1]],
		globalFreq(wref, w, Kvcobar[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[1]],
		tauf, tauc, v, Kvcobar[i], AkPD, GkLF, Gvga,  digital, model))),'-',ms=1, color= colorbar[i], label=r' K=%.5f' %(K(Kvcobar[i], AkPD, GkLF, Gvga)/(2.0*np.pi)))

plt.grid();
plt.ylabel(r'$20log\left|H_k^\textrm{ OL}(j\gamma)\right|$',  rotation=90, fontsize=36, labelpad=30)
ax7.legend(fontsize=15);
ax7.tick_params(axis='both', which='major', labelsize=25, pad=1)


######## Subplot


ax8= plt.subplot(212)
plt.xscale('log')
plt.axhline(y=-180, color='r', linestyle='-.')
					# PhaseclosedloopMutuallyCoupledOnePLL


for i in range(len(Kvcobar)):
	[lineol00]     =    ax8.plot(f, PhaseNet(2.0*np.pi*f,
		globalFreq(wref, w, Kvcobar[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['Omeg'][bar[1]],
		globalFreq(wref, w, Kvcobar[i], AkPD, GkLF,Gvga, tauf, v, digital, maxp, sync_state1, INV)['tau'][bar[1]],
		tauf, tauc, v, Kvcobar[i], AkPD, GkLF, Gvga,  digital, model),'.',ms=1, color= colorbar[i], label=r'K=%.5f' %(K(Kvcobar[i], AkPD, GkLF, Gvga)))

plt.grid();
plt.ylabel(r'phase', rotation=90, fontsize=36, labelpad=30)
plt.xlabel(r'$\textrm{log}_{10}(f)$', fontsize=40,labelpad=-5)
ax8.tick_params(axis='both', which='major', labelsize=25, pad=1)

ax8.legend(fontsize=15)
fig4.set_size_inches(20,10)
if digital == True:
	plt.savefig('plots/digital_bode_plot_Net_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
	plt.savefig('plots/digital_bode_plot_Net_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
else:
	if coupfunction(coupfun)=='sin':
		plt.savefig('plots/analog_sin_bode_plot_Net_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_sin_bode_plot_Net_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)

	if coupfunction(coupfun)=='cos':
		plt.savefig('plots/analog_cos_bode_plot_Net_different_K_%d_%d_%d.pdf' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)
		plt.savefig('plots/analog_cos_bode_plot_Net_different_K_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=150, bbox_inches=0)




plt.show();
