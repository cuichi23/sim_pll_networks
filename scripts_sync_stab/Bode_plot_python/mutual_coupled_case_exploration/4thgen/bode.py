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
from Bode_lib import solveLinStab, globalFreq, linStabEq,Holnodel,Hcl,Holwithdel,fMolnodel,fMcl, fMolwithdel,analytical

# plot parameter
axisLabel = 12;
titleLabel= 10;
dpi_val	  = 150;
figwidth  =	6;
figheight = 5;

# PLL parameters and state information
wref          = 2.0*np.pi*1.0;
wvco          = 0.99*2.0*np.pi*1.0;                                        # global frequency
tau1		  = 0.0*1E-10;
tau_f         = 0.0;
# beta          = np.pi/2.0;
G             = 1.0;
Kvco          = 2.0*np.pi*0.25;
div           = 1.0;
tau_c         = 1.0/(2.0*np.pi);
LF_order	  = 1.0;

K             = G*Kvco*0.5;
# alpha         = K*np.sin(wref*(tau-tau_f)+beta);
w_cutoff      = 1.0/tau_c
# w             = 2.0*np.pi*f


# choose digital vs analog
digital = False;

# choose Full Expression of the characteristic equation vs the expansion of 3d Order,
#True is the Full
#True is the expansion
expansion=False;
# choose phase or anti-phase synchroniazed states,

sync_state1='inphase';															# choose between inphase, antiphase, entrainment
sync_state2='antiphase';
sync_state3='entrainment';
# define parameters, intrinsic freq, coupling strength, feedback-delay, integration-time LF, divisor of divider

w1    = 1.0*2.0*np.pi				#2.0*np.pi*24E9;
# K    = 0.15*2.0*np.pi			#2.0*np.pi*250E6;
tauf = tau_f
tauc = tau_c;
v	 = div;
c	 = 3E8;
maxp = 17;
###### python library for bode plots (lti)
# list_of_coefficients
# system        = signal.lti([alpha], [div*tau_c, div, alpha])
f             = np.logspace(-7.0, 7.0, num=100000, endpoint=True, base=10.0)
w             = 2.0*np.pi*f
# w, mag, phase1 = signal.bode(system,w)
#
#
# print(Hcl(w_cutoff, wref, wvco, tau1, tau_f, div, tau/_c, K))

fol         = np.vectorize(fMolnodel)
fcl         = np.vectorize(fMcl)
fcl1        = np.vectorize(fMolwithdel)

# print(np.argwhere(Hcl(w, wref, wvco, tau, tau_f, div, tau_c, K)<0.999999*Hcl(w_cutoff, wref, wvco, tau, tau_f, div, tau_c, K))[0])
# print(np.argwhere(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau, tau_f, div, tau_c, K)))<-3.001)[0])
# print(np.where(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau, tau_f, div, tau_c, K)))>-2.9999999999999 and 20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau, tau_f, div, tau_c, K)))<-3.000000000000001))
axis_color  = 'lightgoldenrodyellow'
# f = logspace(-3,3) # frequencies from 10**1 to 10**5
# print(fM2(2.0*np.pi*f))
fig    = plt.figure(num=0, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')



fig1= plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig2, (, ) = plt.subplots(2)

print(180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, div, tau_c, K))
ax= plt.subplot(211)
fig1.suptitle(r'magnitude and phase-response for $H_{ol}*H_{FB}$')
plt.title(r'phase margin of $\omega_{gc}= %0.5f$ and $\omega_c^{\rm LF}= %0.5f$' %(180.0+fcl1(w[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,
															wref, wvco, tau1, tau_f, div, tau_c, K)))<0.00000000001)[0]], wref, wvco, tau1, tau_f, div, tau_c, K), 180.0+fcl1(w_cutoff, wref, wvco, tau1, tau_f, div, tau_c, K)), fontsize=8)
plt.ylabel('dB', fontsize=axisLabel)
plt.xscale('log');
[linecl01] 	=	ax.plot(f, 20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K))),'-', label=r'magnitude,',color='red',);
[vlinecl01]	=	ax.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K)))<0.00000000001)[0]],
							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
							'k--',linestyle='--', label=r'$\omega_{gc}$')
[vlinecl02]	=	ax.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, div, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, div, tau_c, K))[0]],
							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, div, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, div, tau_c, K))[0]]),
							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')

plt.grid();

plt.legend();

ax0= plt.subplot(212)
plt.xlabel(r'f in [Hz]', fontsize=axisLabel); plt.ylabel('degrees', fontsize=axisLabel)
plt.xscale('log')

# [line02] 	= 	ax3.plot(f, fM2(2.0*np.pi*f, wref, wvco, tau, tau_f, div, tau_c, K), label=r'phase-response', lineWidth=2.5);
[linecl03] 	= 	ax0.plot(f, fcl1(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K),'.', label=r'phase-response',  color='red',);
[vlinecl03]	=	ax0.plot((f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K)))<0.00000000001)[0]],
							f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K)))<0.00000000001)[0]]),(-300.0,300.0),
							'k--',linestyle='--', label=r'$\omega_{gc}$')
[vlinecl04]	=	ax0.plot((f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, div, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, div, tau_c, K))[0]],
							f[np.argwhere(Holwithdel(w, wref, wvco, tau1, tau_f, div, tau_c, K)>0.999999*Holwithdel(w_cutoff, wref, wvco, tau1, tau_f, div, tau_c, K))[0]]),
							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')

plt.grid();

plt.legend();




fig2= plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig2, (, ) = plt.subplots(2)
fig2.suptitle(r'magnitude and phase-response for $H_{cl}$')


ax2= plt.subplot(211)
plt.title(r'phase margin of $\omega_c^{\rm LF}= %0.5f$ and $\omega_c^{\rm loop}$ gain$\rightarrow-3$dB$= %0.5f$' %(180.0+fMcl(1.0/(tau_c), wref, wvco, tau1, tau_f, div, tau_c, K), 180.0+fMcl(w[np.argwhere(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K)))<-3.00000000001)[0]], wref, wvco, tau1, tau_f, div, tau_c, K)), fontsize=8)
plt.ylabel('dB', fontsize=axisLabel)
plt.xscale('log');
# [line00] 	=	ax2.plot(f, 20.0*log10(abs(H(2.0*np.pi*f, wref, wvco, tau, tau_f, div, tau_c, K))), label=r'magnitude,', lineWidth=2.5 );
[linecl1] 	=	ax2.plot(f, 20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K))),'-', label=r'magnitude,',color='red',);
[vlinecl1]	=	ax2.plot((f[np.argwhere(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K)))<-3.00000000001)[0]],
							f[np.argwhere(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K)))<-3.00000000001)[0]]),(-300.0,30.0),
							'k--',linestyle='--', label=r'$\omega_c^{\rm loop}$ gain$\rightarrow-3$dB')
[vlinecl11]	=	ax2.plot((f[np.argwhere(Hcl(w, wref, wvco, tau1, tau_f, div, tau_c, K)<0.999999*Hcl(w_cutoff, wref, wvco, tau1, tau_f, div, tau_c, K))[0]],
							f[np.argwhere(Hcl(w, wref, wvco, tau1, tau_f, div, tau_c, K)<0.999999*Hcl(w_cutoff, wref, wvco, tau1, tau_f, div, tau_c, K))[0]]),
							(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')

# [line]=ax2.plot((-3.0,-3.0),(-10000,10000.0),'k--',linestyle='--', label=r'cutoff Frequency= %0.2f' %-3.0)
plt.grid();

plt.legend();

ax3= plt.subplot(212)

plt.xlabel(r'f in [Hz]', fontsize=axisLabel); plt.ylabel('degrees', fontsize=axisLabel)
plt.xscale('log')

# [line02] 	= 	ax3.plot(f, fM2(2.0*np.pi*f, wref, wvco, tau, tau_f, div, tau_c, K), label=r'phase-response', lineWidth=2.5);
[linecl3] 	= 	ax3.plot(f, fcl(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K),'.', label=r'phase-response',  color='red',);
[vlinecl3]	=	ax3.plot((f[np.argwhere(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K)))<-3.00000000001)[0]],
					f[np.argwhere(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K)))<-3.00000000001)[0]]),(-300.0,30.0),
					'k--',linestyle='--', label=r'$\omega_c^{\rm loop}$ gain$\rightarrow-3$dB')
[vlinecl33]	=	ax3.plot((f[np.argwhere(Hcl(w, wref, wvco, tau1, tau_f, div, tau_c, K)<0.999999*Hcl(w_cutoff, wref, wvco, tau1, tau_f, div, tau_c, K))[0]],
f[np.argwhere(Hcl(w, wref, wvco, tau1, tau_f, div, tau_c, K)<0.999999*Hcl(w_cutoff, wref, wvco, tau1, tau_f, div, tau_c, K))[0]]),
					(-300.0,30.0),'k--',linestyle='--', label=r'$\omega_c^{\rm LF}=1/\tau_c$',color='blue')
# [line]=ax.plot((x,x),(0.998,1.0021),'k--',linestyle='--', label=r'cutoff Frequency= %0.2f' %(min_delay*(w)/(2.0*np.pi)/v))
plt.grid();

plt.legend();

fig3= plt.figure(num=3, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# fig2, (, ) = plt.subplots(2)
fig3.suptitle('magnitude and phase-response for the $H_{ol}$')

ax4= plt.subplot(211)
plt.xlabel(r'f in [Hz]', fontsize=axisLabel); plt.ylabel('dB', fontsize=axisLabel)
plt.xscale('log');
# [line00] 	=	ax2.plot(f, 20.0*log10(abs(H(2.0*np.pi*f, wref, wvco, tau, tau_f, div, tau_c, K))), label=r'magnitude,', lineWidth=2.5 );
[lineol1] 	=	ax4.plot(f, 20.0*log10(abs(Holnodel(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K))),'-', label=r'magnitude,',color='red',);
plt.grid();
plt.legend();

ax5= plt.subplot(212)
plt.xlabel(r'f in [Hz]', fontsize=axisLabel); plt.ylabel('degrees', fontsize=axisLabel)
plt.xscale('log')

# [line02] 	= 	ax3.plot(f, fM2(2.0*np.pi*f, wref, wvco, tau, tau_f, div, tau_c, K), label=r'phase-response', lineWidth=2.5);
[lineol3] 	= 	ax5.plot(f, fol(2.0*np.pi*f, wref, wvco, tau1, tau_f, div, tau_c, K),'.', label=r'phase-response',  color='red',);
# [line]=ax.plot((x,x),(0.998,1.0021),'k--',linestyle='--', label=r'cutoff Frequency= %0.2f' %(min_delay*(w)/(2.0*np.pi)/v))
plt.grid();


#
#
# fig4         = figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# ax6          = fig4.add_subplot(111)
#
# # plot grid, labels, define intial values
# plt.suptitle(r'The analytic expression characteristic equation ');
# plt.title(r'with cross coupling delay $\tau=%0.2f$' %tau1, fontsize=8)
#
# plt.grid()
# plt.xlabel(r'$\omega\tau/2\pi$', fontsize=18)
# plt.ylabel(r'$\sigma/\omega$, $\gamma/\omega$', fontsize=18)
# # s=[0.0,0.5]
# # draw the initial plot																																							wref, w, Omega, tau, tauf, K, tauc, v, digital, maxp, sync_state
# # [dotanalytical]=ax6.plot(np.asarray(2.0*np.pi/(wref))*analytical(wref,wvco, wref,0.0,tauf, K, tauc, div, digital, maxp, sync_state3), np.asarray(2.0*np.pi/(wref))*analytical(wref,wvco, wref,0.0,tauf, K, tauc, div, digital, maxp, sync_state3), 'o', color='black', label=r'$\sigma$=Re$(\lambda_max)$')
# [lineSigmaIn] = ax6.plot(np.asarray(wref/(2.0*np.pi))*globalFreq(wref, wvco, K, tau_f, div, digital, maxp, sync_state3)['tau'], np.asarray(2.0*np.pi/(wref))*solveLinStab(wref, wvco, globalFreq(wref, wvco, K, tau_f, div, digital, maxp, sync_state3)['Omeg'],
# 					globalFreq(wref, wvco, K, tau_f, div, digital, maxp, sync_state3)['tau'], tau_f, K, tauc, div, digital, maxp, sync_state3, expansion)['ReMax'],
# 					linewidth=2, color='red', label=r'$\sigma$=Re$(\lambda_max)$')
# 					# globalFreq(wref, wvco, K, tau_f, div, digital, maxp, sync_state3)['tau'], tau_f, K, tauc, div, digital, maxp, sync_state3, expansion)['ReMax'])
#
# analytical(wref,wvco, globalFreq(wref, wvco, K, tau_f, div, digital, maxp, sync_state3)['Omeg'],globalFreq(wref, wvco, K, tau_f, div, digital, maxp, sync_state3)['tau'],tauf, K, tauc, div, digital, maxp, sync_state3)
#
# #
# # [lineGammaIn] = ax6.plot(np.asarray(wref/(2.0*np.pi))*globalFreq(wref, wvco, K, tau_f, div, digital, maxp, sync_state3)['tau'], np.asarray(1.0/(wref))*solveLinStab(wref, wvco, globalFreq(wref, wvco, K, tau_f, div, digital, maxp, sync_state3)['Omeg'],
# # 					globalFreq(wref, wvco, K, tau_f, div, digital, maxp, sync_state3)['tau'], tau_f, K, tauc, div, digital, maxp, sync_state3, expansion)['ImMax'],
# # 					'.', color='blue', label=r'$\gamma$=Im$(\lambda)$')
# # print(solveLinStab(wref, 0.01,  wref, 0.01, tau_f, K, tauc, div, digital, maxp, sync_state3, expansion)['ReMax'])
# [vlinetau]	=	ax6.plot((tau1,tau1),(-4.0, 4.10), 'k--',linestyle='--', label=r'$\tau$ slider ')
# #
# # fig4= plt.figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
# ax6          = fig4.add_subplot(111)
#
# # plot grid, labels, define intial values
# plt.title(r'Im($\lambda$) vs Re($\lambda$) inphase');
# plt.grid()
# plt.xlabel(r'Re($\lambda$)', fontsize=18)
# plt.ylabel(r'Im($\lambda$)', fontsize=18)
# tauf_0  = tauf;#tauf*w/(2.0*np.pi);
# tauc_0  = tauc;
# K_0     = K;
# v_0		= div;
# c_0		= c;
# # draw the initial plot
# [lineNyq] = ax6.plot(solveLinStab(wref, tau1, tauf_0, K_0, tauc_0, v_0, digital, maxp, sync_state1, expansion)['ReMax'], solveLinStab(wref, tau1, tauf_0, K_0, tauc_0, v_0, digital, maxp, sync_state1, expansion)['ImMax'], '.', color='red', label=r'$\sigma$=Re$(\lambda)$')
#
# plt.legend();


tau_f_slider_ax  = fig.add_axes([0.15, 0.67, 0.65, 0.1], facecolor=axis_color)
tau_f_slider     = Slider(tau_f_slider_ax, r'$\tau^f$', 0.0, 1.0, valinit=tau_f)

tau_slider_ax  = fig.add_axes([0.15, 0.45, 0.65, 0.1], facecolor=axis_color)
tau_slider     = Slider(tau_slider_ax, r'$\tau$', 0.0, 3.0*2.0*np.pi/wref, valinit=tau1,valfmt='%0.5f')
# Draw another slider
wvco_slider_ax = fig.add_axes([0.15, 0.23, 0.65, 0.1], facecolor=axis_color)
wvco_slider    = Slider(wvco_slider_ax, r'$\omega_{vco}$', 2.0*(7.1/8.0)*np.pi, (8.9/8.0)*2.0*np.pi, valinit=wvco)
# Draw another slider
tau_c_slider_ax  = fig.add_axes([0.15, 0.01, 0.65, 0.1], facecolor=axis_color)
tau_c_slider     = Slider(tau_c_slider_ax, r'$\tau^c$', (1.0/(10.0*2.0*np.pi)), (1.0/(0.1*2.0*np.pi)), valinit=tau_c)



print('phase margin of the cut off freq=', 180.0+fMcl(1.0/(tau_c_slider.val), wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K))
print('phase margin of the -3dB cut off =',180.0+fMcl(w[np.argwhere(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K)))<-3.00000000001)[0]], wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K))




def sliders_on_changed(val):
	global digital

	print('phase margin of the cut off freq=', 180.0+fMcl(1.0/(tau_c_slider.val), wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K))
	print('phase margin of the -3dB cut off =',180.0+fMcl(w[np.argwhere(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K)))<-3.00000000001)[0]], wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K))

	linecl01.set_ydata(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K))));
	linecl01.set_xdata(f);
	linecl03.set_ydata(fcl1(2.0*np.pi*f, wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K));
	linecl03.set_xdata(f);


	linecl1.set_ydata(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K))));
	linecl1.set_xdata(f);

	vlinecl1.set_xdata(f[np.argwhere(20.0*log10(abs(Hcl(2.0*np.pi*f,  wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K)))<-3.00000000001)[0]]);
	vlinecl3.set_xdata(f[np.argwhere(20.0*log10(abs(Hcl(2.0*np.pi*f,  wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K)))<-3.00000000001)[0]]);
	vlinecl01.set_xdata(f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,  wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K)))<0.00000000001)[0]]);
	vlinecl03.set_xdata(f[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f,  wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K)))<0.00000000001)[0]]);
	vlinecl11.set_xdata(f[np.argwhere(Hcl(w,  wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K)<0.999999*Hcl(w_cutoff,  wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K))[0]]);
	vlinecl33.set_xdata(f[np.argwhere(Hcl(w,  wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K)<0.999999*Hcl(w_cutoff,  wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K))[0]]);
	vlinetau.set_xdata(tau_slider.val)

	linecl3.set_ydata(fcl(2.0*np.pi*f, wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K));
	linecl3.set_xdata(f);

	lineol1.set_ydata(20.0*log10(abs(Holnodel(2.0*np.pi*f, wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K))));
	lineol1.set_xdata(f);
	# line02.set_ydata(fM2(2.0*np.pi*f, wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K));
	# line02.set_xdata(f);
	lineol3.set_ydata(fol(2.0*np.pi*f, wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K));
	lineol3.set_xdata(f);
	lineSigmaIn.set_ydata(solveLinStab(wref, wvco_slider.val, globalFreq(wref, wvco_slider.val, K, tau_f_slider.val, div, digital, maxp, sync_state3)['Omeg'],
	 					globalFreq(wref, wvco_slider.val, K, tau_f_slider.val, v, digital, maxp, sync_state3)['tau'],
						tau_f_slider.val, K, tau_c_slider.val, div, digital, maxp, sync_state3, expansion)['ReMax'])

	lineSigmaIn.set_xdata(np.asarray(wref/(2.0*np.pi))*globalFreq(wref, wvco_slider.val,K, tau_f_slider.val, div, digital, maxp, sync_state3)['tau'])
	# lineGammaIn.set_ydata(solveLinStab(wref, wvco_slider.val, globalFreq(wref, wvco_slider.val, K, tau_f_slider.val, div, digital, maxp, sync_state3)['Omeg'],
	# 					globalFreq(wref, wvco_slider.val, K, tau_f_slider.val, v, digital, maxp, sync_state3)['tau'],
	# 					 tau_f_slider.val, K, tau_c_slider.val,div, digital, maxp, sync_state3, expansion)['ImMax'])
	#
	# lineGammaIn.set_xdata(np.asarray(wref/(2.0*np.pi))* globalFreq(wref, wvco_slider.val, K, tau_f_slider.val, div, digital, maxp, sync_state3)['tau'])
	ax6.set_title(r' with cross coupling delay $\tau=%0.5f$' %tau_slider.val);
	ax2.set_title(r'phase margin of $\omega_c^{\rm LF}= %0.5f$ and $\omega_c^{\rm loop}$ gain$\rightarrow-3$dB$= %0.5f$' %(180.0+fMcl(1.0/(tau_c_slider.val), wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K),
					180.0+fMcl(w[np.argwhere(20.0*log10(abs(Hcl(2.0*np.pi*f, wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K)))<-3.00000000001)[0]],
					wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K)))
	ax.set_title(r'phase margin of $\omega_{gc}= %0.5f$' %(180.0+fcl1(w[np.argwhere(20.0*log10(abs(Holwithdel(2.0*np.pi*f, wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K)))<0.00000000001)[0]],
					wref, wvco_slider.val, tau_slider.val, tau_f_slider.val, div, tau_c_slider.val, K)))

	# analytical(wref,wvco, globalFreq(wref, wvco, K, tau_f, div, digital, maxp, sync_state3)['Omeg'],globalFreq(wref, wvco, K, tau_f, div, digital, maxp, sync_state3)['tau'],tauf, K, tauc, div, digital, maxp, sync_state3)

	analytical(wref, wvco_slider.val, globalFreq(wref, wvco_slider.val, K, tau_f_slider.val, div, digital, maxp, sync_state3)['Omeg'],
					0.0, tau_f_slider.val, K, tau_c_slider.val, div, digital, maxp, sync_state3)
	# lineNyq.set_ydata( solveLinStab(wref,tau_slider.val, tau_f_slider.val, K_0, tau_c_slider.val, v_0, digital, maxp, sync_state1, expansion)['ImMax'])
	# lineNyq.set_xdata( solveLinStab(wref,tau_slider.val, tau_f_slider.val, K_0, tau_c_slider.val, v_0, digital, maxp, sync_state1, expansion)['ReMax'])

# 	# recompute the ax.dataLim
	ax.relim();
	ax0.relim();
	ax2.relim()
	ax3.relim();
	ax4.relim()
	ax5.relim();
	ax6.relim()
# 	# update ax.viewLim using the new dataLim
	ax.autoscale_view();
	ax0.autoscale_view();

	ax2.autoscale_view();
	ax3.autoscale_view();
	ax4.autoscale_view();
	ax5.autoscale_view();
	ax6.autoscale_view()
	plt.draw()
	fig.canvas.draw_idle();
	fig1.canvas.draw_idle();
	fig2.canvas.draw_idle();
	fig3.canvas.draw_idle();
	fig4.canvas.draw_idle();
	# fig5.canvas.draw_idle();

tau_f_slider.on_changed(sliders_on_changed)
tau_slider.on_changed(sliders_on_changed)
wvco_slider.on_changed(sliders_on_changed)
tau_c_slider.on_changed(sliders_on_changed)
plt.show();
