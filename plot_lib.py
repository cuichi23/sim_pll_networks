#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.ma as ma
import matplotlib
import codecs
import csv
import os, gc, sys
if not os.environ.get('SGE_ROOT') == None:																				# this environment variable is set within the queue network, i.e., if it exists, 'Agg' mode to supress output
	print('NOTE: \"matplotlib.use(\'Agg\')\"-mode active, plots are not shown on screen, just saved to results folder!\n')
	matplotlib.use('Agg') #'%pylab inline'
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import optimize
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#from scipy.interpolate import spline
from scipy.special import lambertw
from scipy.signal import square
import itertools
import math

import evaluation_lib as eva

import datetime
now = datetime.datetime.now()

''' Enable automatic carbage collector '''
gc.enable();

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
		'color'  : 'black',
		'weight' : 'normal',
		'size'   : 16,
		}

annotationfont = {
		'family' : 'monospace',
		'color'  : (0, 0.27, 0.08),
		'weight' : 'normal',
		'size'   : 14,
		}

# plot parameter
axisLabel = 60;
tickSize  = 35;
titleLabel= 10;
dpi_val	  = 150;
figwidth  = 6;
figheight = 5;

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def prepareDictsForPlotting(dictPLL, dictNet):

	if dictPLL['cutFc'] == None:
		dictPLL.update({'cutFc': np.inf})

	if not np.abs(dictPLL['intrF']) > 1E-17:									# for f=0, there would otherwies be a float division by zero
		dictPLL.update({'intrF': 1})
		print('Since intrinsic frequency was zero: for plotting set to one to generate boundaries!')


	return dictPLL, dictNet

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotPSD(dictPLL, dictNet, dictData, plotList=[], saveData=False):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	f = []; Pxx_db = [];
	if plotList:
		for i in range(len(plotList)):											# calculate spectrum of signals for the oscillators specified in the list
			ftemp, Pxx_temp = eva.calcSpectrum( dictData['phi'][:,plotList[i]], dictPLL, dictNet )
			f.append(ftemp[0]); Pxx_db.append(Pxx_temp[0])
	else:
		plotList = [];
		for i in range(len(dictData['phi'][0,:])):								# calculate spectrum of signals for all oscillators
			ftemp, Pxx_temp = eva.calcSpectrum( dictData['phi'][:,i], dictPLL, dictNet )
			f.append(ftemp[0]); Pxx_db.append(Pxx_temp[0]); plotList.append(i)

	peak_power_val = [];
	fig1 = plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig1.canvas.set_window_title('spectral density of synchronized state')		# plot spectrum
	fig1.set_size_inches(20,10)

	for i in range(len(f)):
		peak_freq_coup1 = np.argmax(Pxx_db[i]);									# find the principle peak of the free-running SLL
		peak_freq1_val  = f[i][peak_freq_coup1];
		peak_power_val.append(Pxx_db[i][peak_freq_coup1]);
		plt.plot(f[i], Pxx_db[i], '-', label='PLL%i' %(plotList[i]))
	plt.title(r'power spectrum $\Delta f=$%0.5E, peak at $Pxx^\textrm{peak}$=%0.2f' %((f[0][2]-f[0][1]), peak_power_val[0]), fontsize=axisLabel)
	plt.xlabel('frequencies [Hz]', fontsize= axisLabel); plt.ylabel('P [dBm]', fontsize = axisLabel);
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	plt.legend()
	plt.grid()
	try:
		plt.ylim([np.min(Pxx_db[i][peak_freq_coup1:]), peak_power_val[0]+5]);
	except:
		print('Could not determine peak_power_val!')
	plt.xlim(0, 12.5*dictPLL['intrF']);
	plt.savefig('results/powerdensity_dB_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/powerdensity_dB_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=150)
	plt.savefig('results/powerdensity_dB_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
	plt.xlim(peak_freq1_val-1.2*np.min(dictPLL['coupK']),peak_freq1_val+1.2*np.max(dictPLL['coupK']));
	plt.plot(f[0][peak_freq_coup1-int(0.1*dictPLL['intrF']/(f[0][2]-f[0][1]))], Pxx_db[0][peak_freq_coup1], 'r*',
			 f[0][peak_freq_coup1+int(0.1*dictPLL['intrF']/(f[0][2]-f[0][1]))], Pxx_db[0][peak_freq_coup1], 'r*')
	plt.savefig('results/powerdensity1stHarm_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/powerdensity1stHarm_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=150)
	plt.savefig('results/powerdensity1stHarm_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
	plt.xlim(0, 8.5*dictPLL['intrF']);

	fig2 = plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')	# plot spectrum
	fig2.canvas.set_window_title('one-sided spectral density')
	fig2.set_size_inches(20,10);

	xHz = 0.001; onsidedPSD_params = []; oneSidPSDwidthsm3dB = [];				# distance from the principle peak to measure damping
	Freqres = f[0][3]-f[0][2]; linestyle = ['-', '--', '-', '--', '-', '--'];
	for i in range (len(f)):
		peak_freq1_val = 0; coup1_delt_3dB = 0;
		# mutually coupled SLL1
		peak_freq_coup1 = np.argmax(Pxx_db[i]);									# find the index of the principle peak (max dB) of the free-running SLL
		peak_freq1_val  = f[i][peak_freq_coup1];								# use the above index to identify the frequency of the peaks location
		coup1_times_X	= np.argmax(f[i] >= 2.25*peak_freq1_val);				# find the index for the frequency being 2.25 times that of the peak
		m3dB_freqcind1  = peak_freq_coup1 + np.where(Pxx_db[i][peak_freq_coup1:]<=(Pxx_db[i][peak_freq_coup1]-3.0))[0][0]; # find the index associated to a power drop of -3dB w.r.t. the peak's value
		print('\n\nm3dB_freqcind1:', m3dB_freqcind1, '\n\n')
		m3dB_freqc_val  = f[i][m3dB_freqcind1];									# extract the frequency at -3dB

		coup1_delt_3dB  = np.abs( m3dB_freqc_val - peak_freq1_val - Freqres )

		print('Calculating: f[peak]-f[-3dBm]=', peak_freq1_val,'-',m3dB_freqc_val,'=',coup1_delt_3dB, '\n power 1st harmonic', Pxx_db[i][peak_freq_coup1],'\n')
		print('\nPxx_db[',i,'][',peak_freq_coup1,']-3.0=',Pxx_db[i][peak_freq_coup1]-3.0)
		#print('TEST Pxx_dBm[',i,'][ peak_freq_coup1+np.argmin(Pxx_db[',i,'][',peak_freq_coup1,':]<=(Pxx_db[',i,'][',peak_freq_coup1,']-3)) ]: ',
		#					Pxx_db[i][ peak_freq_coup1+np.argmin(Pxx_db[i][peak_freq_coup1:].copy()<=(Pxx_db[i][peak_freq_coup1]-3.0)) ],
		#					' -> frequency where PxxMax-3dB:', m3dB_freqc_val,'Hz')
		print('np.where(Pxx_db[',i,'][',peak_freq_coup1,':]<=(Pxx_db[',i,'][',peak_freq_coup1,']-3))[0][0]=', np.where(Pxx_db[i][peak_freq_coup1:]<=(Pxx_db[i][peak_freq_coup1]-3.0))[0][0], '\n')
		# calculate linear function between first two points of one-sided powerspectrum: y = slope * x + yinter, interPol{1,2} are the relate to the y-coordinates between which Pxx-3dB lies
		# interpolP1 = peak_freq_coup1 + np.where(Pxx_db[i][peak_freq_coup1:] > (Pxx_db[i][peak_freq_coup1]-3.0))[0][-1]	# find the results in the PSD vectors that are adjacent to the -3dB point
		# interpolP2 = peak_freq_coup1 + np.where(Pxx_db[i][peak_freq_coup1:] < (Pxx_db[i][peak_freq_coup1]-3.0))[0][0]		# PROBLEM IF Pxx goes up again!!!!!!! sketch to see
		interpolP1 = peak_freq_coup1 + np.where(Pxx_db[i][peak_freq_coup1:] < (Pxx_db[i][peak_freq_coup1]-3.0))[0][0]	# find the first point in the PSD smaller than PxxMax-3dB and then take the prior point
		interpolP2 = interpolP1-1																						# as the second to interpolate
		#print('{interpolP1, interpolP2}:', interpolP1, interpolP2)
		slope  = ( Pxx_db[i][interpolP2]-Pxx_db[i][interpolP1]) / ( f[i][interpolP2]-f[i][interpolP1])
		yinter = Pxx_db[i][interpolP1] - slope * f[i][interpolP1]
		#print('{slope, yinter}:', slope, yinter)
		# slope  = ( Pxx_db[i][peak_freq_coup1+1]-Pxx_db[i][peak_freq_coup1]) / ( f[i][peak_freq_coup1+1]-f[i][peak_freq_coup1])
		# yinter = Pxx_db[i][peak_freq_coup1] - slope * f[i][peak_freq_coup1]
		fm3dB  = ( Pxx_db[i][peak_freq_coup1]-3.0 - yinter ) / slope
		oneSidPSDwidthsm3dB.append(fm3dB-f[i][peak_freq_coup1])
		print('\nwidths (HWHM) of PSDs obtained from one-sided PSD with interpolation for all oscis:', oneSidPSDwidthsm3dB)
		print('Mean of widths of PSDs obtained from one-sided PSD with interpolation for all oscis:', np.mean(oneSidPSDwidthsm3dB))
		print('Std of widths of PSDs obtained from one-sided PSD with interpolation for all oscis:', np.std(oneSidPSDwidthsm3dB), '\n')
		print('For the mutually coupled SLL',i,' we find the principle peak at f =', peak_freq1_val, ', -3dBm delta_f =', coup1_delt_3dB, ', and hence a quality factor Q = ', peak_freq1_val/(2*(fm3dB-f[i][peak_freq_coup1])))
		plt.plot(10.0*np.log10(fm3dB-peak_freq1_val+Freqres), Pxx_db[i][peak_freq_coup1]-3.0, 'r*', markersize=2)
		if coup1_delt_3dB == 0:
			print('frequency resolution of power spectrum too large or power spectrum approaching delta-like peak!')
		try:
			onsidedPSD_params.append([peak_freq1_val, coup1_delt_3dB])
		except:
			onsidedPSD_params.append([0, 0])
		if (m3dB_freqcind1 < 3.1 and m3dB_freqcind1 > 2.9):
			#plt.plot(10.0*np.log10(1E-12), Pxx_db[i][peak_freq_coup1], 'r*', markersize=2)
			plt.plot(10.0*np.log10(f[i][m3dB_freqcind1]-peak_freq1_val), Pxx_db[i][m3dB_freqcind1], 'r+', markersize=2)
		else:
			print('CHECK frequency resolution of power spectrum and noise strength. Cannot use this method.')
		if dictNet['topology'] == 'compareEntrVsMutual':
			plt.plot(10.0*np.log10(f[i][peak_freq_coup1:coup1_times_X]-peak_freq1_val+Freqres), Pxx_db[i][peak_freq_coup1:coup1_times_X], linestyle[i], label='PSD PLL%i' %(i), markersize=2)
		else:
			plt.plot(10.0*np.log10(f[i][peak_freq_coup1:coup1_times_X]-peak_freq1_val+Freqres), Pxx_db[i][peak_freq_coup1:coup1_times_X], label='PSD PLL%i' %(plotList[i]), markersize=2)
	try:
		plt.title(r'$\gamma_0^{(\textrm{PSDfit})}=$%0.4E, $\gamma_1=$%0.4E, $\gamma_2=$%0.4E' %(params[0][0], params[1][0], params[2][0]), fontdict = titlefont)
	except:
		print('No (two-sided, 1st harmonic) PSD fits available!')
	if dictNet['Nx']*dictNet['Ny'] == 2:
		onsidedPSD_params.append([0, 0])										# necessary, otherwise error on write-out to csv file
	#plt.plot(10.0*np.log10(powerspecPLL1['f'][0][peak_freq_coup1:coup1_times_X].copy()-peak_freq1_val), !!!!! , 'y-', label=r'$1/f^2$')
	plt.legend()
	# plt.xlim([0,f01+20*max(Kvco1,Kvco2)]);	#plt.ylim(-100,0);
	plt.xlabel(r'$10\log_{10}\left(f-f_{\rm peak}\right)$ [Hz]', fontsize=axisLabel); plt.ylabel(r'$P$ [dBm]', fontsize=axisLabel)
	plt.tick_params(axis='both', which='major', labelsize=35); plt.grid();
	plt.savefig('results/onsidedPSD_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/onsidedPSD_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=150)
	plt.savefig('results/onsidedPSD_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)

	fig1 = plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	for i in range (len(f)):
		plt.plot(oneSidPSDwidthsm3dB[i]+f[i][peak_freq_coup1], Pxx_db[i][peak_freq_coup1]-3.0, 'r*', markersize=2)

	if saveData:
		np.savez('results/powerSpec_K%.2f_Fc%.4f_FOm%.2f_tau%.2f_c%.7e_%d_%d_%d.npz' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), powerspec=np.array([f, Pxx_db]))

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotPhases2pi(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	labeldict 	= {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 	= {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$', 'phi': r'$\phi$', 't': r'$t$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color		= ['blue', 'red', 'purple', 'cyan', 'green', 'yellow'] #'magenta'
	linet		= ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig3 = plt.figure(num=3, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig3.canvas.set_window_title('phases')  									# plot the phase


	plt.plot(dictData['t'], dictData['phi']%(2*np.pi), linewidth=1, linestyle=linet[0])
	plt.plot(dictData['t'][dictNet['max_delay_steps']-1], dictData['phi'][int(dictNet['max_delay_steps'])-1,0]%(2*np.pi)+0.05,'go')

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict = labelfont)
	plt.ylabel(r'$\theta(t)_{\textrm{mod}\,2\pi}$', fontdict = labelfont)
	# plt.legend()

	plt.savefig('results/phases2pi-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/phases2pi-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotPhasesInf(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	labeldict 	= {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 	= {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$', 'phi': r'$\phi$', 't': r'$t$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color		= ['blue', 'red', 'purple', 'cyan', 'green', 'yellow'] #'magenta'
	linet		= ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig4 = plt.figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig4.canvas.set_window_title('phases')  							 		# plot the phase

	plt.plot(dictData['t'], dictData['phi'], linewidth=1, linestyle=linet[0])
	plt.plot(dictData['t'][dictNet['max_delay_steps']-1], dictData['phi'][int(dictNet['max_delay_steps'])-1,0]+0.05,'go')

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict = labelfont)
	plt.ylabel(r'$\theta(t)$', fontdict = labelfont)
	# plt.legend()

	plt.savefig('results/phasesInf-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/phasesInf-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotFrequency(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	labeldict 	= {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 	= {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$', 'phi': r'$\phi$', 't': r'$t$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color		= ['blue', 'red', 'purple', 'cyan', 'green', 'yellow'] #'magenta'
	linet		= ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig5 = plt.figure(num=5, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig5.canvas.set_window_title('frequency')  							 		# plot the phase

	phidot = np.diff(dictData['phi'], axis=0)/dictPLL['dt'];
	plt.plot(dictData['t'][0:-1], phidot, linewidth=1, linestyle=linet[0])
	plt.plot(dictData['t'][dictNet['max_delay_steps']-1], phidot[int(dictNet['max_delay_steps'])-1,0]+0.05,'go')

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict = labelfont)
	plt.ylabel(r'$\theta(t)$', fontdict = labelfont)
	# plt.legend()

	plt.xlabel(r'$t\,[T_{\omega}]$', fontdict = labelfont)
	plt.ylabel(r'$\dot{\phi}(t)$ [rad Hz]', fontdict = labelfont)
	plt.savefig('results/freq-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/freq-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
	plt.xlim([np.mean(dictPLL['transmission_delay']) - 25*1.0/(dictPLL['intrF']), np.mean(dictPLL['transmission_delay']) + 35*1.0/(dictPLL['intrF'])]);
	plt.ylim([0.99*np.min(phidot[0:int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+25*1.0/(dictPLL['intrF']*dictPLL['dt']))-1,:]), 1.01*np.max(phidot[0:int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+25*1.0/(dictPLL['intrF']*dictPLL['dt']))-1,:])]);
	plt.savefig('results/freqInit-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/freqInit-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
	plt.xlim([np.mean(dictPLL['transmission_delay']) - 25*1.0/(dictPLL['intrF']), np.mean(dictPLL['transmission_delay']) + 400*1.0/(dictPLL['intrF'])]);
	plt.ylim([np.min(phidot[int(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] - 25*1.0/(dictPLL['sampleF']/dictPLL['intrF'])):int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+400*1.0/(dictPLL['intrF']*dictPLL['dt']))-1,:])-0.05, np.max(phidot[int(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] - 25*1.0/(dictPLL['sampleF']/dictPLL['intrF'])):int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+400*1.0/(dictPLL['intrF']*dictPLL['dt']))-1,:])+0.05]);
	plt.savefig('results/freqInit1-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/freqInit1-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
	plt.xlim([0, dictData['t'][-1]]);
	plt.ylim([np.min(phidot[int(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']+0.5*dictPLL['sampleF']/dictPLL['intrF']):,:])-0.05, np.max(phidot[int(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']+0.5*dictPLL['sampleF']/dictPLL['intrF']):,:])+0.05]);

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotOrderPara(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)
	r, orderparam, F1 = eva.obtainOrderParam(dictPLL, dictNet, dictData)

	fig6 = plt.figure(num=6, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig6.canvas.set_window_title('order parameter over time')					# plot the order parameter in dependence of time

	plt.plot(dictData['t'], orderparam)
	plt.plot(np.mean(dictPLL['transmission_delay']), orderparam[int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))], 'yo', ms=5)				# mark where the simulation starts
	plt.axvspan(dictData['t'][-int(dictPLL['timeSeriesAverTime']*1.0/(F1*dictPLL['dt']))], dictData['t'][-1], color='b', alpha=0.3)
	plt.title(r'mean order parameter $\bar{R}=$%.2f, and $\bar{\sigma}=$%.4f' %(np.mean(orderparam[-int(round(dictPLL['timeSeriesAverTime']*1.0/(F1*dictPLL['dt']))):]), np.std(orderparam[-int(round(dictPLL['timeSeriesAverTime']*1.0/(F1*dictPLL['dt']))):])), fontdict = titlefont)
	plt.xlabel(r'$t\,[T_{\omega}]$', fontdict = labelfont)
	plt.ylabel(r'$R(t,m_x=%d,m_y=%d )$' %(dictNet['mx'],dictNet['my']), fontdict = labelfont)
	plt.savefig('results/orderP-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/orderP-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
	plt.xlim([dictData['t'][-int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']+75*1.0/(F1*dictPLL['dt'])))], dictData['t'][-1]]); # plt.ylim([]);
	plt.savefig('results/orderPFin-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/orderPFin-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
	plt.xlim([0, np.mean(dictPLL['transmission_delay'])+125*1.0/(F1)]); # plt.ylim([]);
	plt.savefig('results/orderPInit-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/orderPInit-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
	#print('\nlast entry order parameter: R-1 = %.3e' % (orderparam[-1]-1) )
	#print('\nlast entries order parameter: R = ', orderparam[-25:])

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotPhaseRela(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	fig7 = plt.figure(num=7, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig7.canvas.set_window_title('phase relations')
	if not dictNet['topology'] == 'compareEntrVsMutual':
		plt.plot(dictData['t'][::dictPLL['sampleFplot']],((dictData['phi'][::dictPLL['sampleFplot'],0]-dictData['phi'][::dictPLL['sampleFplot'],1]+np.pi)%(2.*np.pi))-np.pi,label=r'$\phi_{0}-\phi_{1}$')			#math.fmod(phi[:,:], 2.*np.pi))
		if not dictNet['Nx']*dictNet['Ny'] == 2:
			plt.plot(dictData['t'][::dictPLL['sampleFplot']],((dictData['phi'][::dictPLL['sampleFplot'],1]-dictData['phi'][::dictPLL['sampleFplot'],2]+np.pi)%(2.*np.pi))-np.pi,label=r'$\phi_{1}-\phi_{2}$')
			plt.plot(dictData['t'][::dictPLL['sampleFplot']],((dictData['phi'][::dictPLL['sampleFplot'],0]-dictData['phi'][::dictPLL['sampleFplot'],2]+np.pi)%(2.*np.pi))-np.pi,label=r'$\phi_{0}-\phi_{2}$')
		plt.plot(np.mean(dictPLL['transmission_delay']), ((dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),0]-dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),1]+np.pi)%(2.*np.pi))-np.pi, 'yo', ms=5)
		#plt.axvspan(t[-int(5.5*1.0/(F1*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
		if dictNet['Nx']*dictNet['Ny']>=3:
			plt.title(r'phases $\phi_{0}=%.4f$, $\phi_{1}=%.4f$, $\phi_{R}=%.4f$  [rad]' %( (-1)*(np.mod(dictData['phi'][-10][2]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi),
																np.mod(dictData['phi'][-10][1]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi - (np.mod(dictData['phi'][-10][2]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi), 0 ), fontdict = titlefont)
		else:
			plt.title(r'phases [rad]', fontdict = titlefont)
		plt.xlabel(r'$t\,[T_{\omega}]$', fontdict = labelfont)
		plt.ylabel(r'$\phi_k(t)-\phi_0(t)$', fontdict = labelfont)
		plt.legend();
		plt.savefig('results/phaseRela-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
		plt.savefig('results/phaseRela-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
	else:
		plt.plot(dictData['t'][-int(25*1.0/(F1*dt)):],((dictData['phi'][-int(25*1.0/(F1*dt)):,0]-dictData['phi'][-int(25*1.0/(F1*dt)):,1]+np.pi)%(2.*np.pi))-np.pi,'-',label=r'$\phi_{0}-\phi_{1}$ mutual')
		plt.plot(dictData['t'][-int(25*1.0/(F1*dt)):],((dictData['phi'][-int(25*1.0/(F1*dt)):,3]-dictData['phi'][-int(25*1.0/(F1*dt)):,2]+np.pi)%(2.*np.pi))-np.pi,'--',label=r'$\phi_{3}-\phi_{2}$ entrain')
		#plt.plot((t[-int(12*1.0/(F1*dt)):]*dt),((dictData['phi'][-int(12*1.0/(F1*dt)):,0]-dictData['phi'][-int(12*1.0/(F1*dt)):,5]+np.pi)%(2.*np.pi))-np.pi,'-',label=r'$\phi_{0}-\phi_{5}$ mutual  vs freeRef')
		#plt.plot((t[-int(12*1.0/(F1*dt)):]*dt),((dictData['phi'][-int(12*1.0/(F1*dt)):,3]-dictData['phi'][-int(12*1.0/(F1*dt)):,5]+np.pi)%(2.*np.pi))-np.pi,'--',label=r'$\phi_{3}-\phi_{5}$ entrain vs freeRef')
		#plt.plot((t[-int(12*1.0/(F1*dt)):]*dt),((dictData['phi'][-int(12*1.0/(F1*dt)):,0]-dictData['phi'][-int(12*1.0/(F1*dt)):,4]+np.pi)%(2.*np.pi))-np.pi,'-.',label=r'$\phi_{0}-\phi_{4}$ mutual  vs freePLL')
		#plt.plot((t[-int(12*1.0/(F1*dt)):]*dt),((dictData['phi'][-int(12*1.0/(F1*dt)):,3]-dictData['phi'][-int(12*1.0/(F1*dt)):,4]+np.pi)%(2.*np.pi))-np.pi,'-.',label=r'$\phi_{3}-\phi_{4}$ entrain vs freePLL')
		#plt.plot(np.mean(dictPLL['transmission_delay']), ((dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),0]-dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),1]+np.pi)%(2.*np.pi))-np.pi, 'yo', ms=5)
		plt.axvspan(dictData['t'][-int(5.5*1.0/(F1*dt))], dictData['t'][-1], color='b', alpha=0.3)
		plt.title(r'phases-differences between the clocks', fontdict = titlefont)
		plt.xlabel(r'$t\,[T_{\omega}]$', fontdict = labelfont)
		plt.ylabel(r'$\phi_k(t)-\phi_0(t)$', fontdict = labelfont)
		plt.legend();
		plt.savefig('results/phaseRela-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
		plt.savefig('results/phaseRela-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)


	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotPhaseDiff(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	fig8 = plt.figure(num=8, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig8.canvas.set_window_title('phase configuration with respect to the phase of osci 0')
	if len(dictData['phi'][:,0]) > dictPLL['treshold_maxT_to_plot'] * (1.0/(dictPLL['intrF']*dictPLL['dt'])):	# (1.0/(dictPLL['intrF']*dictPLL['dt'])) steps for one period
		for i in range(len(dictData['phi'][0,:])):
			labelname = r'$\phi_{%i}$-$\phi_{0}$' %(i);
			plt.plot((dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+dictPLL['treshold_maxT_to_plot']*1.0/(F1*dt)):dictPLL['sampleFplot']]*dt), ((dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+dictPLL['treshold_maxT_to_plot']*1.0/(F1*dt)):dictPLL['sampleFplot'],i]-dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+dictPLL['treshold_maxT_to_plot']*1.0/(F1*dt)):dictPLL['sampleFplot'],0]+np.pi)%(2.*np.pi))-np.pi,label=labelname)			#math.fmod(phi[:,:], 2.*np.pi))
		plt.plot(np.mean(dictPLL['transmission_delay']), ((dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),0]-dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),1]+np.pi)%(2.*np.pi))-np.pi, 'yo', ms=5)
		#plt.axvspan(t[-int(5.5*1.0/(F1*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
		if dictNet['Nx']*dictNet['Ny']>=3:
			plt.title(r'phase differences $\Delta\phi_{10}=%.4f$, $\Delta\phi_{20}=%.4f$  [rad]' %( np.mod(dictData['phi'][-10][1]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi,
																		np.mod(dictData['phi'][-10][2]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi ), fontdict = titlefont)
		else:
			plt.title(r'phase differences $\Delta\phi_{10}=%.4f$ [rad]' %( np.mod(dictData['phi'][-10][1]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi ), fontdict = titlefont)
		plt.xlabel(r'$t\,[T_{\omega}]$', fontsize = axisLabel)
		plt.ylabel(r'$\phi_k(t)-\phi_0(t)$', fontsize = axisLabel)
		plt.legend();
	else:
		for i in range(len(dictData['phi'][0,:])):
			labelname = r'$\phi_{%i}$-$\phi_{0}$' %(i);
			plt.plot((dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot']]),
					((dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot'],i]-dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot'],0]+np.pi)%(2.*np.pi))-np.pi,label=labelname)			#math.fmod(phi[:,:], 2.*np.pi)) , #int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+25*1.0/(F1*dt))
		plt.plot(np.mean(dictPLL['transmission_delay']), ((dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),0]-dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),1]+np.pi)%(2.*np.pi))-np.pi, 'yo', ms=5)
		#plt.axvspan(t[-int(5.5*1.0/(F1*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
		if dictNet['Nx']*dictNet['Ny']>=3:
			plt.title(r'phase differences $\Delta\phi_{10}=%.4f$, $\Delta\phi_{20}=%.4f$  [rad]' %( np.mod(dictData['phi'][-10][1]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi,
																		np.mod(dictData['phi'][-10][2]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi ), fontdict = titlefont)
		else:
			plt.title(r'phase differences $\Delta\phi_{10}=%.4f$ [rad]' %( np.mod(dictData['phi'][-10][1]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi ), fontdict = titlefont)
		plt.xlabel(r'$t\,[T_{\omega}]$', fontsize = axisLabel)
		plt.ylabel(r'$\phi_k(t)-\phi_0(t)$', fontsize = axisLabel)
		plt.legend();
	plt.savefig('results/phaseConf-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/phaseConf-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotClockTime(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	labeldict 	= {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 	= {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$', 'phi': r'$\phi$', 't': r'$t$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color		= ['blue', 'red', 'purple', 'cyan', 'green', 'yellow'] #'magenta'
	linet		= ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig9 = plt.figure(num=9, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig9.canvas.set_window_title('clock time') 							 		# plot the clocks' time

	plt.plot(dictData['t'], dictData['clock'], linewidth=1, linestyle=linet[0])

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict = labelfont)
	plt.ylabel(r'count $\frac{T}{2}$', fontdict = labelfont)
	# plt.legend()

	plt.savefig('results/clockTime-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.pdf' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/clockTime-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotFreqAndOrderP(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotPhasesAndPhaseRelations(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotFreqAndOrderP_cutAxis(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotPhasesAndPhaseRelations_cutAxis(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	return None


#############################################################################################################################################################################



# ################################################################################################################################################################################
#
# fig110 = plt.figure()
# fig110.set_size_inches(20,10)
#
# ax011 = fig110.add_subplot(211)
#
# plt.axvspan(0, t[int(1.0*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))]*dictPLL['dt'], color='b', alpha=0.25)
# for i in range(len(phidot[0,:])):
# 	plt.plot((t[0:-1:dictPLL['sampleFplot']]*dictPLL['dt']), phidot[::dictPLL['sampleFplot'],i]/(2.0*np.pi*F1), label='PLL%i' %(i), linewidth=1)
#
# plt.ylabel(r'$f(t)/f0=\dot{\phi}(t)/\omega$',rotation=90, fontsize=60, labelpad=40)
# ax011.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.xlim([0, t[-1]*dictPLL['dt']])
# plt.grid();
#
# ax012 = fig110.add_subplot(212)
#
# plt.plot((t*dictPLL['dt'])[::dictPLL['sampleFplot']], orderparam[::dictPLL['sampleFplot']], linewidth=1.5)
# plt.xlabel(r'$t\,[T_{\omega}]$', fontsize=60, labelpad=-5)
# plt.ylabel(r'$R(t)$',rotation=90, fontsize=60, labelpad=40)
# ax012.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.xlim([0, t[-1]*dictPLL['dt']])
# plt.grid();
#
# plt.savefig('results/freq_orderP_allT_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
# plt.savefig('results/freq_orderP_allT_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=150)
# ax011.set_ylim([0, 1.05*np.max(phidot[::dictPLL['sampleFplot'],:]/(2.0*np.pi*F1))])
# ax012.set_ylim([0, 1.05])
# plt.savefig('results/freq_orderP_allT_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
# plt.savefig('results/freq_orderP_allT_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=150)
#
# ################################################################################################################################################################################
# # plot instantaneous frequency and phase-differences
# multPrior = 5.0; multAfter = 95.0;
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multPrior/(F1*dictPLL['dt']):
# 	priorStart = multPrior/(F1*dictPLL['dt'])											# start plot 'multPrior' periods before start of simulation
# else:
# 	priorStart = int(0.5*np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multAfter/(F1*dictPLL['dt']):
# 	afterStart = multPrior/(F1*dictPLL['dt'])											# end plot after 'multAfter' periods following the start of the simulation
# else:
# 	afterStart = int(0.25*Tsim/dt)
# multStartFin = 110.0; multEndFin = 15.0;
# if len(t) > multStartFin/(F1*dictPLL['dt']):
# 	multStartFin = multStartFin/(F1*dictPLL['dt'])										# start plot 'multStartFin' periods before end of simulation
# else:
# 	multStartFin = int(0.65*Tsim/dt)
# if multStartFin > multEndFin:
# 	multEndFin = multEndFin/(F1*dictPLL['dt'])											# end plot after 'multEndFin' periods before end of simulation
# else:
# 	multEndFin = int(0.95*Tsim/dt)
#
# xstart1 = int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])-priorStart);		xstart2 = len(t)-int(multStartFin);
# xend1	= int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+afterStart);		xend2   = len(t)-int(multEndFin);
# xmin1 	= t[xstart1]*dictPLL['dt']; xmax1 	= t[xend1]*dictPLL['dt'];
# xmin2 	= t[xstart2]*dictPLL['dt']; xmax2 	= t[xend2]*dictPLL['dt'];
#
# fig111 = plt.figure(figsize=(figwidth,figheight))
# fig111.set_size_inches(20,10)
#
# ax111 = fig111.add_subplot(221)
#
# ''' PLOT HERE THE FIRST PART '''
# plt.axvspan(0, t[int(1.0*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))]*dictPLL['dt'], color='b', alpha=0.25)
# for i in range(len(phidot[0,:])):
# 	plt.plot((t[xstart1:xend1:dictPLL['sampleFplot']]*dictPLL['dt']), phidot[xstart1:xend1:dictPLL['sampleFplot'],i]/(2.0*np.pi*F1), label='PLL%i' %(i), linewidth=1)
#
# plt.ylabel(r'$f(t)/f0=\dot{\phi}(t)/\omega$',rotation=90, fontsize=60, labelpad=40)
# ax111.set_xlim(xmin1, xmax1)
# ax111.set_ylim(0.99*np.min([np.min(phidot[xstart1:xend1:dictPLL['sampleFplot'],:]/(2.0*np.pi*F1)), np.min(phidot[xstart2:xend2:dictPLL['sampleFplot'],:]/(2.0*np.pi*F1))]),
# 			   1.01*np.max([np.max(phidot[xstart1:xend1:dictPLL['sampleFplot'],:]/(2.0*np.pi*F1)), np.max(phidot[xstart2:xend2:dictPLL['sampleFplot'],:]/(2.0*np.pi*F1))]) )
#
# ax111.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.grid();
#
# ax112 = fig111.add_subplot(222)
#
# ''' PLOT HERE THE SECOND PART '''
# for i in range(len(phidot[0,:])):
# 	plt.plot((t[xstart2:xend2:dictPLL['sampleFplot']]*dictPLL['dt']), phidot[xstart2:xend2:dictPLL['sampleFplot'],i]/(2.0*np.pi*F1), label='PLL%i' %(i), linewidth=1)
#
# #plt.xlabel(r'$time$',fontsize=60,labelpad=-5)
# ax112.set_xlim(xmin2, xmax2)
# ax112.set_ylim(0.99*np.min([np.min(phidot[xstart1:xend1:dictPLL['sampleFplot'],:]/(2.0*np.pi*F1)), np.min(phidot[xstart2:xend2:dictPLL['sampleFplot'],:]/(2.0*np.pi*F1))]),
# 			   1.01*np.max([np.max(phidot[xstart1:xend1:dictPLL['sampleFplot'],:]/(2.0*np.pi*F1)), np.max(phidot[xstart2:xend2:dictPLL['sampleFplot'],:]/(2.0*np.pi*F1))]) )
#
# ax112.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.grid();
# ax111.spines['right'].set_visible(False)
# ax112.spines['left'].set_visible(False)
# # ax4.yaxis.tick_left()
# ax112.tick_params(labelright='off')
# ax112.yaxis.set_ticks_position('right')
# # ax5.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
# # ax4.tick_params(labelleft='off')
# # ax5.yaxis.tick_right()
# d = .00015 # how big to make the diagonal lines in axes coordinates
#
# kwargs2 = dict(transform=ax111.transAxes)#, color='k', clip_on=False)
# ax111.plot((1-d,1+d), (-d,+d), **kwargs2)
# ax111.plot((1-d,1+d),(1-d,1+d), **kwargs2)
#
# kwargs2.update(transform=ax112.transAxes)  # switch to the bottom axes
# ax112.plot((-d,+d), (1-d,1+d), **kwargs2)
# ax112.plot((-d,+d), (-d,+d), **kwargs2)
#
# fig111.subplots_adjust(hspace=0.01)
#
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ax121 = fig111.add_subplot(223)
#
# ''' PLOT HERE THE FIRST PART '''
# plt.axvspan(0, t[int(1.0*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))]*dictPLL['dt'], color='b', alpha=0.25)
# plt.plot((t*dictPLL['dt'])[xstart1:xend1:dictPLL['sampleFplot']], orderparam[xstart1:xend1:dictPLL['sampleFplot']], linewidth=1.5)
#
# plt.ylabel(r'$R(t)$',rotation=0, fontsize=60, labelpad=40)
# ax121.set_xlim(xmin1, xmax1)
# ax121.set_ylim(0, 1.05)
#
# ax121.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.grid();
#
# ax122 = fig111.add_subplot(224)
#
# ''' PLOT HERE THE SECOND PART '''
# plt.plot((t*dictPLL['dt'])[xstart2:xend2:dictPLL['sampleFplot']], orderparam[xstart2:xend2:dictPLL['sampleFplot']], linewidth=1.5)
#
# plt.xlabel(r'$t\,[T_{\omega}]$', fontsize=60, labelpad=-5)
# ax122.set_xlim(xmin2, xmax2)
# ax122.set_ylim(0, 1.05)
#
# ax122.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.grid()
#
# ax121.spines['right'].set_visible(False)
# ax122.spines['left'].set_visible(False)
# # ax4.yaxis.tick_left()
# ax122.tick_params(labelright='off')
# ax122.yaxis.set_ticks_position('right')
# # ax5.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
# # ax4.tick_params(labelleft='off')
# # ax5.yaxis.tick_right()
# d = .00015 # how big to make the diagonal lines in axes coordinates
#
# kwargs2 = dict(transform=ax121.transAxes)#, color='k', clip_on=False)
# ax121.plot((1-d,1+d), (-d,+d), **kwargs2)
# ax121.plot((1-d,1+d),(1-d,1+d), **kwargs2)
#
# kwargs2.update(transform=ax122.transAxes)  # switch to the bottom axes
# ax122.plot((-d,+d), (1-d,1+d), **kwargs2)
# ax122.plot((-d,+d), (-d,+d), **kwargs2)
#
# plt.savefig('results/freq_orderP_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
# plt.savefig('results/freq_orderP_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=150)
#
# multPrior = 5.0; multAfter = 20.0;
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multPrior/(F1*dictPLL['dt']):
# 	priorStart = multPrior/(F1*dictPLL['dt'])											# start plot 'multPrior' periods before start of simulation
# else:
# 	priorStart = int(0.1*np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multAfter/(F1*dictPLL['dt']):
# 	afterStart = multPrior/(F1*dictPLL['dt'])											# end plot after 'multAfter' periods following the start of the simulation
# else:
# 	afterStart = int(0.1*Tsim/dt)
# multStartFin = 40.0; multEndFin = 15.0;
# if len(t) > multStartFin/(F1*dictPLL['dt']):
# 	multStartFin = multStartFin/(F1*dictPLL['dt'])										# start plot 'multStartFin' periods before end of simulation
# else:
# 	multStartFin = int(0.85*Tsim/dt)
# if multStartFin > multEndFin:
# 	multEndFin = multEndFin/(F1*dictPLL['dt'])											# end plot after 'multEndFin' periods before end of simulation
# else:
# 	multEndFin = int(0.95*Tsim/dt)
#
# xstart1 = int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])-priorStart);		xstart2 = len(t)-int(multStartFin);
# xend1	= int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+afterStart);		xend2   = len(t)-int(multEndFin);
# xmin1 	= t[xstart1]*dictPLL['dt']; xmax1 	= t[xend1]*dictPLL['dt'];
# xmin2 	= t[xstart2]*dictPLL['dt']; xmax2 	= t[xend2]*dictPLL['dt'];
#
# ax111.set_xlim(xmin1, xmax1)
# ax112.set_xlim(xmin2, xmax2)
# ax121.set_xlim(xmin1, xmax1)
# ax122.set_xlim(xmin2, xmax2)
#
# plt.savefig('results/freq_orderP_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
# plt.savefig('results/freq_orderP_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=150)
#
# ################################################################################################################################################################################
# multPrior = 5.0; multAfter = 95.0;
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multPrior/(F1*dictPLL['dt']):
# 	priorStart = multPrior/(F1*dictPLL['dt'])											# start plot 'multPrior' periods before start of simulation
# else:
# 	priorStart = int(0.5*np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multAfter/(F1*dictPLL['dt']):
# 	afterStart = multPrior/(F1*dictPLL['dt'])											# end plot after 'multAfter' periods following the start of the simulation
# else:
# 	afterStart = int(0.25*Tsim/dt)
# multStartFin = 110.0; multEndFin = 15.0;
# if len(t) > multStartFin/(F1*dictPLL['dt']):
# 	multStartFin = multStartFin/(F1*dictPLL['dt'])										# start plot 'multStartFin' periods before end of simulation
# else:
# 	multStartFin = int(0.65*Tsim/dt)
# if multStartFin > multEndFin:
# 	multEndFin = multEndFin/(F1*dictPLL['dt'])											# end plot after 'multEndFin' periods before end of simulation
# else:
# 	multEndFin = int(0.95*Tsim/dt)
#
# xstart1 = int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])-priorStart);		xstart2 = len(t)-int(multStartFin);
# xend1	= int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+afterStart);		xend2   = len(t)-int(multEndFin);
# xmin1 	= t[xstart1]*dictPLL['dt']; xmax1 	= t[xend1]*dictPLL['dt'];
# xmin2 	= t[xstart2]*dictPLL['dt']; xmax2 	= t[xend2]*dictPLL['dt'];
#
# fig211 = plt.figure(figsize=(figwidth,figheight))
# fig211.set_size_inches(20,10)
#
# ax211 = fig211.add_subplot(221)
#
# ''' PLOT HERE THE FIRST PART '''
# plt.axvspan(0, t[int(1.0*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))]*dictPLL['dt'], color='b', alpha=0.25)
# for i in range(len(phi[0,:])):
# 	plt.plot((t[xstart1:xend1:dictPLL['sampleFplot']]*dictPLL['dt']), phi[xstart1:xend1:dictPLL['sampleFplot'],i]%(2.*np.pi), label='PLL%i' %(i), linewidth=1)
#
# plt.ylabel(r'$\phi_k(t)$',rotation=0, fontsize=60, labelpad=40)
# ax211.set_xlim(xmin1, xmax1)
#
# ax211.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.grid();
#
# ax212 = fig211.add_subplot(222)
#
# ''' PLOT HERE THE SECOND PART '''
# for i in range(len(phi[0,:])):
# 	plt.plot((t[xstart2:xend2:dictPLL['sampleFplot']]*dictPLL['dt']), phi[xstart2:xend2:dictPLL['sampleFplot'],i]%(2.*np.pi), label='PLL%i' %(i), linewidth=1)
#
# plt.xlabel(r'$t\,[T_{\omega}]$', fontsize=60, labelpad=-5)
# ax212.set_xlim(xmin2, xmax2)
#
# ax212.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.grid();
# ax211.spines['right'].set_visible(False)
# ax212.spines['left'].set_visible(False)
# # ax4.yaxis.tick_left()
# ax212.tick_params(labelright='off')
# ax212.yaxis.set_ticks_position('right')
# # ax5.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
# # ax4.tick_params(labelleft='off')
# # ax5.yaxis.tick_right()
# d = .00015 # how big to make the diagonal lines in axes coordinates
#
# kwargs2 = dict(transform=ax211.transAxes)#, color='k', clip_on=False)
# ax211.plot((1-d,1+d), (-d,+d), **kwargs2)
# ax211.plot((1-d,1+d),(1-d,1+d), **kwargs2)
#
# kwargs2.update(transform=ax212.transAxes)  # switch to the bottom axes
# ax212.plot((-d,+d), (1-d,1+d), **kwargs2)
# ax212.plot((-d,+d), (-d,+d), **kwargs2)
#
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ second row plot
#
# ax221 = fig211.add_subplot(223)
#
# ''' PLOT HERE THE FIRST PART '''
# plt.axvspan(0, t[int(1.0*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))]*dictPLL['dt'], color='b', alpha=0.25)
# for i in range(len(phi[0,:])):
# 	labelname = r'$\phi_{%i}$-$\phi_{0}$' %(i);
# 	plt.plot((t[xstart1:xend1:dictPLL['sampleFplot']]*dictPLL['dt']),
# 			((phi[xstart1:xend1:dictPLL['sampleFplot'],i]-phi[xstart1:xend1:dictPLL['sampleFplot'],0]+np.pi)%(2.*np.pi))-np.pi, label=labelname, linewidth=1)
#
# plt.ylabel(r'$\phi_k(t)-\phi_0(t)$',rotation=90, fontsize=60, labelpad=40)
# ax221.set_xlim(xmin1, xmax1)
# ax221.set_ylim(-1.01*np.pi, 1.01*np.pi)
#
# ax221.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.grid();
#
# ax222 = fig211.add_subplot(224)
#
# ''' PLOT HERE THE SECOND PART '''
# for i in range(len(phi[0,:])):
# 	labelname = r'$\phi_{%i}$-$\phi_{0}$' %(i);
# 	plt.plot((t[xstart2:xend2:dictPLL['sampleFplot']]*dictPLL['dt']),
# 			((phi[xstart2:xend2:dictPLL['sampleFplot'],i]-phi[xstart2:xend2:dictPLL['sampleFplot'],0]+np.pi)%(2.*np.pi))-np.pi, label=labelname, linewidth=1)
#
# plt.xlabel(r'$t\,[T_{\omega}]$', fontsize=60, labelpad=-5)
# ax222.set_xlim(xmin2, xmax2)
# ax222.set_ylim(-1.01*np.pi, 1.01*np.pi)
#
# ax222.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.grid()
#
# ax221.spines['right'].set_visible(False)
# ax222.spines['left'].set_visible(False)
# # ax4.yaxis.tick_left()
# ax222.tick_params(labelright='off')
# ax222.yaxis.set_ticks_position('right')
# # ax5.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
# # ax4.tick_params(labelleft='off')
# # ax5.yaxis.tick_right()
# d = .00015 # how big to make the diagonal lines in axes coordinates
#
# kwargs2 = dict(transform=ax221.transAxes)#, color='k', clip_on=False)
# ax221.plot((1-d,1+d), (-d,+d), **kwargs2)
# ax221.plot((1-d,1+d),(1-d,1+d), **kwargs2)
#
# kwargs2.update(transform=ax222.transAxes)  # switch to the bottom axes
# ax222.plot((-d,+d), (1-d,1+d), **kwargs2)
# ax222.plot((-d,+d), (-d,+d), **kwargs2)
#
# plt.savefig('results/phases_phaseDiff_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
# plt.savefig('results/phases_phaseDiff_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=150)
#
# multPrior = 5.0; multAfter = 20.0;
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multPrior/(F1*dictPLL['dt']):
# 	priorStart = multPrior/(F1*dictPLL['dt'])											# start plot 'multPrior' periods before start of simulation
# else:
# 	priorStart = int(0.1*np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multAfter/(F1*dictPLL['dt']):
# 	afterStart = multPrior/(F1*dictPLL['dt'])											# end plot after 'multAfter' periods following the start of the simulation
# else:
# 	if Tsim > 2*delay:
# 		afterStart = int(2.0*np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])
# 	else:
# 		afterStart = int(0.05*Tsim/dt)
# multStartFin = 40.0; multEndFin = 15.0;
# if len(t) > multStartFin/(F1*dictPLL['dt']):
# 	multStartFin = multStartFin/(F1*dictPLL['dt'])										# start plot 'multStartFin' periods before end of simulation
# else:
# 	multStartFin = int(0.85*Tsim/dt)
# if multStartFin > multEndFin:
# 	multEndFin = multEndFin/(F1*dictPLL['dt'])											# end plot after 'multEndFin' periods before end of simulation
# else:
# 	multEndFin = int(0.95*Tsim/dt)
#
# xstart1 = int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])-priorStart);		xstart2 = len(t)-int(multStartFin);
# xend1	= int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+afterStart);		xend2   = len(t)-int(multEndFin);
# xmin1 	= t[xstart1]*dictPLL['dt']; xmax1 	= t[xend1]*dictPLL['dt'];
# xmin2 	= t[xstart2]*dictPLL['dt']; xmax2 	= t[xend2]*dictPLL['dt'];
#
# ax211.set_xlim(xmin1, xmax1)
# ax212.set_xlim(xmin2, xmax2)
# ax221.set_xlim(xmin1, xmax1)
# ax222.set_xlim(xmin2, xmax2)
#
# plt.savefig('results/phases_phaseDiff_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=300)
# plt.savefig('results/phases_phaseDiff_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=150)
#
# ################################################################################################################################################################################
