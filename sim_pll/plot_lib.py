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

from sim_pll import evaluation_lib as eva

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
		'size'   : 36,
		}

annotationfont = {
		'family' : 'monospace',
		'color'  : (0, 0.27, 0.08),
		'weight' : 'normal',
		'size'   : 14,
		}

# plot parameter
axisLabel 			= 60;
tickSize  			= 25;
titleLabel			= 10;
dpi_val	  			= 150;
figwidth  			= 10;#8;
figheight 			= 5;
plot_size_inches_x 	= 10;
plot_size_inches_y 	= 5;
labelpadxaxis       = 10;
labelpadyaxis       = 20;

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def prepareDictsForPlotting(dictPLL, dictNet):

	if dictPLL['cutFc'] is None:
		dictPLL.update({'cutFc': np.inf})

	if dictPLL['transmission_delay'] is None:
		dictPLL.update({'transmission_delay': 0})

	if not np.abs(np.min(dictPLL['intrF'])) > 1E-17: # for f=0, there would otherwise be a float division by zero
		dictPLL.update({'intrF': 1})
		print('Since intrinsic frequency was zero: for plotting set to one to generate boundaries!')

	return dictPLL, dictNet

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotPSD(dictPLL, dictNet, dictData, plotlist=[], saveData=False):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	f = []
	Pxx_db = []
	peak_power_val = []
	index_of_highest_peak = []
	value_of_highest_peak = []
	frequency_of_max_peak = []
	# compute the PSDs either of a list of oscillators or for all of them
	if plotlist:
		print('\nPlotting PSD according to given plotlist:', plotlist)
		for i in range(len(plotlist)):											# calculate spectrum of signals for the oscillators specified in the list
			ftemp, Pxx_temp = eva.calcSpectrum(dictData['phi'][:, plotlist[i]], dictPLL, dictNet, plotlist[i], dictPLL['percent_of_Tsim'])
			f.append(ftemp[0]); Pxx_db.append(Pxx_temp[0])
	else:
		plotlist = [];
		for i in range(len(dictData['phi'][0,:])):								# calculate spectrum of signals for all oscillators
			ftemp, Pxx_temp = eva.calcSpectrum(dictData['phi'][:,i], dictPLL, dictNet, i, dictPLL['percent_of_Tsim'])
			f.append(ftemp[0]); Pxx_db.append(Pxx_temp[0]); plotlist.append(i)


	fig1 = plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig1.canvas.set_window_title('spectral density of synchronized state')		# plot spectrum
	fig1.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.xlabel('frequencies [Hz]', fontdict = labelfont, labelpad=labelpadxaxis); plt.ylabel('P [dBm]', fontdict = labelfont, labelpad=labelpadyaxis);
	plt.tick_params(axis='both', which='major', labelsize=tickSize)

	for i in range(len(f)):
		index_of_highest_peak.append( np.argmax(Pxx_db[i]) )					# find the principle peak
		frequency_of_max_peak.append( f[i][index_of_highest_peak[i]] )			# save the frequency where the maximum peak is found
		peak_power_val.append( Pxx_db[i][index_of_highest_peak[i]] )			# save the peak power value

		plt.plot(f[i], Pxx_db[i], '-', label='PLL%i' %(plotlist[i]))


	plt.title(r'power spectrum $\Delta f=$%0.5E, peak at $Pxx_0^\textrm{peak}$=%0.2f' %((f[0][2]-f[0][1]), peak_power_val[0]), fontdict = labelfont)
	plt.legend(loc='upper right')
	plt.grid()

	try:
		plt.ylim([np.min(Pxx_db[0][index_of_highest_peak[0]:]), np.max(peak_power_val)+5]);
	except:
		print('Could not set ylim accordingly!')
	plt.xlim(0, 12.5*np.min(dictPLL['intrF']));
	plt.savefig('results/powerdensity_dB_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('results/powerdensity_dB_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	plt.xlim(frequency_of_max_peak[i]-1.2*np.min(dictPLL['coupK']),frequency_of_max_peak[i]+1.2*np.max(dictPLL['coupK']));
	for i in range(len(f)):
		plt.plot(f[i][index_of_highest_peak[i]-int(0.1*np.min(dictPLL['intrF'])/(f[0][2]-f[0][1]))], Pxx_db[i][index_of_highest_peak[i]], 'r*',
			 	 f[i][index_of_highest_peak[i]+int(0.1*np.min(dictPLL['intrF'])/(f[0][2]-f[0][1]))], Pxx_db[i][index_of_highest_peak[i]], 'r*')
	plt.savefig('results/powerdensity1stHarm_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('results/powerdensity1stHarm_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	freq_res_bins_both_peak_sides = 13
	try:
		minima_of_all_psd_in_zoom_range = []
		for i in range(len(f)):
			minima_of_all_psd_in_zoom_range.append(np.min(Pxx_db[i][(index_of_highest_peak[i]-freq_res_bins_both_peak_sides):(index_of_highest_peak[i]+freq_res_bins_both_peak_sides)]))
			print('minimum in plot range of PLL%i = %0.2f'%(i, minima_of_all_psd_in_zoom_range[i]))
		plt.ylim([np.min(minima_of_all_psd_in_zoom_range)-3, np.max(peak_power_val)+3]);
	except:
		print('Could not set ylim accordingly!')
	plt.xlim(frequency_of_max_peak[i]-freq_res_bins_both_peak_sides*(f[0][2]-f[0][1]),frequency_of_max_peak[i]+freq_res_bins_both_peak_sides*(f[0][2]-f[0][1]));
	plt.savefig('results/powerdensity1stHarmCloseZoom_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('results/powerdensity1stHarmCloseZoom_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	try:
		print('np.min(Pxx_db[i][:])=%02f, peak_power_val[0]=%02f'%(np.min(Pxx_db[i][5:]), peak_power_val[0]))
		plt.ylim([np.min(Pxx_db[i][5:]), peak_power_val[0]+5]);
	except:
		print('np.min(Pxx_db[i][:]), peak_power_val[0] either Inf or NAN.')
	plt.xlim(0, 8.5*np.min(dictPLL['intrF']));

	fig2 = plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')	# plot spectrum
	fig2.canvas.set_window_title('one-sided spectral density')
	fig2.set_size_inches(plot_size_inches_x, plot_size_inches_y);

	xHz = 0.001; onsidedPSD_params = []; oneSidPSDwidthsm3dB = []; quality_factors = []; # distance from the principle peak to measure damping
	Freqres = f[0][3]-f[0][2]; linestyle = ['-', '--', '-', '--', '-', '--'];
	for i in range (len(f)):
		frequency_of_max_peak[i] = 0; coup1_delt_3dB = 0;
		# mutually coupled SLL1
		index_of_highest_peak[i] = np.argmax(Pxx_db[i]);									# find the index of the principle peak (max dB) of the free-running SLL
		frequency_of_max_peak[i]  = f[i][index_of_highest_peak[i]];								# use the above index to identify the frequency of the peaks location
		coup1_times_X	= np.argmax(f[i] >= 2.25*frequency_of_max_peak[i]);				# find the index for the frequency being 2.25 times that of the peak
		m3dB_freqcind1  = index_of_highest_peak[i] + np.where(Pxx_db[i][index_of_highest_peak[i]:]<=(Pxx_db[i][index_of_highest_peak[i]]-3.0))[0][0]; # find the index associated to a power drop of -3dB w.r.t. the peak's value
		print('\n\nm3dB_freqcind1:', m3dB_freqcind1, '\n\n')
		m3dB_freqc_val  = f[i][m3dB_freqcind1];									# extract the frequency at -3dB

		coup1_delt_3dB  = np.abs( m3dB_freqc_val - frequency_of_max_peak[i] - Freqres )

		print('Calculating: f[peak]-f[-3dBm]=', frequency_of_max_peak[i],'-',m3dB_freqc_val,'=',coup1_delt_3dB, '\n power 1st harmonic', Pxx_db[i][index_of_highest_peak[i]],'\n')
		print('\nPxx_db[',i,'][',index_of_highest_peak[i],']-3.0=',Pxx_db[i][index_of_highest_peak[i]]-3.0)
		#print('TEST Pxx_dBm[',i,'][ index_of_highest_peak[i]+np.argmin(Pxx_db[',i,'][',index_of_highest_peak[i],':]<=(Pxx_db[',i,'][',index_of_highest_peak[i],']-3)) ]: ',
		#					Pxx_db[i][ index_of_highest_peak[i]+np.argmin(Pxx_db[i][index_of_highest_peak[i]:].copy()<=(Pxx_db[i][index_of_highest_peak[i]]-3.0)) ],
		#					' -> frequency where PxxMax-3dB:', m3dB_freqc_val,'Hz')
		print('np.where(Pxx_db[',i,'][',index_of_highest_peak[i],':]<=(Pxx_db[',i,'][',index_of_highest_peak[i],']-3))[0][0]=', np.where(Pxx_db[i][index_of_highest_peak[i]:]<=(Pxx_db[i][index_of_highest_peak[i]]-3.0))[0][0], '\n')
		# calculate linear function between first two points of one-sided powerspectrum: y = slope * x + yinter, interPol{1,2} are the relate to the y-coordinates between which Pxx-3dB lies
		# interpolP1 = index_of_highest_peak[i] + np.where(Pxx_db[i][index_of_highest_peak[i]:] > (Pxx_db[i][index_of_highest_peak[i]]-3.0))[0][-1]	# find the results in the PSD vectors that are adjacent to the -3dB point
		# interpolP2 = index_of_highest_peak[i] + np.where(Pxx_db[i][index_of_highest_peak[i]:] < (Pxx_db[i][index_of_highest_peak[i]]-3.0))[0][0]		# PROBLEM IF Pxx goes up again!!!!!!! sketch to see
		interpolP1 = index_of_highest_peak[i] + np.where(Pxx_db[i][index_of_highest_peak[i]:] < (Pxx_db[i][index_of_highest_peak[i]]-3.0))[0][0]	# find the first point in the PSD smaller than PxxMax-3dB and then take the prior point
		interpolP2 = interpolP1-1																						# as the second to interpolate
		#print('{interpolP1, interpolP2}:', interpolP1, interpolP2)
		slope  = ( Pxx_db[i][interpolP2]-Pxx_db[i][interpolP1]) / ( f[i][interpolP2]-f[i][interpolP1])
		yinter = Pxx_db[i][interpolP1] - slope * f[i][interpolP1]
		#print('{slope, yinter}:', slope, yinter)
		# slope  = ( Pxx_db[i][index_of_highest_peak[i]+1]-Pxx_db[i][index_of_highest_peak[i]]) / ( f[i][index_of_highest_peak[i]+1]-f[i][index_of_highest_peak[i]])
		# yinter = Pxx_db[i][index_of_highest_peak[i]] - slope * f[i][index_of_highest_peak[i]]
		fm3dB  = ( Pxx_db[i][index_of_highest_peak[i]]-3.0 - yinter ) / slope
		oneSidPSDwidthsm3dB.append(fm3dB-f[i][index_of_highest_peak[i]])
		quality_factors.append(frequency_of_max_peak[i]/(2*(fm3dB-f[i][index_of_highest_peak[i]])))
		print('For the mutually coupled SLL',i,' we find the principle peak at f =', frequency_of_max_peak[i], ', -3dBm delta_f =', coup1_delt_3dB, ', and hence a quality factor Q = ', frequency_of_max_peak[i]/(2*(fm3dB-f[i][index_of_highest_peak[i]])))
		plt.plot(10.0*np.log10(fm3dB-frequency_of_max_peak[i]+Freqres), Pxx_db[i][index_of_highest_peak[i]]-3.0, 'r*', markersize=2)
		if coup1_delt_3dB == 0:
			print('frequency resolution of power spectrum too large or power spectrum approaching delta-like peak!')
		try:
			onsidedPSD_params.append([frequency_of_max_peak[i], coup1_delt_3dB])
		except:
			onsidedPSD_params.append([0, 0])
		if (m3dB_freqcind1 < 3.1 and m3dB_freqcind1 > 2.9):
			#plt.plot(10.0*np.log10(1E-12), Pxx_db[i][index_of_highest_peak[i]], 'r*', markersize=2)
			plt.plot(10.0*np.log10(f[i][m3dB_freqcind1]-frequency_of_max_peak[i]), Pxx_db[i][m3dB_freqcind1], 'r+', markersize=2)
		else:
			print('CHECK frequency resolution of power spectrum and noise strength. Cannot use this method.')
		if dictNet['topology'] == 'compareEntrVsMutual':
			plt.plot(10.0*np.log10(f[i][index_of_highest_peak[i]:coup1_times_X]-frequency_of_max_peak[i]+Freqres), Pxx_db[i][index_of_highest_peak[i]:coup1_times_X], linestyle[i], label='PSD PLL%i' %(i), markersize=2)
		else:
			plt.plot(10.0*np.log10(f[i][index_of_highest_peak[i]:coup1_times_X]-frequency_of_max_peak[i]+Freqres), Pxx_db[i][index_of_highest_peak[i]:coup1_times_X], label='PSD PLL%i' %(plotlist[i]), markersize=2)
	try:
		plt.title(r'$\gamma_0^{(\textrm{PSDfit})}=$%0.4E, $\gamma_1=$%0.4E, $\gamma_2=$%0.4E' %(params[0][0], params[1][0], params[2][0]), fontdict = titlefont)
	except:
		print('No (two-sided, 1st harmonic) PSD fits available!')
	if dictNet['Nx']*dictNet['Ny'] == 2:
		onsidedPSD_params.append([0, 0])										# necessary, otherwise error on write-out to csv file
	#plt.plot(10.0*np.log10(powerspecPLL1['f'][0][index_of_highest_peak[i]:coup1_times_X].copy()-frequency_of_max_peak[i]), !!!!! , 'y-', label=r'$1/f^2$')
	plt.legend(loc='upper right')
	# plt.xlim([0,f01+20*max(Kvco1,Kvco2)]);	#plt.ylim(-100,0);
	plt.xlabel(r'$10\log_{10}\left(f-f_{\rm peak}\right)$ [Hz]', fontdict = labelfont, labelpad=labelpadxaxis); plt.ylabel(r'$P$ [dBm]', fontdict = labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=35); plt.grid();
	plt.savefig('results/onsidedPSD_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('results/onsidedPSD_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	print('\nwidths (HWHM) of PSDs obtained from one-sided PSD with interpolation for all oscis:', oneSidPSDwidthsm3dB)
	print('Mean of widths of PSDs obtained from one-sided PSD with interpolation for all oscis:', np.mean(oneSidPSDwidthsm3dB))
	print('Std of widths of PSDs obtained from one-sided PSD with interpolation for all oscis:', np.std(oneSidPSDwidthsm3dB), '\n')
	print('all quality factors obtained from the one-sided PSD:', quality_factors)
	print('Mean all quality factors obtained from the one-sided PSD:', np.mean(quality_factors))
	print('Std all quality factors obtained from the one-sided PSD:', np.std(quality_factors), '\n')

	fig1 = plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	for i in range (len(f)):
		plt.plot(oneSidPSDwidthsm3dB[i]+f[i][index_of_highest_peak[i]], Pxx_db[i][index_of_highest_peak[i]]-3.0, 'r*', markersize=2)

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
	fig3.set_size_inches(plot_size_inches_x, plot_size_inches_y)


	plt.plot(dictData['t'], dictData['phi']%(2*np.pi), linewidth=1, linestyle=linet[0])
	plt.plot(dictData['t'][dictNet['max_delay_steps']-1], dictData['phi'][int(dictNet['max_delay_steps'])-1,0]%(2*np.pi)+0.05,'go')

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict = labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$\theta(t)_{\textrm{mod}\,2\pi}$', fontdict = labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	# plt.legend(loc='upper right')

	plt.savefig('results/phases2pi-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('results/phases2pi-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

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
	fig4.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.plot(dictData['t'], dictData['phi'], linewidth=1, linestyle=linet[0])
	plt.plot(dictData['t'][dictNet['max_delay_steps']-1], dictData['phi'][int(dictNet['max_delay_steps'])-1,0]+0.05,'go')

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict = labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$\theta(t)$', fontdict = labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	# plt.legend(loc='upper right')

	plt.savefig('results/phasesInf-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('results/phasesInf-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

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
	fig5.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	phidot = np.diff(dictData['phi'], axis=0)/dictPLL['dt'];
	plt.plot(dictData['t'][0:-1], phidot, linewidth=1, linestyle=linet[0])
	plt.plot(dictData['t'][dictNet['max_delay_steps']-1], phidot[int(dictNet['max_delay_steps'])-1,0]+0.001,'go')

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict = labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$\theta(t)$', fontdict = labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	# plt.legend(loc='upper right')

	plt.xlabel(r'$t\,[T_{\omega}]$', fontdict = labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$\dot{\phi}(t)$ [rad Hz]', fontdict = labelfont, labelpad=labelpadyaxis)
	plt.savefig('results/freq-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('results/freq-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.xlim([np.mean(dictPLL['transmission_delay']) - 25*1.0/(np.min(dictPLL['intrF'])), np.mean(dictPLL['transmission_delay']) + 35*1.0/(np.min(dictPLL['intrF']))]);
	plt.ylim([0.99*np.min(phidot[0:int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+25*1.0/(np.min(dictPLL['intrF'])*dictPLL['dt']))-1,:]), 1.01*np.max(phidot[0:int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+25*1.0/(np.min(dictPLL['intrF'])*dictPLL['dt']))-1,:])]);
	plt.savefig('results/freqInit-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('results/freqInit-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.xlim([np.mean(dictPLL['transmission_delay']) - 25*1.0/(np.min(dictPLL['intrF'])), np.mean(dictPLL['transmission_delay']) + 400*1.0/(np.min(dictPLL['intrF']))]);
	plt.ylim([np.min(phidot[int(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] - 25*1.0/(dictPLL['sampleF']/np.min(dictPLL['intrF']))):int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+400*1.0/(np.min(dictPLL['intrF'])*dictPLL['dt']))-1,:])-0.05, np.max(phidot[int(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] - 25*1.0/(dictPLL['sampleF']/np.min(dictPLL['intrF']))):int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+400*1.0/(np.min(dictPLL['intrF'])*dictPLL['dt']))-1,:])+0.05]);
	plt.savefig('results/freqInit1-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('results/freqInit1-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.xlim([0, dictData['t'][-1]]);
	plt.ylim([np.min(phidot[int(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']+0.5*dictPLL['sampleF']/np.min(dictPLL['intrF'])):,:])-0.05, np.max(phidot[int(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']+0.5*dictPLL['sampleF']/np.min(dictPLL['intrF'])):,:])+0.05]);

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotOrderPara(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	fig6 = plt.figure(num=6, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig6.canvas.set_window_title('order parameter over time')					# plot the order parameter in dependence of time
	fig6.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.plot(dictData['t'], dictData['orderParam'])
	plt.plot(np.mean(dictPLL['transmission_delay']), dictData['orderParam'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))], 'yo', ms=5)				# mark where the simulation starts
	if -int(dictPLL['timeSeriesAverTime']*1.0/(dictData['F1']*dictPLL['dt'])) >= 0:
		plt.axvspan(dictData['t'][-int(dictPLL['timeSeriesAverTime']*1.0/(dictData['F1']*dictPLL['dt']))], dictData['t'][-1], color='b', alpha=0.3)
	plt.title(r'mean order parameter $\bar{R}=$%.2f, and $\bar{\sigma}=$%.4f' %(np.mean(dictData['orderParam'][-int(round(dictPLL['timeSeriesAverTime']*1.0/(dictData['F1']*dictPLL['dt']))):]), np.std(dictData['orderParam'][-int(round(dictPLL['timeSeriesAverTime']*1.0/(dictData['F1']*dictPLL['dt']))):])), fontdict = titlefont)
	plt.xlabel(r'$t\,[T_{\omega}]$', fontdict = labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$R(t,m_x=%d,m_y=%d )$' %(dictNet['mx'],dictNet['my']), fontdict = labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)

	plt.savefig('results/orderP-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/orderP-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	try:
		plt.xlim([dictData['t'][-int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']+75*1.0/(dictData['F1']*dictPLL['dt'])))], dictData['t'][-1]]); # plt.ylim([]);
	except:
		plt.xlim([0, 15])
	plt.savefig('results/orderPFin-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/orderPFin-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.xlim([0, np.mean(dictPLL['transmission_delay'])+125*1.0/(dictData['F1'])]); # plt.ylim([]);
	plt.savefig('results/orderPInit-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/orderPInit-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	#print('\nlast entry order parameter: R-1 = %.3e' % (dictData['orderParam'][-1]-1) )
	#print('\nlast entries order parameter: R = ', dictData['orderParam'][-25:])

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotPhaseRela(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	fig7 = plt.figure(num=7, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig7.canvas.set_window_title('phase relations')
	fig7.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	if not dictNet['topology'] == 'compareEntrVsMutual':
		plt.plot(dictData['t'][::dictPLL['sampleFplot']],((dictData['phi'][::dictPLL['sampleFplot'],0]-dictData['phi'][::dictPLL['sampleFplot'],1]+np.pi)%(2.*np.pi))-np.pi,label=r'$\phi_{0}-\phi_{1}$')			#math.fmod(phi[:,:], 2.*np.pi))
		if not dictNet['Nx']*dictNet['Ny'] == 2:
			plt.plot(dictData['t'][::dictPLL['sampleFplot']],((dictData['phi'][::dictPLL['sampleFplot'],1]-dictData['phi'][::dictPLL['sampleFplot'],2]+np.pi)%(2.*np.pi))-np.pi,label=r'$\phi_{1}-\phi_{2}$')
			plt.plot(dictData['t'][::dictPLL['sampleFplot']],((dictData['phi'][::dictPLL['sampleFplot'],0]-dictData['phi'][::dictPLL['sampleFplot'],2]+np.pi)%(2.*np.pi))-np.pi,label=r'$\phi_{0}-\phi_{2}$')
		plt.plot(np.mean(dictPLL['transmission_delay']), ((dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),0]-dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),1]+np.pi)%(2.*np.pi))-np.pi, 'yo', ms=5)
		#plt.axvspan(t[-int(5.5*1.0/(dictData['F1']*dictPLL['dt']))]*dictPLL['dt'], t[-1]*dictPLL['dt'], color='b', alpha=0.3)
		if dictNet['Nx']*dictNet['Ny']>=3:
			plt.title(r'phases $\phi_{0}=%.4f$, $\phi_{1}=%.4f$, $\phi_{R}=%.4f$  [rad]' %( (-1)*(np.mod(dictData['phi'][-10][2]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi),
																np.mod(dictData['phi'][-10][1]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi - (np.mod(dictData['phi'][-10][2]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi), 0 ), fontdict = titlefont)
		else:
			plt.title(r'phases [rad]', fontdict = titlefont)
		plt.xlabel(r'$t\,[T_{\omega}]$', fontdict = labelfont, labelpad=labelpadxaxis)
		plt.ylabel(r'$\phi_k(t)-\phi_0(t)$', fontdict = labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)

		plt.legend(loc='upper right');
		plt.savefig('results/phaseRela-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
		plt.savefig('results/phaseRela-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	else:
		plt.plot(dictData['t'][-int(25*1.0/(dictData['F1']*dictPLL['dt'])):],((dictData['phi'][-int(25*1.0/(dictData['F1']*dictPLL['dt'])):,0]-dictData['phi'][-int(25*1.0/(dictData['F1']*dictPLL['dt'])):,1]+np.pi)%(2.*np.pi))-np.pi,'-',label=r'$\phi_{0}-\phi_{1}$ mutual')
		plt.plot(dictData['t'][-int(25*1.0/(dictData['F1']*dictPLL['dt'])):],((dictData['phi'][-int(25*1.0/(dictData['F1']*dictPLL['dt'])):,3]-dictData['phi'][-int(25*1.0/(dictData['F1']*dictPLL['dt'])):,2]+np.pi)%(2.*np.pi))-np.pi,'--',label=r'$\phi_{3}-\phi_{2}$ entrain')
		#plt.plot((t[-int(12*1.0/(dictData['F1']*dictPLL['dt'])):]*dictPLL['dt']),((dictData['phi'][-int(12*1.0/(dictData['F1']*dictPLL['dt'])):,0]-dictData['phi'][-int(12*1.0/(dictData['F1']*dictPLL['dt'])):,5]+np.pi)%(2.*np.pi))-np.pi,'-',label=r'$\phi_{0}-\phi_{5}$ mutual  vs freeRef')
		#plt.plot((t[-int(12*1.0/(dictData['F1']*dictPLL['dt'])):]*dictPLL['dt']),((dictData['phi'][-int(12*1.0/(dictData['F1']*dictPLL['dt'])):,3]-dictData['phi'][-int(12*1.0/(dictData['F1']*dictPLL['dt'])):,5]+np.pi)%(2.*np.pi))-np.pi,'--',label=r'$\phi_{3}-\phi_{5}$ entrain vs freeRef')
		#plt.plot((t[-int(12*1.0/(dictData['F1']*dictPLL['dt'])):]*dictPLL['dt']),((dictData['phi'][-int(12*1.0/(dictData['F1']*dictPLL['dt'])):,0]-dictData['phi'][-int(12*1.0/(dictData['F1']*dictPLL['dt'])):,4]+np.pi)%(2.*np.pi))-np.pi,'-.',label=r'$\phi_{0}-\phi_{4}$ mutual  vs freePLL')
		#plt.plot((t[-int(12*1.0/(dictData['F1']*dictPLL['dt'])):]*dictPLL['dt']),((dictData['phi'][-int(12*1.0/(dictData['F1']*dictPLL['dt'])):,3]-dictData['phi'][-int(12*1.0/(dictData['F1']*dictPLL['dt'])):,4]+np.pi)%(2.*np.pi))-np.pi,'-.',label=r'$\phi_{3}-\phi_{4}$ entrain vs freePLL')
		#plt.plot(np.mean(dictPLL['transmission_delay']), ((dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),0]-dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),1]+np.pi)%(2.*np.pi))-np.pi, 'yo', ms=5)
		plt.axvspan(dictData['t'][-int(5.5*1.0/(dictData['F1']*dictPLL['dt']))], dictData['t'][-1], color='b', alpha=0.3)
		plt.title(r'phases-differences between the clocks', fontdict = titlefont)
		plt.xlabel(r'$t\,[T_{\omega}]$', fontdict = labelfont, labelpad=labelpadxaxis)
		plt.ylabel(r'$\phi_k(t)-\phi_0(t)$', fontdict = labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)

		plt.legend(loc='upper right');
		plt.savefig('results/phaseRela-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
		plt.savefig('results/phaseRela-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)


	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotPhaseDiff(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	fig8 = plt.figure(num=8, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig8.canvas.set_window_title('phase configuration with respect to the phase of osci 0')
	fig8.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	if len(dictData['phi'][:,0]) > dictPLL['treshold_maxT_to_plot'] * (1.0/(np.min(dictPLL['intrF'])*dictPLL['dt'])):	# (1.0/(np.min(dictPLL['intrF'])*dictPLL['dt'])) steps for one period
		for i in range(len(dictData['phi'][0,:])):
			labelname = r'$\phi_{%i}$-$\phi_{0}$' %(i);
			plt.plot((dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+dictPLL['treshold_maxT_to_plot']*1.0/(dictData['F1']*dictPLL['dt'])):dictPLL['sampleFplot']]*dictPLL['dt']), ((dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+dictPLL['treshold_maxT_to_plot']*1.0/(dictData['F1']*dictPLL['dt'])):dictPLL['sampleFplot'],i]-dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+dictPLL['treshold_maxT_to_plot']*1.0/(dictData['F1']*dictPLL['dt'])):dictPLL['sampleFplot'],0]+np.pi)%(2.*np.pi))-np.pi,label=labelname)			#math.fmod(phi[:,:], 2.*np.pi))
		plt.plot(np.mean(dictPLL['transmission_delay']), ((dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),0]-dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),1]+np.pi)%(2.*np.pi))-np.pi, 'yo', ms=5)
		#plt.axvspan(t[-int(5.5*1.0/(dictData['F1']*dictPLL['dt']))]*dictPLL['dt'], t[-1]*dictPLL['dt'], color='b', alpha=0.3)
		if dictNet['Nx']*dictNet['Ny']>=3:
			plt.title(r'phase differences $\Delta\phi_{10}=%.4f$, $\Delta\phi_{20}=%.4f$  [rad]' %( np.mod(dictData['phi'][-10][1]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi,
																		np.mod(dictData['phi'][-10][2]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi ), fontdict = titlefont)
		else:
			plt.title(r'phase differences $\Delta\phi_{10}=%.4f$ [rad]' %( np.mod(dictData['phi'][-10][1]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi ), fontdict = titlefont)
		plt.xlabel(r'$t\,[T_{\omega}]$', fontdict = labelfont, labelpad=labelpadxaxis)
		plt.ylabel(r'$\phi_k(t)-\phi_0(t)$', fontdict = labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)
		plt.legend(loc='upper right');
	else:
		shift2piWin = -np.pi/2													# this controls how the [0, 2pi) interval is shifted
		for i in range(len(dictData['phi'][0,:])):
			labelname = r'$\phi_{%i}$-$\phi_{0}$' %(i);
			plt.plot((dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot']]),
					((dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot'],i]-dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot'],0]-shift2piWin)%(2*np.pi))+shift2piWin,label=labelname)			#math.fmod(phi[:,:], 2.*np.pi)) , #int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+25*1.0/(dictData['F1']*dictPLL['dt']))
		plt.plot(np.mean(dictPLL['transmission_delay']), ((dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),0]-dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),1]-shift2piWin)%(2*np.pi))+shift2piWin, 'yo', ms=5)
		#plt.axvspan(t[-int(5.5*1.0/(dictData['F1']*dictPLL['dt']))]*dictPLL['dt'], t[-1]*dictPLL['dt'], color='b', alpha=0.3)
		if dictNet['Nx']*dictNet['Ny']>=3:
			plt.title(r'phase differences $\Delta\phi_{10}=%.4f$, $\Delta\phi_{20}=%.4f$  [rad]' %( np.mod(dictData['phi'][-10][1]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi,
																		np.mod(dictData['phi'][-10][2]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi ), fontdict = titlefont)
		else:
			plt.title(r'phase differences $\Delta\phi_{10}=%.4f$ [rad]' %( np.mod(dictData['phi'][-10][1]-dictData['phi'][-10][0]+np.pi, 2.0*np.pi)-np.pi ), fontdict = titlefont)
		plt.xlabel(r'$t\,[T_{\omega}]$', fontdict = labelfont, labelpad=labelpadxaxis)
		plt.ylabel(r'$\phi_k(t)-\phi_0(t)$', fontdict = labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)
		plt.legend(loc='upper right');
	plt.savefig('results/phaseConf-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/phaseConf-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

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
	fig9.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.plot(dictData['t'], dictData['clock_counter'], linewidth=1, linestyle=linet[0])

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict = labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'count $\frac{T}{2}$', fontdict = labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	# plt.legend(loc='upper right')

	plt.savefig('results/clockTime-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/clockTime-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotCtrlSigDny(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	color		= ['blue', 'red', 'purple', 'cyan', 'green', 'yellow'] #'magenta'
	linet		= ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig10 = plt.figure(num=10, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig10.canvas.set_window_title('time-series control voltage')		 		# plot the time evolution of the control signal
	fig10.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.plot(dictData['t'], dictData['ctrl'], linewidth=1, linestyle=linet[0])

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict = labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$V_\textrm{ctrl}(t)$', fontdict = labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	# plt.legend(loc='upper right')

	plt.savefig('results/ctrlSig-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/ctrlSig-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotOscSignal(dictPLL, dictNet, dictData, plotEveryDt=1):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	color		= ['blue', 'red', 'purple', 'cyan', 'green', 'yellow'] #'magenta'
	linet		= ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig11 = plt.figure(num=11, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig11.canvas.set_window_title('time-series signals')		 		   # plot the time evolution of the control signal
	fig11.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	ax11 = fig11.add_subplot(211)

	for i in range(len(dictData['phi'][0,:])):
		plt.plot((dictData['t'][::plotEveryDt]), dictPLL['vco_out_sig'](dictData['phi'][::plotEveryDt,i]), label='sig PLL%i' %(i))
		plt.ylabel(r'$s( \theta(t) )$', fontdict = labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)

	ax12 = fig11.add_subplot(212)

	for i in range(len(dictData['phi'][0,:])):
		plt.plot((dictData['t'][::plotEveryDt]), dictPLL['vco_out_sig'](dictData['phi'][::plotEveryDt,i]/dictPLL['div']), label='sig PLL%i' %(i))
		plt.ylabel(r'$s( \theta(t)/ %i )$'%dictPLL['div'], fontdict = labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)


	plt.savefig('results/sig_and_divSig-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/sig_and_divSig-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_instfreq_vs_timedependent_parameter(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	param_name = dictNet['special_case']		#'timeDepInjectLockCoupStr', 'timeDepTransmissionDelay', 'timeDepChangeOfCoupStr', 'distanceDepTransmissionDelay'
	if param_name == 'timeDepTransmissionDelay':
		dyn_x_label = r'$\frac{\tau\omega}{2\pi}$'
		x_axis_scaling = np.mean(dictPLL['intrF'])
	elif param_name == 'timeDepChangeOfCoupStr':
		dyn_x_label = r'$\frac{2\pi K}{\omega}$'
		x_axis_scaling = np.mean(1.0/dictPLL['intrF'])
	elif param_name == 'timeDepInjectLockCoupStr':
		dyn_x_label = r'$\frac{2\pi K}{\omega}$'
		x_axis_scaling = np.mean(1.0 / dictPLL['intrF'])

	y_axis_scaling = (2.0*np.pi*np.mean(dictPLL['intrF']))

	fig12 = plt.figure(num=12, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig12.canvas.set_window_title('instantaneous frequency as function of time-dependent parameter')
	fig12.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.plot(dictData['timeDependentParameter'][0,0:len(dictData['phi'][:,0])-1]*x_axis_scaling, (np.diff(dictData['phi'], axis=0)/dictPLL['dt'])/y_axis_scaling, 'b-')

	plt.xlabel(dyn_x_label, fontdict = labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$\frac{\dot{\theta}_k(t)}{\omega}$', fontdict = labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	# plt.legend(loc='upper right')

	plt.savefig('results/instFreq_vs_'+param_name+'%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/instFreq_vs_'+param_name+'%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	if dictPLL['div'] != 1:
		fig1212 = plt.figure(num=1212, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
		fig1212.canvas.set_window_title('instantaneous frequency as function of time-dependent parameter')
		fig1212.set_size_inches(plot_size_inches_x, plot_size_inches_y)

		plt.plot(dictData['timeDependentParameter'][0,0:len(dictData['phi'][:,0])-1]*x_axis_scaling, (np.diff(dictData['phi']/dictPLL['div'], axis=0)/dictPLL['dt'])/y_axis_scaling, 'b-')

		plt.xlabel(dyn_x_label, fontdict = labelfont, labelpad=labelpadxaxis)
		plt.ylabel(r'$\frac{\dot{\theta}^\textrm{HF}_k(t)}{v\,\omega}$', fontdict = labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)
		# plt.legend(loc='upper right')

		plt.savefig('results/instDivFreq_vs_'+param_name+'%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
		plt.savefig('results/instDivFreq_vs_'+param_name+'%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotPhasesAndPhaseRelations(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	fig13 = plt.figure(num=13, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig13.canvas.set_window_title('time-series phases and phase-differences')	# time-series phases and phase-differences
	fig13.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotFreqAndOrderP_cutAxis(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	fig14 = plt.figure(num=14, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig14.canvas.set_window_title('time-series frequency and order parameter')	# frequency and order parameter
	fig14.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotPhasesAndPhaseRelations_cutAxis(dictPLL, dictNet, dictData):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	fig15 = plt.figure(num=15, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig15.canvas.set_window_title('time-series phases and phase-differences')	# time-series phases and phase-differences
	fig15.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	return None

#############################################################################################################################################################################

def deltaThetaDot_vs_deltaTheta(dictPLL, dictNet, deltaTheta, deltaThetaDot, color, alpha):

	#dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)
	fig16 = plt.figure(num=16, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig16.canvas.set_window_title('time-series phases and phase-differences')	# time-series phases and phase-differences


	plt.plot(deltaTheta, deltaThetaDot, '-', color=color, alpha=alpha)			# plot trajectory
	plt.plot(deltaTheta[0], deltaThetaDot[0], 'o', color=color, alpha=alpha)	# plot initial dot
	plt.plot(deltaTheta[-1], deltaThetaDot[-1], 'x', color=color, alpha=alpha)	# plot final state cross

	fig17 = plt.figure(num=17, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig17.canvas.set_window_title('time-series phases and phase-differences')	# time-series phases and phase-differences
	fig17.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.plot(deltaTheta[0], deltaThetaDot[0], 'o', color=color, alpha=alpha)	# plot initial dot

	return None

#############################################################################################################################################################################

def plotOrderPvsTimeDepPara(dictPLL, dictNet, dictData, dictAlgo):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	fig18 = plt.figure(num=18, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig18.canvas.set_window_title('order parameter as function of time-dependent parameter')	# time-series phases and phase-differences
	fig18.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.plot(dictData['timeDependentParameter'][0], dictData['orderParam'], 'b-')

	plt.xlabel(r'$K$', fontdict = labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$R(t)$', fontdict = labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	# plt.legend(loc='upper right')

	plt.savefig('results/orderPvstimeDepPara_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day))
	plt.savefig('results/orderPvstimeDepPara_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotFreqAndPhaseDiff(dictPLL, dictNet, dictData, plotlist=[], ylim_percent_of_min_val=0.995, ylim_percent_of_max_val=1.005):

	phase_diff_zero_2pi = 2					# set to 1 if phase differences to be plotted in [0, 2pi), to 0 if plotting in [-pi, +pi) and to 2 if plotting in [-pi/2, 3pi/2]

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	labeldict = {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$',	'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 = {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$', 'phi': r'$\phi$', 't': r'$t$', 'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color = ['blue', 'red', 'purple', 'cyan', 'green', 'yellow'] #'magenta'
	linet = ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	fig19 = plt.figure(num=19, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig19.canvas.set_window_title('time-series frequency and phase difference')	# frequency and order parameter
	fig19.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	ax011 = fig19.add_subplot(211)

	plt.axvspan(dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))], dictData['t'][int(1.0*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))], color='b', alpha=0.25)
	phidot = np.diff(dictData['phi'], axis=0)/dictPLL['dt']
	if not dictPLL['intrF'] == 0:
		if isinstance(dictPLL['intrF'], list) or isinstance(dictPLL['intrF'], np.ndarray):
			phidot = phidot / (2.0*np.pi*np.mean(dictPLL['intrF']))
			ylabelname = r'$\frac{\dot{\theta}_k(t)}{\bar{\omega}_k}$'
		else:
			phidot = phidot / (2.0*np.pi*dictPLL['intrF'])
			ylabelname = r'$\frac{\dot{\theta}_k(t)}{\omega}$'
	else:
		ylabelname = r'$\dot{\theta}_k(t)$'
	if not plotlist:
		plt.plot(dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot']],
				phidot[int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot']], linewidth=2, linestyle=linet[0])
	else:
		plt.plot(dictData['t'][int(0.75 * np.round(np.mean(dictPLL['transmission_delay']) / dictPLL['dt'])):-1:dictPLL['sampleFplot']],
				phidot[int(0.75 * np.round(np.mean(dictPLL['transmission_delay']) / dictPLL['dt']))::dictPLL['sampleFplot'], plotlist], linewidth=2, linestyle=linet[0])
	# plt.plot(dictData['t'][dictNet['max_delay_steps']-1], phidot[int(dictNet['max_delay_steps'])-1,0]+0.001,'go')

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict = labelfont, labelpad=labelpadxaxis)
	plt.ylabel(ylabelname, fontdict=labelfont, labelpad=labelpadyaxis)
	ax011.tick_params(axis='both', which='major', labelsize=tickSize, pad=1)
	ax011.set_xlim([dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))], dictData['t'][-1]])

	mean_freq_ts = np.mean(phidot[int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot']])
	max_freq_ts  = np.max(phidot[int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot']])
	min_freq_ts  = np.min(phidot[int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot']])

	ax011.set_ylim([ylim_percent_of_min_val*min_freq_ts, ylim_percent_of_max_val*max_freq_ts])
	plt.grid();

	ax012 = fig19.add_subplot(212)

	if phase_diff_zero_2pi == 0:												# plot phase-differences in [-pi, pi] interval
		shift2piWin = np.pi
	elif phase_diff_zero_2pi == 1:												# plot phase-differences in [0, 2*pi] interval
		shift2piWin = 0.0
	elif phase_diff_zero_2pi == 2:												# plot phase-differences in [-pi/2, 3*pi/2] interval
		shift2piWin = 0.5*np.pi

	if not dictNet['topology'] == 'compareEntrVsMutual':
		if not dictNet['Nx']*dictNet['Ny'] == 2:
			if not plotlist:
				for i in range(len(dictData['phi'][0,:])):
					labelname = r'$\phi_{%i}$-$\phi_{0}$' %(i)
					plt.plot((dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot']]),
							((dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot'], i]-dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot'], 0]+shift2piWin)%(2*np.pi))-shift2piWin,label=labelname)
			else:
				for i in plotlist:
					labelname = r'$\phi_{%i}$-$\phi_{0}$' %(i)
					plt.plot((dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot']]),
							((dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot'], i]-dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot'], 0]+shift2piWin)%(2*np.pi))-shift2piWin,label=labelname)
		else:
			labelname = r'$\phi_{1}$-$\phi_{0}$'
			plt.plot((dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot']]),
					((dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot'], 1]-dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot'], 0]+shift2piWin)%(2*np.pi))-shift2piWin,label=labelname)
	else:
		plt.plot(dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot']],
		((dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot'], 0]-dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot'], 1]+shift2piWin)%(2.*np.pi))-shift2piWin,'-',linewidth=2,label=r'$\phi_{0}-\phi_{1}$ mutual')
		plt.plot(dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot']],
		((dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot'], 3]-dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot'], 2]+shift2piWin)%(2.*np.pi))-shift2piWin,'--',linewidth=2,label=r'$\phi_{3}-\phi_{2}$ entrain')
		# plt.plot((t[int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot']]*dictPLL['dt']),((dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot'],0]-dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot'],5]+np.pi)%(2.*np.pi))-np.pi,'-',linewidth=2,label=r'$\phi_{0}-\phi_{5}$ mutual  vs freeRef')
		# plt.plot((t[int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot']]*dictPLL['dt']),((dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot'],3]-dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot'],5]+np.pi)%(2.*np.pi))-np.pi,'--',linewidth=2,label=r'$\phi_{3}-\phi_{5}$ entrain vs freeRef')
		# plt.plot((t[int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot']]*dictPLL['dt']),((dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot'],0]-dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot'],4]+np.pi)%(2.*np.pi))-np.pi,'-.',linewidth=2,label=r'$\phi_{0}-\phi_{4}$ mutual  vs freePLL')
		# plt.plot((t[int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot']]*dictPLL['dt']),((dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot'],3]-dictData['phi'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot'],4]+np.pi)%(2.*np.pi))-np.pi,'-.',linewidth=2,label=r'$\phi_{3}-\phi_{4}$ entrain vs freePLL')
		# plt.plot(np.mean(dictPLL['transmission_delay']), ((dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),0]-dictData['phi'][int(round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])),1]+np.pi)%(2.*np.pi))-np.pi, 'yo', ms=5)
		# plt.axvspan(dictData['t'][-int(5.5*1.0/(dictPLL['intrF']*dictPLL['dt']))], dictData['t'][-1], color='b', alpha=0.3)

	plt.xlabel(r'$\omega t/2\pi$', fontdict=labelfont, labelpad=-5)
	plt.ylabel(r'$\Delta\theta_{k0}(t)$', rotation=90, fontdict=labelfont, labelpad=40)
	ax012.tick_params(axis='both', which='major', labelsize=tickSize, pad=1)
	ax012.set_xlim([dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))], dictData['t'][-1]])
	#ax012.set_ylim([-np.pi, np.pi])
	plt.legend(loc='upper right'); plt.grid();

	plt.savefig('results/freq_phaseDiff_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('results/freq_phaseDiff_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)


	return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotFreqAndOrderPar(dictPLL, dictNet, dictData, plotlist=[]):

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	labeldict 	= {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 	= {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$', 'phi': r'$\phi$', 't': r'$t$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color		= ['blue', 'red', 'purple', 'cyan', 'green', 'yellow'] #'magenta'
	linet		= ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)

	fig20 = plt.figure(num=20, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig20.canvas.set_window_title('time-series frequency and order parameter')	# frequency and order parameter
	fig20.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	ax011 = fig20.add_subplot(211)

	plt.axvspan(dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))], dictData['t'][int(1.0*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))], color='b', alpha=0.25)
	phidot = np.diff(dictData['phi'], axis=0)/dictPLL['dt']
	if not dictPLL['intrF'] == 0:
		if isinstance(dictPLL['intrF'], list) or isinstance(dictPLL['intrF'], np.ndarray):
			phidot = phidot / (2.0*np.pi*np.mean(dictPLL['intrF']))
			ylabelname = r'$\frac{\dot{\theta}_k(t)}{\bar{\omega}_k}$'
		else:
			phidot = phidot / (2.0*np.pi*dictPLL['intrF'])
			ylabelname = r'$\frac{\dot{\theta}_k(t)}{\omega}$'
	else:
		ylabelname = r'$\dot{\theta}_k(t)$'
	if not plotlist:
		plt.plot(dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot']],
				phidot[int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))::dictPLL['sampleFplot']], linewidth=2, linestyle=linet[0])
	else:
		plt.plot(dictData['t'][int(0.75 * np.round(np.mean(dictPLL['transmission_delay']) / dictPLL['dt'])):-1:dictPLL['sampleFplot']],
				phidot[int(0.75 * np.round(np.mean(dictPLL['transmission_delay']) / dictPLL['dt']))::dictPLL['sampleFplot'], plotlist], linewidth=2, linestyle=linet[0])
	# plt.plot(dictData['t'][dictNet['max_delay_steps']-1], phidot[int(dictNet['max_delay_steps'])-1,0]+0.001,'go')

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict = labelfont, labelpad=labelpadxaxis)
	plt.ylabel(ylabelname, fontdict = labelfont, labelpad=labelpadyaxis)
	ax011.tick_params(axis='both', which='major', labelsize=tickSize, pad=1)
	ax011.set_xlim([dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))], dictData['t'][-1]])
	plt.grid();

	ax012 = fig20.add_subplot(212)

	plt.plot(dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot']], dictData['orderParam'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])):-1:dictPLL['sampleFplot']], linewidth=1.5)
	plt.xlabel(r'$\omega t/2\pi$', fontdict=labelfont, labelpad=-5)
	plt.ylabel(r'$R(t)$',rotation=90, fontdict=labelfont, labelpad=40)
	ax012.tick_params(axis='both', which='major', labelsize=tickSize, pad=1)
	ax012.set_xlim([dictData['t'][int(0.75*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))], dictData['t'][-1]])
	plt.grid();

	plt.savefig('results/freq_orderPar_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('results/freq_orderPar_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)


	return None


#############################################################################################################################################################################
def plot_histogram(dictPLL: dict, dictNet: dict, dictData: dict, at_index: int = -1, plot_variable: str = 'phase', plotlist: list = [],
															prob_density: bool = True, number_of_bins: int = 25, rel_plot_width: float = 1):
	"""Function that plots a histogram of either the phases or the frequencies.

			Args:
				dictNet:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
				dictPLL:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
				dictData: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
				plot_variable: can either be 'phase' (default) or 'frequency'
				plotlist: list of a subset of the oscillators to be plotted

			Returns:
				saves plotted data to files
		"""

	phase_diff_zero_2pi = 2;  # set to 0 if plotting in (-inf, inf), to 1 of to be plotted in [-pi, pi), to 2 if to be plotted in [-pi/2, 3*pi/2), and to 3 if to be plotted in [0, 2pi)
	dictPLL, dictNet = prepareDictsForPlotting(dictPLL, dictNet)
	if at_index == -1:	# translate since in case of plotting frequency 2 indixes are needed and the '-1' syntax does not work
		at_index = len(dictData['phi'][:, 0])-1
	if at_index == 0:  # in case one chooses to plot a histogram at the very beginning, one needs to correct to allow for the differentiation/the generalized function
		at_index = 1

	if plotlist == []:
		plotlist = [i for i in range(dictNet['Nx'] * dictNet['Ny'])]

	if plot_variable == 'frequency':
		plot_func = lambda x, along_dim: np.diff(x, along_dim)/dictPLL['dt']
		phase_diff_zero_2pi = 0 # since the frequency is not a cyclic variable
		x_label_string = r'$\dot{\theta}(t=%0.2f)$'%(at_index*dictPLL['dt'])
		y_label_string = r'$\textrm{hist}\left(\dot{\theta}\right)$'
	elif plot_variable == 'phase':
		plot_func = lambda x, along_dim: x
		x_label_string = r'$\theta(t=%0.2f)$'%(at_index*dictPLL['dt'])
		y_label_string = r'$\textrm{hist}\left(\theta\right)$'
	else:
		print('Function parameter plot_variable not yet defined, extend the plot function in plot_lib.py!')

	if phase_diff_zero_2pi == 1:	# plot phase-differences in [-pi, pi] interval
		shift2piWin = np.pi
	elif phase_diff_zero_2pi == 2:	# plot phase-differences in [-pi/2, 3*pi/2] interval
		shift2piWin = 0.5*np.pi
	elif phase_diff_zero_2pi == 3:	# plot phase-differences in [0, 2*pi] interval
		shift2piWin = 0

	fig = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig.canvas.set_window_title('histogram of %s at time=%0.2f'%(plot_variable, at_index*dictPLL['dt']))  # frequency and order parameter
	fig.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	if phase_diff_zero_2pi == 0:	# plot phase differences in [0, inf), i.e., we use the unwrapped phases that have counted the cycles/periods
		print('histogram_data:', plot_func(dictData['phi'][(at_index-1):at_index, plotlist], 0)[0])
		plt.hist(plot_func(dictData['phi'][(at_index-1):at_index, plotlist], 0)[0], bins=number_of_bins, rwidth=rel_plot_width, density=prob_density)
	else:
		print('histogram_data:', ((dictData['phi'][at_index, plotlist] + shift2piWin) % (2 * np.pi)) - shift2piWin)
		plt.hist((((dictData['phi'][at_index, plotlist] + shift2piWin) % (2 * np.pi)) - shift2piWin), bins=number_of_bins, rwidth=rel_plot_width, density=prob_density)

	plt.xlabel(x_label_string)
	plt.ylabel(y_label_string)

	plt.savefig('results/histogram_%s_atTime%0.2f_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (plot_variable, at_index*dictPLL['dt'], np.mean(dictPLL['coupK']),
		np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('results/histogram_%s_atTime%0.2f_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (plot_variable, at_index*dictPLL['dt'], np.mean(dictPLL['coupK']),
		np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)

	return None


#################################################################################################################################################################################



# ################################################################################################################################################################################
#
# fig110 = plt.figure()
# fig110.set_size_inches(plot_size_inches_x, plot_size_inches_y)
#
# ax011 = fig110.add_subplot(211)
#
# plt.axvspan(0, t[int(1.0*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))]*dictPLL['dt'], color='b', alpha=0.25)
# for i in range(len(phidot[0,:])):
# 	plt.plot((t[0:-1:dictPLL['sampleFplot']]*dictPLL['dt']), phidot[::dictPLL['sampleFplot'],i]/(2.0*np.pi*dictData['F1']), label='PLL%i' %(i), linewidth=1)
#
# plt.ylabel(r'$f(t)/f0=\dot{\phi}(t)/\omega$',rotation=90, fontsize=60, labelpad=40)
# ax011.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.xlim([0, t[-1]*dictPLL['dt']])
# plt.grid();
#
# ax012 = fig110.add_subplot(212)
#
# plt.plot((t*dictPLL['dt'])[::dictPLL['sampleFplot']], dictData['orderParam'][::dictPLL['sampleFplot']], linewidth=1.5)
# plt.xlabel(r'$t\,[T_{\omega}]$', fontsize=60, labelpad=-5)
# plt.ylabel(r'$R(t)$',rotation=90, fontsize=60, labelpad=40)
# ax012.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.xlim([0, t[-1]*dictPLL['dt']])
# plt.grid();
#
# plt.savefig('results/freq_orderP_allT_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
# plt.savefig('results/freq_orderP_allT_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
# ax011.set_ylim([0, 1.05*np.max(phidot[::dictPLL['sampleFplot'],:]/(2.0*np.pi*dictData['F1']))])
# ax012.set_ylim([0, 1.05])
# plt.savefig('results/freq_orderP_allT_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
# plt.savefig('results/freq_orderP_allT_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
#
# ################################################################################################################################################################################
# # plot instantaneous frequency and phase-differences
# multPrior = 5.0; multAfter = 95.0;
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multPrior/(dictData['F1']*dictPLL['dt']):
# 	priorStart = multPrior/(dictData['F1']*dictPLL['dt'])											# start plot 'multPrior' periods before start of simulation
# else:
# 	priorStart = int(0.5*np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multAfter/(dictData['F1']*dictPLL['dt']):
# 	afterStart = multPrior/(dictData['F1']*dictPLL['dt'])											# end plot after 'multAfter' periods following the start of the simulation
# else:
# 	afterStart = int(0.25*Tsim/dt)
# multStartFin = 110.0; multEndFin = 15.0;
# if len(t) > multStartFin/(dictData['F1']*dictPLL['dt']):
# 	multStartFin = multStartFin/(dictData['F1']*dictPLL['dt'])										# start plot 'multStartFin' periods before end of simulation
# else:
# 	multStartFin = int(0.65*Tsim/dt)
# if multStartFin > multEndFin:
# 	multEndFin = multEndFin/(dictData['F1']*dictPLL['dt'])											# end plot after 'multEndFin' periods before end of simulation
# else:
# 	multEndFin = int(0.95*Tsim/dt)
#
# xstart1 = int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])-priorStart);		xstart2 = len(t)-int(multStartFin);
# xend1	= int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+afterStart);		xend2   = len(t)-int(multEndFin);
# xmin1 	= t[xstart1]*dictPLL['dt']; xmax1 	= t[xend1]*dictPLL['dt'];
# xmin2 	= t[xstart2]*dictPLL['dt']; xmax2 	= t[xend2]*dictPLL['dt'];
#
# fig111 = plt.figure(figsize=(figwidth,figheight))
# fig111.set_size_inches(plot_size_inches_x, plot_size_inches_y)
#
# ax111 = fig111.add_subplot(221)
#
# ''' PLOT HERE THE FIRST PART '''
# plt.axvspan(0, t[int(1.0*np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt']))]*dictPLL['dt'], color='b', alpha=0.25)
# for i in range(len(phidot[0,:])):
# 	plt.plot((t[xstart1:xend1:dictPLL['sampleFplot']]*dictPLL['dt']), phidot[xstart1:xend1:dictPLL['sampleFplot'],i]/(2.0*np.pi*dictData['F1']), label='PLL%i' %(i), linewidth=1)
#
# plt.ylabel(r'$f(t)/f0=\dot{\phi}(t)/\omega$',rotation=90, fontsize=60, labelpad=40)
# ax111.set_xlim(xmin1, xmax1)
# ax111.set_ylim(0.99*np.min([np.min(phidot[xstart1:xend1:dictPLL['sampleFplot'],:]/(2.0*np.pi*dictData['F1'])), np.min(phidot[xstart2:xend2:dictPLL['sampleFplot'],:]/(2.0*np.pi*dictData['F1']))]),
# 			   1.01*np.max([np.max(phidot[xstart1:xend1:dictPLL['sampleFplot'],:]/(2.0*np.pi*dictData['F1'])), np.max(phidot[xstart2:xend2:dictPLL['sampleFplot'],:]/(2.0*np.pi*dictData['F1']))]) )
#
# ax111.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.grid();
#
# ax112 = fig111.add_subplot(222)
#
# ''' PLOT HERE THE SECOND PART '''
# for i in range(len(phidot[0,:])):
# 	plt.plot((t[xstart2:xend2:dictPLL['sampleFplot']]*dictPLL['dt']), phidot[xstart2:xend2:dictPLL['sampleFplot'],i]/(2.0*np.pi*dictData['F1']), label='PLL%i' %(i), linewidth=1)
#
# #plt.xlabel(r'$time$',fontsize=60,labelpad=-5)
# ax112.set_xlim(xmin2, xmax2)
# ax112.set_ylim(0.99*np.min([np.min(phidot[xstart1:xend1:dictPLL['sampleFplot'],:]/(2.0*np.pi*dictData['F1'])), np.min(phidot[xstart2:xend2:dictPLL['sampleFplot'],:]/(2.0*np.pi*dictData['F1']))]),
# 			   1.01*np.max([np.max(phidot[xstart1:xend1:dictPLL['sampleFplot'],:]/(2.0*np.pi*dictData['F1'])), np.max(phidot[xstart2:xend2:dictPLL['sampleFplot'],:]/(2.0*np.pi*dictData['F1']))]) )
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
# plt.plot((t*dictPLL['dt'])[xstart1:xend1:dictPLL['sampleFplot']], dictData['orderParam'][xstart1:xend1:dictPLL['sampleFplot']], linewidth=1.5)
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
# plt.plot((t*dictPLL['dt'])[xstart2:xend2:dictPLL['sampleFplot']], dictData['orderParam'][xstart2:xend2:dictPLL['sampleFplot']], linewidth=1.5)
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
# plt.savefig('results/freq_orderP_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
# plt.savefig('results/freq_orderP_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
#
# multPrior = 5.0; multAfter = 20.0;
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multPrior/(dictData['F1']*dictPLL['dt']):
# 	priorStart = multPrior/(dictData['F1']*dictPLL['dt'])											# start plot 'multPrior' periods before start of simulation
# else:
# 	priorStart = int(0.1*np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multAfter/(dictData['F1']*dictPLL['dt']):
# 	afterStart = multPrior/(dictData['F1']*dictPLL['dt'])											# end plot after 'multAfter' periods following the start of the simulation
# else:
# 	afterStart = int(0.1*Tsim/dt)
# multStartFin = 40.0; multEndFin = 15.0;
# if len(t) > multStartFin/(dictData['F1']*dictPLL['dt']):
# 	multStartFin = multStartFin/(dictData['F1']*dictPLL['dt'])										# start plot 'multStartFin' periods before end of simulation
# else:
# 	multStartFin = int(0.85*Tsim/dt)
# if multStartFin > multEndFin:
# 	multEndFin = multEndFin/(dictData['F1']*dictPLL['dt'])											# end plot after 'multEndFin' periods before end of simulation
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
# plt.savefig('results/freq_orderP_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
# plt.savefig('results/freq_orderP_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
#
# ################################################################################################################################################################################
# multPrior = 5.0; multAfter = 95.0;
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multPrior/(dictData['F1']*dictPLL['dt']):
# 	priorStart = multPrior/(dictData['F1']*dictPLL['dt'])											# start plot 'multPrior' periods before start of simulation
# else:
# 	priorStart = int(0.5*np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multAfter/(dictData['F1']*dictPLL['dt']):
# 	afterStart = multPrior/(dictData['F1']*dictPLL['dt'])											# end plot after 'multAfter' periods following the start of the simulation
# else:
# 	afterStart = int(0.25*Tsim/dt)
# multStartFin = 110.0; multEndFin = 15.0;
# if len(t) > multStartFin/(dictData['F1']*dictPLL['dt']):
# 	multStartFin = multStartFin/(dictData['F1']*dictPLL['dt'])										# start plot 'multStartFin' periods before end of simulation
# else:
# 	multStartFin = int(0.65*Tsim/dt)
# if multStartFin > multEndFin:
# 	multEndFin = multEndFin/(dictData['F1']*dictPLL['dt'])											# end plot after 'multEndFin' periods before end of simulation
# else:
# 	multEndFin = int(0.95*Tsim/dt)
#
# xstart1 = int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])-priorStart);		xstart2 = len(t)-int(multStartFin);
# xend1	= int(np.round(np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])+afterStart);		xend2   = len(t)-int(multEndFin);
# xmin1 	= t[xstart1]*dictPLL['dt']; xmax1 	= t[xend1]*dictPLL['dt'];
# xmin2 	= t[xstart2]*dictPLL['dt']; xmax2 	= t[xend2]*dictPLL['dt'];
#
# fig211 = plt.figure(figsize=(figwidth,figheight))
# fig211.set_size_inches(plot_size_inches_x, plot_size_inches_y)
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
# plt.savefig('results/phases_phaseDiff_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
# plt.savefig('results/phases_phaseDiff_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
#
# multPrior = 5.0; multAfter = 20.0;
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multPrior/(dictData['F1']*dictPLL['dt']):
# 	priorStart = multPrior/(dictData['F1']*dictPLL['dt'])											# start plot 'multPrior' periods before start of simulation
# else:
# 	priorStart = int(0.1*np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])
# if np.mean(dictPLL['transmission_delay'])/dictPLL['dt'] > multAfter/(dictData['F1']*dictPLL['dt']):
# 	afterStart = multPrior/(dictData['F1']*dictPLL['dt'])											# end plot after 'multAfter' periods following the start of the simulation
# else:
# 	if Tsim > 2*delay:
# 		afterStart = int(2.0*np.mean(dictPLL['transmission_delay'])/dictPLL['dt'])
# 	else:
# 		afterStart = int(0.05*Tsim/dt)
# multStartFin = 40.0; multEndFin = 15.0;
# if len(t) > multStartFin/(dictData['F1']*dictPLL['dt']):
# 	multStartFin = multStartFin/(dictData['F1']*dictPLL['dt'])										# start plot 'multStartFin' periods before end of simulation
# else:
# 	multStartFin = int(0.85*Tsim/dt)
# if multStartFin > multEndFin:
# 	multEndFin = multEndFin/(dictData['F1']*dictPLL['dt'])											# end plot after 'multEndFin' periods before end of simulation
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
# plt.savefig('results/phases_phaseDiff_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
# plt.savefig('results/phases_phaseDiff_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dictPLL['coupK']), np.mean(dictPLL['cutFc']), np.mean(dictPLL['syncF']), np.mean(dictPLL['transmission_delay']), np.mean(dictPLL['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val)
#
# ################################################################################################################################################################################
