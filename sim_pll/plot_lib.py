#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import random
import numpy as np
import numpy.ma as ma
import matplotlib
import codecs
import csv
import os, gc, sys

if not os.environ.get('SGE_ROOT') == None:  # this environment variable is set within the queue network, i.e., if it exists, 'Agg' mode to supress output
	print('NOTE: \"matplotlib.use(\'Agg\')\"-mode active, plots are not shown on screen, just saved to results folder!\n')
	matplotlib.use('Agg')  # '%pylab inline'
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import optimize
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.ndimage.filters import uniform_filter1d
# from scipy.interpolate import spline
from scipy.special import lambertw
from scipy.signal import square
import allantools
import itertools
import math

from sim_pll import evaluation_lib as eva

import datetime
import time

now = datetime.datetime.now()

''' Enable automatic carbage collector '''
gc.enable()

''' All plots in latex mode '''
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.rcParams['agg.path.chunksize'] = 10000

''' STYLEPACKS '''
titlefont = {
	'family': 'serif',
	'color': 'black',
	'weight': 'normal',
	'size': 9,
}

labelfont = {
	'family': 'sans-serif',
	'color': 'black',
	'weight': 'normal',
	'size': 36,
}

annotationfont = {
	'family': 'monospace',
	'color': (0, 0.27, 0.08),
	'weight': 'normal',
	'size': 14,
}

# plot parameter
axisLabel = 60
tickSize = 25
titleLabel = 10
dpi_val = 150
figwidth = 10  # 8
figheight = 5
plot_size_inches_x = 10
plot_size_inches_y = 5
labelpadxaxis = 10
labelpadyaxis = 20


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def prepareDictsForPlotting(dict_pll, dict_net):
	if dict_pll['cutFc'] is None:
		dict_pll.update({'cutFc': np.inf})

	if dict_pll['transmission_delay'] is None:
		dict_pll.update({'transmission_delay': 0})

	if not np.abs(np.min(dict_pll['intrF'])) > 1E-17:  # for f=0, there would otherwise be a float division by zero
		dict_pll.update({'intrF': 1})
		print('Since intrinsic frequency was zero: for plotting set to one to generate boundaries!')

	return dict_pll, dict_net


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_power_spectral_density(dict_pll: dict, dict_net: dict, dict_data: dict, plotlist=[], saveData=False):
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	f = []
	Pxx_db = []
	peak_power_val = []
	index_of_highest_peak = []
	value_of_highest_peak = []
	frequency_of_max_peak = []
	# compute the PSDs either of a list of oscillators or for all of them
	if plotlist:
		print('\nPlotting PSD according to given plotlist:', plotlist)
		for i in range(len(plotlist)):  # calculate spectrum of signals for the oscillators specified in the list
			ftemp, Pxx_temp = eva.calcSpectrum(dict_data['phi'][:, plotlist[i]], dict_pll, dict_net, plotlist[i], dict_pll['percent_of_Tsim'])
			f.append(ftemp[0])
			# print('Test Pxx_temp[0]:', Pxx_temp[0])
			Pxx_db.append(Pxx_temp[0])
	else:
		plotlist = []
		for i in range(len(dict_data['phi'][0, :])):  # calculate spectrum of signals for all oscillators
			ftemp, Pxx_temp = eva.calcSpectrum(dict_data['phi'][:, i], dict_pll, dict_net, i, dict_pll['percent_of_Tsim'])
			f.append(ftemp[0])
			# print('Test Pxx_temp[0]:', Pxx_temp[0])
			Pxx_db.append(Pxx_temp[0])
			plotlist.append(i)

	fig1 = plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig1.canvas.manager.set_window_title('spectral density of synchronized state')  # plot spectrum
	fig1.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.xlabel('frequencies [Hz]', fontdict=labelfont, labelpad=labelpadxaxis)
	plt.ylabel('P [dBm]', fontdict=labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)

	for i in range(len(f)):
		# print('Test:', Pxx_db[i])
		index_of_highest_peak.append(np.argmax(Pxx_db[i]))  # find the principle peak
		frequency_of_max_peak.append(f[i][index_of_highest_peak[i]])  # save the frequency where the maximum peak is found
		peak_power_val.append(Pxx_db[i][index_of_highest_peak[i]])  # save the peak power value

		plt.plot(f[i], Pxx_db[i], '-', label='PLL%i' % (plotlist[i]))

	plt.title(r'power spectrum $\Delta f=$%0.5E, peak at $Pxx_0^\textrm{peak}$=%0.2f' % ((f[0][2] - f[0][1]), peak_power_val[0]), fontdict=labelfont)
	plt.legend(loc='upper right')
	plt.grid()

	try:
		plt.ylim([np.min(Pxx_db[0][index_of_highest_peak[0]:]), np.max(peak_power_val) + 5])
	except:
		print('Could not set ylim accordingly!')
	plt.xlim(0, 12.5 * np.min(dict_pll['intrF']))
	plt.savefig('results/powerdensity_dB_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/powerdensity_dB_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	plt.xlim(0, 3.8 * np.min(dict_pll['intrF']))
	plt.savefig('results/powerdensityLessFreq_dB_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val)
	plt.savefig('results/powerdensityLessFreq_dB_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val)

	plt.xlim(frequency_of_max_peak[i] - 1.2 * np.min(dict_pll['coupK']), frequency_of_max_peak[i] + 1.2 * np.max(dict_pll['coupK']))
	for i in range(len(f)):
		plt.plot(f[i][index_of_highest_peak[i] - int(0.1 * np.min(dict_pll['intrF']) / (f[0][2] - f[0][1]))], Pxx_db[i][index_of_highest_peak[i]], 'r*',
				 f[i][index_of_highest_peak[i] + int(0.1 * np.min(dict_pll['intrF']) / (f[0][2] - f[0][1]))], Pxx_db[i][index_of_highest_peak[i]], 'r*')
	plt.savefig('results/powerdensity1stHarm_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/powerdensity1stHarm_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	freq_res_bins_both_peak_sides = 13
	try:
		minima_of_all_psd_in_zoom_range = []
		for i in range(len(f)):
			minima_of_all_psd_in_zoom_range.append(np.min(Pxx_db[i][(index_of_highest_peak[i] - freq_res_bins_both_peak_sides):(index_of_highest_peak[i] + freq_res_bins_both_peak_sides)]))
			print('minimum in plot range of PLL%i = %0.2f' % (i, minima_of_all_psd_in_zoom_range[i]))
		plt.ylim([np.min(minima_of_all_psd_in_zoom_range) - 3, np.max(peak_power_val) + 3])
	except:
		print('Could not set ylim accordingly!')
	plt.xlim(frequency_of_max_peak[i] - freq_res_bins_both_peak_sides * (f[0][2] - f[0][1]), frequency_of_max_peak[i] + freq_res_bins_both_peak_sides * (f[0][2] - f[0][1]))
	plt.savefig('results/powerdensity1stHarmCloseZoom_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/powerdensity1stHarmCloseZoom_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	try:
		print('np.min(Pxx_db[i][:])=%02f, peak_power_val[0]=%02f' % (np.min(Pxx_db[i][5:]), peak_power_val[0]))
		plt.ylim([np.min(Pxx_db[i][5:]), peak_power_val[0] + 5])
	except:
		print('np.min(Pxx_db[i][:]), peak_power_val[0] either Inf or NAN.')
	plt.xlim(0, 8.5 * np.min(dict_pll['intrF']))

	fig2 = plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')  # plot spectrum
	fig2.canvas.manager.set_window_title('one-sided spectral density')
	fig2.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	xHz = 0.001
	onsidedPSD_params = []
	oneSidPSDwidthsm3dB = []
	quality_factors = []  # distance from the principle peak to measure damping
	Freqres = f[0][3] - f[0][2]
	linestyle = ['-', '--', '-', '--', '-', '--']
	for i in range(len(f)):
		frequency_of_max_peak[i] = 0
		coup1_delt_3dB = 0
		# mutually coupled SLL1
		index_of_highest_peak[i] = np.argmax(Pxx_db[i])  # find the index of the principle peak (max dB) of the free-running SLL
		frequency_of_max_peak[i] = f[i][index_of_highest_peak[i]]  # use the above index to identify the frequency of the peaks location
		coup1_times_X = np.argmax(f[i] >= 2.25 * frequency_of_max_peak[i])  # find the index for the frequency being 2.25 times that of the peak
		m3dB_freqcind1 = index_of_highest_peak[i] + np.where(Pxx_db[i][index_of_highest_peak[i]:] <= (Pxx_db[i][index_of_highest_peak[i]] - 3.0))[0][0]  # find the index associated to a power drop of -3dB w.r.t. the peak's value
		print('\n\nm3dB_freqcind1:', m3dB_freqcind1, '\n\n')
		m3dB_freqc_val = f[i][m3dB_freqcind1]  # extract the frequency at -3dB

		coup1_delt_3dB = np.abs(m3dB_freqc_val - frequency_of_max_peak[i] - Freqres)

		print('Calculating: f[peak]-f[-3dBm]=', frequency_of_max_peak[i], '-', m3dB_freqc_val, '=', coup1_delt_3dB, '\n power 1st harmonic', Pxx_db[i][index_of_highest_peak[i]], '\n')
		print('\nPxx_db[', i, '][', index_of_highest_peak[i], ']-3.0=', Pxx_db[i][index_of_highest_peak[i]] - 3.0)
		# print('TEST Pxx_dBm[',i,'][ index_of_highest_peak[i]+np.argmin(Pxx_db[',i,'][',index_of_highest_peak[i],':]<=(Pxx_db[',i,'][',index_of_highest_peak[i],']-3)) ]: ',
		#					Pxx_db[i][ index_of_highest_peak[i]+np.argmin(Pxx_db[i][index_of_highest_peak[i]:].copy()<=(Pxx_db[i][index_of_highest_peak[i]]-3.0)) ],
		#					' -> frequency where PxxMax-3dB:', m3dB_freqc_val,'Hz')
		print('np.where(Pxx_db[', i, '][', index_of_highest_peak[i], ':]<=(Pxx_db[', i, '][', index_of_highest_peak[i], ']-3))[0][0]=',
			  np.where(Pxx_db[i][index_of_highest_peak[i]:] <= (Pxx_db[i][index_of_highest_peak[i]] - 3.0))[0][0], '\n')
		# calculate linear function between first two points of one-sided powerspectrum: y = slope * x + yinter, interPol{1,2} are the relate to the y-coordinates between which Pxx-3dB lies
		# interpolP1 = index_of_highest_peak[i] + np.where(Pxx_db[i][index_of_highest_peak[i]:] > (Pxx_db[i][index_of_highest_peak[i]]-3.0))[0][-1]	# find the results in the PSD vectors that are adjacent to the -3dB point
		# interpolP2 = index_of_highest_peak[i] + np.where(Pxx_db[i][index_of_highest_peak[i]:] < (Pxx_db[i][index_of_highest_peak[i]]-3.0))[0][0]		# PROBLEM IF Pxx goes up again!!!!!!! sketch to see
		interpolP1 = index_of_highest_peak[i] + np.where(Pxx_db[i][index_of_highest_peak[i]:] < (Pxx_db[i][index_of_highest_peak[i]] - 3.0))[0][
			0]  # find the first point in the PSD smaller than PxxMax-3dB and then take the prior point
		interpolP2 = interpolP1 - 1  # as the second to interpolate
		# print('{interpolP1, interpolP2}:', interpolP1, interpolP2)
		slope = (Pxx_db[i][interpolP2] - Pxx_db[i][interpolP1]) / (f[i][interpolP2] - f[i][interpolP1])
		yinter = Pxx_db[i][interpolP1] - slope * f[i][interpolP1]
		# print('{slope, yinter}:', slope, yinter)
		# slope  = ( Pxx_db[i][index_of_highest_peak[i]+1]-Pxx_db[i][index_of_highest_peak[i]]) / ( f[i][index_of_highest_peak[i]+1]-f[i][index_of_highest_peak[i]])
		# yinter = Pxx_db[i][index_of_highest_peak[i]] - slope * f[i][index_of_highest_peak[i]]
		fm3dB = (Pxx_db[i][index_of_highest_peak[i]] - 3.0 - yinter) / slope
		oneSidPSDwidthsm3dB.append(fm3dB - f[i][index_of_highest_peak[i]])
		quality_factors.append(frequency_of_max_peak[i] / (2 * (fm3dB - f[i][index_of_highest_peak[i]])))
		print('For the mutually coupled SLL', i, ' we find the principle peak at f =', frequency_of_max_peak[i], ', -3dBm delta_f =', coup1_delt_3dB, ', and hence a quality factor Q = ',
			  frequency_of_max_peak[i] / (2 * (fm3dB - f[i][index_of_highest_peak[i]])))
		plt.plot(10.0 * np.log10(fm3dB - frequency_of_max_peak[i] + Freqres), Pxx_db[i][index_of_highest_peak[i]] - 3.0, 'r*', markersize=2)
		if coup1_delt_3dB == 0:
			print('frequency resolution of power spectrum too large or power spectrum approaching delta-like peak!')
		try:
			onsidedPSD_params.append([frequency_of_max_peak[i], coup1_delt_3dB])
		except:
			onsidedPSD_params.append([0, 0])
		if (m3dB_freqcind1 < 3.1 and m3dB_freqcind1 > 2.9):
			# plt.plot(10.0*np.log10(1E-12), Pxx_db[i][index_of_highest_peak[i]], 'r*', markersize=2)
			plt.plot(10.0 * np.log10(f[i][m3dB_freqcind1] - frequency_of_max_peak[i]), Pxx_db[i][m3dB_freqcind1], 'r+', markersize=2)
		else:
			print('CHECK frequency resolution of power spectrum and noise strength. Cannot use this method.')
		if dict_net['topology'] == 'compareEntrVsMutual':
			plt.plot(10.0 * np.log10(f[i][index_of_highest_peak[i]:coup1_times_X] - frequency_of_max_peak[i] + Freqres), Pxx_db[i][index_of_highest_peak[i]:coup1_times_X], linestyle[i],
					 label='PSD PLL%i' % (i), markersize=2)
		else:
			plt.plot(10.0 * np.log10(f[i][index_of_highest_peak[i]:coup1_times_X] - frequency_of_max_peak[i] + Freqres), Pxx_db[i][index_of_highest_peak[i]:coup1_times_X],
					 label='PSD PLL%i' % (plotlist[i]), markersize=2)
	try:
		plt.title(r'$\gamma_0^{(\textrm{PSDfit})}=$%0.4E, $\gamma_1=$%0.4E, $\gamma_2=$%0.4E' % (params[0][0], params[1][0], params[2][0]), fontdict=titlefont)
	except:
		print('No (two-sided, 1st harmonic) PSD fits available!')
	if dict_net['Nx'] * dict_net['Ny'] == 2:
		onsidedPSD_params.append([0, 0])  # necessary, otherwise error on write-out to csv file
	# plt.plot(10.0*np.log10(powerspecPLL1['f'][0][index_of_highest_peak[i]:coup1_times_X].copy()-frequency_of_max_peak[i]), !!!!! , 'y-', label=r'$1/f^2$')
	plt.legend(loc='upper right')
	# plt.xlim([0,f01+20*max(Kvco1,Kvco2)])	#plt.ylim(-100,0)
	plt.xlabel(r'$10\log_{10}\left(f-f_{\rm peak}\right)$ [Hz]', fontdict=labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$P$ [dBm]', fontdict=labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=35)
	plt.grid()
	plt.savefig('results/onsidedPSD_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/onsidedPSD_dBm_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	print('\nwidths (HWHM) of PSDs obtained from one-sided PSD with interpolation for all oscis:', oneSidPSDwidthsm3dB)
	print('Mean of widths of PSDs obtained from one-sided PSD with interpolation for all oscis:', np.mean(oneSidPSDwidthsm3dB))
	print('Std of widths of PSDs obtained from one-sided PSD with interpolation for all oscis:', np.std(oneSidPSDwidthsm3dB), '\n')
	print('all quality factors obtained from the one-sided PSD:', quality_factors)
	print('Mean all quality factors obtained from the one-sided PSD:', np.mean(quality_factors))
	print('Std all quality factors obtained from the one-sided PSD:', np.std(quality_factors), '\n')

	fig1 = plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	for i in range(len(f)):
		plt.plot(oneSidPSDwidthsm3dB[i] + f[i][index_of_highest_peak[i]], Pxx_db[i][index_of_highest_peak[i]] - 3.0, 'r*', markersize=2)

	if saveData:
		np.savez('results/powerSpec_K%.2f_Fc%.4f_FOm%.2f_tau%.2f_c%.7e_%d_%d_%d.npz' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				 powerspec=np.array([f, Pxx_db]))

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_phases_two_pi_periodic(dict_pll: dict, dict_net: dict, dict_data: dict):
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	labeldict = {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$',
				 'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 = {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$', 'phi': r'$\phi$', 't': r'$t$',
				  'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color = ['blue', 'red', 'purple', 'cyan', 'green', 'yellow']  # 'magenta'
	linet = ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig3 = plt.figure(num=3, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig3.canvas.manager.set_window_title('phases')  # plot the phase
	fig3.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.plot(dict_data['t'], dict_data['phi'] % (2 * np.pi), linewidth=1, linestyle=linet[0])
	plt.plot(dict_data['t'][dict_net['max_delay_steps'] - 1], dict_data['phi'][int(dict_net['max_delay_steps']) - 1, 0] % (2 * np.pi) + 0.05, 'go')

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict=labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$\theta(t)_{\textrm{mod}\,2\pi}$', fontdict=labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	# plt.legend(loc='upper right')

	plt.savefig('results/phases2pi-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/phases2pi-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_phases_unwrapped(dict_pll: dict, dict_net: dict, dict_data: dict):
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	labeldict = {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$',
				 'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 = {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$', 'phi': r'$\phi$', 't': r'$t$',
				  'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color = ['blue', 'red', 'purple', 'cyan', 'green', 'yellow']  # 'magenta'
	linet = ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig4 = plt.figure(num=4, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig4.canvas.manager.set_window_title('phases')  # plot the phase
	fig4.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.plot(dict_data['t'], dict_data['phi'], linewidth=1, linestyle=linet[0])
	plt.plot(dict_data['t'][dict_net['max_delay_steps'] - 1], dict_data['phi'][int(dict_net['max_delay_steps']) - 1, 0] + 0.05, 'go')

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict=labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$\theta(t)$', fontdict=labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	# plt.legend(loc='upper right')

	plt.savefig('results/phasesInf-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/phasesInf-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_inst_frequency(dict_pll: dict, dict_net: dict, dict_data: dict):
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	labeldict = {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$',
				 'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 = {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$', 'phi': r'$\phi$', 't': r'$t$',
				  'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color = ['blue', 'red', 'purple', 'cyan', 'green', 'yellow']  # 'magenta'
	linet = ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig5 = plt.figure(num=5, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig5.canvas.manager.set_window_title('frequency')  # plot the phase
	fig5.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	phidot = np.diff(dict_data['phi'], axis=0) / dict_pll['dt']
	plt.plot(dict_data['t'][0:-1], phidot, linewidth=1, linestyle=linet[0])
	plt.plot(dict_data['t'][dict_net['max_delay_steps'] - 1], phidot[int(dict_net['max_delay_steps']) - 1, 0] + 0.001, 'go')

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict=labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$\theta(t)$', fontdict=labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	# plt.legend(loc='upper right')

	plt.xlabel(r'$t\,[T_{\omega}]$', fontdict=labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$\dot{\phi}(t)$ [rad Hz]', fontdict=labelfont, labelpad=labelpadyaxis)
	plt.savefig('results/freq-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/freq-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.xlim([np.mean(dict_pll['transmission_delay']) - 25 * 1.0 / (np.min(dict_pll['intrF'])), np.mean(dict_pll['transmission_delay']) + 35 * 1.0 / (np.min(dict_pll['intrF']))])
	plt.ylim([0.99 * np.min(phidot[0:int(np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']) + 25 * 1.0 / (np.min(dict_pll['intrF']) * dict_pll['dt'])) - 1, :]),
			  1.01 * np.max(phidot[0:int(np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']) + 25 * 1.0 / (np.min(dict_pll['intrF']) * dict_pll['dt'])) - 1, :])])
	plt.savefig('results/freqInit-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/freqInit-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.xlim([np.mean(dict_pll['transmission_delay']) - 25 * 1.0 / (np.min(dict_pll['intrF'])), np.mean(dict_pll['transmission_delay']) + 400 * 1.0 / (np.min(dict_pll['intrF']))])
	plt.ylim([np.min(phidot[int(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'] - 25 * 1.0 / (dict_pll['sampleF'] / np.min(dict_pll['intrF']))):int(
		np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']) + 400 * 1.0 / (np.min(dict_pll['intrF']) * dict_pll['dt'])) - 1, :]) - 0.05, np.max(phidot[int(np.mean(
		dict_pll['transmission_delay']) / dict_pll['dt'] - 25 * 1.0 / (dict_pll['sampleF'] / np.min(dict_pll['intrF']))):int(
		np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']) + 400 * 1.0 / (np.min(dict_pll['intrF']) * dict_pll['dt'])) - 1, :]) + 0.05]);
	plt.savefig('results/freqInit1-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/freqInit1-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.xlim([0, dict_data['t'][-1]]);
	plt.ylim([np.min(phidot[int(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'] + 0.5 * dict_pll['sampleF'] / np.min(dict_pll['intrF'])):, :]) - 0.05,
			  np.max(phidot[int(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'] + 0.5 * dict_pll['sampleF'] / np.min(dict_pll['intrF'])):, :]) + 0.05]);

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_order_parameter(dict_pll: dict, dict_net: dict, dict_data: dict):
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	fig6 = plt.figure(num=6, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig6.canvas.manager.set_window_title('order parameter over time')  # plot the order parameter in dependence of time
	fig6.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.plot(dict_data['t'], dict_data['order_parameter'])
	plt.plot(np.mean(dict_pll['transmission_delay']), dict_data['order_parameter'][int(round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))], 'yo', ms=5)  # mark where the simulation starts
	if -int(dict_pll['timeSeriesAverTime'] * 1.0 / (dict_data['F1'] * dict_pll['dt'])) >= 0:
		plt.axvspan(dict_data['t'][-int(dict_pll['timeSeriesAverTime'] * 1.0 / (dict_data['F1'] * dict_pll['dt']))], dict_data['t'][-1], color='b', alpha=0.3)
	plt.title(r'mean order parameter $\bar{R}=$%.2f, and $\bar{\sigma}=$%.4f' % (
	np.mean(dict_data['order_parameter'][-int(round(dict_pll['timeSeriesAverTime'] * 1.0 / (dict_data['F1'] * dict_pll['dt']))):]),
	np.std(dict_data['order_parameter'][-int(round(dict_pll['timeSeriesAverTime'] * 1.0 / (dict_data['F1'] * dict_pll['dt']))):])), fontdict=titlefont)
	plt.xlabel(r'$t\,[T_{\omega}]$', fontdict=labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$R(t,m_x=%d,m_y=%d )$' % (dict_net['mx'], dict_net['my']), fontdict=labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)

	plt.savefig('results/orderP-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				bbox_inches="tight")
	plt.savefig('results/orderP-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	try:
		plt.xlim([dict_data['t'][-int(round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'] + 75 * 1.0 / (dict_data['F1'] * dict_pll['dt'])))], dict_data['t'][-1]]);  # plt.ylim([]);
	except:
		plt.xlim([0, 15])
	plt.savefig('results/orderPFin-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				bbox_inches="tight")
	plt.savefig('results/orderPFin-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.xlim([0, np.mean(dict_pll['transmission_delay']) + 125 * 1.0 / (dict_data['F1'])])  # plt.ylim([]);
	plt.savefig('results/orderPInit-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				bbox_inches="tight")
	plt.savefig('results/orderPInit-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.xlim([0, np.mean(dict_pll['transmission_delay']) + 2250 * 1.0 / (dict_data['F1'])])  # plt.ylim([]);
	plt.savefig('results/orderPInit1-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				bbox_inches="tight")
	plt.savefig('results/orderPInit1-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	# print('\nlast entry order parameter: R-1 = %.3e' % (dict_data['order_parameter'][-1]-1) )
	# print('\nlast entries order parameter: R = ', dict_data['order_parameter'][-25:])

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_phase_difference_wrt_to_osci_kzero(dict_pll: dict, dict_net: dict, dict_data: dict):
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	fig7 = plt.figure(num=7, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig7.canvas.manager.set_window_title('phase relations')
	fig7.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	if not dict_net['topology'] == 'compareEntrVsMutual':
		plt.plot(dict_data['t'][::dict_pll['sampleFplot']], ((dict_data['phi'][::dict_pll['sampleFplot'], 0] - dict_data['phi'][::dict_pll['sampleFplot'], 1] + np.pi) % (2. * np.pi)) - np.pi,
				 label=r'$\phi_{0}-\phi_{1}$')  # math.fmod(phi[:,:], 2.*np.pi))
		if not dict_net['Nx'] * dict_net['Ny'] == 2:
			plt.plot(dict_data['t'][::dict_pll['sampleFplot']], ((dict_data['phi'][::dict_pll['sampleFplot'], 1] - dict_data['phi'][::dict_pll['sampleFplot'], 2] + np.pi) % (2. * np.pi)) - np.pi,
					 label=r'$\phi_{1}-\phi_{2}$')
			plt.plot(dict_data['t'][::dict_pll['sampleFplot']], ((dict_data['phi'][::dict_pll['sampleFplot'], 0] - dict_data['phi'][::dict_pll['sampleFplot'], 2] + np.pi) % (2. * np.pi)) - np.pi,
					 label=r'$\phi_{0}-\phi_{2}$')
		plt.plot(np.mean(dict_pll['transmission_delay']), ((dict_data['phi'][int(round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])), 0] - dict_data['phi'][
			int(round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])), 1] + np.pi) % (2. * np.pi)) - np.pi, 'yo', ms=5)
		# plt.axvspan(t[-int(5.5*1.0/(dict_data['F1']*dict_pll['dt']))]*dict_pll['dt'], t[-1]*dict_pll['dt'], color='b', alpha=0.3)
		if dict_net['Nx'] * dict_net['Ny'] >= 3:
			plt.title(r'phases $\phi_{0}=%.4f$, $\phi_{1}=%.4f$, $\phi_{R}=%.4f$  [rad]' % ((-1) * (np.mod(dict_data['phi'][-10][2] - dict_data['phi'][-10][0] + np.pi, 2.0 * np.pi) - np.pi),
					np.mod(dict_data['phi'][-10][1] - dict_data['phi'][-10][0] + np.pi, 2.0 * np.pi) - np.pi - (np.mod(dict_data['phi'][-10][2] - dict_data['phi'][-10][0] + np.pi, 2.0 * np.pi) - np.pi), 0),
					fontdict=titlefont)
		else:
			plt.title(r'phases [rad]', fontdict=titlefont)
		plt.xlabel(r'$t\,[T_{\omega}]$', fontdict=labelfont, labelpad=labelpadxaxis)
		plt.ylabel(r'$\phi_k(t)-\phi_0(t)$', fontdict=labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)

		plt.legend(loc='upper right');
		plt.savefig('results/phaseRela-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					bbox_inches="tight")
		plt.savefig('results/phaseRela-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")
	else:
		plt.plot(dict_data['t'][-int(25 * 1.0 / (dict_data['F1'] * dict_pll['dt'])):],
				 ((dict_data['phi'][-int(25 * 1.0 / (dict_data['F1'] * dict_pll['dt'])):, 0] - dict_data['phi'][-int(25 * 1.0 / (dict_data['F1'] * dict_pll['dt'])):, 1] + np.pi) % (2. * np.pi)) - np.pi,
				 '-', label=r'$\phi_{0}-\phi_{1}$ mutual')
		plt.plot(dict_data['t'][-int(25 * 1.0 / (dict_data['F1'] * dict_pll['dt'])):],
				 ((dict_data['phi'][-int(25 * 1.0 / (dict_data['F1'] * dict_pll['dt'])):, 3] - dict_data['phi'][-int(25 * 1.0 / (dict_data['F1'] * dict_pll['dt'])):, 2] + np.pi) % (2. * np.pi)) - np.pi,
				 '--', label=r'$\phi_{3}-\phi_{2}$ entrain')
		# plt.plot((t[-int(12*1.0/(dict_data['F1']*dict_pll['dt'])):]*dict_pll['dt']),((dict_data['phi'][-int(12*1.0/(dict_data['F1']*dict_pll['dt'])):,0]-dict_data['phi'][-int(12*1.0/(dict_data['F1']*dict_pll['dt'])):,5]+np.pi)%(2.*np.pi))-np.pi,'-',label=r'$\phi_{0}-\phi_{5}$ mutual  vs freeRef')
		# plt.plot((t[-int(12*1.0/(dict_data['F1']*dict_pll['dt'])):]*dict_pll['dt']),((dict_data['phi'][-int(12*1.0/(dict_data['F1']*dict_pll['dt'])):,3]-dict_data['phi'][-int(12*1.0/(dict_data['F1']*dict_pll['dt'])):,5]+np.pi)%(2.*np.pi))-np.pi,'--',label=r'$\phi_{3}-\phi_{5}$ entrain vs freeRef')
		# plt.plot((t[-int(12*1.0/(dict_data['F1']*dict_pll['dt'])):]*dict_pll['dt']),((dict_data['phi'][-int(12*1.0/(dict_data['F1']*dict_pll['dt'])):,0]-dict_data['phi'][-int(12*1.0/(dict_data['F1']*dict_pll['dt'])):,4]+np.pi)%(2.*np.pi))-np.pi,'-.',label=r'$\phi_{0}-\phi_{4}$ mutual  vs freePLL')
		# plt.plot((t[-int(12*1.0/(dict_data['F1']*dict_pll['dt'])):]*dict_pll['dt']),((dict_data['phi'][-int(12*1.0/(dict_data['F1']*dict_pll['dt'])):,3]-dict_data['phi'][-int(12*1.0/(dict_data['F1']*dict_pll['dt'])):,4]+np.pi)%(2.*np.pi))-np.pi,'-.',label=r'$\phi_{3}-\phi_{4}$ entrain vs freePLL')
		# plt.plot(np.mean(dict_pll['transmission_delay']), ((dict_data['phi'][int(round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])),0]-dict_data['phi'][int(round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])),1]+np.pi)%(2.*np.pi))-np.pi, 'yo', ms=5)
		plt.axvspan(dict_data['t'][-int(5.5 * 1.0 / (dict_data['F1'] * dict_pll['dt']))], dict_data['t'][-1], color='b', alpha=0.3)
		plt.title(r'phases-differences between the clocks', fontdict=titlefont)
		plt.xlabel(r'$t\,[T_{\omega}]$', fontdict=labelfont, labelpad=labelpadxaxis)
		plt.ylabel(r'$\phi_k(t)-\phi_0(t)$', fontdict=labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)

		plt.legend(loc='upper right');
		plt.savefig('results/phaseRela-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					bbox_inches="tight")
		plt.savefig('results/phaseRela-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_phase_difference(dict_pll: dict, dict_net: dict, dict_data: dict):
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	fig8 = plt.figure(num=8, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig8.canvas.manager.set_window_title('phase configuration with respect to the phase of osci 0')
	fig8.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	if len(dict_data['phi'][:, 0]) > dict_pll['treshold_maxT_to_plot'] * (1.0 / (np.min(dict_pll['intrF']) * dict_pll['dt'])):  # (1.0/(np.min(dict_pll['intrF'])*dict_pll['dt'])) steps for one period
		for i in range(len(dict_data['phi'][0, :])):
			labelname = r'$\phi_{%i}$-$\phi_{0}$' % (i)
			plt.plot((dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):int(
				np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']) + dict_pll['treshold_maxT_to_plot'] * 1.0 / (dict_data['F1'] * dict_pll['dt'])):dict_pll['sampleFplot']] * dict_pll[
						  'dt']), ((dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):int(
				np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']) + dict_pll['treshold_maxT_to_plot'] * 1.0 / (dict_data['F1'] * dict_pll['dt'])):dict_pll['sampleFplot'], i] -
									dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):int(
										np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']) + dict_pll['treshold_maxT_to_plot'] * 1.0 / (dict_data['F1'] * dict_pll['dt'])):dict_pll[
														'sampleFplot'], 0] + np.pi) % (2. * np.pi)) - np.pi, label=labelname)  # math.fmod(phi[:,:], 2.*np.pi))
		plt.plot(np.mean(dict_pll['transmission_delay']), ((dict_data['phi'][int(round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])), 0] - dict_data['phi'][
			int(round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])), 1] + np.pi) % (2. * np.pi)) - np.pi, 'yo', ms=5)
		# plt.axvspan(t[-int(5.5*1.0/(dict_data['F1']*dict_pll['dt']))]*dict_pll['dt'], t[-1]*dict_pll['dt'], color='b', alpha=0.3)
		if dict_net['Nx'] * dict_net['Ny'] >= 3:
			plt.title(r'phase differences $\Delta\phi_{10}=%.4f$, $\Delta\phi_{20}=%.4f$  [rad]' % (np.mod(dict_data['phi'][-10][1] - dict_data['phi'][-10][0] + np.pi, 2.0 * np.pi) - np.pi,
																									np.mod(dict_data['phi'][-10][2] - dict_data['phi'][-10][0] + np.pi, 2.0 * np.pi) - np.pi),
					  fontdict=titlefont)
		else:
			plt.title(r'phase differences $\Delta\phi_{10}=%.4f$ [rad]' % (np.mod(dict_data['phi'][-10][1] - dict_data['phi'][-10][0] + np.pi, 2.0 * np.pi) - np.pi), fontdict=titlefont)
		plt.xlabel(r'$t\,[T_{\omega}]$', fontdict=labelfont, labelpad=labelpadxaxis)
		plt.ylabel(r'$\phi_k(t)-\phi_0(t)$', fontdict=labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)
		plt.legend(loc='upper right');
	else:
		shift2piWin = -np.pi / 2  # this controls how the [0, 2pi) interval is shifted
		for i in range(len(dict_data['phi'][0, :])):
			labelname = r'$\phi_{%i}$-$\phi_{0}$' % (i);
			plt.plot((dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']]),
					 ((dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], i] - dict_data['phi'][int(0.75 * np.round(
						 np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], 0] - shift2piWin) % (2 * np.pi)) + shift2piWin,
					 label=labelname)  # math.fmod(phi[:,:], 2.*np.pi)) , #int(np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])+25*1.0/(dict_data['F1']*dict_pll['dt']))
		plt.plot(np.mean(dict_pll['transmission_delay']), ((dict_data['phi'][int(round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])), 0] - dict_data['phi'][
			int(round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])), 1] - shift2piWin) % (2 * np.pi)) + shift2piWin, 'yo', ms=5)
		# plt.axvspan(t[-int(5.5*1.0/(dict_data['F1']*dict_pll['dt']))]*dict_pll['dt'], t[-1]*dict_pll['dt'], color='b', alpha=0.3)
		if dict_net['Nx'] * dict_net['Ny'] >= 3:
			plt.title(r'phase differences $\Delta\phi_{10}=%.4f$, $\Delta\phi_{20}=%.4f$  [rad]' % (np.mod(dict_data['phi'][-10][1] - dict_data['phi'][-10][0] + np.pi, 2.0 * np.pi) - np.pi,
																									np.mod(dict_data['phi'][-10][2] - dict_data['phi'][-10][0] + np.pi, 2.0 * np.pi) - np.pi),
					  fontdict=titlefont)
		else:
			plt.title(r'phase differences $\Delta\phi_{10}=%.4f$ [rad]' % (np.mod(dict_data['phi'][-10][1] - dict_data['phi'][-10][0] + np.pi, 2.0 * np.pi) - np.pi), fontdict=titlefont)
		plt.xlabel(r'$t\,[T_{\omega}]$', fontdict=labelfont, labelpad=labelpadxaxis)
		plt.ylabel(r'$\phi_k(t)-\phi_0(t)$', fontdict=labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)
		plt.legend(loc='upper right');
	plt.savefig('results/phaseConf-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				bbox_inches="tight")
	plt.savefig('results/phaseConf-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_clock_time_in_period_fractions(dict_pll: dict, dict_net: dict, dict_data: dict):
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	labeldict = {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$',
				 'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 = {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$', 'phi': r'$\phi$', 't': r'$t$',
				  'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color = ['blue', 'red', 'purple', 'cyan', 'green', 'yellow']  # 'magenta'
	linet = ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig9 = plt.figure(num=9, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig9.canvas.manager.set_window_title('clock time')  # plot the clocks' time
	fig9.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.plot(dict_data['t'], dict_data['clock_counter'], linewidth=1, linestyle=linet[0])

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict=labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'count $\frac{T}{2}$', fontdict=labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	# plt.legend(loc='upper right')

	plt.savefig('results/clockTime-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				bbox_inches="tight")
	plt.savefig('results/clockTime-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plotCtrlSigDny(dict_pll: dict, dict_net: dict, dict_data: dict):
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	color = ['blue', 'red', 'purple', 'cyan', 'green', 'yellow']  # 'magenta'
	linet = ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig10 = plt.figure(num=10, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig10.canvas.manager.set_window_title('time-series control voltage')  # plot the time evolution of the control signal
	fig10.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.plot(dict_data['t'], dict_data['ctrl'], linewidth=1, linestyle=linet[0])

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict=labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$V_\textrm{ctrl}(t)$', fontdict=labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	# plt.legend(loc='upper right')

	plt.savefig('results/ctrlSig-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				bbox_inches="tight")
	plt.savefig('results/ctrlSig-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_periodic_output_signal_from_phase(dict_pll: dict, dict_net: dict, dict_data: dict, plotEveryDt=1):
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	color = ['blue', 'red', 'purple', 'cyan', 'green', 'yellow']  # 'magenta'
	linet = ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig11 = plt.figure(num=11, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig11.canvas.manager.set_window_title('time-series signals')  # plot the time evolution of the control signal
	fig11.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	ax11 = fig11.add_subplot(211)

	for i in range(len(dict_data['phi'][0, :])):
		plt.plot((dict_data['t'][::plotEveryDt]), dict_pll['vco_out_sig'](dict_data['phi'][::plotEveryDt, i]), label='sig PLL%i' % (i))
		plt.ylabel(r'$s( \theta(t) )$', fontdict=labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)

	ax12 = fig11.add_subplot(212)

	for i in range(len(dict_data['phi'][0, :])):
		plt.plot((dict_data['t'][::plotEveryDt]), dict_pll['vco_out_sig'](dict_data['phi'][::plotEveryDt, i] / dict_pll['div']), label='sig PLL%i' % (i))
		plt.ylabel(r'$s( \theta(t)/ %i )$' % dict_pll['div'], fontdict=labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)

	plt.savefig('results/sig_and_divSig-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				bbox_inches="tight")
	plt.savefig('results/sig_and_divSig-t_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_instantaneous_freqs_vs_time_dependent_parameter(dict_pll: dict, dict_net: dict, dict_data: dict):
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	color = ['blue', 'red', 'purple', 'cyan', 'green', 'yellow']  # 'magenta'
	linet = ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	opacity_reverse_change = 0.35

	param_name = dict_net['special_case']  # 'timeDepInjectLockCoupStr', 'timeDepTransmissionDelay', 'timeDepChangeOfCoupStr', 'distanceDepTransmissionDelay'
	if param_name == 'timeDepTransmissionDelay':
		dyn_x_label = r'$\frac{\tau\omega}{2\pi}$'
		x_axis_scaling = np.mean(dict_pll['intrF'])
	elif param_name == 'timeDepChangeOfCoupStr':
		dyn_x_label = r'$\frac{2\pi K}{\omega}$'
		x_axis_scaling = np.mean(1.0 / dict_pll['intrF'])
	elif param_name == 'timeDepInjectLockCoupStr':
		dyn_x_label = r'$\frac{2\pi K}{\omega}$'
		x_axis_scaling = np.mean(1.0 / dict_pll['intrF'])
	elif param_name == 'timeDepChangeOfIntrFreq':
		if dict_data['only_change_freq_of_reference']:
			dyn_x_label = r'$\frac{\omega_R}{\bar{\omega}_{k\neq R}}$'
			x_axis_scaling = np.mean(1.0 / dict_pll['intrF'][1:])
		else:
			dyn_x_label = r'$\omega$'
			x_axis_scaling = 1

	y_axis_scaling = (2.0 * np.pi * np.mean(dict_pll['intrF']))

	fig12 = plt.figure(num=12, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig12.canvas.manager.set_window_title('instantaneous frequency as function of time-dependent parameter')
	fig12.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	if dict_net['typeOfTimeDependency'] == 'triangle':
		labelname = r'$\frac{\dot{\theta}_k(t)}{\omega}$ $\uparrow$'
		plt.plot(dict_data['timeDependentParameter'][0, 0:int(dict_net['index_max_value_time_dependent_parameter'][0])-1:dict_pll['sampleFplot']] * x_axis_scaling,
				 (np.diff(dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]), :], axis=0) / dict_pll['dt'])[::dict_pll['sampleFplot']] / y_axis_scaling,
				 'b', linestyle=linet[0], label=labelname)
		labelname = r'$\frac{\dot{\theta}_k(t)}{\omega} $\uparrow$'
		plt.plot(dict_data['timeDependentParameter'][0, int(dict_net['index_max_value_time_dependent_parameter'][0])+2::dict_pll['sampleFplot']] * x_axis_scaling,
				(np.diff(dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0])+1:, :], axis=0) / dict_pll['dt'])[::dict_pll['sampleFplot']] / y_axis_scaling,
				'c', linestyle=linet[1], alpha=opacity_reverse_change, label=labelname)
	else:
		plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0]) - 1] * x_axis_scaling, (np.diff(dict_data['phi'], axis=0) / dict_pll['dt']) / y_axis_scaling, linewidth=2, linestyle=linet[0])

	plt.xlabel(dyn_x_label, fontdict=labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$\frac{\dot{\theta}_k(t)}{\omega}$', fontdict=labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	# plt.legend(loc='upper right')

	plt.savefig('results/instFreq_vs_' + param_name + '%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				bbox_inches="tight")
	plt.savefig('results/instFreq_vs_' + param_name + '%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	if dict_pll['div'] != 1:
		fig1212 = plt.figure(num=1212, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
		fig1212.canvas.manager.set_window_title('instantaneous frequency of coupling signlas as function of time-dependent parameter')
		fig1212.set_size_inches(plot_size_inches_x, plot_size_inches_y)

		if dict_net['typeOfTimeDependency'] == 'triangle':
			plt.plot(dict_data['timeDependentParameter'][0, 0:int(dict_net['index_max_value_time_dependent_parameter'][0]) - dict_pll['sampleFplot']:dict_pll['sampleFplot']] * x_axis_scaling,
					 (np.diff(dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], :] / dict_pll['div'], axis=0) / dict_pll['dt']) / y_axis_scaling, 'b',
					 linestyle=linet[0])

			plt.plot(dict_data['timeDependentParameter'][0, int(dict_net['index_max_value_time_dependent_parameter'][0]) - dict_pll['sampleFplot']::dict_pll['sampleFplot']] * x_axis_scaling,
					 (np.diff(dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:-1:dict_pll['sampleFplot'], :] / dict_pll['div'], axis=0) / dict_pll['dt']) / y_axis_scaling,
					 'c', linestyle=linet[1], alpha=opacity_reverse_change)
		else:
			plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0]) - 1] * x_axis_scaling,
					 (np.diff(dict_data['phi'] / dict_pll['div'], axis=0) / dict_pll['dt']) / y_axis_scaling, 'b-')



		plt.xlabel(dyn_x_label, fontdict=labelfont, labelpad=labelpadxaxis)
		plt.ylabel(r'$\frac{\dot{\theta}^\textrm{HF}_k(t)}{v\,\omega}$', fontdict=labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)
		# plt.legend(loc='upper right')

		plt.savefig('results/instDivFreq_vs_' + param_name + '_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					bbox_inches="tight")
		plt.savefig('results/instDivFreq_vs_' + param_name + '_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def plot_order_parameter_vs_time_dependent_parameter_div_and_undiv(dict_pll: dict, dict_net: dict, dict_data: dict):
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	opacity_reverse_change = 0.35
	color = ['blue', 'red', 'purple', 'cyan', 'green', 'yellow']  # 'magenta'
	linet = ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	param_name = dict_net['special_case']  # 'timeDepInjectLockCoupStr', 'timeDepTransmissionDelay', 'timeDepChangeOfCoupStr', 'distanceDepTransmissionDelay'
	if param_name == 'timeDepTransmissionDelay':
		dyn_x_label = r'$\frac{\tau\omega}{2\pi}$'
		x_axis_scaling = np.mean(dict_pll['intrF'])
	elif param_name == 'timeDepChangeOfCoupStr':
		dyn_x_label = r'$\frac{2\pi K}{\omega}$'
		x_axis_scaling = np.mean(1.0 / dict_pll['intrF'])
	elif param_name == 'timeDepInjectLockCoupStr':
		dyn_x_label = r'$\frac{2\pi K}{\omega}$'
		x_axis_scaling = np.mean(1.0 / dict_pll['intrF'])
	elif param_name == 'timeDepChangeOfIntrFreq':
		if dict_data['only_change_freq_of_reference']:
			dyn_x_label = r'$\frac{\omega_R}{\bar{\omega}_{k\neq R}}$'
			x_axis_scaling = np.mean(1.0 / dict_pll['intrF'][1:])
		else:
			dyn_x_label = r'$\omega$'
			x_axis_scaling = 1

	fig13 = plt.figure(num=13, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig13.canvas.manager.set_window_title('order parameter as function of time-dependent parameter')  # time-series phases and phase-differences
	fig13.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	if dict_net['typeOfTimeDependency'] == 'triangle':
		labelname = r'R(t) $\uparrow$'
		plt.plot(dict_data['timeDependentParameter'][0, 0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot']] * x_axis_scaling,
				dict_data['order_parameter'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot']], 'b', linestyle=linet[0], label=labelname)
		labelname = r'R(t) $\downarrow$'
		plt.plot(dict_data['timeDependentParameter'][0, int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:-1:dict_pll['sampleFplot']] * x_axis_scaling,
				dict_data['order_parameter'][int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:-1:dict_pll['sampleFplot']], 'c', linestyle=linet[1], alpha=opacity_reverse_change, label=labelname)
	else:
		plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['order_parameter'][:])] * x_axis_scaling, dict_data['order_parameter'], 'b-')

	plt.xlabel(dyn_x_label, fontdict=labelfont, labelpad=labelpadxaxis)
	plt.ylabel(r'$R(t)$', fontdict=labelfont, labelpad=labelpadyaxis)
	plt.tick_params(axis='both', which='major', labelsize=tickSize)
	plt.legend(loc='upper right')

	plt.savefig('results/orderP_vs_' + param_name + '_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				bbox_inches="tight")
	plt.savefig('results/orderP_vs_' + param_name + '_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	if dict_pll['div'] != 1:
		fig1313 = plt.figure(num=1313, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
		fig1313.canvas.manager.set_window_title('order parameter of divided phases as a function of time-dependent parameter')  # time-series phases and phase-differences
		fig1313.set_size_inches(plot_size_inches_x, plot_size_inches_y)

		if dict_net['typeOfTimeDependency'] == 'triangle':
			labelname = r'R(t) $\uparrow$'
			plt.plot(dict_data['timeDependentParameter'][0, 0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot']] * x_axis_scaling,
					dict_data['order_parameter_divided_phases'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot']], 'b', linestyle=linet[0], label=labelname)
			labelname = r'R(t) $\downarrow$'
			plt.plot(dict_data['timeDependentParameter'][0, int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:-1:dict_pll['sampleFplot']] * x_axis_scaling,
					dict_data['order_parameter_divided_phases'][int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:-1:dict_pll['sampleFplot']], 'c',
					 linestyle=linet[1], alpha=opacity_reverse_change, label=labelname)
		else:
			plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['order_parameter_divided_phases'][:])] * x_axis_scaling, dict_data['order_parameter_divided_phases'], 'b-')

		plt.xlabel(dyn_x_label, fontdict=labelfont, labelpad=labelpadxaxis)
		plt.ylabel(r'$R_\textrm{div}(t)$', fontdict=labelfont, labelpad=labelpadyaxis)
		plt.tick_params(axis='both', which='major', labelsize=tickSize)
		plt.legend(loc='upper right')

		plt.savefig('results/orderP_divPhases_vs_' + param_name + '_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
			np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					bbox_inches="tight")
		plt.savefig('results/orderP_divPhases_vs_' + param_name + '_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
			np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_phase_differences_vs_time_dependent_parameter_divided_or_undivided(dict_pll: dict, dict_net: dict, dict_data: dict, plotlist: list = [],
																			phase_diff_wrap_to_interval: np.int = 1, phases_of_divided_signals: bool = True):
	plt.clf()

	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)
	division = 1
	x_axis_scaling = 1
	opacity_reverse_change = 0.35

	color = ['blue', 'red', 'purple', 'cyan', 'green', 'yellow']  # 'magenta'
	linet = ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	param_name = dict_net['special_case']  # 'timeDepInjectLockCoupStr', 'timeDepTransmissionDelay', 'timeDepChangeOfCoupStr', 'distanceDepTransmissionDelay'
	if param_name == 'timeDepTransmissionDelay':
		dyn_x_label = r'$\frac{\tau\omega}{2\pi}$'
		x_axis_scaling = np.mean(dict_pll['intrF'])
	elif param_name == 'timeDepChangeOfCoupStr':
		dyn_x_label = r'$\frac{2\pi K}{\omega}$'
		x_axis_scaling = np.mean(1.0 / dict_pll['intrF'])
	elif param_name == 'timeDepInjectLockCoupStr':
		dyn_x_label = r'$\frac{2\pi K}{\omega}$'
		x_axis_scaling = np.mean(1.0 / dict_pll['intrF'])
	elif param_name == 'timeDepChangeOfIntrFreq':
		if dict_data['only_change_freq_of_reference']:
			dyn_x_label = r'$\frac{\omega_R}{\bar{\omega}_{k\neq R}}$'
			x_axis_scaling = np.mean(1.0 / dict_pll['intrF'][1:])
		else:
			dyn_x_label = r'$\omega$'
			x_axis_scaling = 1

	if phase_diff_wrap_to_interval == 1:  	# plot phase-differences in [-pi, pi] interval
		shift2piWin = np.pi
	elif phase_diff_wrap_to_interval == 2:  # plot phase-differences in [-pi/2, 3*pi/2] interval
		shift2piWin = 0.5 * np.pi
	elif phase_diff_wrap_to_interval == 3:  # plot phase-differences in [0, 2*pi] interval
		shift2piWin = 0.0

	if phases_of_divided_signals and dict_pll['div'] != 1:
		division = dict_pll['div']
		add_to_savename_phase = 'div'
		dyn_y_label = r'$\frac{\Delta\theta_{k0}(t)}{v}$'
	else:
		division = 1
		add_to_savename_phase = ''
		dyn_y_label = r'$\Delta\theta_{k0}(t)$'

	fig14 = plt.figure(num=14, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig14.canvas.manager.set_window_title('phase-differences (' + add_to_savename_phase + ') as function of time-dependent parameter')  # frequency and order parameter
	fig14.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	if not dict_net['topology'] == 'compareEntrVsMutual':
		if not dict_net['Nx'] * dict_net['Ny'] == 2:
			if not plotlist:
				for i in range(len(dict_data['phi'][0, :])):
					if dict_net['typeOfTimeDependency'] == 'triangle':
						labelname = r'$\frac{\phi_{%i}-\phi_{0}}{v}$ $\uparrow$' % (i)
						plt.plot(dict_data['timeDependentParameter'][0, 0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot']] * x_axis_scaling,
								((dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], i] / division -
								dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin,
								label=labelname, linestyle=linet[0])
						labelname = r'$\frac{\phi_{%i}-\phi_{0}}{v}$ $\downarrow$' % (i)
						plt.plot(dict_data['timeDependentParameter'][0, int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:-1:dict_pll['sampleFplot']] * x_axis_scaling,
								((dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:-1:dict_pll['sampleFplot'], i] / division -
								dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:-1:dict_pll['sampleFplot'], 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin,
								label=labelname, linestyle=linet[1], alpha=opacity_reverse_change)
					else:
						labelname = r'$\frac{\phi_{%i}-\phi_{0}}{v}$' % (i)
						plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0])] * x_axis_scaling,
									((dict_data['phi'][:, i] / division - dict_data['phi'][:, 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname)
					# int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']
			else:
				for i in plotlist:
					# labelname = r'$\frac{\phi_{%i}-\phi_{0}}{v}$' % (i)
					# plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0])] * x_axis_scaling,
					# 		 ((dict_data['phi'][:, i] / division - dict_data['phi'][:, 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname)
					if dict_net['typeOfTimeDependency'] == 'triangle':
						labelname = r'$\frac{\phi_{%i}-\phi_{0}}{v}$ $\uparrow$' % (i)
						plt.plot(dict_data['timeDependentParameter'][0, 0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot']] * x_axis_scaling,
								((dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], i] / division -
								dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin,
								label=labelname, linestyle=linet[0])
						labelname = r'$\frac{\phi_{%i}-\phi_{0}}{v}$ $\downarrow$' % (i)
						plt.plot(dict_data['timeDependentParameter'][0, int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:-1:dict_pll['sampleFplot']] * x_axis_scaling,
								((dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:-1:dict_pll['sampleFplot'], i] / division -
								dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:-1:dict_pll['sampleFplot'], 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin,
								label=labelname, linestyle=linet[1], alpha=opacity_reverse_change)
					else:
						labelname = r'$\frac{\phi_{%i}-\phi_{0}}{v}$' % (i)
						plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0])] * x_axis_scaling,
									((dict_data['phi'][:, i] / division - dict_data['phi'][:, 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname)
		else:
			if dict_net['typeOfTimeDependency'] == 'triangle':
				labelname = r'$\frac{\phi_{1}-\phi_{0}}{v}$ $\uparrow$'
				plt.plot(dict_data['timeDependentParameter'][0, 0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot']] * x_axis_scaling,
						 ((dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], 1] / division -
						   dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin,
						 label=labelname, linestyle=linet[0])
				labelname = r'$\frac{\phi_{1}-\phi_{0}}{v}$ $\downarrow$'
				plt.plot(dict_data['timeDependentParameter'][0, int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:-1:dict_pll['sampleFplot']] * x_axis_scaling,
						 ((dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:-1:dict_pll['sampleFplot'], 1] / division -
						   dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:-1:dict_pll['sampleFplot'], 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin,
						 label=labelname, linestyle=linet[1], alpha=opacity_reverse_change)
			else:
				labelname = r'$\frac{\phi_{1}-\phi_{0}}{v}$'
				plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0])] * x_axis_scaling,
						 ((dict_data['phi'][:, 1] / division - dict_data['phi'][:, 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname)
			# labelname = r'$\frac{\phi_{1}$-$\phi_{0}}{v}$'
			# plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0])] * x_axis_scaling,
			# 		 ((dict_data['phi'][:, 1] / division - dict_data['phi'][:, 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname)
	else:
		plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0])] * x_axis_scaling,
				((dict_data['phi'][:, 0] / division - dict_data['phi'][:, 1] / division + shift2piWin) % (2. * np.pi)) - shift2piWin, '-', linewidth=2,
				label=r'$\frac{\phi_{0}-\phi_{1}}{v}$ mutual')
		plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0])] * x_axis_scaling,
				((dict_data['phi'][:, 3] / division - dict_data['phi'][:, 2] / division + shift2piWin) % (2. * np.pi)) - shift2piWin, '--', linewidth=2,
				label=r'$\frac{\phi_{3}-\phi_{2}}{v}$ entrain')

	plt.xlabel(dyn_x_label, fontdict=labelfont, labelpad=-5)
	plt.ylabel(dyn_y_label, rotation=90, fontdict=labelfont, labelpad=40)
	plt.tick_params(axis='both', which='major', labelsize=tickSize, pad=1)
	plt.xlim([dict_net['min_max_rate_timeDepPara'][0] * x_axis_scaling, dict_net['min_max_rate_timeDepPara'][1] * x_axis_scaling])
	# ax012.set_ylim([-np.pi, np.pi])
	plt.legend(loc='upper right')
	plt.grid()

	plt.savefig('results/freq_phaseDiff' + add_to_savename_phase + '_vs_' + param_name + '_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/freq_phaseDiff' + add_to_savename_phase + '_vs_' + param_name + '_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def plot_inst_frequency_and_phase_difference_vs_time_dependent_parameter_divided_or_undivided(dict_pll: dict, dict_net: dict, dict_data: dict, phases_of_divided_signals: bool = True,
																			frequency_of_divided_signals: bool = True, plotlist: list = [], phase_diff_wrap_to_interval: np.int = 1,
																		 		ylim_percent_of_min_val: np.float = 0.995, ylim_percent_of_max_val: np.float = 1.005):
	# clear the figure in case more than one is to be plotted
	plt.clf()
	# set to 1 if plotting in [-pi, +pi) and to 2 if plotting in [-pi/2, 3pi/2] or to 3 if phase differences to be plotted in [0, 2pi)
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)
	division = 1
	x_axis_scaling = 1
	opacity_reverse_change = 0.35

	labeldict = {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$', 'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 = {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$', 'phi': r'$\phi$', 't': r'$t$', 'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color = ['blue', 'red', 'purple', 'cyan', 'green', 'yellow']  # 'magenta'
	linet = ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	param_name = dict_net['special_case']  # 'timeDepInjectLockCoupStr', 'timeDepTransmissionDelay', 'timeDepChangeOfCoupStr', 'distanceDepTransmissionDelay'
	if param_name == 'timeDepTransmissionDelay':
		dyn_x_label = r'$\frac{\tau\omega}{2\pi}$'
		x_axis_scaling = np.mean(dict_pll['intrF'])
	elif param_name == 'timeDepChangeOfCoupStr':
		dyn_x_label = r'$\frac{2\pi K}{\omega}$'
		x_axis_scaling = np.mean(1.0 / dict_pll['intrF'])
	elif param_name == 'timeDepInjectLockCoupStr':
		dyn_x_label = r'$\frac{2\pi K}{\omega}$'
		x_axis_scaling = np.mean(1.0 / dict_pll['intrF'])
	elif param_name == 'timeDepChangeOfIntrFreq':
		if dict_data['only_change_freq_of_reference']:
			dyn_x_label = r'$\frac{\omega_R}{\bar{\omega}_{k\neq R}}$'
			x_axis_scaling = np.mean(1.0 / dict_pll['intrF'][1:])
		else:
			dyn_x_label = r'$\omega$'
			x_axis_scaling = 1

	if phases_of_divided_signals and dict_pll['div'] != 1:
		#fig141.canvas.manager.set_window_title('frequency and phase-differences (divided) vs time-dependent parameter')  # frequency and order parameter
		y_label_name_1 = r'$\frac{\Delta\theta_{k0}(t)}{v}$'
		division = dict_pll['div']
		add_to_savename_phase = 'div'
	else:
		#fig141.canvas.manager.set_window_title('frequency and phase-differences (undivided) vs time-dependent parameter')  # frequency and order parameter
		y_label_name_1 = r'$\Delta\theta_{k0}(t)$'
		division = dict_pll['div']
		add_to_savename_phase = ''
	# whether or not the frequency is plotted from the phases of the divided signal
	# print('dict_data[*phi*]', dict_data['phi'])
	# print('dict_data[*phi*][0,:]', dict_data['phi'][0, :])
	# print('dict_data[*phi*][:,0]', dict_data['phi'][:, 0])
	if frequency_of_divided_signals:
		phidot = np.diff(dict_data['phi'] / dict_pll['div'], axis=0) / dict_pll['dt']
	else:
		phidot = np.diff(dict_data['phi'], axis=0) / dict_pll['dt']
	# manage the normalization with respect to the mean intrinsic frequency, excluding the reference in the entrainment case
	if isinstance(dict_pll['intrF'], list) or isinstance(dict_pll['intrF'], np.ndarray):
		if 'entrain' in dict_net['topology']: 												#if dict_net['topology'].find('entrain') != -1:
			phidot = phidot / (2.0 * np.pi * np.mean(dict_pll['intrF'][1:]))
		else:
			phidot = phidot / (2.0 * np.pi * np.mean(dict_pll['intrF']))
		if frequency_of_divided_signals:
			ylabelname = r'$\frac{\dot{\theta}_k(t)}{v\,\bar{\omega}_k}$'
			add_to_savename_freq = 'div'
		else:
			ylabelname = r'$\frac{\dot{\theta}_k(t)}{\bar{\omega}_k}$'
			add_to_savename_freq = ''
	else:
		if not dict_pll['intrF'] == 0:
			phidot = phidot / (2.0 * np.pi * dict_pll['intrF'])
			if frequency_of_divided_signals:
				ylabelname = r'$\frac{\dot{\theta}_k(t)}{v\,\omega}$'
				add_to_savename_freq = 'div'
			else:
				ylabelname = r'$\frac{\dot{\theta}_k(t)}{\omega}$'
				add_to_savename_freq = ''
		else:
			if frequency_of_divided_signals:
				ylabelname = r'$\frac{\dot{\theta}_k(t)}{v}$'
				add_to_savename_freq = 'div'
			else:
				ylabelname = r'$\dot{\theta}_k(t)$'
				add_to_savename_freq = ''

	# for triangle time dependence separate the growing and shrinking port
	if dict_net['typeOfTimeDependency'] == 'triangle':
		phidot_increase = phidot[0:int(dict_net['index_max_value_time_dependent_parameter'][0]), :]
		phidot_decrease = phidot[int(dict_net['index_max_value_time_dependent_parameter'][0]) + 1:, :]

	fig141 = plt.figure(num=141, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig141.canvas.manager.set_window_title('frequency (' + add_to_savename_freq + ') and phase-differences (' + add_to_savename_phase + ') vs time-dependent parameter')  # frequency and order parameter
	fig141.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	ax011 = fig141.add_subplot(211)
	# plt.plot(dict_data['t'][dict_net['max_delay_steps']-1], phidot[int(dict_net['max_delay_steps'])-1,0]+0.001,'go')
	if not plotlist:
		# print('phidot_increase', phidot_increase)
		# print('phidot_increase[:, 0]', phidot_increase[:, 0])
		# print('phidot_increase[0, :]', phidot_increase[0, :])
		# sys.exit()
		if dict_net['typeOfTimeDependency'] == 'triangle':
			plt.plot(dict_data['timeDependentParameter'][0, 0:int(dict_net['index_max_value_time_dependent_parameter'][0])] * x_axis_scaling, #:dict_pll['sampleFplot']
							phidot_increase[:, ::dict_pll['sampleFplot']], linewidth=2, linestyle=linet[0])
			plt.plot(dict_data['timeDependentParameter'][0, int(dict_net['index_max_value_time_dependent_parameter'][0])+2:] * x_axis_scaling, #:dict_pll['sampleFplot']
							phidot_decrease[:, ::dict_pll['sampleFplot']], linewidth=2, linestyle=linet[1], alpha=opacity_reverse_change)
		else:
			plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0]):dict_pll['sampleFplot']] * x_axis_scaling,
							phidot[::dict_pll['sampleFplot'], :], linewidth=2, linestyle=linet[0])
	else:
		if dict_net['typeOfTimeDependency'] == 'triangle':
			plt.plot(dict_data['timeDependentParameter'][0, 0:int(dict_net['index_max_value_time_dependent_parameter'][0])-1] * x_axis_scaling,
					phidot_increase[plotlist, 0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot']], linewidth=2, linestyle=linet[0])
			plt.plot(dict_data['timeDependentParameter'][0, int(dict_net['index_max_value_time_dependent_parameter'][0])+2:] * x_axis_scaling,
					phidot_decrease[plotlist, int(dict_net['index_max_value_time_dependent_parameter'][0])+1::dict_pll['sampleFplot']], linewidth=2, linestyle=linet[1], alpha=opacity_reverse_change)
		else:
			plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0]):dict_pll['sampleFplot']] * x_axis_scaling,
					 phidot[::dict_pll['sampleFplot'], plotlist], linewidth=2, linestyle=linet[0])

	plt.xlabel(dyn_x_label, fontdict=labelfont, labelpad=labelpadxaxis)
	plt.ylabel(ylabelname, fontdict=labelfont, labelpad=labelpadyaxis)
	ax011.tick_params(axis='both', which='major', labelsize=tickSize, pad=1)
	ax011.set_xlim([dict_net['min_max_rate_timeDepPara'][0] * x_axis_scaling, dict_net['min_max_rate_timeDepPara'][1] * x_axis_scaling])

	mean_freq_ts = np.mean(phidot[int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']])
	max_freq_ts = np.max(phidot[int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']])
	min_freq_ts = np.min(phidot[int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']])

	ax011.set_ylim([ylim_percent_of_min_val * min_freq_ts, ylim_percent_of_max_val * max_freq_ts])
	plt.grid()

	ax012 = fig141.add_subplot(212)

	if phase_diff_wrap_to_interval == 1:  # plot phase-differences in [-pi, pi] interval
		shift2piWin = np.pi
	elif phase_diff_wrap_to_interval == 2:  # plot phase-differences in [-pi/2, 3*pi/2] interval
		shift2piWin = 0.5 * np.pi
	elif phase_diff_wrap_to_interval == 3:  # plot phase-differences in [0, 2*pi] interval
		shift2piWin = 0.0

	if len(dict_data['timeDependentParameter'][:, 0]) > 1:
		print('\n\nNOTE: The time-dependent parameter is set different over the connections or properties of the PLLs, we plot however against the parameter associated to k=0! Think of how to generalize this!')

	if not dict_net['topology'] == 'compareEntrVsMutual':
		if not dict_net['Nx'] * dict_net['Ny'] == 2:
			if not plotlist:
				if dict_net['typeOfTimeDependency'] == 'triangle':
					for i in range(len(dict_data['phi'][0, :])):
						labelname = r'$\phi_{%i}$-$\phi_{0}$ $\uparrow$' % (i)
						plt.plot(dict_data['timeDependentParameter'][0, 0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot']] * x_axis_scaling,
								((dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], i] / division -
								dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin,
								label=labelname, linestyle=linet[0])
						labelname = r'$\phi_{%i}$-$\phi_{0}$ $\downarrow$' % (i)
						plt.plot(dict_data['timeDependentParameter'][0, int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot']] * x_axis_scaling,
								((dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot'], i] / division -
								dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot'], 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin,
								label=labelname, linestyle=linet[1], alpha=opacity_reverse_change)
				else:
					for i in range(len(dict_data['phi'][0, :])):
						labelname = r'$\phi_{%i}$-$\phi_{0}$' % (i)
						plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0]):dict_pll['sampleFplot']] * x_axis_scaling,
								 ((dict_data['phi'][::dict_pll['sampleFplot'], i]/division - dict_data['phi'][::dict_pll['sampleFplot'], 0]/division + shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname)
			else:
				for i in plotlist:											# directly run index i over the elements of plotlist (integer)
					if dict_net['typeOfTimeDependency'] == 'triangle':
						labelname = r'$\phi_{%i}$-$\phi_{0}$' % (i)
						plt.plot(dict_data['timeDependentParameter'][0, 0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot']] * x_axis_scaling,
								((dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], i] / division -
								dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin,
								label=labelname, linestyle=linet[0])
						plt.plot(dict_data['timeDependentParameter'][ii, int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot']] * x_axis_scaling,
								((dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot'], i] / division -
								dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot'], 0] / division + shift2piWin) % (2 * np.pi)) - shift2piWin,
								label=labelname, linestyle=linet[1], alpha=opacity_reverse_change)
					else:
						labelname = r'$\phi_{%i}$-$\phi_{0}$' % (i)
						plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0]):dict_pll['sampleFplot']] * x_axis_scaling,
								 ((dict_data['phi'][::dict_pll['sampleFplot'], i]/division - dict_data['phi'][::dict_pll['sampleFplot'], 0]/division + shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname)
		else:
			labelname = r'$\phi_{1}$-$\phi_{0}$'
			if dict_net['typeOfTimeDependency'] == 'triangle':
				plt.plot(dict_data['timeDependentParameter'][0, int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot']] * x_axis_scaling,
						((dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], 1] / division -
						dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], 0] / division +
						shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname, linestyle=linet[0])
				plt.plot(dict_data['timeDependentParameter'][0, int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot']] * x_axis_scaling,
						((dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot'], 1] / division -
						dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot'], 0] / division +
						shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname, linestyle=linet[1], alpha=opacity_reverse_change)
			else:
				plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0]):dict_pll['sampleFplot']] * x_axis_scaling,
					 ((dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], 1]/division - dict_data['phi'][int(0.75 * np.round(
						 np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], 0]/division + shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname)
	else:
		if dict_net['typeOfTimeDependency'] == 'triangle':
			# print('Needs to be implemented!')
			# sys.exit()
			plt.plot(dict_data['timeDependentParameter'][0, 0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot']] * x_axis_scaling,
					((dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], 0] / division -
					dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], 1] / division + shift2piWin) % (2. * np.pi)) - shift2piWin,
					linewidth=2, label=r'$\phi_{0}-\phi_{1}$ mutual', linestyle=linet[0])
			plt.plot(dict_data['timeDependentParameter'][0, int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot']] * x_axis_scaling,
					((dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot'], 0] / division -
					dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot'], 1] / division + shift2piWin) % (2. * np.pi)) - shift2piWin,
					linewidth=2, label=r'$\phi_{0}-\phi_{1}$ mutual', linestyle=linet[1], alpha=opacity_reverse_change)

			plt.plot(dict_data['timeDependentParameter'][0, 0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot']] * x_axis_scaling,
					((dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], 3] / division -
					dict_data['phi'][0:int(dict_net['index_max_value_time_dependent_parameter'][0]):dict_pll['sampleFplot'], 2] / division + shift2piWin) % (2. * np.pi)) - shift2piWin,
					linewidth=2, label=r'$\phi_{3}-\phi_{2}$ entrain', linestyle=linet[2])
			plt.plot(dict_data['timeDependentParameter'][0, int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot']] * x_axis_scaling,
					((dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot'], 3] / division -
					dict_data['phi'][int(dict_net['index_max_value_time_dependent_parameter'][0])+1:-1:dict_pll['sampleFplot'], 2] / division + shift2piWin) % (2. * np.pi)) - shift2piWin,
					linewidth=2, label=r'$\phi_{3}-\phi_{2}$ entrain', linestyle=linet[3], alpha=opacity_reverse_change)
		else:
			plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0]):dict_pll['sampleFplot']] * x_axis_scaling,
					((dict_data['phi'][::dict_pll['sampleFplot'], 0]/division - dict_data['phi'][::dict_pll['sampleFplot'], 1]/division + shift2piWin) % (2. * np.pi)) - shift2piWin, '-', linewidth=2,
					label=r'$\phi_{0}-\phi_{1}$ mutual')
			plt.plot(dict_data['timeDependentParameter'][0, 0:len(dict_data['phi'][:, 0]):dict_pll['sampleFplot']] * x_axis_scaling,
					((dict_data['phi'][::dict_pll['sampleFplot'], 3]/division - dict_data['phi'][::dict_pll['sampleFplot'], 2]/division + shift2piWin) % (2. * np.pi)) - shift2piWin, '--', linewidth=2,
					label=r'$\phi_{3}-\phi_{2}$ entrain')
	# plt.plot((t[int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot']]*dict_pll['dt']),((dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],0]-dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],5]+np.pi)%(2.*np.pi))-np.pi,'-',linewidth=2,label=r'$\phi_{0}-\phi_{5}$ mutual  vs freeRef')
	# plt.plot((t[int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot']]*dict_pll['dt']),((dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],3]-dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],5]+np.pi)%(2.*np.pi))-np.pi,'--',linewidth=2,label=r'$\phi_{3}-\phi_{5}$ entrain vs freeRef')
	# plt.plot((t[int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot']]*dict_pll['dt']),((dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],0]-dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],4]+np.pi)%(2.*np.pi))-np.pi,'-.',linewidth=2,label=r'$\phi_{0}-\phi_{4}$ mutual  vs freePLL')
	# plt.plot((t[int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot']]*dict_pll['dt']),((dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],3]-dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],4]+np.pi)%(2.*np.pi))-np.pi,'-.',linewidth=2,label=r'$\phi_{3}-\phi_{4}$ entrain vs freePLL')
	# plt.plot(np.mean(dict_pll['transmission_delay']), ((dict_data['phi'][int(round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])),0]-dict_data['phi'][int(round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])),1]+np.pi)%(2.*np.pi))-np.pi, 'yo', ms=5)
	# plt.axvspan(dict_data['t'][-int(5.5*1.0/(dict_pll['intrF']*dict_pll['dt']))], dict_data['t'][-1], color='b', alpha=0.3)

	plt.xlabel(dyn_x_label, fontdict=labelfont, labelpad=-5)
	plt.ylabel(y_label_name_1, rotation=90, fontdict=labelfont, labelpad=40)
	ax012.tick_params(axis='both', which='major', labelsize=tickSize, pad=1)
	ax012.set_xlim([dict_net['min_max_rate_timeDepPara'][0] * x_axis_scaling, dict_net['min_max_rate_timeDepPara'][1] * x_axis_scaling])
	# ax012.set_ylim([-np.pi, np.pi])
	plt.legend(loc='upper right')
	plt.grid()

	plt.savefig('results/freq_' + add_to_savename_freq + '_phaseDiff_' + add_to_savename_phase + '_vs_' + param_name + '_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (np.mean(dict_pll['coupK']),
				np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/freq_' + add_to_savename_freq + '_phaseDiff_' + add_to_savename_phase + '_vs_' + param_name + '_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (np.mean(dict_pll['coupK']),
				np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_phase_relations_of_divided_signal(dict_pll: dict, dict_net: dict, dict_data: dict, plotlist=[], phase_diff_wrap_to_interval=1):

	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	fig15 = plt.figure(num=15, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig15.canvas.manager.set_window_title('time-series phase-differences (divided)')  # time-series phases and phase-differences
	fig15.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	if phase_diff_wrap_to_interval == 1:  # plot phase-differences in [-pi, pi] interval
		shift2piWin = np.pi
	elif phase_diff_wrap_to_interval == 2:  # plot phase-differences in [-pi/2, 3*pi/2] interval
		shift2piWin = 0.5 * np.pi
	elif phase_diff_wrap_to_interval == 3:  # plot phase-differences in [0, 2*pi] interval
		shift2piWin = 0.0

	if not dict_net['topology'] == 'compareEntrVsMutual':
		if not dict_net['Nx'] * dict_net['Ny'] == 2:
			if not plotlist:
				for i in range(len(dict_data['phi'][0, :])):
					labelname = r'$\frac{\phi_{%i}-\phi_{0}}{v}$' % (i)
					plt.plot((dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']]),
							 ((dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], i]/dict_pll['div'] - dict_data['phi'][int(0.75 * np.round(
								 np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], 0]/dict_pll['div'] + shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname)
			else:
				for i in plotlist:
					labelname = r'$\frac{\phi_{%i}-\phi_{0}}{v}$' % (i)
					plt.plot((dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']]),
							 ((dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], i]/dict_pll['div'] - dict_data['phi'][int(0.75 * np.round(
								 np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], 0]/dict_pll['div'] + shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname)
		else:
			labelname = r'$\frac{\phi_{1}$-$\phi_{0}}{v}$'
			plt.plot((dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']]),
					 ((dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], 1]/dict_pll['div'] - dict_data['phi'][int(0.75 * np.round(
						 np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], 0]/dict_pll['div'] + shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname)
	else:
		plt.plot(dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot']],
				 ((dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot'], 0]/dict_pll['div'] - dict_data['phi'][int(0.75 * np.round(
					 np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot'], 1]/dict_pll['div'] + shift2piWin) % (2. * np.pi)) - shift2piWin, '-', linewidth=2,
				 label=r'$\frac{\phi_{0}-\phi_{1}}{v}$ mutual')
		plt.plot(dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot']],
				 ((dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot'], 3]/dict_pll['div'] - dict_data['phi'][int(0.75 * np.round(
					 np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot'], 2]/dict_pll['div'] + shift2piWin) % (2. * np.pi)) - shift2piWin, '--', linewidth=2,
				 label=r'$\frac{\phi_{3}-\phi_{2}}{v}$ entrain')

	plt.xlabel(r'$\omega t/2\pi$', fontdict=labelfont, labelpad=-5)
	plt.ylabel(r'$\frac{\Delta\theta_{k0}(t)}{v}$', rotation=90, fontdict=labelfont, labelpad=40)
	plt.tick_params(axis='both', which='major', labelsize=tickSize, pad=1)
	plt.xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))], dict_data['t'][-1]])
	# ax012.set_ylim([-np.pi, np.pi])
	plt.legend(loc='upper right')
	plt.grid()

	plt.savefig('results/freq_phaseDiff_dividedSignal_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/freq_phaseDiff_dividedSignal_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	return None



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def plotPhasesAndPhaseRelations_cutAxis(dict_pll: dict, dict_net: dict, dict_data: dict):
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	fig16 = plt.figure(num=16, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig16.canvas.manager.set_window_title('time-series phases and phase-differences')  # time-series phases and phase-differences
	fig16.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def deltaThetaDot_vs_deltaTheta(dict_pll, dict_net, deltaTheta, deltaThetaDot, color, alpha):
	# dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)
	fig17 = plt.figure(num=17, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig17.canvas.manager.set_window_title('time-series phases and phase-differences')  # time-series phases and phase-differences

	plt.plot(deltaTheta, deltaThetaDot, '-', color=color, alpha=alpha)  # plot trajectory
	plt.plot(deltaTheta[0], deltaThetaDot[0], 'o', color=color, alpha=alpha)  # plot initial dot
	plt.plot(deltaTheta[-1], deltaThetaDot[-1], 'x', color=color, alpha=alpha)  # plot final state cross

	fig18 = plt.figure(num=18, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig18.canvas.manager.set_window_title('time-series phases and phase-differences')  # time-series phases and phase-differences
	fig18.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	plt.plot(deltaTheta[0], deltaThetaDot[0], 'o', color=color, alpha=alpha)  # plot initial dot

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def plot_inst_frequency_and_phase_difference(dict_pll: dict, dict_net: dict, dict_algo: dict, dict_data: dict, phases_of_divided_signals: bool = False, plotlist: list = [],
											 						phase_diff_wrap_to_interval: np.int = 1, ylim_percent_of_min_val: np.float = 0.995, ylim_percent_of_max_val: np.float = 1.005):
	# set to 1 if plotting in [-pi, +pi) and to 2 if plotting in [-pi/2, 3pi/2] or to 3 if phase differences to be plotted in [0, 2pi)

	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	labeldict = {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$', 'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 = {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$', 'phi': r'$\phi$', 't': r'$t$', 'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color = ['blue', 'red', 'purple', 'cyan', 'green', 'yellow']  # 'magenta'
	linet = ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig19 = plt.figure(num=19, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	if phases_of_divided_signals and dict_pll['div'] != 1:
		fig19.canvas.manager.set_window_title('time-series frequency and phase-differences (divided)')  # frequency and order parameter
		y_label_name_1 = r'$\frac{\Delta\theta_{k0}(t)}{v}$'
		division = dict_pll['div']
	else:
		fig19.canvas.manager.set_window_title('time-series frequency and phase-differences (undivided)')  # frequency and order parameter
		y_label_name_1 = r'$\Delta\theta_{k0}(t)$'
		division = 1
	fig19.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	ax011 = fig19.add_subplot(211)

	plt.axvspan(dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))],
				dict_data['t'][int(1.0 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))], color='b', alpha=0.25)
	phidot = np.diff(dict_data['phi'], axis=0) / dict_pll['dt']

	if isinstance(dict_pll['intrF'], list) or isinstance(dict_pll['intrF'], np.ndarray):
		if dict_net['topology'].find('entrain') != -1:
			phidot = phidot / (2.0 * np.pi * np.mean(dict_pll['intrF'][1:]))
		else:
			phidot = phidot / (2.0 * np.pi * np.mean(dict_pll['intrF']))
		ylabelname = r'$\frac{\dot{\theta}_k(t)}{\bar{\omega}_k}$'
	else:
		if not dict_pll['intrF'] == 0:
			phidot = phidot / (2.0 * np.pi * dict_pll['intrF'])
			ylabelname = r'$\frac{\dot{\theta}_k(t)}{\omega}$'
		else:
			ylabelname = r'$\dot{\theta}_k(t)$'

	if not plotlist:
		plt.plot(dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot']],
				 phidot[int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']], linewidth=2, linestyle=linet[0])
	else:
		plt.plot(dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot']],
				 phidot[int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], plotlist], linewidth=2, linestyle=linet[0])
	# plt.plot(dict_data['t'][dict_net['max_delay_steps']-1], phidot[int(dict_net['max_delay_steps'])-1,0]+0.001,'go')

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict=labelfont, labelpad=labelpadxaxis)
	plt.ylabel(ylabelname, fontdict=labelfont, labelpad=labelpadyaxis)
	if dict_algo['parameter_space_sweeps'] == 'two_parameter_sweep':
		plt.title(r'%s $= %0.2f$, %s $= %0.2f$' % (dict_algo['param_id_0'].replace('_', ' '), dict_pll['intrF'][0], dict_algo['param_id_1'].replace('_', ' '), dict_pll['transmission_delay']))
	ax011.tick_params(axis='both', which='major', labelsize=tickSize, pad=1)
	ax011.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))], dict_data['t'][-1]])

	mean_freq_ts = np.mean(phidot[int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']])
	max_freq_ts = np.max(phidot[int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']])
	min_freq_ts = np.min(phidot[int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']])

	ax011.set_ylim([ylim_percent_of_min_val * min_freq_ts, ylim_percent_of_max_val * max_freq_ts])
	plt.grid()

	ax012 = fig19.add_subplot(212)

	if phase_diff_wrap_to_interval == 1:  # plot phase-differences in [-pi, pi] interval
		shift2piWin = np.pi
	elif phase_diff_wrap_to_interval == 2:  # plot phase-differences in [-pi/2, 3*pi/2] interval
		shift2piWin = 0.5 * np.pi
	elif phase_diff_wrap_to_interval == 3:  # plot phase-differences in [0, 2*pi] interval
		shift2piWin = 0.0

	if not dict_net['topology'] == 'compareEntrVsMutual':
		if not dict_net['Nx'] * dict_net['Ny'] == 2:
			if not plotlist:
				for i in range(len(dict_data['phi'][0, :])):
					labelname = r'$\phi_{%i}$-$\phi_{0}$' % (i)
					plt.plot((dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']]),
							 ((dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], i]/division - dict_data['phi'][int(0.75 * np.round(
								 np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], 0]/division + shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname)
			else:
				for i in plotlist:
					labelname = r'$\phi_{%i}$-$\phi_{0}$' % (i)
					plt.plot((dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']]),
							 ((dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], i]/division - dict_data['phi'][int(0.75 * np.round(
								 np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], 0]/division + shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname)
		else:
			labelname = r'$\phi_{1}$-$\phi_{0}$'
			plt.plot((dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']]),
					 ((dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], 1]/division - dict_data['phi'][int(0.75 * np.round(
						 np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], 0]/division + shift2piWin) % (2 * np.pi)) - shift2piWin, label=labelname)
	else:
		plt.plot(dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot']],
				 ((dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot'], 0]/division - dict_data['phi'][int(0.75 * np.round(
					 np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot'], 1]/division + shift2piWin) % (2. * np.pi)) - shift2piWin, '-', linewidth=2,
				 label=r'$\phi_{0}-\phi_{1}$ mutual')
		plt.plot(dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot']],
				 ((dict_data['phi'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot'], 3]/division - dict_data['phi'][int(0.75 * np.round(
					 np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot'], 2]/division + shift2piWin) % (2. * np.pi)) - shift2piWin, '--', linewidth=2,
				 label=r'$\phi_{3}-\phi_{2}$ entrain')
	# plt.plot((t[int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot']]*dict_pll['dt']),((dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],0]-dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],5]+np.pi)%(2.*np.pi))-np.pi,'-',linewidth=2,label=r'$\phi_{0}-\phi_{5}$ mutual  vs freeRef')
	# plt.plot((t[int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot']]*dict_pll['dt']),((dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],3]-dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],5]+np.pi)%(2.*np.pi))-np.pi,'--',linewidth=2,label=r'$\phi_{3}-\phi_{5}$ entrain vs freeRef')
	# plt.plot((t[int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot']]*dict_pll['dt']),((dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],0]-dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],4]+np.pi)%(2.*np.pi))-np.pi,'-.',linewidth=2,label=r'$\phi_{0}-\phi_{4}$ mutual  vs freePLL')
	# plt.plot((t[int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot']]*dict_pll['dt']),((dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],3]-dict_data['phi'][int(0.75*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])):-1:dict_pll['sampleFplot'],4]+np.pi)%(2.*np.pi))-np.pi,'-.',linewidth=2,label=r'$\phi_{3}-\phi_{4}$ entrain vs freePLL')
	# plt.plot(np.mean(dict_pll['transmission_delay']), ((dict_data['phi'][int(round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])),0]-dict_data['phi'][int(round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])),1]+np.pi)%(2.*np.pi))-np.pi, 'yo', ms=5)
	# plt.axvspan(dict_data['t'][-int(5.5*1.0/(dict_pll['intrF']*dict_pll['dt']))], dict_data['t'][-1], color='b', alpha=0.3)

	plt.xlabel(r'$\omega t/2\pi$', fontdict=labelfont, labelpad=-5)
	plt.ylabel(y_label_name_1, rotation=90, fontdict=labelfont, labelpad=40)
	ax012.tick_params(axis='both', which='major', labelsize=tickSize, pad=1)
	ax012.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))], dict_data['t'][-1]])
	# ax012.set_ylim([-np.pi, np.pi])
	plt.legend(loc='upper right')
	plt.grid()

	plt.savefig('results/freq_phaseDiff_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/freq_phaseDiff_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	try:
		ax011.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))],
						dict_data['t'][int(5.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))]])
		ax012.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))],
						dict_data['t'][int(5.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))]])

		plt.savefig('results/freq_phaseDiff_5tauInit_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")
		plt.savefig('results/freq_phaseDiff_5tauInit_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")
	except:
		print('Time series not sufficiently long!')

	try:
		ax011.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))],
						dict_data['t'][int(20.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))]])
		ax012.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))],
						dict_data['t'][int(20.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))]])

		plt.savefig('results/freq_phaseDiff_20tauInit_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")
		plt.savefig('results/freq_phaseDiff_20tauInit_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")
	except:
		print('Time series not sufficiently long!')

	try:
		ax011.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))],
						dict_data['t'][int(50.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))]])
		ax012.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))],
						dict_data['t'][int(50.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))]])

		plt.savefig('results/freq_phaseDiff_50tauInit_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")
		plt.savefig('results/freq_phaseDiff_50tauInit_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")
	except:
		print('Time series not sufficiently long!')

	ax011.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))], dict_data['t'][-1]])
	ax012.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))], dict_data['t'][-1]])

	return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def plot_inst_frequency_and_order_parameter(dict_pll: dict, dict_net: dict, dict_data: dict, plotlist: list = [], order_param_of_divided_signals: bool = True):
	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	labeldict = {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$',
				 'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 = {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$', 'phi': r'$\phi$', 't': r'$t$',
				  'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color = ['blue', 'red', 'purple', 'cyan', 'green', 'yellow']  # 'magenta'
	linet = ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	fig20 = plt.figure(num=20, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig20.canvas.manager.set_window_title('time-series frequency and order parameter')  # frequency and order parameter
	fig20.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	ax011 = fig20.add_subplot(211)

	plt.axvspan(dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))],
				dict_data['t'][int(1.0 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))], color='b', alpha=0.25)
	phidot = np.diff(dict_data['phi'], axis=0) / dict_pll['dt']

	if isinstance(dict_pll['intrF'], list) or isinstance(dict_pll['intrF'], np.ndarray):
		if dict_net['topology'].find('entrain') != -1:
			phidot = phidot / (2.0 * np.pi * np.mean(dict_pll['intrF'][1:]))
		else:
			phidot = phidot / (2.0 * np.pi * np.mean(dict_pll['intrF']))
		ylabelname = r'$\frac{\dot{\theta}_k(t)}{\bar{\omega}_k}$'
	else:
		if not dict_pll['intrF'] == 0:
			phidot = phidot / (2.0 * np.pi * dict_pll['intrF'])
			ylabelname = r'$\frac{\dot{\theta}_k(t)}{\omega}$'
		else:
			ylabelname = r'$\dot{\theta}_k(t)$'

	if not plotlist:
		plt.plot(dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot']],
				 phidot[int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot']], linewidth=2, linestyle=linet[0])
	else:
		plt.plot(dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot']],
				 phidot[int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))::dict_pll['sampleFplot'], plotlist], linewidth=2, linestyle=linet[0])
	# plt.plot(dict_data['t'][dict_net['max_delay_steps']-1], phidot[int(dict_net['max_delay_steps'])-1,0]+0.001,'go')

	plt.xlabel(r'$\frac{\omega t}{2\pi}$', fontdict=labelfont, labelpad=labelpadxaxis)
	plt.ylabel(ylabelname, fontdict=labelfont, labelpad=labelpadyaxis)
	ax011.tick_params(axis='both', which='major', labelsize=tickSize, pad=1)
	ax011.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))], dict_data['t'][-1]])
	plt.grid()

	ax012 = fig20.add_subplot(212)

	plt.plot(dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot']],
			 dict_data['order_parameter'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot']], linewidth=1.5, linestyle=linet[0])
	if order_param_of_divided_signals:
		plt.plot(dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot']],
				 dict_data['order_parameter_divided_phases'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt'])):-1:dict_pll['sampleFplot']], linewidth=1.5, linestyle=linet[2])
	plt.xlabel(r'$\omega t/2\pi$', fontdict=labelfont, labelpad=-5)
	plt.ylabel(r'$R(t)$', rotation=90, fontdict=labelfont, labelpad=40)
	ax012.tick_params(axis='both', which='major', labelsize=tickSize, pad=1)
	ax012.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))], dict_data['t'][-1]])
	plt.grid()

	plt.savefig('results/freq_orderPar_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/freq_orderPar_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
	np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
				dpi=dpi_val, bbox_inches="tight")

	try:
		ax011.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))],
						dict_data['t'][int(5.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))]])
		ax012.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))],
						dict_data['t'][int(5.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))]])

		plt.savefig('results/freq_orderPar_5tauInit_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")
		plt.savefig('results/freq_orderPar_5tauInit_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")
	except:
		print('Time series not sufficiently long!')

	try:
		ax011.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))],
						dict_data['t'][int(20.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))]])
		ax012.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))],
						dict_data['t'][int(20.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))]])

		plt.savefig('results/freq_orderPar_20tauInit_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")
		plt.savefig('results/freq_orderPar_20tauInit_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")
	except:
		print('Time series not sufficiently long!')

	try:
		ax011.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))],
						dict_data['t'][int(50.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))]])
		ax012.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))],
						dict_data['t'][int(50.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))]])

		plt.savefig('results/freq_orderPar_50tauInit_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")
		plt.savefig('results/freq_orderPar_50tauInit_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (
		np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day),
					dpi=dpi_val, bbox_inches="tight")
	except:
		print('Time series not sufficiently long!')

	ax011.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))], dict_data['t'][-1]])
	ax012.set_xlim([dict_data['t'][int(0.75 * np.round(np.mean(dict_pll['transmission_delay']) / dict_pll['dt']))], dict_data['t'][-1]])

	return None


#############################################################################################################################################################################
def plot_histogram(dict_pll: dict, dict_net: dict, dict_data: dict, at_index: int = -1, plot_variable: str = 'phase', phase_wrap=0, plotlist: list = [],
				   prob_density: bool = True, number_of_bins: int = 25, rel_plot_width: float = 1):
	"""Function that plots a histogram of either the phases or the frequencies.

			Args:
				dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
				dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
				dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
				plot_variable: can either be 'phase' (default), 'phase-difference' w.r.t. oscillator k=0, or 'frequency'
				phase_wrap: set to 0 if plotting in (-inf, inf), to 1 of to be plotted in [-pi, pi), to 2 if to be plotted in [-pi/2, 3*pi/2), and to 3 if to be plotted in [0, 2pi)
				plotlist: list of a subset of the oscillators to be plotted

			Returns:
				saves plotted data to files
		"""

	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)

	if at_index == -1:  # translate since in case of plotting frequency 2 indixes are needed and the '-1' syntax does not work
		at_index = len(dict_data['phi'][:, 0]) - 2
	if at_index == 0:  # in case one chooses to plot a histogram at the very beginning, one needs to correct to allow for the differentiation/the generalized function
		at_index = 3

	if plotlist == []:
		plotlist = [i for i in range(dict_net['Nx'] * dict_net['Ny'])]

	if plot_variable == 'frequency':
		plot_func = lambda x, along_dim: np.diff(x, axis=along_dim) / dict_pll['dt']
		x_label_string = r'$\dot{\theta}(t=%0.2f)$' % (at_index * dict_pll['dt'])
		y_label_string = r'$\textrm{hist}\left(\dot{\theta}\right)$'
	elif plot_variable == 'phase':
		x_label_string = r'$\theta(t=%0.2f)$' % (at_index * dict_pll['dt'])
		y_label_string = r'$\textrm{hist}\left(\theta\right)$'
	elif plot_variable == 'phase-difference':
		x_label_string = r'$\Delta\theta(t=%0.2f)$' % (at_index * dict_pll['dt'])
		y_label_string = r'$\textrm{hist}\left(\Delta\theta\right)$'
	else:
		print('Function parameter plot_variable not yet defined, extend the plot function in plot_lib.py!')

	if phase_wrap == 1:  # plot phase-differences in [-pi, pi] interval
		shift2piWin = np.pi
	elif phase_wrap == 2:  # plot phase-differences in [-pi/2, 3*pi/2] interval
		shift2piWin = 0.5 * np.pi
	elif phase_wrap == 3:  # plot phase-differences in [0, 2*pi] interval
		shift2piWin = 0

	fig = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')  # fig, ax =
	fig.canvas.manager.set_window_title('histogram of %s at time=%0.2f' % (plot_variable, at_index * dict_pll['dt']))  # frequency and order parameter
	fig.set_size_inches(plot_size_inches_x, plot_size_inches_y)
	# ax = ax.ravel()

	if plot_variable == 'frequency':
		print('phases to calculate frequency data:', dict_data['phi'][(at_index - 3):(at_index - 1), plotlist])
		print('frequency histogram_data:', plot_func(dict_data['phi'][(at_index - 3):(at_index - 1), plotlist], 0))
		plt.hist(plot_func(dict_data['phi'][(at_index - 3):(at_index - 1), plotlist], 0)[0], bins=number_of_bins, rwidth=rel_plot_width, density=prob_density)
	elif plot_variable == 'phase' and phase_wrap == 0:  # plot phase differences in [-inf, inf), i.e., we use the unwrapped phases that have counted the cycles/periods
		plt.hist(dict_data['phi'][at_index, plotlist], bins=number_of_bins, rwidth=rel_plot_width, density=prob_density)
	elif plot_variable == 'phase' and phase_wrap != 0:
		# print('histogram_data (wrapping if phase):', ((dict_data['phi'][at_index, plotlist] + shift2piWin) % (2 * np.pi)) - shift2piWin)
		plt.hist((((dict_data['phi'][at_index, plotlist] + shift2piWin) % (2.0 * np.pi)) - shift2piWin), bins=number_of_bins, rwidth=rel_plot_width, density=prob_density)
	elif plot_variable == 'phase-difference' and phase_wrap == 0:  # plot phase differences in [-inf, inf), i.e., we use the unwrapped phases that have counted the cycles/periods
		plt.hist(dict_data['phi'][at_index, plotlist] - dict_data['phi'][at_index - 1, 0], bins=number_of_bins, rwidth=rel_plot_width, density=prob_density)
	elif plot_variable == 'phase-difference' and phase_wrap != 0:
		# print('histogram_data (wrapping if phase):', ((dict_data['phi'][at_index, plotlist] + shift2piWin) % (2 * np.pi)) - shift2piWin)
		plt.hist((((dict_data['phi'][at_index, plotlist] - dict_data['phi'][at_index - 1, 0] + shift2piWin) % (2.0 * np.pi)) - shift2piWin), bins=number_of_bins, rwidth=rel_plot_width,
				 density=prob_density)

	plt.xlabel(x_label_string)
	plt.ylabel(y_label_string)

	plt.savefig('results/histogram_%s_atTime%0.2f_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' % (plot_variable, at_index * dict_pll['dt'], np.mean(dict_pll['coupK']),
																									  np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']),
																									  np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/histogram_%s_atTime%0.2f_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' % (plot_variable, at_index * dict_pll['dt'], np.mean(dict_pll['coupK']),
																									  np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']),
																									  np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")

	return None  # ax


#################################################################################################################################################################################

def plot_time_dependent_parameter(dict_pll: dict, dict_net: dict, dict_data: dict, time_dependent_parameter: np.ndarray, y_label: str):
	"""Function that plots the time dependence of a variable or parameter.

				Args:
					dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
					dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
					dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
					time_dependent_parameter: array of the time-dependent parameter/variable
					y_label: name of that variable or parameter, also used to label the y-axis of the plot

				Returns:
					saves plotted data to files
			"""

	fig = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig.canvas.manager.set_window_title('time dependence of %s' % (y_label))  # frequency and order parameter
	fig.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	t = np.arange(0, dict_net['max_delay_steps'] + dict_pll['sim_time_steps']) * dict_pll['dt']
	plt.plot(t, time_dependent_parameter)
	plt.xlabel('time')
	plt.ylabel(y_label)

	plt.savefig('results/time_dependence_%s_start%0.2f_end_%0.2f_ratePerPeriod_%0.5f_%d_%d_%d.png' % (y_label,
						dict_net['min_max_rate_timeDepPara'][0], dict_net['min_max_rate_timeDepPara'][1],
							dict_net['min_max_rate_timeDepPara'][2], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/time_dependence_%s_start%0.2f_end_%0.2f_ratePerPeriod_%0.5f_%d_%d_%d.svg' % (y_label,
						dict_net['min_max_rate_timeDepPara'][0], dict_net['min_max_rate_timeDepPara'][1],
							dict_net['min_max_rate_timeDepPara'][2], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")

	return None


# ################################################################################################################################################################################

def plot_allan_variance(dict_pll: dict, dict_net: dict, dict_data: dict, t_transients_decayed: float, plotlist: list = [0], type_def_allan: str = 'overlapping_adev', phase_or_freq: str = 'frequency',
						max_integration_time: float = 1000):
	"""Function that plots the Allan variance of the a frequency or phase time-series. Uses the Allan variance python library: https://allantools.readthedocs.io/en/latest/.

				Args:
					dict_net:  contains as parameters the information about properties of the network to be simulated, the initial conditions and the synchronized states under investigation
					dict_pll:  contains as parameters the information about properties of the PLLs, its components, the time delays, the types of signals exchanged
					dict_data: contains the results of the simulation, i.e., the phases of all oscillators, time dependent parameters, etc.
					t_transients_decayed: time after which transients have decayed to choose the part of the time-series to be analyzed
					plotlist: list of all indexes of oscillators to be plotted/evaluated
					type_def_allan: what type of Allan variance should be plotted, choose from 'overlapping_adev', 'modified_adev'
					phase_or_freq: whether to generate of the frequency or the phase time-series (i.e., plotting frequency stability vs plotting ....): 'frequency', 'phase'

				Returns:
					saves plotted data to files
			"""

	dict_pll, dict_net = prepareDictsForPlotting(dict_pll, dict_net)
	largest_averaging_window = max_integration_time
	# A: make sure, that transients are not incorporated
	# HOWTO 2) to determine the time to solutions, we find the time at the which the asymptotic value of the order parameter has been reached, we count from the time when we start increasing one of the coupling strengths
	order_param_change_treshold = 0.01
	smoothing = True
	if smoothing:  # we find the last transient changes of the order parameter before its fluctuations/changes become smaller than the threshold
		temp = np.where(uniform_filter1d((np.diff(dict_data['order_parameter'][dict_net['max_delay_steps']:]) / dict_pll['dt']), size=int(0.5 * np.mean(dict_pll['intrF']) / dict_pll['dt']),
										 mode='reflect') > order_param_change_treshold)
	# print('temp=', temp[0])
	else:
		temp = np.where((np.diff(dict_data['order_parameter'][dict_net['max_delay_steps']:]) / dict_pll['dt']) > order_param_change_treshold)
	# print('temp=', temp[0])

	if not len(temp[0]) == 0:
		time_to_transient_decay = temp[0][-1] * dict_pll['dt']
	else:
		time_to_transient_decay = 0.2 * dict_net['Tsim']

	if t_transients_decayed > time_to_transient_decay:
		print('Initially provided time at which transients have decayed is larger then that measured from the order parameter. Adjusted to new time!')
		t_transients_decayed = time_to_transient_decay
	else:
		print('Initially provided time at which transients have decayed is smaller then that measured from the order parameter. Adjusted to new time!')
		t_transients_decayed = time_to_transient_decay

	max_integration_time = dict_net['Tsim'] - t_transients_decayed - 5

	# test whether time set for transients to have decayed is smaller than the simulation time
	if dict_net['Tsim'] - t_transients_decayed >= max_integration_time:
		print('Please adjust the t_transients_decayed to be smaller than the length of the time-series simulated.')

	# list of averaging time/integration time, x-axis
	tau_integration = np.logspace(start=0, stop=np.log10(largest_averaging_window), num=50, base=10)
	if phase_or_freq == 'frequency':
		y = np.diff(dict_data['phi'][int(t_transients_decayed / dict_pll['dt']):-1, plotlist], axis=0) / dict_pll['dt']
		data_type = 'freq'
	elif phase_or_freq == 'phase':
		print('Phase data transformed into phase in seconds for the allanvariance library used.')
		sys.exit()  # use instantaneous frequency and phase to calculate a time  time-series in seconds, see docs of allanlib
		y = dict_data['phi'][int(t_transients_decayed / dict_pll['dt']):-1, plotlist]
		data_type = 'phase'

	# sample rate in Hz of the input data
	sample_rate = 1 / dict_pll['dt']
	t2_data_all_clocks = []
	ad_data_all_clocks = []

	for i in plotlist:
		# print('Data:', y[:, i])
		# compute the overlapping ADEV or... or...
		if type_def_allan == 'overlapping_adev':
			print('Calculating overlapping Allan variance for oscillator %i!' % (i))
			(t2, ad, ade, adn) = allantools.oadev(y[:, i], rate=sample_rate, data_type=data_type, taus=tau_integration)
			t2_data_all_clocks.append(t2)
			ad_data_all_clocks.append(ad)
		elif type_def_allan == 'modified_adev':
			print('Calculating modified Allan variance for oscillator %i!' % (i))
			(t2, ad, ade, adn) = allantools.mdev(y[:, i], rate=sample_rate, data_type=data_type, taus=tau_integration)
			t2_data_all_clocks.append(t2)
			ad_data_all_clocks.append(ad)
		else:
			print('Other types of Allan variances need to be implemented! See: https://allantools.readthedocs.io/en/latest/readme_copy.html#minimal-example-phase-data')

	# create here a reference from a GWN source
	if type_def_allan == 'overlapping_adev':
		(t2_data_gwn_ref, ad_data_gwn_ref, errors, ns) = allantools.oadev(np.random.normal(0, np.sqrt(dict_pll['noiseVarVCO'] * dict_pll['dt']), len(y[:, 0])))
	elif type_def_allan == 'modified_adev':
		(t2_data_gwn_ref, ad_data_gwn_ref, errors, ns) = allantools.mdev(np.random.normal(0, np.sqrt(dict_pll['noiseVarVCO'] * dict_pll['dt']), len(y[:, 0])))
	else:
		print('Other types of Allan variances need to be implemented! See: https://allantools.readthedocs.io/en/latest/readme_copy.html#minimal-example-phase-data')

	fig = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig.canvas.manager.set_window_title('%s Allan variance' % (type_def_allan))  # frequency and order parameter
	fig.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	colors = ['b', 'r', 'c', 'g', 'k', 'm']
	symbol = ['d', 'D', 's', 'o', 'v', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', 'x', 'X']

	plt.loglog(t2_data_gwn_ref, ad_data_gwn_ref, 'y--', label=r'GWN(0, $\sigma_\textrm{vco}$)')
	for i in range(len(plotlist)):
		color_symbol = colors[i % len(colors)] + symbol[np.random.randint(0, len(symbol) - 1)]
		print('Picked symbol: ', color_symbol)
		plt.loglog(t2_data_all_clocks, ad_data_all_clocks, color_symbol, markersize=2, label='PLL%i' % (plotlist[i]))

	title_string = 'Allan variance of oscillators in loglog'
	plt.title(title_string)
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys())
	plt.xlabel(r'$\tau$')
	plt.ylabel(r'$\sigma_\textrm{y}(\tau)$')

	plt.savefig('results/%s_Allan_fsample%0.2f_%d_%d_%d.png' % (type_def_allan, dict_pll['sampleF'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/%s_Allan_fsample%0.2f_%d_%d_%d.svg' % (type_def_allan, dict_pll['sampleF'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")

	return None


#################################################################################################################################################################################

def plot_order_param_vs_parameter_space(pool_data: dict, average_time_order_parameter_in_periods: np.float, axis_normalization=True, add_scatter_plots: bool = False):
	""" Function that plots the last value and an average of the order parameter in a 2d parameter space in two individual plots.
		Each plot comes as a scatterplot and an imshow.

		Args:
			pool_data: contains the results of all simulations (realizations), i.e., the phases of all oscillators, time dependent parameters, etc.
			average_time_order_parameter_in_periods: determines over how many periods of the intrinsic PLL frequency averages of the order parameter are performed
			axis_normalization: whether the axis of the plot are normalized or not: default True
			add_scatter_plots: if true, also plot scatter plots of parameter space

		Returns:
			saves plotted data to files
	"""

	dict_net = pool_data[0][0]['dict_net']
	dict_algo = pool_data[0][0]['dict_algo']

	# prepare colormap for scatter plot that is always in [0, 1] or [min(results), max(results)]
	cdict = {
		'red': ((0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
		'green': ((0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
		'blue': ((0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
	}
	# cmaps['Diverging'] = [ 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

	# if (topology == 'entrainOne' or topology == 'entrainAll'):
	# 	# cmap=plt.cm.get_cmap('Blues', 6)
	# 	cmap 	  = plt.cm.get_cmap('seismic', 256);
	# 	cmap1 	  = plt.cm.get_cmap('seismic', 256);
	# 	colormap  = eva.shiftedColorMap(cmap,  start=0.0, midpoint=absR_conf, stop=1.0, name='shiftedcmap')
	# 	colormap1 = eva.shiftedColorMap(cmap1, start=0.0, midpoint=absR_conf, stop=1.0, name='shiftedcmap1')
	# 	print('absR_conf=', absR_conf, ', min(results[:,1])=', min(results[:,1]), ', max(results[:,1])=', max(results[:,1]))
	# else:
	colormap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

	# extract the results from the data dictionary for plotting '''
	results = []
	for i in range(dict_algo['paramDiscretization'][0] * dict_algo['paramDiscretization'][1]):
		if 'entrain' in dict_net['topology'] and (isinstance(pool_data[0][i]['dict_pll']['intrF'], list) or isinstance(pool_data[0][i]['dict_pll']['intrF'], np.ndarray)):
			averaging_time_as_index = np.int(average_time_order_parameter_in_periods * np.mean(pool_data[0][i]['dict_pll']['intrF'][1:]) / pool_data[0][i]['dict_pll']['dt'])
		else:
			averaging_time_as_index = np.int(average_time_order_parameter_in_periods * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt'])
		results.append(
			[pool_data[0][i]['dict_data']['order_parameter'][-1], np.mean(pool_data[0][i]['dict_data']['order_parameter'][-averaging_time_as_index:]), np.std(pool_data[0][i]['dict_data']['order_parameter'][-averaging_time_as_index:])])
	results = np.array(results, dtype=object)

	# set the normalization of the axis
	normalization_x = 1
	normalization_y = 1
	x_label = 'tbs'
	y_label = 'tbs'
	if axis_normalization:
		if dict_algo['param_id_0'] == 'intrF':
			if 'entrain' in dict_net['topology'] and (isinstance(pool_data[0][0]['dict_pll']['intrF'], list) or isinstance(pool_data[0][0]['dict_pll']['intrF'], np.ndarray)):
				normalization_x = 1 / np.mean(pool_data[0][0]['dict_pll']['intrF'][1:])
			else:
				normalization_x = 1 / np.mean(pool_data[0][0]['dict_pll']['intrF'])
			x_label = r'$f_\textrm{R}/\langle f \rangle$'
		else:
			x_label = r'$f$\,[Hz]'
			print('Introduce normalization in plot_lib.plot_order_param_vs_parameter_space() function!')
		if dict_algo['param_id_1'] == 'transmission_delay':
			if 'entrain' in dict_net['topology'] and (isinstance(pool_data[0][0]['dict_pll']['intrF'], list) or isinstance(pool_data[0][0]['dict_pll']['intrF'], np.ndarray)):
				normalization_y = np.mean(pool_data[0][0]['dict_pll']['intrF'][1:])
			else:
				normalization_y = np.mean(pool_data[0][0]['dict_pll']['intrF'])
			y_label = r'$\tau_{kl}/T_{\omega}$'
		else:
			y_label = r'$\tau_{kl}$'
			print('Introduce normalization in plot_lib.plot_order_param_vs_parameter_space() function!')

	# start plotting
	fig1 = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig1.canvas.manager.set_window_title('parameter space %s vs %s' % (dict_algo['param_id_0'], dict_algo['param_id_1']))
	fig1.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	tempresults = results[:, 0].reshape(dict_algo['paramDiscretization'][0], dict_algo['paramDiscretization'][1])
	tempresults = np.transpose(tempresults)
	# print('tempresults:', tempresults)
	tempresults_ma = ma.masked_where(tempresults < 0, tempresults)  # Create masked array
	# print('tempresult_ma:', tempresults_ma)
	# print('initPhiPrime0:', initPhiPrime0)
	plt.imshow(tempresults_ma.astype(float), interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower',
			   extent=(dict_algo['min_max_range_parameter_0'][0], dict_algo['min_max_range_parameter_0'][1], dict_algo['min_max_range_parameter_1'][0], dict_algo['min_max_range_parameter_1'][1]),
			   vmin=0, vmax=1)
	plt.title(r'last $R(t)$')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.xlim([1.05 * dict_algo['min_max_range_parameter_0'][0], 1.05 * dict_algo['min_max_range_parameter_0'][1]])
	plt.ylim([1.05 * dict_algo['min_max_range_parameter_1'][0], 1.05 * dict_algo['min_max_range_parameter_1'][1]])
	plt.colorbar()

	plt.savefig('results/param_space_%s_vs_%s_lastR_imshow_%d_%d_%d.png' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/param_space_%s_vs_%s_lastR_imshow_%d_%d_%d.svg' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")

	fig2 = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig2.canvas.manager.set_window_title('parameter space %s vs %s' % (dict_algo['param_id_0'], dict_algo['param_id_1']))
	fig2.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	tempresults = results[:, 1].reshape(dict_algo['paramDiscretization'][0], dict_algo['paramDiscretization'][1])
	tempresults = np.transpose(tempresults)
	# print('tempresults:', tempresults)
	tempresults_ma = ma.masked_where(tempresults < 0, tempresults)  # Create masked array
	# print('tempresult_ma:', tempresults_ma)
	# print('initPhiPrime0:', initPhiPrime0)
	plt.imshow(tempresults_ma.astype(float), interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower',
			   extent=(dict_algo['min_max_range_parameter_0'][0], dict_algo['min_max_range_parameter_0'][1], dict_algo['min_max_range_parameter_1'][0], dict_algo['min_max_range_parameter_1'][1]),
			   vmin=0, vmax=1)
	plt.title(r'mean $R(t)$')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.xlim([1.05 * dict_algo['min_max_range_parameter_0'][0], 1.05 * dict_algo['min_max_range_parameter_0'][1]])
	plt.ylim([1.05 * dict_algo['min_max_range_parameter_1'][0], 1.05 * dict_algo['min_max_range_parameter_1'][1]])
	plt.colorbar()

	plt.savefig('results/param_space_%s_vs_%s_meanR_imshow_%d_%d_%d.png' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/param_space_%s_vs_%s_meanR_imshow_%d_%d_%d.svg' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")

	if add_scatter_plots:
		fig3 = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
		fig3.canvas.manager.set_window_title('parameter space %s vs %s' % (dict_algo['param_id_0'], dict_algo['param_id_1']))
		fig3.set_size_inches(plot_size_inches_x, plot_size_inches_y)

		plt.clf()
		ax = plt.subplot(1, 1, 1)
		ax.set_aspect('equal')

		plt.scatter(dict_algo['allPoints'][:, 0], dict_algo['allPoints'][:, 1], c=results[:, 0], alpha=0.5, cmap=colormap, vmin=0, vmax=1)
		plt.title(r'last $R(t)$')
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.xlim([1.05 * dict_algo['min_max_range_parameter_0'][0], 1.05 * dict_algo['min_max_range_parameter_0'][1]])
		plt.ylim([1.05 * dict_algo['min_max_range_parameter_1'][0], 1.05 * dict_algo['min_max_range_parameter_1'][1]])
		plt.colorbar()
		plt.savefig('results/param_space_%s_vs_%s_lastR_imshow_%d_%d_%d.png' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
		plt.savefig('results/param_space_%s_vs_%s_lastR_imshow_%d_%d_%d.svg' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")

		fig4 = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
		fig4.canvas.manager.set_window_title('parameter space %s vs %s' % (dict_algo['param_id_0'], dict_algo['param_id_1']))
		fig4.set_size_inches(plot_size_inches_x, plot_size_inches_y)

		plt.clf()
		ax = plt.subplot(1, 1, 1)
		ax.set_aspect('equal')
		plt.scatter(dict_algo['allPoints'][:, 0], dict_algo['allPoints'][:, 1], c=results[:, 1], alpha=0.5, cmap=colormap, vmin=0, vmax=1)
		plt.title(r'mean $R(t)$')
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.xlim([1.05 * dict_algo['min_max_range_parameter_0'][0], 1.05 * dict_algo['min_max_range_parameter_0'][1]])
		plt.ylim([1.05 * dict_algo['min_max_range_parameter_1'][0], 1.05 * dict_algo['min_max_range_parameter_1'][1]])
		plt.colorbar()
		plt.savefig('results/param_space_%s_vs_%s_meanR_scatter_%d_%d_%d.png' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
		plt.savefig('results/param_space_%s_vs_%s_meanR_scatter_%d_%d_%d.svg' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")

	return None


# ################################################################################################################################################################################


def plot_final_phase_configuration_vs_parameter_space(pool_data: dict, average_time_phase_difference_in_periods: np.float, phase_wrap: np.int = 1, std_treshold_determine_time_dependency: np.float = 0.1*np.pi,
													  axis_normalization: bool = True, phase_diff_wrt_osci_k_0: bool = False, plot_if_time_dependent: bool = False, add_scatter_plots: bool = False):
	""" Function that plots the last value of the phase and the std of the phase over the average time in a 2d parameter space in two individual plots.
		Each plot comes as a scatterplot and an imshow.

		Args:
			pool_data: contains the results of all simulations (realizations), i.e., the phases of all oscillators, time dependent parameters, etc.
			average_time_phase_difference_in_periods: determines over how many periods of the intrinsic PLL frequency averages of the order parameter are performed
			phase_wrap: determines the representation of the 2pi periodic phase
			axis_normalization: whether the axis of the plot are normalized or not: default True
			phase_diff_wrt_osci_k_0: whether the phase difference is calculated with respect to oscillator zero, otherwise w.r.t. nearest neighbor
			plot_if_time_dependent: if True, also plot results in the parameter space where the phase-differences are still time-dependent
			add_scatter_plots: if true, also plot scatter plots of parameter space

		Returns:
			saves plotted data to files
	"""

	if phase_wrap == 1:  # plot phase-differences in [-pi, pi) interval
		shift2piWin = np.pi
		imshow_min_val = -np.pi
		imshow_max_val = np.pi-0.000001
	elif phase_wrap == 2:  # plot phase-differences in [-pi/2, 3*pi/2) interval
		shift2piWin = 0.5 * np.pi
		imshow_min_val = -np.pi/2
		imshow_max_val = 3*np.pi/2 - 0.000001
	elif phase_wrap == 3:  # plot phase-differences in [0, 2*pi) interval
		shift2piWin = 0
		imshow_min_val = 0
		imshow_max_val = 2*np.pi - 0.000001

	dict_net = pool_data[0][0]['dict_net']
	dict_algo = pool_data[0][0]['dict_algo']

	# prepare colormap for scatter plot that is always in [0, 1] or [min(results), max(results)]
	cdict = {
		'red': ((0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
		'green': ((0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
		'blue': ((0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
	}
	# cmaps['Diverging'] = [ 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

	# if (topology == 'entrainOne' or topology == 'entrainAll'):
	# 	# cmap=plt.cm.get_cmap('Blues', 6)
	# 	cmap 	  = plt.cm.get_cmap('seismic', 256);
	# 	cmap1 	  = plt.cm.get_cmap('seismic', 256);
	# 	colormap  = eva.shiftedColorMap(cmap,  start=0.0, midpoint=absR_conf, stop=1.0, name='shiftedcmap')
	# 	colormap1 = eva.shiftedColorMap(cmap1, start=0.0, midpoint=absR_conf, stop=1.0, name='shiftedcmap1')
	# 	print('absR_conf=', absR_conf, ', min(results[:,1])=', min(results[:,1]), ', max(results[:,1])=', max(results[:,1]))
	# else:
	colormap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

	# extract the results from the data dictionary for plotting '''
	j0 = 0
	# loop over all parameter pairs of the parameter space, indexed by "i", loops over index "j" represent the mutual phase differences
	beta_kl = 999 + np.zeros([len(pool_data[0][0]['dict_data']['phi'][-1, 1:]), dict_algo['paramDiscretization'][0] * dict_algo['paramDiscretization'][1]])
	std_beta_kl = 999 + np.zeros([len(pool_data[0][0]['dict_data']['phi'][-1, 1:]), dict_algo['paramDiscretization'][0] * dict_algo['paramDiscretization'][1]])
	mean_beta_kl = 999 + np.zeros([len(pool_data[0][0]['dict_data']['phi'][-1, 1:]), dict_algo['paramDiscretization'][0] * dict_algo['paramDiscretization'][1]])
	predicted_beta_kl = 999 + np.zeros([len(pool_data[0][0]['dict_data']['phi'][-1, 1:]), dict_algo['paramDiscretization'][0] * dict_algo['paramDiscretization'][1]])
	print('\nbeta_kl:', beta_kl, '\nstd_beta_kl', std_beta_kl)
	for i in range(dict_algo['paramDiscretization'][0] * dict_algo['paramDiscretization'][1]):
		print('\n\ni=', i)
		if 'entrain' in dict_net['topology'] and (isinstance(pool_data[0][i]['dict_pll']['intrF'], list) or isinstance(pool_data[0][i]['dict_pll']['intrF'], np.ndarray)):
			averaging_time_as_index = np.int(average_time_phase_difference_in_periods * np.mean(pool_data[0][i]['dict_pll']['intrF'][1:]) / pool_data[0][i]['dict_pll']['dt'])
		else:
			averaging_time_as_index = np.int(average_time_phase_difference_in_periods * np.mean(pool_data[0][i]['dict_pll']['intrF']) / pool_data[0][i]['dict_pll']['dt'])
		if np.isnan(pool_data[0][i]['dict_net']['phiInitConfig']).any() and 'entrain' in dict_net['topology']:
			print('Set all betas and std_betas to np.nan since for this parameter set no solution exists given the inverse coupling function to be evaluated.')
			predicted_beta_kl[:, i] = np.nan
			beta_kl[:, i] = np.nan
			std_beta_kl[:, i] = -999
			mean_beta_kl[:, i] = np.nan
		else:
			if phase_diff_wrt_osci_k_0:
				j0 = 1
				for j in range(j0, len(pool_data[0][i]['dict_data']['phi'][-1, 1:])):
					predicted_beta_kl[j, i] = (pool_data[0][i]['dict_net']['phiInitConfig'][j] - pool_data[0][i]['dict_net']['phiInitConfig'][0] + shift2piWin) % (2.0 * np.pi) - shift2piWin
					std_beta_kl[j, i] = np.std(((pool_data[0][i]['dict_data']['phi'][-averaging_time_as_index:, j]
														- pool_data[0][i]['dict_data']['phi'][-averaging_time_as_index:, 0] + shift2piWin) % (2.0 * np.pi)) - shift2piWin)
					if not plot_if_time_dependent and std_beta_kl[j, i] > std_treshold_determine_time_dependency:
						beta_kl[j, i] = np.nan
						mean_beta_kl[j, i] = np.nan
						print('Set beta_%i%i[%i,%i] to np.nan since std_beta_%i%i[%i,%i]=%0.2f' % (j, 0, j, i, j, 0, j, i, std_beta_kl[j, i]))
					else:
						beta_kl[j, i] = ((pool_data[0][i]['dict_data']['phi'][-1, j] - pool_data[0][i]['dict_data']['phi'][-1, 0] + shift2piWin) % (2.0 * np.pi)) - shift2piWin
						mean_beta_kl[j, i] = np.mean(((pool_data[0][i]['dict_data']['phi'][-averaging_time_as_index:, j]
													   - pool_data[0][i]['dict_data']['phi'][-averaging_time_as_index:, 0] + shift2piWin) % (2.0 * np.pi)) - shift2piWin)
					print('Calculated beta_%i0=beta_%i-beta_0.' % (j, j), ' Hence, beta_%i0=' % j, beta_kl[j, i], ', and std(beta_%i0)=' % j, std_beta_kl[j, i])
			else:
				j0 = 0
				for j in range(j0, len(pool_data[0][i]['dict_data']['phi'][-1, 1:])):
					print('j=', j)
					# print('pool_data[0][i][*dict_net*][*phiInitConfig*]:', pool_data[0][i]['dict_net']['phiInitConfig'])
					predicted_beta_kl[j, i] = (pool_data[0][i]['dict_net']['phiInitConfig'][j] - pool_data[0][i]['dict_net']['phiInitConfig'][j+1] + shift2piWin) % (2.0 * np.pi) - shift2piWin
					std_beta_kl[j, i] = np.std(((pool_data[0][i]['dict_data']['phi'][-averaging_time_as_index:, j]
														- pool_data[0][i]['dict_data']['phi'][-averaging_time_as_index:, j+1] + shift2piWin) % (2.0 * np.pi)) - shift2piWin)
					if not plot_if_time_dependent and std_beta_kl[j, i] > std_treshold_determine_time_dependency:
						beta_kl[j, i] = np.nan
						mean_beta_kl[j, i] = np.nan
						print('Set beta_%i%i[%i,%i] to np.nan since std_beta_%i%i[%i,%i]=%0.2f' % (j, j+1, j, i, j, j+1, j, i, std_beta_kl[j, i]))
						if (dict_algo['paramDiscretization'][0] * dict_algo['paramDiscretization'][1] < 10
								and pool_data[0][i]['dict_pll']['intrF'][0] == dict_algo['scanValues'][0, 1][0] and pool_data[0][i]['dict_pll']['transmission_delay'] == dict_algo['scanValues'][1, 1]):
							plot_inst_frequency_and_phase_difference(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_algo'], pool_data[0][i]['dict_data'], True, [], 2)
							# plt.draw()
							# plt.show()
					else:
						beta_kl[j, i] = ((pool_data[0][i]['dict_data']['phi'][-1, j] - pool_data[0][i]['dict_data']['phi'][-1, j+1] + shift2piWin) % (2.0 * np.pi)) - shift2piWin
						mean_beta_kl[j, i] = np.mean(((pool_data[0][i]['dict_data']['phi'][-averaging_time_as_index:, j]
													   - pool_data[0][i]['dict_data']['phi'][-averaging_time_as_index:, j + 1] + shift2piWin) % (2.0 * np.pi)) - shift2piWin)
						if (dict_algo['paramDiscretization'][0] * dict_algo['paramDiscretization'][1] < 10
								and pool_data[0][i]['dict_pll']['intrF'][0] == dict_algo['scanValues'][0, 1][0] and pool_data[0][i]['dict_pll']['transmission_delay'] == dict_algo['scanValues'][1, 1]):
							plot_inst_frequency_and_phase_difference(pool_data[0][i]['dict_pll'], pool_data[0][i]['dict_net'], pool_data[0][i]['dict_algo'], pool_data[0][i]['dict_data'], True, [], 2)
							# plt.draw()
							# plt.show()

					print('Calculated beta_%i%i=beta_%i-beta_%i.' % (j, j + 1, j, j + 1), ' It is beta_%i%i=' % (j, j+1), beta_kl[j, i], ', and std(beta_%i%i)=' % (j, j+1), std_beta_kl[j, i])
					# print('\nfrom: phi[-average_index:, j], phi[-average_index:, j+1]\n', pool_data[0][i]['dict_data']['phi'][-averaging_time_as_index:, j], '\n',
					#		pool_data[0][i]['dict_data']['phi'][-averaging_time_as_index:, j + 1])

	# set the normalization of the axis
	normalization_x = 1
	normalization_y = 1
	x_label = 'tbs'
	y_label = 'tbs'
	if axis_normalization:
		if dict_algo['param_id_0'] == 'intrF':
			if 'entrain' in dict_net['topology'] and (isinstance(pool_data[0][0]['dict_pll']['intrF'], list) or isinstance(pool_data[0][0]['dict_pll']['intrF'], np.ndarray)):
				normalization_x = 1 / np.mean(pool_data[0][0]['dict_pll']['intrF'][1:])
			else:
				normalization_x = 1 / np.mean(pool_data[0][0]['dict_pll']['intrF'])
			x_label = r'$f_\textrm{R}/\langle f \rangle$'
		else:
			x_label = r'$f$\,[Hz]'
			print('Introduce normalization in plot_lib.plot_order_param_vs_parameter_space() function!')
		if dict_algo['param_id_1'] == 'transmission_delay':
			if 'entrain' in dict_net['topology'] and (isinstance(pool_data[0][0]['dict_pll']['intrF'], list) or isinstance(pool_data[0][0]['dict_pll']['intrF'], np.ndarray)):
				normalization_y = np.mean(pool_data[0][0]['dict_pll']['intrF'][1:])
			else:
				normalization_y = np.mean(pool_data[0][0]['dict_pll']['intrF'])
			y_label = r'$\tau_{kl}/T_{\omega}$'
		else:
			y_label = r'$\tau_{kl}$'
			print('Introduce normalization in plot_lib.plot_order_param_vs_parameter_space() function!')

	figs = []
	for j in range(j0, len(pool_data[0][0]['dict_data']['phi'][-1, 1:])):

		if phase_diff_wrt_osci_k_0:
			jj = 0
			phi_string = r'$\phi_{%i0}$' % j
		else:
			jj = j+1
			phi_string = r'$\phi_{%i%i}$' % (j, j+1)

		print('\nbeta_%i%i[%i,:]=' % (j, jj, j), beta_kl[j, :])
		print('\nstd_beta_%i%i[%i,:]=' % (j, jj, j), std_beta_kl[j, :])

		# start plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		fig1 = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
		fig1.canvas.manager.set_window_title('final phase differences in parameter space %s vs %s' % (dict_algo['param_id_0'].replace('_', ' '), dict_algo['param_id_1'].replace('_', ' ')))
		fig1.set_size_inches(plot_size_inches_x, plot_size_inches_y)

		tempresults = beta_kl[j, :].reshape(dict_algo['paramDiscretization'][0], dict_algo['paramDiscretization'][1])
		tempresults = np.transpose(tempresults)
		# print('tempresults:', tempresults)
		tempresults_ma = ma.masked_where(tempresults == np.nan, tempresults)  # Create masked array
		# print('tempresult_ma:', tempresults_ma)

		plt.imshow(tempresults_ma.astype(float), interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower',
				   extent=(dict_algo['min_max_range_parameter_0'][0], dict_algo['min_max_range_parameter_0'][1], dict_algo['min_max_range_parameter_1'][0], dict_algo['min_max_range_parameter_1'][1]),
				   vmin=imshow_min_val, vmax=imshow_max_val)
		plt.title(r'last $\Delta$'+phi_string)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.xlim([1.05 * dict_algo['min_max_range_parameter_0'][0], 1.05 * dict_algo['min_max_range_parameter_0'][1]])
		plt.ylim([1.05 * dict_algo['min_max_range_parameter_1'][0], 1.05 * dict_algo['min_max_range_parameter_1'][1]])
		plt.colorbar()

		plt.savefig('results/param_space_%s_vs_%s_final_beta_%i%i_%d_%d_%d.png' % (dict_algo['param_id_0'], dict_algo['param_id_1'], j, jj, now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
		plt.savefig('results/param_space_%s_vs_%s_final_beta_%i%i_%d_%d_%d.svg' % (dict_algo['param_id_0'], dict_algo['param_id_1'], j, jj, now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		fig21 = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
		fig21.canvas.manager.set_window_title('mean of phase differences of %0.2fT parameter space %s vs %s' % (average_time_phase_difference_in_periods, dict_algo['param_id_0'].replace('_', ' '),
																												dict_algo['param_id_1'].replace('_', ' ')))
		fig21.set_size_inches(plot_size_inches_x, plot_size_inches_y)

		tempresults = mean_beta_kl[j, :].reshape(dict_algo['paramDiscretization'][0], dict_algo['paramDiscretization'][1])
		tempresults = np.transpose(tempresults)
		# print('tempresults:', tempresults)
		tempresults_ma = ma.masked_where(tempresults == np.nan, tempresults)  # Create masked array
		# print('tempresult_ma:', tempresults_ma)

		plt.imshow(tempresults_ma.astype(float), interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower',
				   extent=(dict_algo['min_max_range_parameter_0'][0], dict_algo['min_max_range_parameter_0'][1], dict_algo['min_max_range_parameter_1'][0], dict_algo['min_max_range_parameter_1'][1]),
				   vmin=imshow_min_val, vmax=imshow_max_val)
		plt.title(r'$\bar{\Delta}$' + phi_string)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.xlim([1.05 * dict_algo['min_max_range_parameter_0'][0], 1.05 * dict_algo['min_max_range_parameter_0'][1]])
		plt.ylim([1.05 * dict_algo['min_max_range_parameter_1'][0], 1.05 * dict_algo['min_max_range_parameter_1'][1]])
		plt.colorbar()

		plt.savefig('results/param_space_%s_vs_%s_mean_beta_%i%i_%d_%d_%d.png' % (dict_algo['param_id_0'], dict_algo['param_id_1'], j, jj, now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
		plt.savefig('results/param_space_%s_vs_%s_mean_beta_%i%i_%d_%d_%d.svg' % (dict_algo['param_id_0'], dict_algo['param_id_1'], j, jj, now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		fig22 = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
		fig22.canvas.manager.set_window_title('predicted phase differences in parameter space %s vs %s' % (dict_algo['param_id_0'].replace('_', ' '), dict_algo['param_id_1'].replace('_', ' ')))
		fig22.set_size_inches(plot_size_inches_x, plot_size_inches_y)

		tempresults = predicted_beta_kl[j, :].reshape(dict_algo['paramDiscretization'][0], dict_algo['paramDiscretization'][1])
		tempresults = np.transpose(tempresults)

		plt.imshow(tempresults.astype(float), interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower',
				   extent=(dict_algo['min_max_range_parameter_0'][0], dict_algo['min_max_range_parameter_0'][1], dict_algo['min_max_range_parameter_1'][0], dict_algo['min_max_range_parameter_1'][1]),
				   vmin=std_treshold_determine_time_dependency, vmax=2 * np.pi)
		plt.title(r'$\Delta_\textrm{theory}$' + phi_string)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.xlim([1.05 * dict_algo['min_max_range_parameter_0'][0], 1.05 * dict_algo['min_max_range_parameter_0'][1]])
		plt.ylim([1.05 * dict_algo['min_max_range_parameter_1'][0], 1.05 * dict_algo['min_max_range_parameter_1'][1]])
		plt.colorbar()

		plt.savefig('results/param_space_%s_vs_%s_predicted_beta_%i%i_%d_%d_%d.png' % (dict_algo['param_id_0'], dict_algo['param_id_1'], j, jj, now.year, now.month, now.day), dpi=dpi_val,	bbox_inches="tight")
		plt.savefig('results/param_space_%s_vs_%s_predicted_beta_%i%i_%d_%d_%d.svg' % (dict_algo['param_id_0'], dict_algo['param_id_1'], j, jj, now.year, now.month, now.day), dpi=dpi_val,	bbox_inches="tight")
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		fig23 = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
		fig23.canvas.manager.set_window_title('standard deviation of phase differences of %0.2fT parameter space %s vs %s' % (average_time_phase_difference_in_periods,
																															  dict_algo['param_id_0'].replace('_', ' '),
																															  dict_algo['param_id_1'].replace('_', ' ')))
		fig23.set_size_inches(plot_size_inches_x, plot_size_inches_y)

		tempresults = std_beta_kl[j, :].reshape(dict_algo['paramDiscretization'][0], dict_algo['paramDiscretization'][1])
		tempresults = np.transpose(tempresults)
		# print('tempresults:', tempresults)
		# tempresults_ma = ma.masked_where(tempresults == np.nan, tempresults)  # Create masked array
		tempresults_ma = ma.masked_where(tempresults < std_treshold_determine_time_dependency, tempresults)  # Create masked array
		# print('tempresult_ma:', tempresults_ma)

		plt.imshow(tempresults_ma.astype(float), interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower',
				   extent=(dict_algo['min_max_range_parameter_0'][0], dict_algo['min_max_range_parameter_0'][1], dict_algo['min_max_range_parameter_1'][0], dict_algo['min_max_range_parameter_1'][1]),
				   vmin=std_treshold_determine_time_dependency, vmax=2 * np.pi)
		plt.title(r'std($\Delta$' + phi_string + ')')
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.xlim([1.05 * dict_algo['min_max_range_parameter_0'][0], 1.05 * dict_algo['min_max_range_parameter_0'][1]])
		plt.ylim([1.05 * dict_algo['min_max_range_parameter_1'][0], 1.05 * dict_algo['min_max_range_parameter_1'][1]])
		plt.colorbar()

		plt.savefig('results/param_space_%s_vs_%s_std_beta_%i%i_%d_%d_%d.png' % (dict_algo['param_id_0'], dict_algo['param_id_1'], j, jj, now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
		plt.savefig('results/param_space_%s_vs_%s_std_beta_%i%i_%d_%d_%d.svg' % (dict_algo['param_id_0'], dict_algo['param_id_1'], j, jj, now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		if add_scatter_plots:
			fig3 = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
			fig3.canvas.manager.set_window_title(r'last $\beta_{kl}$ in parameter space %s vs %s' % (dict_algo['param_id_0'], dict_algo['param_id_1']))
			fig3.set_size_inches(plot_size_inches_x, plot_size_inches_y)

			plt.clf()
			ax = plt.subplot(1, 1, 1)
			ax.set_aspect('equal')

			plt.scatter(dict_algo['allPoints'][:, 0], dict_algo['allPoints'][:, 1], c=beta_kl[j, :], alpha=0.5, cmap=colormap, vmin=np.min(beta_kl[j, :]), vmax=np.max(beta_kl[j, :]))
			plt.title(r'last $\Delta$'+phi_string)
			plt.xlabel(x_label)
			plt.ylabel(y_label)
			plt.xlim([1.05 * dict_algo['min_max_range_parameter_0'][0], 1.05 * dict_algo['min_max_range_parameter_0'][1]])
			plt.ylim([1.05 * dict_algo['min_max_range_parameter_1'][0], 1.05 * dict_algo['min_max_range_parameter_1'][1]])
			plt.colorbar()
			plt.savefig('results/param_space_%s_vs_%s_beta_%i%i_%d_%d_%d.png' % (dict_algo['param_id_0'], dict_algo['param_id_1'], j, jj, now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
			plt.savefig('results/param_space_%s_vs_%s_beta_%i%i_%d_%d_%d.svg' % (dict_algo['param_id_0'], dict_algo['param_id_1'], j, jj, now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
			# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			fig4 = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
			fig4.canvas.manager.set_window_title(r'std of $\beta_{kl}$ in parameter space %s vs %s' % (dict_algo['param_id_0'], dict_algo['param_id_1']))
			fig4.set_size_inches(plot_size_inches_x, plot_size_inches_y)

			plt.clf()
			ax = plt.subplot(1, 1, 1)
			ax.set_aspect('equal')
			plt.scatter(dict_algo['allPoints'][:, 0], dict_algo['allPoints'][:, 1], c=std_beta_kl[j, :], alpha=0.5, cmap=colormap, vmin=np.min(std_beta_kl[j, :]), vmax=np.max(std_beta_kl[j, :]))
			plt.title(r'std($\Delta$'+phi_string+')')
			plt.xlabel(x_label)
			plt.ylabel(y_label)
			plt.xlim([1.05 * dict_algo['min_max_range_parameter_0'][0], 1.05 * dict_algo['min_max_range_parameter_0'][1]])
			plt.ylim([1.05 * dict_algo['min_max_range_parameter_1'][0], 1.05 * dict_algo['min_max_range_parameter_1'][1]])
			plt.colorbar()
			plt.savefig('results/param_space_%s_vs_%s_std_beta_%i%i_%d_%d_%d.png' % (dict_algo['param_id_0'], dict_algo['param_id_1'], j, jj, now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
			plt.savefig('results/param_space_%s_vs_%s_std_beta_%i%i_%d_%d_%d.svg' % (dict_algo['param_id_0'], dict_algo['param_id_1'], j, jj, now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")

			figs.append([fig1, fig21, fig22, fig23, fig3, fig4])

		figs.append([fig1, fig21, fig22, fig23])

	return None


# ################################################################################################################################################################################


def plot_order_parameter_vs_initial_phase_configuration_space(pool_data: dict, average_time_order_parameter_in_periods: np.float, axis_normalization=True):
	"""Function that plots the last value and an average of the order parameter as a function of the initial phase-configuration  in two individual plots.
		The initial phase configuration is expressed as the phase differences for 2d and 3d phase spaces. Each plot comes as a scatterplot and an imshow.

		Args:
			pool_data: contains the results of all simulations (realizations), i.e., the phases of all oscillators, time dependent parameters, etc.
			average_time_order_parameter_in_periods: determines over how many periods of the intrinsic PLL frequency averages of the order parameter are performed
			axis_normalization: whether the axis of the plot are normalized or not: default True

		Returns:
			saves plotted data to files
	"""

	dict_net = pool_data[0][0]['dict_net']
	dict_algo = pool_data[0][0]['dict_algo']

	# prepare colormap for scatter plot that is always in [0, 1] or [min(results), max(results)]
	cdict = {
		'red': ((0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
		'green': ((0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
		'blue': ((0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
	}
	# cmaps['Diverging'] = [ 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

	# if (topology == 'entrainOne' or topology == 'entrainAll'):
	# 	# cmap=plt.cm.get_cmap('Blues', 6)
	# 	cmap 	  = plt.cm.get_cmap('seismic', 256);
	# 	cmap1 	  = plt.cm.get_cmap('seismic', 256);
	# 	colormap  = eva.shiftedColorMap(cmap,  start=0.0, midpoint=absR_conf, stop=1.0, name='shiftedcmap')
	# 	colormap1 = eva.shiftedColorMap(cmap1, start=0.0, midpoint=absR_conf, stop=1.0, name='shiftedcmap1')
	# 	print('absR_conf=', absR_conf, ', min(results[:,1])=', min(results[:,1]), ', max(results[:,1])=', max(results[:,1]))
	# else:
	colormap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)



	# we want to plot all the m-twist locations in rotated phase space: calculate phases, rotate and then plot into the results
	twist_points = np.zeros((dict_net['Nx']*dict_net['Ny'], dict_net['Nx']*dict_net['Ny']), dtype=np.float)  # twist points in physical phase space
	twist_pointsR = np.zeros((dict_net['Nx']*dict_net['Ny'], dict_net['Nx']*dict_net['Ny']), dtype=np.float)  # twist points in rotated phase space
	alltwistP = []

	if F_Omeg > 0:  		# for f=0, there would otherwise be a float division by zero
		F1 = F_Omeg
	else:
		F1 = 1.1

	if N == 2:  			# this part is for calculating the points of m-twist solutions in the rotated space, they are plotted later
		d1 = 0
		d2 = 1
		pass
	if N == 3:
		d1 = 1
		d2 = 2
		for i in range(dict_net['Nx']*dict_net['Ny']):
			twistdelta = (2.0 * np.pi * i / (1.0 * dict_net['Nx']*dict_net['Ny']))
			twist_points[i, :] = np.array([0.0, twistdelta, 2.0 * twistdelta])  # write m-twist phase configuation in phase space of phases
			# print(i,'-twist points:\n', twist_points[i,:], '\n')
			for m in range(-2, 3):
				for n in range(-2, 3):
					vtemp = twist_points[i, :] + np.array([0.0, 2.0 * np.pi * m, 2.0 * np.pi * n])
					alltwistP.append(vtemp)
		# print('vtemp:', vtemp, '\n')

		if 'entrain' in dict_net['topology']:
			alltwistP = phiConfig  # phase-configuration of entrained synced state
			if not len(phiConfig) == 0:
				R_config = eva.real_part_kuramoto_order_parameter(dict_net['phiInitConfig'])
				absR_conf = np.abs(R_config)
			alltwistPR = np.transpose(eva.rotate_phases(np.transpose(alltwistP), isInverse=True))  # express the points in rotated phase space
			print('value of unadjusted order parameter of the expected phase-configuration:', absR_conf)
		# phi_constant_expected = eva.rotate_phases(phiMr, isInverse=False)
		# r = eva.calcKuramotoOrderParEntrainSelfOrgState(phi[-int(numb_av_T*1.0/(F1*dt)):, :], phi_constant_expected);
		# order_parameter = eva.calcKuramotoOrderParEntrainSelfOrgState(phi[:, :], phi_constant_expected);
		else:
			alltwistP = np.array(alltwistP)
			# print('alltwistP:\n', alltwistP, '\n')
			alltwistPR = np.transpose(eva.rotate_phases(np.transpose(alltwistP), isInverse=True))  # express the points in rotated phase space
	# print('alltwistP rotated (alltwistPR):\n', alltwistPR, '\n')

	fig1 = plt.figure(figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig1.canvas.manager.set_window_title('parameter space of initial phase configuration')
	fig1.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	ax.set_aspect('equal')
	plt.scatter(allPoints[:, 0] + phiMr[d1], allPoints[:, 1] + phiMr[d2], c=results[:, 0], alpha=0.5, cmap=colormap, vmin=0, vmax=1)
	plt.title(r'mean $R(t,m=%d)$, constant dim: $\phi_0^{\prime}=%.2f$' % (int(k), initPhiPrime0))
	if N == 3:
		plt.xlabel(r'$\phi_1^{\prime}$')
		plt.ylabel(r'$\phi_2^{\prime}$')
	elif N == 2:
		plt.xlabel(r'$\phi_0^{\prime}$')
		plt.ylabel(r'$\phi_1^{\prime}$')
	if (N == 3 and not (topology == "square-open" or topology == "chain" or topology == "entrainOne" or topology == "entrainAll")):
		plt.plot(alltwistPR[:, 1], alltwistPR[:, 2], 'yo', ms=8)
	if (N == 3 and (topology == "entrainOne" or topology == "entrainAll")):
		plt.plot(alltwistPR[1], alltwistPR[2], 'yo', ms=2)
	plt.xlim([1.05 * allPoints[:, 0].min() + phiMr[d1], 1.05 * allPoints[:, 0].max() + phiMr[d1]])
	plt.ylim([1.05 * allPoints[:, 1].min() + phiMr[d2], 1.05 * allPoints[:, 1].max() + phiMr[d2]])
	plt.colorbar()

	t = np.arange(0, dict_net['max_delay_steps'] + dict_pll['sim_time_steps']) * dict_pll['dt']
	plt.plot(t, time_dependent_parameter)
	plt.xlabel('time')
	plt.ylabel(y_label)

	plt.savefig('results/param_space_%s_vs_%s_lastR_scatter_%d_%d_%d.png' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/param_space_%s_vs_%s_lastR_scatter_%d_%d_%d.svg' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")

	plt.savefig('results/param_space_%s_vs_%s_meanR_scatter_%d_%d_%d.png' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/param_space_%s_vs_%s_meanR_scatter_%d_%d_%d.svg' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")

	plt.savefig('results/param_space_%s_vs_%s_lastR_imshow_%d_%d_%d.png' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/param_space_%s_vs_%s_lastR_imshow_%d_%d_%d.svg' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")

	plt.savefig('results/param_space_%s_vs_%s_meanR_imshow_%d_%d_%d.png' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")
	plt.savefig('results/param_space_%s_vs_%s_meanR_imshow_%d_%d_%d.svg' % (dict_algo['param_id_0'], dict_algo['param_id_1'], now.year, now.month, now.day), dpi=dpi_val, bbox_inches="tight")

	''' IMPORTANT: since we add the perturbations additively, here we need to shift allPoints around the initial phases of the respective m-twist state, using phiMr '''
	plt.figure(1)
	# plot the mean of the order parameter over a period 2T
	plt.clf()
	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')
	plt.scatter(allPoints[:, 0] + phiMr[d1], allPoints[:, 1] + phiMr[d2], c=results[:, 0], alpha=0.5, cmap=colormap, vmin=0, vmax=1)
	plt.title(r'mean $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' % (int(k), initPhiPrime0))
	if N == 3:
		plt.xlabel(r'$\phi_1^{\prime}$')
		plt.ylabel(r'$\phi_2^{\prime}$')
	elif N == 2:
		plt.xlabel(r'$\phi_0^{\prime}$')
		plt.ylabel(r'$\phi_1^{\prime}$')
	if (N == 3 and not (topology == "square-open" or topology == "chain" or topology == "entrainOne" or topology == "entrainAll")):
		plt.plot(alltwistPR[:, 1], alltwistPR[:, 2], 'yo', ms=8)
	if (N == 3 and (topology == "entrainOne" or topology == "entrainAll")):
		plt.plot(alltwistPR[1], alltwistPR[2], 'yo', ms=2)
	plt.xlim([1.05 * allPoints[:, 0].min() + phiMr[d1], 1.05 * allPoints[:, 0].max() + phiMr[d1]])
	plt.ylim([1.05 * allPoints[:, 1].min() + phiMr[d2], 1.05 * allPoints[:, 1].max() + phiMr[d2]])
	plt.colorbar()
	plt.savefig('results/rot_red_PhSpac_meanR_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_%d_%d_%d.pdf' % (K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
	plt.savefig('results/rot_red_PhSpac_meanR_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_%d_%d_%d.png' % (K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure(2)
	plt.clf()
	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')
	plt.scatter(allPoints[:, 0] + phiMr[d1], allPoints[:, 1] + phiMr[d2], c=results[:, 1], alpha=0.5, cmap=colormap, vmin=0.0, vmax=1.0)
	plt.title(r'last $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' % (int(k), initPhiPrime0))
	if N == 3:
		plt.xlabel(r'$\phi_1^{\prime}$')
		plt.ylabel(r'$\phi_2^{\prime}$')
	elif N == 2:
		plt.xlabel(r'$\phi_0^{\prime}$')
		plt.ylabel(r'$\phi_1^{\prime}$')
	if (N == 3 and not (topology == "square-open" or topology == "chain" or topology == "entrainOne" or topology == "entrainAll")):
		plt.plot(alltwistPR[:, 1], alltwistPR[:, 2], 'yo', ms=8)
	if (N == 3 and (topology == "entrainOne" or topology == "entrainAll")):
		plt.plot(alltwistPR[1], alltwistPR[2], 'yo', ms=2)
	plt.xlim([1.05 * allPoints[:, 0].min() + phiMr[d1], 1.05 * allPoints[:, 0].max() + phiMr[d1]])
	plt.ylim([1.05 * allPoints[:, 1].min() + phiMr[d2], 1.05 * allPoints[:, 1].max() + phiMr[d2]])
	plt.colorbar()
	plt.savefig('results/rot_red_PhSpac_lastR_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_%d_%d_%d.pdf' % (K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
	plt.savefig('results/rot_red_PhSpac_lastR_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_%d_%d_%d.png' % (K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure(3)
	plt.clf()
	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')
	tempresults = results[:, 0].reshape((paramDiscretization, paramDiscretization))  # np.flipud()
	tempresults = np.transpose(tempresults)
	# print('tempresults:', tempresults)
	tempresults_ma = ma.masked_where(tempresults < 0, tempresults)  # Create masked array
	# print('tempresult_ma:', tempresults_ma)
	# print('initPhiPrime0:', initPhiPrime0)
	plt.imshow(tempresults_ma.astype(float), interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower',
			   extent=(allPoints[:, 0].min() + phiMr[d1], allPoints[:, 0].max() + phiMr[d1], allPoints[:, 1].min() + phiMr[d2], allPoints[:, 1].max() + phiMr[d2]), vmin=0, vmax=1)
	plt.title(r'mean $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' % (int(k), initPhiPrime0))
	if N == 3:
		plt.xlabel(r'$\phi_1^{\prime}$')
		plt.ylabel(r'$\phi_2^{\prime}$')
	elif N == 2:
		plt.xlabel(r'$\phi_0^{\prime}$')
		plt.ylabel(r'$\phi_1^{\prime}$')
	if (N == 3 and not (topology == "square-open" or topology == "chain" or topology == "entrainOne" or topology == "entrainAll")):
		plt.plot(alltwistPR[:, 1], alltwistPR[:, 2], 'yo', ms=8)
	if (N == 3 and (topology == "entrainOne" or topology == "entrainAll")):
		plt.plot(alltwistPR[1], alltwistPR[2], 'yo', ms=2)
	plt.xlim([1.05 * allPoints[:, 0].min() + phiMr[d1], 1.05 * allPoints[:, 0].max() + phiMr[d1]])
	plt.ylim([1.05 * allPoints[:, 1].min() + phiMr[d2], 1.05 * allPoints[:, 1].max() + phiMr[d2]])
	plt.colorbar()
	plt.savefig('results/imsh_PhSpac_meanR_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_%d_%d_%d.pdf' % (K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
	plt.savefig('results/imsh_PhSpac_meanR_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_%d_%d_%d.png' % (K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure(4)
	plt.clf()
	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')
	tempresults = results[:, 1].reshape((paramDiscretization, paramDiscretization))  # np.flipud()
	tempresults = np.transpose(tempresults)
	tempresults_ma = ma.masked_where(tempresults < 0, tempresults)  # Create masked array
	plt.imshow(tempresults_ma.astype(float), interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower',
			   extent=(allPoints[:, 0].min() + phiMr[d1], allPoints[:, 0].max() + phiMr[d1], allPoints[:, 1].min() + phiMr[d2], allPoints[:, 1].max() + phiMr[d2]), vmin=0, vmax=1)
	plt.title(r'last $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' % (int(k), initPhiPrime0))
	if N == 3:
		plt.xlabel(r'$\phi_1^{\prime}$')
		plt.ylabel(r'$\phi_2^{\prime}$')
	elif N == 2:
		plt.xlabel(r'$\phi_0^{\prime}$')
		plt.ylabel(r'$\phi_1^{\prime}$')
	if (N == 3 and not (topology == "square-open" or topology == "chain" or topology == "entrainOne" or topology == "entrainAll")):
		plt.plot(alltwistPR[:, 1], alltwistPR[:, 2], 'yo', ms=8)
	if (N == 3 and (topology == "entrainOne" or topology == "entrainAll")):
		plt.plot(alltwistPR[1], alltwistPR[2], 'yo', ms=2)
	plt.xlim([1.05 * allPoints[:, 0].min() + phiMr[d1], 1.05 * allPoints[:, 0].max() + phiMr[d1]])
	plt.ylim([1.05 * allPoints[:, 1].min() + phiMr[d2], 1.05 * allPoints[:, 1].max() + phiMr[d2]])
	plt.colorbar()
	plt.savefig('results/imsh_PhSpac_lastR_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_%d_%d_%d.pdf' % (K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
	plt.savefig('results/imsh_PhSpac_lastR_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_%d_%d_%d.png' % (K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	# plt.figure(5)
	# plt.clf()
	# ax = plt.subplot(1, 1, 1)
	# ax.set_aspect('equal')
	# plt.scatter(allPoints[:,0]+phiMr[d1], allPoints[:,1]+phiMr[d2], c=results[:,0], alpha=0.5, cmap='jet')#, vmin=0, vmax=1)
	# plt.title(r'mean $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' %(int(k) ,initPhiPrime0) )
	# if N==3:
	# 	plt.xlabel(r'$\phi_1^{\prime}$')
	# 	plt.ylabel(r'$\phi_2^{\prime}$')
	# elif N==2:
	# 	plt.xlabel(r'$\phi_0^{\prime}$')
	# 	plt.ylabel(r'$\phi_1^{\prime}$')
	# if N==3 and topology != "square-open" and topology != "chain":
	# 	plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=4)
	# plt.xlim([1.05*allPoints[:,0].min()+phiMr[d1], 1.05*allPoints[:,0].max()+phiMr[d1]])
	# plt.ylim([1.05*allPoints[:,1].min()+phiMr[d2], 1.05*allPoints[:,1].max()+phiMr[d2]])
	# plt.colorbar()
	# plt.savefig('results/rot_redColor_PhSpac_meanR_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
	# plt.savefig('results/rot_redColor_PhSpac_meanR_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure(5)
	plt.clf()
	my_cmap = matplotlib.cm.get_cmap('jet')
	my_cmap.set_under('w')
	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')
	plt.scatter(allPoints[:, 0] + phiMr[d1], allPoints[:, 1] + phiMr[d2], c=results[:, 0], s=10, alpha=0.5, cmap=my_cmap, vmin=0.0, vmax=1.0)  # , vmin=0, vmax=1)
	plt.title(r'mean $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' % (int(k), initPhiPrime0))
	if N == 3:
		plt.xlabel(r'$\phi_1^{\prime}$')
		plt.ylabel(r'$\phi_2^{\prime}$')
	elif N == 2:
		plt.xlabel(r'$\phi_0^{\prime}$')
		plt.ylabel(r'$\phi_1^{\prime}$')
	if (N == 3 and not (topology == "square-open" or topology == "chain" or topology == "entrainOne" or topology == "entrainAll")):
		plt.plot(alltwistPR[:, 1], alltwistPR[:, 2], 'yo', ms=8)
	if (N == 3 and (topology == "entrainOne" or topology == "entrainAll")):
		plt.plot(alltwistPR[1], alltwistPR[2], 'yo', ms=2)
	plt.xlim([1.05 * allPoints[:, 0].min() + phiMr[d1], 1.05 * allPoints[:, 0].max() + phiMr[d1]])
	plt.ylim([1.05 * allPoints[:, 1].min() + phiMr[d2], 1.05 * allPoints[:, 1].max() + phiMr[d2]])
	plt.colorbar()
	plt.savefig('results/rot_redColor_PhSpac_meanR_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_%d_%d_%d.pdf' % (K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
	plt.savefig('results/rot_redColor_PhSpac_meanR_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_%d_%d_%d.png' % (K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure(6)
	plt.clf()
	my_cmap = matplotlib.cm.get_cmap('jet')
	my_cmap.set_under('w')
	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')
	plt.scatter(allPoints[:, 0] + phiMr[d1], allPoints[:, 1] + phiMr[d2], c=results[:, 0], s=5, alpha=0.5, cmap=my_cmap, vmin=0.0, vmax=1.0)  # , vmin=0, vmax=1)
	plt.title(r'mean $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' % (int(k), initPhiPrime0))
	if N == 3:
		plt.xlabel(r'$\phi_1^{\prime}$')
		plt.ylabel(r'$\phi_2^{\prime}$')
	elif N == 2:
		plt.xlabel(r'$\phi_0^{\prime}$')
		plt.ylabel(r'$\phi_1^{\prime}$')
	if (N == 3 and not (topology == "square-open" or topology == "chain" or topology == "entrainOne" or topology == "entrainAll")):
		plt.plot(alltwistPR[:, 1], alltwistPR[:, 2], 'yo', ms=8)
	if (N == 3 and (topology == "entrainOne" or topology == "entrainAll")):
		plt.plot(alltwistPR[1], alltwistPR[2], 'yo', ms=2)
	plt.xlim([1.05 * allPoints[:, 0].min() + phiMr[d1], 1.05 * allPoints[:, 0].max() + phiMr[d1]])
	plt.ylim([1.05 * allPoints[:, 1].min() + phiMr[d2], 1.05 * allPoints[:, 1].max() + phiMr[d2]])
	plt.colorbar()
	plt.savefig('results/rot_redColor5_PhSpac_meanR_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_%d_%d_%d.pdf' % (K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
	plt.savefig('results/rot_redColor5_PhSpac_meanR_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_%d_%d_%d.png' % (K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.draw()
	if show_plot:
		plt.show()
	plt.show()

	return None

# ################################################################################################################################################################################


#
# fig110 = plt.figure()
# fig110.set_size_inches(plot_size_inches_x, plot_size_inches_y)
#
# ax011 = fig110.add_subplot(211)
#
# plt.axvspan(0, t[int(1.0*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt']))]*dict_pll['dt'], color='b', alpha=0.25)
# for i in range(len(phidot[0,:])):
# 	plt.plot((t[0:-1:dict_pll['sampleFplot']]*dict_pll['dt']), phidot[::dict_pll['sampleFplot'],i]/(2.0*np.pi*dict_data['F1']), label='PLL%i' %(i), linewidth=1)
#
# plt.ylabel(r'$f(t)/f0=\dot{\phi}(t)/\omega$',rotation=90, fontsize=60, labelpad=40)
# ax011.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.xlim([0, t[-1]*dict_pll['dt']])
# plt.grid();
#
# ax012 = fig110.add_subplot(212)
#
# plt.plot((t*dict_pll['dt'])[::dict_pll['sampleFplot']], dict_data['order_parameter'][::dict_pll['sampleFplot']], linewidth=1.5)
# plt.xlabel(r'$t\,[T_{\omega}]$', fontsize=60, labelpad=-5)
# plt.ylabel(r'$R(t)$',rotation=90, fontsize=60, labelpad=40)
# ax012.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.xlim([0, t[-1]*dict_pll['dt']])
# plt.grid();
#
# plt.savefig('results/freq_orderP_allT_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches = "tight")
# plt.savefig('results/freq_orderP_allT_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches = "tight")
# ax011.set_ylim([0, 1.05*np.max(phidot[::dict_pll['sampleFplot'],:]/(2.0*np.pi*dict_data['F1']))])
# ax012.set_ylim([0, 1.05])
# plt.savefig('results/freq_orderP_allT_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches = "tight")
# plt.savefig('results/freq_orderP_allT_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches = "tight")
#
# ################################################################################################################################################################################
# # plot instantaneous frequency and phase-differences
# multPrior = 5.0; multAfter = 95.0;
# if np.mean(dict_pll['transmission_delay'])/dict_pll['dt'] > multPrior/(dict_data['F1']*dict_pll['dt']):
# 	priorStart = multPrior/(dict_data['F1']*dict_pll['dt'])											# start plot 'multPrior' periods before start of simulation
# else:
# 	priorStart = int(0.5*np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])
# if np.mean(dict_pll['transmission_delay'])/dict_pll['dt'] > multAfter/(dict_data['F1']*dict_pll['dt']):
# 	afterStart = multPrior/(dict_data['F1']*dict_pll['dt'])											# end plot after 'multAfter' periods following the start of the simulation
# else:
# 	afterStart = int(0.25*Tsim/dt)
# multStartFin = 110.0; multEndFin = 15.0;
# if len(t) > multStartFin/(dict_data['F1']*dict_pll['dt']):
# 	multStartFin = multStartFin/(dict_data['F1']*dict_pll['dt'])										# start plot 'multStartFin' periods before end of simulation
# else:
# 	multStartFin = int(0.65*Tsim/dt)
# if multStartFin > multEndFin:
# 	multEndFin = multEndFin/(dict_data['F1']*dict_pll['dt'])											# end plot after 'multEndFin' periods before end of simulation
# else:
# 	multEndFin = int(0.95*Tsim/dt)
#
# xstart1 = int(np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])-priorStart);		xstart2 = len(t)-int(multStartFin);
# xend1	= int(np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])+afterStart);		xend2   = len(t)-int(multEndFin);
# xmin1 	= t[xstart1]*dict_pll['dt']; xmax1 	= t[xend1]*dict_pll['dt'];
# xmin2 	= t[xstart2]*dict_pll['dt']; xmax2 	= t[xend2]*dict_pll['dt'];
#
# fig111 = plt.figure(figsize=(figwidth,figheight))
# fig111.set_size_inches(plot_size_inches_x, plot_size_inches_y)
#
# ax111 = fig111.add_subplot(221)
#
# ''' PLOT HERE THE FIRST PART '''
# plt.axvspan(0, t[int(1.0*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt']))]*dict_pll['dt'], color='b', alpha=0.25)
# for i in range(len(phidot[0,:])):
# 	plt.plot((t[xstart1:xend1:dict_pll['sampleFplot']]*dict_pll['dt']), phidot[xstart1:xend1:dict_pll['sampleFplot'],i]/(2.0*np.pi*dict_data['F1']), label='PLL%i' %(i), linewidth=1)
#
# plt.ylabel(r'$f(t)/f0=\dot{\phi}(t)/\omega$',rotation=90, fontsize=60, labelpad=40)
# ax111.set_xlim(xmin1, xmax1)
# ax111.set_ylim(0.99*np.min([np.min(phidot[xstart1:xend1:dict_pll['sampleFplot'],:]/(2.0*np.pi*dict_data['F1'])), np.min(phidot[xstart2:xend2:dict_pll['sampleFplot'],:]/(2.0*np.pi*dict_data['F1']))]),
# 			   1.01*np.max([np.max(phidot[xstart1:xend1:dict_pll['sampleFplot'],:]/(2.0*np.pi*dict_data['F1'])), np.max(phidot[xstart2:xend2:dict_pll['sampleFplot'],:]/(2.0*np.pi*dict_data['F1']))]) )
#
# ax111.tick_params(axis='both', which='major', labelsize=35, pad=1)
# plt.grid();
#
# ax112 = fig111.add_subplot(222)
#
# ''' PLOT HERE THE SECOND PART '''
# for i in range(len(phidot[0,:])):
# 	plt.plot((t[xstart2:xend2:dict_pll['sampleFplot']]*dict_pll['dt']), phidot[xstart2:xend2:dict_pll['sampleFplot'],i]/(2.0*np.pi*dict_data['F1']), label='PLL%i' %(i), linewidth=1)
#
# #plt.xlabel(r'$time$',fontsize=60,labelpad=-5)
# ax112.set_xlim(xmin2, xmax2)
# ax112.set_ylim(0.99*np.min([np.min(phidot[xstart1:xend1:dict_pll['sampleFplot'],:]/(2.0*np.pi*dict_data['F1'])), np.min(phidot[xstart2:xend2:dict_pll['sampleFplot'],:]/(2.0*np.pi*dict_data['F1']))]),
# 			   1.01*np.max([np.max(phidot[xstart1:xend1:dict_pll['sampleFplot'],:]/(2.0*np.pi*dict_data['F1'])), np.max(phidot[xstart2:xend2:dict_pll['sampleFplot'],:]/(2.0*np.pi*dict_data['F1']))]) )
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
# plt.axvspan(0, t[int(1.0*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt']))]*dict_pll['dt'], color='b', alpha=0.25)
# plt.plot((t*dict_pll['dt'])[xstart1:xend1:dict_pll['sampleFplot']], dict_data['order_parameter'][xstart1:xend1:dict_pll['sampleFplot']], linewidth=1.5)
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
# plt.plot((t*dict_pll['dt'])[xstart2:xend2:dict_pll['sampleFplot']], dict_data['order_parameter'][xstart2:xend2:dict_pll['sampleFplot']], linewidth=1.5)
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
# plt.savefig('results/freq_orderP_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches = "tight")
# plt.savefig('results/freq_orderP_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches = "tight")
#
# multPrior = 5.0; multAfter = 20.0;
# if np.mean(dict_pll['transmission_delay'])/dict_pll['dt'] > multPrior/(dict_data['F1']*dict_pll['dt']):
# 	priorStart = multPrior/(dict_data['F1']*dict_pll['dt'])											# start plot 'multPrior' periods before start of simulation
# else:
# 	priorStart = int(0.1*np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])
# if np.mean(dict_pll['transmission_delay'])/dict_pll['dt'] > multAfter/(dict_data['F1']*dict_pll['dt']):
# 	afterStart = multPrior/(dict_data['F1']*dict_pll['dt'])											# end plot after 'multAfter' periods following the start of the simulation
# else:
# 	afterStart = int(0.1*Tsim/dt)
# multStartFin = 40.0; multEndFin = 15.0;
# if len(t) > multStartFin/(dict_data['F1']*dict_pll['dt']):
# 	multStartFin = multStartFin/(dict_data['F1']*dict_pll['dt'])										# start plot 'multStartFin' periods before end of simulation
# else:
# 	multStartFin = int(0.85*Tsim/dt)
# if multStartFin > multEndFin:
# 	multEndFin = multEndFin/(dict_data['F1']*dict_pll['dt'])											# end plot after 'multEndFin' periods before end of simulation
# else:
# 	multEndFin = int(0.95*Tsim/dt)
#
# xstart1 = int(np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])-priorStart);		xstart2 = len(t)-int(multStartFin);
# xend1	= int(np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])+afterStart);		xend2   = len(t)-int(multEndFin);
# xmin1 	= t[xstart1]*dict_pll['dt']; xmax1 	= t[xend1]*dict_pll['dt'];
# xmin2 	= t[xstart2]*dict_pll['dt']; xmax2 	= t[xend2]*dict_pll['dt'];
#
# ax111.set_xlim(xmin1, xmax1)
# ax112.set_xlim(xmin2, xmax2)
# ax121.set_xlim(xmin1, xmax1)
# ax122.set_xlim(xmin2, xmax2)
#
# plt.savefig('results/freq_orderP_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches = "tight")
# plt.savefig('results/freq_orderP_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches = "tight")
#
# ################################################################################################################################################################################
# multPrior = 5.0; multAfter = 95.0;
# if np.mean(dict_pll['transmission_delay'])/dict_pll['dt'] > multPrior/(dict_data['F1']*dict_pll['dt']):
# 	priorStart = multPrior/(dict_data['F1']*dict_pll['dt'])											# start plot 'multPrior' periods before start of simulation
# else:
# 	priorStart = int(0.5*np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])
# if np.mean(dict_pll['transmission_delay'])/dict_pll['dt'] > multAfter/(dict_data['F1']*dict_pll['dt']):
# 	afterStart = multPrior/(dict_data['F1']*dict_pll['dt'])											# end plot after 'multAfter' periods following the start of the simulation
# else:
# 	afterStart = int(0.25*Tsim/dt)
# multStartFin = 110.0; multEndFin = 15.0;
# if len(t) > multStartFin/(dict_data['F1']*dict_pll['dt']):
# 	multStartFin = multStartFin/(dict_data['F1']*dict_pll['dt'])										# start plot 'multStartFin' periods before end of simulation
# else:
# 	multStartFin = int(0.65*Tsim/dt)
# if multStartFin > multEndFin:
# 	multEndFin = multEndFin/(dict_data['F1']*dict_pll['dt'])											# end plot after 'multEndFin' periods before end of simulation
# else:
# 	multEndFin = int(0.95*Tsim/dt)
#
# xstart1 = int(np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])-priorStart);		xstart2 = len(t)-int(multStartFin);
# xend1	= int(np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])+afterStart);		xend2   = len(t)-int(multEndFin);
# xmin1 	= t[xstart1]*dict_pll['dt']; xmax1 	= t[xend1]*dict_pll['dt'];
# xmin2 	= t[xstart2]*dict_pll['dt']; xmax2 	= t[xend2]*dict_pll['dt'];
#
# fig211 = plt.figure(figsize=(figwidth,figheight))
# fig211.set_size_inches(plot_size_inches_x, plot_size_inches_y)
#
# ax211 = fig211.add_subplot(221)
#
# ''' PLOT HERE THE FIRST PART '''
# plt.axvspan(0, t[int(1.0*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt']))]*dict_pll['dt'], color='b', alpha=0.25)
# for i in range(len(phi[0,:])):
# 	plt.plot((t[xstart1:xend1:dict_pll['sampleFplot']]*dict_pll['dt']), phi[xstart1:xend1:dict_pll['sampleFplot'],i]%(2.*np.pi), label='PLL%i' %(i), linewidth=1)
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
# 	plt.plot((t[xstart2:xend2:dict_pll['sampleFplot']]*dict_pll['dt']), phi[xstart2:xend2:dict_pll['sampleFplot'],i]%(2.*np.pi), label='PLL%i' %(i), linewidth=1)
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
# plt.axvspan(0, t[int(1.0*np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt']))]*dict_pll['dt'], color='b', alpha=0.25)
# for i in range(len(phi[0,:])):
# 	labelname = r'$\phi_{%i}$-$\phi_{0}$' %(i);
# 	plt.plot((t[xstart1:xend1:dict_pll['sampleFplot']]*dict_pll['dt']),
# 			((phi[xstart1:xend1:dict_pll['sampleFplot'],i]-phi[xstart1:xend1:dict_pll['sampleFplot'],0]+np.pi)%(2.*np.pi))-np.pi, label=labelname, linewidth=1)
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
# 	plt.plot((t[xstart2:xend2:dict_pll['sampleFplot']]*dict_pll['dt']),
# 			((phi[xstart2:xend2:dict_pll['sampleFplot'],i]-phi[xstart2:xend2:dict_pll['sampleFplot'],0]+np.pi)%(2.*np.pi))-np.pi, label=labelname, linewidth=1)
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
# plt.savefig('results/phases_phaseDiff_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches = "tight")
# plt.savefig('results/phases_phaseDiff_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches = "tight")
#
# multPrior = 5.0; multAfter = 20.0;
# if np.mean(dict_pll['transmission_delay'])/dict_pll['dt'] > multPrior/(dict_data['F1']*dict_pll['dt']):
# 	priorStart = multPrior/(dict_data['F1']*dict_pll['dt'])											# start plot 'multPrior' periods before start of simulation
# else:
# 	priorStart = int(0.1*np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])
# if np.mean(dict_pll['transmission_delay'])/dict_pll['dt'] > multAfter/(dict_data['F1']*dict_pll['dt']):
# 	afterStart = multPrior/(dict_data['F1']*dict_pll['dt'])											# end plot after 'multAfter' periods following the start of the simulation
# else:
# 	if Tsim > 2*delay:
# 		afterStart = int(2.0*np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])
# 	else:
# 		afterStart = int(0.05*Tsim/dt)
# multStartFin = 40.0; multEndFin = 15.0;
# if len(t) > multStartFin/(dict_data['F1']*dict_pll['dt']):
# 	multStartFin = multStartFin/(dict_data['F1']*dict_pll['dt'])										# start plot 'multStartFin' periods before end of simulation
# else:
# 	multStartFin = int(0.85*Tsim/dt)
# if multStartFin > multEndFin:
# 	multEndFin = multEndFin/(dict_data['F1']*dict_pll['dt'])											# end plot after 'multEndFin' periods before end of simulation
# else:
# 	multEndFin = int(0.95*Tsim/dt)
#
# xstart1 = int(np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])-priorStart);		xstart2 = len(t)-int(multStartFin);
# xend1	= int(np.round(np.mean(dict_pll['transmission_delay'])/dict_pll['dt'])+afterStart);		xend2   = len(t)-int(multEndFin);
# xmin1 	= t[xstart1]*dict_pll['dt']; xmax1 	= t[xend1]*dict_pll['dt'];
# xmin2 	= t[xstart2]*dict_pll['dt']; xmax2 	= t[xend2]*dict_pll['dt'];
#
# ax211.set_xlim(xmin1, xmax1)
# ax212.set_xlim(xmin2, xmax2)
# ax221.set_xlim(xmin1, xmax1)
# ax222.set_xlim(xmin2, xmax2)
#
# plt.savefig('results/phases_phaseDiff_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.png' %(np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches = "tight")
# plt.savefig('results/phases_phaseDiff_2_K%.4f_Fc%.4f_FOm%.4f_tau%.4f_c%.7e_%d_%d_%d.svg' %(np.mean(dict_pll['coupK']), np.mean(dict_pll['cutFc']), np.mean(dict_pll['syncF']), np.mean(dict_pll['transmission_delay']), np.mean(dict_pll['noiseVarVCO']), now.year, now.month, now.day), dpi=dpi_val, bbox_inches = "tight")
#
# ################################################################################################################################################################################
