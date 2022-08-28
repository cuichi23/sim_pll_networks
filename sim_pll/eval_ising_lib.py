#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import sys, gc
import inspect

import networkx as nx
import numpy as np
import scipy
from scipy import signal
import multiprocessing as mp
from multiprocessing import Pool, freeze_support
import itertools
from itertools import permutations as permu
from itertools import combinations as combi
import matplotlib
import os, pickle
if not os.environ.get('SGE_ROOT') is None:										# this environment variable is set within the queue network, i.e., if it exists, 'Agg' mode to supress output
	print('NOTE: \"matplotlib.use(\'Agg\')\"-mode active, plots are not shown on screen, just saved to results folder!\n')
	matplotlib.use('Agg') #'%pylab inline'
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from scipy.ndimage.filters import uniform_filter1d
import time
import datetime
import pandas as pd

from sim_pll import plot_lib
from sim_pll import evaluation_lib as eva

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


def evaluateSimulationIsing(poolData: dict, phase_wrap=0, number_of_histogram_bins=25, prob_density=False, order_param_solution=0.0, number_of_expected_oscis_in_one_group=10) -> None:
	"""
		Evaluates the simulations with SHIL, solving the MAX-cut problem in this case

		Args:
			poolData: [dict] contains all the data of the simulations to be evaluated and the settings
			phase_wrap: [integer] whether phases are wrapped into the interval 0) [0, 2*pi), 1) [-pi, pi), or 2) [-pi/2, 3*pi/2)
			number_of_histogram_bins: [integer] the number of bins of the histogram of phases plotted for the final state of the simulation
			prob_density: [boolean] whether histograms are normalized to become probability densities
			order_param_solution: [float] expected order parameter of the asymptotic state if the correct benchmark solution has been found
			number_of_expected_oscis_in_one_group: [integer] number of oscillators in the one of the two groups that form asymptotically

		TODO:
		1) reorganize to a class
		2) two modes, benchmark mode with different benchmark problems and operation mode, i.e., reading out the result of an in silico annealing
		3) structure into functions and simplify

		Returns:
		"""

	# plot parameter
	axisLabel = 9
	legendLab = 6
	tickSize = 5
	titleLabel = 9
	dpi_val = 150
	figwidth = 6
	figheight = 5
	linewidth = 0.8
	plot_size_inches_x = 10
	plot_size_inches_y = 5
	labelpadxaxis = 10
	labelpadyaxis = 20
	alpha = 0.5

	threshold_realizations_plot = 8

	if phase_wrap == 1:				# plot phase-differences in [-pi, pi) interval
		shift2piWin = np.pi
	elif phase_wrap == 2:			# plot phase-differences in [-pi/2, 3*pi/2) interval
		shift2piWin = 0.5*np.pi
	elif phase_wrap == 3:			# plot phase-differences in [0, 2*pi) interval
		shift2piWin = 0

	#unit_cell = PhaseDifferenceCell(poolData[0][0]['dictNet']['Nx']*poolData[0][0]['dictNet']['Ny'])
	threshold_statState = np.pi/15
	plotEveryDt = 1
	numberColsPlt = 3
	numberColsPlt_widePlt = 1
	number_of_intrinsic_periods_smoothing = 1.5
	print('For smoothing of phase-differences and order parameters we average over %0.2f periods of the ensemble mean intrinsic frequency.' % number_of_intrinsic_periods_smoothing)

	fig16, ax16 = plt.subplots(int(np.ceil(len(poolData[0][:])/numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig16.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, phase relations')					# phase relations
	if isinstance( poolData[0][0]['dictPLL']['cutFc'], np.float):
		fig16.suptitle(r'parameters: $f=$%0.2f Hz, $K=$%0.2f Hz/V, $K^\textrm{I}=$%0.2f Hz/V, $f_c=$%0.2f Hz, $\tau=$%0.2f s'%(poolData[0][0]['dictPLL']['intrF'], poolData[0][0]['dictPLL']['coupK'], poolData[0][0]['dictPLL']['coupStr_2ndHarm'], poolData[0][0]['dictPLL']['cutFc'], poolData[0][0]['dictPLL']['transmission_delay']))
	fig16.subplots_adjust(hspace=0.4, wspace=0.4)
	ax16 = ax16.ravel()

	fig161, ax161 = plt.subplots(int(np.ceil(len(poolData[0][:]) / numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig161.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, smoothed out phase relations')  # phase relations
	if isinstance(poolData[0][0]['dictPLL']['cutFc'], np.float):
		fig161.suptitle(r'parameters: $f=$%0.2f Hz, $K=$%0.2f Hz/V, $K^\textrm{I}=$%0.2f Hz/V, $f_c=$%0.2f Hz, $\tau=$%0.2f s' % (
		poolData[0][0]['dictPLL']['intrF'], poolData[0][0]['dictPLL']['coupK'], poolData[0][0]['dictPLL']['coupStr_2ndHarm'], poolData[0][0]['dictPLL']['cutFc'],
		poolData[0][0]['dictPLL']['transmission_delay']))
	fig161.subplots_adjust(hspace=0.4, wspace=0.4)
	ax161 = ax161.ravel()

	fig17, ax17 = plt.subplots(int(np.ceil(len(poolData[0][:])/numberColsPlt_widePlt)), numberColsPlt_widePlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig17.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, inst. frequencies')					# inst. frequencies
	fig17.subplots_adjust(hspace=0.4, wspace=0.4)
	ax17 = ax17.ravel()

	fig18, ax18 = plt.subplots(int(np.ceil(len(poolData[0][:])/numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig18.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, order parameter')					# order parameter
	fig18.subplots_adjust(hspace=0.4, wspace=0.4)
	ax18 = ax18.ravel()

	fig19, ax19 = plt.subplots(int(np.ceil(len(poolData[0][:])/numberColsPlt_widePlt)), numberColsPlt_widePlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig19.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, signals')							# signals
	fig19.subplots_adjust(hspace=0.4, wspace=0.4)
	ax19 = ax19.ravel()

	fig20, ax20 = plt.subplots(int(np.ceil(len(poolData[0][:]) / numberColsPlt)), numberColsPlt, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig20.canvas.manager.set_window_title('Ising different initial conditions, fixed topology, histograms')  # signals
	fig20.subplots_adjust(hspace=0.4, wspace=0.4)
	ax20 = ax20.ravel()

	fig99, ax99 = plt.subplots(1, 1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig99.canvas.manager.set_window_title('Network view of result.')  # network

	if len(poolData[0][:]) > threshold_realizations_plot: # only plot when many realizations are computed for overview
		fig21, ax21 = plt.subplots(1, 1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
		fig21.canvas.manager.set_window_title('all order parameters (solution correct: solid, incorrect: dashed line)')  # all order parameters
		fig21.set_size_inches(plot_size_inches_x, plot_size_inches_y)

		fig211, ax211 = plt.subplots(1, 1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
		fig211.canvas.manager.set_window_title('all order parameters smoothed (solution correct: solid, incorrect: dashed line)')  # all order parameters
		fig211.set_size_inches(plot_size_inches_x, plot_size_inches_y)

		fig22, ax22 = plt.subplots(1, 1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
		fig22.canvas.manager.set_window_title('distribution of times to solution')  # all order parameters
		fig22.set_size_inches(plot_size_inches_x, plot_size_inches_y)

	# print('poolData in eva.evaluateSimulationIsing(poolData):', poolData)

	print('For evaluation of asymptotic order parameter we average over %0.2f periods of the ensemble mean intrinsic frequency.' % number_of_intrinsic_periods_smoothing)

	sol_time = []
	success_count = 0
	success_count_test1 = 0
	group_oscillators_maxcut = np.zeros([len(poolData[0][:]), len(poolData[0][0]['dictData']['phi'][0, :])])

	# loop over the realizations
	for i in range(len(poolData[0][:])):
		deltaTheta = np.zeros([len(poolData[0][i]['dictData']['phi'][0, :]), len(poolData[0][i]['dictData']['phi'][:, 0])])
		signalOut  = np.zeros([len(poolData[0][i]['dictData']['phi'][0, :]), len(poolData[0][i]['dictData']['phi'][:, 0])])

		thetaDot = np.diff( poolData[0][i]['dictData']['phi'][:, :], axis=0 ) / poolData[0][i]['dictPLL']['dt']				# compute frequencies and order parameter
		r, orderparam, F1 = eva.obtainOrderParam(poolData[0][i]['dictPLL'], poolData[0][i]['dictNet'], poolData[0][i]['dictData'])

		ax18[i].plot( poolData[0][i]['dictData']['t'][::plotEveryDt], orderparam[::plotEveryDt], label=r'$R_\textrm{final}=%0.2f$'%(orderparam[-1]), linewidth=linewidth )

		# HOWTO 1) to determine whether the correct solution has be found, we test for the asymptotic value of the order parameter
		order_param_diff_expected_value_threshold = 0.01
		correct_solution_test0 = False
		if np.abs(np.mean(orderparam[-int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']):]) - order_param_solution) < order_param_diff_expected_value_threshold:
			print('Order parameter predicted for solution=%0.2f has been reached. Averaged over last %i periods of the intrinsic frequency for realization %i.' % (order_param_solution, number_of_intrinsic_periods_smoothing, i))
			success_count += 1					# to calculate the probability of finding the correct solutions
			correct_solution_test0 = True		# this is needed to decide for which realizations we need to measure the time to solution

		# HOWTO 2) to determine whether the correct solution has be found, we also test for mutual phase-differences between the oscillators
		group1 = 0
		group2 = 0
		correct_solution_test1 = False
		final_phase_oscillator = []
		for j in range(len(poolData[0][i]['dictData']['phi'][0, :])):
			# calculate mean phase difference over an interval of 'number_of_intrinsic_periods_smoothing' periods at the end of all oscillators with respect to oscillator zero
			# interval_index = -int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt'])
			temp_phase_diff = np.mean(poolData[0][i]['dictData']['phi'][-int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']):, 0] - poolData[0][i]['dictData']['phi'][-int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']):, j])
			print('Realization %i, mean phase difference {mod 2pi into [-pi, pi)} between k=0 and k=%i is deltaPhi=%0.2f'%(i, j, ((temp_phase_diff+np.pi) % (2*np.pi))-np.pi))
			if np.abs(((temp_phase_diff+np.pi) % (2*np.pi))-np.pi) < np.pi / 2:
				group1 += 1
				group_oscillators_maxcut[i, j] = -1
				final_phase_oscillator.append('zero')
			else: 																		# elif np.abs(((temp_phase_diff + np.pi) % (2 * np.pi)) - np.pi) >= np.pi / 2:
				group2 += 1
				group_oscillators_maxcut[i, j] = +1
				final_phase_oscillator.append('pi')
			# else:																		# this should in principle never happen!
			# 	group_oscillators_maxcut[i, j] = 0
			# 	final_phase_oscillator.append('diff')

		if not group1+group2 == len(poolData[0][i]['dictData']['phi'][0, :]):
			print('ERROR: check!')
			sys.exit()
		if group1 == number_of_expected_oscis_in_one_group or group2 == number_of_expected_oscis_in_one_group:
			success_count_test1 += 1
			correct_solution_test1 = True

		# HOWTO 3) to determine the time to solutions; find the time at the which the asymptotic value of the order parameter has been reached, we count from the start time increasing one of the coupling strengths
		order_param_std_threshold = 0.005
		smoothing = True
		if smoothing:
			if correct_solution_test1:
				# when the derivative of the order parameter is close to zero, we expect that the asymptotic state has been reached
				# here we look for the cases where this is NOT the case yet, then the last entry of the resulting vector will be the transition time from transient to asymptotic dynamics
				# note that in poolData[0][i]['dictData']['tstep_annealing_start'] the delay steps are ALREADY included!
				derivative_order_param_smoothed = (np.diff( uniform_filter1d( orderparam[poolData[0][i]['dictNet']['max_delay_steps']:],
					size=int(15 * number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']), mode='reflect') ) / poolData[0][i]['dictPLL']['dt'])

				rolling_std_derivative_order_param_smoothed = pd.Series(derivative_order_param_smoothed).rolling(int(15 * number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt'])).std()

				#temp = np.where( (np.diff( uniform_filter1d( orderparam[(poolData[0][i]['dictData']['tstep_annealing_start'] + poolData[0][i]['dictNet']['max_delay_steps']):],
				#	size=int(15 * number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']), mode='reflect') ) / poolData[0][i]['dictPLL']['dt']) > order_param_change_threshold )
				min_std = np.min(rolling_std_derivative_order_param_smoothed)
				print('min_std:', min_std)
				max_std = np.max(rolling_std_derivative_order_param_smoothed)
				order_param_std_threshold = 0.075 * (max_std - min_std) + min_std
				print('Realization %i, order_param_std_threshold to %0.05f, for {min_std, max_std} = {%0.4f,%0.4f} '%(i, order_param_std_threshold, min_std, max_std))
				temp = np.where(rolling_std_derivative_order_param_smoothed[(poolData[0][i]['dictData']['tstep_annealing_start']-poolData[0][i]['dictNet']['max_delay_steps']):] > order_param_std_threshold)

				# plt.plot(derivative_order_param_smoothed, 'b')
				# plt.plot(rolling_std_derivative_order_param_smoothed, 'r--')
				# plt.plot(temp[0][-1], rolling_std_derivative_order_param_smoothed[temp[0][-1]], 'cd')
				# plt.plot((poolData[0][i]['dictData']['tstep_annealing_start']-poolData[0][i]['dictNet']['max_delay_steps']) * poolData[0][i]['dictPLL']['dt'], 0, 'yd')

				# print('temp=', temp[0])
				if not len(temp[0]) == 0:
					# subtract from the last time when the transient dynamics caused order parameter fluctuations above the threshold the time when the annealing process started
					# the subtraction of the initial delay history is already done since we only search from tau onwards for the time at which the fluctuations fulfill the conditions
					sol_time.append((temp[0][-1]) * poolData[0][i]['dictPLL']['dt'])
				else:
					sol_time.append(np.inf)
				# print('sol_time=', sol_time)
				# plt.draw()
				# plt.show()
			else:
				sol_time.append(np.inf)
		else:
			if correct_solution_test1:
				derivative_order_param_smoothed = (np.diff(orderparam[poolData[0][i]['dictNet']['max_delay_steps']:] / poolData[0][i]['dictPLL']['dt']))
				rolling_std_derivative_order_param_smoothed = pd.Series(derivative_order_param_smoothed).rolling(int(15 * number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt'])).std()
				temp = np.where(rolling_std_derivative_order_param_smoothed[(poolData[0][i]['dictData']['tstep_annealing_start'] - poolData[0][i]['dictNet']['max_delay_steps']):] > order_param_std_threshold)
				sol_time.append(temp[0][-1] * poolData[0][i]['dictPLL']['dt'])
			else:
				sol_time.append(np.inf)

		if correct_solution_test1:
			ax18[i].plot(poolData[0][i]['dictData']['t'][poolData[0][i]['dictData']['tstep_annealing_start']], 0, 'cd', markersize=1.5)
			ax18[i].plot(poolData[0][i]['dictData']['t'][poolData[0][i]['dictNet']['max_delay_steps']:-1:plotEveryDt], derivative_order_param_smoothed[::plotEveryDt], 'r', linewidth=0.5, alpha=0.35)
			ax18[i].plot(poolData[0][i]['dictData']['t'][poolData[0][i]['dictNet']['max_delay_steps']:-1:plotEveryDt], rolling_std_derivative_order_param_smoothed[::plotEveryDt], 'k', linewidth=0.5, alpha=0.35)

		if correct_solution_test0 and sol_time[i] != np.inf:
			ax18[i].plot(sol_time[i] + poolData[0][i]['dictData']['t'][poolData[0][i]['dictData']['tstep_annealing_start']], rolling_std_derivative_order_param_smoothed[int(sol_time[i] / poolData[0][i]['dictPLL']['dt'])], 'c*', markersize=2.5)

		if len(poolData[0][:]) > threshold_realizations_plot:
			ax21.plot(poolData[0][i]['dictData']['t'][(poolData[0][i]['dictData']['tstep_annealing_start'] + poolData[0][i]['dictNet']['max_delay_steps'])], -0.05, 'cd', markersize=1)
			if correct_solution_test1:
				ax21.plot(poolData[0][i]['dictData']['t'][::plotEveryDt], orderparam[::plotEveryDt], '-', label=r'$R_\textrm{final}=%0.2f$' % (orderparam[-1]), linewidth=linewidth)
				ax211.plot(poolData[0][i]['dictData']['t'][::plotEveryDt], uniform_filter1d(orderparam[::plotEveryDt],
							size=int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']), mode='reflect'), '-', label=r'$R_\textrm{final}=%0.2f$' % (orderparam[-1]), linewidth=linewidth)
				if sol_time[i] != np.inf:
					ax21.plot(sol_time[i] + poolData[0][i]['dictNet']['max_delay_steps'], -0.05, 'c*', markersize=1)
			else:
				ax21.plot(poolData[0][i]['dictData']['t'][::plotEveryDt], orderparam[::plotEveryDt], '--', label=r'$R_\textrm{final}=%0.2f$' % (orderparam[-1]), linewidth=linewidth)
				ax211.plot(poolData[0][i]['dictData']['t'][::plotEveryDt], uniform_filter1d(orderparam[::plotEveryDt],
							size=int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']), mode='reflect'), '--', label=r'$R_\textrm{final}=%0.2f$' % (orderparam[-1]), linewidth=linewidth)


		if phase_wrap == 0:  # plot phase differences in [-inf, inf), i.e., we use the unwrapped phases that have counted the cycles/periods
			ax20[i].hist(poolData[0][i]['dictData']['phi'][-3, :] - poolData[0][i]['dictData']['phi'][-2, 0], bins=number_of_histogram_bins, rwidth=0.9, density=prob_density)
		elif phase_wrap != 0:
			# print('histogram_data (wrapping if phase):', ((dictData['phi'][at_index, plotlist] + shift2piWin) % (2 * np.pi)) - shift2piWin)
			ax20[i].hist((((poolData[0][i]['dictData']['phi'][-3, :] - poolData[0][i]['dictData']['phi'][-2, 0] + shift2piWin) % (2.0 * np.pi)) - shift2piWin), bins=number_of_histogram_bins, rwidth=0.9, density=prob_density)

		for j in range(len(poolData[0][i]['dictData']['phi'][0, :])):
			if shift2piWin != 0:
				deltaTheta[j] = (((poolData[0][i]['dictData']['phi'][:, 0] - poolData[0][i]['dictData']['phi'][:, j]) + shift2piWin) % (2.0 * np.pi)) - shift2piWin 		# calculate phase-differnce w.r.t. osci k=0
			else:
				deltaTheta[j] = poolData[0][i]['dictData']['phi'][:, 0] - poolData[0][i]['dictData']['phi'][:, j]
			signalOut[j] = poolData[0][i]['dictPLL']['vco_out_sig'](poolData[0][i]['dictData']['phi'][:, j])				# generate signals for all phase histories

			# # save in which binarized state the oscillator was at the end of the realization, averaged over
			# if np.mean(deltaTheta[j][-int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']):-1]) - 0 < 0.2:
			# 	group_oscillators_maxcut[i, j] = -1
			# 	final_phase_oscillator.append('zero')
			# elif np.mean(deltaTheta[j][-int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']):-1]) - np.pi < 0.2:
			# 	group_oscillators_maxcut[i, j] = 1
			# 	final_phase_oscillator.append('pi')
			# else:
			# 	group_oscillators_maxcut[i, j] = 0
			# 	final_phase_oscillator.append('diff')

			if j == 0:
				linestyle = '--'
			else:
				linestyle = '-'

			ax16[i].plot( poolData[0][i]['dictData']['t'][::plotEveryDt], deltaTheta[j, ::plotEveryDt], linestyle, linewidth=linewidth, label='sig PLL%i' %(j))
			ax161[i].plot(poolData[0][i]['dictData']['t'], uniform_filter1d(deltaTheta[j, :], size=int(number_of_intrinsic_periods_smoothing * np.mean(poolData[0][i]['dictPLL']['intrF']) / poolData[0][i]['dictPLL']['dt']), mode='reflect'), linestyle, linewidth=linewidth, label='sig PLL%i' % (j))
			ax19[i].plot( poolData[0][i]['dictData']['t'][::plotEveryDt], poolData[0][i]['dictPLL']['vco_out_sig'](poolData[0][i]['dictData']['phi'][::plotEveryDt, j]), linewidth=linewidth, label='sig PLL%i' %(j))
			ax17[i].plot( poolData[0][i]['dictData']['t'][1::plotEveryDt], thetaDot[::plotEveryDt, j], linewidth=linewidth, label='sig PLL%i' %(j))

		print('working on realization %i results from sim:'%i, poolData[0][i]['dictNet'], '\n', poolData[0][i]['dictPLL'], '\n', poolData[0][i]['dictData'], '\n\n')

		if i == int( len(poolData[0][:]) / 2 ):
			ax16[i].set_ylabel(r'$\Delta\theta(t)$', fontsize=axisLabel)
			ax161[i].set_ylabel(r'$\langle\Delta\theta(t)\rangle_{%0.1f T}$' % number_of_intrinsic_periods_smoothing, fontsize=axisLabel)
			ax17[i].set_ylabel(r'$\dot{\theta}(t)$ in radHz', fontsize=axisLabel)
			ax18[i].set_ylabel(r'$R(t)$', fontsize=axisLabel)
			ax19[i].set_ylabel(r'$s(t)$', fontsize=axisLabel)
			ax20[i].set_ylabel(r'$H\left(\Delta\theta(t)\right)$', fontsize=axisLabel)
		if i == len(poolData[0][:])-2:
			ax16[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
			ax161[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
			ax18[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
			ax20[i].set_xlabel(r'$\Delta\theta(t)$ in $[rad]$', fontsize=axisLabel)
		if i == len(poolData[0][:])-1:
			ax17[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
			ax19[i].set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
		if len(poolData[0][:]) > threshold_realizations_plot and i == len(poolData[0][:])-1:
			ax21.set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
			ax21.set_ylabel(r'$R(t)$', fontsize=axisLabel)
			ax21.tick_params(labelsize=tickSize)
			ax211.set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
			ax211.set_ylabel(r'$\langle R(t) \rangle_{%0.1f T}$'%(number_of_intrinsic_periods_smoothing), fontsize=axisLabel)
			ax211.tick_params(labelsize=tickSize)
			ax22.set_xlabel(r'$t$ in $[s]$', fontsize=axisLabel)
			ax22.set_ylabel(r'$p(t)$', fontsize=axisLabel)
			ax22.tick_params(labelsize=tickSize)

		ax16[i].tick_params(labelsize=tickSize)
		ax161[i].tick_params(labelsize=tickSize)
		ax17[i].tick_params(labelsize=tickSize)
		ax18[i].tick_params(labelsize=tickSize)
		ax18[i].legend(loc='lower right', fontsize=legendLab)
		ax19[i].tick_params(labelsize=tickSize)
		ax20[i].tick_params(labelsize=tickSize)

		#if i == 0:
		print('Plotting the network and its binarized asymptotic state.')
		color_map = []
		if isinstance(poolData[0][i]['dictPLL']['gPDin'], list):
			network_graph = nx.from_numpy_array(np.array(poolData[0][i]['dictPLL']['gPDin']))
		else:
			network_graph = nx.from_numpy_array(poolData[0][i]['dictPLL']['gPDin'])
		print('len(final_phase_oscillator)=', len(final_phase_oscillator))
		for node in network_graph:
			print('Working on node %i'%(node))
			if final_phase_oscillator[node] == 'zero':
				color_map.append('green')
			elif final_phase_oscillator[node] == 'pi':
				color_map.append('blue')
			else:
				color_map.append('red')
		plt.figure(9999-i)
		nx.draw(network_graph, node_color=color_map, with_labels=True, pos=nx.spring_layout(network_graph))
		plt.savefig('results/network_asymptotic_state_r%i_%d_%d_%d.svg' % (i, now.year, now.month, now.day), dpi=dpi_val)
		plt.savefig('results/network_asymptotic_state_r%i_%d_%d_%d.png' % (i, now.year, now.month, now.day), dpi=dpi_val)

	sol_time = np.array(sol_time)
	sol_time_without_inf_entries = sol_time[sol_time != np.inf]
	if len(sol_time_without_inf_entries) == 0:
		print('All times to solution were evaluated as np.inf and hence the mean time so solution is np.inf!')
		sol_time_without_inf_entries = np.array([np.inf])
	print('success_count: ', success_count, 'len(poolData[0][:]: ', len(poolData[0][:]), 'sol_time:', sol_time)
	results_string = 'Final evaluation:\n1) for a total of %i realizations, success probability (evaluation R) p1 = %0.4f\n2) and success probability evaluating groups separated by pi is p2 = %0.4f\n3) average time to solution = %0.4f seconds, i.e., %0.2f mean intrinsic periods.\n4) average time to solution without infinity entries= %0.4f seconds, i.e., %0.2f mean intrinsic periods.\n5) fastest and slowest time to solution in multiples of periods: {%0.2f, %0.2f}'%(
		len(poolData[0][:]), success_count / len(poolData[0][:]), success_count_test1 / len(poolData[0][:]),
		np.mean(sol_time),
		np.mean(sol_time)/np.mean(poolData[0][i]['dictPLL']['intrF']),
		np.mean(sol_time_without_inf_entries),
		np.mean(sol_time_without_inf_entries)/np.mean(poolData[0][i]['dictPLL']['intrF']),
		np.min(sol_time)/np.mean(poolData[0][i]['dictPLL']['intrF']),
		np.max(sol_time)/np.mean(poolData[0][i]['dictPLL']['intrF']))
	if len(poolData[0][:]) > threshold_realizations_plot:
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		#ax21.text(0.25*poolData[0][0]['dictData']['t'][-1], 0.2, results_string, horizontalalignment='left', verticalalignment='bottom', bbox=props, fontsize=9)
		ax211.text(0.25 * poolData[0][0]['dictData']['t'][-1], 0.2, results_string, horizontalalignment='left', verticalalignment='bottom', bbox=props, fontsize=9)

	print(results_string)

	if np.all(np.isfinite(sol_time[:])):
		ax22.hist(sol_time, bins=15, rwidth=0.9, density=prob_density)

	ax16[0].legend(loc='upper right', fontsize=legendLab)
	ax17[0].legend(loc='upper right', fontsize=legendLab)
	ax19[0].legend(loc='upper right', fontsize=legendLab)

	fig16.savefig('results/phase_relations_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
	fig161.savefig('results/phase_relations_smoothed_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
	fig17.savefig('results/frequencies_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
	fig18.savefig('results/order_parameter_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
	fig19.savefig('results/signals_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
	fig20.savefig('results/histograms_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
	#fig99.savefig('results/network_asymptotic_state_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
	fig16.savefig('results/phase_relations_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
	fig161.savefig('results/phase_relations_smoothed_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
	fig17.savefig('results/frequencies_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
	fig18.savefig('results/order_parameter_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
	fig19.savefig('results/signals_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
	fig20.savefig('results/histograms_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
	# fig99.savefig('results/network_asymptotic_state_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
	if len(poolData[0][:]) > threshold_realizations_plot:
		fig21.savefig('results/all_order_parameters_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
		fig21.savefig('results/all_order_parameters_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
		fig211.savefig('results/all_order_parameters_smoothed_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
		fig211.savefig('results/all_order_parameters_smoothed_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)
		fig22.savefig('results/hist_soltime_density_%d_%d_%d.svg' % (now.year, now.month, now.day), dpi=dpi_val)
		fig22.savefig('results/hist_soltime_density_%d_%d_%d.png' % (now.year, now.month, now.day), dpi=dpi_val)

	plt.draw()
	plt.show()

	return None


class EvaluateAndPlotIsingMachineSimulation:
	"""Methods to evaluate automatically the time-series of Ising Machine simulations for networks of electronic oscillators. Plot the results.

		Attributes:
			pll_id: the oscillator's identity
			intrinsic_freq: intrinsic frequency in Hz
			intrinsic_freq_rad: intrinsic frequency in radHz

	"""
	def __init__(self, poolData: dict, phase_wrap=0, number_of_histogram_bins=25, prob_density=False, order_param_solution=0.0, number_of_expected_oscis_in_one_group=10) -> None:
		"""
		Args:
			poolData: [dict] contains all the data of the simulations to be evaluated and the settings
			phase_wrap: [integer] whether phases are wrapped into the interval 0) [0, 2*pi), 1) [-pi, pi), or 2) [-pi/2, 3*pi/2)
			number_of_histogram_bins: [integer] the number of bins of the histogram of phases plotted for the final state of the simulation
			prob_density: [boolean] whether histograms are normalized to become probability densities
			order_param_solution: [float] expected order parameter of the asymptotic state if the correct benchmark solution has been found
			number_of_expected_oscis_in_one_group: [integer] number of oscillators in the one of the two groups that form asymptotically
		"""

		if phase_wrap == 1:  	# plot phase-differences in [-pi, pi) interval
			self.shift2piWin = np.pi
		elif phase_wrap == 2:  # plot phase-differences in [-pi/2, 3*pi/2) interval
			self.shift2piWin = 0.5 * np.pi
		elif phase_wrap == 3:  # plot phase-differences in [0, 2*pi) interval
			self.shift2piWin = 0

	def function(self, argument) -> None:
		"""
		Does XY.

		Args:
			argument: explanation

		Returns:
		"""

		return None
