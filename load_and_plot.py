#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.ma as ma
import matplotlib
import codecs
import csv
import os, gc, sys
if not os.environ.get('SGE_ROOT') == None:	# this environment variable is set within the queue network, i.e., if it exists, 'Agg' mode to supress output
	print('NOTE: \"matplotlib.use(\'Agg\')\"-mode active, plots are not shown on screen, just saved to results folder!\n')
	matplotlib.use('Agg') #'%pylab inline'
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
from scipy import optimize
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import time
import pickle

from sim_pll import plot_lib
from sim_pll import sim_lib as sim
from sim_pll import evaluation_lib as eva
from sim_pll import setup
from sim_pll import coupling_fct_lib as coupfct
from sim_pll import check_dicts_lib as chk_dicts
from sim_pll import synctools_interface_lib as synctools_interface
#import palettable.colorbrewer.diverging as colormap_diver
#from palettable.colorbrewer.diverging import PuOr_7

import datetime
now = datetime.datetime.now()

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
axisLabel 			= 60
tickSize  			= 25
titleLabel			= 10
dpi_val	  			= 150
figwidth  			= 10#8
figheight 			= 5
plot_size_inches_x 	= 10
plot_size_inches_y 	= 5
labelpadxaxis       = 10
labelpadyaxis       = 20
#colormapSyncStab 	= colormap_diver.PuOr_7.mpl_colormap


################################################################################
# load data
folder		 = '/home/cuichi/Documents/MPI_PKS_Docs/2019_VIP+/Programs/2021_simPLL_pub/results/'#'/home/cuichi/data-z2/simPLL_2/1_CH_success/results/'
################################################################################
filenamePLL  = folder+'dict_pll_K0.013_tau1.730_Fc0.008_mx0_my-999_N4_topoentrainOne-ring_0:43_2022_10_13' #_K0.013_tau1.730_Fc0.008_mx0_my-999_N4_topoentrainOne-ring_12:7_2022_10_11'
filenameNet  = folder+'dict_net_K0.013_tau1.730_Fc0.008_mx0_my-999_N4_topoentrainOne-ring_0:43_2022_10_13' #_K0.013_tau1.730_Fc0.008_mx0_my-999_N4_topoentrainOne-ring_12:7_2022_10_11'
filenameData = folder+'pool_data_K0.013_tau1.730_Fc0.008_mx0_my-999_N4_topoentrainOne-ring_0:43_2022_10_13' #_K0.013_tau1.730_Fc0.008_mx0_my-999_N4_topoentrainOne-ring_12:7_2022_10_11'
filenameAlgo = folder+'dict_algo_K0.013_tau1.730_Fc0.008_mx0_my-999_N4_topoentrainOne-ring_0:43_2022_10_13' #_K0.013_tau1.730_Fc0.008_mx0_my-999_N4_topoentrainOne-ring_12:7_2022_10_11'
################################################################################
if 'pool_data' in filenameData:
	pool_data = pickle.load(open(filenameData, 'rb'))
else:
	dict_data = pickle.load(open(filenameData, 'rb'))
	dict_pll  = pickle.load(open(filenamePLL, 'rb'))
	dict_net  = pickle.load(open(filenameNet, 'rb'))
	dict_algo = pickle.load(open(filenameAlgo, 'rb'))
################################################################################
# if necessary update parameters related to plotting
################################################################################
if 'pool_data' in filenameData:
	for i in range(len(pool_data[0][:])):
		pool_data[0][i]['dict_pll'].update({'PSD_freq_resolution': 1E-5})
		pool_data[0][i]['dict_pll'].update({'sampleFplot': 1000})
		# pool_data[0][i]['dict_pll'].update({'intrF': 1})
		# print('updated intrinsic freq:', pool_data[0][i]['dict_pll']['intrF'])
		# pool_data[0][i]['dict_pll'].update({'orderLF': 1})
		# dict_pll, dict_net, dict_algo = chk_dicts.check_dicts_consistency(dict_pll, dict_net, dict_algo)
else:
	dict_pll.update({'PSD_freq_resolution': 1E-5})
	dict_pll.update({'sampleFplot': 1000})
	dict_pll.update({'intrF': 1})
	dict_pll.update({'orderLF': 1})
	print('updated intrinsic freq:', dict_pll['intrF'])
	# if not dict_algo:
	# 	dict_algo={
	# 		'parameter_space_sweeps': 'listOfInitialPhaseConfigurations',		# pick method for setting realizations 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations'
	# 		'paramDiscretization': [5, 3],#3										# parameter discetization for brute force parameter space scans
	# 		'min_max_range_parameter_0': [0.95, 1.05]									# specifies within which min and max value to linspace the detuning
	# 	}
	# else:
	# 	print('dict_algo has been loaded!')
	dict_pll, dict_net, dict_algo = chk_dicts.check_dicts_consistency(dict_pll, dict_net, dict_algo)
################################################################################
################################################################################

labeldict 	= {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\omega\tau/2\pi$', 'K': r'$\frac{K}{\omega}$', 'fric': r'$\gamma$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$', 'alphaK': r'$2\alpha(K)$', 'alphaTau': r'$2\alpha(\tau)$'}

''' prepare colormap for scatter plot that is always in [0, 1] or [min(results), max(results)] '''
cdict = {
  'red'  :  ( (0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
  'green':  ( (0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
  'blue' :  ( (0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
}
colormap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

if 'pool_data' in filenameData:
	if pool_data[0][0]['dict_algo']['parameter_space_sweeps']   == 'testNetworkMotifIsing':
		eva.evaluateSimulationIsing(pool_data)
	elif pool_data[0][0]['dict_algo']['parameter_space_sweeps'] == 'listOfInitialPhaseConfigurations':
		eva.evaluateSimulationsChrisHoyer(pool_data)
	elif pool_data[0][0]['dict_algo']['parameter_space_sweeps'] == 'classicBruteForceMethodRotatedSpace':
		print('Implement evaluation as in the old version! Copy plots, etc...')
		sys.exit()
	elif pool_data[0][0]['dict_algo']['parameter_space_sweeps'] == 'one_parameter_sweep' and 'entrain' in pool_data[0][0]['dict_net']['topology']:
		eva.evaluate_entrainment_of_mutual_sync(pool_data, average_time_for_time_series_in_periods=3.5)
else:
	if dict_net['special_case'] == 'timeDepChangeOfIntrFreq':
		phase_diff_wrap_to_interval = 3
		if phase_diff_wrap_to_interval == 1:  # plot phase-differences in [-pi, pi] interval
			shift2piWin = np.pi
		elif phase_diff_wrap_to_interval == 2:  # plot phase-differences in [-pi/2, 3*pi/2] interval
			shift2piWin = 0.5 * np.pi
		elif phase_diff_wrap_to_interval == 3:  # plot phase-differences in [0, 2*pi] interval
			shift2piWin = 0.0
		sampling = 100
		inst_freq = np.diff(dict_data['phi'] / dict_pll['div'], axis=0) / (2 * np.pi * dict_pll['dt'])
		phase_diffs = np.zeros([len(dict_data['phi'][0:-1:sampling, 0]), 3])
		phase_diffs[:, 0] = ((dict_data['phi'][0:-1:sampling, 1] / dict_pll['div'] - dict_data['phi'][0:-1:sampling, 2] / dict_pll['div'] + shift2piWin) % (2 * np.pi)) - shift2piWin
		phase_diffs[:, 1] = ((dict_data['phi'][0:-1:sampling, 1] / dict_pll['div'] - dict_data['phi'][0:-1:sampling, 3] / dict_pll['div'] + shift2piWin) % (2 * np.pi)) - shift2piWin
		phase_diffs[:, 2] = ((dict_data['phi'][0:-1:sampling, 2] / dict_pll['div'] - dict_data['phi'][0:-1:sampling, 3] / dict_pll['div'] + shift2piWin) % (2 * np.pi)) - shift2piWin
		dict_for_chris_adiabatic = {'time_dependent_parameter': dict_data['timeDependentParameter'][0, 0:-1:sampling],
									'instantaneous_frequencies_Hz': inst_freq[0:-1:sampling, :], 'phase_differences_rad': phase_diffs, 'control_signal': dict_data['ctrl'][:, 0:-1:sampling]}
		np.save('results/dict_for_chris_adiabatic_tau-%0.3f_topology-%s.npy' % (dict_pll['transmission_delay'], dict_net['topology']), dict_for_chris_adiabatic)
		sys.exit()
	# run evaluations
	order_parameter, order_parameter_divided_phases, F1 = eva.compute_order_parameter(dict_pll, dict_net, dict_data)
	dict_data.update({'order_parameter': order_parameter, 'order_parameter_divided_phases': order_parameter_divided_phases, 'F1': F1})

	#dict_pll.update({'vco_out_sig': coupfct.sine})

	#setup.setup_time_dependent_parameter(dict_net, dict_pll, dict_data)
	sim.plot_results_simulation(dict_net, dict_pll, dict_algo, dict_data)

	# dict_pllsyncTool = dict_pll.copy()
	# dict_pllsyncTool.update({'transmission_delay': 3})
	# dict_pllsyncTool.update({'coupK': dict_pll['coupK']/2})
	# tau1, omega1, tau2, omega2 = synctools_interface.generate_delay_plot(dict_pllsyncTool, dict_net)
	#
	# param_name = dict_net['special_case']										#'timeDepInjectLockCoupStr', 'timeDepTransmissionDelay', 'timeDepChangeOfCoupStr', 'distanceDepTransmissionDelay'
	# if param_name == 'timeDepTransmissionDelay':
	# 	dyn_x_label = r'$\frac{\tau\omega}{2\pi}$'
	# 	x_axis_scaling = np.mean(dict_pll['intrF'])
	# elif param_name == 'timeDepChangeOfCoupStr':
	# 	dyn_x_label = r'$\frac{2\pi K}{\omega}$'
	# 	x_axis_scaling = np.mean(1.0/dict_pll['intrF'])
	# y_axis_scaling = (2.0*np.pi*np.mean(dict_pll['intrF']))
	#
	# fig12 = plt.figure(num=12, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	# fig12.canvas.set_window_title('instantaneous frequency as function of time-dependent parameter')
	# fig12.set_size_inches(plot_size_inches_x, plot_size_inches_y)
	#
	# plt.plot(dict_data['timeDependentParameter'][0,0:len(dict_data['phi'][:,0])-1]*x_axis_scaling, (np.diff(dict_data['phi'], axis=0)/dict_pll['dt'])/y_axis_scaling, 'b-')
	# plt.plot(tau1, omega1, 'c+', tau2, omega2, 'g*')
	#
	# plt.xlabel(dyn_x_label, fontdict = labelfont, labelpad=labelpadxaxis)
	# plt.ylabel(r'$\frac{\dot{\theta}_k(t)}{\omega}$', fontdict = labelfont, labelpad=labelpadyaxis)
	# plt.tick_params(axis='both', which='major', labelsize=tickSize)

	# plot_lib.plotOrderPara(dict_pll, dict_net, dict_data)
	# #plot_lib.plotPhaseRela(dict_pll, dict_net, dict_data)
	# #plot_lib.plotPhaseDiff(dict_pll, dict_net, dict_data)
	# #plot_lib.plotClockTime(dict_pll, dict_net, dict_data)
	# #plot_lib.plotOscSignal(dict_pll, dict_net, dict_data)
	# #plot_lib.plotFrequency(dict_pll, dict_net, dict_data)
	# plot_lib.plotFreqAndPhaseDiff(dict_pll, dict_net, dict_data)
	# #plot_lib.plotFreqAndOrderPar(dict_pll, dict_net, dict_data)
	# plot_lib.plotPSD(dict_pll, dict_net, dict_data, [], saveData=False)

plt.draw()
plt.show()
