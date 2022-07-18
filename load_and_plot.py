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
#colormapSyncStab 	= colormap_diver.PuOr_7.mpl_colormap


################################################################################
# load data
folder		 = '/home/cuichi/Documents/MPI_PKS_Docs/2019_VIP+/Programs/2021_simPLL_pub/results/'#'/home/cuichi/data-z2/simPLL_2/1_CH_success/results/'
################################################################################
filenamePLL  = folder+'dictPLL_K0.050_tau2.969_Fc0.000_mx0_my-999_N2_toporing_22:13_2021_11_23'
filenameNet  = folder+'dictNet_K0.050_tau2.969_Fc0.000_mx0_my-999_N2_toporing_22:13_2021_11_23'
filenameData = folder+'dictData_K0.050_tau2.969_Fc0.000_mx0_my-999_N2_toporing_22:13_2021_11_23'
filenameAlgo = folder+'dictAlgo_K0.050_tau2.969_Fc0.000_mx0_my-999_N2_toporing_22:13_2021_11_23'
################################################################################
if 'poolData' in filenameData:
	poolData = pickle.load(open(filenameData, 'rb'))
else:
	dictData = pickle.load(open(filenameData, 'rb'))
	dictPLL  = pickle.load(open(filenamePLL, 'rb'))
	dictNet  = pickle.load(open(filenameNet, 'rb'))
	dictAlgo = pickle.load(open(filenameAlgo, 'rb'))
################################################################################
# if necessary update parameters related to plotting
################################################################################
dictPLL.update({'PSD_freq_resolution': 1E-5})
dictPLL.update({'sampleFplot': 1000})
dictPLL.update({'intrF': 1})
dictPLL.update({'orderLF': 1})
print('updated intrinsic freq:', dictPLL['intrF'])
# if not dictAlgo:
# 	dictAlgo={
# 		'bruteForceBasinStabMethod': 'listOfInitialPhaseConfigurations',		# pick method for setting realizations 'classicBruteForceMethodRotatedSpace', 'listOfInitialPhaseConfigurations'
# 		'paramDiscretization': [5, 3],#3										# parameter discetization for brute force parameter space scans
# 		'min_max_range_parameter': [0.95, 1.05]									# specifies within which min and max value to linspace the detuning
# 	}
# else:
# 	print('dictAlgo has been loaded!')
dictPLL, dictNet, dictAlgo = chk_dicts.check_dicts_consistency(dictPLL, dictNet, dictAlgo)
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
colormap  	= matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

if 'poolData' in filenameData:
	if poolData[0][0]['dictAlgo']['bruteForceBasinStabMethod']   == 'testNetworkMotifIsing':
		eva.evaluateSimulationIsing(poolData)
	elif poolData[0][0]['dictAlgo']['bruteForceBasinStabMethod'] == 'listOfInitialPhaseConfigurations':
		eva.evaluateSimulationsChrisHoyer(poolData)
	elif poolData[0][0]['dictAlgo']['bruteForceBasinStabMethod'] == 'classicBruteForceMethodRotatedSpace':
		print('Implement evaluation as in the old version! Copy plots, etc...'); sys.exit()
else:
	# run evaluations
	r, orderParam, F1 	= eva.obtainOrderParam(dictPLL, dictNet, dictData)
	dictData.update({'orderParam': orderParam, 'R': r, 'F1': F1})

	#dictPLL.update({'vco_out_sig': coupfct.sine})

	sim.plot_results_simulation(dictNet, dictPLL, dictData)

	# dictPLLsyncTool = dictPLL.copy()
	# dictPLLsyncTool.update({'transmission_delay': 3})
	# dictPLLsyncTool.update({'coupK': dictPLL['coupK']/2})
	# tau1, omega1, tau2, omega2 = synctools_interface.generate_delay_plot(dictPLLsyncTool, dictNet)
	#
	# param_name = dictNet['special_case']										#'timeDepInjectLockCoupStr', 'timeDepTransmissionDelay', 'timeDepChangeOfCoupStr', 'distanceDepTransmissionDelay'
	# if param_name == 'timeDepTransmissionDelay':
	# 	dyn_x_label = r'$\frac{\tau\omega}{2\pi}$'
	# 	x_axis_scaling = np.mean(dictPLL['intrF'])
	# elif param_name == 'timeDepChangeOfCoupStr':
	# 	dyn_x_label = r'$\frac{2\pi K}{\omega}$'
	# 	x_axis_scaling = np.mean(1.0/dictPLL['intrF'])
	# y_axis_scaling = (2.0*np.pi*np.mean(dictPLL['intrF']))
	#
	# fig12 = plt.figure(num=12, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	# fig12.canvas.set_window_title('instantaneous frequency as function of time-dependent parameter')
	# fig12.set_size_inches(plot_size_inches_x, plot_size_inches_y)
	#
	# plt.plot(dictData['timeDependentParameter'][0,0:len(dictData['phi'][:,0])-1]*x_axis_scaling, (np.diff(dictData['phi'], axis=0)/dictPLL['dt'])/y_axis_scaling, 'b-')
	# plt.plot(tau1, omega1, 'c+', tau2, omega2, 'g*')
	#
	# plt.xlabel(dyn_x_label, fontdict = labelfont, labelpad=labelpadxaxis)
	# plt.ylabel(r'$\frac{\dot{\theta}_k(t)}{\omega}$', fontdict = labelfont, labelpad=labelpadyaxis)
	# plt.tick_params(axis='both', which='major', labelsize=tickSize)

	# plot_lib.plotOrderPara(dictPLL, dictNet, dictData)
	# #plot_lib.plotPhaseRela(dictPLL, dictNet, dictData)
	# #plot_lib.plotPhaseDiff(dictPLL, dictNet, dictData)
	# #plot_lib.plotClockTime(dictPLL, dictNet, dictData)
	# #plot_lib.plotOscSignal(dictPLL, dictNet, dictData)
	# #plot_lib.plotFrequency(dictPLL, dictNet, dictData)
	# plot_lib.plotFreqAndPhaseDiff(dictPLL, dictNet, dictData)
	# #plot_lib.plotFreqAndOrderPar(dictPLL, dictNet, dictData)
	# plot_lib.plotPSD(dictPLL, dictNet, dictData, [], saveData=False)

plt.draw()
plt.show()
