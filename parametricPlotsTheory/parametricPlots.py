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
#from scipy.interpolate import spline
from scipy.special import lambertw
from scipy.signal import square
import itertools
import math
import time
import pickle

import synctools_interface_lib as synctools
from function_lib import solveLinStab
import function_lib as fct_lib

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
axisLabel = 44;
tickSize  = 20;
titleLabel= 10;
dpi_val   = 150;
figwidth  = 10;
figheight = 5;

def preparePlotting(params):

	# calculate gamma
	params = evaluateEq(params)

	# prepare scatter plot matrices or 2D plot
	if not params['discrP'] == None:
		params = prepare2D(params)
	else:
		params = prepareScatt(params)

	#print('params[*y*]',  params['y'])
	#print('params[*x1*]', params['x1'])
	#print('params[*x2*]', params['x2'])
	#print('shape(params[*x1*]), params[*x2*]', np.shape(params['x1']), np.shape(params['x2']))

	return params

# ******************************************************************************

def makePlotsFromSynctoolsResults(figID, x, y, z, rescale_x, rescale_y, rescale_z, x_label, y_label, z_label,
										x_identifier, y_identifier, z_identifier, mask_treshold=None, colormap=cm.coolwarm):

	fig = plt.figure(num=figID, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig.set_size_inches(8,6)

	plt.clf()
	ax = plt.subplot(1, 1, 1)
	#ax.set_aspect('equal')

	plt.rc('xtick', labelsize=tickSize)    										# fontsize of the tick labels
	plt.rc('ytick', labelsize=tickSize)	    									# fontsize of the tick labels

	if z_identifier == 'ReLambda':												# fix the colormap to clearly distinguish the zero
		colormap = shiftedColorMap(colormap, np.min(np.array(z)[np.isnan(np.array(z))==False]), np.max(np.array(z)[np.isnan(np.array(z))==False]), name='test')

	#tempresults = np.array(z).reshape((len(x), len(y)))
	tempresults = z
	tempresults = np.transpose(tempresults)
	if mask_treshold:
		tempresults_ma = ma.masked_where(tempresults < mask_treshold, tempresults)	# create masked array
	else:
		tempresults_ma = tempresults

	#print('tempresults_ma:', tempresults_ma, '\ttype(tempresults_ma)', type(tempresults_ma), '\tmin(tempresults_ma):', np.min(tempresults_ma), '\tmax(tempresults_ma):', np.max(tempresults_ma))

	plt.imshow(tempresults_ma.astype(float)*rescale_z, interpolation='nearest', cmap=colormap, aspect='auto', origin='lower',
			extent=(x.min()*rescale_x, x.max()*rescale_x, y.min()*rescale_y, y.max()*rescale_y),
			vmin=np.min(tempresults_ma[np.isnan(tempresults_ma)==False]),
			vmax=np.max(tempresults_ma[np.isnan(tempresults_ma)==False]) )

	plt.title(r'%s in the %s vs %s parameter plane'%(z_label, x_label, y_label))
	plt.xlabel(x_label, fontsize=axisLabel)
	plt.ylabel(y_label, rotation=0, fontsize=axisLabel)
	plt.xlim([np.min(x)*rescale_x, np.max(x)*rescale_x])
	plt.ylim([np.min(y)*rescale_y, np.max(y)*rescale_y])
	plt.colorbar(); plt.tight_layout();
	plt.savefig('plots/imshow_%s_in_%s_vs_%s_%d_%d_%d.svg' %(z_identifier, x_identifier, y_identifier, now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('plots/imshow_%s_in_%s_vs_%s_%d_%d_%d.png' %(z_identifier, x_identifier, y_identifier, now.year, now.month, now.day), dpi=dpi_val)

	return None

# ******************************************************************************

def shiftedColorMap(cmap, min_val, max_val, name):
	'''Function to offset the "center" of a colormap. Useful for data with a negative min and positive max and you want the middle of the colormap's dynamic range to be at zero. Adapted from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

	Input
	-----
	  cmap : The matplotlib colormap to be altered.
	  start : Offset from lowest point in the colormap's range.
		  Defaults to 0.0 (no lower ofset). Should be between
		  0.0 and `midpoint`.
	  midpoint : The new center of the colormap. Defaults to
		  0.5 (no shift). Should be between 0.0 and 1.0. In
		  general, this should be  1 - vmax/(vmax + abs(vmin))
		  For example if your data range from -15.0 to +5.0 and
		  you want the center of the colormap at 0.0, `midpoint`
		  should be set to  1 - 5/(5 + 15)) or 0.75
	  stop : Offset from highets point in the colormap's range.
		  Defaults to 1.0 (no upper ofset). Should be between
		  `midpoint` and 1.0.'''
	epsilon = 0.001
	start, stop = 0.0, 1.0
	min_val, max_val = min(0.0, min_val), max(0.0, max_val) 					# Edit #2
	midpoint = 1.0 - max_val/(max_val + abs(min_val))
	cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
	# regular index to compute the colors
	reg_index = np.linspace(start, stop, 257)
	# shifted index to match the data
	shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)])
	for ri, si in zip(reg_index, shift_index):
		if abs(si - midpoint) < epsilon:
			r, g, b, a = cmap(0.5) # 0.5 = original midpoint.
		else:
			r, g, b, a = cmap(ri)
		cdict['red'].append((si, r, r))
		cdict['green'].append((si, g, g))
		cdict['blue'].append((si, b, b))
		cdict['alpha'].append((si, a, a))
	newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
	plt.register_cmap(cmap=newcmap)
	return newcmap

# ******************************************************************************

def plotParametric(params):

	params = preparePlotting(params)

	labeldict 	= {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\omega\tau}{2\pi}$', 'K': r'$K$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$', 'alphaK': r'$2\alpha(K)$', 'alphaTau': r'$2\alpha(\tau)$'}

	''' prepare colormap for scatter plot that is always in [0, 1] or [min(results), max(results)] '''
	cdict = {
	  'red'  :  ( (0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
	  'green':  ( (0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
	  'blue' :  ( (0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
	}
	colormap  = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

	fig1 = plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig1.set_size_inches(8,6)

	plt.clf()
	ax = plt.subplot(1, 1, 1)
	#ax.set_aspect('equal')

	tempresults = params['yy'].reshape((len(params['x1']), len(params['x2'])))   #np.flipud()
	tempcond = params['CondStab'].reshape((len(params['x1']), len(params['x2'])))
	#tempcond2 = params['noInstCond2'].reshape((len(params['x1']), len(params['x2'])))
	tempresults = np.transpose(tempresults)
	tempcond = np.transpose(tempcond)
	#tempcond2 = np.transpose(tempcond2)
	#print('tempresults:', tempresults)
	#tempresults_ma = tempresults
	tempresults_ma = ma.masked_where(tempresults < 0, tempresults)				# Create masked array
	tempresults_ma1 = ma.masked_where(tempresults >= 0, tempresults)			# Create masked array
	#print('tempresult_ma:', tempresults_ma)
	#print('initPhiPrime0:', initPhiPrime0)
	cmap_choice 	= cm.PuOr													# cm.coolwarm
	cmap_neg_alpha 	= colors.ListedColormap(['black'])
	cmap_cond 		= colors.ListedColormap(['yellow'])

	try:
		plt.imshow(tempresults_ma.astype(float), interpolation='nearest', cmap=cmap_choice, aspect='auto', origin='lower',
				extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
				vmin=np.min(tempresults_ma[np.isnan(tempresults_ma)==False]), vmax=np.max(tempresults_ma[np.isnan(tempresults_ma)==False]) )
				#vmin=np.min(tempresults_ma), vmax=np.max(tempresults_ma) )
		plt.colorbar();
		plt.imshow(tempresults_ma1.astype(float), interpolation='nearest', cmap=cmap_neg_alpha, aspect='auto', origin='lower',
				extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
				vmin=np.min(tempresults_ma1), vmax=np.max(tempresults_ma1) )
		plt.imshow(tempcond.astype(float), interpolation='nearest', cmap=cmap_cond, aspect='auto', origin='lower',
				extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
				vmin=np.min(tempcond), vmax=np.max(tempcond) )
	except:
		print('Exception mode when plotting imshow!');
		plt.imshow(tempresults_ma.astype(float), interpolation='nearest', cmap=cmap_choice, aspect='auto', origin='lower',
				extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
				vmin=np.min(tempresults_ma), vmax=np.max(tempresults_ma) )
				#vmin=np.min(tempresults_ma), vmax=np.max(tempresults_ma) )
		plt.imshow(tempresults_ma1.astype(float), interpolation='nearest', cmap=cmap_neg_alpha, aspect='auto', origin='lower',
				extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
				vmin=np.min(tempresults_ma1), vmax=np.max(tempresults_ma1) )
		plt.imshow(tempcond.astype(float), interpolation='nearest', cmap=cmap_cond, aspect='auto', origin='lower',
				extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
				vmin=np.min(tempcond), vmax=np.max(tempcond) )

	if ( params['loopP1'] == 'alphaK' and ( isinstance(params['K'], np.ndarray) or isinstance(params['K'], list)) ):
		print('Plot condition region into the plot!')
		plt.plot(2*params['xscatt'], 2*params['xscatt'], 'k-', linewidth=1);
		plt.plot(2*params['xscatt'], (1+np.sqrt(1-params['zeta']**2))*2*params['xscatt'], 'r--', linewidth=1); #*2*params['alpha'][:,0]

	plt.xlabel(labeldict[params['loopP1']], fontsize=axisLabel)
	plt.ylabel(labeldict[params['loopP2']], fontsize=axisLabel, rotation=0)
	plt.xlim([np.min(params['xscatt']), np.max(params['xscatt'])])
	plt.ylim([np.min(params['yscatt']), np.max(params['yscatt'])])
	plt.tight_layout();
	plt.savefig('plots/imshow_%s_vs_%s_%d_%d_%d.svg' %(params['loopP1'], params['loopP2'], now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('plots/imshow_%s_vs_%s_%d_%d_%d.png' %(params['loopP1'], params['loopP2'], now.year, now.month, now.day), dpi=dpi_val)

################################################################################

	fig2 = plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig2.set_size_inches(8,6)

	plt.clf()
	ax = plt.subplot(1, 1, 1)
	#ax.set_aspect('equal')

	cmap_choice = cm.RdYlBu														# cm.coolwarm
	ytemp = params['yy']
	for i in range(len(ytemp)):
		if (ytemp[i] > 0 and np.isnan(ytemp[i])==False):
			ytemp[i] = 1

	#ytemp[np.isnan(params['y'])==False] = 0
	tempresults = ytemp.reshape((len(params['x1']), len(params['x2'])))   #np.flipud()
	tempresults = np.transpose(tempresults)
	tempresults_ma = tempresults

	plt.imshow(tempresults_ma.astype(float), interpolation='nearest', cmap=cmap_choice, aspect='auto', origin='lower',
			extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
			vmin=np.min(tempresults_ma[np.isnan(tempresults_ma)==False]), vmax=np.max(tempresults_ma[np.isnan(tempresults_ma)==False]) )
			#vmin=np.min(tempresults_ma), vmax=np.max(tempresults_ma) )
	plt.imshow(tempcond.astype(float), interpolation='nearest', cmap=cmap_cond, aspect='auto', origin='lower',
			extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
			vmin=np.min(tempcond), vmax=np.max(tempcond) )

	if ( params['loopP1'] == 'alphaK' and ( isinstance(params['K'], np.ndarray) or isinstance(params['K'], list)) ):
		print('Plot condition region into the plot!')
		plt.plot(2*params['xscatt'], 2*params['xscatt'], 'k-', linewidth=1);
		plt.plot(2*params['xscatt'], (1+np.sqrt(1-params['zeta']**2))*2*params['xscatt'], 'r--', linewidth=1); #*2*params['alpha'][:,0]

	plt.xlabel(labeldict[params['loopP1']], fontsize=axisLabel)
	plt.ylabel(labeldict[params['loopP2']], fontsize=axisLabel, rotation=0)
	plt.xlim([np.min(params['xscatt']), np.max(params['xscatt'])])
	plt.ylim([np.min(params['yscatt']), np.max(params['yscatt'])])
	plt.tight_layout(); # plt.colorbar();
	plt.savefig('plots/imshow1color_%s_vs_%s_%d_%d_%d.svg' %(params['loopP1'], params['loopP2'], now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('plots/imshow1color_%s_vs_%s_%d_%d_%d.png' %(params['loopP1'], params['loopP2'], now.year, now.month, now.day), dpi=dpi_val)

################################################################################

	# fig3 = plt.figure(num=3, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	# fig3.set_size_inches(8,6)
	#
	# plt.clf()
	# ax = plt.subplot(1, 1, 1)
	# #ax.set_aspect('equal')
	#
	# plt.scatter(params['xscatt'], params['yscatt'], s=1.5, c=params['y'], alpha=1, edgecolor='', cmap=colormap,
	# 				vmin=np.min(params['y'][np.isnan(params['y'])==False]), vmax=np.max(params['y'][np.isnan(params['y'])==False]))
	#
	# plt.xlabel(labeldict[params['loopP1']], fontsize=axisLabel)
	# plt.ylabel(labeldict[params['loopP2']], fontsize=axisLabel)
	# plt.xlim([np.min(params['xscatt']), np.max(params['xscatt'])])
	# plt.ylim([np.min(params['yscatt']), np.max(params['yscatt'])])
	# plt.colorbar()
	# plt.savefig('plots/scatter_%s_vs_%s_%d_%d_%d.svg' %(params['loopP1'], params['loopP2'], now.year, now.month, now.day), dpi=dpi_val)
	# plt.savefig('plots/scatter_%s_vs_%s_%d_%d_%d.png' %(params['loopP1'], params['loopP2'], now.year, now.month, now.day), dpi=dpi_val)

	plt.draw()
	plt.show()

	return None

# ******************************************************************************

def plot2D(params):

	params		= preparePlotting(params)

	labeldict 	= {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\frac{\Omega\tau}{2\pi}$', 'K': r'$K$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	labeldict1 	= {'wc': r'$\omega_\textrm{c}$', 'tau': r'$\tau$', 'K': r'$K$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$'}
	color		= ['blue', 'red', 'purple', 'cyan', 'green', 'yellow'] #'magenta'
	linet		= ['-', '-.', '--', ':', 'densily dashdotdotted', 'densely dashed']

	fig1 = plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig1.canvas.set_window_title('gamma vs %s' %params['loopP2'])				# plot the gamma versus XY
	plt.clf()

	#print('shape(params[*x1*])', np.shape(params['x1']), '\t shape(params[*y*])', np.shape(params['y']))

	for i in range(len(params[params['discrP']])):
		lab1 = r''+labeldict1[params['discrP']]+'=%0.4f' % params[params['discrP']][i]
		lab2 = r''+labeldict1[params['discrP']]+r'=%0.4f ($\gamma$)' % params[params['discrP']][i]
		#print('lab', lab, '\t type(lab)', type(lab)); #sys.exit()
		#print('shape(params[*x1*][0,:]), shape(params[*y*][i,:])', np.shape(params['x1'][0,:]), np.shape(params['y'][i,:]))
		plt.plot(params['x1'][0,:], params['y'][i,:], 'r*', markersize=8)#color=color[i], linewidth=1, linestyle=linet[i], label=lab1)
		#plt.plot(params['x1'], params['charEq_Im'][i,:], color=color[i], linewidth=3, linestyle='dotted', alpha=0.2, label=lab2)

	plt.xlabel(labeldict[params['loopP2']], fontdict = labelfont)
	plt.ylabel(r'$\gamma$', fontdict = labelfont)
	plt.legend()

	plt.savefig('plots/2D_%s_vs_%s_and%s_%d_%d_%d.svg' %(params['loopP2'], 'gamma', params['loopP1'], now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('plots/2D_%s_vs_%s_and%s_%d_%d_%d.png' %(params['loopP2'], 'gamma', params['loopP1'], now.year, now.month, now.day), dpi=dpi_val)

	#############################################################################################################################################

	fig2 = plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
	fig2.canvas.set_window_title('sigma vs %s' %params['loopP2'])				# plot the gamma versus XY
	plt.clf()

	#print('shape(params[*x1*])', np.shape(params['x1']), '\t shape(params[*y*])', np.shape(params['y']))

	for i in range(len(params[params['discrP']])):
		lab1 = r''+labeldict1[params['discrP']]+'=%0.4f' % params[params['discrP']][i]
		lab2 = r''+labeldict1[params['discrP']]+r'=%0.4f ($\sigma$)' % params[params['discrP']][i]
		#print('lab', lab, '\t type(lab)', type(lab)); #sys.exit()
		plt.plot(params['x1'][0,:], params['y1'][i,:], color=color[i], linewidth=1, linestyle=linet[i], label=lab1)
		plt.plot(params['x1'][0,:], params['charEq_Re'][i,:], color=color[i], linewidth=3, linestyle='dotted', alpha=0.2, label=lab2)

	plt.axhline(y=0)
	plt.xlabel(labeldict[params['loopP2']], fontdict = labelfont)
	plt.ylabel(r'$\sigma$', fontdict = labelfont)
	plt.legend()

	plt.savefig('plots/2D_%s_vs_%s_and%s_%d_%d_%d.svg' %(params['loopP2'], 'gamma', params['loopP1'], now.year, now.month, now.day), dpi=dpi_val)
	plt.savefig('plots/2D_%s_vs_%s_and%s_%d_%d_%d.png' %(params['loopP2'], 'gamma', params['loopP1'], now.year, now.month, now.day), dpi=dpi_val)

	plt.draw()
	plt.show()

	return None

# ******************************************************************************

def prepareScatt(params):

	if  ( params['loopP1'] == 'tau' and params['loopP2'] == 'wc' ):

		print('Prepare %s vs %s plot!'%(params['loopP1'], params['loopP2']))
		scatt = np.array(list(itertools.product(*np.array([params['x1'], params['x2']]))))
		scale = np.array(list(itertools.product(*np.array([params['Omeg'][:,0], params['x2']])))) #rescale only in x-direction, Omega independent of y-direction
		params.update({'xscatt': scatt[:,0]*params['w']/(2.0*np.pi)})#scale[:,0]
		params.update({'yscatt': scatt[:,1]/params['w']})#

	elif( params['loopP1'] == 'wc' and params['loopP2'] == 'tau' ):

		print('Prepare %s vs %s plot!'%(params['loopP1'], params['loopP2']))
		scatt = np.array(list(itertools.product(*np.array([params['x1'], params['x2']]))))
		scale = np.array(list(itertools.product(*np.array([params['Omeg'][0,:], params['x1']]))))
		params.update({'xscatt': scatt[:,0]/params['w']})
		params.update({'yscatt': scatt[:,1]*params['w']/(2.0*np.pi)}) #scale[:,0]

	elif( params['loopP1'] == 'K' and params['loopP2'] == 'wc' ):

		print('Prepare %s vs %s plot!'%(params['loopP1'], params['loopP2']))
		scatt = np.array(list(itertools.product(*np.array([params['x1'], params['x2']]))))
		params.update({'xscatt': scatt[:,0]/params['w']})
		params.update({'yscatt': scatt[:,1]/params['w']})

	elif( params['loopP1'] == 'tau' and params['loopP2'] == 'K' ):

		print('Prepare %s vs %s plot!'%(params['loopP1'], params['loopP2']))
		scatt  = np.array(list(itertools.product(*np.array([params['x1'], params['x2']]))))
		scalex = np.array(list(itertools.product(*np.array([params['Omeg'][:,0], params['x2']]))))
		scaley = np.array(list(itertools.product(*np.array([params['Omeg'][0,:], params['x1']]))))
		#print('shape(scatt)', np.shape(scatt)); sys.exit()
		params.update({'xscatt': scatt[:,0]/(2.0*np.pi/params['w'])})#(2.0*np.pi/scalex[:,0])})
		params.update({'yscatt': scatt[:,1]/params['w']})#(2.0*np.pi/scaley[:,0])})#			params['w']})

	elif( params['loopP1'] == 'alpha' and params['loopP2'] == 'wc' ):

		print('Prepare %s vs %s plot!'%(params['loopP1'], params['loopP2']))
		if ( isinstance(params['K'], np.ndarray) or isinstance(params['K'], list) and params['loopP1'] == 'alpha' ):
			params.update({'loopP1': 'alphaK'})
		elif ( isinstance(params['tau'], np.ndarray) or isinstance(params['tau'], list) and params['loopP1'] == 'alpha' ):
			params.update({'loopP1': 'alphaTau'})

		rescale_x = 1
		if params['rescale'] == '2alpha':
			rescale_x = 2

		print('alpha', params['x1'][:,0])

		# if params['loopP1'] == 'alphaTau':
		# 	scatt = np.array(list(itertools.product(*np.array([params['tau'], params['x2']]))))
		scatt = np.array(list(itertools.product(*np.array([params['x1'][:,0], params['x2']]))))
		params.update({'xscatt': scatt[:,0]*rescale_x})
		params.update({'yscatt': scatt[:,1]/params['w']})

	else:

		print('Prepare unspecified plot! No rescaling.')
		scatt = np.array(list(itertools.product(*np.array([params['x1'], params['x2']]))))
		params.update({'xscatt': xscatt[:,0]})
		params.update({'yscatt': yscatt[:,1]})

	print('params[*y*]',  params['y'])

	#print('params[*y*][0,:]', params['y'][0,:], ' \nparams[*y*][:,0]', params['y'][:,0])

	# print('Here we pick the largest gamma! Check when studying systems with N>2.')
	# ytemp = [];
	# for i in range(len(params['y'][:,0])):
	# 	if np.all(np.isnan(params['y'][i,:])==True):
	# 		ytemp.append(params['y'][i,0])
	# 	else:
	# 		if np.all(params['y'][i,np.isnan(params['y'][i,:])==False]) == 0:
	# 			ytemp.append(np.max(params['y'][i,:]))
	# 		else:
	# 			ytemp.append(np.max(params['y'][i,np.isnan(params['y'][i,:])==False]))

	#save results in pickle file
	filename = 'results/params_%s_vs_%s_%d:%d_%d_%d_%d'%(params['loopP1'], params['loopP2'], now.hour, now.minute, now.year, now.month, now.day)
	f 		 = open(filename,'wb')
	pickle.dump(params,f)
	f.close()

	print('Here we pick the largest gamma! Check when studying systems with N>2.')
	ytemp = [];
	for i in range(len(params['y'][:,0])):
		if np.all(np.isnan(params['y'][i,:])==True):
			ytemp.append(params['y'][i,0])
		else:
			ytemp.append(np.max(params['y'][i,np.isnan(params['y'][i,:])==False]))

	params.update({'yy': np.array(ytemp)})

	ytemp 		= np.array(ytemp)
	maxGamma 	= np.max(ytemp[np.isnan(ytemp)==False])							# for the case that one wants to set the negative alpha from -0.1 to max ytemp
	ytemp[ytemp == -0.1] = maxGamma

	params.update({'y': np.array(ytemp)})
	#params.update({'y': np.max(params['y'][:,:], axis=0)})

	#print('params[*y*]', params['y'])

	return params

# ******************************************************************************

def prepare2D(params):

	# pick out maximum value of the gamma solutions
	params.update({'y': np.max(params['y'][:,:], axis=1)})
	#print('params[*y*]', params['y'], '\t np.shape(params[*y*])', np.shape(params['y']))

	# reshape
	params['y']  = params['y'].reshape(len(params[params['discrP']]), len(params['x1']))
	params['y1'] = params['y1'].reshape(len(params[params['discrP']]), len(params['x1']))
	params['charEq_Re'] = params['charEq_Re'].reshape(len(params[params['discrP']]), len(params['x1']))
	params['charEq_Im'] = params['charEq_Im'].reshape(len(params[params['discrP']]), len(params['x1']))

	# for i in range(len(params[params['discrP']])):
	# 	for j in range(len(params['x1'])):
	# 		if (np.abs(params['K']*params['hp'](params['Omeg'][j]*params['x1'][j]))*np.abs(params['zeta'])*
	# 			np.abs(np.sin(params['x1'][j]*params['y'][i,j]))<=params['y'][i,j]): #params[params['discrP']][i]*
	# 			params['y'][i,j] = -0.5

	# rescale x-axis if plot against delay tau
	if  ( params['loopP1'] == 'wc' and params['loopP2'] == 'tau' ):
			params['x1'] = params['Omeg'] * params['x1'] / (2.0*np.pi)

	return params

# ******************************************************************************

def evaluateEq(params):

	sol = [];

	K 		= params.get('K')
	wc 		= params.get('wc')
	hp		= params.get('hp')
	tau		= params.get('tau')
	psi		= params.get('psi')
	zeta	= params.get('zeta')
	Omeg	= params.get('Omeg')
	alpha	= params.get('alpha')
	beta	= params.get('beta')

	loopP1	= params.get('loopP1')
	loopP2	= params.get('loopP2')

	if  ( loopP1 == 'K' and loopP2 == 'wc' ):

		print('Calculate x=K and y=wc!')
		for k in range(len(K)):
			#print('K[k], Omeg[k]', K[k], Omeg[k])
			#a = K[k] * hp( -Omeg[k]*tau + beta )
			for l in range(len(wc)):
				#a = K[k] * hp( -Omeg[k,l]*tau + beta )
				#print('a_old:', a, '\ta_synctools:', alpha[k,l]); time.sleep(0.5)
				a = alpha[k,l]
				#sol, cond_noInst = equationGamma(tau, a, wc[l], zeta, psi)
				#sol.append(sol)
				sol.append(equationGamma(tau, a, wc[l], zeta, psi))

	elif( loopP1 == 'wc' and loopP2 == 'K' ):

		print('Calculate x=K and y=wc!')
		for k in range(len(wc)):
			for l in range(len(K)):
				a = alpha[k,l]
				sol.append(equationGamma(tau, a, wc[k], zeta, psi))

	elif( loopP1 == 'alpha' and loopP2 == 'wc' ):

		if ( isinstance(params['K'], np.ndarray) or isinstance(params['K'], list) and params['loopP1'] == 'alpha' ):
			alphaK_alphaTau_len = len(params['K'])
			print('Plotting vs alpha: identified alpha(K)!')
		elif ( isinstance(params['tau'], np.ndarray) or isinstance(params['tau'], list) and params['loopP1'] == 'alpha' ):
			alphaK_alphaTau_len = len(params['tau'])
			print('Plotting vs alpha: identified alpha(tau)!')

		print('Calculate x=alpha and y=wc!')
		for k in range(alphaK_alphaTau_len):
			if ( isinstance(params['tau'], np.ndarray) or isinstance(params['tau'], list) and params['loopP1'] == 'alpha' ):
				tautemp = tau[k]
			else:
				tautemp = tau
			for l in range(len(wc)):
				a = alpha[k,l]
				sol.append(equationGamma(tautemp, a, wc[l], zeta, psi))

	elif( loopP1 == 'K' and loopP2 == 'tau' ):

		print('Calculate x=K and y=tau!')
		for k in range(len(K)):
			for l in range(len(tau)):
				a = alpha[k,l]
				sol.append(equationGamma(tau[l], a, wc, zeta, psi))

	elif( loopP1 == 'tau' and loopP2 == 'K' ):

		print('Calculate x=K and y=tau!')
		for k in range(len(tau)):
			for l in range(len(K)):
				a = alpha[k,l]
				sol.append(equationGamma(tau[k], a, wc, zeta, psi))

	elif( loopP1 == 'wc' and loopP2 == 'tau'):

		print('Calculate x=wc and y=tau!')
		for k in range(len(wc)):
			for l in range(len(tau)):
				a = alpha[k,l]
				if a > 0:
					sol.append(equationGamma(tau[l], a, wc[k], zeta, psi))
				else:
					sol.append([0, 0, 0, 0])

	elif( loopP1 == 'tau' and loopP2 == 'wc' ):

		print('Calculate x=tau and y=wc!')
		for k in range(len(tau)):
			for l in range(len(wc)):
				a = alpha[k,l]
				sol.append(equationGamma(tau[k], a, wc[l], zeta, psi))

	params.update({'y': np.array(sol)})

	#print('params[*y*]', params['y'])

	if not params['discrP'] == None:
		params['x1']=params.pop(loopP2)
	else:
		params['x1']=params.pop(loopP1)											# saves all alpha values under dict key 'x1'
		params['x2']=params.pop(loopP2)

	return params

# ******************************************************************************

# potentially cythonize!
def equationSigma(tau, a, wc, zeta, gamma=0, psi=-np.pi):
	''' USE for gammas without the condition
		np.abs( a*zeta*np.sin(gamma[i]*tau) ) <= gamma[i] and np.abs(gamma[i]) > zero_treshold '''
	# real zeta, approximation for small gamma*tau
	#sigma = -wc/2.0 + 1.0/tau * lambertw( -0.5*np.exp( 0.5*wc*tau )*wc*a*zeta*tau**2 )
	# real zeta, no approximation, need to calculate gamma assuming sigma=0, hence only valid for sigma << 1
	sigma = -wc/2.0 + 1.0/tau * lambertw( -(0.5/gamma)*np.exp( 0.5*wc*tau )*np.sin(gamma*tau-psi)*wc*a*np.abs(zeta)*tau )
	# real zeta, approximation of setting sigma**2 to zero in real part of char. equation,
	# need to calculate gamma assuming sigma=0, hence only valid for sigma << 1
	#sigma = gamma**2/wc - a + 1.0/tau * lambertw( np.exp( -(gamma**2/wc - a)*tau )*zeta*np.cos(gamma*tau)*a*tau )

	#print('type(sigma)', type(sigma))
	return sigma

# ******************************************************************************

# potentially cythonize!
def equationGamma(tau, a, wc, zeta, psi=-np.pi):

	if isinstance(zeta, list) or isinstance(zeta, np.ndarray):
		zetlen = len(zeta)
	elif isinstance(zeta, int) or isinstance(zeta, float):
		zetlen = 1; zeta = np.array([zeta]); psi = np.array([psi]);
	else:
		print('Error, zeta needs to be value or list!')

	result_gamma = []
	all_gamma	 = []
	cond_noInst  = []

	for i in range(zetlen):

		print('Compute gammas for set (tau, alpha, wc, zeta, psi):', tau, a, wc, zeta[i], psi[i])

		zero_treshold1 = 1E-17
		zero_treshold2 = 1E-17


		#A = wc**2.0 - 2.0*np.abs(a)*wc
		#B = (1.0-zeta**2.0)*(np.abs(a)*wc)**2.0
		A = wc**2.0 - 2.0*a*wc
		B = (1.0-zeta[i]**2.0)*(a*wc)**2.0
		# print('A:', A); print('B:', B);
		gamma = np.array([ +np.sqrt(0.5*(-A+A*np.sqrt(1.0-4.0*B/(A**2.0)))), +np.sqrt(0.5*(-A-A*np.sqrt(1.0-4.0*B/(A**2.0)))),
						   -np.sqrt(0.5*(-A+A*np.sqrt(1.0-4.0*B/(A**2.0)))), -np.sqrt(0.5*(-A-A*np.sqrt(1.0-4.0*B/(A**2.0)))) ])

		all_gamma.append(gamma)
		cond_noInst.append(0)
		print('Result before condition:', gamma); #time.sleep(0.25)

		for j in range(len(gamma)):													# this condition can also be checked in prepare2D

			if np.abs(gamma[j]) > zero_treshold2:
				if ( gamma[j] )**2 - np.abs(a)*wc*(1.0-np.abs(zeta[i])*np.cos( gamma[j]*tau-psi[i]) )  < 0.0: #if fullfilled, stable synced state!
					if   gamma[j] > 0.0 and -wc*(gamma[j] + np.abs(a)*np.abs(zeta[i])*np.sin( gamma[j]*tau-psi[i]) ) < 0.0: #  wc+np.abs(a)*np.abs(zeta)*tau*np.abs( np.cos( gamma[j]*tau-psi) ) >0.0) or (    ( gamma[j] )**2 - np.abs(a)*wc*(1.0-np.abs(zeta)*np.abs( np.cos( gamma[j]*tau-psi) ) ) < 0.0 and  wc+np.abs(a)*np.abs(zeta)*tau*np.abs( np.cos( gamma[j]*tau-psi) ) >0.0):
						gamma[j] = None
					elif gamma[j] < 0.0 and -wc*(gamma[j] - np.abs(a)*np.abs(zeta[i])*np.sin( gamma[j]*tau-psi[i]) ) > 0.0: #  wc+np.abs(a)*np.abs(zeta)*tau*np.abs( np.cos( gamma[j]*tau-psi) ) >0.0) or (    ( gamma[j] )**2 - np.abs(a)*wc*(1.0-np.abs(zeta)*np.abs( np.cos( gamma[j]*tau-psi) ) ) < 0.0 and  wc+np.abs(a)*np.abs(zeta)*tau*np.abs( np.cos( gamma[j]*tau-psi) ) >0.0):
						gamma[j] = None

			elif np.abs(gamma[j]) < zero_treshold2:									# gamma = 0 means that there is no instability
				gamma[j] = None

		print('Result after condition:', gamma); #time.sleep(0.25)

		# ADD condition that checks whether gamma fulfill Re and Im part equation, or which is closer...

			# if gamma[j] != 0:
			# 	if ( gamma[j] )**2 - np.abs(a)*wc*(1.0-np.abs(zeta)*np.cos( gamma[j]*tau-psi) )  < 0.0:
			# 		if   gamma[j] > 0.0 and -wc*(gamma[j] + np.abs(a)*np.abs(zeta)*np.sin( gamma[j]*tau-psi) ) < 0.0: #  wc+np.abs(a)*np.abs(zeta)*tau*np.abs( np.cos( gamma[j]*tau-psi) ) >0.0) or (    ( gamma[j] )**2 - np.abs(a)*wc*(1.0-np.abs(zeta)*np.abs( np.cos( gamma[j]*tau-psi) ) ) < 0.0 and  wc+np.abs(a)*np.abs(zeta)*tau*np.abs( np.cos( gamma[j]*tau-psi) ) >0.0):
			# 			gamma[j] = None
			# 		elif gamma[j] < 0.0 and -wc*(gamma[j] - np.abs(a)*np.abs(zeta)*np.sin( gamma[j]*tau-psi) ) > 0.0: #  wc+np.abs(a)*np.abs(zeta)*tau*np.abs( np.cos( gamma[j]*tau-psi) ) >0.0) or (    ( gamma[j] )**2 - np.abs(a)*wc*(1.0-np.abs(zeta)*np.abs( np.cos( gamma[j]*tau-psi) ) ) < 0.0 and  wc+np.abs(a)*np.abs(zeta)*tau*np.abs( np.cos( gamma[j]*tau-psi) ) >0.0):
			# 			gamma[j] = None

		#print('Gammas left:', gamma); #time.sleep(1)

		if a <= zero_treshold1:														# for alpha < 0, the state for the above parameters IS UNSTABLE
			#print('Set gamma to zero since alpha<0!'); time.sleep(2)
			gamma[:] = -0.1;

		result_gamma.append(gamma)

	gamma = np.concatenate(result_gamma).tolist()

	return gamma#, all_gamma
