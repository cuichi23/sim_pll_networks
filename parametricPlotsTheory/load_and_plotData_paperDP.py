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

sys.path.append('..')

import synctools_interface_lib as synctools
from function_lib import solveLinStab
import function_lib as fct_lib
import parametricPlots as paraPlot
import palettable.colorbrewer.diverging as colormap_diver
from palettable.colorbrewer.diverging import PuOr_7

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
		'size'   : 16,
		}

annotationfont = {
		'family' : 'monospace',
		'color'  : (0, 0.27, 0.08),
		'weight' : 'normal',
		'size'   : 14,
		}

# plot parameter
axisLabel 			= 44;
tickSize  			= 30;
titleLabel			= 10;
dpi_val   			= 150;
figwidth  			= 10;
figheight 			= 6;
labelpadxaxis		= 10;
labelpadyaxis		= 20
colormapSyncStab 	= colormap_diver.PuOr_7.mpl_colormap

################################################################################
################################################################################
filename = 'results/params_tau_vs_K_19:34_2021_7_9'
################################################################################
################################################################################

params 	 = pickle.load(open(filename, 'rb'))
#print(params); sys.exit()
################################################################################
print('Here we pick the largest gamma! Check when studying systems with N>2.')
ytemp = [];
for i in range(len(params['y'][:,0])):
	if np.all(np.isnan(params['y'][i,:])==True):
		ytemp.append(params['y'][i,0])
	else:
		ytemp.append(np.max(params['y'][i,np.isnan(params['y'][i,:])==False]))

params.update({'yy': np.array(ytemp)})

ytemp 		= np.array(ytemp)
maxGamma 	= np.max(ytemp[np.isnan(ytemp)==False])								# for the case that one wants to set the negative alpha from -0.1 to max ytemp
ytemp[ytemp == -999] = maxGamma+1


params.update({'y': np.array(ytemp)})

fzeta = 1-np.sqrt(1-np.abs(params['zeta'][0] )**2)

if ( params['loopP1'] == 'K' and params['loopP2'] == 'fric' ):
	CondStab = np.zeros([len(params['x1']), len(params['x2'])]);
	for i in range(len(params['x1'])):
		for j in range(len(params['x2'])):

			if params['wc']*params['x2'][j]**2/(2*params['alpha'][i,j]) > fzeta:
				CondStab[i,j] = 0
			elif params['wc']*params['x2'][j]**2/(2*params['alpha'][i,j]) > 1:
				CondStab[i,j] = 1

			else:
				CondStab[i,j] = None

elif ( params['loopP1'] == 'tau' and params['loopP2'] == 'K' ):
	CondStab = np.zeros([len(params['x1']), len(params['x2'])]);
	for i in range(len(params['x1'])):
		for j in range(len(params['x2'])):

			if params['wc']*params['fric']**2/(2*params['alpha'][i,j]) > fzeta:
				CondStab[i,j] = 0
			elif params['wc']*params['fric']**2/(2*params['alpha'][i,j]) > 1:
				CondStab[i,j] = 1
			else:
				CondStab[i,j] = None

elif ( params['loopP1'] == 'tau' and params['loopP2'] == 'fric' ):
	CondStab = np.zeros([len(params['x1']), len(params['x2'])]);
	for i in range(len(params['x1'])):
		for j in range(len(params['x2'])):

			if params['wc']*params['x2'][j]**2/(2*params['alpha'][i,j]) > fzeta:
				CondStab[i,j] = 0
			elif params['wc']*params['x2'][j]**2/(2*params['alpha'][i,j]) > 1:
				CondStab[i,j] = 1
			else:
				CondStab[i,j] = None

elif ( params['loopP1'] == 'tau' and params['loopP2'] == 'wc' ):
	CondStab = np.zeros([len(params['x1']), len(params['x2'])]);
	for i in range(len(params['x1'])):
		for j in range(len(params['x2'])):

			if params['x2'][j]*params['fric']**2/(2*params['alpha'][i,j]) > fzeta:
				CondStab[i,j] = 0
			elif params['x2'][j]*params['fric']**2/(2*params['alpha'][i,j]) > 1:
				CondStab[i,j] = 1
			else:
				CondStab[i,j] = None

elif ( params['loopP1'] == 'K' and params['loopP2'] == 'wc' ):
	CondStab = np.zeros([len(params['x1']), len(params['x2'])]);
	#print(len(params['x1']))
	#print(len(params['x2']))
	for i in range(len(params['x1'])):
		for j in range(len(params['x2'])):


			if (params['x2'][j]*params['fric']**2)/(2*params['alpha'][i,j]) > fzeta:
				CondStab[i,j] = 0
			elif (params['x2'][j]*params['fric']**2)/(2*params['alpha'][i,j]) > 1:
				CondStab[i,j] = 1
			else:
				CondStab[i,j] = None
params.update({'CondStab': CondStab})

# save results in pickle file

filename = 'results/paramsCURED_%s_vs_%s_%d:%d_%d_%d_%d'%(params['loopP1'], params['loopP2'], now.hour, now.minute, now.year, now.month, now.day)
f 		 = open(filename,'wb')
pickle.dump(params,f)
f.close()

################################################################################

labeldict 	= {'wc': r'$\frac{\omega_\textrm{c}}{\omega}$', 'tau': r'$\omega\tau/2\pi$', 'K': r'$\frac{K}{\omega}$', 'fric': r'$\gamma$',
					'a': r'$\alpha$', 'Omeg': r'$\Omega$', 'zeta': r'$\zeta$', 'beta': r'$\beta$', 'alphaK': r'$2\alpha(K)$', 'alphaTau': r'$2\alpha(\tau)$'}

''' prepare colormap for scatter plot that is always in [0, 1] or [min(results), max(results)] '''
cdict = {
  'red'  :  ( (0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
  'green':  ( (0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
  'blue' :  ( (0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
}
colormap  = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

fig1 = plt.figure(num=1, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig1.set_size_inches(figwidth,figheight)

plt.clf()
ax = plt.subplot(1, 1, 1)
#ax.set_aspect('equal')
ytemp = params['yy']
for i in range(len(ytemp)):
	if (ytemp[i] != -999 and np.isnan(ytemp[i])==False):
		ytemp[i] = 1
	elif (ytemp[i] == -999 and np.isnan(ytemp[i])==False):
		ytemp[i] = -0.1
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
tempresults_ma11 = ma.masked_where(tempresults > 1., tempresults)			# Create masked array
#print('tempresult_ma:', tempresults_ma)
#print('initPhiPrime0:', initPhiPrime0)
#cmap_choice 	= colors.ListedColormap(['lightskyblue'])													# cm.coolwarm
#cmap_neg_alpha 	= colors.ListedColormap(['brown'])
#cmap_cond 		= colors.ListedColormap(['navy'])
#cmap_unstabl 		= colors.ListedColormap(['tomato'])

cmap_choice 	= colors.ListedColormap(['white'])													# cm.coolwarm
cmap_neg_alpha 	= colors.ListedColormap(['#12808a'])
cmap_cond 		= colors.ListedColormap(['#e0e1d2'])
cmap_unstabl 		= colors.ListedColormap(['#9908ff'])
try:
	plt.imshow(tempresults_ma.astype(float), interpolation='nearest', cmap=cmap_choice, aspect='auto', origin='lower',
			extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
			vmin=np.min(tempresults_ma[np.isnan(tempresults_ma)==False]), vmax=np.max(tempresults_ma[np.isnan(tempresults_ma)==False]) )
	plt.imshow(tempresults_ma11.astype(float), interpolation='nearest', cmap=cmap_unstabl, aspect='auto', origin='lower',
			extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
			vmin=np.min(tempresults_ma11[np.isnan(tempresults_ma11)==False]), vmax=np.max(tempresults_ma11[np.isnan(tempresults_ma11)==False]), alpha=1 )
			#vmin=np.min(tempresults_ma), vmax=np.max(tempresults_ma) )
	#plt.colorbar();
	plt.imshow(tempresults_ma1.astype(float), interpolation='nearest', cmap=cmap_neg_alpha, aspect='auto', origin='lower',
			extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
			vmin=np.min(tempresults_ma1), vmax=np.max(tempresults_ma1) , alpha=1)
	plt.imshow(tempcond.astype(float), interpolation='nearest', cmap=cmap_cond, aspect='auto', origin='lower',
			extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
			vmin=np.min(tempcond), vmax=np.max(tempcond) )
except:
	print('Exception mode when plotting imshow!');
	plt.imshow(tempresults_ma.astype(float), interpolation='nearest', cmap=cmap_choice, aspect='auto', origin='lower',
			extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
			vmin=np.min(tempresults_ma), vmax=np.max(tempresults_ma) )
	plt.imshow(tempresults_ma11.astype(float), interpolation='nearest', cmap=cmap_unstabl, aspect='auto', origin='lower',
			extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
			vmin=np.min(tempresults_ma11[np.isnan(tempresults_ma11)==False]), vmax=np.max(tempresults_ma11[np.isnan(tempresults_ma11)==False]) , alpha=1)
			#vmin=np.min(tempresults_ma), vmax=np.max(tempresults_ma) )
	plt.imshow(tempresults_ma1.astype(float), interpolation='nearest', cmap=cmap_neg_alpha, aspect='auto', origin='lower',
			extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
			vmin=np.min(tempresults_ma1), vmax=np.max(tempresults_ma1) , alpha=1)
	plt.imshow(tempcond.astype(float), interpolation='nearest', cmap=cmap_cond, aspect='auto', origin='lower',
			extent=(params['xscatt'].min(), params['xscatt'].max(), params['yscatt'].min(), params['yscatt'].max()),
			vmin=np.min(tempcond), vmax=np.max(tempcond) )

if ( params['loopP1'] == 'alphaK' and ( isinstance(params['K'], np.ndarray) or isinstance(params['K'], list)) ):
	print('Plot condition region into the plot!')
	plt.plot(2*params['xscatt'], 2*params['xscatt'], 'k-', linewidth=1);
	plt.plot(2*params['xscatt'], (1+np.sqrt(1-params['zeta']**2))*2*params['xscatt'], 'r--', linewidth=1); #*2*params['alpha'][:,0]

plt.xlabel(labeldict[params['loopP1']], fontsize=axisLabel, labelpad=labelpadxaxis)
plt.ylabel(labeldict[params['loopP2']], fontsize=axisLabel, rotation=0, labelpad=labelpadyaxis)
plt.tick_params(axis='both', which='major', labelsize=tickSize)
plt.xlim([np.min(params['xscatt']), np.max(params['xscatt'])])
plt.ylim([np.min(params['yscatt']), np.max(params['yscatt'])])
plt.tight_layout();
plt.savefig('plots/imshow_%s_vs_%s_%d_%d_%d.svg' %(params['loopP1'], params['loopP2'], now.year, now.month, now.day), dpi=dpi_val)
plt.savefig('plots/imshow_%s_vs_%s_%d_%d_%d.png' %(params['loopP1'], params['loopP2'], now.year, now.month, now.day), dpi=dpi_val)
plt.savefig('plots/imshow_%s_vs_%s_%d_%d_%d.pdf' %(params['loopP1'], params['loopP2'], now.year, now.month, now.day), dpi=dpi_val)
################################################################################

fig2 = plt.figure(num=2, figsize=(figwidth, figheight), dpi=dpi_val, facecolor='w', edgecolor='k')
fig2.set_size_inches(figwidth,figheight)

plt.clf()
ax = plt.subplot(1, 1, 1)
#ax.set_aspect('equal')

cmap_choice = cm.RdYlBu														# cm.coolwarm
ytemp = params['yy']
for i in range(len(ytemp)):
	if (ytemp[i] != -999 and np.isnan(ytemp[i])==False):
		ytemp[i] = 1
	elif (ytemp[i] == -999 and np.isnan(ytemp[i])==False):
		ytemp[i] = -0.1
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

plt.xlabel(labeldict[params['loopP1']], fontsize=axisLabel, labelpad=labelpadxaxis)
plt.ylabel(labeldict[params['loopP2']], fontsize=axisLabel, rotation=0, labelpad=labelpadyaxis)
plt.tick_params(axis='both', which='major', labelsize=tickSize)
plt.xlim([np.min(params['xscatt']), np.max(params['xscatt'])])
plt.ylim([np.min(params['yscatt']), np.max(params['yscatt'])])
plt.tight_layout(); # plt.colorbar();
plt.savefig('plots/imshow1color_%s_vs_%s_%d_%d_%d.svg' %(params['loopP1'], params['loopP2'], now.year, now.month, now.day), dpi=dpi_val)
plt.savefig('plots/imshow1color_%s_vs_%s_%d_%d_%d.png' %(params['loopP1'], params['loopP2'], now.year, now.month, now.day), dpi=dpi_val)
plt.savefig('plots/imshow1color_%s_vs_%s_%d_%d_%d.pdf' %(params['loopP1'], params['loopP2'], now.year, now.month, now.day), dpi=dpi_val)
################################################################################

print('Prepare %s vs %s plot!'%(params['loopP1'], params['loopP2']))

if  ( params['loopP1'] == 'tau' and params['loopP2'] == 'wc' ):

	paraPlot.makePlotsFromSynctoolsResults(100, params['x1'], params['x2'], params['Omeg'], params['w']/(2.0*np.pi), 1.0/params['w'], 1.0,
					r'$\frac{\omega\tau}{2\pi}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\Omega$', 'tau', 'K', 'Omeg', None, cm.coolwarm)
	paraPlot.makePlotsFromSynctoolsResults(101, params['x1'], params['x2'], params['ReLambSynctools'], params['w']/(2.0*np.pi), 1.0/params['w'], params['w']/(2.0*np.pi),
					r'$\frac{\omega\tau}{2\pi}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\frac{\textrm{Re}(\lambda)\omega}{2\pi}$', 'tau', 'wc', 'ReLambda', None, colormapSyncStab)
	paraPlot.makePlotsFromSynctoolsResults(102, params['x1'], params['x2'], params['ImLambSynctools'], params['w']/(2.0*np.pi), 1.0/params['w'], 1.0/params['w'],
					r'$\frac{\omega\tau}{2\pi}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\frac{\textrm{Im}(\lambda)}{\omega}$', 'tau', 'wc', 'ImLambda', None, colormapSyncStab)

elif( params['loopP1'] == 'tau' and params['loopP2'] == 'K' ):

	paraPlot.makePlotsFromSynctoolsResults(100, params['x1'], params['x2'], params['Omeg'], params['w']/(2.0*np.pi), 1.0/params['w'], 1.0,
					r'$\frac{\omega\tau}{2\pi}$', r'$\frac{K}{\omega}$', r'$\Omega$', 'tau', 'K', 'Omeg', None, cm.coolwarm)
	paraPlot.makePlotsFromSynctoolsResults(101, params['x1'], params['x2'], params['ReLambSynctools'], params['w']/(2.0*np.pi), 1.0/params['w'], params['w']/(2.0*np.pi),
					r'$\frac{\omega\tau}{2\pi}$', r'$\frac{K}{\omega}$', r'$\frac{\textrm{Re}(\lambda)\omega}{2\pi}$', 'tau', 'K', 'ReLambda', None, colormapSyncStab)
	paraPlot.makePlotsFromSynctoolsResults(102, params['x1'], params['x2'], params['ImLambSynctools'], params['w']/(2.0*np.pi), 1.0/params['w'], 1.0/params['w'],
					r'$\frac{\omega\tau}{2\pi}$', r'$\frac{K}{\omega}$', r'$\frac{\textrm{Im}(\lambda)}{\omega}$', 'tau', 'K', 'ImLambda', None, colormapSyncStab)

elif( params['loopP1'] == 'K' and params['loopP2'] == 'wc' ):

	paraPlot.makePlotsFromSynctoolsResults(100, params['x1'], params['x2'], params['Omeg'], 1.0/params['w'], 1.0/params['w'], 1.0,
					r'$\frac{K}{\omega}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\Omega$', 'K', 'wc', 'Omeg', None, cm.coolwarm)
	paraPlot.makePlotsFromSynctoolsResults(101, params['x1'], params['x2'], params['ReLambSynctools'], 1.0/params['w'], 1.0/params['w'], params['w']/(2.0*np.pi),
					r'$\frac{K}{\omega}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\frac{\textrm{Re}(\lambda)\omega}{2\pi}$', 'K', 'wc', 'ReLambda', None, colormapSyncStab)
	paraPlot.makePlotsFromSynctoolsResults(102, params['x1'], params['x2'], params['ImLambSynctools'], 1.0/params['w'], 1.0/params['w'], 1.0/params['w'],
					r'$\frac{K}{\omega}$', r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\frac{\textrm{Im}(\lambda)}{\omega}$', 'K', 'wc', 'ImLambda', None, colormapSyncStab)

elif( params['loopP1'] == 'K' and params['loopP2'] == 'fric' ):

	#		 makePlotsFromSynctoolsResults(figID, x, y,  z, rescale_x, rescale_y, rescale_z, x_label, y_label, z_label, x_identifier, y_identifier, z_identifier)
	paraPlot.makePlotsFromSynctoolsResults(100, params['x1'], params['x2'], params['Omeg'], 1.0/params['w'], 1, 1.0,
					r'$\frac{K}{\omega}$', r'$\gamma$', r'$\Omega$', 'K', 'fric', 'Omeg', None, cm.coolwarm)
	paraPlot.makePlotsFromSynctoolsResults(101, params['x1'], params['x2'], params['ReLambSynctools'], 1.0/params['w'], 1, params['w']/(2.0*np.pi),
					r'$\frac{K}{\omega}$', r'$\gamma$', r'$\frac{\textrm{Re}(\lambda)\omega}{2\pi}$', 'K', 'fric', 'ReLambda', None, colormapSyncStab)
	paraPlot.makePlotsFromSynctoolsResults(102, params['x1'], params['x2'], params['ImLambSynctools'], 1.0/params['w'], 1, 1.0/params['w'],
					r'$\frac{K}{\omega}$', r'$\gamma$', r'$\frac{\textrm{Im}(\lambda)}{\omega}$', 'K', 'fric', 'ImLambda', None, colormapSyncStab)

elif( params['loopP1'] == 'tau' and params['loopP2'] == 'fric' ):

	#		 makePlotsFromSynctoolsResults(figID, x, y,  z, rescale_x, rescale_y, rescale_z, x_label, y_label, z_label, x_identifier, y_identifier, z_identifier)
	paraPlot.makePlotsFromSynctoolsResults(100, params['x1'], params['x2'], params['Omeg'], 1.0/params['w'], 1, 1.0,
					r'$\frac{\omega\tau}{2\pi}$', r'$\gamma$', r'$\Omega$', 'K', 'fric', 'Omeg', None, cm.coolwarm)
	paraPlot.makePlotsFromSynctoolsResults(101, params['x1'], params['x2'], params['ReLambSynctools'], 1.0/params['w'], 1, params['w']/(2.0*np.pi),
					r'$\frac{\omega\tau}{2\pi}$', r'$\gamma$', r'$\frac{\textrm{Re}(\lambda)\omega}{2\pi}$', 'K', 'fric', 'ReLambda', None, colormapSyncStab)
	paraPlot.makePlotsFromSynctoolsResults(102, params['x1'], params['x2'], params['ImLambSynctools'], 1.0/params['w'], 1, 1.0/params['w'],
					r'$\frac{\omega\tau}{2\pi}$', r'$\gamma$', r'$\frac{\textrm{Im}(\lambda)}{\omega}$', 'K', 'fric', 'ImLambda', None, colormapSyncStab)

elif( params['loopP1'] == 'wc' and params['loopP2'] == 'fric' ):

	#		 makePlotsFromSynctoolsResults(figID, x, y,  z, rescale_x, rescale_y, rescale_z, x_label, y_label, z_label, x_identifier, y_identifier, z_identifier)
	paraPlot.makePlotsFromSynctoolsResults(100, params['x1'], params['x2'], params['Omeg'], 1.0/w, 1, 1.0,
					r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\gamma$', r'$\Omega$', 'K', 'fric', 'Omeg', None, cm.coolwarm)
	paraPlot.makePlotsFromSynctoolsResults(101, params['x1'], params['x2'], params['ReLambSynctools'], 1.0/params['w'], 1, params['w']/(2.0*np.pi),
					r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\gamma$', r'$\frac{\textrm{Re}(\lambda)\omega}{2\pi}$', 'K', 'fric', 'ReLambda', None, colormapSyncStab)
	paraPlot.makePlotsFromSynctoolsResults(102, params['x1'], params['x2'], params['ImLambSynctools'], 1.0/params['w'], 1, 1.0/params['w'],
					r'$\frac{\omega_\textrm{c}}{\omega}$', r'$\gamma$', r'$\frac{\textrm{Im}(\lambda)}{\omega}$', 'K', 'fric', 'ImLambda', None, colormapSyncStab)

plt.draw(); #plt.show();

plt.draw()
plt.show()
