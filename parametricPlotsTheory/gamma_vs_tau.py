#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import numpy as np
import datetime
import os, gc
now = datetime.datetime.now()

import parametricPlots as paraPlot

w 		= 2.0*np.pi*1.0
K 		= 2.0*np.pi*0.4 #0.37/2
z 		= -1
wc		= 2.0*np.pi*np.array([0.014])#, 0.14, 0.9])

h  		= lambda x: -np.cos(x)
hp 		= lambda x: np.sin(x)

beta 	= 0

tau 	= np.arange(0, 10, 0.0001)
p   	= np.arange(0, 8, 0.0001)

# provide the parameter that is NOT the x-axis as the outer loop, i.e., loopP1
loopP1	= 'wc'																	# discrete parameter
loopP2 	= 'tau'																	# x-axis
discrP	= 'wc'																	# plot for different values of this parameter if given

######################################################################################################################################
# no user input below (unless you know what you are doing!)

params  = {'p': p, 'h': h, 'hp': hp, 'w': w, 'K': K, 'wc': wc,
			'tau': tau, 'zeta': z, 'beta': beta, 'loopP1': loopP1, 'loopP2': loopP2, 'discrP': discrP}

if ( isinstance(params[params['discrP']], int) or isinstance(params[params['discrP']], float) ):
	#print('type(params[params[*discrP*]])', type(params[params['discrP']]))
	params[params['discrP']] = [params[params['discrP']]]
	#print('type(params[params[*discrP*]])', type(params[params['discrP']]))

paraPlot.plot2D(params)
