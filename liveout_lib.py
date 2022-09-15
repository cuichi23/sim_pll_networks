#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import evaluation_lib as eva
import numpy as np
import numpy.ma as ma
import matplotlib
import codecs
import csv
import os, gc
if not os.environ.get('SGE_ROOT') == None:																				# this environment variable is set within the queue network, i.e., if it exists, 'Agg' mode to supress output
	print('NOTE: \"matplotlib.use(\'Agg\')\"-mode active, plots are not shown on screen, just saved to results folder!\n')
	matplotlib.use('Agg') #'%pylab inline'
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import optimize
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#from scipy.interpolate import spline
from scipy.signal import square
import math

import datetime
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


# see: https://makersportal.com/blog/2018/8/14/real-time-graphing-in-python
# and or
#
# https://www.google.com/search?q=real+time+graphical+output+python&hl=en&sxsrf=ALeKk02FiCTvM9VTKA9VrSkxyE8bWnmmYg%3A1616324318246&ei=3iZXYMGzDoPAlAbzrbeAAw&oq=real+time+graphical+output+python&gs_lcp=Cgdnd3Mtd2l6EAM6BwgAEEcQsAM6BQgAEJECOgIIADoCCC46BAguEEM6CAguEMcBEKMCOgQIABBDOgUIABDJAzoICC4QxwEQrwE6BggAEBYQHjoICAAQFhAKEB46CAgAEAgQBxAeOggIABAIEA0QHjoECAAQHjoGCAAQCBAeOgoIABAIEA0QChAeOgYIABANEB5Q8aAvWI-JMGD_lTBoBHACeACAAacBiAGqF5IBBDI3LjWYAQCgAQGqAQdnd3Mtd2l6yAEIwAEB&sclient=gws-wiz&ved=0ahUKEwjBmtKmncHvAhUDIMUKHfPWDTAQ4dUDCAw&uact=5#kpvalbx=_uitXYIe-C6KAi-gPp-yvuA497
# https://medium.com/intel-student-ambassadors/live-graph-simulation-using-python-matplotlib-and-pandas-30ea4e50f883
#
# connect to audio?
# https://medium.com/quick-code/graphing-real-time-audio-with-python-213be536b094

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')

def live_plotter(x_vec, y1_data, line1, identifier='', pause_time=0.01):
	if line1==[]:
		# this is the call to matplotlib that allows dynamic plotting
		plt.ion()
		fig = plt.figure(figsize=(13,6))
		ax  = fig.add_subplot(111)
		# create a variable for the line so we can later update it
		line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)
		#update plot label/title
		plt.ylabel('Y Label')
		plt.title('Title: {}'.format(identifier))
		plt.show()

	# after the figure, axis, and line are created, we only need to update the y-data
	line1.set_ydata(y1_data)
	# adjust limits if new data goes beyond bounds
	if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
		plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
	# this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
	plt.pause(pause_time)

	# return line so we can update it again in the next iteration
	return line1
