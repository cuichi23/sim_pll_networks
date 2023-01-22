#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.signal import square
import matplotlib
import os
if not os.environ.get('SGE_ROOT') is None:										# this environment variable is set within the queue network, i.e., if it exists, 'Agg' mode to supress output
	print('NOTE: \"matplotlib.use(\'Agg\')\"-mode active, plots are not shown on screen, just saved to results folder!\n')
	matplotlib.use('Agg') #'%pylab inline'
import matplotlib.pyplot as plt

def cutTimeSeriesOfIntegerPeriod(Fsim: np.float, Tsim: np.float, delay: np.float, syncF: np.float, maxK: np.float, phase_or_signal: np.ndarray, psd_id: np.int, percentOfTsim: np.float, signal_given: bool=False):

	print('Fct: cutTimeSeriesOfIntegerPeriod -- trying to extract an integer number of periods from the time-series of PLL %i!' % psd_id)
	# Tsim = 1000; Fsim = 125; f = 0.999; phiInit = 0.0; percentOfTsim = 0.75;
	if not signal_given:
		signal = square(phase_or_signal, duty=0.5)											# history [-tau, 0) and [0, Tsim]
	else:
		signal = phase_or_signal

	siglen = len(signal)
	analyzeL = int(percentOfTsim*siglen)
	#print(analyzeL)
	testplot = True
	#print('analyze', analyzeL1/,' periods')

	f = syncF + maxK

	# find rising edge close to t[-analyzeL] and t[-1] -- NOTE f represents Omega here
	widthWinI = 3.3
	widthWin1E = 3.3
	widthWin2E = 3.05
	widthWin3E = 3.55
	indexesLowStateI = np.where(signal[-analyzeL-int(widthWinI*Fsim/f):-analyzeL] != 1)
	indexesHigStateI = np.where(signal[-analyzeL-int(widthWinI*Fsim/f):-analyzeL] !=-1)
	indexesLowStateE = np.where(signal[-int(widthWin1E*Fsim/f):] != 1)
	indexesHigStateE = np.where(signal[-int(widthWin1E*Fsim/f):] !=-1)
	#print('indexesLowStateI and indexesHigStateI:', indexesLowStateI, '\t', indexesHigStateI)
	#print('np.shape(indexesLowStateI) and np.shape(indexesHigStateI):', np.shape(indexesLowStateI), '\t', np.shape(indexesHigStateI))

	# plt.figure(1)
	# shift = 0
	# plt.plot(signal[-analyzeL-int(widthWinI*Fsim/f)-shift:-analyzeL+shift], 'g')
	# plt.plot(signal[-int(widthWinI*Fsim/f)-shift:], 'r')
	# plt.draw(); plt.show()

	# if there are not high and low states in the monitored area. the search window needs too be widened
	if not ( np.shape(indexesLowStateI)[1]>0 and np.shape(indexesHigStateI)[1]>0 and np.shape(indexesLowStateE)[1]>0 and np.shape(indexesHigStateE)[1]>0 ):
		print('WIDER WINDOW')
		widthWinI  = 6.3;
		widthWin1E = 6.3; widthWin2E = 6.05; widthWin3E = 6.55;
		indexesLowStateI = np.where(signal[-analyzeL-int(widthWinI*Fsim/f):-analyzeL]!= 1)
		indexesHigStateI = np.where(signal[-analyzeL-int(widthWinI*Fsim/f):-analyzeL]!=-1)
		indexesLowStateE = np.where(signal[-int(widthWin1E*Fsim/f):]!= 1)
		indexesHigStateE = np.where(signal[-int(widthWin1E*Fsim/f):]!=-1)

	#print('signal:', signal)
	#print('signal analyzed for LowStateI:', signal[-analyzeL-int(widthWinI*Fsim/f):-analyzeL])
	#print('indexesLowStateI and indexesHigStateI:', indexesLowStateI, '\t', indexesHigStateI)
	#print('np.shape(indexesLowStateI) and np.shape(indexesHigStateI):', np.shape(indexesLowStateI), '\t', np.shape(indexesHigStateI))
	#print('len(indexesLowStateI) and len(indexesHigStateI):', len(indexesLowStateI), '\t', len(indexesHigStateI))

	if indexesLowStateI[0][0] == 0 and indexesHigStateI[0][0] != 0:
		firstRisingEdgeIndI 		= indexesHigStateI[0][0];					# this specifies the first index of the high state in this search window
		indexTimeFirstRisingEdgeI	= siglen-analyzeL-int(widthWinI*Fsim/f)+firstRisingEdgeIndI
		print('First edge in the inital time window will be rising at time t=%0.5f!'%(indexTimeFirstRisingEdgeI/Fsim))
		print('signal[indexTimeFirstRisingEdgeI-7:indexTimeFirstRisingEdgeI+7]', signal[indexTimeFirstRisingEdgeI-7:indexTimeFirstRisingEdgeI+7])

		if indexesLowStateE[0][0] == 0:											# we found a rising edge above... now we need to find another one at the end, if the first state in the search window is a low state
																				# then the first edge must be a rising edge, this one we choose below
			print('First edge in the final time window is rising!')
			firstRisingEdgeIndE 		= indexesHigStateE[0][0];
			indexTimeFirstRisingEdgeE	= siglen-int(widthWin1E*Fsim/f)+firstRisingEdgeIndE
			print('signal[indexTimeFirstRisingEdgeE-7:indexTimeFirstRisingEdgeE+7]', signal[indexTimeFirstRisingEdgeE-7:indexTimeFirstRisingEdgeE+7])

		else: # change window to find other edge, - shift window by -T/2 in time
			##### look below for potentially more elegant way (given the predicted frequency is met)
			print('Try to find a rising edge in last window, when shifted to widthWin2E!')
			indexesLowStateE 		= np.where(signal[-int(widthWin2E*Fsim/f):]!=  1)
			indexesHigStateE 		= np.where(signal[-int(widthWin2E*Fsim/f):]!= -1)
			if indexesLowStateE[0][0] == 0:
				print('First edge in the by widthWin2E shifted final time window is rising!')
				firstRisingEdgeIndE 		= indexesHigStateE[0][0];
				indexTimeFirstRisingEdgeE	= siglen-int(widthWin2E*Fsim/f)+firstRisingEdgeIndE
				print('signal[indexTimeFirstRisingEdgeE-7:indexTimeFirstRisingEdgeE+7]', signal[indexTimeFirstRisingEdgeE-7:indexTimeFirstRisingEdgeE+7])
			else:
				print('Try to find a rising edge in last window, when shifted to widthWin3E!')
				indexesLowStateE 		= np.where(signal[-int(widthWin3E*Fsim/f):]!=  1)
				indexesHigStateE 		= np.where(signal[-int(widthWin3E*Fsim/f):]!= -1)
				if indexesLowStateE[0][0] == 0:
					print('First edge in the by widthWin3E shifted final time window is rising!')
					firstRisingEdgeIndE 		= indexesHigStateE[0][0];
					indexTimeFirstRisingEdgeE	= siglen-int(widthWin3E*Fsim/f)+firstRisingEdgeIndE
					print('signal[indexTimeFirstRisingEdgeE-7:indexTimeFirstRisingEdgeE+7]', signal[indexTimeFirstRisingEdgeE-7:indexTimeFirstRisingEdgeE+7])
				else:
					print('\n\n...debug!1\n\n');
		print('analyzed nsteps divided by steps equivalent to a period', (indexTimeFirstRisingEdgeE - indexTimeFirstRisingEdgeI) / (Fsim/syncF) )
		if testplot:
			t = np.arange(0,Tsim+delay,1/Fsim)
			timeFirstRisingEdgeI		= t[indexTimeFirstRisingEdgeI]
			timeFirstRisingEdgeE		= t[indexTimeFirstRisingEdgeE]
			print('timeFirstRisingEdgeE[indexTimeFirstRisingEdgeE]:', timeFirstRisingEdgeE)
			print('timeFirstRisingEdgeI[indexTimeFirstRisingEdgeI]:', timeFirstRisingEdgeI)
			print('analyzed time divided by period of synced state (DO NOT expect an integer number for noisy simulation!)', (timeFirstRisingEdgeE - timeFirstRisingEdgeI) * syncF )
			plt.figure(1000+psd_id); plt.clf();
			plt.plot(t[-analyzeL-int(widthWinI*Fsim/f):-analyzeL], signal[-analyzeL-int(widthWinI*Fsim/f):-analyzeL], 'b-')
			plt.plot(t[-int(widthWin1E*Fsim/f):], signal[-int(widthWin1E*Fsim/f):], 'r-')
			plt.plot(t[-int(widthWin3E*Fsim/f):], signal[-int(widthWin3E*Fsim/f):], 'g--')
			plt.plot(t[-int(widthWin3E*Fsim/f):-int(widthWin2E*Fsim/f)], signal[-int(widthWin3E*Fsim/f):-int(widthWin2E*Fsim/f)], 'c-')
			plt.plot(t[indexTimeFirstRisingEdgeI], signal[indexTimeFirstRisingEdgeI], 'b*', t[indexTimeFirstRisingEdgeE], signal[indexTimeFirstRisingEdgeE], 'r*')
			plt.draw();
		startIndexFFT	= indexTimeFirstRisingEdgeI
		endIndexFFT		= indexTimeFirstRisingEdgeE
	elif indexesLowStateI[0][0] != 0 and indexesHigStateI[0][0] == 0:
		print('First edge in the inital time window will be falling!')
		firstFallinEdgeIndI 		= indexesLowStateI[0][0];
		indexTimeFirstFallinEdgeI	= siglen-analyzeL-int(widthWinI*Fsim/f)+firstFallinEdgeIndI
		print('signal[indexTimeFirstFallinEdgeI-7:indexTimeFirstFallinEdgeI+7]', signal[indexTimeFirstFallinEdgeI-7:indexTimeFirstFallinEdgeI+7])

		if indexesHigStateE[0][0] == 0:
			print('First edge in the final time window is falling!')
			firstFallinEdgeIndE 		= indexesLowStateE[0][0];
			indexTimeFirstFallinEdgeE	= siglen-int(widthWin1E*Fsim/f)+firstFallinEdgeIndE
			print('signal[indexTimeFirstFallinEdgeE-7:indexTimeFirstFallinEdgeE+7]', signal[indexTimeFirstFallinEdgeE-7:indexTimeFirstFallinEdgeE+7])
		else: # change window to find other edge, - shift window by -T/2 in time
	##### more elegant?
			# print('Try to find a falling edge in last window, !')
			# firstRisingEdgeIndE 		= indexesHigStateE[0][0];
			# indexTimeFirstRisinEdgeE	= siglen-analyzeL-int(widthWin1E*Fsim/f)+firstRisinEdgeIndE
			# indexesLowStateEshift		= np.where(signal[(indexTimeFirstRisinEdgeE-int(0.75*(Fsim/f))):]!=  1)
			# indexesHigStateEshift		= np.where(signal[(indexTimeFirstRisinEdgeE-int(0.75*(Fsim/f))):]!= -1)
			# if indexesHigStateE[0][0] == 0:
			# 	print('First edge in the by widthWin2E shifted final time window is falling!')
			# 	firstFallinEdgeIndE 		= indexesLowStateE[0][0];
			# 	indexTimeFirstFallinEdgeE	= (indexTimeFirstRisinEdgeE-int(0.75*(Fsim/f)))+firstFallinEdgeIndEshift
			# 	timeFirstFallinEdgeE		= t[indexTimeFirstFallinEdgeE]
			# 	print('timeFirstFallinEdgeE[indexTimeFirstFallinEdgeE]:', timeFirstFallinEdgeE)
			# else:
			# 	print('\n\n...debug!\n\n');
	##### old
			print('Try to find a falling edge in last window, when shifted to widthWin2E!')
			indexesLowStateE 		= np.where(signal[-int(widthWin2E*Fsim/f):]!=  1)
			indexesHigStateE 		= np.where(signal[-int(widthWin2E*Fsim/f):]!= -1)
			if indexesHigStateE[0][0] == 0:
				print('First edge in the by widthWin2E shifted final time window is falling!')
				firstFallinEdgeIndE 		= indexesLowStateE[0][0];
				indexTimeFirstFallinEdgeE	= siglen-int(widthWin2E*Fsim/f)+firstFallinEdgeIndE
				print('signal[indexTimeFirstFallinEdgeE-7:indexTimeFirstFallinEdgeE+7]', signal[indexTimeFirstFallinEdgeE-7:indexTimeFirstFallinEdgeE+7])
			else:
				print('Try to find a falling edge in last window, when shifted to widthWin3E!')
				indexesLowStateE 		= np.where(signal[-int(widthWin3E*Fsim/f):]!=  1)
				indexesHigStateE 		= np.where(signal[-int(widthWin3E*Fsim/f):]!= -1)
				if indexesHigStateE[0][0] == 0:
					print('First edge in the widthWin3E shifted final time window is falling!')
					firstFallinEdgeIndE 		= indexesLowStateE[0][0];
					indexTimeFirstFallinEdgeE	= siglen-int(widthWin3E*Fsim/f)+firstFallinEdgeIndE
					print('signal[indexTimeFirstFallinEdgeE-7:indexTimeFirstFallinEdgeE+7]', signal[indexTimeFirstFallinEdgeE-7:indexTimeFirstFallinEdgeE+7])
				else:
					print('\n\n...debug! Did not find a rising edge in the search window at the end of the time-series.\n\n');
		print('total # of analyzed nsteps divided by # of steps equivalent to the expected deterministic period, i.e., # of analyzed periods:', (indexTimeFirstFallinEdgeE - indexTimeFirstFallinEdgeI) / (Fsim/syncF) )
		if testplot:
			t = np.arange(0,Tsim+delay,1/Fsim)
			timeFirstFallinEdgeI		= t[indexTimeFirstFallinEdgeI]
			timeFirstFallinEdgeE		= t[indexTimeFirstFallinEdgeE]
			print('timeFirstFallinEdgeE[indexTimeFirstFallinEdgeE]:', timeFirstFallinEdgeE)
			print('timeFirstFallinEdgeI[indexTimeFirstFallinEdgeI]:', timeFirstFallinEdgeI)
			print('analyzed time divided by period of synced state (DO NOT expect an integer number for noisy simulation!)', (timeFirstFallinEdgeE - timeFirstFallinEdgeI) * syncF )
			plt.figure(2000+psd_id); plt.clf();
			plt.plot(t[-analyzeL-int(widthWinI*Fsim/f):-analyzeL], signal[-analyzeL-int(widthWinI*Fsim/f):-analyzeL], 'b-')
			plt.plot(t[-int(widthWin1E*Fsim/f):], signal[-int(widthWin1E*Fsim/f):], 'r-')
			plt.plot(t[-int(widthWin3E*Fsim/f):], signal[-int(widthWin3E*Fsim/f):], 'g--')
			plt.plot(t[-int(widthWin3E*Fsim/f):-int(widthWin2E*Fsim/f)], signal[-int(widthWin3E*Fsim/f):-int(widthWin2E*Fsim/f)], 'c-')
			plt.plot(t[indexTimeFirstFallinEdgeI], signal[indexTimeFirstFallinEdgeI], 'b*', t[indexTimeFirstFallinEdgeE], signal[indexTimeFirstFallinEdgeE], 'r*')
			plt.draw();
		startIndexFFT	= indexTimeFirstFallinEdgeI
		endIndexFFT		= indexTimeFirstFallinEdgeE

	return np.array([startIndexFFT, endIndexFFT])

# works for f = 1; Fsim = 1000; i.e., Fsim/f is integer

# from scipy.signal import square
# Tsim = 1000; Fsim = 125; f = 0.999; phiInit = 0.0; percentOfTsim = 0.75;
#
# t      = np.arange(0,Tsim,1/Fsim)
# signal = square(2*pi*f*t+phiInit, duty=0.5)
# siglen = len(signal);
# analyzeL0	= percentOfTsim*siglen;
# analyzeL1 	= int((Fsim/f)*int(percentOfTsim*siglen)/(Fsim/f));
# #print('np.abs(analyzeL0-analyzeL1)', np.abs(analyzeL0-analyzeL1))
# analyzeL  	= int(analyzeL1)
#print('analyze', analyzeL1/(Fsim/f),' periods')
#
# analyzeL1 = int((Fsim/f)*int(percentOfTsim*siglen)/(Fsim/f));
# print('np.abs(analyzeL0-analyzeL1)', np.abs(analyzeL0-analyzeL1))
# analyzeL  = int(analyzeL1)
# print('analyze', analyzeL1/(Fsim/f),' periods')
# # test whether this is an integer number of periods between those rising edges -- not necessary since we rising/falling/rising...
# timeshift = t[-1]-t[-analyzeL];
# figure(); plt.clf(); plot(t[-int(4*Fsim/f):], signal[-int(4*Fsim/f):], 'b-'); plot(t[-analyzeL:-analyzeL+int(4*Fsim/f)]+timeshift, signal[-analyzeL:-analyzeL+int(4*Fsim/f)], 'c--');
# errorPeriod =  np.abs(Fsim/f-int(Fsim/f)); print('Error perid:', errorPeriod, '\t error over analzeL:', errorPeriod*analyzeL/(Fsim/f))
