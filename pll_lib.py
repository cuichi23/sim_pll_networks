## @package PLL library
#  Documentation for this module.
#
#  authors: Alexandros Pollakis, Daniel Platz, Deborah Schmidt, Lucas Wetzel (wetztel.lucas[at]gmail.com)

#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import sys, gc
import inspect
import numpy as np
#cimport numpy as np
#cimport cython
import networkx as nx
from scipy.signal import sawtooth
from scipy.signal import square
from scipy.stats import cauchy
from scipy.integrate import solve_ivp

#import matplotlib
import matplotlib.pyplot as plt
import datetime
import time

import setup

''' Enable automatic carbage collector '''
gc.enable();

#%%cython --annotate -c=-O3 -c=-march=native

'''PLL library
authors: Alexandros Pollakis, Daniel Platz, Deborah Schmidt, Lucas Wetzel (lwetzel[at]pks.mpg.de)
'''

# # system/measurement
# class Algorithm:
# 	"""An algorithm class"""
# 	def __init__(self,idx_self,pll1x,pll2x,pll1y,pll2y,pll1z,pll2z,algo):		# takes two PLLs and computes something
# 		self.pll_listX	= [pll1x, pll2x]
# 		self.pll_listY	= [pll1y, pll2y]
# 		self.pll_listZ	= [pll1z, pll2z]
# 		self.algorithm	= algo
#
# 	def next(self,idx_time,phi,ext_field):
#
# 		phi				= [pll1.next_antenna(idx_time,phi,ext_field) for pll in self.pll_listX]
#
# 		monitor_ctrl1	= self.pll1.lf.read_ctrl()
# 		monitor_ctrl2	= self.pll2.lf.read_ctrl()
# 		countPLL1 		= self.pll1.read_periods(phi)							# reads the current count of periods of PLL1
# 		countPLL2 		= self.pll2.read_periods(phi)							# reads the current count of periods of PLL2
#
# 	return phiPLLpair, countPLL1, countPLL2,
#
# 	def init_counters(self,phi0):
# 		self.pll1.reset(phi0)													# resets the counters of the PLLs, storing their current phase as new reference
# 		self.pll2.reset(phi0)													# counter states are computed as np.floor((phi-phi0)/(2*np.pi))

# PLL
class PhaseLockedLoop:
	""" A phase-locked loop class, handles signal flow between the PLL components

		Args:

	    Returns:

	    Raises:
	"""
	def __init__(self,idx_self,delayer,pdc,lf,vco,antenna,counter):				# sets PLL properties as given when an object of this class is created (see line 253), where pll_list is created
		""" initialization of a PLL object

			Args:
				instance self.delayer 	delayer
				instance self.pdc 		phase detector and combiner
				instance self.lf 		loop filter
				instance self.vco 		voltage controlled oscillator
				instance self.antenna	antenna
				instance self.counter	counter
				integer self.idx_self	identity

		    Returns:
				PLL instance	PLL with defined cross-coupling time delays to its coupling neighbors in the network, intrinisc frequency, loop filter, coupling strength,
								and many additional options such as additional injection of signals, noise, heterogeneity, etc.

		    Raises:
		"""
		self.delayer 	= delayer
		self.pdc 		= pdc
		self.lf 		= lf
		self.vco 		= vco
		self.antenna	= antenna
		self.counter	= counter
		self.idx_self	= idx_self

	def next(self,idx_time,philength,phi):
		""" Function that evolves the PLL forward in time by one increment based on the external signals and internal dynamics.
			1) delayer obtains past states x_delayed of neighbors in the network coupled to this oscillator and current state x of the oscillator itself
			2) from these states the input phase relations are evaluated by the phase detector and combiner which yields the PD signal x_comb (averaged over all inputs)
			3) the PD signal x_comb is fed into the loop filter and yields the control signal x_ctrl
			4) the voltage controlled oscillator's next function evolves the phases dependent on the control signal x_ctrl

			Args:
				int self		identity
				int idx_time	current time within simulation
				int philength	length of container that stores the phases of the oscillators, needed for organizing the cyclic memory to handle the delay
				np.ndarray phi	holds the phases of all oscillators for at least the time [-tau_max, 0], denotes the memory necessary for the time delay

		    Returns:
				np.ndarray phi	updated with the next time increment

		    Raises:
		"""
		x, x_delayed  	= self.delayer.next(idx_time%philength,phi,idx_time)	# this gets the values of a signal x at time t, and time t-tau, from the delayer (x is the matrix phi here)
		x_comb 		  	= self.pdc.next(x,x_delayed,0,idx_time)					# the phase detector signal is computed, antenna input set to zero; COMBINER GETS ABSOLUTE TIME
		x_ctrl		 	= self.lf.next(x_comb)									# the filtering at the loop filter is applied to the phase detector signal
		phi 		  	= self.vco.next(x_ctrl)[0]								# the control signal is used to update the VCO and thereby evolving the phase
		return phi#, x_ctrl

	def clock_periods_count(self,idx_time,current_phi):							# count periods of oscillations
		""" Function that counts the periods of oscillations that have passed for the oscillator. This enables the derivation of a time,
			i.e., extends the oscillator to become a clock.

			Args:
				int self				identity
				int idx_time			current time within simulation
				np.ndarray current_phi	the current phase of the oscillator

		    Returns:
				int count 				the number of cycles counted

		    Raises:
		"""
		count			= self.counter.read_periods(current_phi)
		return count

	def clock_halfperiods_count(self,idx_time,current_phi):						# count every half a period of the oscillations
		""" Function that counts TWICE per period for oscillations of the oscillator, e.g., counting the falling and rising edge of digital signal.
			This enables the derivation of a time, i.e., extends the oscillator to become a clock.

			Args:
				int self				identity
				int idx_time			current time within simulation
				np.ndarray current_phi	the current phase of the oscillator

		    Returns:
				int count 				the number of cycles counted

		    Raises:
		"""
		count			= self.counter.read_halfperiods(current_phi)
		return count

	def clock_reset(self,current_phi):											# reset all clock counters
		""" Function that resets the clock count of the oscillator.

			Args:
				int self				identity
				np.ndarray current_phi	the current phase of the oscillator (as new reference for the zero count)

		    Returns:
				None

		    Raises:
		"""
		count			= self.counter.reset(current_phi)
		return None

	def next_antenna(self,idx_time,philength,phi,ext_field):
		""" Function that implements an additional input signal to the oscillator received via an antenna. NOT YET IMPLEMENTED.

			Args:

		    Returns:

		    Raises:
		"""
		x, x_delayed  	= self.delayer.next(idx_time%philength,phi)				# this gets the values of a signal x at time t, and time t-tau, from the delayer (x is the matrix phi here)
		antenna_rx		= self.antenna.listen(idx_time,ext_field)				# antenna listens to the external signal
		x_comb 		  	= self.pdc.next(x,x_delayed,antenna_rx,idx_time)		# the phase detector signal is computed
		x_ctrl		 	= self.lf.next(x_comb)									# the filtering at the loop filter is applied to the phase detector signal
		phi 		  	= self.vco.next(x_ctrl)[0]								# the control signal is used to update the VCO and thereby evolving the phase
		return phi

	def setup_hist_reverse(self):												# set the initial history of the PLLs, all evolving with frequency Omega of synched state, provided at start
		""" Function that sets the history of the oscillator using the voltage controlled oscillator. Given the required phase at the start of the simulation,
			the function calls the VCO's function to set the history/memory backwards in time until the memory is filled.

			Args:
				int self		identity

		    Returns:
				np.array phi	an array with the memory of the particular oscillator, to be written into the phi container that holds all oscillators' phases

		    Raises:
		"""
		phi 			= self.vco.set_initial_reverse()[0]						# [0] vector-entry zero of the two values saved to phi which is returned
		return phi

	def set_delta_pertubation(self,idx_time,phi,phiS,inst_Freq):
		""" Function that applies a delta-like perturbation to the phase of the oscillator at the start of the simulation.
			Corrects the internal state of the oscillator with respect to the perturbation, calculated the initial control signal.

			Args:
				int self				identity
				int idx_time			current time within simulation
				float phi				the current phase of the oscillator
				phiS					the perturbation to the current phase
				float inst_freq 		the instantaneous frequency of the oscillator

		    Returns:
				float phi				the current phase of the oscillator perturbed by phiS

		    Raises:
		"""
		x_ctrl 			= self.lf.set_initial_control_signal(phi,inst_Freq)		# the filtering at the loop filter is applied to the phase detector signal
		phi 			= self.vco.delta_perturbation(phi,phiS,x_ctrl)[0]
		return phi

	def next_magic(self,idx_time,philength,phi):								# this is the closed-loop case of the free running PLL
		""" Function that evolves the voltage controlled oscillator in a closed loop configuration when there is no external input, i.e., free-running closed loop PLL.
			1) delayer obtains the current state x of the oscillator itself (potentially delayed by a feedback delay)
			2) this state is fed in the phase detector and combiner for zero external signal which yields the PD signal x_comb
			3) the PD signal x_comb is fed into the loop filter and yields the control signal x_ctrl
			4) the voltage controlled oscillator's next function evolves the phases dependent on the control signal x_ctrl

			Args:
				int self		identity
				int idx_time	current time within simulation
				int philength	length of container that stores the phases of the oscillators, needed for organizing the cyclic memory to handle the delay
				np.ndarray phi	holds the phases of all oscillators for at least the time [-tau_max, 0], denotes the memory necessary for the time delay

		    Returns:
				np.ndarray phi	updated with the next time increment

		    Raises:
		"""
		x, x_delayed  	= self.delayer.next(idx_time%philength,phi,idx_time)	# this gets the values of a signal x at time t, and time t-tau, from the delayer (x is the matrix phi here)
		x_comb 		  	= self.pdc.next(x,0,0,idx_time)							# the phase detector signal is computed, antenna input set to zero; COMBINER GETS ABSOLUTE TIME
		x_ctrl		 	= self.lf.next(x_comb)									# the filtering at the loop filter is applied to the phase detector signal
		phi 			= self.vco.next(x_ctrl)[0]								# [0] vector-entry zero of the two values returned
		return phi

	def next_free(self):														# evolve phase of the VCO as if there was no loop, hence x_ctrl = 0.0, i.e., only VCO
		""" Function that evolves the voltage controlled oscillator when there is no external input, i.e., free-running open loop PLL.

			Args:
				int self				identity

		    Returns:
				float phi				the updated phase

		    Raises:
		"""
		phi 			= self.vco.next(0.0)[0]
		return phi

################################################################################

# LF: y = integral du x(t) * p(t-u)												# this can be expressed in terms of a differential equation with the help of the Laplace transform
class LowPass:
	""" A lowpass filter class

		Args:

	    Returns:

	    Raises:
	"""
	def __init__(self,idx_self,dictPLL,dictNet):								#,Fc,dt,K,F_Omeg,F,cLF=0,Trelax=0,y=0,y_old=0):

		self.dt      = dictPLL['dt']											# set time-step
		self.Fc 	 = self.set_from_value_or_list(idx_self, dictPLL['cutFc'], dictNet['Nx']*dictNet['Ny']) # set cut-off frequency [Hz]
		self.K_Hz	 = self.set_from_value_or_list(idx_self, dictPLL['coupK'], dictNet['Nx']*dictNet['Ny']) # set coupling strength [Hz]
		self.intrF   = self.set_from_value_or_list(idx_self, dictPLL['intrF'], dictNet['Nx']*dictNet['Ny']) # intrinsic frequency of VCO - here needed for x_k^C(0) [Hz]
		self.syncF   = dictPLL['syncF']											# provide freq. of synchronized state under investigation - here needed for x_k^C(0) [Hz]
		self.LForder = dictPLL['orderLF']										# provides the LF order

		self.idx		= idx_self
		self.Omega  	= 2.0*np.pi*self.syncF									# angular frequency of synchronized state
		self.K 	  		= 2.0*np.pi*self.K_Hz									# provide coupling strength - here needed for x_k^C(0)
		self.instantF  	= None													# instantaneous frequency [Hz]
		self.fric_coeff = self.set_from_value_or_list(idx_self, dictPLL['friction_coefficient'], dictNet['Nx']*dictNet['Ny']) # friction coefficient

		self.y 			= None													# denotes the control signal, output of the LF
		self.dydt		= None													# denotes the time derivative of the control signal, output of the LF

		if not self.Fc == None and self.LForder > 0:
			self.wc 	= 2.0*np.pi*self.Fc										# angular cut-off frequency of the loop filter for a=1, filter of first order
			self.beta 	= self.dt*self.wc
			if   self.LForder == 1:
				print('I am the loop filter of PLL%i: first order, a=%i. Friction coefficient set to %0.2f.'%(self.idx, self.LForder, self.fric_coeff))
				self.evolve = lambda xPD: (1.0-self.beta*self.fric_coeff)*self.y + self.beta*xPD
			elif self.LForder == 2:
				print('I am the loop filter of PLL%i: second order, a=%i. Friction coefficient set to %0.2f.'%(self.idx, self.LForder, self.fric_coeff))
				self.evolve = lambda xPD: self.solve_2nd_orderOrdDiffEq(xPD)
			elif self.LForder > 2:
				print('Loop filters of order higher two are NOT implemented. Aborting!'); sys.exit()
		elif self.Fc == None:
			print('No cut-off frequency defined (None), hence simulating without loop filter!')
			self.evolve = lambda xPD: xPD
		else:
			print('Problem in LF class!'); sys.exit()

		a 	   = self.LForder;
		self.b = 1.0 / ( 2.0*np.pi*self.Fc * a )								# https://www.electronics-tutorials.ws/filter/filter_2.html	QUESTION CHRIS: cut-off freq wc = 1 / RC or w(@-3dB) = wc sqrt( 2^(1/n) -1 )
		self.t = np.array([0, self.dt])

	#***************************************************************************

	def set_from_value_or_list(self,idx_self,set_vars,numberPLLs):
		if ( ( isinstance(set_vars, list) or isinstance(set_vars, np.ndarray) ) and len(set_vars) == numberPLLs ):
			if len(set_vars) > 1:												# additional test, in case a single float has been case into a list or array
				return set_vars[idx_self]										# set individual value
			else:
				return set_vars													# set value for all
		elif ( isinstance(set_vars, float) or isinstance(set_vars, int) or set_vars == None):
			return set_vars														# set value for all
		else:
			print('Error in LF constructor setting a variable!'); sys.exit()

	def controlSig(self, t, z, xPD):
		x = z[0]
		y = z[1]
		# print('Solving control signal with 2nd order LF. Initial conditions are:', self.dydt, ',\t', self.y*(1+2.0/self.b)); time.sleep(2)
		return [y, (1.0/self.b**2)*(xPD-x)-(2.0/self.b)*y]	# -self.y-(self.dydt+(2.0*self.y)/self.b)

	def solve_2nd_orderOrdDiffEq(self, xPD):
																				# optional: try to implement via odeint as shown here: https://www.epythonguru.com/2020/07/second-order-differential-equation.html
		func = lambda t, z: self.controlSig(t, z, xPD)

		sol = solve_ivp(func, [self.t[0], self.t[1]], [2*self.y/self.b, self.dydt], method='RK45', t_eval=self.t, dense_output=False, events=None, vectorized=False, rtol = 1e-5)
		#print('sol: ', sol)
		y 			= sol.y[0][1]												# control signal value at time t
		self.dydt   = sol.y[1][1]												# derivative of control signal at time t
		# print('self.y:', self.y, '\ty: ', y, '\tdydt:', self.dydt); time.sleep(1)
		return y

	def set_initial_control_signal(self,inst_Freq,prior_inst_Freq):				# set the control signal for the last time step of the history, in the case the history is a synched state

		#print('REWORK: setting of initial time-derivative of control signal in case of second order LFs.')
		self.instantF = inst_Freq												# calculate the instantaneous frequency for the last time step of the history
		#self.y = (self.F_Omeg - self.F) / (self.K)								# calculate the state of the LF at the last time step of the history, it is needed for the simulation of the network
		if self.K_Hz != 0:														# this if-call is fine, since it will only be evaluated once
			self.y 	  = (self.instantF - self.intrF) / (self.K_Hz)				# calculate the state of the LF at the last time step of the history, it is needed for the simulation of the network
			yNminus1  = (prior_inst_Freq - self.intrF) / (self.K_Hz)			# calculate the state of the LF at the last time step of the history, it is needed for the simulation of the network
			self.dydt = (self.y - yNminus1) / self.dt							# calculate the change of the state of the LF at the last time step of the history
			print('Set initial ctrl signal! self.instantF, self.intrF, self.K_Hz', self.instantF, ' ', self.intrF, ' ', self.K_Hz)
		else:
			self.y = 0.0
		#print('Set initial control signal of PLL %i to:' %self.idx, self.y)
		return self.y

	def next(self,xPD):										      				# this updates y=x_k^{C}(t), the control signal, using the input x=x_k^{PD}(t), the phase-detector signal

		#print('Current PD signal of PLL %i:' %self.idx, xPD); time.sleep(1)
		self.y = self.evolve(xPD)
		#print('Current control signal of PLL %i:' %self.idx, self.y)
		return self.y

	def monitor_ctrl(self):														# monitor ctrl signal
		return self.y

################################################################################

# VCO: d_phi / d_t = omega + K * x
class VoltageControlledOscillator:
	"""A voltage controlled oscillator class

		Args:

	    Returns:

	    Raises:
	"""
	def __init__(self,idx_self,dictPLL,dictNet):
		self.idx_self 	= idx_self												# assigns the index
		self.Omega 		= 2.0*np.pi*dictPLL['syncF']							# set angular frequency of synchronized state under investigation
		if dictPLL['fric_coeff_PRE_vs_PRR'] == 'PRR':
			self.omega 	= 2.0*np.pi*self.set_from_value_or_list(idx_self, dictPLL['intrF'], dictNet['Nx']*dictNet['Ny']) # set intrinsic frequency of the VCO
		elif dictPLL['fric_coeff_PRE_vs_PRR'] == 'PRE':
			self.omega 	= 2.0*np.pi*self.set_from_value_or_list(idx_self, dictPLL['intrF']/dictPLL['friction_coefficient'], dictNet['Nx']*dictNet['Ny']) # set intrinsic frequency of the VCO
		self.K 			= 2.0*np.pi*self.set_from_value_or_list(idx_self, dictPLL['coupK'], dictNet['Nx']*dictNet['Ny']) # set coupling strength
		self.c 			= self.set_from_value_or_list(idx_self, dictPLL['noiseVarVCO'], dictNet['Nx']*dictNet['Ny'])	 # noise strength -- provide the variance of the GWN process
		self.dt 		= dictPLL['dt']											# set time step with which the equations are evolved
		self.phi 		= None													# this is the internal representation of phi, NOT the container in simulateNetwork
		self.responVCO	= dictPLL['responseVCO']								# defines a nonlinar VCO response, either set to 'linear' or the nonlinear expression
		self.idx		= idx_self

		if 	 dictPLL['typeOfHist'] == 'syncState':								# set initial frequency according to the parameter in 1params.txt
			print('I am the VCO of PLL%i with intrinsic frequency f=%0.2f Hz and K=%0.2f Hz, initially in a synchronized state.'%(self.idx_self, self.omega/(2.0*np.pi), self.K/(2.0*np.pi)))
			self.init_freq = self.Omega
		elif dictPLL['typeOfHist'] == 'freeRunning':
			print('I am the VCO of PLL%i with intrinsic frequency f=%0.2f Hz and K=%0.2f Hz, initially in free running.'%(self.idx_self, self.omega/(2.0*np.pi), self.K/(2.0*np.pi)))
			self.init_freq = self.omega
		else:
			print('\nSet typeOfHist dict entry correctly!'); sys.exit()

		if self.c > 0:															# create noisy VCO output
			print('VCO output noise is enabled!')
			if self.responVCO == 'linear':										# this simulates a linear response of the VCO
				self.evolvePhi = lambda w, K, x_ctrl, c, dt: ( w + K * x_ctrl ) * dt + np.random.normal(loc=0.0, scale=np.sqrt( c * dt ))
			elif not self.responVCO == 'linear':								# this simulates a user defined nonlinear VCO response
				print('\nself.responVCO:',self.responVCO,'\n')
				self.evolvePhi = lambda w, K, x_ctrl, c, dt: self.responVCO(w, K, x_ctrl) * dt + np.random.normal(loc=0.0, scale=np.sqrt( c * dt ))
		elif self.c == 0:														# create non-noisy VCO output
			if self.responVCO == 'linear':
				self.evolvePhi = lambda w, K, x_ctrl, c, dt: ( w + K * x_ctrl ) * dt
			elif not self.responVCO == 'linear':
				self.evolvePhi = lambda w, K, x_ctrl, c, dt: self.responVCO(w, K, x_ctrl) * dt

		test = self.evolvePhi(self.omega, self.K, 0.01, self.c, self.dt)
		if not ( isinstance(test, float) or isinstance(test, int) ):
			print('Specified VCO response function unknown, check VCO initialization in pll_lib!'); sys.exit()

	#***************************************************************************

	def set_from_value_or_list(self,idx_self,set_vars,numberPLLs):
		if ( ( isinstance(set_vars, list) or isinstance(set_vars, np.ndarray) ) and len(set_vars) == numberPLLs ):
			if len(set_vars) > 1:												# additional test, in case a single float has been case into a list or array
				return set_vars[idx_self]										# set individual value
			else:
				return set_vars													# set value for all
		elif ( isinstance(set_vars, float) or isinstance(set_vars, int) ):
			return set_vars														# set value for all
		else:
			print('Error in VCO constructor setting a variable!'); sys.exit()

	def evolveCouplingStrength(self,new_value_or_list,dictNet):

		self.K	= 2.0*np.pi*self.set_from_value_or_list(self.idx_self, new_value_or_list/self.K, dictNet['Nx']*dictNet['Ny'])
		#print('Injection lock coupling strength for PLL%i changed, new value:'%self.idx_self, self.K2nd_k); #time.sleep(1)
		return None

	def next(self,x_ctrl):														# compute change of phase per time-step due to intrinsic frequency and noise (if non-zero variance)
		self.d_phi 	= self.evolvePhi(self.omega, self.K, x_ctrl, self.c, self.dt)
		self.phi 	= self.phi + self.d_phi
		return self.phi, self.d_phi

	def delta_perturbation(self, phi, phiPert, x_ctrl):							# sets a delta-like perturbation 0-dt, the last time-step of the history
		self.d_phi 	= phiPert + self.evolvePhi(self.init_freq, self.K, x_ctrl, self.c, self.dt)
		self.phi 	= self.phi + self.d_phi
		return self.phi, self.d_phi

	def add_perturbation(self, phiPert):										# adds additional perturbation to current state
		self.phi 	= self.phi + phiPert
		return self.phi

	def set_initial_forward(self):												# sets the phase history of the VCO with the frequency of the synchronized state under investigation
		self.d_phi 	= self.evolvePhi(self.init_freq, 0.0, 0.0, self.c, self.dt)
		self.phi 	= self.phi + self.d_phi
		return self.phi, self.d_phi

	def set_initial_reverse(self):												# sets the phase history of the VCO with the frequency of the synchronized state under investigation
		self.d_phi 	= self.evolvePhi(-self.init_freq, 0.0, 0.0, self.c, self.dt)
		#print('In reverse fct of PLL%i self.phi, self.d_phi:'%self.idx, self.phi, self.d_phi); time.sleep(0.5)
		self.phi 	= self.phi + self.d_phi
		return self.phi, self.d_phi

################################################################################

# y = 1 / n * sum h( x_delayed_neighbours - x_self )
# print('Phasedetector and Combiner: sawtooth')
class PhaseDetectorCombiner:													# this class creates PD objects, these are responsible to detect the phase differences and combine the results
	"""A phase detector and combiner class

		Args:

	    Returns:

	    Raises:
	"""
	def __init__(self,idx_self,dictPLL,dictNet):
		# print('Phasedetector and Combiner: sin(x)')
		self.idx_self 		= idx_self											# assigns the index
		self.omega 			= 2.0*np.pi*np.mean(dictPLL['intrF'])				# set intrinsic frequency in radHz
		self.K 				= 2.0*np.pi*self.set_from_value_or_list(idx_self, dictPLL['coupK'], dictNet['Nx']*dictNet['Ny']) # set coupling strength
		self.dt				= dictPLL['dt']										# set time step
		self.div 			= dictPLL['div']									# set the divider
		self.h 				= dictPLL['coup_fct_sig']							# set the type of PD coupling function
		self.hp				= dictPLL['derivative_coup_fct']					# derivative of coupling function h
		self.hf				= dictPLL['vco_out_sig']							# set the type of VCO output signal, needed for HF cases
		self.a 				= dictPLL['antenna_sig']							# set the type of wireless signal
		self.K2nd_k			= self.set_from_value_or_list(idx_self, dictPLL['coupStr_2ndHarm']/np.array(dictPLL['coupK']), dictNet['Nx']*dictNet['Ny']) # set coupling strength for injection of 2nd harmonic, divide by PLL coupling strength as this is later multiplied again
		self.actRx			= 0													# PLL dynamic independent of antenna input

		self.idx_neighbors 	= [n for n in dictPLL['G'].neighbors(self.idx_self)]# for networkx > v1.11
		print('I am the phase detector of PLL%i, the frequency division is %i:'%(self.idx_self, self.div))
		if isinstance(dictPLL['gPDin'], np.ndarray) or isinstance(dictPLL['gPDin'], list):
			tempG_kl 		= [dictPLL['gPDin'][self.idx_self,i] for i in self.idx_neighbors]# pick the entries
			self.G_kl		= np.array(tempG_kl)							# the gain of each individual input gain, together with heterogeneous coupling strength: K_kl
			print('PD has different gains for each input signal! Hence: G_kl are introduced. CHECK THESE CASES AGAIN! self.G_kl[%i,l]'%self.idx_self, self.G_kl); #time.sleep(1)
		elif ((isinstance(dictPLL['gPDin'], int) or isinstance(dictPLL['gPDin'], np.float)) and dictPLL['extra_coup_sig'] == 'injection2ndHarm'):
			self.G_kl = dictPLL['gPDin'] + np.zeros(dictNet['Nx']*dictNet['Ny']-1)
		else:
			self.G_kl = 1

		if dictPLL['includeCompHF'] == False:
			print('High frequency components assumed to be ideally damped!')
			# depending on the coupling function for the Kuramoto like model with ideally damped HF terms this implements an XOR (triangular) or mixer PD (cos/sin)
			if dictPLL['antenna'] == True and dictPLL['extra_coup_sig'] == None:
				print('Extra signal to coupling!')
				self.compute	= lambda x_ext, ant_in, x_feed, idx_time:  np.mean( self.G_kl * self.h( ( x_ext - x_feed ) / self.div ) + self.actRx * self.h( ant_in - x_feed / self.div ) )
			elif dictPLL['antenna'] == False and dictPLL['extra_coup_sig'] == 'injection2ndHarm':
				print('Setup PLL with injection locking signal! Initial self.K2nd_k=', self.K2nd_k, 'Hz');
				if self.omega == 0 and not dictPLL['syncF'] == 0:
					self.omega = 2*np.pi*dictPLL['syncF']

				#self.compute	= lambda x_ext, ant_in, x_feed, idx_time: np.mean( self.G_kl * self.h( ( x_ext - x_feed ) / self.div ) ) - self.K2nd_k * self.h( ( 2.0 * self.omega * idx_time * self.dt ) / self.div )
				#self.compute	= lambda x_ext, ant_in, x_feed, idx_time: np.sum( self.G_kl * self.h( ( x_ext - x_feed ) / self.div ) ) - self.K2nd_k * self.h( 2.0 * x_feed / self.div )
				#self.compute	= lambda x_ext, ant_in, x_feed, idx_time: np.mean( self.G_kl * self.h( ( x_ext - x_feed ) / self.div ) ) - self.K2nd_k * self.h( -( 2.0 * ( self.omega * idx_time * self.dt ) - x_feed ) / self.div )
				#self.compute	= lambda x_ext, ant_in, x_feed, idx_time: np.mean( self.G_kl * self.h( ( x_ext - x_feed ) / self.div ) ) - self.K2nd_k * self.h( ( 2.0 * ( self.omega * idx_time * self.dt - x_feed ) ) / self.div )
				self.compute	= lambda x_ext, ant_in, x_feed, idx_time: np.mean( self.G_kl * self.h( ( x_ext - x_feed ) / self.div ) ) - np.mean( self.K2nd_k * self.h( ( 2.0 * ( x_ext - x_feed ) ) / self.div ) )

				#self.compute	= self.coupling_function_InjectLocking();
				#self.compute	= lambda x_ext, ant_in, x_feed, idx_time:  np.array(self.G_kl)@self.h( ( x_ext - x_feed ) / self.div ) - self.K2nd_k * self.h( ( 2.0 * self.omega * idx_time  * self.dt ) / self.div )
				#self.compute	= lambda x_ext, ant_in, x_feed, idx_time:  np.mean( self.G_kl * self.h( ( x_ext - x_feed ) / self.div ) ) - np.mean( self.G_kl * self.K2nd_k * self.h( ( 2.0*x_ext - x_feed ) / self.div ) )
				#self.compute	= lambda x_ext, ant_in, x_feed, idx_time:  np.mean( self.G_kl * self.h( ( x_ext - x_feed ) / self.div ) ) - np.mean ( self.K2nd_k * self.h( ( 2 * np.append(x_ext, x_feed) ) / self.div ) )
				#self.compute	= lambda x_ext, ant_in, x_feed, idx_time:  np.mean( self.G_kl * self.h( ( 2 * x_ext - x_feed ) / self.div ) )
			else:
				print('Simulating coupling function h(.) of the phase-differences as specified in dictPLL. The individial feed-forward path gains are G_%il=%0.2f'%(self.idx_self, self.G_kl))
				self.compute	= lambda x_ext, ant_in, x_feed, idx_time:  np.mean( self.G_kl * self.h( ( x_ext - x_feed ) / self.div ) )

		elif dictPLL['includeCompHF'] == True:
			print('High frequency components actived!')

			# depending on the coupling function this implements an XOR (triangular) or mixer PD (cos/sin) including the HF terms
			if dictPLL['antenna'] == True and dictPLL['extra_coup_sig'] == None:
				if dictPLL['typeVCOsig'] == 'analogHF':							# this becomes the coupling function for analog VCO output signals
					self.compute= lambda x_ext, ant_in, x_feed, idx_time: np.mean( self.G_kl * self.hf( x_ext / self.div ) * self.hf( x_feed / self.div )
																					+ self.actRx * self.a( ant_in ) * self.hf( x_feed / self.div ) )

				elif dictPLL['typeVCOsig'] == 'digitalHF':						# this becomes the coupling function for digital VCO output signals
					self.compute= lambda x_ext, ant_in, x_feed, idx_time: np.mean( self.G_kl * ( self.hf( x_ext / self.div )*(1.0-self.hf( x_feed / self.div ))
																				+ (1.0-self.hf( x_ext / self.div ))*self.hf( x_feed / self.div ) )
																				+ self.h( ant_in )*(1.0-self.hf( x_feed / self.div ))
																				+ (1.0-self.hf( ant_in ))*self.hf( x_feed / self.div ))
			elif dictPLL['antenna'] == False and dictPLL['extra_coup_sig'] == 'injection2ndHarm':
				print('Setup PLL with injection locking signal!');
				self.compute	= lambda x_ext, ant_in, x_feed, idx_time: np.mean( self.G_kl * ( self.h( ( 2 * x_ext - x_feed ) / self.div )
																				+ self.h( ( 2 * self.omega * idx_time * self.dt ) / self.div ) ) )
			else:
				if dictPLL['typeVCOsig'] == 'analogHF':							# this becomes the coupling function for analog VCO output signals
					self.compute= lambda x_ext, ant_in, x_feed, idx_time: np.mean( self.G_kl * self.hf( x_ext / self.div ) * self.hf( x_feed / self.div ) )

				elif dictPLL['typeVCOsig'] == 'digitalHF':						# this becomes the coupling function for digital VCO output signals
					self.compute= lambda x_ext, ant_in, x_feed, idx_time: np.mean( self.G_kl * ( self.hf( x_ext / self.div )*(1.0-self.hf( x_feed / self.div ))
																				+ (1.0-self.hf( x_ext / self.div ))*self.hf( x_feed / self.div ) ) )
			print('High frequency components activated, using:', inspect.getsourcelines(self.compute)[0][0])
		else:
			print('Phase detector and combiner problem, dictPLL[*includeCompHF*] should either be True or False, check PhaseDetectorCombiner in pll_lib! ')

	#***************************************************************************

	def set_from_value_or_list(self,idx_self,set_vars,numberPLLs):
		if ( ( isinstance(set_vars, list) or isinstance(set_vars, np.ndarray) ) and len(set_vars) == numberPLLs ):
			if len(set_vars) > 1:												# additional test, in case a single float has been case into a list or array
				return set_vars[idx_self]										# set individual value
			else:
				return set_vars													# set value for all
		elif ( isinstance(set_vars, float) or isinstance(set_vars, int) ):
			return set_vars														# set value for all
		else:
			print('Error in PDC constructor setting a variable!'); sys.exit()

	# def coupling_function_InjectLocking(self):
	# 	xctrl = lambda x_ext, ant_in, x_feed, idx_time:   np.array(self.G_kl)@self.h( ( x_ext - x_feed ) / self.div ) - self.K2nd_k * self.h( ( 2.0 * self.omega * idx_time ) / self.div )
	# 	#self.compute	= lambda x_ext, ant_in, x_feed, idx_time:   np.mean( self.G_kl * self.h( ( x_ext - x_feed ) / self.div ) ) - self.K2nd_k * self.h( ( 2 * self.omega * idx_time ) / self.div )
	# 	return xctrl

	def evolveCouplingStrengthInjectLock(self,new_value_or_list,dictNet):

		self.K2nd_k	= self.set_from_value_or_list(self.idx_self, new_value_or_list/self.K, dictNet['Nx']*dictNet['Ny'])
		#print('Injection lock coupling strength for PLL%i changed, new value:'%self.idx_self, self.K2nd_k); #time.sleep(1)
		return None

	def next(self,x_feed,x_delayed,antenna_in,idx_time=0):						# gets time-series results at delayed time and current time to calculate phase differences
		#print('self.idx_neighbors:', self.idx_neighbors)
		#print('x_delayed:', x_delayed, '\tx_feed:', x_feed)
		#print('idx_time in pdc.next: ', idx_time)
		#print('Next function of PDC, np.shape(x_feed)=',np.shape(x_feed),'\tnp.shape(x_delayed)=',np.shape(x_delayed))
		try:
			#x_feed = x[self.idx_self]											# extract own state (phase) at time t and save to x_feed
			if self.idx_neighbors:												# check whether there are neighbors
				#x_neighbours = x_delayed[self.idx_neighbors]					# extract the states of all coupling neighbors at t-tau and save to x_neighbours
				#self.y = self.compute( x_neighbours, antenna_in, x_feed  )		--> replaced these by choosing in delayer already!
				self.y = self.compute( x_delayed, antenna_in, x_feed, idx_time )
			else:
				print('No neighbors found, setting self.y to zero! Check for digital PLLs or other types.')
				self.y = 0.0;
			return self.y														# return control signal
		except:
			print('\n\nCHECK phase detector next() function!\n\n')
			self.y = 0.0;

			return self.y

################################################################################

# delayer
class Delayer:
	"""A delayer class

		Args:

	    Returns:

	    Raises:
	"""
	def __init__(self,idx_self,dictPLL,dictNet,dictData): #delay,dt,feedback_delay,std_dist_delay)

		self.dt				= dictPLL['dt']
		self.idx_self		= idx_self
		self.phi_array_len  = None												# this is being set after all (random) delays have been drawn
		self.idx_neighbors = [n for n in dictPLL['G'].neighbors(self.idx_self)]# for networkx > v1.11
		print('\nI am the delayer of PLL%i, my neighbors have indexes:'%self.idx_self, self.idx_neighbors)
		self.temp_array 	= np.zeros(dictNet['Nx']*dictNet['Ny'])				# use to collect tau_kl for PLL k

		if ( ( isinstance(dictPLL['transmission_delay'], float) or isinstance(dictPLL['transmission_delay'], int) ) and not dictNet['special_case'] == 'timeDepTransmissionDelay'):
			self.transmit_delay 		= dictPLL['transmission_delay']
			self.transmit_delay_steps 	= int(np.round( self.transmit_delay / self.dt ))	# when initialized, the delay in time-steps is set to delay_steps
			if ( self.transmit_delay_steps == 0 and self.transmit_delay > 0 ):
				print('Transmission delay set nonzero but smaller than the time-step "dt", hence "self.transmit_delay_steps" < 1 !'); sys.exit()
			elif ( self.transmit_delay_steps == 0 and self.transmit_delay == 0 ):
				print('Transmission delay set to zero!')
			#self.get_delayed_states		= lambda;
			self.pick_delayed_phases = lambda phi, t, abs_t, tau: phi[(t-tau)%self.phi_array_len, self.idx_neighbors]

		elif ( dictNet['special_case'] == 'timeDepTransmissionDelay' ):

			print('Time dependent transmission delay set!')
			time_dep_delay = setup.setupTimeDependentParameter(dictNet, dictPLL, dictData, parameter='transmission_delay', afterTsimPercent=0.25, forAllPLLsDifferent=False)
			selfidx_or_ident = 0;												# this is the case if all transmission delays have the same time dependence
			if len(time_dep_delay[:,0]) == dictNet['Nx']*dictNet['Ny']:			# if there is a matrix, i.e., different time-dependencies for different delay, then use this
				print('Test');
				selfidx_or_ident = self.idx_self
			dictPLL.update({'transmission_delay': time_dep_delay})
			plt.figure(1234)
			plt.plot(np.arange(0,len(dictPLL['transmission_delay'][selfidx_or_ident,:]))*dictPLL['dt'], dictPLL['transmission_delay'][selfidx_or_ident,:])
			plt.xlabel('time'); plt.ylabel('delay value [s]'); plt.title('time-dependence of transmission delay over simulation time')
			plt.draw(); plt.show()
			self.transmit_delay 		= dictPLL['transmission_delay'][selfidx_or_ident,:]	# each object only knows its own sending delay time dependence
			self.transmit_delay_steps 	= [int(np.round(delay / self.dt)) for delay in self.transmit_delay] # when initialized, the delay in time-steps is set to delay_steps

			if ( self.transmit_delay_steps == 0 and self.transmit_delay > 0 ):
				print('Transmission delay set nonzero but smaller than the time-step "dt", hence "self.transmit_delay_steps" < 1 !'); sys.exit()
			#self.get_delayed_states		= lambda;
			self.pick_delayed_phases = lambda phi, t, abs_t, tau: phi[(t-tau[abs_t])%self.phi_array_len, self.idx_neighbors]

		# calculate tranmission delays steps, here pick for each Delayer individually but the same for each input l or even tau_kl
		elif ( isinstance(dictPLL['transmission_delay'], list) or isinstance(dictPLL['transmission_delay'], np.ndarray) and not dictNet['special_case'] == 'timeDepTransmissionDelay' ):
			if np.array(dictPLL['transmission_delay']).ndim == 1:				# tau_k case
				self.transmit_delay_steps= int(np.round(dictPLL['transmission_delay'][idx_self] / self.dt))
				self.pick_delayed_phases = lambda phi, t, abs_t, tau_k: phi[(t-tau_k)%self.phi_array_len, self.idx_neighbors]

			elif np.array(dictPLL['transmission_delay']).ndim == 2: 			# tau_kl case
				print('Delayer has different delays for each input signal! Hence: tau_kl are introduced.')
				tempTauSteps_kl			 = [int(np.round(dictPLL['transmission_delay'][self.idx_self,i]/ self.dt)) for i in self.idx_neighbors]# pick the entries and store in a list for each PLL
				self.transmit_delay_steps= np.array(tempTauSteps_kl)			# save as an array to object
				self.pick_delayed_phases = lambda phi, t, abs_t, tau_kl: [phi[(t-tau_kl[i])%self.phi_array_len, self.idx_neighbors[i]] for i in range(len(self.idx_neighbors))]
				#other possible option if needed -- return phi-slice of the dimension of the phi container and pick neighbors in PDC!
				#self.pick_delayed_phases = self.pick_delayed_phases_taukl

		if ( isinstance(dictPLL['feedback_delay'], float) or isinstance(dictPLL['feedback_delay'], int) ):
			self.feedback_delay 		= dictPLL['feedback_delay']
			self.feedback_delay_steps 	= int(np.round( self.feedback_delay / self.dt ))	# when initialized, the delay in time-steps is set to delay_steps
			if ( self.feedback_delay_steps == 0 and self.feedback_delay > 0 ):
				print('Feedback set to nonzero but is smaller than the time-step "dt", hence "self.feedback_delay_steps" < 1 !'); sys.exit()

	#***************************************************************************

	# def pick_delayed_phases_taukl(phi, t, tau_kl):
	#
	# 	for i in range(self.idx_neighbors):
	# 		self.temp_array[self.idx_neighbors[i]] = phi[(t-tau_kl[i])%self.phi_array_len, self.idx_neighbors[i]]
	#
	# 	return transmission_delayed_phases

	def evolveDelayInTime(self,new_value):

		self.transmit_delay 		= new_value
		self.transmit_delay_steps 	= int(np.round( self.transmit_delay / self.dt ))	# when initialized, the delay in time-steps is set to delay_steps

		return None

	def next(self,idx_time,x,abs_t):

		#print('in delayer next: self.transmit_delay_steps=', self.transmit_delay_steps)
		transmission_delayed_phases = self.pick_delayed_phases(x, idx_time, abs_t, self.transmit_delay_steps )
		feedback_delayed_phase		= x[(idx_time-self.feedback_delay_steps)%self.phi_array_len, self.idx_self]
		#print('PLL%i with its feedback_delayed_phase [in steps]:'%self.idx_self,feedback_delayed_phase,', receives transmission_delayed_phases [in steps]:',transmission_delayed_phases,' from self.idx_neighbors:', self.idx_neighbors)
		#time.sleep(1)

		return np.asarray(feedback_delayed_phase), np.asarray(transmission_delayed_phases) # x is is the time-series from which the values at t-tau_kl^f and t-tau_kl are returned

################################################################################

# antenna
class Antenna:
	"""A antenna class

		Args:

	    Returns:

	    Raises:
	"""
	def __init__(self,idx_self,dictPLL,dictNet):

		self.dt			 = dictPLL['dt']
		self.posX 		 = dictPLL['posX']
		self.posY 		 = dictPLL['posY']
		self.posZ 		 = dictPLL['posZ']

		self.signalRx	 = dictPLL['antenna_sig']
		self.networkF	 = dictNet['freq_beacons']

		self.stateListen = dictPLL['initAntennaState']
		self.signalHeard = []

	#***************************************************************************

	def listen(self,idx_time,ext_field):

		antenna_in = self.signalRx(self.networkF*idx_time*self.dt)
		#antenna_in = 0
		return antenna_in

	def toggle_activation():

		if self.stateListen:
			self.stateListen = False
		elif not self.stateListen:
			self.stateListen = True
		return None

################################################################################

# counter
class Counter:
	"""A counter class

		Args:

	    Returns:

	    Raises:
	"""
	def __init__(self,idx_self,dictPLL):

		self.phase_init = 0

	#***************************************************************************

	def reset(self,phi0):
		self.phase_init = phi0
		return None

	def read_periods(self,phi):
		return np.floor((phi-self.phase_init)/(2.0*np.pi))

	def read_halfperiods(self,phi):
		return np.floor((phi-self.phase_init)/(np.pi))
