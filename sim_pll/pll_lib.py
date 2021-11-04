## @package PLL library
#  Documentation for this module.
#
#  authors: Alexandros Pollakis, Daniel Platz, Deborah Schmidt, Lucas Wetzel (wetztel.lucas[at]gmail.com)

#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import sys, gc
import inspect
from typing import Tuple, Optional

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

from sim_pll import setup

''' Enable automatic carbage collector '''
gc.enable()

#%%cython --annotate -c=-O3 -c=-march=native

'''PLL library
authors: Alexandros Pollakis, Daniel Platz, Deborah Schmidt, Lucas Wetzel (lwetzel[at]pks.mpg.de)
'''


def get_from_value_or_list(pll_id, input, pll_count):
	"""
	Convenience method return a value or value from list based on oscillator id, depending on the type
	of the input (scalar or list / array)

	Args:
		pll_id: the oscillator's identity
		input: the input data (scalar or list / array)
		pll_count: number of oscillators in the network

	Returns:
		the value or value from list, based on the oscillator id

	"""
	if ((isinstance(input, list) or isinstance(input, np.ndarray)) and len(input) == pll_count):
		if len(input) > 1:  # additional test, in case a single float has been case into a list or array
			return input[pll_id]  # set individual value
		else:
			return input  # set value for all
	elif (isinstance(input, float) or isinstance(input, int)):
		return input  # set value for all
	else:
		print('Error in constructor setting a variable!')
		sys.exit()


class LowPassFilter:
	""" Implements a first or second order RC low pass filter.
		Mathematically that can be represented by an integral over the impulse response of the filter and
		the signals. For convenience we reexpress this as a first or second order differential equation
		using Laplace transformation.

		Attributes:
			dt: time increment
			cutoff_freq_Hz: cutoff frequency
			K_Hz: coupling strength in Hz
			intr_freq_Hz: intrinsic frequency of SCO in Hz, needed to calculate the initial control signal at t=0
			sync_freq_Hz: frequency of synchronized state under investigation in Hz, needed to calculate the initial control signal at t=0
			order_loop_filter: specifies the order of the loop filter, here first or second order - 1 RC stage or 2 sequential RC stages
			pll_id: the oscillator's identity
			sync_freq_rad: frequency of synchronized states in radHz (Omega)
			K_rad: coupling strength in radHz
			instantaneous_freq_Hz: instantaneous frequency in Hz
			friction_coefficient: damping of friction coefficient of the dynamical model
			control_signal: denotes the control signal, output of the loop filter stage
			derivative_control_signal: denotes the time derivative of the control signal, output of the loop filter stage
			cutoff_freq_rad: angular cut-off frequency of the loop filter
			beta: helper variable
			evolve: defines the differential equation that evolves the loop filter in time
			b: https://www.electronics-tutorials.ws/filter/filter_2.html	TODO QUESTION CHRIS: cut-off freq wc = 1 / RC or w(@-3dB) = wc sqrt( 2^(1/n) -1 )
			t: array that defines the time span for the solve_ivp function that solves the second order differential equation numerically
	"""
	def __init__(self, pll_id: int, dict_pll: dict, dict_net: dict):
		"""
		Args:
			pll_id: the oscillator's identity
			dict_pll: oscillator related properties and parameters
			dict_net: network related properties and parameters
		"""

		self.dt = dict_pll['dt']
		self.cutoff_freq_Hz 	 = get_from_value_or_list(pll_id, dict_pll['cutFc'], dict_net['Nx'] * dict_net['Ny'])
		self.K_Hz	 = get_from_value_or_list(pll_id, dict_pll['coupK'], dict_net['Nx'] * dict_net['Ny'])
		self.intr_freq_Hz   = get_from_value_or_list(pll_id, dict_pll['intrF'], dict_net['Nx'] * dict_net['Ny'])
		self.sync_freq_Hz   = dict_pll['syncF']
		self.order_loop_filter = dict_pll['orderLF']

		self.pll_id		= pll_id
		self.sync_freq_rad  	= 2.0 * np.pi * self.sync_freq_Hz
		self.K_rad 	  		= 2.0 * np.pi * self.K_Hz
		self.instantaneous_freq_Hz  	= None
		self.friction_coefficient = get_from_value_or_list(pll_id, dict_pll['friction_coefficient'], dict_net['Nx'] * dict_net['Ny'])

		self.control_signal 			= None
		self.derivative_control_signal		= None

		if not self.cutoff_freq_Hz == None and self.order_loop_filter > 0:
			self.cutoff_freq_rad 	= 2.0 * np.pi * self.cutoff_freq_Hz
			self.beta 	= self.dt*self.cutoff_freq_rad
			if   self.order_loop_filter == 1:
				print('I am the loop filter of PLL%i: first order, a=%i. Friction coefficient set to %0.2f.' % (self.pll_id, self.order_loop_filter, self.friction_coefficient))
				self.evolve = lambda xPD: (1.0 - self.beta * self.friction_coefficient) * self.control_signal + self.beta * xPD
			elif self.order_loop_filter == 2:
				print('I am the loop filter of PLL%i: second order, a=%i. Friction coefficient set to %0.2f.' % (self.pll_id, self.order_loop_filter, self.friction_coefficient))
				self.evolve = lambda xPD: self.solve_2nd_order_ordinary_diff_eq(xPD)
			elif self.order_loop_filter > 2:
				print('Loop filters of order higher two are NOT implemented. Aborting!'); sys.exit()
		elif self.cutoff_freq_Hz == None:
			print('No cut-off frequency defined (None), hence simulating without loop filter!')
			self.evolve = lambda xPD: xPD
		else:
			print('Problem in LF class!'); sys.exit()

		a = self.order_loop_filter
		self.b = 1.0 / (2.0 * np.pi * self.cutoff_freq_Hz * a)
		self.t = np.array([0, self.dt])

	def second_order_ordinary_diff_eq(self, t, z, phase_detector_output):
		""" Defines the second order ordinary differential equation of the second order loop filter as a set of two
		first order coupled differential equations.
		"""
		x = z[0]
		y = z[1]
		# print('Solving control signal with 2nd order LF. Initial conditions are:', self.dydt, ',\t', self.y*(1+2.0/self.b)); time.sleep(2)
		return [y, (1.0/self.b**2) * (phase_detector_output - x) - (2.0 / self.b) * y]	# -self.y-(self.dydt+(2.0*self.y)/self.b)

	def solve_2nd_order_ordinary_diff_eq(self, phase_detector_output):
		""" Evolves the output of the second order loop filter by one time increment.

		Args:
			phase_detector_output: the input signal to the signal loop filter

		Returns:
			control signal
		"""
		# TODO define function with known solution and test
		# optional: try to implement via odeint as shown here: https://www.epythonguru.com/2020/07/second-order-differential-equation.html
		func = lambda t, z: self.second_order_ordinary_diff_eq(t, z, phase_detector_output)
		sol = solve_ivp(func, [self.t[0], self.t[1]], [2 * self.control_signal / self.b, self.derivative_control_signal], method='RK45', t_eval=self.t, dense_output=False, events=None, vectorized=False, rtol=1e-5)
		#print('sol: ', sol)
		control_signal 			= sol.y[0][1]
		self.derivative_control_signal   = sol.y[1][1]
		# print('self.control_signal:', self.control_signal, '\tcontrol_signal: ', control_signal, '\tderivative_control_signal:', self.derivative_control_signal); time.sleep(1)
		return control_signal

	def set_initial_control_signal(self, instantaneous_freq_Hz: float, prior_instantaneous_freq_Hz: float) -> np.float:
		""" Sets the initial control signal for the last time step depending on the history.

		 Args:
		 	instantaneous_freq_Hz: instantaneous frequency in Hz at the end of the history
		 	prior_instantaneous_freq_Hz: instantaneous frequency in Hz at the time step before the end of the history

		Returns:
			initial control signal
		"""

		#print('REWORK: setting of initial time-derivative of control signal in case of second order LFs.')
		# TODO clean up
		self.instantaneous_freq_Hz = instantaneous_freq_Hz												# calculate the instantaneous frequency for the last time step of the history
		#self.y = (self.sync_freq_Hz - self.intr_freq_Hz) / (self.K_Hz)								# calculate the state of the LF at the last time step of the history, it is needed for the simulation of the network
		if self.K_Hz != 0:														# this if-call is fine, since it will only be evaluated once
			self.control_signal 	  = (self.instantaneous_freq_Hz - self.intr_freq_Hz) / (self.K_Hz)				# calculate the state of the LF at the last time step of the history, it is needed for the simulation of the network
			yNminus1  = (prior_instantaneous_freq_Hz - self.intr_freq_Hz) / (self.K_Hz)			# calculate the state of the LF at the last time step of the history, it is needed for the simulation of the network
			self.derivative_control_signal = (self.control_signal - yNminus1) / self.dt							# calculate the change of the state of the LF at the last time step of the history
			print('Set initial ctrl signal! self.instantF, self.intrF, self.K_Hz', self.instantaneous_freq_Hz, ' ', self.intr_freq_Hz, ' ', self.K_Hz)
		else:
			self.control_signal = 0.0
		#print('Set initial control signal of PLL %i to:' %self.pll_id, self.control_signal)
		return self.control_signal

	def next(self, phase_detector_output: np.float) -> np.float:
		""" This function evolves the control signal in time according to the input signal from the phase detector and
		 combiner.

		Args:
			phase_detector_output: the input signal to the signal loop filter

		Returns:
			control signal
		"""
		#print('Current PD signal of PLL %i:' %self.pll_id, phase_detector_output); time.sleep(1)
		self.control_signal = self.evolve(phase_detector_output)
		#print('Current control signal of PLL %i:' %self.pll_id, self.control_signal)
		return self.control_signal

	def get_control_signal(self) -> np.float:
		"""
		Returns:
			control signal
		"""
		return self.control_signal


class SignalControlledOscillator:
	"""A signal controlled oscillator is an autonomous oscillator that can change its instantaneous frequency as a
	function of the control signal.

	SCO: d_phi / d_t = f(x_ctrl) = omega + K * x_ctrl + O(epsilon > 1)

	Attributes:
		pll_id: the oscillator's identity
		sync_freq_rad: frequency of synchronized states in radHz (Omega)
		intr_freq_rad: intrinsic frequency of free running closed loop oscillator in radHz (omega)
		K_rad: coupling strength in radHz
		c: noise strength -- provides the variance of the GWN process
		dt: time increment
		phi: this is the internal representation of the oscillator's phase, NOT the container in simulateNetwork
		response_vco: defines a nonlinear VCO response, either set to 'linear' or the nonlinear expression
		init_freq: defines the initial frequency of the signal controlled oscillator according to the phase history
		evolve_phi: function defining how the phase evolves in time, e.g., with or without noise, linear vs. nonlinear
		d_phi: stores the phase increment between the current and prior simulation step
	"""
	def __init__(self, pll_id, dict_pll, dict_net):
		"""
		Args:
			pll_id: the oscillator's identity
			dict_pll: oscillator related properties and parameters
			dict_net: network related properties and parameters
		"""
		self.d_phi = None
		self.pll_id = pll_id
		self.sync_freq_rad 	= 2.0 * np.pi * get_from_value_or_list(pll_id, dict_pll['syncF'], dict_net['Nx'] * dict_net['Ny'])     #dict_pll['syncF']
		if dict_pll['fric_coeff_PRE_vs_PRR'] == 'PRR':
			self.intr_freq_rad 	= 2.0 * np.pi * get_from_value_or_list(pll_id, dict_pll['intrF'], dict_net['Nx'] * dict_net['Ny'])
		elif dict_pll['fric_coeff_PRE_vs_PRR'] == 'PRE':
			if isinstance(dict_pll['intrF'], list):
				intrinsic_freqs_temp = np.array(dict_pll['intrF'])
			else:
				intrinsic_freqs_temp = dict_pll['intrF']
			self.intr_freq_rad 	= 2.0 * np.pi * get_from_value_or_list(pll_id, intrinsic_freqs_temp / dict_pll['friction_coefficient'], dict_net['Nx'] * dict_net['Ny'])
		self.K_rad 		= 2.0 * np.pi * get_from_value_or_list(pll_id, dict_pll['coupK'], dict_net['Nx'] * dict_net['Ny'])
		self.c 			= get_from_value_or_list(pll_id, dict_pll['noiseVarVCO'], dict_net['Nx'] * dict_net['Ny'])
		self.dt 		= dict_pll['dt']
		self.phi: Optional[float] = None
		self.response_vco = dict_pll['responseVCO']

		if dict_pll['typeOfHist'] == 'syncState':
			print('I am the VCO of PLL%i with intrinsic frequency f=%0.2f Hz and K=%0.2f Hz, initially in a synchronized state.' % (self.pll_id, self.intr_freq_rad / (2.0 * np.pi), self.K_rad / (2.0 * np.pi)))
			self.init_freq = self.sync_freq_rad
		elif dict_pll['typeOfHist'] == 'freeRunning':
			print('I am the VCO of PLL%i with intrinsic frequency f=%0.2f Hz and K=%0.2f Hz, initially in free running.' % (self.pll_id, self.intr_freq_rad / (2.0 * np.pi), self.K_rad / (2.0 * np.pi)))
			self.init_freq = self.intr_freq_rad
		else:
			print('\nSet typeOfHist dict entry correctly!'); sys.exit()

		if self.c > 0:															# create noisy VCO output
			print('VCO output noise is enabled!')
			if self.response_vco == 'linear':										# this simulates a linear response of the VCO
				self.evolve_phi = lambda w, K, x_ctrl, c, dt: (w + K * x_ctrl) * dt + np.random.normal(loc=0.0, scale=np.sqrt(c * dt))
			elif not self.response_vco == 'linear':								# this simulates a user defined nonlinear VCO response
				print('\nself.responVCO:', self.response_vco, '\n')
				self.evolve_phi = lambda w, K, x_ctrl, c, dt: self.response_vco(w, K, x_ctrl) * dt + np.random.normal(loc=0.0, scale=np.sqrt(c * dt))
		elif self.c == 0:														# create non-noisy VCO output
			if self.response_vco == 'linear':
				self.evolve_phi = lambda w, K, x_ctrl, c, dt: (w + K * x_ctrl) * dt
			elif not self.response_vco == 'linear':
				self.evolve_phi = lambda w, K, x_ctrl, c, dt: self.response_vco(w, K, x_ctrl) * dt

		test = self.evolve_phi(self.intr_freq_rad, self.K_rad, 0.01, self.c, self.dt)
		if not ( isinstance(test, float) or isinstance(test, int) ):
			print('Specified VCO response function unknown, check VCO initialization in pll_lib!'); sys.exit()

	def evolve_coupling_strength(self, new_coupling_strength_value_or_list, dict_net: dict) -> None:
		"""
		Evolves the values of the cross coupling strength in time -- more precisely the VCO sensitivity.

		Args:
			new_coupling_strength_value_or_list: the new value for the cross coupling strength,
			either a scalar or an array with individual values for each signal controlled oscillator in the network
			dict_net: network related properties and parameters
		"""

		self.K_rad	= 2.0 * np.pi * get_from_value_or_list(self.pll_id, new_coupling_strength_value_or_list / self.K_rad, dict_net['Nx'] * dict_net['Ny'])
		#print('Injection lock coupling strength for PLL%i changed, new value:'%self.idx_self, self.K2nd_k); #time.sleep(1)
		return None

	def next(self, control_signal) -> Tuple[np.float, np.float]:
		"""
		Evolves the instantaneous output frequency of the signal controlled oscillator according to the control signal
		and the dynamic noise.

		Args:
			control_signal: the result of the phase detection and signal processing to control the output frequency of
			the signal controlled oscillator

		Returns:
			tuple of the next phase and phase increment
		"""
		self.d_phi 	= self.evolve_phi(self.intr_freq_rad, self.K_rad, control_signal, self.c, self.dt)
		self.phi 	= self.phi + self.d_phi
		return self.phi, self.d_phi

	def delta_perturbation(self, phase_perturbation: np.float, control_signal: np.float) -> Tuple[np.float, np.float]:
		"""
		Sets a delta-like perturbation.

		Args:
			phase_perturbation: a delta like perturbation to the current phase
			control_signal: the input signal to the signal controlled oscillator
		Returns:
			a tuple of the perturbed phase and the phase increment
		"""
		self.d_phi 	= phase_perturbation + self.evolve_phi(self.init_freq, self.K_rad, control_signal, self.c, self.dt)
		self.phi 	= self.phi + self.d_phi
		return self.phi, self.d_phi

	def add_perturbation(self, phase_perturbation: np.float) -> np.float:
		"""
		Adds user defined perturbation to current state.

		Args:
			phase_perturbation: a delta like perturbation to the current phase
		Returns:
			the perturbed phase
		"""
		self.phi 	= self.phi + phase_perturbation
		return self.phi

	def set_initial_forward(self) -> Tuple[np.float, np.float]:
		"""
		Sets the phase history of the signal controlled oscillator based on the frequency for the initial frequency.
		Starts at time t - tau_max (the maximum time delay) and evolves the history until the time at which the
		simulation starts.

		Returns:
			a tuple of the perturbed phase and the phase increment
		"""
		self.d_phi 	= self.evolve_phi(self.init_freq, 0.0, 0.0, self.c, self.dt)
		self.phi 	= self.phi + self.d_phi
		return self.phi, self.d_phi

	def set_initial_reverse(self) -> Tuple[np.float, np.float]:
		"""
		Sets the phase history of the signal controlled oscillator based on the frequency for the initial frequency.
		Starts at the time at which the simulation starts and evolves the history until time t - tau_max
		(the maximum time delay).

		Returns:
			a tuple of the perturbed phase and the phase increment
		"""
		self.d_phi 	= self.evolve_phi(-self.init_freq, 0.0, 0.0, self.c, self.dt)
		#print('In reverse fct of PLL%i self.phi, self.d_phi:'%self.idx, self.phi, self.d_phi); time.sleep(0.5)
		self.phi 	= self.phi + self.d_phi
		return self.phi, self.d_phi


class PhaseDetectorCombiner:
	"""A phase detector and combiner class. It creates PD objects, these are responsible to detect the phase differences
	and combine the results if there is more than one input. Available phase detectors are:
	- Multiplier phase detector for analog signals
	- XOR phase detectors for digital signals
	- phase frequency detectors (PFD)

	Note that the high frequency components are usually assumed to be damped ideally. There is an option however to
	include the high frequency components of the phase detector signal.

	The result of the phase detector and combiner has the following structure:
	y_k = 1 / n_k * sum_l * c_kl * coup_fct( delayed_neighbour_phase_l - feedback_delayed_phase_k )

	Attributes:
		pll_id: the oscillator's identity
		intr_freq_rad: intrinsic frequency in radHz
		K_rad: coupling strength in radHz
		dt: time increment
		div: the division value of the frequency divider
		h: the coupling function of the phase detector (HF components ideally damped)
		hp: derivative of coupling function h
		hf: the function of the VCO output signal, needed for HF cases (HF components included)
		a: the wave form of the wireless signal
		K2nd_k: coupling strength for injection of 2nd harmonic, divide by PLL coupling strength as this is later multiplied again
		activate_Rx:  turns antenna input off or on (0 or 1)
		idx_neighbors: the identifiers of all neighbors from whom a signal is received
		G_kl: component of feed forward path gain matrix: defines the gain of the feed forward path of oscillator k with respect to input l
		compute: function that defines the dynamics based on the relation of input and feedback signal
		y: phase detector output

	"""
	def __init__(self, pll_id: int, dict_pll: dict, dict_net: dict):
		"""
		Args:
			pll_id: the oscillator's identity
			dict_pll: oscillator related properties and parameters
			dict_net: network related properties and parameters
		"""
		# print('Phasedetector and Combiner: sin(x)')
		self.pll_id 		= pll_id
		self.intr_freq_rad 			= 2.0 * np.pi * np.mean(dict_pll['intrF'])
		self.K_rad 				= 2.0 * np.pi * get_from_value_or_list(pll_id, dict_pll['coupK'], dict_net['Nx'] * dict_net['Ny'])
		self.dt				= dict_pll['dt']
		self.div 			= dict_pll['div']
		self.h 				= dict_pll['coup_fct_sig']
		self.hp				= dict_pll['derivative_coup_fct']
		self.hf				= dict_pll['vco_out_sig']
		self.a 				= dict_pll['antenna_sig']
		self.K2nd_k			= get_from_value_or_list(pll_id, dict_pll['coupStr_2ndHarm'] / np.array(dict_pll['coupK']), dict_net['Nx'] * dict_net['Ny'])
		self.activate_Rx	= 0
		self.y = None

		self.idx_neighbors 	= [n for n in dict_pll['G'].neighbors(self.pll_id)]# for networkx > v1.11
		print('I am the phase detector of PLL%i, the frequency division is %i:' % (self.pll_id, self.div))
		if isinstance(dict_pll['gPDin'], np.ndarray) or isinstance(dict_pll['gPDin'], list):
			tempG_kl 		= [dict_pll['gPDin'][self.pll_id, i] for i in self.idx_neighbors]
			self.G_kl		= np.array(tempG_kl)
			print('PD has different gains for each input signal! Hence: G_kl are introduced. CHECK THESE CASES AGAIN! self.G_kl[%i,l]' % self.pll_id, self.G_kl)
			#time.sleep(1)
		elif ((isinstance(dict_pll['gPDin'], int) or isinstance(dict_pll['gPDin'], np.float)) and dict_pll['extra_coup_sig'] == 'injection2ndHarm'):
			self.G_kl = dict_pll['gPDin'] + np.zeros(dict_net['Nx'] * dict_net['Ny'] - 1)
		else:
			self.G_kl = 1

		if dict_pll['includeCompHF'] == False:
			print('High frequency components assumed to be ideally damped!')
			# depending on the coupling function for the Kuramoto like model with ideally damped HF terms this implements an XOR (triangular) or mixer PD (cos/sin)
			if dict_pll['antenna'] == True and dict_pll['extra_coup_sig'] == None:
				print('Extra signal to coupling!')
				self.compute	= lambda x_ext, ant_in, x_feed, idx_time:  np.mean(self.G_kl * self.h( ( x_ext - x_feed ) / self.div ) + self.activate_Rx * self.h(ant_in - x_feed / self.div))
			elif dict_pll['antenna'] == False and dict_pll['extra_coup_sig'] == 'injection2ndHarm':
				print('Setup PLL with injection locking signal! Initial self.K2nd_k=', self.K2nd_k, 'Hz')
				if self.intr_freq_rad == 0 and not dict_pll['syncF'] == 0:
					self.intr_freq_rad = 2 * np.pi * dict_pll['syncF']

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
				print('Simulating coupling function h(.) of the phase-differences as specified in dictPLL. The individial feed-forward path gains are G_%il=%0.2f' % (self.pll_id, self.G_kl))
				self.compute	= lambda x_ext, ant_in, x_feed, idx_time:  np.mean( self.G_kl * self.h( ( x_ext - x_feed ) / self.div ) )

		elif dict_pll['includeCompHF'] == True:
			print('High frequency components actived!')

			# depending on the coupling function this implements an XOR (triangular) or mixer PD (cos/sin) including the HF terms
			if dict_pll['antenna'] == True and dict_pll['extra_coup_sig'] == None:
				if dict_pll['typeVCOsig'] == 'analogHF':							# this becomes the coupling function for analog VCO output signals
					self.compute= lambda x_ext, ant_in, x_feed, idx_time: np.mean(self.G_kl * self.hf( x_ext / self.div ) * self.hf( x_feed / self.div )
																				  + self.activate_Rx * self.a(ant_in) * self.hf(x_feed / self.div))

				elif dict_pll['typeVCOsig'] == 'digitalHF':						# this becomes the coupling function for digital VCO output signals
					self.compute= lambda x_ext, ant_in, x_feed, idx_time: np.mean( self.G_kl * ( self.hf( x_ext / self.div )*(1.0-self.hf( x_feed / self.div ))
																				+ (1.0-self.hf( x_ext / self.div ))*self.hf( x_feed / self.div ) )
																				+ self.h( ant_in )*(1.0-self.hf( x_feed / self.div ))
																				+ (1.0-self.hf( ant_in ))*self.hf( x_feed / self.div ))
			elif dict_pll['antenna'] == False and dict_pll['extra_coup_sig'] == 'injection2ndHarm':
				print('Setup PLL with injection locking signal!');
				self.compute	= lambda x_ext, ant_in, x_feed, idx_time: np.mean(self.G_kl * (self.h( ( 2 * x_ext - x_feed ) / self.div )
																							   + self.h((2 * self.intr_freq_rad * idx_time * self.dt) / self.div)))
			else:
				if dict_pll['typeVCOsig'] == 'analogHF':							# this becomes the coupling function for analog VCO output signals
					self.compute= lambda x_ext, ant_in, x_feed, idx_time: np.mean( self.G_kl * self.hf( x_ext / self.div ) * self.hf( x_feed / self.div ) )

				elif dict_pll['typeVCOsig'] == 'digitalHF':						# this becomes the coupling function for digital VCO output signals
					self.compute= lambda x_ext, ant_in, x_feed, idx_time: np.mean( self.G_kl * ( self.hf( x_ext / self.div )*(1.0-self.hf( x_feed / self.div ))
																				+ (1.0-self.hf( x_ext / self.div ))*self.hf( x_feed / self.div ) ) )
			print('High frequency components activated, using:', inspect.getsourcelines(self.compute)[0][0])
		else:
			print('Phase detector and combiner problem, dictPLL[*includeCompHF*] should either be True or False, check PhaseDetectorCombiner in pll_lib! ')

	def evolve_coupling_strength_inject_lock(self, new_coupling_strength_injection_locking, dict_net: dict) -> None:
		"""
		Evolves the values of the coupling strength of the second harmonic injection signals in time.

		Args:
			new_coupling_strength_injection_locking: the new value for the coupling strength of the injection signals,
			either a scalar or an array with individual values for each oscillator in the network
			dict_net: network related properties and parameters
		"""
		self.K2nd_k	= get_from_value_or_list(self.pll_id, new_coupling_strength_injection_locking / self.K_rad, dict_net['Nx'] * dict_net['Ny'])
		#print('Injection lock coupling strength for PLL%i changed, new value:'%self.idx_self, self.K2nd_k); #time.sleep(1)

	def next(self, feedback_delayed_phases: np.ndarray, transmission_delayed_phases: np.ndarray, antenna_in: float, index_current_time: int = 0) -> np.float:
		"""
		Evaluates delayed phase states of coupling partners and feedback delayed phase state of itself to yield phase
		detector output.

		Args:
			feedback_delayed_phases: the feedback delayed phases of all oscillators in the network at the current time
			transmission_delayed_phases: the transmission delayed phases of all oscillators in the network at the current time
			antenna_in: antenna input signal phase
			index_current_time: current time step

		Returns:
			phase detector output
		"""
		#print('self.idx_neighbors:', self.idx_neighbors)
		#print('x_delayed:', x_delayed, '\tx_feed:', x_feed)
		#print('idx_time in pdc.next: ', idx_time)
		#print('Next function of PDC, np.shape(x_feed)=',np.shape(x_feed),'\tnp.shape(x_delayed)=',np.shape(x_delayed))
		try:
			#x_feed = x[self.idx_self]											# extract own state (phase) at time t and save to x_feed
			if self.idx_neighbors:												# check whether there are neighbors
				#x_neighbours = x_delayed[self.idx_neighbors]					# extract the states of all coupling neighbors at t-tau and save to x_neighbours
				#self.y = self.compute( x_neighbours, antenna_in, x_feed  )		--> replaced these by choosing in delayer already!
				self.y = self.compute(transmission_delayed_phases, antenna_in, feedback_delayed_phases, index_current_time)
			else:
				print('No neighbors found, setting self.y to zero! Check for digital PLLs or other types.')
				self.y = 0.0
			return self.y
		except:
			print('\n\nCHECK phase detector next() function!\n\n')
			self.y = 0.0

			return self.y


class Delayer:
	"""The delayer obtains the past states of a PLLs coupling partners and its own past or current state.

		Attributes:
			pll_id: the identifier of the PLL this delayer belongs to
			dt: time increment
			phi_array_len: length of container that stores the phases
			neighbor_ids: list of the ids of neighbors from whom signals are being received
			transmit_delay: the transmission delay of the incoming signal
			transmit_delay_steps: transmission delay in multiples of the time increment
			pick_delayed_phases: function that defines how phases of incoming signals are picked
			feedback_delay: the feedback delay of the oscillators internal loop
			feedback_delay_steps: feedback delay in multiples of the time increment

	"""
	def __init__(self, pll_id: int, dict_pll: dict, dict_net: dict, dict_data: dict) -> None:
		"""
		Args:
			pll_id: the oscillator's identity
			dict_pll: oscillator related properties and parameters
			dict_net: network related properties and parameters
			dict_data: database for simulation results
		"""
		self.dt = dict_pll['dt']
		self.pll_id = pll_id

		# this is being set after all (random) delays have been drawn
		self.phi_array_len = None
		# each Delayers neighbors are either given by a coupling topology stored in dict_pll['G'] or are updated during the simulation given the positions and a signal propagation speed
		if not dict_net['special_case'] == 'distanceDepTransmissionDelay':
			self.neighbor_ids = [n for n in dict_pll['G'].neighbors(self.pll_id)]	# for networkx > v1.11
		else:
			self.neighbor_ids = None											# to be set via fuction self.set_list_of_current_neighbors(list_of_neighbors)
		print('\nI am the delayer of PLL%i, my neighbors (initially) have indexes:' % self.pll_id, self.neighbor_ids)

		if ( (isinstance(dict_pll['transmission_delay'], float) or isinstance(dict_pll['transmission_delay'], int)) and not dict_net['special_case'] == 'timeDepTransmissionDelay'):
			self.transmit_delay 		= dict_pll['transmission_delay']
			self.transmit_delay_steps 	= int(np.round( self.transmit_delay / self.dt ))	# when initialized, the delay in time-steps is set to delay_steps
			if ( self.transmit_delay_steps == 0 and self.transmit_delay > 0 ):
				print('Transmission delay set nonzero but smaller than the time-step "dt", hence "self.transmit_delay_steps" < 1 !'); sys.exit()
			elif ( self.transmit_delay_steps == 0 and self.transmit_delay == 0 ):
				print('Transmission delay set to zero!')
			#self.get_delayed_states		= lambda;
			self.pick_delayed_phases = lambda phi, t, abs_t, tau: phi[(t-tau)%self.phi_array_len, self.neighbor_ids]

		# calculate tranmission delays steps, here pick for each Delayer individually but the same for each input l or even for each connection tau_kl individually
		elif ( isinstance(dict_pll['transmission_delay'], list) or isinstance(dict_pll['transmission_delay'], np.ndarray)
												or dict_net['special_case'] == 'distanceDepTransmissionDelay' and not dict_net['special_case'] == 'timeDepTransmissionDelay' ):

			if np.array(dict_pll['transmission_delay']).ndim == 1:				# tau_k case -- all inputs are subject to the same transmission delay
				print('Delayer has different delays for each oscillator! Hence: tau_k are introduced and all incoming signals are subject to the same time delay.')
				self.transmit_delay_steps= int(np.round(dict_pll['transmission_delay'][self.pll_id] / self.dt))
				self.pick_delayed_phases = lambda phi, t, abs_t, tau_k: phi[(t-tau_k)%self.phi_array_len, self.neighbor_ids]

			elif np.array(dict_pll['transmission_delay']).ndim == 2: 			# tau_kl case -- all iputs can have different transmission delay values -- here we provide a 2D matrix of time delays
				print('Delayer has different delays for each input signal! Hence: tau_kl are introduced via a matrix with dimensions %ix%i.'%(dict_net['Nx']*dict_net['Ny'], dict_net['Nx']*dict_net['Ny']))
				# pick all time delays of the neighbors of oscillator k an save those as a list to self.transmit_delay_steps, dict_pll['transmission_delay'] is an (Nx*Ny) x (Nx*Ny) matrix
				# generate a list with all time delays in simulation steps for all neighbors
				tempTauSteps_kl			 = [int(np.round(dict_pll['transmission_delay'][self.pll_id, i] / self.dt)) for i in self.neighbor_ids]
				self.transmit_delay_steps= np.array(tempTauSteps_kl)			# save as an array to object
				self.pick_delayed_phases = lambda phi, t, abs_t, tau_kl: [phi[(t-tau_kl[i])%self.phi_array_len, self.neighbor_ids[i]] for i in range(len(self.neighbor_ids))]

		# time-dependent delays as defined by a function
		elif (dict_net['special_case'] == 'timeDepTransmissionDelay'):

			print('Time dependent transmission delay set!')
			time_dep_delay = setup.setupTimeDependentParameter(dict_net, dict_pll, dict_data, parameter='transmission_delay', afterTsimPercent=0.25, forAllPLLsDifferent=False)

			if len(time_dep_delay[:,0]) == dict_net['Nx']*dict_net['Ny']:		# if there is a matrix, i.e., different time-dependencies for different delay, then use this
				print('Test')
				selfidx_or_ident = self.pll_id
			else:																# this is the case if all transmission delays have the same time dependence
				selfidx_or_ident = 0
			dict_pll.update({'transmission_delay': time_dep_delay})

			# TODO make a test of this! externalize to a test lib
			plt.figure(1234)
			plt.plot(np.arange(0, len(dict_pll['transmission_delay'][selfidx_or_ident, :])) * dict_pll['dt'], dict_pll['transmission_delay'][selfidx_or_ident, :])
			plt.xlabel('time'); plt.ylabel('delay value [s]'); plt.title('time-dependence of transmission delay over simulation time')
			plt.draw(); plt.show()

			self.transmit_delay 		= dict_pll['transmission_delay'][selfidx_or_ident, :]	# each object only knows its own sending delay time dependence
			self.transmit_delay_steps 	= [int(np.round(delay / self.dt)) for delay in self.transmit_delay] # when initialized, the delay in time-steps is set to delay_steps

			if self.transmit_delay_steps == 0 and self.transmit_delay > 0:
				print('Transmission delay set nonzero but smaller than the time-step "dt", hence "self.transmit_delay_steps" < 1 !'); sys.exit()

			#self.get_delayed_states		= lambda;
			self.pick_delayed_phases = lambda phi, t, abs_t, tau: phi[(t-tau[abs_t])%self.phi_array_len, self.neighbor_ids]

		if isinstance(dict_pll['feedback_delay'], float) or isinstance(dict_pll['feedback_delay'], int):
			self.feedback_delay 		= dict_pll['feedback_delay']
			self.feedback_delay_steps 	= int(np.round( self.feedback_delay / self.dt ))	# when initialized, the delay in time-steps is set to delay_steps
			if self.feedback_delay_steps == 0 and self.feedback_delay > 0:
				print('Feedback set to nonzero but is smaller than the time-step "dt", hence "self.feedback_delay_steps" < 1 !'); sys.exit()

	def set_list_of_current_neighbors(self, list_of_neighbors: list) -> None:
		"""Set the internal list of neighbors of the Delayer.

		Args:
			list_of_neighbors: list of the current coupling neighbors of the Delayer
		"""
		self.neighbor_ids = list_of_neighbors

	def get_list_of_current_neighbors(self) -> list:
		"""Get the internal list of neighbors of the Delayer.

		Returns:
			self.neighbor_ids, the pll_id's of all the neighbors of the current Delayer
		"""
		return self.neighbor_ids

	def set_current_transmit_delay_steps(self, current_transmit_delay_steps: list) -> None:
		"""Set the internal list of the current signal propagation time delay (in simulation steps) of the Delayer according to the, e.g., distances.

		Args:
			current_transmit_delay_steps: the current time delay values in simulation steps for the reception of signals from each neighbor
		"""
		self.transmit_delay_steps = current_transmit_delay_steps

	def get_current_transmit_delay_steps(self) -> list:
		"""Get the internal list of neighbors of the Delayer.

		Returns:
			self.transmit_delay_steps, the current time delay values in simulation steps for the reception of signals from each neighbor
		"""
		return self.transmit_delay_steps

	def evolve_delay_in_time(self, new_delay_value: float) -> None:
		"""Assigns a new delay value to an incoming signal. So far this only works if all incoming delays are equal.

		Args:
			new_delay_value: the new delay value

		"""
		self.transmit_delay = new_delay_value
		self.transmit_delay_steps = int(np.round( self.transmit_delay / self.dt ))

	def next(self, index_current_time_cyclic: int, phase_memory: np.ndarray, index_current_time_absolute: int) -> Tuple[np.ndarray, np.ndarray]:
		"""Assigns a new delay value to an incoming signal. So far this only works if all incoming delays are equal.

		Args:
			index_current_time_cyclic: the current time index limited by the size of the phase memory
			phase_memory: a container for the history of all phases
			index_current_time_absolute: the absolute current time index within the simulation

		Returns:
			a tuple of arrays representing the feedback delayed phase states of the oscillator and the transmission delayed phase states of the neighbors

		"""
		#print('in delayer next: self.transmit_delay_steps=', self.transmit_delay_steps)
		transmission_delayed_phases = self.pick_delayed_phases(phase_memory, index_current_time_cyclic, index_current_time_absolute, self.transmit_delay_steps)
		feedback_delayed_phase		= phase_memory[(index_current_time_cyclic - self.feedback_delay_steps) % self.phi_array_len, self.pll_id]
		#print('PLL%i with its feedback_delayed_phase [in steps]:'%self.idx_self,feedback_delayed_phase,', receives transmission_delayed_phases [in steps]:',transmission_delayed_phases,' from self.idx_neighbors:', self.idx_neighbors)
		#time.sleep(1)

		return np.asarray(feedback_delayed_phase), np.asarray(transmission_delayed_phases) # x is is the time-series from which the values at t-tau_kl^f and t-tau_kl are returned


class Counter:
	"""Counts cycles or half cycles of periodic processes using the phase variable.

	Counts multiples of PI or 2*PI to count half or full cycles.

	Attributes:
		reference_phase: the phase at which counting starts
	"""
	def __init__(self):
		self.reference_phase: float = 0

	def reset(self, reference_phase: float) -> None:
		"""Resets the clock counter by setting the reference phase.

		Args:
			reference_phase: the phase at which counting starts

		"""
		self.reference_phase = reference_phase
		return None

	def read_periods(self, current_phase: float) -> int:
		"""
		Calculates number of fully completed periods given by the difference between the current and reference phase.

		Args:
			current_phase: phase at which periods should be calculated

		Returns:
			fully completed periods at the given phase

		"""
		return np.floor((current_phase - self.reference_phase) / (2.0 * np.pi))

	def read_half_periods(self, current_phase: float) -> int:
		"""
		Calculates number of half periods given by the difference between the current and reference phase.

		Args:
			current_phase: phase at which half periods should be calculated

		Returns:
			half periods at the given phase

		"""
		return np.floor((current_phase - self.reference_phase) / np.pi)


class PhaseLockedLoop:
	""" This class represents an oscillator / phase-locked loop (PLL), handles signal flow between its components.
		The oscillator can receive external signals and compare those with its own state, filter / process the result
		of this comparison and adjust the output frequency accordingly.

		This defines an oscillator, the cross coupling time delays with which it receives external signals from other
		entities of the network, its intrinsic frequency, the properties of the loop filter or internal signal
		processing, the interaction / coupling strength, and many additional options such as additional injection of
		signals, noise, heterogeneity, etc.

		It allows to implement first and second order Kuramoto models with delayed and non delayed coupling for
		different coupling topologies, heterogeneous oscillators and dynamic and quenched noise.

		Attributes:
			delayer: 						organizes delayed coupling
			phase_detector_combiner: 		extracts phase relations between feedback and external signals
			low_pass_filter: 				filters high frequency components (first or second order low pass)
			signal_controlled_oscillator 	determines instantaneous frequency of autonomous oscillator with respect to an input signal
			counter:						derives a clock / time from the oscillators clocking signal
			pll_id:							identifier of the oscillator within the network
	"""

	def __init__(self, pll_id: int, delayer: Delayer, phase_detector_combiner: PhaseDetectorCombiner,
				 low_pass_filter: LowPassFilter, signal_controlled_oscillator: SignalControlledOscillator,
				 counter: Counter):
		"""
			Args:
				delayer: 						organizes delayed coupling
				phase_detector_combiner: 		extracts phase relations between feedback and external signals
				low_pass_filter: 				filters high frequency components (first or second order low pass)
				signal_controlled_oscillator 	determines instantaneous frequency of autonomous oscillator with respect to an input signal
				counter:						derives a clock / time from the oscillators clocking signal
				pll_id:							identifier of the oscillator within the network
				pll_rx_signal_distance_treshold distance treshold beyond which no signal from other oscillators can be received (e.g., in wireless coupling)
				pll_coordinate_vector_3d		coordinate of the oscillator in a 3D coordinate system (x, y, z)
				pll_diff_var_vector_3d			variance of Gaussian white noise driven diffusion of the oscillator in a 3D coordinate system (var_x, var_y, var_z)
				pll_speed_vector_3d				speed of the oscillator in a 3D coordinate system (velocity_x, velocity_y, velocity_z)

		"""
		self.delayer = delayer
		self.phase_detector_combiner = phase_detector_combiner
		self.low_pass_filter = low_pass_filter
		self.signal_controlled_oscillator = signal_controlled_oscillator
		self.counter = counter
		self.pll_id = pll_id

		self.signal_propagation_speed = None
		self.pll_coordinate_vector_3d = None
		self.pll_diff_var_vector_3d = None
		self.geometry_of_treshold = None
		self.pll_speed_vector_3d = None
		self.distance_treshold = None


	def next(self, index_current_time_absolute: int, length_phase_memory: int, phase_memory: np.ndarray) -> np.ndarray:
		""" Function that evolves the oscillator forward in time by one increment based on the external signals and internal dynamics.
			1) delayer obtains past states of neighbors in the network coupled to this oscillator and the current state of the oscillator itself
			2) from these states the input phase relations are evaluated by the phase detector and combiner which yields the PD signal (averaged over all inputs)
			3) the PD signal is fed into the loop filter and yields the control signal
			4) the voltage controlled oscillator evolves the phases according to the control signal

			Args:
				index_current_time_absolute: current time within simulation
				length_phase_memory: length of container that stores the phases of the oscillators, needed for organizing the cyclic memory to handle the delay
				phase_memory: holds the phases of all oscillators for at least the time [-tau_max, 0], denotes the memory necessary for the time delay

			Returns:
				updated phase incrementing the time by one step

			Raises:
		"""
		feedback_delayed_phases, transmission_delayed_phases = self.delayer.next(index_current_time_absolute % length_phase_memory, phase_memory, index_current_time_absolute)

		phase_detector_output = self.phase_detector_combiner.next(feedback_delayed_phases, transmission_delayed_phases, 0, index_current_time_absolute)

		control_signal = self.low_pass_filter.next(phase_detector_output)

		updated_phase = self.signal_controlled_oscillator.next(control_signal)[0]
		return updated_phase

	def clock_periods_count(self, current_phase_state: np.ndarray) -> int:
		""" Function that counts the periods of oscillations that have passed for the oscillator. This enables the derivation of a time,
			i.e., extends the oscillator to become a clock.

			Args:
				current_phase_state:	the current phase of the oscillator

			Returns:
				the number of cycles counted
		"""
		return self.counter.read_periods(current_phase_state)

	def clock_halfperiods_count(self, current_phase_state: np.ndarray) -> int:
		""" Function that counts TWICE per period for oscillations of the oscillator, e.g., counting the falling and rising edge of digital signal.
			This enables the derivation of a time, i.e., extends the oscillator to become a clock.

			Args:
				current_phase_state:	the current phase of the oscillator

			Returns:
				the number of half cycles counted
		"""
		return self.counter.read_half_periods(current_phase_state)

	def clock_reset(self, current_phase_state: np.ndarray) -> None:
		""" Function that resets the clock count of the oscillator.

			Args:
				current_phase_state:	the current phase of the oscillator (as new reference for the zero count)
		"""
		self.counter.reset(current_phase_state)

	def setup_hist_reverse(self) -> float:
		""" Function that sets the history of the oscillator using the voltage controlled oscillator. Given the required phase at the start of the simulation,
			the function calls the VCO's function to set the history/memory backwards in time until the memory is filled.

			Returns:
				an array with the memory of the particular oscillator, to be written into the phi container that holds all oscillators' phases
		"""
		return self.signal_controlled_oscillator.set_initial_reverse()[0]

	def set_delta_perturbation(self, perturbation, instantaneous_frequency, prior_to_instantaneous_frequency) -> float:
		""" Function that applies a delta-like perturbation to the phase of the oscillator at the start of the simulation.
			Corrects the internal state of the oscillator with respect to the perturbation, calculated the initial control signal.

			Args:
				perturbation:							the perturbation to the current phase
				instantaneous_frequency:		 		the current instantaneous frequency of the oscillator
				prior_to_instantaneous_frequency: 		the previous (time increment) instantaneous frequency of the oscillator

			Returns:
				the current perturbed phase of the oscillator
		"""
		# the filtering at the loop filter is applied to the phase detector signal
		control_signal = self.low_pass_filter.set_initial_control_signal(instantaneous_frequency, prior_to_instantaneous_frequency)
		return self.signal_controlled_oscillator.delta_perturbation(perturbation, control_signal)[0]

	def next_no_external_input(self, index_current_time: int, length_phase_memory: int, phase_memory: np.ndarray) -> float:
		""" Function that evolves the voltage controlled oscillator in a closed loop configuration when there is no external input, i.e., free-running closed loop PLL.
			1) delayer obtains the current state of the oscillator itself (potentially delayed by a feedback delay)
			2) this state is fed in the phase detector and combiner for zero external signal which yields the PD signal
			3) the PD signal is fed into the loop filter and yields the control signal
			4) the voltage controlled oscillator's next function evolves the phases dependent on the control signal

			Args:
				index_current_time:		current time within simulation
				length_phase_memory:	length of container that stores the phases of the oscillators, needed for organizing the cyclic memory to handle the delay
				phase_memory:			holds the phases of all oscillators for at least the time [-tau_max, 0], denotes the memory necessary for the time delay

			Returns:
				updated phase incrementing the time by one step

			Raises:
		"""
		current_phase_state, delayed_phase_state = self.delayer.next(index_current_time % length_phase_memory, phase_memory, index_current_time)
		phase_detector_output = self.phase_detector_combiner.next(current_phase_state, 0, 0, index_current_time)
		control_signal = self.low_pass_filter.next(phase_detector_output)
		return self.signal_controlled_oscillator.next(control_signal)[0]

	def next_free_running_open_loop(self) -> float:
		""" Function that evolves the voltage controlled oscillator when there is no external input, i.e., free-running open loop PLL.

			Returns:
				the updated phase
		"""
		return self.signal_controlled_oscillator.next(0)[0]

	def set_position_3d(self, position_vector_in_3d_cartesian_coordinates: np.ndarray) -> None:
		""" Function that sets the position, speed and variance of the diffusion which is driven by Gaussian white noise.

			Args:
				position_vector_in_3d_cartesian_coordinates: sets the position of the oscillator in a cartesian coordinate system

			Raises:
		"""

		self.pll_coordinate_vector_3d = position_vector_in_3d_cartesian_coordinates

	def set_speed_3d(self, speed_vector_in_3d_cartesian_coordinates: np.ndarray) -> None:
		""" Function that sets the position, speed and variance of the diffusion which is driven by Gaussian white noise.

			Args:
 				speed_vector_in_3d_cartesian_coordinates: sets the speed of the oscillator in 3D space

			Raises:
		"""

		self.pll_speed_vector_3d = speed_vector_in_3d_cartesian_coordinates

	def set_gwn_diffusion_variance_3d(self, variance_gwn_diffusion_cartesian_coordinates: np.ndarray) -> None:
		""" Function that sets the position, speed and variance of the diffusion which is driven by Gaussian white noise.

			Args:
				variance_gwn_diffusion_cartesian_coordinates: sets the variance of the Gaussian white noise driven diffusion of the oscillator

			Raises:
		"""

		self.pll_diff_var_vector_3d = variance_gwn_diffusion_cartesian_coordinates

	def get_position_3d(self) -> np.ndarray:
		""" Function that get the current position.

			Returns:
				an np.array with the position of the oscillator
				access: return_array[0:3] --> coordinate components x, y, z

			Raises:
		"""

		return self.pll_coordinate_vector_3d

	def get_speed_3d(self) -> np.ndarray:
		""" Function that get the current speed.

			Returns:
				an np.array with the speed of the oscillator
				access: return_array[0:3] --> speed components x, y, z

			Raises:
		"""

		return self.pll_speed_vector_3d

	def get_gwn_diffusion_variance_3d(self) -> np.ndarray:
		""" Function that get the current variance of the Gaussian white noise diffusion process.

			Returns:
				an np.array with the variance of the Gaussian white noise driven diffusion in a cartesian coordinate system
				access: return_array[0:3] --> variance components x, y, z

			Raises:
		"""

		return self.pll_diff_var_vector_3d

	def set_receive_sig_distance_treshold(self, distance_treshold: np.ndarray, geometry_of_treshold: str) -> None:
		""" Function that sets a distance treshold beyond which no signal from other oscillators are received. Can have different shapes, circular, rectangular, etc.

			Args:
				distance_treshold: defines the radius of a circle or rectangle from within which range signals from other oscillators can be received
 				geometry_of_treshold: defines the geometry

			Raises:
		"""

		self.distance_treshold = distance_treshold
		self.geometry_of_treshold = geometry_of_treshold

	def evolve_position_in_3d(self) -> np.ndarray:
		""" Function that evolves the position in 3d space according to the diffusion coefficient and the deterministic speed components.

			Args:
			Raises:
				Error if position, speed, and diffusion coefficient are not set.
		"""

		self.pll_coordinate_vector_3d = ( self.pll_coordinate_vector_3d + self.pll_speed_vector_3d * self.delayer.dt
												+ np.random.normal(loc=0.0, scale=np.sqrt(self.pll_diff_var_vector_3d * self.delayer.dt)) )
		return self.get_position_3d()


	# def update_list_of_neighbors_in_coupling_range(self, all_plls_positions: np.ndarray, distance_treshold: np.ndarray, geometry_of_treshold: str) -> None:
	# 	"""Function that evaluates the distances so all other oscillators and stores the ids of those that are in the defined coupling range.
	#
	# 		Args:
	# 			all_plls_positions: contains the current position of all entities
	# 			distance_treshold: defines the radius of a circle or rectangle from within which range signals from other oscillators can be received
	#  			geometry_of_treshold: defines the geometry
	#
	# 		Returns:
	# 			TODO
	# 	"""
	# 	list_of_neighbors_in_range = []
	# 	for i in range(len(all_plls_position[:,0])):
	# 		if ( i ~= self.pll_id and np.sqrt( (all_plls_position[self.pll_id,0]-all_plls_position[i,0])**2 + (all_plls_position[self.pll_id,1]-all_plls_position[i,1])**2
	# 						+ (all_plls_position[self.pll_id,2]-all_plls_position[i,2])**2 ) < distance_treshold ):
	# 			list_of_neighbors_in_range.append(i)
	#
	# 	self.delayer.set_list_of_current_neighbors(list_of_neighbors_in_range)

	# def update_propagation_time_delay_matrix_of_network(self, all_plls_positions: np.ndarray) -> np.ndarray:
	# 	"""Function that updates the propagation time delays between each pair of oscillators in the system.
	#
	# 		Args:
	# 			all_plls_positions: contains the current position of all entities
	# 			distance_treshold: defines the radius of a circle or rectangle from within which range signals from other oscillators can be received
	#  			geometry_of_treshold: defines the geometry
	#
	# 		Returns:
	# 			an np.ndarray with signal propagation times between the oscillators positions
	# 	"""
	#
	# 	self.delayer.set_current_transmit_delay_steps(current_transmit_delay_steps)
	# 	return propagation_time_delay_in_steps_matrix

	# def obtain_transmission_time_delay_from_distances(self, first_position: np.ndarray, second_position: np.ndarray, signal_propagation_speed: np.float) -> np.float:
	# 	"""Function that calculates the distance between two coordinate vectors x_1 and x_2 and returns the time it takes to go from x_1 to x_2 at a given
	# 		constant propagation speed.
	#
	# 		Args:
	# 			first_position: contains the position of a first entity
	# 			second_position: contains the position of a second entity
	# 			signal_propagation_speed: the speed with which the signal propagates
	#
	# 		Returns:
	# 			signal propagation time between positions x1 and x2
	# 	"""
	#
	# 	return np.sqrt( (first_position[0]-second_position[0])**2 + (first_position[1]-second_position[1])**2 + (first_position[2]-second_position[2])**2 ) / signal_propagation_speed


class Space:
	""" This class represents a rectangular space in which oscillators can exist. It can have closed or periodic boundary conditions.

		With the definition of 3d space, the oscillators mutual coupling and the associated transmission time delays can be computed according to the distances between
		the oscillators and  the signal propagation velocity.

		Attributes:
			signal_propagation_speed: the velocity with which signals propagate in this space
			dimensions_xyz: the length of the boundaries in x-, y- and z-direction
	"""

	def __init__(self, signal_propagation_speed: np.float, dimensions_xyz: np.ndarray):
		"""
			Args:
				signals_tracked_currently: N x N matrix in which all current potential signal exchange events are tracked, the entry contains the time of signal emission t_e
				signal_propagation_speed: the velocity with which signals propagate in this space
				dimensions_xyz: the length of the boundaries in x, y, and z-direction
		"""
		self.signals_tracked_currently = None
		self.signal_propagation_speed = signal_propagation_speed
		self.dimensions_xyz = dimensions_xyz

	def update_adjacency_matrix_for_all_plls_potentially_receiving_a_signal(self, all_pll_positions: np.ndarray, distance_treshold: np.ndarray, geometry_of_treshold: str) -> np.ndarray:
		"""Function that updates the adjacency matrix by checking for each oscillator whether it is within the coupling of another in its vicinity
 			according to the distance treshold and geometry of the treshold.
			If all distance tresholds are equal, any connection found is bidirectional and hence only one side of the diagonal of the coupling function needs to be checked.

			Args:
				all_pll_positions: contains the current position of all entities
				distance_treshold: defines the radius of a circle or rectangle from within which range signals from other oscillators can be received
	 			geometry_of_treshold: defines the geometry

			Returns:
				TODO
		"""
		# use all current positions to calculate the current adjacency matrix, since we assume the signal propagation to have equal velocity in either direction,
		# we only need to compute one side of the matrix of oscillators who are in potential coupling range at a time t (symmetric about the main diagonal)
		num_plls = len(all_pll_positions[:])
		temp_adjacency_matrix = np.empty([num_plls, num_plls])
		temp_adjacency_matrix.fill(np.nan)
		for i in range(num_plls):
			for j in range(i+1, num_plls):
				# THIS IS NOT FASTER: #np.sqrt( np.linalg.norm( np.subtract( all_plls_position[i,0], all_plls_position[j,0]) ) )**2 )
				distance_of_pair_ij = ( np.sqrt( (all_pll_positions[i,0]-all_pll_positions[j,0])**2 + (all_pll_positions[i,1]-all_pll_positions[j,1])**2
														+ (all_pll_positions[i,2]-all_pll_positions[j,2])**2 ) )
				if distance_of_pair_ij < distance_treshold:
					temp_adjacency_matrix[i, j] = 1
					temp_adjacency_matrix[j, i] = 1

		return temp_adjacency_matrix
