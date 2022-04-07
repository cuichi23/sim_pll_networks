## @package PLL library
#  Documentation for this module.
#
#  authors: Deborah Schmidt, Lucas Wetzel (wetztel.lucas[at]gmail.com)

#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import sys
import gc
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

''' Enable automatic carbage collector '''
gc.enable()

# %%cython --annotate -c=-O3 -c=-march=native

''' SPLL library
	authors: Deborah Schmidt, Lucas Wetzel (lwetzel[at]pks.mpg.de)
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
	if (isinstance(input, list) or isinstance(input, np.ndarray)) and len(input) == pll_count:
		if len(input) > 1:  # additional test, in case a single float has been case into a list or array
			return input[pll_id]  # set individual value
		else:
			return input  # set value for all
	elif isinstance(input, float) or isinstance(input, int):
		return input  # set value for all
	else:
		print('Error in PLL component constructor setting a variable using get_from_value_or_list() function! Check whether enough values for each oscillator are defined!')
		sys.exit()


class SignalControlledOscillator:
	"""A signal controlled oscillator is an autonomous oscillator that can change its instantaneous frequency as a function of the control signal.

	SCO: d_phi / d_t = f(x_ctrl) = omega + K * x_ctrl + O(epsilon > 1)

	Attributes:
		pll_id: the oscillator's identity
		sync_freq_rad: frequency of synchronized states in radHz (Omega)
		intr_freq_rad: intrinsic frequency of free running closed loop oscillator in radHz (omega)
		fric_coeff: friction coefficient
		K_rad: coupling strength in radHz
		c: noise strength -- provides the variance of the GWN process
		dt: time increment
		phi: this is the internal representation of the oscillator's phase, NOT the container in simulateNetwork
		response_vco: defines a nonlinear VCO response, either set to 'linear' or the nonlinear expression
		init_freq: defines the initial frequency of the signal controlled oscillator according to the phase history
		evolve_phi: function defining how the phase evolves in time, e.g., with or without noise, linear vs. nonlinear
		d_phi: stores the phase increment between the current and prior simulation step
	"""
	def __init__(self, pll_id, spll, dict_net):
		"""
		Args:
			pll_id: the oscillator's identity
			dict_pll: oscillator related properties and parameters
			dict_net: network related properties and parameters
		"""
		self.d_phi = None
		self.pll_id = pll_id
		self.instantaneous_freq = spll.initial_instantaneous_frequency
		self.periodic_output_signal = spll.periodic_output_signal

	def run_oscillations(self) -> float:
		"""
		Generates the periodic output signal according to the instantaneous frequency of the SPLL, driven by the machine's system time.

		Args:
		"""
		return self.periodic_output_signal(2.0*np.pi*self.instantaneous_freq*1E-9*time.time_ns())




def run_osci(update_in_ns, frequenz):
	l = []
	counter = 0
	run_osci = True
	t = time.time_ns()
	while run_osci:
		if time.time_ns() - t >= update_in_ns:
			t = time.time_ns()
			l.append(t)
			print('PING:', t)
			counter = counter + 1
		if counter > 10000:
			run_osci = False
	plt.plot(np.sin(2.0*np.pi*frequenz*1E-9*np.array(l)))





class SoftwarePhaseLockedLoop:
	""" This class represents an oscillator / software hase-locked loop (SPLL), handles signal flow between its components.
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
				pll_rx_signal_distance_threshold distance threshold beyond which no signal from other oscillators can be received (e.g., in wireless coupling)
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

	# NOTE: found better solution, of ok... delete this when you see it
	# def next_N_sequential_LFs(self, index_current_time_absolute: int, length_phase_memory: int, phase_memory: np.ndarray, order_filter: int) -> np.ndarray:
	# 	""" Function that evolves the oscillator forward in time by one increment based on the external signals and internal dynamics.
	# 		1) delayer obtains past states of neighbors in the network coupled to this oscillator and the current state of the oscillator itself
	# 		2) from these states the input phase relations are evaluated by the phase detector and combiner which yields the PD signal (averaged over all inputs)
	# 		3) the PD signal is fed into the loop filter and yields the control signal
	# 		4) the voltage controlled oscillator evolves the phases according to the control signal
	#
	# 		Args:
	# 			index_current_time_absolute: current time within simulation
	# 			length_phase_memory: length of container that stores the phases of the oscillators, needed for organizing the cyclic memory to handle the delay
	# 			phase_memory: holds the phases of all oscillators for at least the time [-tau_max, 0], denotes the memory necessary for the time delay
	# 			order_filter: integer that
	#
	# 		Returns:
	# 			updated phase incrementing the time by one step
	#
	# 		Raises:
	# 	"""
	# 	feedback_delayed_phases, transmission_delayed_phases = self.delayer.next(index_current_time_absolute % length_phase_memory, phase_memory, index_current_time_absolute)
	#
	# 	phase_detector_output = self.phase_detector_combiner.next(feedback_delayed_phases, transmission_delayed_phases, 0, index_current_time_absolute)
	#
	# 	input = phase_detector_output
	# 	for i in range(order_filter):											# apply first order loop filter (identical RC) sequentially
	# 		input = self.low_pass_filter.next(input)
	#
	# 	control_signal = input
	#
	# 	updated_phase = self.signal_controlled_oscillator.next(control_signal)[0]
	# 	return updated_phase

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

