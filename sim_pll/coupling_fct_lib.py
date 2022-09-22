#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import sys, gc
import numpy as np
import random
#cimport numpy as np
#cimport cython
import networkx as nx
from scipy.signal import sawtooth
from scipy.signal import square
from scipy.stats import cauchy


def neg_cosine(x: np.float) -> np.float:
	"""
		Computes the negative cosine of the argument x.

		Args:
			x: argument

		Returns:
			np.float value of function
		"""
	return -1.0*np.cos(x)


def cosine(x: np.float) -> np.float:
	"""
		Computes the cosine of the argument x.

		Args:
			x: argument

		Returns:
			np.float value of function
		"""
	return np.cos(x)


def inverse_cosine(y: np.float, branch: str = 'positive') -> np.float:
	"""
		Computes the inverse of the cosine of the argument y.

		Args:
			y: argument
			branch: HERE: dummy variable to chose negative or positive branch

		Returns:
			np.float value of function
		"""
	# if y>=0:
	# 	return +np.arccos(y)
	# else:
	# 	return -np.arccos(y)
	return np.arccos(y)


def neg_sine(x: np.float) -> np.float:
	"""
		Computes the negative sine of the argument x.

		Args:
			x: argument

		Returns:
			np.float value of function
		"""
	return -1.0*np.sin(x)


def sine(x: np.float):
	"""
		Computes the sine of the argument x.

		Args:
			x: argument

		Returns:
			np.float value of function
		"""
	return np.sin(x)


def inverse_sine(y: np.float, branch: str = 'positive') -> np.float:
	"""
		Computes the inverse sine of the argument y.

		Args:
			y: argument
			branch: HERE: dummy variable to chose negative or positive branch

		Returns:
			np.float value of function
		"""
	# if y>=0:
	# 	return np.arcsin(y)
	# else:
	# 	return np.pi+np.arcsin(y)
	np.arcsin(y)


def triangular(x: np.float) -> np.float:
	"""
		Computes the triangular function of the argument x.

		Args:
			x: argument

		Returns:
			np.float value of function
		"""
	return sawtooth(x, width=0.5)


def deriv_triangular(x: np.float) -> np.float:
	"""
		Computes the derivative of the triangular function of the argument x.

		Args:
			x: argument

		Returns:
			np.float value of function
		"""
	return (2.0/np.pi)*square(x, duty=0.5)


def inverse_triangular(y: np.float, branch: str = 'positive') -> np.float:
	"""
		Computes the inverse of the triangular function of the argument y.

		Args:
			y: argument
			branch: chose negative or positive branch

		Returns:
			result of either of the two branches
		"""

	if branch == 'positive' and np.abs(y) <= 1:
		return (np.pi/2)*(y+1)
	elif branch == 'negative' and np.abs(y) <= 1:
		return -(np.pi/2)*(y+1)
	else:
		return np.nan


def square_wave(x: np.float, duty=0.5) -> np.float:
	"""
		Computes the square wave function with peak to peak amplitude one and shifted to the interval [0, 1] of the argument x.

		Args:
			x: argument
			duty: determines the duty cycle of the square wave

		Returns:
			tuple of results of the two branches
		"""
	return 0.5*(1.0+square(x, duty=0.5))


def square_wave_symm_zero(x: np.float, duty=0.5) -> np.float:
	"""
		Computes the functional value of the square wave function with peak to peak amplitude two and symmetric about zero of the argument x.

		Args:
			x: argument
			duty: determines the duty cycle of the square wave

		Returns:
			functional value of square wave function at x
		"""
	return 0.5*(square(x, duty=0.5))


def pfd(x: np.float) -> np.float:
	"""
		Computes phase frequency detector function (pfd) of the argument x.

		Args:
			x: argument

		Returns:
			results of phase detection with a pfd
		"""
	return 0.5*(np.sign(x)*(1+sawtooth(1*x*np.sign(x), width=1)))


def inverse_linear(y: np.float) -> np.float:
	"""
		Computes the inverse a linear function of the argument y.

		Args:
			y: argument

		Returns:
			np.float value of function
		"""
	return y


def nonlinear_response_vco_3rd_gen(loop_filter_signal: np.float, voltage_bias: np.float, normalized=True, vco_parameter_1=20.7923E9, vco_parameter_2=2.1944E9) -> np.float:
	"""
		Computes the nonlinear response of the VCO according to the functional form derived for the 3rd generation prototypes.

		Args:
			loop_filter_signal: argument
			voltage_bias: (pre)bias voltage to set the frequency of the VCO
			normalized: whether all parameters are normalized by the mean intrinsic frequency or not
			vco_parameter_1: first fit parameter of VCO response function
			vco_parameter_2: second fit parameter of VCO response function

		Returns:
			the frequency of the VCO in Hertz - if loop_filter_signal is zero: free-running, otherwise closed-loop
		"""
	return vco_parameter_1 + vco_parameter_2 * np.sqrt(voltage_bias + loop_filter_signal)


def nonlinear_response_vco_3rd_gen_calculate_voltage_bias(frequency_of_vco_in_hz, vco_parameter_1=20.7923E9, vco_parameter_2=2.1944E9):
	"""
		Computes the necessary voltage bias in volts to achieve a specific intrinsic VCO frequency according to the nonlinear VCO response curve.
		Assumes that there is no additional input from the control/loop filter signal.

		Args:
			frequency_of_vco_in_hz: argument [Hz]
			vco_parameter_1: first fit parameter of VCO response function
			vco_parameter_2: second fit parameter of VCO response function

		Returns:
			the bias voltage(s) for zero control/loop filter signal for a given intrinsic VCO('s) frequency in Hertz
		"""
	if isinstance(frequency_of_vco_in_hz, list):
		frequency_of_vco_in_hz = np.array(frequency_of_vco_in_hz)
	return ((frequency_of_vco_in_hz - vco_parameter_1) / vco_parameter_2) ** 2


def inverse_nonlinear_response_vco_3rd_gen(frequency_of_vco_in_hz: np.float, voltage_bias: np.float, normalized=True, vco_parameter_1=20.7923E9, vco_parameter_2=2.1944E9) -> np.float:
	"""
		Computes the inverse nonlinear response of the VCO according to the functional form derived for the 3rd generation prototypes.

		Args:
			frequency_of_vco_in_hz: argument [Hz]
			voltage_bias: prebias voltage to set the frequency of the VCO
			normalized: whether all parameters are normalized by the mean intrinsic frequency or not
			vco_parameter_1: first fit parameter of VCO response function
			vco_parameter_2: second fit parameter of VCO response function

		Returns:
			the loop_filter_signal for a given frequency at fixed bias frequency
		"""
	return ((frequency_of_vco_in_hz - vco_parameter_1) / vco_parameter_2) ** 2 - voltage_bias