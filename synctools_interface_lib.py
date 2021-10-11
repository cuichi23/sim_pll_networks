import datetime
import os

# Required when plot windows should not be displayed
#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import inspect, copy

import synctools_lib as st
import coupling_fct_lib as coupfct

TOPO_0D_GLOBAL 				= 'global'
TOPO_1D_RING 				= 'ring'
TOPO_1D_CHAIN 				= 'chain'
TOPO_1D_ENTRAINONE 			= 'entrainOne'
TOPO_1D_ENTRAINALL 			= 'entrainAll'
TOPO_2D_CUBIC_OPEN 			= 'square-open'
TOPO_2D_CUBIC_PERIODIC 		= 'square-periodic'
TOPO_2D_HEXAGONAL_OPEN 		= 'hexagon-open'
TOPO_2D_HEXAGONAL_PERIODIC 	= 'hexagon-periodic'
TOPO_2D_OCTAGONAL_OPEN 		= 'octagon-open'
TOPO_2D_OCTAGONAL_PERIODIC 	= 'octagon-periodic'

# mixer+1sig shift: +/-np.sin(x), mixer: +/-np.cos(x), XOR: sawtooth(x,width=0.5), PFD: 0.5*(np.sign(x)*(1+sawtooth(1*x*np.sign(x), width=1)))
COUPLING_FUNCTION_TRIANGLE 		= coupfct.triangular 							#'sawtooth'
COUPLING_FUNCTION_PFD 			= coupfct.pfd									#'0.5*(np.sign(x)*(1+sawtooth(1*x*np.sign(x))'
COUPLING_FUNCTION_COS 			= coupfct.cosine								#'np.cos'
COUPLING_FUNCTION_NEGCOS		= coupfct.neg_cosine							#'-np.cos'
COUPLING_FUNCTION_SIN 			= coupfct.sine									#'np.sin'
COUPLING_FUNCTION_NEGSIN		= coupfct.neg_sine								#'-np.sin'
#COUPLING_FUNCTION_SINCOS 		= 'sincos'
#COUPLING_FUNCTION_TRIANGSHIFT 	= 'triangshift'

# #############################################################################


def generate_delay_plot(dictPLL, dictNet, isRadians=True, filename=None, max_delay_range=5.0):
	# Setup sweep factory and create state list
	n_points = 50 * max_delay_range
	if isRadians:
		if dictPLL['transmission_delay'] > 0.5 * max_delay_range:
			tau_min = ( dictPLL['transmission_delay'] - 0.5 * max_delay_range ) / (dictPLL['intrF'] / (2 * np.pi))
			tau_max = ( dictPLL['transmission_delay'] + 0.5 * max_delay_range ) / (dictPLL['intrF'] / (2 * np.pi))
		else:
			tau_min = 0.0
			tau_max = max_delay_range / (dictPLL['intrF'] / (2 * np.pi))
		f  = dictPLL['intrF'] / (2 * np.pi)
		fc = dictPLL['cutFc'] / (2 * np.pi)
		kc = dictPLL['coupK'] / (2 * np.pi)
	else:
		if dictPLL['transmission_delay'] > 0.5 * max_delay_range:
			tau_min = ( dictPLL['transmission_delay'] - 0.5 * max_delay_range ) / dictPLL['intrF']
			tau_max = ( dictPLL['transmission_delay'] + 0.5 * max_delay_range ) / dictPLL['intrF']
		else:

			tau_min = 0.0
			tau_max = max_delay_range / dictPLL['intrF']
		f  = dictPLL['intrF']
		fc = dictPLL['cutFc']
		kc = dictPLL['coupK']
	tau = np.linspace(tau_min, tau_max, n_points); #print('tau_min, tau_max, tau: ', tau_min, tau_max, tau)
	dictPLL.update({'transmission_delay': tau})
	sf  = SweepFactory(dictPLL, dictNet, isRadians=isRadians)
	fsl = sf.sweep()
	if dictNet['mx'] == 0 and dictNet['my'] == -999:
		dictTemp = copy.deepcopy(dictNet); dictTemp.update({'mx': 1});
		sf1 = SweepFactory(dictPLL, dictTemp, isRadians=isRadians); fsl1 = sf1.sweep()
	elif dictNet['mx'] == 1 and dictNet['my'] == -999:
		dictTemp = copy.deepcopy(dictNet); dictTemp.update({'mx': 0});
		sf1 = SweepFactory(dictPLL, dictTemp, isRadians=isRadians); fsl1 = sf1.sweep()

	# Create parameter string
	str_para = ''
	str_para += 'v = %i   mx = %i   my = %i' % (dictPLL['div'], dictNet['mx'], dictNet['my'])
	str_para += '\n%s topology' % dictNet['topology']
	str_para += ' N = %i   Nx = %i   Ny = %i' % (dictNet['Nx']*dictNet['Ny'], dictNet['Nx'], dictNet['Ny'])
	str_para += '\nF = %.2f Hz   Fc = %.2f Hz   Kc = %.2f Hz' % (f, fc, kc)

	# Create figure
	plt.figure(num=1, figsize=(8, 8.2))

	plt.subplot(2, 1, 1)
	plt.title(str_para)
	plt.plot(fsl.get_tau(), fsl.get_omega(), 'b.')
	if ( dictNet['mx'] == 0 or dictNet['mx'] == 1 ) and dictNet['my'] == -999:
		plt.plot(fsl1.get_tau(), fsl1.get_omega(), 'k+', alpha=0.5)
	plt.grid(True, ls='--')
	plt.xlabel('delay [s]')
	plt.ylabel('sync. frequency [rad/s]')
	plt.tight_layout()

	plt.subplot(2, 1, 2)
	plt.axhline(0, color='k')
	plt.plot(fsl.get_tau(), np.real(fsl.get_l()), '.')
	if ( dictNet['mx'] == 0 or dictNet['mx'] == 1 ) and dictNet['my'] == -999:
		plt.plot(fsl1.get_tau(), fsl1.get_l(), 'k+', alpha=0.5)
	plt.grid(True, ls='--')
	plt.xlabel('delay [s]')
	plt.ylabel('stability [rad/s]')
	plt.tight_layout()
	plt.draw()

	# Check if results folder exists
	if not os.path.isdir('results'):
		try:
			os.makedirs('results')
		except:
			raise Exception('results folder does not exist and could not be created')

	# Save figure
	if filename == None:
		dt = datetime.datetime.now()
		str_time = dt.strftime('%Y%m%d_%H%M%S')
		filename = os.path.join('results', 'delay_plot_' + str_time)
	plt.savefig(filename + '.png', dpi=150)
	plt.savefig(filename + '.pdf')

	# Create figure
	plt.figure(num=2, figsize=(8, 8.2))

	plt.subplot(2, 1, 1)
	plt.title(str_para)
	plt.plot(fsl.get_tau()*fsl.get_omega()/(2*np.pi), fsl.get_omega(), '.')
	if ( dictNet['mx'] == 0 or dictNet['mx'] == 1 ) and dictNet['my'] == -999:
		plt.plot(fsl1.get_tau()*fsl1.get_omega()/(2*np.pi), fsl1.get_omega(), 'k+', alpha=0.5)
	plt.grid(True, ls='--')
	plt.xlabel('Omega delay')
	plt.ylabel('sync. frequency [rad/s]')
	plt.tight_layout()

	plt.subplot(2, 1, 2)
	plt.axhline(0, color='k')
	plt.plot(fsl.get_tau()*fsl.get_omega()/(2*np.pi), np.real(fsl.get_l()), '.')
	if ( dictNet['mx'] == 0 or dictNet['mx'] == 1 ) and dictNet['my'] == -999:
		plt.plot(fsl1.get_tau()*fsl1.get_omega()/(2*np.pi), np.real(fsl1.get_l()), 'k+', alpha=0.5)
	plt.grid(True, ls='--')
	plt.xlabel('Omega delay [s]')
	plt.ylabel('stability [rad/s]')
	plt.tight_layout()
	plt.draw()

	# Check if results folder exists
	if not os.path.isdir('results'):
		try:
			os.makedirs('results')
		except:
			raise Exception('results folder does not exist and could not be created')

	# Save figure
	if filename == None:
		dt = datetime.datetime.now()
		str_time = dt.strftime('%Y%m%d_%H%M%S')
		filename = os.path.join('results', 'delayOmeg_plot_' + str_time)
	else:
		dt = datetime.datetime.now()
		str_time = dt.strftime('%Y%m%d_%H%M%S')
		filename = os.path.join('results', 'delayOmeg_plot_' + str_time)
	plt.savefig(filename + '.png', dpi=150)
	plt.savefig(filename + '.pdf')

	# Show figure
	plt.show()




class SweepFactory(object):
	'''Sweeps a system parameters of a coupled PLL system

	   One of the class attributes should be given as a np.ndarray. This will be the swept parameter

	   Attributes
	   ----------
	   n : int/np.ndarray
		   number of oscillators
	   w : float/np.ndarray
		   intrinsic angular frequency
	   k : float/np.ndarray
		   coupling constant
	   tau : float/np.ndarray
			 delay
	   h : callable/list of callables
		   coupling function
	   wc : float/np.ndarray
			(angular) cut-off frequency of low-pass filter
	   m : int
		   twist number
	   v : int/np.ndarray
		   divider for cross-coupling
	   tsim : float
			  simulation time
	'''
	def __init__(self, dictPLL, dictNet, isRadians=True):
		self.n 			= dictNet['Nx']*dictNet['Ny']
		self.nx 		= dictNet['Nx']
		self.ny 		= dictNet['Ny']
		self.tau 		= dictPLL['transmission_delay']
		self.h 			= dictPLL['coup_fct_sig']
		self.m 			= dictNet['mx']
		self.mx 		= dictNet['mx']
		self.my 		= dictNet['my']
		self.tsim 		= dictNet['Tsim']
		self.topology 	= dictNet['topology']
		self.c 			= 0                     								# just dummy variable here
		self.v 			= dictPLL['div']
		self.fric		= dictPLL['friction_coefficient']
		self.dummy		= np.array([self.n])

		if dictPLL['fric_coeff_PRE_vs_PRR'] == 'PRE':							# distinguish between the Kuramoto model as in the PRR paper Wetzel, Metevier, Gupta or the PRE paper Prousalis, Wetzel
			self.fric_omega = 1.0												# PRE: Omega = omega/gamma + K/gamma * h[-Omega Tau], while PRR: Omega = omega + K/gamma * h[-Omega Tau]
		elif dictPLL['fric_coeff_PRE_vs_PRR'] == 'PRR':
			self.fric_omega = 1.0/self.fric

		# if parameters provided in rad*Hz
		if isRadians:
			self.w    = dictPLL['intrF']
			self.k    = dictPLL['coupK']
			self.wc   = dictPLL['cutFc']
		# if parameters provided in Hz, multiply by 2pi, as needed in the phase model
		else:
			self.w    = 2.0*np.pi*dictPLL['intrF']           					# here, w = f
			self.k    = 2.0*np.pi*dictPLL['coupK']           					# here, k is given in Hz instead rad*Hz
			self.wc   = 2.0*np.pi*dictPLL['cutFc']           					# here, wc = fc

		# Identify and store swept variable
		self.key_sweep = self._identify_swept_variable()
		self.values_sweep = self[self.key_sweep]								# use python object as dict, self.values_sweep contains self.m
		self[self.key_sweep] = self.values_sweep[0]								# self[self.key_sweep] now only has the first value (the one to start with)


	def _identify_swept_variable(self):
		'''Identify the swept variable

		   Returns
		   -------
		   var_str  :  str
					   name string of the swept variable
		'''
		if type(self.n) is np.ndarray:
			return 'n'
		elif type(self.nx) is np.ndarray:
			return 'nx'
		elif type(self.ny) is np.ndarray:
			return 'ny'
		elif type(self.tau) is np.ndarray:
			return 'tau'
		elif type(self.w) is np.ndarray:
			return 'w'
		elif type(self.k) is np.ndarray:
			return 'k'
		elif type(self.wc) is np.ndarray:
			return 'wc'
		elif type(self.c) is np.ndarray:
			return 'c'
		elif type(self.m) is np.ndarray:
			return 'mx'
		elif type(self.v) is np.ndarray:
			return 'v'
		elif type(self.fric) is np.ndarray:
			return 'fric'
		else:
			return 'dummy'

	def __getitem__(self, key):
		return self.__dict__[key]

	def __setitem__(self, key, value):
		self.__dict__[key] = value

	def init_system(self):
		# Initilaize coupling function

		#print('inspect.getsourcelines(self.h)[0][0]', inspect.getsourcelines(self.h)[0][0][27:])

		if self.h == COUPLING_FUNCTION_TRIANGLE:								# inspect.getsourcelines(self.h)[0][0][27:35]
			h_func = st.Triangle(1.0 / (2.0 * np.pi))
		elif self.h == COUPLING_FUNCTION_PFD:									# inspect.getsourcelines(self.h)[0][0][27:70]
			h_func = st.PFD(1.0 / (2.0 * np.pi))
		elif self.h == COUPLING_FUNCTION_COS:									# inspect.getsourcelines(self.h)[0][0][27:33]
			h_func = st.Cos(1.0 / (2.0 * np.pi))
		elif self.h == COUPLING_FUNCTION_SIN:									# inspect.getsourcelines(self.h)[0][0][27:33]
			h_func = st.Sin(1.0 / (2.0 * np.pi))
		elif self.h == COUPLING_FUNCTION_NEGCOS:								# inspect.getsourcelines(self.h)[0][0][27:34]
			#h_func = st.NegCos(1.0 / (2.0 * np.pi))
			h_func = st.Cos(1.0 / (2.0 * np.pi), -1.0)							# the -1.0 refers to a sign change of the amplitude!
		elif self.h == COUPLING_FUNCTION_NEGSIN:								# inspect.getsourcelines(self.h)[0][0][27:34]
			#h_func = st.NegSin(1.0 / (2.0 * np.pi))
			h_func = st.Sin(1.0 / (2.0 * np.pi), -1.0)
		#elif inspect.getsourcelines(self.h)[0][0][12:19] == COUPLING_FUNCTION_TRIANGSHIFT:
		#	h_func = st.Triangle(1.0 / (2.0 * np.pi))
		#elif inspect.getsourcelines(self.h)[0][0][12:19] == COUPLING_FUNCTION_SINCOS:
		#	h_func = st.Sin(1.0 / (2.0 * np.pi)) + 0.8 * st.Cos(6.0 * 1.0 / (2.0 * np.pi))
		else:
			raise Exception('Non-valid coupling function string')

		# Initialize arrangement/ and coupling
		if self.topology == TOPO_0D_GLOBAL:
			arr = st.Ring(self.n)
			g = st.AllToAll(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
		elif self.topology == TOPO_1D_CHAIN:
			arr = st.Chain(self.n)
			g = st.NearestNeighbor(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
		elif self.topology == TOPO_1D_RING:
			arr = st.Ring(self.n)
			g = st.NearestNeighbor(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
		elif self.topology == TOPO_1D_ENTRAINONE:
			arr = st.Chain(self.n)
			#g = st.
		elif self.topology == TOPO_1D_ENTRAINALL:
			arr = st.Ring(self.n)
			#g = st.
		elif self.topology == TOPO_2D_CUBIC_OPEN:
			arr = st.OpenCubic2D(self.nx, self.ny)
			g = st.CubicNearestNeighbor(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
		elif self.topology == TOPO_2D_CUBIC_PERIODIC:
			arr = st.PeriodicCubic2D(self.nx, self.ny)
			g = st.CubicNearestNeighbor(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
		elif self.topology == TOPO_2D_HEXAGONAL_OPEN:
			arr = st.OpenCubic2D(self.nx, self.ny)
			g = st.CubicHexagonal(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
		elif self.topology == TOPO_2D_HEXAGONAL_PERIODIC:
			arr = st.PeriodicCubic2D(self.nx, self.ny)
			g = st.CubicHexagonal(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
		elif self.topology == TOPO_2D_OCTAGONAL_OPEN:
			arr = st.OpenCubic2D(self.nx, self.ny)
			g = st.CubicOctagonal(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
		elif self.topology == TOPO_2D_OCTAGONAL_PERIODIC:
			arr = st.PeriodicCubic2D(self.nx, self.ny)
			g = st.CubicOctagonal(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
		else:
			raise Exception('Non-valid topology string')

		# Initialize singel pll
		pll = st.Pll(self.w, self.wc, self.v, self.fric, self.fric_omega)		# hand over PLL parameters

		# Initialize system
		pll_sys = st.PllSystem(pll, g)

		return pll_sys

	def get_states(self, pll_sys):
		# 0d global coupling
		if self.topology == TOPO_0D_GLOBAL:
			state_def = st.TwistDefinition(pll_sys, 0)
		# 1d twist state
		elif self.topology == TOPO_1D_RING:
			state_def = st.TwistDefinition(pll_sys, self.m)
		# 1d global sync state for non-periodic boundary conditions
		elif self.topology == TOPO_1D_CHAIN and self.m == 0:
			state_def = st.TwistDefinition(pll_sys, 0)
		# 1d Checkerboard states for non-periodic boundary conditions and m > 0
		elif self.topology == TOPO_1D_CHAIN:
			state_def = st.CheckerboardDefinition(pll_sys)
		# Global sync state for open 2d cubic lattice
		elif (self.topology == TOPO_2D_CUBIC_OPEN or self.topology == TOPO_2D_HEXAGONAL_OPEN or self.topology == TOPO_2D_OCTAGONAL_OPEN) and self.mx == 0 and self.my == 0:
			state_def = st.CubicTwistDefinition(pll_sys, 0, 0)
		# Checkerboard state for open cubic 2d lattice
		elif (self.topology == TOPO_2D_CUBIC_OPEN or self.topology == TOPO_2D_HEXAGONAL_OPEN or self.topology == TOPO_2D_OCTAGONAL_OPEN) and (self.mx > 0 or self.my > 0):
			state_def = st.CubicCheckerboardDefinition(pll_sys)
		# Twist states for periodic cubic 2d lattice
		elif (self.topology == TOPO_2D_CUBIC_PERIODIC or self.topology == TOPO_2D_HEXAGONAL_PERIODIC or self.topology == TOPO_2D_OCTAGONAL_PERIODIC):
			state_def = st.CubicTwistDefinition(pll_sys, self.mx, self.my)
		else:
			raise Exception('Interface does not support topology yet.')

		return state_def.get_states()

	def sweep(self):
		'''Performs sweep

		   Determines the possible globally synchronized states, their angular frequencies and their linear stability

		   Returns
		   -------
		   fsl  :  FlatStateList
				   flat list of the possible states
		'''
		# Set up sweep loop
		fsl = FlatStateList(sweep_factory=self)
		for i in range(len(self.values_sweep)):
			msg_str = 'Sweep value: %.3e' % self.values_sweep[i]
			# print msg_str + '\r',
			print(msg_str + '\r', end=" ")

			# Set new value for sweep variable
			self[self.key_sweep] = self.values_sweep[i]

			# Construct system
			pll_sys = self.init_system()

			# Get states
			s = self.get_states(pll_sys)
			fsl.add_states(s)

		return fsl

# ##############################################################################

class FlatStateList(object):
	'''Flat list of TwistStates'''
	def __init__(self, tsim=0.0, sweep_factory=None):
		self.states = []
		self.n = 0
		self.tsim = tsim
		self.sweep_factory = sweep_factory

	def add_states(self, s):
		'''Adds a single or a list of twist states to the list

		   Parameters
		   ----------
		   s : Twist
		   State or list of TwistStates
			   state or list of states that should be added
		'''
		if isinstance(s, st.SyncState):
			self.states.append(s)
			self.n = len(self.states)
		elif isinstance(s, list):
			for el in s:
				self.states.append(el)
			self.n = len(self.states)
		else:
			raise Exception('Non-valid object for storage in FlatStateList')

	def get_n(self):
		'''Returns an array of the number of oscillators of the states in the list'''
		if self.n > 0:
			x = np.zeros(self.n)
			for i in range(self.n):
				x[i] = self.states[i].sys.g.arr.get_n()
			return x
		else:
			return None

	def get_v(self):
		'''Returns an array of the number of oscillators of the states in the list'''
		if self.n > 0:
			x = np.zeros(self.n)
			for i in range(self.n):
				x[i] = self.states[i].sys.pll.v
			return x
		else:
			return None

	def get_fric(self):
		'''Returns an array of the number of oscillators of the states in the list'''
		if self.n > 0:
			x = np.zeros(self.n)
			for i in range(self.n):
				x[i] = self.states[i].sys.pll.fric
			return x
		else:
			return None

	def get_w(self, isRadians=True):
		'''Returns an array of the intrinsic frequencies of oscillators of the states in the list

		   Parameters
		   ----------
		   isRadians : bool
					   frequency is given in radians if True, otherwise in Hertz
		'''
		if isRadians:
			s = 1.0
		else:
			s = 1.0 / (2 * np.pi)

		if self.n > 0:
			x = np.zeros(self.n)
			for i in range(self.n):
				x[i] = s * self.states[i].sys.pll.w
			return x
		else:
			return None

	def get_k(self, isRadians=True):
		'''Returns an array of the coupling constants of the states in the list

		   Parameters
		   ----------
		   isRadians : bool
					   frequency is given in radians if True, otherwise in Hertz
		'''
		if isRadians:
			s = 1.0
		else:
			s = 1.0 / (2 * np.pi)

		if self.n > 0:
			x = np.zeros(self.n)
			for i in range(self.n):
				x[i] = s * self.states[i].sys.g.k
			return x
		else:
			return None

	def get_tau(self):
		'''Returns an array of the delay times of the states in the list'''
		if self.n > 0:
			x = np.zeros(self.n)
			for i in range(self.n):
				x[i] = self.states[i].sys.g.tau
			return x
		else:
			return None

	def get_m(self):
		'''Returns an array of the twist numbers of the states in the list'''
		return self.get_mx()

	def get_omega(self, isRadians=True):
		'''Returns an array of the global synchronization frequencies of the states in the list

		   Parameters
		   ----------
		   isRadians : bool
					   frequency is given in radians if True, otherwise in Hertz
		'''
		if isRadians:
			s = 1.0
		else:
			s = 1.0 / (2 * np.pi)

		if self.n > 0:
			x = np.zeros(self.n)
			for i in range(self.n):
				x[i] = s * self.states[i].omega
			return x
		else:
			return None

	def get_l(self):
		'''Returns an array of the complex linear stability exponent of the states in the list'''
		if self.n > 0:
			x = np.zeros(self.n, dtype=np.complex)
			for i in range(self.n):
				x[i] = self.states[i].get_stability()
			return x
		else:
			return None

	def get_wc(self, isRadians=True):
		'''Returns the low-pass filter cut-off frequency of the states in the list

		   Parameters
		   ----------
		   isRadians : bool
					   frequency is given in radians if True, otherwise in Hertz
		'''
		if isRadians:
			s = 1.0
		else:
			s = 1.0 / (2 * np.pi)

		if self.n > 0:
			x = np.zeros(self.n)
			for i in range(self.n):
				x[i] = s * self.states[i].sys.pll.wc
			return x
		else:
			return None

	def get_tsim(self):
		'''Returns an array of simulation time'''
		if self.n > 0:
			x = np.zeros(self.n)
			for i in range(self.n):
				re_lambda = np.real(self.states[i].get_stability())
				x[i] = 25.0 / np.abs(re_lambda)
			return x
		else:
			return None

	def get_nx(self):
		if self.n > 0:
			x = np.zeros(self.n)
			for i in range(self.n):
				s = self.states[i]
				if isinstance(s.sys.g.arr, st.Linear):
					x[i] = s.sys.g.arr.get_n()
				elif isinstance(s.sys.g.arr, st.Cubic2D):
					x[i] = s.sys.g.arr.nx
				else:
					raise Exception('Topology not yet supported')
			return x
		else:
			return None

	def get_ny(self):
		if self.n > 0:
			x = np.zeros(self.n)
			for i in range(self.n):
				s = self.states[i]
				if isinstance(s.sys.g.arr, st.Linear):
					x[i] = 1
				elif isinstance(s.sys.g.arr, st.Cubic2D):
					x[i] = s.sys.g.arr.ny
				else:
					raise Exception('Topology not yet supported')
			return x
		else:
			return None

	def get_mx(self):
		if self.n > 0:
			x = np.zeros(self.n)
			for i in range(self.n):
				s = self.states[i]
				if isinstance(s, st.Twist):
					x[i] = s.state_def.m
				elif isinstance(s, st.Checkerboard):
					#x[i] = s.sys.g.arr.get_n() / 2
					x[i] = self.sweep_factory.m     # Required by Lucas' code
				elif isinstance(s, st.CubicTwist):
					x[i] = s.state_def.mx
				elif isinstance(s, st.CubicCheckerboard):
					#x[i] = s.sys.g.arr.nx / 2
					x[i] = self.sweep_factory.mx    # Required by Lucas' code
				else:
				   raise Exception('State not supported so far.')
			return x
		else:
			return None

	def get_my(self):
		if self.n > 0:
			x = np.zeros(self.n)
			for i in range(self.n):
				s = self.states[i]
				if isinstance(s, st.Twist):
					x[i] = -999
				elif isinstance(s, st.Checkerboard):
					x[i] = -999
				elif isinstance(s, st.CubicTwist):
					x[i] = s.state_def.my
				elif isinstance(s, st.CubicCheckerboard):
					# x[i] = s.sys.g.arr.ny / 2
					x[i] = self.sweep_factory.my        # Required by Lucas' code
				else:
					raise Exception('State not supported so far.')
			return x
		else:
			return None

	def get_phiConf(self):
		'''Returns an array of the phase configurations of the states in the list'''
		if self.n > 0:
			x = []
			for i in range(self.n):
				x.append( self.states[i].state_def.get_phi() )
			return np.array(x)
		else:
			return None

	def get_coup_fct(self):
		'''Returns the coupling function'''
		h = self.states[0].sys.g.func.get_derivative()
		return h

	def get_parameter_matrix(self, isRadians=True):
		'''Returns a matrix of the numeric parameters the states in the list

		   Parameters
		   ----------
		   isRadians : bool
					   frequency is given in radians if True, otherwise in Hertz
		'''
		if self.n > 0:
			phi_config_vec_len = len( self.get_phiConf()[0] ); #print('Lenght vector phi-configuation:', phi_config_vec_len)
			x = np.zeros((self.n, 14+phi_config_vec_len))
			h = self.get_coup_fct()
			print('coupling function h=',h)
			x[:, 0] = self.get_w(isRadians=isRadians)
			x[:, 1] = self.get_k(isRadians=isRadians)
			x[:, 2] = self.get_wc(isRadians=isRadians)
			x[:, 3] = self.get_tau()
			x[:, 4] = self.get_omega(isRadians=isRadians)
			x[:, 5] = np.real(self.get_l())
			x[:, 6] = np.imag(self.get_l())
			x[:, 7] = self.get_tsim()
			x[:, 8] = self.get_nx()
			x[:, 9] = self.get_ny()
			x[:, 10] = self.get_mx()
			x[:, 11] = self.get_my()
			x[:, 12] = self.get_v()
			# NOTE: steady state loop gain is returned as specified by the function, however inside the coupling function it needs to be set to redHz
			x[:, 13] = ( self.get_k(isRadians=isRadians) / self.get_v() ) * h( self.get_omega(isRadians=True)*self.get_tau() )
			#print('from get_phiConf: ', self.get_phiConf()[0])
			x[:, 14:14+phi_config_vec_len] = self.get_phiConf()[0]
			#print('phi configuration x[:, 13:%i]: '%(13+phi_config_vec_len), x[:, 13:])
			return x
		else:
			return None

	def get_parameter_matrix_nostab(self, isRadians=True):
		'''Returns a matrix of the numeric parameters the states in the list

		   Parameters
		   ----------
		   isRadians : bool
					   frequency is given in radians if True, otherwise in Hertz
		'''
		if self.n > 0:
			phi_config_vec_len = len( self.get_phiConf()[0] ); #print('Lenght vector phi-configuation:', phi_config_vec_len)
			x = np.zeros((self.n, 13+phi_config_vec_len))
			x[:, 0] = self.get_w(isRadians=isRadians)
			x[:, 1] = self.get_k(isRadians=isRadians)
			x[:, 2] = self.get_wc(isRadians=isRadians)
			x[:, 3] = self.get_tau()
			x[:, 4] = self.get_omega(isRadians=isRadians)
			x[:, 5] = 1
			x[:, 6] = 1
			x[:, 7] = 100
			x[:, 8] = self.get_nx()
			x[:, 9] = self.get_ny()
			x[:, 10] = self.get_mx()
			x[:, 11] = self.get_my()
			x[:, 12] = self.get_v()
			#print('from get_phiConf: ', self.get_phiConf()[0])
			x[:, 13:13+phi_config_vec_len] = self.get_phiConf()[0]
			#print('phi configuration x[:, 13:%i]: '%(13+phi_config_vec_len), x[:, 13:])
			return x
		else:
			return None
