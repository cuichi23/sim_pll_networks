"""Parameterized factory for the three configuration dictionaries of sim_pll.

This module replaces the copy-and-modify pattern of the root-level ``dicts_lib_*.py``
variants with a single function. It produces consistent, minimal configurations for
programmatic use (parameter sweeps, the reservoir-computing adapter, tests).

Conventions
-----------
- Frequencies are given in Hz, delays in seconds, phases in radians.
- ``sim_time_periods`` follows the dicts_lib convention: the simulation time is
  given in multiples of the mean intrinsic period and converted to seconds by
  ``check_dicts_lib.check_dicts_consistency`` (called by ``sim_lib.simulate_headless``).
- For ``topology='custom'`` an ``adjacency_matrix`` is required where entry
  ``[k][l] == 1`` means oscillator k receives the signal of oscillator l.
"""
import numpy as np

from sim_pll import coupling_fct_lib as coupfct


#: coupling-function presets mapping to consistent (h, h', VCO output, VCO type) choices,
#: chosen such that check_dicts_consistency never asks for interactive confirmation
COUPLING_PRESETS = {
	'sin': {
		'coup_fct_sig': coupfct.sine,
		'derivative_coup_fct': coupfct.cosine,
		'vco_out_sig': coupfct.sine,
		'typeVCOsig': 'analogHF',
		'PSD_from_signal': coupfct.sine,
	},
	'cos': {
		'coup_fct_sig': coupfct.cosine,
		'derivative_coup_fct': coupfct.neg_sine,
		'vco_out_sig': coupfct.sine,
		'typeVCOsig': 'analogHF',
		'PSD_from_signal': coupfct.sine,
	},
	'triang': {
		'coup_fct_sig': coupfct.triangular,
		'derivative_coup_fct': coupfct.deriv_triangular,
		'vco_out_sig': coupfct.square_wave_symm_zero,
		'typeVCOsig': 'digitalHF',
		'PSD_from_signal': coupfct.square_wave,
	},
}


def generate_dicts(n: int = 3,
				   topology: str = 'global',
				   adjacency_matrix=None,
				   intrinsic_freqs_hz=1.0,
				   coupling_strength_hz=0.1,
				   coupling_gain_matrix=1,
				   cutoff_freqs_hz=0.5,
				   transmission_delay_s=0.0,
				   feedback_delay_s=0.0,
				   divider: int = 1,
				   coupling_function: str = 'sin',
				   sim_time_periods: float = 100.0,
				   sample_freq_hz: float = 125.0,
				   noise_var_vco: float = 0.0,
				   type_of_history: str = 'freeRunning',
				   order_lf: int = 1,
				   phase_perturbation=None,
				   sync_freq_hz: float = None):
	"""Build (dict_pll, dict_net, dict_algo) for a single simulation run.

	Parameters
	----------
	n : int
		Number of oscillators (network is n x 1).
	topology : str
		Any topology accepted by ``setup.setup_topology``, e.g. 'global', 'ring',
		'chain', or 'custom' (requires ``adjacency_matrix``).
	adjacency_matrix : array_like of shape (n, n), optional
		Binary directed coupling for ``topology='custom'``; entry [k][l] == 1 means
		oscillator k receives the signal of oscillator l.
	intrinsic_freqs_hz : float or list of float
		Intrinsic VCO frequencies in Hz (scalar is broadcast to all oscillators).
	coupling_strength_hz : float or list of float
		Coupling strength K in Hz per oscillator.
	coupling_gain_matrix : float or ndarray of shape (n, n)
		Feed-forward gains G_kl of the phase-detector inputs; a matrix realizes
		per-link coupling strengths K_kl = coupK_k * G_kl.
	cutoff_freqs_hz : float or list of float
		Low-pass filter cut-off frequencies in Hz.
	transmission_delay_s : float, list, or matrix
		Signal transmission delay(s) in seconds.
	feedback_delay_s : float
		Feedback path delay in seconds.
	divider : int
		Frequency division factor of the feedback divider.
	coupling_function : str
		One of 'sin', 'cos', 'triang' (see ``COUPLING_PRESETS``).
	sim_time_periods : float
		Simulation time in multiples of the mean intrinsic period.
	sample_freq_hz : float
		Sampling frequency in Hz; the integration time step is dt = 1/sample_freq_hz.
	noise_var_vco : float
		Variance of the Gaussian white noise of the VCO (0 for deterministic runs).
	type_of_history : str
		'freeRunning' or 'syncState' history initialization.
	order_lf : int
		Order of the low-pass loop filter (1 or 2).
	phase_perturbation : list of float, optional
		Initial delta phase perturbation per oscillator (defaults to zero).
	sync_freq_hz : float, optional
		Frequency of the synchronized state in Hz (defaults to the mean intrinsic
		frequency); used for 'syncState' history and time averaging.

	Returns
	-------
	tuple of (dict, dict, dict)
		(dict_pll, dict_net, dict_algo), unchecked; pass them through
		``sim_lib.simulate_headless`` (which runs the consistency check) or run
		``check_dicts_lib.check_dicts_consistency`` manually before simulating.
	"""
	if coupling_function not in COUPLING_PRESETS:
		raise ValueError("coupling_function must be one of %s, got %r."
						 % (sorted(COUPLING_PRESETS), coupling_function))
	if topology == 'custom' and adjacency_matrix is None:
		raise ValueError("topology='custom' requires an adjacency_matrix.")

	preset = COUPLING_PRESETS[coupling_function]
	intrinsic_freqs = _broadcast_to_list(intrinsic_freqs_hz, n)
	if sync_freq_hz is None:
		sync_freq_hz = float(np.mean(intrinsic_freqs))
	if phase_perturbation is None:
		phase_perturbation = [0.0 for _ in range(n)]

	dict_net = {
		'Nx': n,
		'Ny': 1,
		'mx': 0,
		'my': -999,
		'topology': topology,
		'normalize_mutual_coupling_by_inputs': True,
		'Tsim': sim_time_periods,
		'computeFreqAndStab': False,
		'phi_array_mult_tau': 1,
		'phiPerturb': list(phase_perturbation),
		'phiPerturbRot': [],
		'phiInitConfig': [],
		'freq_beacons': 0.0,
		'special_case': 'False',
		'typeOfTimeDependency': 'linear',
		'min_max_rate_timeDepPara': [0.0, 1.0, 1.0],
		'plot_topology_diagnostics': False,
	}
	if adjacency_matrix is not None:
		dict_net['adjacency_matrix'] = np.asarray(adjacency_matrix)

	dict_pll = {
		'intrF': intrinsic_freqs,
		'syncF': sync_freq_hz,
		'coupK': _broadcast_to_list(coupling_strength_hz, n),
		'gPDin': coupling_gain_matrix,
		'gPDin_symmetric': False,
		'cutFc': _broadcast_to_list(cutoff_freqs_hz, n),
		'orderLF': order_lf,
		'div': divider,
		'friction_coefficient': 1,
		'fric_coeff_PRE_vs_PRR': 'PRE',
		'noiseVarVCO': noise_var_vco,
		'feedback_delay': feedback_delay_s,
		'feedback_delay_var': None,
		'transmission_delay': transmission_delay_s,
		'transmission_delay_var': None,
		'distribution_for_delays': None,
		'coup_fct_sig': preset['coup_fct_sig'],
		'derivative_coup_fct': preset['derivative_coup_fct'],
		'branch_of_inverse_coupling_fct_if_applies': 'negative',
		'includeCompHF': False,
		'vco_out_sig': preset['vco_out_sig'],
		'typeVCOsig': preset['typeVCOsig'],
		'responseVCO': 'linear',
		'antenna': False,
		'posX': 0,
		'posY': 0,
		'posZ': 0,
		'initAntennaState': 0,
		'antenna_sig': coupfct.sine,
		'coup_fct_phase_shift': 0,
		'extra_coup_sig': None,
		'shil_generation_through_filter': False,
		'coupStr_2ndHarm': 0,
		'typeOfHist': type_of_history,
		'sampleF': sample_freq_hz,
		'sampleFplot': 1,
		'treshold_maxT_to_plot': 1E6,
		'percentPeriodsAverage': 0.25,
		'PSD_freq_resolution': 1E-5,
		'PSD_from_signal': preset['PSD_from_signal'],
		'signal_propagation_speed': 0.0,
		'space_dimensions_xyz': [10, 10, 10],
	}

	dict_algo = {
		'parameter_space_sweeps': 'single',
		'paramDiscretization': [2, 2],
		'param_id_0': 'None',
		'min_max_range_parameter_0': [0.95, 1.05],
		'param_id_1': 'None',
		'min_max_range_parameter_1': [0.95, 1.05],
		'store_ctrl_and_clock': False,
		'store_phases_tau_array': False,
		'number_of_processes_in_multisim': 1,
	}

	return dict_pll, dict_net, dict_algo


def _broadcast_to_list(value, n: int) -> list:
	"""Return value as a list of length n; scalars are repeated, sequences validated."""
	if isinstance(value, (list, tuple, np.ndarray)):
		values = list(value)
		if len(values) != n:
			raise ValueError("Expected %i per-oscillator values, got %i." % (n, len(values)))
		return [float(v) for v in values]
	return [float(value) for _ in range(n)]
