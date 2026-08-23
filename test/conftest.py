"""Shared pytest configuration for the sim_pll test suite."""
import matplotlib
import pytest

matplotlib.use("Agg")  # headless backend; setup_topology() draws the graph as a side effect


@pytest.fixture
def two_channel_dicts():
	"""Factory for a small periodic lattice configured as in the field study.

	Returns a callable rather than a dict triple. ``check_dicts_consistency`` rewrites
	the dictionaries it is given, including the simulation length, so a triple that has
	been through one run cannot be reused for a second: the two runs come back with
	different array lengths. Every call here builds a fresh triple.

	8x8 oscillators, nearest-neighbour delayed coupling, one loop filter per node, and no
	oscillator noise, so that the bit-identity assertions mean what they say.
	"""
	import contextlib
	import io as _io

	import numpy as np
	from sim_pll import dicts_factory, field_lib as fl

	def build(sim_time_periods=40.0):
		nx = ny = 8
		n = nx * ny
		adjacency = fl.lattice_adjacency(nx, ny, periodic=True, radius=1.0)
		rng = np.random.default_rng(1234)
		with contextlib.redirect_stdout(_io.StringIO()):
			dict_pll, dict_net, dict_algo = dicts_factory.generate_dicts(
				n=n,
				topology='custom',
				adjacency_matrix=adjacency,
				intrinsic_freqs_hz=list(1.0 + 0.05 * rng.standard_normal(n)),
				coupling_strength_hz=0.15,
				cutoff_freqs_hz=0.25,
				transmission_delay_s=8 / 125.0,
				sim_time_periods=sim_time_periods,
				sample_freq_hz=125.0,
				noise_var_vco=0.0,
				coupling_function='sin',
				phase_perturbation=list(rng.uniform(-np.pi, np.pi, n)),
			)
		dict_net['Nx'], dict_net['Ny'] = nx, ny
		return dict_pll, dict_net, dict_algo

	return build
