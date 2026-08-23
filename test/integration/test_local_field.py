"""Integration tests for the local mesoscale field channel of sim_pll.field_lib.

The load-bearing test here is the open-loop equivalence test: with the field coupling
switched off the trajectory must be bit-identical to sim_lib.simulate_headless. The
field channel is meant to be an addition to the model, not a perturbation of it, and
every claim about what changes at finite epsilon depends on the epsilon = 0 baseline
being the unmodified system.
"""
import numpy as np
import pytest

from sim_pll import dicts_factory, sim_lib
from sim_pll import field_lib as fl


NX = NY = 4
N = NX * NY
F0_HZ = 1.0
K_HZ = 0.15
FC_HZ = 0.25
SAMPLE_F_HZ = 125.0
DELAY_S = 8.0 / SAMPLE_F_HZ
SIM_PERIODS = 30.0


def build_lattice_dicts(noise_var_vco=0.0, sim_periods=SIM_PERIODS, transmission_delay_s=DELAY_S,
						intrinsic_freqs_hz=F0_HZ):
	"""Configuration for a periodic NX x NY lattice with nearest-neighbour synaptic coupling."""
	adjacency = fl.lattice_adjacency(NX, NY, periodic=True, radius=1.0)
	dict_pll, dict_net, dict_algo = dicts_factory.generate_dicts(
		n=N,
		topology='custom',
		adjacency_matrix=adjacency,
		intrinsic_freqs_hz=intrinsic_freqs_hz,
		coupling_strength_hz=K_HZ,
		cutoff_freqs_hz=FC_HZ,
		transmission_delay_s=transmission_delay_s,
		sim_time_periods=sim_periods,
		sample_freq_hz=SAMPLE_F_HZ,
		noise_var_vco=noise_var_vco,
		coupling_function='sin',
	)
	dict_net['Nx'] = NX
	dict_net['Ny'] = NY
	return dict_pll, dict_net, dict_algo


# ##############################################################################
# the regression test
# ##############################################################################

@pytest.mark.parametrize('noise_var_vco', [0.0, 1e-6])
def test_zero_epsilon_reproduces_simulate_headless_exactly(noise_var_vco):
	"""Switching the field channel off must leave the existing dynamics untouched.

	Run deterministically and with VCO noise; in both cases the phase trajectory of
	the new driver must agree with the unmodified one to machine precision for the
	same seed.
	"""
	dict_pll, dict_net, dict_algo = build_lattice_dicts(noise_var_vco)
	baseline = sim_lib.simulate_headless(dict_net, dict_pll, dict_algo, seed=1234)

	dict_pll, dict_net, dict_algo = build_lattice_dicts(noise_var_vco)
	kernel = fl.build_kernel(NX, NY, kind='dipole', sigma=1.5)
	off = fl.simulate_with_local_field(
		dict_net, dict_pll, dict_algo,
		field_coupling=fl.LocalFieldCoupling(kernel, epsilon=0.0),
		seed=1234)

	np.testing.assert_allclose(off['dict_data']['phi'], baseline['dict_data']['phi'],
							   rtol=0.0, atol=0.0)


def test_no_field_coupling_object_also_reproduces_the_baseline():
	dict_pll, dict_net, dict_algo = build_lattice_dicts()
	baseline = sim_lib.simulate_headless(dict_net, dict_pll, dict_algo, seed=99)

	dict_pll, dict_net, dict_algo = build_lattice_dicts()
	off = fl.simulate_with_local_field(dict_net, dict_pll, dict_algo,
									   field_coupling=None, seed=99)
	np.testing.assert_allclose(off['dict_data']['phi'], baseline['dict_data']['phi'],
							   rtol=0.0, atol=0.0)


# ##############################################################################
# the field channel does something
# ##############################################################################

def test_finite_epsilon_changes_the_trajectory():
	"""Sanity: if the field never altered anything, every later result would be vacuous."""
	dict_pll, dict_net, dict_algo = build_lattice_dicts(intrinsic_freqs_hz=list(
		np.linspace(0.9, 1.1, N)))
	baseline = sim_lib.simulate_headless(dict_net, dict_pll, dict_algo, seed=7)

	dict_pll, dict_net, dict_algo = build_lattice_dicts(intrinsic_freqs_hz=list(
		np.linspace(0.9, 1.1, N)))
	kernel = fl.build_kernel(NX, NY, kind='dipole', sigma=1.5)
	on = fl.simulate_with_local_field(
		dict_net, dict_pll, dict_algo,
		field_coupling=fl.LocalFieldCoupling(kernel, epsilon=0.5), seed=7)

	assert not np.allclose(on['dict_data']['phi'], baseline['dict_data']['phi'])


def test_field_channel_increases_coherence_of_a_heterogeneous_lattice():
	"""An attractive field must not decrease global coherence in a detuned lattice."""
	kernel = fl.build_kernel(NX, NY, kind='dipole', sigma=1.5)
	# a detuned lattice that the synaptic channel alone cannot lock
	freqs = list(np.linspace(0.85, 1.15, N))

	coherence = {}
	for epsilon in (0.0, 1.0):
		dict_pll, dict_net, dict_algo = build_lattice_dicts(sim_periods=60.0,
															intrinsic_freqs_hz=freqs)
		result = fl.simulate_with_local_field(
			dict_net, dict_pll, dict_algo,
			field_coupling=fl.LocalFieldCoupling(kernel, epsilon=epsilon), seed=3)
		phi = result['dict_data']['phi']
		tail = phi[int(0.6 * len(phi)):, :]
		coherence[epsilon] = float(np.mean(np.abs(np.mean(np.exp(1j * tail), axis=1))))

	assert coherence[1.0] > coherence[0.0]


# ##############################################################################
# recorded diagnostics
# ##############################################################################

def test_field_is_recorded_with_the_right_shape_even_when_inactive():
	dict_pll, dict_net, dict_algo = build_lattice_dicts()
	result = fl.simulate_with_local_field(dict_net, dict_pll, dict_algo,
										  field_coupling=None, seed=5)
	data = result['dict_data']
	assert data['z_local'].shape == data['phi'].shape
	np.testing.assert_allclose(data['r_local'], np.abs(data['z_local']))
	assert data['epsilon'] == 0.0


def test_store_field_can_be_switched_off():
	dict_pll, dict_net, dict_algo = build_lattice_dicts()
	result = fl.simulate_with_local_field(dict_net, dict_pll, dict_algo,
										  field_coupling=None, seed=5, store_field=False)
	assert 'z_local' not in result['dict_data']


def test_mismatched_kernel_size_is_rejected_by_the_driver():
	dict_pll, dict_net, dict_algo = build_lattice_dicts()
	wrong_kernel = fl.build_kernel(3, 3, sigma=1.0)
	with pytest.raises(ValueError, match='kernel is sized for'):
		fl.simulate_with_local_field(
			dict_net, dict_pll, dict_algo,
			field_coupling=fl.LocalFieldCoupling(wrong_kernel, epsilon=0.1), seed=1)
