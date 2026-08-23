"""Unit tests for sim_pll.field_lib, the local mesoscale field channel.

The tests that matter physically are the scaling tests. The whole justification for
identifying the local field potential with a local order parameter is that both scale
as sqrt(N) for independent sources and as N for synchronised ones; if that fails, the
model is not a model of a field.
"""
import numpy as np
import pytest

from sim_pll import field_lib as fl


# ##############################################################################
# geometry and kernels
# ##############################################################################

def test_lattice_positions_match_reshape_convention():
	"""Index k must sit at row k // nx, column k % nx, as np.reshape(phi, (ny, nx)) assumes."""
	positions = fl.lattice_positions(nx=4, ny=3)
	assert positions.shape == (12, 2)
	np.testing.assert_array_equal(positions[0], [0.0, 0.0])
	np.testing.assert_array_equal(positions[3], [3.0, 0.0])
	np.testing.assert_array_equal(positions[4], [0.0, 1.0])
	np.testing.assert_array_equal(positions[11], [3.0, 2.0])


def test_pairwise_distances_are_symmetric_with_zero_diagonal():
	distances = fl.pairwise_distances(nx=4, ny=4)
	np.testing.assert_allclose(distances, distances.T)
	np.testing.assert_allclose(np.diag(distances), 0.0)


def test_periodic_boundaries_wrap_the_lattice():
	"""On a torus the two ends of a row are nearest neighbours, not far apart."""
	periodic = fl.pairwise_distances(nx=6, ny=1, periodic=True)
	open_bc = fl.pairwise_distances(nx=6, ny=1, periodic=False)
	assert periodic[0, 5] == pytest.approx(1.0)
	assert open_bc[0, 5] == pytest.approx(5.0)


def test_kernel_rows_are_normalised_and_non_negative():
	kernel = fl.build_kernel(nx=5, ny=5, kind='dipole', sigma=2.0)
	assert np.all(kernel >= 0.0)
	np.testing.assert_allclose(kernel.sum(axis=1), 1.0)


def test_dipole_kernel_decays_with_distance():
	"""The 1/(1+(r/sigma)^2) form must be monotonically decreasing in r."""
	kernel = fl.build_kernel(nx=9, ny=1, kind='dipole', sigma=2.0, normalize_rows=False)
	row = kernel[4]
	assert row[4] > row[3] > row[2] > row[1] > row[0]


def test_nearest_neighbour_kernel_selects_four_neighbours_on_a_torus():
	kernel = fl.build_kernel(nx=5, ny=5, kind='nearest', include_self=False,
							 normalize_rows=False, periodic=True)
	assert np.all(kernel.sum(axis=1) == 4.0)


def test_cutoff_radius_zeroes_distant_weights():
	kernel = fl.build_kernel(nx=9, ny=1, kind='gaussian', sigma=3.0,
							 cutoff_radius=2.0, normalize_rows=False)
	distances = fl.pairwise_distances(nx=9, ny=1)
	assert np.all(kernel[distances > 2.0] == 0.0)
	assert np.all(kernel[distances <= 2.0] > 0.0)


def test_unknown_kernel_kind_is_rejected():
	with pytest.raises(ValueError, match='unknown kernel kind'):
		fl.build_kernel(nx=3, ny=3, kind='banana')


def test_non_positive_sigma_is_rejected():
	with pytest.raises(ValueError, match='sigma must be positive'):
		fl.build_kernel(nx=3, ny=3, sigma=0.0)


def test_all_zero_kernel_row_is_rejected():
	with pytest.raises(ValueError, match='all-zero row'):
		fl.build_kernel(nx=9, ny=1, kind='gaussian', sigma=1.0,
						cutoff_radius=0.5, include_self=False)


def test_lattice_adjacency_gives_four_neighbours_and_no_self_coupling():
	adjacency = fl.lattice_adjacency(nx=5, ny=5, periodic=True, radius=1.0)
	np.testing.assert_allclose(np.diag(adjacency), 0.0)
	np.testing.assert_allclose(adjacency.sum(axis=1), 4.0)
	np.testing.assert_allclose(adjacency, adjacency.T)


# ##############################################################################
# the order parameter
# ##############################################################################

def test_identical_phases_give_unit_coherence():
	kernel = fl.build_kernel(nx=6, ny=6, sigma=2.0)
	z = fl.local_order_parameter(np.full(36, 0.7), kernel)
	np.testing.assert_allclose(np.abs(z), 1.0)
	np.testing.assert_allclose(np.angle(z), 0.7)


def test_antiphase_pair_cancels():
	"""Two opposed phases with equal weight must cancel exactly."""
	kernel = np.array([[0.5, 0.5], [0.5, 0.5]])
	z = fl.local_order_parameter(np.array([0.0, np.pi]), kernel)
	np.testing.assert_allclose(np.abs(z), 0.0, atol=1e-12)


def test_order_parameter_accepts_a_time_series():
	kernel = fl.build_kernel(nx=4, ny=4, sigma=1.5)
	phi = np.zeros((10, 16))
	z = fl.local_order_parameter(phi, kernel)
	assert z.shape == (10, 16)


def test_mismatched_kernel_size_is_rejected():
	kernel = fl.build_kernel(nx=3, ny=3, sigma=1.0)
	with pytest.raises(ValueError, match='oscillators'):
		fl.local_order_parameter(np.zeros(16), kernel)


def test_incoherent_field_scales_as_sqrt_n_and_coherent_field_scales_as_n():
	"""The claim the whole field analogy rests on.

	An unnormalised weighted sum over N sources grows as sqrt(N) when the phases are
	independent and as N when they are locked. This is the same scaling that makes a
	local field potential a coherence measure, so it must hold here numerically.
	"""
	rng = np.random.default_rng(20260822)
	sizes = np.array([64, 256, 1024, 4096])
	incoherent = []
	coherent = []
	for n in sizes:
		flat_kernel = np.ones((1, n))
		samples = [np.abs(fl.local_order_parameter(rng.uniform(0.0, 2.0 * np.pi, n), flat_kernel))[0]
				   for _ in range(200)]
		incoherent.append(np.mean(samples))
		coherent.append(np.abs(fl.local_order_parameter(np.zeros(n), flat_kernel))[0])

	# fitted exponents of amplitude versus N
	incoherent_exponent = np.polyfit(np.log(sizes), np.log(incoherent), 1)[0]
	coherent_exponent = np.polyfit(np.log(sizes), np.log(coherent), 1)[0]
	assert incoherent_exponent == pytest.approx(0.5, abs=0.05)
	assert coherent_exponent == pytest.approx(1.0, abs=1e-6)


def test_normalised_coherence_of_random_phases_falls_as_one_over_sqrt_n():
	rng = np.random.default_rng(7)
	for n in (256, 1024):
		flat_kernel = np.ones((1, n)) / n
		samples = [np.abs(fl.local_order_parameter(rng.uniform(0.0, 2.0 * np.pi, n), flat_kernel))[0]
				   for _ in range(200)]
		assert np.mean(samples) == pytest.approx(np.sqrt(np.pi) / (2.0 * np.sqrt(n)), rel=0.15)


def test_local_field_keeps_amplitude_information_that_normalisation_discards():
	"""Unnormalised field distinguishes population sizes; normalised coherence does not."""
	small = fl.build_kernel(nx=4, ny=1, kind='nearest', normalize_rows=False, periodic=False)
	phi = np.zeros(4)
	field = fl.local_field(phi, small)
	coherence = np.abs(fl.local_order_parameter(phi, fl.build_kernel(
		nx=4, ny=1, kind='nearest', normalize_rows=True, periodic=False)))
	assert np.max(field) > 1.0
	np.testing.assert_allclose(coherence, 1.0)


def test_local_field_respects_source_amplitudes():
	kernel = np.eye(3)
	field = fl.local_field(np.zeros(3), kernel, amplitudes=np.array([1.0, 2.0, 3.0]))
	np.testing.assert_allclose(field, [1.0, 2.0, 3.0])


# ##############################################################################
# statistics
# ##############################################################################

def test_spatial_spread_separates_coexistence_from_uniform_states():
	"""A chimera-like split state must show higher spatial spread of R than either
	a fully synchronised or a fully incoherent state."""
	rng = np.random.default_rng(3)
	nx = ny = 12
	kernel = fl.build_kernel(nx, ny, sigma=1.5)

	synchronised = np.zeros((1, nx * ny))
	incoherent = rng.uniform(0.0, 2.0 * np.pi, (1, nx * ny))
	split = incoherent.copy()
	grid = split.reshape(1, ny, nx)
	grid[:, :, : nx // 2] = 0.0
	split = grid.reshape(1, nx * ny)

	spread_sync = fl.coherence_statistics(synchronised, kernel)['mean_spatial_std']
	spread_incoherent = fl.coherence_statistics(incoherent, kernel)['mean_spatial_std']
	spread_split = fl.coherence_statistics(split, kernel)['mean_spatial_std']

	assert spread_split > spread_sync
	assert spread_split > spread_incoherent


def test_coherence_statistics_reports_expected_keys_and_shapes():
	kernel = fl.build_kernel(nx=4, ny=4, sigma=1.5)
	stats = fl.coherence_statistics(np.zeros((7, 16)), kernel)
	assert stats['r_local'].shape == (7, 16)
	assert stats['r_global'].shape == (7,)
	assert stats['mean_r_global'] == pytest.approx(1.0)


def test_correlation_length_is_short_for_random_and_long_for_synchronised():
	rng = np.random.default_rng(11)
	nx = ny = 16
	random_length = fl.spatial_correlation_length(rng.uniform(0.0, 2.0 * np.pi, nx * ny), nx, ny)
	sync_length = fl.spatial_correlation_length(np.zeros(nx * ny), nx, ny)
	assert random_length <= 2.0
	assert np.isnan(sync_length)


# ##############################################################################
# the coupling channel
# ##############################################################################

def test_field_drive_vanishes_for_a_fully_synchronised_state():
	"""Nothing to correct when everyone already agrees; the field must exert no torque."""
	kernel = fl.build_kernel(nx=6, ny=6, sigma=2.0)
	coupling = fl.LocalFieldCoupling(kernel, epsilon=0.5)
	np.testing.assert_allclose(coupling.drive(np.full(36, 1.3)), 0.0, atol=1e-12)


def test_field_drive_pulls_a_lagging_oscillator_forward():
	"""An oscillator behind its local mean must receive a positive drive, and vice versa."""
	kernel = np.array([[0.5, 0.5], [0.5, 0.5]])
	coupling = fl.LocalFieldCoupling(kernel, epsilon=1.0)
	drive = coupling.drive(np.array([-0.4, 0.4]))
	assert drive[0] > 0.0
	assert drive[1] < 0.0
	assert drive[0] == pytest.approx(-drive[1])


def test_field_drive_scales_linearly_with_epsilon():
	kernel = fl.build_kernel(nx=5, ny=5, sigma=2.0)
	rng = np.random.default_rng(5)
	phi = rng.uniform(0.0, 2.0 * np.pi, 25)
	weak = fl.LocalFieldCoupling(kernel, epsilon=0.1).drive(phi)
	strong = fl.LocalFieldCoupling(kernel, epsilon=0.4).drive(phi)
	np.testing.assert_allclose(strong, 4.0 * weak, rtol=1e-12)


def test_zero_epsilon_gives_no_drive_and_no_gating():
	kernel = fl.build_kernel(nx=4, ny=4, sigma=1.5)
	coupling = fl.LocalFieldCoupling(kernel, epsilon=0.0)
	rng = np.random.default_rng(1)
	phi = rng.uniform(0.0, 2.0 * np.pi, 16)
	assert not coupling.is_active
	np.testing.assert_allclose(coupling.drive(phi), 0.0)
	np.testing.assert_allclose(coupling.gate(phi), 1.0)


def test_gating_mode_suppresses_the_synaptic_channel_where_coherence_is_high():
	kernel = fl.build_kernel(nx=6, ny=6, sigma=2.0)
	coupling = fl.LocalFieldCoupling(kernel, epsilon=0.8, mode='gating', gate_depth=1.0)
	gate_coherent = coupling.gate(np.zeros(36))
	rng = np.random.default_rng(2)
	gate_incoherent = coupling.gate(rng.uniform(0.0, 2.0 * np.pi, 36))
	np.testing.assert_allclose(gate_coherent, 1.0 - 0.8, atol=1e-12)
	assert np.mean(gate_incoherent) > np.mean(gate_coherent)


def test_gating_mode_contributes_no_additive_drive():
	kernel = fl.build_kernel(nx=4, ny=4, sigma=1.5)
	coupling = fl.LocalFieldCoupling(kernel, epsilon=0.5, mode='gating')
	np.testing.assert_allclose(coupling.drive(np.linspace(0.0, 3.0, 16)), 0.0)


def test_non_square_kernel_is_rejected():
	with pytest.raises(ValueError, match='square matrix'):
		fl.LocalFieldCoupling(np.ones((3, 4)), epsilon=0.1)


def test_unknown_mode_is_rejected():
	with pytest.raises(ValueError, match="'additive' or 'gating'"):
		fl.LocalFieldCoupling(np.eye(4), epsilon=0.1, mode='telepathy')


# ##############################################################################
# attractor classification, baselines, and the two-band gate
# ##############################################################################

def test_participation_number_counts_uniform_weights():
	"""A flat kernel over n sources must have participation number exactly n."""
	assert fl.participation_number(np.ones((4, 4)) / 4.0) == pytest.approx(4.0)


def test_participation_number_of_dipole_kernel_grows_with_the_lattice():
	"""A 1/d^2 tail has logarithmically divergent mass in 2D, so sigma sets no range.

	This is why the dipole kernel must not be described as local: unlike a Gaussian, its
	participation number does not saturate as the lattice grows.
	"""
	dipole = [fl.participation_number(fl.build_kernel(n, n, 'dipole', sigma=1.5))
			  for n in (6, 8, 12, 16)]
	gaussian = [fl.participation_number(fl.build_kernel(n, n, 'gaussian', sigma=1.5))
				for n in (6, 8, 12, 16)]
	assert dipole == sorted(dipole) and dipole[-1] > 2.0 * dipole[0]
	assert gaussian[-1] == pytest.approx(gaussian[-2], rel=0.02)


def test_classification_recognises_the_in_phase_state():
	assert fl.pattern_order_parameters(np.zeros(64), 8, 8)['state'] == 'in-phase'


def test_classification_recognises_the_chequerboard_state():
	positions = fl.lattice_positions(8, 8)
	phases = np.pi * ((positions[:, 0] + positions[:, 1]) % 2)
	result = fl.pattern_order_parameters(phases, 8, 8)
	assert result['state'] == 'chequerboard'
	assert result['chequerboard'] == pytest.approx(1.0)


def test_classification_recognises_a_twist_state_and_its_winding_numbers():
	positions = fl.lattice_positions(8, 8)
	result = fl.pattern_order_parameters(2.0 * np.pi * positions[:, 1] / 8.0, 8, 8)
	assert result['state'] == 'twist'
	assert (result['twist_mx'], result['twist_my']) == (0, 1)
	assert result['twist_max'] == pytest.approx(1.0)
	assert result['in_phase'] < 0.1


def test_classification_calls_random_phases_unordered():
	rng = np.random.default_rng(4)
	assert fl.pattern_order_parameters(rng.uniform(0, 2 * np.pi, 64), 8, 8)['state'] == 'unordered'


def test_random_phase_baseline_matches_the_rayleigh_mean_but_not_its_spread():
	"""The mean follows the marginal Rayleigh result; the spatial spread does not.

	Overlapping kernels correlate the local order parameters across the lattice, so a
	single realization's spatial standard deviation is a downward-biased estimator of the
	marginal one. This is why the baseline is measured rather than derived.
	"""
	kernel = fl.build_kernel(8, 8, 'dipole', sigma=1.5)
	baseline = fl.random_phase_baseline(8, 8, kernel, realizations=300, seed=2)
	n_eff = baseline['participation_number']
	assert baseline['mean_r_local'] == pytest.approx(np.sqrt(np.pi) / 2 / np.sqrt(n_eff), rel=0.05)
	marginal_spread = np.sqrt(1 - np.pi / 4) / np.sqrt(n_eff)
	assert baseline['spatial_std'] < 0.85 * marginal_spread


def test_two_band_gate_rejects_a_mismatched_kernel():
	with pytest.raises(ValueError, match='kernel is sized for'):
		fl.simulate_two_band_gating({'Nx': 8, 'Ny': 1}, {}, {},
									fl.build_kernel(3, 3, sigma=1.0), 0.5, 4)


def test_two_band_gate_rejects_a_network_that_is_not_two_bands():
	with pytest.raises(ValueError, match='expected .* for two bands'):
		fl.simulate_two_band_gating({'Nx': 7, 'Ny': 1}, {}, {},
									fl.build_kernel(2, 2, sigma=1.0), 0.5, 4)


# ##############################################################################
# linear stability of the in-phase state
# ##############################################################################

def test_fourier_symbol_of_a_normalised_kernel_is_one_at_zero_wavevector():
	kernel = fl.build_kernel(8, 8, 'dipole', sigma=1.5)
	assert fl.fourier_symbol(kernel, 8, 8, 0, 0) == pytest.approx(1.0)


def test_chequerboard_symbol_of_nearest_neighbour_adjacency_is_minus_one():
	"""The binding mode of a square lattice with four-neighbour coupling."""
	adjacency = fl.lattice_adjacency(8, 8, periodic=True, radius=1.0)
	normalised = adjacency / adjacency.sum(axis=1, keepdims=True)
	assert fl.fourier_symbol(normalised, 8, 8, 4, 4) == pytest.approx(-1.0)


def test_smooth_field_kernel_suppresses_the_chequerboard_mode():
	kernel = fl.build_kernel(8, 8, 'dipole', sigma=1.5)
	assert abs(fl.fourier_symbol(kernel, 8, 8, 4, 4)) < 0.01


def test_no_field_is_needed_below_a_quarter_cycle():
	"""With cos(Omega tau) > 0 the delayed channel already stabilises the in-phase state."""
	kernel = fl.build_kernel(8, 8, 'dipole', sigma=1.5)
	adjacency = fl.lattice_adjacency(8, 8, periodic=True, radius=1.0)
	adjacency = adjacency / adjacency.sum(axis=1, keepdims=True)
	result = fl.in_phase_stability_threshold(8, 8, kernel, adjacency,
											 collective_frequency_hz=1.0, delay_s=0.1)
	assert result['cos_omega_tau'] > 0
	assert result['epsilon_c'] == 0.0


def test_threshold_approaches_two_at_half_a_cycle_of_loop_phase():
	"""eps_c -> -2 cos(Omega tau), which is 2 when the loop phase is half a cycle."""
	kernel = fl.build_kernel(8, 8, 'dipole', sigma=1.5)
	adjacency = fl.lattice_adjacency(8, 8, periodic=True, radius=1.0)
	adjacency = adjacency / adjacency.sum(axis=1, keepdims=True)
	result = fl.in_phase_stability_threshold(8, 8, kernel, adjacency,
											 collective_frequency_hz=1.0, delay_s=0.5)
	assert result['epsilon_c'] == pytest.approx(2.0, abs=0.02)
	assert result['mode'] == (4, 4)


def test_in_phase_branch_reduces_to_the_intrinsic_frequency_at_zero_delay():
	assert fl.in_phase_collective_frequency(1.0, 0.15, 0.0) == pytest.approx(1.0)


def test_in_phase_branch_satisfies_its_own_self_consistency_condition():
	for delay in (0.064, 0.192, 0.384, 0.496):
		f = fl.in_phase_collective_frequency(1.0, 0.15, delay)
		assert f == pytest.approx(1.0 + 0.15 * np.sin(-2 * np.pi * f * delay), abs=1e-9)


def test_stability_threshold_on_the_in_phase_branch_matches_direct_integration():
	"""The criterion must be evaluated on the in-phase branch, not on a measured frequency.

	Reference values come from direct integration of the full model, which places the
	exchange of stability at 1.00-1.02 and 1.99-2.01 for these two delays.
	"""
	kernel = fl.build_kernel(8, 8, 'dipole', sigma=1.5)
	adjacency = fl.lattice_adjacency(8, 8, periodic=True, radius=1.0)
	adjacency = adjacency / adjacency.sum(axis=1, keepdims=True)
	for delay, expected in ((0.384, 1.013), (0.496, 2.002)):
		frequency = fl.in_phase_collective_frequency(1.0, 0.15, delay)
		result = fl.in_phase_stability_threshold(8, 8, kernel, adjacency, frequency, delay)
		assert result['epsilon_c'] == pytest.approx(expected, abs=0.02)
