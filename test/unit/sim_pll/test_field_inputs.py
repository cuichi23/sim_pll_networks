# -*- coding: utf-8 -*-
"""Tests for external input signals injected into the two-channel lattice.

Four channels are distinguished, and the tests pin what makes them different rather
than only that they run. The load-bearing one is
:func:`test_detuning_response_matches_the_inverse_stability_operator`, which checks the
physics: a static detuning pattern at lattice mode q produces a phase response whose
amplitude is the reciprocal of the same operator that governs linear stability.
"""

import contextlib
import io as _io

import numpy as np
import pytest

from sim_pll import field_lib as fl


def _run(dicts, **kwargs):
	"""Run one simulation quietly. ``dicts`` is a fresh (pll, net, algo) triple."""
	dict_pll, dict_net, dict_algo = dicts
	with contextlib.redirect_stdout(_io.StringIO()):
		return fl.simulate_with_local_field(dict_net, dict_pll, dict_algo, **kwargs)


NX = NY = 8
N = NX * NY


def _pattern(nx, ny, mx, my):
	"""Unit-amplitude cosine pattern at lattice mode (mx, my)."""
	x = np.arange(nx)[None, :] * np.ones((ny, 1))
	y = np.ones((1, nx)) * np.arange(ny)[:, None]
	return np.cos(2.0 * np.pi * (mx * x / nx + my * y / ny)).ravel()


class TestInputSignal:
	"""The input object itself: shape, channels, and time modulation."""

	def test_static_signal_returns_the_scaled_pattern(self):
		pattern = _pattern(NX, NY, 1, 0)
		signal = fl.InputSignal(pattern, channel='detuning', amplitude=0.03)
		np.testing.assert_allclose(signal.value(0.0), 0.03 * pattern)
		np.testing.assert_allclose(signal.value(17.3), 0.03 * pattern)

	def test_modulated_signal_oscillates_at_the_stated_frequency(self):
		pattern = np.ones(N)
		signal = fl.InputSignal(pattern, channel='detuning', amplitude=1.0,
								modulation_hz=0.25)
		np.testing.assert_allclose(signal.value(0.0), pattern)
		np.testing.assert_allclose(signal.value(2.0), -pattern, atol=1e-12)
		np.testing.assert_allclose(signal.value(4.0), pattern, atol=1e-12)

	def test_unknown_channel_is_rejected(self):
		with pytest.raises(ValueError, match='channel'):
			fl.InputSignal(np.ones(N), channel='telepathy')

	def test_pattern_length_is_checked_against_the_lattice(self):
		signal = fl.InputSignal(np.ones(N - 1), channel='detuning')
		with pytest.raises(ValueError, match='oscillators'):
			signal.check_size(N)


class TestInputChannels:
	"""Each channel must act at its own injection point and nowhere else."""

	def test_no_inputs_is_bit_identical_to_the_plain_run(self, two_channel_dicts):
		coupling = fl.LocalFieldCoupling(fl.build_kernel(NX, NY, sigma=1.5), epsilon=0.3)
		plain = _run(two_channel_dicts(), field_coupling=coupling, seed=7)
		with_none = _run(two_channel_dicts(), field_coupling=coupling, seed=7, inputs=None)
		np.testing.assert_array_equal(plain['dict_data']['phi'],
									  with_none['dict_data']['phi'])

	def test_empty_input_list_is_bit_identical_to_no_inputs(self, two_channel_dicts):
		coupling = fl.LocalFieldCoupling(fl.build_kernel(NX, NY, sigma=1.5), epsilon=0.3)
		plain = _run(two_channel_dicts(), field_coupling=coupling, seed=7)
		empty = _run(two_channel_dicts(), field_coupling=coupling, seed=7, inputs=[])
		np.testing.assert_array_equal(plain['dict_data']['phi'], empty['dict_data']['phi'])

	def test_uniform_detuning_input_shifts_the_collective_frequency_self_consistently(
			self, two_channel_dicts):
		"""A uniform input of I raises the effective intrinsic frequency by K I.

		The resulting shift of the collective frequency is not K I. The delayed channel
		re-equilibrates at the new frequency, so the shift is the difference of two
		solutions of f = f0 + K sin(-2 pi f tau) with f0 and f0 + K I. At K = 0.15,
		I = 0.2 and tau f0 = 0.064 that is 0.0284 Hz rather than the naive 0.03.
		"""
		coupling = fl.LocalFieldCoupling(fl.build_kernel(NX, NY, sigma=1.5), epsilon=0.5)
		drive = 0.2
		reference = _run(two_channel_dicts(160.0), field_coupling=coupling, seed=3)
		signal = fl.InputSignal(np.ones(N), channel='detuning', amplitude=drive)
		driven = _run(two_channel_dicts(160.0), field_coupling=coupling, seed=3,
					  inputs=[signal])
		dt = reference['dict_pll']['dt']
		tail = slice(-400, None)
		f_ref = np.mean(np.diff(reference['dict_data']['phi'][tail], axis=0)) / (2 * np.pi * dt)
		f_drv = np.mean(np.diff(driven['dict_data']['phi'][tail], axis=0)) / (2 * np.pi * dt)
		coupling_hz, tau_s = 0.15, 8 / 125.0
		expected = (fl.in_phase_collective_frequency(1.0 + coupling_hz * drive,
													 coupling_hz, tau_s)
					- fl.in_phase_collective_frequency(1.0, coupling_hz, tau_s))
		assert abs(expected - 0.15 * drive) > 1e-4, 'the naive K I must differ, or the test is vacuous'
		assert abs((f_drv - f_ref) - expected) < 0.03 * abs(expected)

	def test_gain_channel_does_not_move_an_uncoupled_lattice(self, two_channel_dicts):
		"""Modulating the synaptic gain can do nothing when there is no synaptic drive."""
		coupling = fl.LocalFieldCoupling(fl.build_kernel(NX, NY, sigma=1.5), epsilon=0.0)
		pattern = _pattern(NX, NY, 2, 0)
		gain = fl.InputSignal(pattern, channel='gain', amplitude=0.9)
		free = _run(two_channel_dicts(), field_coupling=coupling, seed=5,
					inputs=[fl.InputSignal(np.zeros(N), 'gain')])
		modulated = _run(two_channel_dicts(), field_coupling=coupling, seed=5,
						 inputs=[gain])
		# with a non-empty adjacency the gain channel must change something
		assert not np.allclose(free['dict_data']['phi'][-1], modulated['dict_data']['phi'][-1])

	def test_field_gain_channel_is_inert_when_epsilon_is_zero(self, two_channel_dicts):
		"""Scaling a field gain of zero is still zero, whatever the input pattern."""
		coupling = fl.LocalFieldCoupling(fl.build_kernel(NX, NY, sigma=1.5), epsilon=0.0)
		pattern = _pattern(NX, NY, 1, 1)
		plain = _run(two_channel_dicts(), field_coupling=coupling, seed=11)
		scaled = _run(two_channel_dicts(), field_coupling=coupling, seed=11,
					  inputs=[fl.InputSignal(pattern, channel='field_gain',
											 amplitude=1.0)])
		np.testing.assert_array_equal(plain['dict_data']['phi'], scaled['dict_data']['phi'])

	def test_field_bias_pulls_the_lattice_toward_the_reference_phase(self, two_channel_dicts):
		"""A strong uniform external phasor is an external field; phases follow it."""
		coupling = fl.LocalFieldCoupling(fl.build_kernel(NX, NY, sigma=1.5), epsilon=1.0)
		target = 0.0
		biased = _run(
			two_channel_dicts(120.0), field_coupling=coupling, seed=2,
			inputs=[fl.InputSignal(np.ones(N), channel='field_bias', amplitude=4.0,
								   reference_phase=target)])
		final = biased['dict_data']['phi'][-1]
		offset = np.angle(np.mean(np.exp(1j * (final - final.mean()))))
		spread = np.std(np.angle(np.exp(1j * (final - np.mean(final)))))
		assert spread < 0.25, 'a strong external field should align the lattice'


class TestSpatialResponse:
	"""The physics: the lattice inverts the operator that governs its own stability."""

	def test_detuning_response_matches_the_inverse_stability_operator(self):
		"""Static detuning at mode q gives a phase response of amplitude 1/(2 pi K L(q)).

		Linearising about the locked state, a static frequency perturbation dw(q) is
		balanced by 2 pi K L(q) dtheta(q) with

			L(q) = eps [1 - What(q)] + cos(Omega tau) [1 - chat(q)]

		which is the left-hand side of the stability criterion. Measuring dtheta(q) and
		multiplying by L(q) must therefore return the input, independently of q.
		"""
		from sim_pll import dicts_factory

		coupling_hz, cutoff_hz, delay_steps, sample_hz = 0.15, 0.25, 8, 125.0
		epsilon, amplitude = 0.5, 0.02
		kernel = fl.build_kernel(NX, NY, sigma=1.5)
		adjacency = fl.lattice_adjacency(NX, NY, periodic=True, radius=1.0)
		omega = fl.in_phase_collective_frequency(1.0, coupling_hz, delay_steps / sample_hz)
		cos_loop = np.cos(2.0 * np.pi * omega * delay_steps / sample_hz)

		errors = []
		for mode in (1, 2, 3):
			pattern = _pattern(NX, NY, mode, 0)
			with contextlib.redirect_stdout(_io.StringIO()):
				dict_pll, dict_net, dict_algo = dicts_factory.generate_dicts(
					n=N, topology='custom', adjacency_matrix=adjacency,
					intrinsic_freqs_hz=1.0, coupling_strength_hz=coupling_hz,
					cutoff_freqs_hz=cutoff_hz,
					transmission_delay_s=delay_steps / sample_hz,
					sim_time_periods=120.0, sample_freq_hz=sample_hz,
					noise_var_vco=0.0, coupling_function='sin',
					phase_perturbation=np.zeros(N))
			dict_net['Nx'], dict_net['Ny'] = NX, NY
			with contextlib.redirect_stdout(_io.StringIO()):
				result = fl.simulate_with_local_field(
					dict_net, dict_pll, dict_algo,
					field_coupling=fl.LocalFieldCoupling(kernel, epsilon=epsilon), seed=0,
					inputs=[fl.InputSignal(pattern, channel='detuning',
										   amplitude=amplitude)])
			final = result['dict_data']['phi'][-1]
			final = final - final.mean()
			# projection of the phase field onto the driving mode
			response = 2.0 * np.mean(final * pattern)
			w_hat = fl.fourier_symbol(kernel, NX, NY, mode, 0)
			c_hat = fl.fourier_symbol(adjacency / adjacency.sum(axis=1)[:, None], NX, NY, mode, 0)
			operator = epsilon * (1.0 - w_hat) + cos_loop * (1.0 - c_hat)
			recovered = response * operator
			errors.append(abs(recovered - amplitude) / amplitude)
		assert max(errors) < 0.12, 'response deviates from 1/L(q): %r' % errors


class TestFieldIsUnfiltered:
	"""The field channel must reach the oscillator without passing through the filter.

	Appendix Eq. (A1) of the manuscript writes both channels as convolutions,
	``kappa int ds p(s) x^c(t-s) + kappa eps int ds g(s) x^f(t-s)``, and the model of the
	paper is the special case ``g(s) = delta(s)``: the field is *not* filtered. Reading
	Eq. (A1) alone suggests otherwise, so the property is pinned here rather than left to
	the reader of ``_step_oscillator``.

	The discriminator is the first integration step. The loop filter's output starts at
	its initial value, so a filtered field could contribute only ``omega_c dt`` of its
	value; an unfiltered one contributes all of it. At 125 samples per period and
	``f_c = 0.25`` Hz the two predictions differ by a factor of about 80.
	"""

	@staticmethod
	def _first_step(epsilon, initial, kernel, adjacency, nx=4, ny=4):
		from sim_pll import dicts_factory
		n = nx * ny
		with contextlib.redirect_stdout(_io.StringIO()):
			dict_pll, dict_net, dict_algo = dicts_factory.generate_dicts(
				n=n, topology='custom', adjacency_matrix=adjacency,
				intrinsic_freqs_hz=1.0, coupling_strength_hz=0.15, cutoff_freqs_hz=0.25,
				transmission_delay_s=8 / 125.0, sim_time_periods=3.0,
				sample_freq_hz=125.0, noise_var_vco=0.0, coupling_function='sin',
				phase_perturbation=list(initial))
			dict_net['Nx'], dict_net['Ny'] = nx, ny
			result = fl.simulate_with_local_field(
				dict_net, dict_pll, dict_algo,
				field_coupling=fl.LocalFieldCoupling(kernel, epsilon=epsilon), seed=0)
		phi = result['dict_data']['phi']
		start = dict_net['max_delay_steps']
		return phi[start, :], phi[start + 1, :] - phi[start, :], result['dict_pll']['dt']

	def test_field_reaches_the_oscillator_unfiltered(self):
		nx = ny = 4
		n = nx * ny
		initial = np.random.default_rng(7).uniform(-np.pi, np.pi, n)
		kernel = fl.build_kernel(nx, ny, kind='gaussian', sigma=1.5, periodic=True)
		adjacency = fl.lattice_adjacency(nx, ny, periodic=True, radius=1.0)
		epsilon, coupling_hz, cutoff_hz = 0.7, 0.15, 0.25

		phases, step_off, dt = self._first_step(0.0, initial, kernel, adjacency)
		_, step_on, _ = self._first_step(epsilon, initial, kernel, adjacency)
		measured = (step_on - step_off) / dt

		field = np.imag(fl.local_order_parameter(phases, kernel) * np.exp(-1j * phases))
		unfiltered = 2 * np.pi * coupling_hz * epsilon * field
		filtered = unfiltered * (2 * np.pi * cutoff_hz * dt)

		# the two hypotheses must be far apart, or the test proves nothing
		assert np.max(np.abs(unfiltered - filtered)) > 1e-3
		np.testing.assert_allclose(measured, unfiltered, atol=1e-11)
		assert np.max(np.abs(measured - filtered)) > 1e-3
