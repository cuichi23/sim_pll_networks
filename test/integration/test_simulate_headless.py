"""Integration tests for sim_lib.simulate_headless and the dicts_factory module.

Physics baselines follow the self-consistency condition for phase-locked states
of delay-coupled oscillators (triad_analysis_notes.tex, our_notes/notes):

    Omega = omega_k + (K_k / n_k) * sum_l c_kl * h(-Omega * tau_kl - beta_kl)

For identical oscillators in an in-phase state (beta_kl = 0) this reduces to
    f_Omega = f_omega + K_Hz * h(-2*pi*f_Omega*tau).
"""
import numpy as np
import pytest

from sim_pll import sim_lib, dicts_factory
from sim_pll import coupling_fct_lib as coupfct


F0_HZ = 1.0
K_HZ = 0.15
FC_HZ = 0.25
SAMPLE_F_HZ = 125.0
DELAY_S = 8.0 / SAMPLE_F_HZ  # exactly 8 time steps
SIM_PERIODS = 200.0


def run_headless(adjacency, intrinsic_freqs_hz, seed=42, **overrides):
    dict_pll, dict_net, dict_algo = dicts_factory.generate_dicts(
        n=3,
        topology='custom',
        adjacency_matrix=adjacency,
        intrinsic_freqs_hz=intrinsic_freqs_hz,
        coupling_strength_hz=K_HZ,
        cutoff_freqs_hz=FC_HZ,
        transmission_delay_s=DELAY_S,
        sim_time_periods=SIM_PERIODS,
        sample_freq_hz=SAMPLE_F_HZ,
        **overrides,
    )
    return sim_lib.simulate_headless(dict_net, dict_pll, dict_algo, seed=seed)


def locked_frequencies_hz(result, last_fraction=0.25):
    """Mean instantaneous frequency of each oscillator over the last fraction of the run."""
    phi = result['dict_data']['phi']
    dt = result['dict_pll']['dt']
    inst_freq = np.diff(phi, axis=0) / (2.0 * np.pi * dt)
    n_last = int(last_fraction * inst_freq.shape[0])
    return np.mean(inst_freq[-n_last:, :], axis=0)


ALL_TO_ALL = np.ones((3, 3)) - np.eye(3)


class TestSimulateHeadless:

    def test_returns_phase_array_and_time(self):
        result = run_headless(ALL_TO_ALL, F0_HZ)
        phi = result['dict_data']['phi']
        t = result['dict_data']['t']
        assert phi.ndim == 2 and phi.shape[1] == 3
        assert len(t) == phi.shape[0]
        assert np.all(np.isfinite(phi))

    def test_identical_all_to_all_locks_in_phase(self):
        result = run_headless(ALL_TO_ALL, F0_HZ)
        freqs = locked_frequencies_hz(result)
        # all three lock to a common frequency
        assert np.max(freqs) - np.min(freqs) < 1e-6
        # phase differences settle to zero (in-phase state)
        phi = result['dict_data']['phi']
        final_diffs = phi[-1, 1:] - phi[-1, 0]
        wrapped = (final_diffs + np.pi) % (2.0 * np.pi) - np.pi
        assert np.allclose(wrapped, 0.0, atol=1e-3)

    def test_locked_frequency_satisfies_self_consistency(self):
        result = run_headless(ALL_TO_ALL, F0_HZ)
        f_locked = float(np.mean(locked_frequencies_hz(result)))
        prediction = F0_HZ + K_HZ * coupfct.sine(-2.0 * np.pi * f_locked * DELAY_S)
        assert abs(f_locked - prediction) < 1e-3

    def test_chain_entrainment_downstream_locks_to_driver(self):
        # 0 -> 1 -> 2: row k lists senders of k
        chain = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        detuned = [F0_HZ, 1.02 * F0_HZ, 0.98 * F0_HZ]
        result = run_headless(chain, detuned)
        freqs = locked_frequencies_hz(result)
        # node 0 receives nothing and runs free; 1 and 2 inherit its frequency
        assert abs(freqs[0] - F0_HZ) < 1e-6
        assert np.allclose(freqs[1:], freqs[0], atol=1e-4)

    def test_detuned_mutual_pair_locks_to_common_frequency(self):
        pair = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])  # 0<->1, node 2 free
        detuned = [0.97 * F0_HZ, 1.03 * F0_HZ, F0_HZ]
        result = run_headless(pair, detuned)
        freqs = locked_frequencies_hz(result)
        assert abs(freqs[0] - freqs[1]) < 1e-6
        # the locked state satisfies the pair of self-consistency equations
        #   Omega = omega_k + K * h(beta_lk - 2*pi*Omega*tau)   (triad notes, N=2)
        phi = result['dict_data']['phi']
        f_locked = float(freqs[0])
        beta = (phi[-1, 1] - phi[-1, 0] + np.pi) % (2.0 * np.pi) - np.pi
        arg = 2.0 * np.pi * f_locked * DELAY_S
        assert abs(f_locked - (detuned[0] + K_HZ * coupfct.sine(beta - arg))) < 2e-3
        assert abs(f_locked - (detuned[1] + K_HZ * coupfct.sine(-beta - arg))) < 2e-3
        # node 2 is uncoupled and stays at its intrinsic frequency
        assert abs(freqs[2] - F0_HZ) < 1e-6

    def test_seed_reproducibility_with_noise(self):
        kwargs = {'noise_var_vco': 1e-7}
        result_a = run_headless(ALL_TO_ALL, F0_HZ, seed=7, **kwargs)
        result_b = run_headless(ALL_TO_ALL, F0_HZ, seed=7, **kwargs)
        result_c = run_headless(ALL_TO_ALL, F0_HZ, seed=8, **kwargs)
        assert np.array_equal(result_a['dict_data']['phi'], result_b['dict_data']['phi'])
        assert not np.array_equal(result_a['dict_data']['phi'], result_c['dict_data']['phi'])
