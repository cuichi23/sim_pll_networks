"""Integration tests for sim_lib.simulate_headless_driven (continuous-drive entry).

The driven entry must reuse the standard physics exactly: a constant drive at the
base value must reproduce the static simulation, and a driven step must make the
free node's instantaneous frequency follow it.
"""
import numpy as np
import pytest

from sim_pll import sim_lib, dicts_factory
from sim_pll import check_dicts_lib as chk


F0_HZ = 1.0
K_HZ = 0.15
FC_HZ = 0.25
SAMPLE_F_HZ = 125.0
DELAY_S = 8.0 / SAMPLE_F_HZ
SIM_PERIODS = 80.0
ALL_TO_ALL = np.ones((3, 3)) - np.eye(3)


def make_dicts(adjacency=ALL_TO_ALL, intrinsic_freqs_hz=F0_HZ, noise=0.0):
    """Return (dict_net, dict_pll, dict_algo) in the order the headless entries expect."""
    dict_pll, dict_net, dict_algo = dicts_factory.generate_dicts(
        n=3, topology='custom', adjacency_matrix=adjacency,
        intrinsic_freqs_hz=intrinsic_freqs_hz, coupling_strength_hz=K_HZ,
        cutoff_freqs_hz=FC_HZ, transmission_delay_s=DELAY_S,
        sim_time_periods=SIM_PERIODS, sample_freq_hz=SAMPLE_F_HZ, noise_var_vco=noise)
    return dict_net, dict_pll, dict_algo


def drive_length(adjacency=ALL_TO_ALL, intrinsic_freqs_hz=F0_HZ):
    """Total step count (history + sim) for a constant-length drive array.

    check_dicts_consistency sets dt and sim_time_steps; the history length
    max_delay_steps is the transmission delay in integer steps.
    """
    dict_net, dict_pll, dict_algo = make_dicts(adjacency, intrinsic_freqs_hz)
    chk.check_dicts_consistency(dict_pll, dict_net, dict_algo)
    max_delay_steps = int(round(DELAY_S / dict_pll['dt']))
    return max_delay_steps + dict_pll['sim_time_steps']


def inst_freq_hz(result):
    phi = result['dict_data']['phi']
    dt = result['dict_pll']['dt']
    return np.diff(phi, axis=0) / (2.0 * np.pi * dt)


class TestDrivenEntry:

    def test_constant_drive_reproduces_static_run(self):
        static = sim_lib.simulate_headless(*make_dicts(), seed=0)
        n = static['dict_data']['phi'].shape[0]
        driven = sim_lib.simulate_headless_driven(
            *make_dicts(), driven_parameter='intrinsic_freq',
            driven_node_ids=[0, 1, 2], values_per_step=np.full(n, F0_HZ), seed=0)
        assert np.allclose(driven['dict_data']['phi'], static['dict_data']['phi'], atol=1e-9)

    def test_control_signal_is_recorded(self):
        n = drive_length()
        result = sim_lib.simulate_headless_driven(
            *make_dicts(), driven_parameter='intrinsic_freq',
            driven_node_ids=[0], values_per_step=np.full(n, F0_HZ), seed=0)
        ctrl = result['dict_data']['ctrl']
        assert ctrl.shape == result['dict_data']['phi'].shape
        assert np.all(np.isfinite(ctrl))

    def test_driven_step_makes_isolated_node_follow(self):
        """An uncoupled node driven to a new intrinsic frequency runs at that frequency."""
        adjacency = np.zeros((3, 3))
        n = drive_length(adjacency)
        target_hz = 1.3
        result = sim_lib.simulate_headless_driven(
            *make_dicts(adjacency), driven_parameter='intrinsic_freq',
            driven_node_ids=[0], values_per_step=np.full(n, target_hz), seed=0)
        freqs = inst_freq_hz(result)
        assert abs(np.mean(freqs[-200:, 0]) - target_hz) < 1e-2
        assert abs(np.mean(freqs[-200:, 1]) - F0_HZ) < 1e-2

    def test_seed_determinism_with_noise(self):
        n = drive_length()
        vals = np.full(n, F0_HZ)
        kwargs = dict(driven_parameter='intrinsic_freq', driven_node_ids=[0], values_per_step=vals)
        r1 = sim_lib.simulate_headless_driven(*make_dicts(noise=1e-7), seed=5, **kwargs)
        r2 = sim_lib.simulate_headless_driven(*make_dicts(noise=1e-7), seed=5, **kwargs)
        r3 = sim_lib.simulate_headless_driven(*make_dicts(noise=1e-7), seed=6, **kwargs)
        assert np.array_equal(r1['dict_data']['phi'], r2['dict_data']['phi'])
        assert not np.array_equal(r1['dict_data']['phi'], r3['dict_data']['phi'])

    def test_coupling_drive_zero_decouples(self):
        """Driving the coupling gain of a node to 0 removes the coupling into it."""
        detuned = [F0_HZ, 1.05 * F0_HZ, 0.95 * F0_HZ]
        n = drive_length(ALL_TO_ALL, detuned)
        result = sim_lib.simulate_headless_driven(
            *make_dicts(intrinsic_freqs_hz=detuned), driven_parameter='coupling_strength',
            driven_node_ids=[1], values_per_step=np.zeros(n), seed=0)
        freqs = inst_freq_hz(result)
        assert abs(np.mean(freqs[-200:, 1]) - 1.05 * F0_HZ) < 1e-2

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            sim_lib.simulate_headless_driven(
                *make_dicts(), driven_parameter='intrinsic_freq',
                driven_node_ids=[0], values_per_step=np.full(10, F0_HZ), seed=0)

    def test_rejects_unknown_parameter(self):
        n = drive_length()
        with pytest.raises(ValueError):
            sim_lib.simulate_headless_driven(
                *make_dicts(), driven_parameter='delay',
                driven_node_ids=[0], values_per_step=np.full(n, F0_HZ), seed=0)
