"""Unit tests for sim_pll.pll_lib.PhaseLockedLoop (assembled oscillator)."""
import numpy as np
import pytest

from sim_pll import setup
from sim_pll.pll_lib import PhaseLockedLoop
from .conftest import build_component_dicts


INTR_F_HZ = 1.0
SAMPLE_F_HZ = 125.0


def make_pll_list(n=3, adjacency_matrix=None, **overrides):
    dict_pll, dict_net = build_component_dicts(n=n, adjacency_matrix=adjacency_matrix, **overrides)
    dict_data = {}
    pll_list = setup.generate_plls(dict_pll, dict_net, dict_data)
    return pll_list, dict_pll, dict_net


class TestAssembly:

    def test_generate_plls_returns_phase_locked_loops(self):
        pll_list, _, _ = make_pll_list()
        assert len(pll_list) == 3
        assert all(isinstance(p, PhaseLockedLoop) for p in pll_list)
        assert [p.pll_id for p in pll_list] == [0, 1, 2]


class TestClock:

    def test_clock_period_and_half_period_count(self):
        pll_list, _, _ = make_pll_list()
        pll = pll_list[0]
        pll.clock_reset(0.0)
        assert pll.clock_periods_count(2.0 * np.pi) == 1
        assert pll.clock_halfperiods_count(2.0 * np.pi) == 2

    def test_clock_reset_changes_reference(self):
        pll_list, _, _ = make_pll_list()
        pll = pll_list[0]
        pll.clock_reset(2.0 * np.pi)
        assert pll.clock_periods_count(2.0 * np.pi) == 0
        assert pll.clock_periods_count(4.0 * np.pi) == 1


class TestFreeRunningAndReverseHistory:

    def test_free_running_open_loop_advances_at_intrinsic_freq(self):
        pll_list, dict_pll, _ = make_pll_list(intrinsic_freqs_hz=INTR_F_HZ, coupling_strength_hz=0.1)
        pll = pll_list[0]
        pll.signal_controlled_oscillator.phi = 0.0
        intr_freq_rad = pll.signal_controlled_oscillator.intr_freq_rad
        phi = pll.next_free_running_open_loop()
        assert phi == pytest.approx(intr_freq_rad * dict_pll['dt'])

    def test_setup_hist_reverse_steps_backwards(self):
        pll_list, dict_pll, _ = make_pll_list(intrinsic_freqs_hz=INTR_F_HZ)
        pll = pll_list[0]
        pll.signal_controlled_oscillator.phi = 5.0
        intr_freq_rad = pll.signal_controlled_oscillator.intr_freq_rad
        phi = pll.setup_hist_reverse()
        assert phi == pytest.approx(5.0 - intr_freq_rad * dict_pll['dt'])


class TestPositionAccessors:

    def test_set_and_get_position_speed_variance(self):
        pll_list, _, _ = make_pll_list()
        pll = pll_list[0]
        pll.set_position_3d(np.array([1.0, 2.0, 3.0]))
        pll.set_speed_3d(np.array([0.1, 0.2, 0.3]))
        pll.set_gwn_diffusion_variance_3d(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(pll.get_position_3d(), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(pll.get_speed_3d(), np.array([0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(pll.get_gwn_diffusion_variance_3d(), np.array([0.0, 0.0, 0.0]))
