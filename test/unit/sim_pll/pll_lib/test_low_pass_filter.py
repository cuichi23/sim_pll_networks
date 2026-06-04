"""Unit tests for sim_pll.pll_lib.LowPassFilter (first-order step response)."""
import numpy as np
import pytest

from sim_pll.pll_lib import LowPassFilter
from .conftest import build_component_dicts


CUTOFF_HZ = 0.5
SAMPLE_F_HZ = 125.0


def make_filter(cutoff_hz=CUTOFF_HZ, order=1, k_hz=0.1):
    dict_pll, dict_net = build_component_dicts(
        n=3,
        cutoff_freqs_hz=cutoff_hz,
        coupling_strength_hz=k_hz,
        sample_freq_hz=SAMPLE_F_HZ,
        order_lf=order,
    )
    return LowPassFilter(0, dict_pll, dict_net), dict_pll


class TestLowPassFilterFirstOrder:

    def test_beta_matches_discretization(self):
        lpf, dict_pll = make_filter()
        expected_beta = dict_pll['dt'] * 2.0 * np.pi * CUTOFF_HZ
        assert lpf.beta == pytest.approx(expected_beta)
        assert lpf.cutoff_freq_rad == pytest.approx(2.0 * np.pi * CUTOFF_HZ)

    def test_step_response_recursion(self):
        """Driving a constant input u follows c_{n+1} = (1-beta)c_n + beta*u."""
        lpf, dict_pll = make_filter()
        lpf.control_signal = 0.0
        beta = lpf.beta
        u = 0.7
        c = 0.0
        for _ in range(20):
            returned = lpf.next(u)
            c = (1.0 - beta) * c + beta * u  # friction_coefficient == 1
            assert returned == pytest.approx(c)
            assert lpf.get_control_signal() == pytest.approx(c)

    def test_exponential_approach_to_fixed_point(self):
        """Constant input -> control signal converges to that input value."""
        lpf, _ = make_filter()
        lpf.control_signal = 0.0
        u = 1.3
        for _ in range(5000):
            lpf.next(u)
        assert lpf.get_control_signal() == pytest.approx(u, rel=1e-3)

    def test_exponential_rate_matches_2pi_cutfc(self):
        """The deviation from the fixed point decays as (1-beta)^n with beta=2*pi*cutFc*dt."""
        lpf, _ = make_filter()
        lpf.control_signal = 0.0
        u = 1.0
        beta = lpf.beta
        n = 30
        for _ in range(n):
            lpf.next(u)
        predicted = u * (1.0 - (1.0 - beta) ** n)
        assert lpf.get_control_signal() == pytest.approx(predicted)

    def test_set_initial_control_signal_from_history(self):
        """Initial control signal equals (f_inst - f_intr) / K_Hz."""
        lpf, dict_pll = make_filter(k_hz=0.1)
        f_inst = 1.05
        f_prior = 1.04
        initial = lpf.set_initial_control_signal(f_inst, f_prior)
        expected = (f_inst - lpf.intr_freq_Hz) / lpf.K_Hz
        assert initial == pytest.approx(expected)
        assert lpf.get_control_signal() == pytest.approx(expected)

    def test_no_cutoff_is_passthrough(self):
        """cutFc=0 disables the filter; next() returns its input unchanged."""
        lpf, _ = make_filter(cutoff_hz=0.0)
        assert lpf.next(0.42) == pytest.approx(0.42)
        assert lpf.next(-1.7) == pytest.approx(-1.7)


class TestLowPassFilterSecondOrder:

    def test_second_order_sequential_filter_converges(self):
        lpf, _ = make_filter(order=2)
        lpf.control_signal = 0.0
        u = 0.9
        for _ in range(5000):
            lpf.next(u)
        assert lpf.get_control_signal() == pytest.approx(u, rel=1e-2)
