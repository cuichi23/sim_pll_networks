"""Unit tests for sim_pll.pll_lib.SignalControlledOscillator (deterministic, noiseVarVCO=0)."""
import numpy as np
import pytest

from sim_pll.pll_lib import SignalControlledOscillator
from .conftest import build_component_dicts


INTR_F_HZ = 1.0
K_HZ = 0.1
SAMPLE_F_HZ = 125.0


def make_vco(intr_f_hz=INTR_F_HZ, k_hz=K_HZ, history='freeRunning'):
    dict_pll, dict_net = build_component_dicts(
        n=3,
        intrinsic_freqs_hz=intr_f_hz,
        coupling_strength_hz=k_hz,
        sample_freq_hz=SAMPLE_F_HZ,
        noise_var_vco=0.0,
        type_of_history=history,
    )
    return SignalControlledOscillator(0, dict_pll, dict_net), dict_pll


class TestSignalControlledOscillator:

    def test_rad_per_hz_convention(self):
        vco, _ = make_vco()
        assert vco.intr_freq_rad == pytest.approx(2.0 * np.pi * INTR_F_HZ)
        assert vco.K_rad == pytest.approx(2.0 * np.pi * K_HZ)
        assert vco.c == 0.0

    def test_phase_advance_per_step(self):
        """With constant control x, phase advances by (2pi*intrF + K_rad*x)*dt per step."""
        vco, dict_pll = make_vco()
        vco.phi = 0.0
        dt = dict_pll['dt']
        x = 0.5
        expected_dphi = (vco.intr_freq_rad + vco.K_rad * x) * dt
        prev = vco.phi
        for _ in range(10):
            phi, d_phi = vco.next(x)
            assert d_phi == pytest.approx(expected_dphi)
            assert phi == pytest.approx(prev + expected_dphi)
            prev = phi

    def test_next_returns_advanced_phase(self):
        vco, dict_pll = make_vco()
        vco.phi = 1.0
        phi, d_phi = vco.next(0.0)
        # zero control -> free-running advance by intr_freq_rad*dt
        assert d_phi == pytest.approx(vco.intr_freq_rad * dict_pll['dt'])
        assert phi == pytest.approx(1.0 + vco.intr_freq_rad * dict_pll['dt'])

    def test_set_initial_forward_advances_at_init_freq(self):
        vco, dict_pll = make_vco(history='freeRunning')
        vco.phi = 0.0
        phi, d_phi = vco.set_initial_forward()
        # free-running history: init_freq == intr_freq_rad, no coupling contribution
        assert d_phi == pytest.approx(vco.intr_freq_rad * dict_pll['dt'])
        assert phi == pytest.approx(d_phi)

    def test_set_initial_reverse_steps_phase_backwards(self):
        """Reverse history evolution steps the phase backwards by init_freq*dt."""
        vco, dict_pll = make_vco(history='freeRunning')
        vco.phi = 0.0
        phi, d_phi = vco.set_initial_reverse()
        assert d_phi == pytest.approx(-vco.intr_freq_rad * dict_pll['dt'])
        assert phi == pytest.approx(-vco.intr_freq_rad * dict_pll['dt'])

    def test_forward_then_reverse_cancels(self):
        vco, _ = make_vco()
        vco.phi = 2.0
        vco.set_initial_forward()
        vco.set_initial_reverse()
        assert vco.phi == pytest.approx(2.0)

    def test_add_perturbation(self):
        vco, _ = make_vco()
        vco.phi = 0.3
        result = vco.add_perturbation(0.2)
        assert result == pytest.approx(0.5)
        assert vco.phi == pytest.approx(0.5)
