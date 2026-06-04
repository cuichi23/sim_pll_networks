"""Unit tests for sim_pll.pll_lib.Delayer (circular-buffer delay indexing)."""
import numpy as np
import pytest

from sim_pll.pll_lib import Delayer
from .conftest import build_component_dicts


# 0 <-> 1 mutual, 2 receives from both 0 and 1
ADJ = np.array([[0, 1, 0],
                [1, 0, 0],
                [1, 1, 0]])
SAMPLE_F_HZ = 100.0


def make_delayer(pll_id, transmission_delay_s, feedback_delay_s=0.0, array_len=64):
    dict_pll, dict_net = build_component_dicts(
        n=3,
        adjacency_matrix=ADJ,
        transmission_delay_s=transmission_delay_s,
        feedback_delay_s=feedback_delay_s,
        sample_freq_hz=SAMPLE_F_HZ,
    )
    delayer = Delayer(pll_id, dict_pll, dict_net, dict_data={})
    delayer.phi_array_len = array_len
    return delayer, dict_pll


def make_phase_memory(array_len=64, n=3):
    """Distinct, recoverable phase value at every (time, oscillator) cell."""
    phi = np.zeros((array_len, n))
    for t in range(array_len):
        for k in range(n):
            phi[t, k] = 100.0 * t + k
    return phi


class TestScalarDelay:

    def test_delay_steps_from_seconds(self):
        delayer, dict_pll = make_delayer(2, transmission_delay_s=5.0 / SAMPLE_F_HZ)
        assert delayer.transmit_delay_steps == 5

    def test_delayed_phases_indexing(self):
        tau_steps = 5
        delayer, _ = make_delayer(2, transmission_delay_s=tau_steps / SAMPLE_F_HZ)
        phi = make_phase_memory(delayer.phi_array_len)
        t = 20
        feedback, transmission = delayer.next(t, phi, t)
        # senders of oscillator 2 are 0 and 1
        expected = phi[(t - tau_steps) % delayer.phi_array_len, [0, 1]]
        np.testing.assert_allclose(transmission, expected)

    def test_circular_wraparound(self):
        tau_steps = 10
        delayer, _ = make_delayer(2, transmission_delay_s=tau_steps / SAMPLE_F_HZ)
        phi = make_phase_memory(delayer.phi_array_len)
        t = 3  # t - tau goes negative -> wraps around the buffer
        _, transmission = delayer.next(t, phi, t)
        expected = phi[(t - tau_steps) % delayer.phi_array_len, [0, 1]]
        np.testing.assert_allclose(transmission, expected)

    def test_zero_feedback_delay_returns_current_phase(self):
        delayer, _ = make_delayer(2, transmission_delay_s=5.0 / SAMPLE_F_HZ, feedback_delay_s=0.0)
        phi = make_phase_memory(delayer.phi_array_len)
        t = 20
        feedback, _ = delayer.next(t, phi, t)
        assert feedback == pytest.approx(phi[t, 2])

    def test_nonzero_feedback_delay(self):
        fb_steps = 3
        delayer, _ = make_delayer(2, transmission_delay_s=5.0 / SAMPLE_F_HZ,
                                  feedback_delay_s=fb_steps / SAMPLE_F_HZ)
        assert delayer.feedback_delay_steps == fb_steps
        phi = make_phase_memory(delayer.phi_array_len)
        t = 20
        feedback, _ = delayer.next(t, phi, t)
        assert feedback == pytest.approx(phi[(t - fb_steps) % delayer.phi_array_len, 2])


class TestPerLinkDelayMatrix:

    def test_tau_kl_matrix_picks_per_neighbor_delays(self):
        # different delay per (k, l) link, in seconds
        tau_matrix = np.array([[0.0, 0.04, 0.07],
                               [0.04, 0.0, 0.07],
                               [0.02, 0.05, 0.0]])  # row 2: from 0 -> 2 steps, from 1 -> 5 steps
        delayer, _ = make_delayer(2, transmission_delay_s=tau_matrix)
        np.testing.assert_array_equal(delayer.transmit_delay_steps, np.array([2, 5]))
        phi = make_phase_memory(delayer.phi_array_len)
        t = 30
        _, transmission = delayer.next(t, phi, t)
        # neighbor 0 delayed by 2 steps, neighbor 1 delayed by 5 steps
        expected = [phi[(t - 2) % delayer.phi_array_len, 0],
                    phi[(t - 5) % delayer.phi_array_len, 1]]
        np.testing.assert_allclose(transmission, expected)
