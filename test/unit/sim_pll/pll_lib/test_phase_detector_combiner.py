"""Unit tests for sim_pll.pll_lib.PhaseDetectorCombiner.

The phase-detector output for oscillator k is
    y_k = reduce_l( G_kl * h( (phi_l^delayed - phi_k^feedback)/div + phase_shift ) )
where reduce is the mean when normalize_mutual_coupling_by_inputs is True and the
sum when False, l runs over the senders G.neighbors(k), and h is the coupling
function (here sine, div=1, phase_shift=0).
"""
import numpy as np
import pytest

from sim_pll.pll_lib import PhaseDetectorCombiner
from sim_pll import coupling_fct_lib as cpl
from .conftest import build_component_dicts


# 0 <-> 1 mutual, 2 receives from both 0 and 1
ADJ = np.array([[0, 1, 0],
                [1, 0, 0],
                [1, 1, 0]])


def make_pdc(pll_id, normalize=True, gPDin=1):
    dict_pll, dict_net = build_component_dicts(
        n=3,
        adjacency_matrix=ADJ,
        coupling_gain_matrix=gPDin,
        coupling_function='sin',
    )
    dict_net['normalize_mutual_coupling_by_inputs'] = normalize
    return PhaseDetectorCombiner(pll_id, dict_pll, dict_net), dict_pll


class TestNeighborSelection:

    def test_neighbors_follow_digraph(self):
        pdc0, _ = make_pdc(0)
        pdc2, _ = make_pdc(2)
        assert sorted(pdc0.neighbor_ids) == [1]
        assert sorted(pdc2.neighbor_ids) == [0, 1]


class TestScalarGainOutput:

    def test_mean_over_inputs_when_normalized(self):
        pdc, _ = make_pdc(2, normalize=True, gPDin=1)
        phases = np.array([0.3, 0.7, 1.1])          # phi_0, phi_1, phi_2
        own_feedback = phases[2]
        neighbor_phases = phases[pdc.neighbor_ids]  # phases of senders 0 and 1
        y = pdc.next(own_feedback, neighbor_phases, 0)
        expected = np.mean(cpl.sine(neighbor_phases - own_feedback))
        assert y == pytest.approx(expected)

    def test_sum_over_inputs_when_not_normalized(self):
        pdc, _ = make_pdc(2, normalize=False, gPDin=1)
        phases = np.array([0.3, 0.7, 1.1])
        own_feedback = phases[2]
        neighbor_phases = phases[pdc.neighbor_ids]
        y = pdc.next(own_feedback, neighbor_phases, 0)
        expected = np.sum(cpl.sine(neighbor_phases - own_feedback))
        assert y == pytest.approx(expected)

    def test_no_neighbors_yields_zero(self):
        # oscillator 2 sends to nobody in the transposed sense, but here build a
        # node with no senders: a reference that receives nothing.
        adj = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        dict_pll, dict_net = build_component_dicts(n=3, adjacency_matrix=adj)
        pdc = PhaseDetectorCombiner(0, dict_pll, dict_net)
        assert pdc.neighbor_ids == []
        y = pdc.next(np.array(0.5), np.array([]), 0)
        assert y == 0.0


class TestMatrixGainOutput:

    def test_matrix_gPDin_weights_each_input(self):
        gain = np.array([[0.0, 2.0, 0.0],
                         [3.0, 0.0, 0.0],
                         [1.5, 0.5, 0.0]])
        pdc, _ = make_pdc(2, normalize=True, gPDin=gain)
        # G_kl picks gain[2, neighbor_ids] = [1.5, 0.5]
        np.testing.assert_allclose(pdc.G_kl, np.array([1.5, 0.5]))
        phases = np.array([0.2, 0.9, 1.3])
        own_feedback = phases[2]
        neighbor_phases = phases[pdc.neighbor_ids]
        y = pdc.next(own_feedback, neighbor_phases, 0)
        expected = np.mean(pdc.G_kl * cpl.sine(neighbor_phases - own_feedback))
        assert y == pytest.approx(expected)
