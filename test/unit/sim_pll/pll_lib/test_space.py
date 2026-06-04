"""Smoke tests for sim_pll.pll_lib.Space (distance-based adjacency)."""
import numpy as np

from sim_pll.pll_lib import Space


class TestSpace:

    def test_construction_stores_attributes(self):
        space = Space(signal_propagation_speed=2.0, dimensions_xyz=np.array([10, 10, 10]))
        assert space.signal_propagation_speed == 2.0
        np.testing.assert_array_equal(space.dimensions_xyz, np.array([10, 10, 10]))

    def test_adjacency_within_threshold(self):
        space = Space(1.0, np.array([10, 10, 10]))
        positions = np.array([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],   # close to 0
                              [9.0, 0.0, 0.0]])  # far from both
        adjacency = space.update_adjacency_matrix_for_all_plls_potentially_receiving_a_signal(
            positions, distance_threshold=2.0, geometry_of_threshold='spherical')
        assert adjacency[0, 1] == 1 and adjacency[1, 0] == 1
        assert np.isnan(adjacency[0, 2])
        assert np.isnan(adjacency[1, 2])
