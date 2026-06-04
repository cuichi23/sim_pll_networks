"""Unit tests for sim_pll.setup: topology construction and initial phase configurations.

These tests pin down bugs found in the 2026 code audit:
- NameError (undefined Nx/Ny) in the 'hexagon' and 'octagon' topology branches.
- NameError (undefined phiM) in the chequerboard branch mx != 0, my == 0 of generate_phi0.
- twistdelta_x computed from my instead of mx in the periodic-grid twist branch.
"""
import numpy as np
import networkx as nx
import pytest

from sim_pll import setup


def make_dict_net(topology: str, nx_grid: int = 3, ny_grid: int = 3, mx: int = 0, my: int = 0) -> dict:
    """Minimal dict_net accepted by setup.setup_topology / setup.generate_phi0."""
    return {
        'topology': topology,
        'Nx': nx_grid,
        'Ny': ny_grid,
        'mx': mx,
        'my': my,
        'special_case': 'False',
        'phiPerturb': [],
        'phiPerturbRot': [],
        'phiInitConfig': [],
    }


class TestSetupTopology:

    def test_global_topology_is_complete_graph(self):
        graph = setup.setup_topology(make_dict_net('global', 3, 1))
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 3  # complete graph K3

    def test_ring_topology(self):
        graph = setup.setup_topology(make_dict_net('ring', 4, 1))
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 4

    def test_chain_topology(self):
        graph = setup.setup_topology(make_dict_net('chain', 4, 1))
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 3

    def test_square_open_topology(self):
        graph = setup.setup_topology(make_dict_net('square-open', 3, 3))
        assert graph.number_of_nodes() == 9
        assert graph.number_of_edges() == 12  # 2 * Nx * Ny - Nx - Ny

    def test_hexagon_topology_constructs_with_diagonal_edges(self):
        """Regression: 'hexagon' raised NameError (Nx, Ny undefined) before the audit fix."""
        graph = setup.setup_topology(make_dict_net('hexagon', 3, 3))
        assert graph.number_of_nodes() == 9
        # 12 grid edges + 4 main-diagonal edges
        assert graph.number_of_edges() == 16
        # main diagonal (0,0)-(1,1): sorted 2d labels -> integer labels 0 and 4
        assert graph.has_edge(0, 4)

    def test_octagon_topology_constructs_with_both_diagonals(self):
        """Regression: 'octagon' raised NameError (Nx, Ny undefined) before the audit fix."""
        graph = setup.setup_topology(make_dict_net('octagon', 3, 3))
        assert graph.number_of_nodes() == 9
        # 12 grid edges + 4 main-diagonal + 4 anti-diagonal edges
        assert graph.number_of_edges() == 20
        # anti-diagonal (0,1)-(1,0): integer labels 1 and 3
        assert graph.has_edge(1, 3)


class TestGeneratePhi0:

    @staticmethod
    def make_dicts(topology: str, mx: int, my: int, n: int = 3):
        dict_net = make_dict_net(topology, n, n, mx=mx, my=my)
        dict_pll = {'transmission_delay': 0.0, 'intrF': [1.0]}
        dict_algo = {'parameter_space_sweeps': 'single'}
        return dict_net, dict_pll, dict_algo

    def test_chequerboard_x_direction_square_open(self):
        """Regression: mx != 0, my == 0 raised NameError (phiM undefined) before the audit fix."""
        dict_net, dict_pll, dict_algo = self.make_dicts('square-open', mx=1, my=0)
        setup.generate_phi0(dict_net, dict_pll, dict_algo)
        config = np.asarray(dict_net['phiInitConfig'])
        assert config.shape == (9,)
        assert np.all(np.isfinite(config))
        assert np.all((config >= 0.0) & (config < 2.0 * np.pi))

    def test_mx_twist_square_periodic_uses_mx_for_x_increment(self):
        """Regression: twistdelta_x was computed from my (not mx); an mx-only twist came out flat."""
        dict_net, dict_pll, dict_algo = self.make_dicts('square-periodic', mx=1, my=0)
        setup.generate_phi0(dict_net, dict_pll, dict_algo)
        config = np.asarray(dict_net['phiInitConfig'])
        assert config.shape == (9,)
        # an mx=1 twist must not be the all-zero configuration
        assert not np.allclose(config, 0.0)
        # x-increment within one row must equal 2*pi*mx/Nx
        expected_increment = 2.0 * np.pi / 3.0
        increments = np.diff(config[:3]) % (2.0 * np.pi)
        assert np.allclose(increments, expected_increment)

    def test_zero_twist_is_in_phase(self):
        dict_net, dict_pll, dict_algo = self.make_dicts('square-periodic', mx=0, my=0)
        setup.generate_phi0(dict_net, dict_pll, dict_algo)
        assert np.allclose(np.asarray(dict_net['phiInitConfig']), 0.0)
