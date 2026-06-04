"""Shared helpers for the pll_lib component unit tests.

Each PLL component is constructed from ``(pll_id, dict_pll, dict_net, ...)``.
``build_component_dicts`` returns minimal but consistent dictionaries derived
from ``dicts_factory.generate_dicts``, augmented with the two keys the component
constructors read that the factory leaves to the consistency check / setup step:

- ``dict_pll['dt']``  -- the integration time step (1 / sampleF), and
- ``dict_pll['G']``   -- the coupling graph built by ``setup.setup_topology``.
"""
import numpy as np
import pytest

from sim_pll import dicts_factory, setup


def build_component_dicts(n=3, adjacency_matrix=None, **overrides):
    """Build (dict_pll, dict_net) ready to construct a single PLL component.

    Parameters
    ----------
    n : int
        Number of oscillators (network is n x 1).
    adjacency_matrix : array_like, optional
        When given, a 'custom' directed topology is used so that
        ``dict_pll['G'].neighbors(k)`` lists the senders of oscillator k.
    **overrides
        Forwarded to ``dicts_factory.generate_dicts``.

    Returns
    -------
    tuple of (dict, dict)
        (dict_pll, dict_net) with 'dt' and 'G' populated.
    """
    topology = 'custom' if adjacency_matrix is not None else overrides.pop('topology', 'global')
    dict_pll, dict_net, _ = dicts_factory.generate_dicts(
        n=n,
        topology=topology,
        adjacency_matrix=adjacency_matrix,
        **overrides,
    )
    dict_pll['dt'] = 1.0 / dict_pll['sampleF']
    dict_pll['G'] = setup.setup_topology(dict_net)
    return dict_pll, dict_net


@pytest.fixture
def component_dicts():
    """Factory fixture: call it with the same kwargs as ``build_component_dicts``."""
    return build_component_dicts
