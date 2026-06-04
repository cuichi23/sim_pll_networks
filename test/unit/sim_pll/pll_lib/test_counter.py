"""Unit tests for sim_pll.pll_lib.Counter (cycle / half-cycle counting)."""
import numpy as np

from sim_pll.pll_lib import Counter


class TestCounter:

    def test_initial_reference_phase_is_zero(self):
        assert Counter().reference_phase == 0

    def test_reset_sets_reference_phase(self):
        counter = Counter()
        counter.reset(1.5)
        assert counter.reference_phase == 1.5

    def test_read_periods(self):
        counter = Counter()
        assert counter.read_periods(0.0) == 0
        assert counter.read_periods(1.0) == 0
        assert counter.read_periods(2.0 * np.pi) == 1
        assert counter.read_periods(7.0 * np.pi) == 3

    def test_read_half_periods(self):
        counter = Counter()
        assert counter.read_half_periods(0.0) == 0
        assert counter.read_half_periods(1.0) == 0
        assert counter.read_half_periods(2.0 * np.pi) == 2
        assert counter.read_half_periods(3.0 * np.pi) == 3

    def test_counting_relative_to_reference(self):
        counter = Counter()
        counter.reset(2.0 * np.pi)
        assert counter.read_periods(2.0 * np.pi) == 0
        assert counter.read_periods(4.0 * np.pi) == 1
