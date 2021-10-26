import unittest

import numpy as np

from sim_pll.pll_lib import Counter


class TestCounter(unittest.TestCase):

    def setUp(self):
        # this method is called before every test
        self.counter = Counter()

    def tearDown(self) -> None:
        # this method is called after every test
        pass

    def test___init__(self):
        self.assertEqual(0, self.counter.reference_phase)

    def test_reset(self):
        self.counter.reset(1.5)
        self.assertEqual(1.5, self.counter.reference_phase)

    def test_read_periods(self):
        self.assertEqual(0, self.counter.read_periods(0.0))
        self.assertEqual(0, self.counter.read_periods(1.0))
        self.assertEqual(1, self.counter.read_periods(np.pi * 2.0))

    def test_read_half_periods(self):
        self.assertEqual(0, self.counter.read_half_periods(0.0))
        self.assertEqual(0, self.counter.read_half_periods(1.0))
        self.assertEqual(2, self.counter.read_half_periods(np.pi * 2.0))


if __name__ == '__main__':
    unittest.main()