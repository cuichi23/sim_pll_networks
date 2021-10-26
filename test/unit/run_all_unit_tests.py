import unittest

from test.unit.sim_pll import test_sim_lib
from test.unit.sim_pll.pll_lib import test_counter, test_pll_lib, test_delayer, test_low_pass_filter, \
    test_phase_detector_combiner, test_phase_locked_loop, test_signal_controlled_oscillator, test_space


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromModule(test_sim_lib))
    suite.addTests(loader.loadTestsFromModule(test_pll_lib))
    suite.addTests(loader.loadTestsFromModule(test_delayer))
    suite.addTests(loader.loadTestsFromModule(test_low_pass_filter))
    suite.addTests(loader.loadTestsFromModule(test_phase_detector_combiner))
    suite.addTests(loader.loadTestsFromModule(test_phase_locked_loop))
    suite.addTests(loader.loadTestsFromModule(test_signal_controlled_oscillator))
    suite.addTests(loader.loadTestsFromModule(test_counter))
    suite.addTests(loader.loadTestsFromModule(test_space))

    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("Success")
        exit(0)
    else:
        print("Failed")
        exit(1)


main()
