"""Unit tests for sim_pll.coupling_fct_lib.

These tests pin down the *actual* amplitude conventions used in the code, which
follow scipy.signal.sawtooth/square rather than the textbook shifted versions:

- ``triangular`` is ``sawtooth(x, width=0.5)``: it has period 2*pi, equals -1 at
  x = 0, 0 at x = pi/2, and +1 at x = pi (peak-to-peak amplitude 2).
- ``deriv_triangular`` is ``(2/pi)*square(x, duty=0.5)``: a +/-(2/pi) square wave,
  the analytic slope of ``triangular`` away from the kinks at multiples of pi.
- ``square_wave`` maps to [0, 1]; ``square_wave_symm_zero`` to [-0.5, 0.5].
"""
import numpy as np
import pytest

from sim_pll import coupling_fct_lib as cpl


TWO_PI = 2.0 * np.pi
SAMPLES = np.linspace(-3.0, 3.0, 25)


class TestPeriodicity:
    """Every coupling waveform must satisfy h(x) = h(x + 2*pi)."""

    @pytest.mark.parametrize("func", [
        cpl.sine, cpl.cosine, cpl.neg_sine, cpl.neg_cosine,
        cpl.triangular, cpl.deriv_triangular,
        cpl.square_wave, cpl.square_wave_symm_zero,
    ])
    def test_period_two_pi(self, func):
        for x in SAMPLES:
            assert func(x) == pytest.approx(func(x + TWO_PI), abs=1e-12)


class TestKeyValues:
    """Amplitude conventions as implemented (verified against the code)."""

    def test_sine_zero(self):
        assert cpl.sine(0.0) == pytest.approx(0.0)
        assert cpl.sine(np.pi / 2.0) == pytest.approx(1.0)

    def test_cosine_zero(self):
        assert cpl.cosine(0.0) == pytest.approx(1.0)
        assert cpl.neg_cosine(0.0) == pytest.approx(-1.0)

    def test_triangular_convention(self):
        # sawtooth(x, width=0.5): -1 at 0, 0 at pi/2, +1 at pi
        assert cpl.triangular(0.0) == pytest.approx(-1.0)
        assert cpl.triangular(np.pi / 2.0) == pytest.approx(0.0)
        assert cpl.triangular(np.pi) == pytest.approx(1.0)

    def test_square_wave_range(self):
        # square_wave maps to {0, 1}; square_wave_symm_zero to {-0.5, 0.5}
        assert cpl.square_wave(0.0) == pytest.approx(1.0)
        assert cpl.square_wave(np.pi) == pytest.approx(0.0)
        assert cpl.square_wave_symm_zero(0.0) == pytest.approx(0.5)
        assert cpl.square_wave_symm_zero(np.pi) == pytest.approx(-0.5)


class TestDerivativeConsistency:
    """Numerical derivative of triangular matches deriv_triangular away from kinks."""

    @pytest.mark.parametrize("x", [-2.0, -1.0, 0.5, 1.0, 2.0, 2.5])
    def test_numeric_derivative_matches(self, x):
        h = 1e-6
        numeric = (cpl.triangular(x + h) - cpl.triangular(x - h)) / (2.0 * h)
        assert numeric == pytest.approx(cpl.deriv_triangular(x), abs=1e-4)

    def test_deriv_triangular_amplitude(self):
        # +2/pi on the rising branch (0, pi), -2/pi on the falling branch (pi, 2*pi)
        assert cpl.deriv_triangular(0.5) == pytest.approx(2.0 / np.pi)
        assert cpl.deriv_triangular(-0.5) == pytest.approx(-2.0 / np.pi)


class TestInverseRoundTrip:
    """Inverse functions recover the argument on their declared branch."""

    @pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 2.0, 3.0])
    def test_inverse_cosine_round_trip(self, x):
        # arccos is the inverse of cos on [0, pi]
        assert cpl.inverse_cosine(cpl.cosine(x)) == pytest.approx(x)

    @pytest.mark.parametrize("x", [0.2, 0.5, 1.0, 2.0, 3.0])
    def test_inverse_triangular_round_trip_positive_branch(self, x):
        # the positive branch of inverse_triangular (phase_wrap=2) recovers |x|
        # for x on the rising part of the triangular wave, x in (0, pi)
        y = cpl.triangular(x)
        assert cpl.inverse_triangular(y, branch='positive', phase_wrap=2) == pytest.approx(x)

    def test_inverse_triangular_out_of_range_returns_nan(self):
        assert np.isnan(cpl.inverse_triangular(2.0, branch='positive', phase_wrap=1))

    def test_inverse_linear_is_identity(self):
        assert cpl.inverse_linear(0.7) == pytest.approx(0.7)


class TestPfd:
    """Phase-frequency-detector waveform is odd about zero."""

    @pytest.mark.parametrize("x", [0.3, 1.0, 2.0, 3.0])
    def test_pfd_is_odd(self, x):
        assert cpl.pfd(x) == pytest.approx(-cpl.pfd(-x))
