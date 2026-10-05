"""
Unit tests for forecast metrics.

Each test uses a tiny hand-computed example so the expected value
is verifiable by mental arithmetic.
"""
import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import wape, bias, mase, prediction_interval_coverage


class TestWAPE:
    """Tests for Weighted Absolute Percentage Error."""

    def test_perfect_forecast(self):
        """WAPE should be 0 for perfect forecast."""
        actual = np.array([10, 20, 30])
        forecast = np.array([10, 20, 30])
        assert wape(actual, forecast) == 0.0

    def test_hand_computed(self):
        """
        Hand computation:
            actual   = [10, 20, 30]  → sum = 60
            forecast = [12, 18, 33]
            errors   = [2,  2,  3]   → sum = 7
            WAPE = 7 / 60 = 0.11667
        """
        actual = np.array([10, 20, 30])
        forecast = np.array([12, 18, 33])
        result = wape(actual, forecast)
        expected = 7.0 / 60.0
        assert abs(result - expected) < 1e-10, f"Got {result}, expected {expected}"

    def test_all_zeros_actual(self):
        """WAPE should be inf when all actuals are 0."""
        actual = np.array([0, 0, 0])
        forecast = np.array([1, 2, 3])
        assert wape(actual, forecast) == np.inf

    def test_with_zeros_in_actual(self):
        """
        WAPE handles zeros in actuals (unlike MAPE).
            actual   = [0, 10, 20]  → sum = 30
            forecast = [2, 10, 18]
            errors   = [2, 0,  2]   → sum = 4
            WAPE = 4 / 30 = 0.13333
        """
        actual = np.array([0, 10, 20])
        forecast = np.array([2, 10, 18])
        result = wape(actual, forecast)
        expected = 4.0 / 30.0
        assert abs(result - expected) < 1e-10

    def test_symmetry(self):
        """WAPE is symmetric in the error direction (uses absolute)."""
        actual = np.array([10, 20])
        # Over-forecast
        f1 = np.array([15, 25])
        # Under-forecast by same amount
        f2 = np.array([5, 15])
        assert abs(wape(actual, f1) - wape(actual, f2)) < 1e-10


class TestBias:
    """Tests for Normalized Bias."""

    def test_perfect_forecast(self):
        """Bias should be 0 for perfect forecast."""
        actual = np.array([10, 20, 30])
        forecast = np.array([10, 20, 30])
        assert bias(actual, forecast) == 0.0

    def test_over_forecast(self):
        """
        Hand computation:
            actual   = [10, 20]   → sum_abs = 30
            forecast = [15, 25]
            diff     = [5, 5]     → sum = 10
            Bias = 10 / 30 = 0.33333 (positive = over-forecast)
        """
        actual = np.array([10, 20])
        forecast = np.array([15, 25])
        result = bias(actual, forecast)
        expected = 10.0 / 30.0
        assert abs(result - expected) < 1e-10
        assert result > 0, "Over-forecast should be positive"

    def test_under_forecast(self):
        """
        Hand computation:
            actual   = [10, 20]   → sum_abs = 30
            forecast = [8, 16]
            diff     = [-2, -4]   → sum = -6
            Bias = -6 / 30 = -0.2 (negative = under-forecast)
        """
        actual = np.array([10, 20])
        forecast = np.array([8, 16])
        result = bias(actual, forecast)
        expected = -6.0 / 30.0
        assert abs(result - expected) < 1e-10
        assert result < 0, "Under-forecast should be negative"

    def test_all_zeros_actual(self):
        """Bias should be inf when all actuals are 0."""
        actual = np.array([0, 0, 0])
        forecast = np.array([1, 2, 3])
        assert bias(actual, forecast) == np.inf


class TestMASE:
    """Tests for Mean Absolute Scaled Error."""

    def test_hand_computed(self):
        """
        Hand computation with season=2:
            insample = [1, 3, 2, 4, 3, 5]
            seasonal naive errors = |insample[2:] - insample[:-2]|
                = |[2,4,3,5] - [1,3,2,4]| = [1,1,1,1]
                → mean = 1.0

            actual   = [6, 7]
            forecast = [5, 8]
            errors   = |[6,7] - [5,8]| = [1, 1]
                → mean = 1.0

            MASE = 1.0 / 1.0 = 1.0
        """
        insample = np.array([1, 3, 2, 4, 3, 5])
        actual = np.array([6, 7])
        forecast = np.array([5, 8])
        result = mase(actual, forecast, insample, season=2)
        assert abs(result - 1.0) < 1e-10

    def test_beats_naive(self):
        """
        MASE < 1 means model beats the seasonal naive.
            insample = [10, 20, 12, 22, 14, 24]  (season=2)
            naive errors = |[12,22,14,24] - [10,20,12,22]| = [2,2,2,2]
                → mean = 2.0

            actual   = [16, 26]
            forecast = [15.5, 26.5]
            errors   = [0.5, 0.5]  → mean = 0.5

            MASE = 0.5 / 2.0 = 0.25
        """
        insample = np.array([10, 20, 12, 22, 14, 24])
        actual = np.array([16, 26])
        forecast = np.array([15.5, 26.5])
        result = mase(actual, forecast, insample, season=2)
        expected = 0.25
        assert abs(result - expected) < 1e-10
        assert result < 1.0, "Model should beat naive (MASE < 1)"

    def test_constant_insample(self):
        """MASE should be inf if in-sample has zero seasonal variation."""
        insample = np.array([5, 5, 5, 5, 5, 5])
        actual = np.array([6, 7])
        forecast = np.array([5, 5])
        result = mase(actual, forecast, insample, season=2)
        assert result == np.inf


class TestPredictionIntervalCoverage:
    """Tests for prediction interval coverage."""

    def test_all_covered(self):
        """All actuals below upper bound → coverage = 1.0"""
        actual = np.array([1, 2, 3, 4, 5])
        upper = np.array([10, 10, 10, 10, 10])
        assert prediction_interval_coverage(actual, upper) == 1.0

    def test_none_covered(self):
        """All actuals above upper bound → coverage = 0.0"""
        actual = np.array([10, 20, 30])
        upper = np.array([1, 2, 3])
        assert prediction_interval_coverage(actual, upper) == 0.0

    def test_hand_computed(self):
        """
        actual = [5, 15, 10, 25, 8]
        upper  = [10, 10, 10, 10, 10]
        covered = [T, F, T, F, T] → 3/5 = 0.6
        """
        actual = np.array([5, 15, 10, 25, 8])
        upper = np.array([10, 10, 10, 10, 10])
        result = prediction_interval_coverage(actual, upper)
        assert abs(result - 0.6) < 1e-10

    def test_exact_boundary(self):
        """Actual == upper bound counts as covered (<=)."""
        actual = np.array([10])
        upper = np.array([10])
        assert prediction_interval_coverage(actual, upper) == 1.0
