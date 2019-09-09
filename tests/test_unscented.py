import unittest
import numpy
import numpy.testing as npt

from unscented import sigmas_from_stats, stats_from_sigmas


class TestSigmaPoints(unittest.TestCase):
    def setUp(self):
        self.x = numpy.ones(4)
        self.P = numpy.array([[5, 4, 0, 0], [4, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_stats_to_sigma_to_stats_roundtrip(self):
        for h in numpy.linspace(0.01, 10, 100):
            sigmas, weights = sigmas_from_stats(self.x, self.P, h, dim=2)
            x, P = stats_from_sigmas(sigmas, weights)
            npt.assert_almost_equal(x, self.x)
            npt.assert_almost_equal(P, self.P)
