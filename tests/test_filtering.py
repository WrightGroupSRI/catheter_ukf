import unittest
import numpy
import numpy.testing as npt

from statespace import evolve_state, observe_state
from filtering import predict, update


class TestFilterPoints(unittest.TestCase):
    def setUp(self):
        self.x0 = numpy.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 2])
        self.P0 = numpy.eye(12)
        self.P0[6, 6] = 0
        self.P0[9, 9] = 0
        self.Q = 0.5  # numpy.eye(12)
        self.R = numpy.eye(6)

    def test_exact_obs_const_state(self):

        true_x = self.x0
        x, P = true_x, self.P0
        z = observe_state(true_x)

        for _ in range(1000):
            true_x = evolve_state(true_x, 1)
            x, P = predict(x, P, self.Q, 1)
            z = observe_state(true_x)
            x, P = update(x, P, self.R, z)

            print("- - - - - - - - - - - -")
            print("dx:")
            print(x - true_x)
            print("dz:")
            print(observe_state(x) - z)
