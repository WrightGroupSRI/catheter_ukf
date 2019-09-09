""" """

import numpy
from . import riemannian_ukf
from . import statespace
from . import unscented


class UKF(object):
    """Keeps track of the parameters required to use the unscented Kalman
    filter designed for tracking our catheter.

    Args:
        coil_distance: The distance between the coils in mm.
        tip_distance: The distance from the tip to the tip-adjacent coil in mm.
        h: Determines the spread in the sigma points. If the filter seems to
           diverge consider reducing this value.
    """

    def __init__(self, coil_distance=7.8, tip_distance=9.0, h=0.0001):
        s = statespace.States(coil_distance, tip_distance)
        u = unscented.Unscented(h)
        self._ukf = riemannian_ukf.RiemannianUKF(s, u)
        self.Q = self._create_transition_cov(s.tip_offset)
        self.R = self._create_measurement_cov()

    def predict(self, x, P, dt):
        """Perform the predict phase of the unscented Kalman filter.
        
        Args:
            x: The a-priori state estimate.
            P: The a-priori error covariance, based at x.
            dt: The timestep.
        
        Returns:
            x: The predicted state estimate.
            P: The predicted error covariance.
        """
        return self._ukf.predict(x, P, self.Q, dt)

    def update(self, x, P, z):
        """Perform the update phase of the unscented Kalman filter.
        
        Args:
            x: The predicted state estimate.
            P: The predicted error covariance, based at x.
            z: The observation.
        
        Returns:
            x: The a-posteriori state estimate.
            P: The a-posteriori error covariance.
        """
        return self._ukf.update(x, P, self.R, z)

    def filter(self, x, P, z, dt):
        """Perform a combination predict/update step using this unscented
        Kalman filter.
        
        Args:
            x: The a-priori state estimate.
            P: The a-priori error covariance, based at x.
            z: The observation.
            dt: The timestep.
        
        Returns:
            x: The a-posteriori state estimate.
            P: The a-posteriori error covariance.
        """
        x, P = self.predict(x, P, dt)
        return self.update(x, P, z)

    def estimate_initial_state(self, dist_coord, prox_coord):
        p = 0.5*(dist_coord+prox_coord)
        d = 0.5*(dist_coord-prox_coord)
        d = d / numpy.linalg.norm(d, ord=2)
        x = numpy.concatenate((p, numpy.zeros(6), d, numpy.zeros(6)))

        Px, Pv, Pu = 1, 1, 1
        c = self._linear_angular_ratio(self._ukf.statespace.tip_offset)
        P = numpy.array([Px, Pv, Pu])
        P = numpy.concatenate((P, c * P))
        P = numpy.repeat(P, 3)
        P = numpy.diag(P)
        return x, P

    def tip_and_coils(self, x):
        center = x[0:3]
        direction = x[9:12]

        tip_coord = center + (self._ukf.statespace.tip_offset) * direction
        dist_coord = center + (self._ukf.statespace.coil_offset) * direction
        prox_coord = center - (self._ukf.statespace.coil_offset) * direction

        return tip_coord, dist_coord, prox_coord

    @classmethod
    def _linear_angular_ratio(cls, tip_offset):
        return (1.0 / tip_offset) ** 2.0

    @classmethod
    def _create_transition_cov(cls, tip_offset):
        # Relative noise of positions, velocities, and accelerations
        Qx, Qv, Qu = 1e-12, 1e0, 1e0
        c = cls._linear_angular_ratio(tip_offset)
        # This is Q
        Q = numpy.array([Qx, Qv, Qu])
        Q = numpy.concatenate((Q, c * Q))
        Q = numpy.repeat(Q, 3)
        Q = numpy.diag(Q)
        return Q

    @staticmethod
    def _create_measurement_cov():
        R = 0.0010 * numpy.block(
            [
                [1.0 * numpy.eye(3), 0.6 * numpy.eye(3)],
                [0.6 * numpy.eye(3), 1.0 * numpy.eye(3)],
            ]
        )
        return R
