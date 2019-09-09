"""This module implements the Kalman equations for a UKF on a riemannian
manifold.

For reference, see:
    Unscented Kalman Filtering on Riemannian Manifolds
        by
    Soren Hauberg + Prancois Lauze + Kim Steenstrup Pedersen
"""

import numpy


class RiemannianUKF(object):
    """Keep track of parameters used for applying an unscented Kalman filter
    in a Riemannian setting.
    
    These parameters are:
        statespace: Knows how to manipulate states in local and global coords.
        unscented: Knows how to compute sigma points and covariances.
    """

    def __init__(self, statespace, unscented):
        self.statespace = statespace
        self.unscented = unscented

    def predict(self, x, P, Q, dt):
        """Perform the predict phase of the unscented Kalman filter.
        
        Args:
            x: The a-priori state estimate.
            P: The a-priori error covariance, based at x.
            Q: The transition covariance, based at x.
            dt: The timestep.
        
        Returns:
            x: The predicted state estimate.
            P: The predicted error covariance.
        """
        return predict(x, P, Q, dt, self.statespace, self.unscented)

    def update(self, x, P, R, z):
        """Perform the update phase of the unscented Kalman filter.

        Note that this makes the assumption that the `observation space` is a
        vector space.

        Args:
            x: The predicted state estimate.
            P: The predicted error covariance, based at x.
            R: The observation covariance.
            z: The observation.
        
        Returns:
            x: The a-posteriori state estimate.
            P: The a-posteriori error covariance.
        """
        return update(x, P, R, z, self.statespace, self.unscented)


def predict(x, P, Q, dt, statespace, unscented):
    """Perform the predict phase of the unscented Kalman filter.
    
    Args:
        x: The a-priori state estimate.
        P: The a-priori error covariance, based at x.
        Q: The transition covariance, based at x.
        dt: The timestep.
        statespace: Knows how to manipulate states in local and global coords.
        unscented: Knows how to compute sigma points and covariances.
    
    Returns:
        x: The predicted state estimate.
        P: The predicted error covariance.
    """

    xt = statespace.evolve_state(x, dt)

    local_x = statespace.local_identity(x)
    local_sigmas, w = unscented.sigmas_from_stats(local_x, P)
    for i in range(local_sigmas.shape[1]):
        g = statespace.local_to_global(x, local_sigmas[:, i])
        g = statespace.evolve_state(g, dt)
        local_sigmas[:, i] = statespace.global_to_local(xt, g)

    _, Pt = unscented.stats_from_sigmas(local_sigmas, w)
    return xt, Pt + dt * statespace.local_transition_cov(xt, Q)


def update(x, P, R, z, statespace, unscented):
    """Perform the update phase of the unscented Kalman filter.

    Note that this makes the assumption that the `observation space` is a
    vector space.

    Args:
        x: The predicted state estimate.
        P: The predicted error covariance, based at x.
        R: The observation covariance.
        z: The observation.
        statespace: Knows how to manipulate states in local and global coords.
        unscented: Knows how to compute sigma points and covariances.
    
    Returns:
        x: The a-posteriori state estimate.
        P: The a-posteriori error covariance.
    """

    local_x = statespace.local_identity(x)
    local_sigmas, w = unscented.sigmas_from_stats(local_x, P)

    os = numpy.zeros((6, local_sigmas.shape[1]))  # Should query from statespace
    for i in range(local_sigmas.shape[1]):
        g = statespace.local_to_global(x, local_sigmas[:, i])
        os[:, i] = statespace.observe_state(g)

    _, S = unscented.stats_from_sigmas(os, w)
    S = S + R

    C = local_sigmas @ (w * os).T
    K = C @ numpy.linalg.inv(S)

    new_local_x = K @ (z - statespace.observe_state(x))
    new_x = statespace.local_to_global(x, new_local_x)

    new_P = P - K @ S @ K.T
    new_P = 0.5 * (new_P + new_P.T)  # Symmetrize (a no-op in theory)

    # rebase new_p
    new_local_sigmas, w = unscented.sigmas_from_stats(local_x, new_P)
    for i in range(new_local_sigmas.shape[1]):
        g = statespace.local_to_global(x, new_local_sigmas[:, i])
        new_local_sigmas[:, i] = statespace.global_to_local(new_x, g)

    # get cov matrix at new_x
    _, new_P = unscented.stats_from_sigmas(new_local_sigmas, w)

    return new_x, new_P
