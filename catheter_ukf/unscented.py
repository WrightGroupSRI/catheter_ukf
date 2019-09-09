"""This module implements the `unscented transform` required for the UKF."""

import numpy
import scipy.linalg


class Unscented(object):
    """Keep track of parameters required for the unscented transform.
    
    These parameters are:
        h: Determines the spread in the sigma points.
    """

    def __init__(self, h):
        self.h = h

    def sigmas_from_stats(self, x, P):
        """Compute sigma points and weights from the given parameters.

        Args:
            x: The mean of the sigma points.
            P: The covariance of the sigma points.
        
        Returns:
            sigmas: The calculated sigma points.
            weights: The calculated weights.
        """
        return sigmas_from_stats(x, P, self.h)

    def stats_from_sigmas(self, sigmas, weights):
        """Compute the mean and covariance of the given sigma points.

        Args:
            sigmas: The sigma points.
            weights: The weights.
        
        Returns:
            x: The weighted mean of the sigma points.
            P: The weighted covariance of the sigma points.
        """
        return stats_from_sigmas(sigmas, weights)


def sigmas_from_stats(x, P, h):
    """Compute sigma points and weights from the given parameters.
    
    Compute a set of sigma points and weights representing the given mean
    and covariance.
    
    Args:
        x: The mean of the sigma points.
        P: The covariance of the sigma points.
        h: The `spread` of the sigma points. Controls distance between sigmas.
    
    Returns:
        sigmas: The calculated sigma points.
        weights: The calculated weights.
    """

    # Calculate required sqrt... sqrt(Q) where Q = (M + h) * P
    M = P.shape[0]
    sqrt_Q = scipy.linalg.sqrtm((M + h) * P)
    sqrt_Q = numpy.real(sqrt_Q)  # Should include a sanity check somewhere?

    # Calculate sigmas
    sigmas = numpy.repeat(numpy.reshape(x, (len(x), 1)), (2 * M) + 1, axis=1)
    sigmas[:, 1::2] += sqrt_Q
    sigmas[:, 2::2] -= sqrt_Q

    # Calculate weights
    weights = numpy.repeat(1.0 / (2 * (h + M)), 2 * M + 1)
    weights[0] = h / (h + M)

    return sigmas, weights


def stats_from_sigmas(sigmas, weights):
    """Compute the mean and covariance of the given sigma points.
    
    Compute the mean and covariance represented by a set of sigma points
    and weights.
    
    Args:
        sigmas: The sigma points.
        weights: The weights.
    
    Returns:
        x: The weighted mean of the sigma points.
        P: The weighted covariance of the sigma points.
    """

    x = numpy.average(sigmas, weights=weights, axis=1)
    P = numpy.cov(sigmas, aweights=weights, bias=1)
    return x, P
