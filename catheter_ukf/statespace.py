"""This module implements various state manipulation functions for the UKF.

States are represented by 3+3+3+3+3+3 = 18 parameters. These are:
    x: In R3. The position of the midpoint between the coils.
    v: In R3. The velocity of x.
    a: In R3. The acceleration of x.
    q: In S2 (as a subset of R3). The direction from x to the catheter's tip.
    w: In R2 (as a subset of R3). The (rotational) velocity of q.
    u: In R2 (as a subset of R3). The (rotational) acceleration of q.

If q is a point on the unit sphere, then w and u are tangent to the sphere at
q.

Roll of the catheter is not represented.
"""

import numpy
import quaternion


class States(object):
    """Keep track of parameters used for state transition and observation.
    
    These parameters are:
        coil_distance: The distance between the coils in mm.
        tip_distance: The distance from the tip to the tip-adjacent coil in mm.
    """

    def __init__(self, coil_distance, tip_distance):
        self.coil_offset = coil_distance / 2.0
        self.tip_offset = tip_distance + self.coil_offset

    def evolve_state(self, s, dt):
        """Calculate the evolution of the given state after the given duration.
        
        Args:
            s: The starting state.
            dt: The duration.

        Returns the evolved state.
        """
        return evolve_state(s, dt)

    def observe_state(self, s):
        """ """
        return observe_state(s, self.coil_offset)

    def tip_from_state(self, s):
        """ """
        return tip_from_state(s, self.tip_offset)

    def global_to_local(self, global_base, global_coordinate):
        """ """
        return global_to_local(global_base, global_coordinate)

    def local_to_global(self, global_base, local_coordinate):
        """ """
        return local_to_global(global_base, local_coordinate)

    def local_identity(self, global_base):
        """ """
        return numpy.zeros(18)

    def local_transition_cov(self, s, Q):
        """ """
        return local_transition_cov(s, Q)


def Rot(base, v):
    x = numpy.cross(base, v)
    return quaternion.as_rotation_matrix(quaternion.from_rotation_vector(x))


def Exp(base, v):
    return numpy.matmul(Rot(base, v), base)


def Log(base, p):
    v = numpy.cross(numpy.cross(base, p), base)
    s = numpy.linalg.norm(v, ord=2)
    if numpy.isclose(s, 0.0):
        # s == 0 implies v == (0, 0, 0)
        return numpy.zeros(3)
    else:
        # Need to worry about values near 0/0?
        return (numpy.arcsin(s) / s) * v


def pack_global_state(x, v, a, q, w, u):
    # project state onto manifold.
    q = q / numpy.linalg.norm(q, ord=2)
    w = w - numpy.dot(w, q) * q  # q is unit
    u = u - numpy.dot(u, q) * q
    # pack.
    return numpy.concatenate((x, v, a, q, w, u))


def pack_local_state(x, v, a, q, w, u):
    return numpy.concatenate((x, v, a, q, w, u))


def unpack_state(s):
    return s[0:3], s[3:6], s[6:9], s[9:12], s[12:15], s[15:18]


def evolve_state(s, dt):
    x, v, a, q, w, u = unpack_state(s)
    R = Rot(q, dt * w + 0.5 * dt * dt * u)
    return pack_global_state(
        x + dt * v + 0.5 * dt * dt * a,
        v + dt * a,
        a,
        numpy.matmul(R, q),
        numpy.matmul(R, w + dt * u),
        numpy.matmul(R, u),
    )


def local_transition_cov(s, Q):
    _, _, _, q, _, _ = unpack_state(s)
    P = numpy.eye(18)
    P[12:15, 12:15] -= numpy.outer(q, q)
    P[15:18, 15:18] -= numpy.outer(q, q)
    return P @ Q @ P.T


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Local coordinates are expanded around a base point (x, v, a, q, w, u).
# Exp and Log are used to put all the direction stuff into the tangent space
# of the sphere at q, using w as the origin.


def global_to_local(base, g):
    cx, cv, ca, cq, cw, cu = unpack_state(base)
    gx, gv, ga, gq, gw, gu = unpack_state(g)
    lx = gx - cx
    lv = gv - cv
    la = ga - ca
    lq = Log(cq, gq)
    R = Rot(cq, lq)
    lw = numpy.linalg.solve(R, gw) - cw
    lu = numpy.linalg.solve(R, gu) - cu
    return pack_local_state(lx, lv, la, lq, lw, lu)


def local_to_global(base, l):
    cx, cv, ca, cq, cw, cu = unpack_state(base)
    lx, lv, la, lq, lw, lu = unpack_state(l)
    gx = cx + lx
    gv = cv + lv
    ga = ca + la
    gq = Exp(cq, lq)
    R = Rot(cq, lq)
    gw = numpy.matmul(R, cw + lw)
    gu = numpy.matmul(R, cu + lu)
    return pack_global_state(gx, gv, ga, gq, gw, gu)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Observations are made up of a simultaneous measurement of both catheter
# coil positions. They have 3+3 = 6 parameters.


def pack_observation(a, b):
    return numpy.concatenate((a, b))


def unpack_observation(o):
    return o[0:3], o[3:6]


def observe_state(s, coil_offset):
    x, _, _, q, _, _ = unpack_state(s)
    return pack_observation(x + coil_offset * q, x - coil_offset * q)


def tip_from_state(s, tip_offset):
    x, _, _, q, _, _ = unpack_state(s)
    return x + tip_offset * q
