import unittest
import numpy
import numpy.testing as npt

from statespace import (
    Exp,
    Log,
    unpack_state,
    pack_state,
    evolve_state,
    observe_state,
    unpack_observation,
    pack_observation,
    global_to_local,
    local_to_global,
)


class TestGeometricFuncs(unittest.TestCase):
    def setUp(self):
        self.a = numpy.array([1, 0, 0])
        self.b = numpy.array([0, 1, 0])
        self.ab = numpy.array([0, 0.5 * numpy.pi, 0])
        self.ba = numpy.array([0.5 * numpy.pi, 0, 0])

    def test_exp_and_log(self):
        b = Exp(self.a, self.ab)
        npt.assert_almost_equal(self.b, b)

        ab = Log(self.a, self.b)
        npt.assert_almost_equal(self.ab, ab)

        a = Exp(self.b, self.ba)
        npt.assert_almost_equal(self.a, a)

        ba = Log(self.b, self.a)
        npt.assert_almost_equal(self.ba, ba)


class TestGlobalRepr(unittest.TestCase):
    def setUp(self):
        self.s = numpy.array([1, 2, 3, 4, 5, 6, 1, 0, 0, 0, 1, 0])
        self.x = numpy.array([1, 2, 3])
        self.v = numpy.array([4, 5, 6])
        self.q = numpy.array([1, 0, 0])
        self.w = numpy.array([0, 1, 0])

        self.s2 = numpy.array(
            [
                5,
                7,
                9,
                4,
                5,
                6,
                numpy.cos(1),
                numpy.sin(1),
                0,
                -numpy.sin(1),
                numpy.cos(1),
                0,
            ]
        )

    def test_pack(self):
        packed = pack_state(self.x, self.v, self.q, self.w)
        npt.assert_almost_equal(self.s, packed)

    def test_unpack(self):
        x, v, q, w = unpack_state(self.s)
        npt.assert_almost_equal(self.x, x)
        npt.assert_almost_equal(self.v, v)
        npt.assert_almost_equal(self.q, q)
        npt.assert_almost_equal(self.w, w)

    def test_evolve(self):
        s2 = evolve_state(self.s, 1)
        npt.assert_almost_equal(self.s2, s2)


class TestLocalRepr(unittest.TestCase):
    def setUp(self):
        self.s = numpy.array([1, 2, 3, 4, 5, 6, 1, 0, 0, 0, 1, 0])

    def test_global_to_local_id(self):
        l = global_to_local(self.s, self.s)
        npt.assert_almost_equal(l, numpy.zeros(12))

    def test_local_to_global_id(self):
        g = local_to_global(self.s, numpy.zeros(12))
        npt.assert_almost_equal(g, self.s)


class TestObserveRepr(unittest.TestCase):
    def setUp(self):
        self.offset = 1.0
        self.s = numpy.array([1, 2, 3, 4, 5, 6, 1, 0, 0, 0, 1, 0])
        self.o = numpy.array([2, 2, 3, 0, 2, 3])
        self.a = numpy.array([2, 2, 3])
        self.b = numpy.array([0, 2, 3])

    def test_pack(self):
        o = pack_observation(self.a, self.b)
        npt.assert_almost_equal(self.o, o)

    def test_unpack(self):
        a, b = unpack_observation(self.o)
        npt.assert_almost_equal(self.a, a)
        npt.assert_almost_equal(self.b, b)

    def test_observe(self):
        o = observe_state(self.s, self.offset)
        npt.assert_almost_equal(self.o, o)


if __name__ == "__main__":
    unittest.main()
