import time

from qsprpred.extra.utils.parallel import DaskJITGenerator
from qsprpred.utils.parallel import batched_generator
from qsprpred.utils.testing.base import QSPRTestCase


class TestDaskGenerator(QSPRTestCase):

    @staticmethod
    def func(x):
        time.sleep(1)
        return x ** 2

    @staticmethod
    def func_batched(x):
        time.sleep(1)
        return [i ** 2 for i in x]

    def testSimple(self):
        generator = (x for x in range(10))
        p_generator = DaskJITGenerator(self.nCPU)
        self.assertListEqual(
            [x ** 2 for x in range(10)],
            sorted(p_generator(
                generator,
                self.func,
            ))
        )

    def testBatched(self):
        generator = batched_generator(range(10), 2)
        p_generator = DaskJITGenerator(self.nCPU)
        self.assertListEqual(
            [[0, 1], [4, 9], [16, 25], [36, 49], [64, 81]],
            sorted(p_generator(
                generator,
                self.func_batched
            ))
        )
