import time
from concurrent import futures

from parameterized import parameterized

from .parallel import batched_generator, MultiprocessingJITGenerator, \
    PebbleJITGenerator, ThreadsJITGenerator
from .testing.base import QSPRTestCase


class TestMultiProcGenerators(QSPRTestCase):

    @staticmethod
    def func(x):
        return x ** 2

    @staticmethod
    def func_batched(x):
        return [i ** 2 for i in x]

    @staticmethod
    def func_timeout(x):
        time.sleep(x)
        return x ** 2

    @staticmethod
    def func_args(x, *args, **kwargs):
        return x, args, kwargs

    @parameterized.expand([
        (None, MultiprocessingJITGenerator),
        (1, PebbleJITGenerator),
        (None, MultiprocessingJITGenerator),
    ])
    def testSimple(self, timeout, pool_type):
        generator = (x for x in range(10))
        p_generator = pool_type(self.nCPU) if not timeout else pool_type(
            self.nCPU, timeout=timeout
        )
        self.assertListEqual(
            [x ** 2 for x in range(10)],
            sorted(p_generator(
                generator,
                self.func,
            ))
        )

    @parameterized.expand([
        (None, MultiprocessingJITGenerator),
        (1, PebbleJITGenerator),
        (None, MultiprocessingJITGenerator),
    ])
    def testBatched(self, timeout, pool_type):
        generator = batched_generator(range(10), 2)
        p_generator = pool_type(self.nCPU) if not timeout else pool_type(
            self.nCPU, timeout=timeout
        )
        self.assertListEqual(
            [[0, 1], [4, 9], [16, 25], [36, 49], [64, 81]],
            sorted(p_generator(
                generator,
                self.func_batched
            ))
        )

    def testTimeout(self):
        generator = (x for x in [1, 2, 10])
        timeout = 4
        p_generator = PebbleJITGenerator(self.nCPU, timeout=timeout)
        result = list(p_generator(
            generator,
            self.func_timeout
        ))
        self.assertListEqual([1, 4], result[0:-1])
        self.assertIsInstance(result[-1], futures.TimeoutError)
        self.assertTrue(str(timeout) in str(result[-1]))

    @parameterized.expand([
        ((0,), {"A": 1}, MultiprocessingJITGenerator),
        (None, {"A": 1}, MultiprocessingJITGenerator),
        ((0,), None, MultiprocessingJITGenerator),
    ])
    def testArgs(self, args, kwargs, pool_type):
        generator = (x for x in range(10))
        p_generator = pool_type(self.nCPU)
        result = list(p_generator(
            generator,
            self.func_args,
            *args or (),
            **kwargs or {},
        ))
        result = sorted(result, key=lambda x: x[0])
        for idx, res in enumerate(result):
            self.assertEqual(
                (idx, args if args else (), kwargs if kwargs else {}),
                res
            )


class TestThreadedGenerators(QSPRTestCase):
    """Test processing using a pool of threads."""

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
        p_generator = ThreadsJITGenerator(self.nCPU)
        self.assertListEqual(
            [x ** 2 for x in range(10)],
            sorted(p_generator(
                generator,
                self.func,
            ))
        )

    def testBatched(self):
        generator = batched_generator(range(10), 2)
        p_generator = ThreadsJITGenerator(self.nCPU)
        self.assertListEqual(
            [[0, 1], [4, 9], [16, 25], [36, 49], [64, 81]],
            sorted(p_generator(
                generator,
                self.func_batched
            ))
        )
