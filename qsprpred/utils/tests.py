import time
from concurrent import futures

from parameterized import parameterized

from .parallel import parallel_jit_generator, batched_generator
from .testing.base import QSPRTestCase


class TestParallel(QSPRTestCase):

    @staticmethod
    def func(x):
        return x**2

    @staticmethod
    def func_batched(x):
        return [i**2 for i in x]

    @staticmethod
    def func_timeout(x):
        time.sleep(x)
        return x**2

    @staticmethod
    def func_args(x, *args, **kwargs):
        return x, args, kwargs

    @parameterized.expand([
        (None, "multiprocessing"),
        (1, "pebble"),
    ])
    def test_simple(self, timeout, pool_type):
        generator = (x for x in range(10))
        self.assertListEqual(
            [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
            list(parallel_jit_generator(
                    generator,
                    self.func,
                    self.nCPU,
                    pool_type=pool_type,
                    timeout=timeout,
                 ))
        )

    @parameterized.expand([
        (None, "multiprocessing"),
        (1, "pebble"),
    ])
    def test_batched(self, timeout, pool_type):
        generator = batched_generator(range(10), 2)
        self.assertListEqual(
            [[0, 1], [4, 9], [16, 25], [36, 49], [64, 81]],
            list(parallel_jit_generator(
                generator,
                self.func_batched,
                self.nCPU,
                pool_type,
                timeout=timeout,
            ))
        )

    def test_timeout(self):
        generator = (x for x in [1, 2, 10])
        timeout = 4
        result = list(parallel_jit_generator(
            generator,
            self.func_timeout,
            self.nCPU,
            pool_type="pebble",
            timeout=timeout
        ))
        self.assertListEqual([1, 4], result[0:-1])
        self.assertIsInstance(result[-1], futures.TimeoutError)
        self.assertTrue(str(timeout) in str(result[-1]))

    @parameterized.expand([
        ((0,), {"A": 1}),
        (None, {"A": 1}),
        ((0,), None),
    ])
    def test_arguments(self, args, kwargs):
        generator = (x for x in range(10))
        for idx, result in enumerate(parallel_jit_generator(
            generator,
            self.func_args,
            self.nCPU,
            args=args,
            kwargs=kwargs,
        )):
            self.assertEqual(
                (idx, args if args else (), kwargs if kwargs else {}),
                result
            )
