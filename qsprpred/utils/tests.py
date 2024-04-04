import time
from concurrent import futures
from unittest import skipIf

import torch
from parameterized import parameterized

from .parallel import batched_generator, PebbleJITGenerator, \
    MultiprocessingJITGenerator, TorchJITGenerator, ThreadsJITGenerator
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
        (None, TorchJITGenerator),
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
        (None, TorchJITGenerator),
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
        (None, {"A": 1}, PebbleJITGenerator),
        ((0,), None, ThreadsJITGenerator),
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


@skipIf(not torch.cuda.is_available(), "CUDA not available. Skipping...")
class TestMultiGPUGenerators(QSPRTestCase):

    @staticmethod
    def func(x, gpu=None):
        assert gpu is not None
        device = torch.device(f"cuda:{gpu}")
        ret = (torch.tensor([x], device=device) ** 2).item()
        time.sleep(1)
        return ret

    @staticmethod
    def func_batched(x, gpu=None):
        assert gpu is not None
        device = torch.device(f"cuda:{gpu}")
        time.sleep(1)
        return (torch.tensor(x, device=device) ** 2).tolist()

    @parameterized.expand([
        (1,),
        (2,),
    ])
    def testSimple(self, jobs_per_gpu):
        generator = (x for x in range(10))
        p_generator = TorchJITGenerator(
            len(self.GPUs),
            use_gpus=self.GPUs,
            jobs_per_gpu=jobs_per_gpu,
            worker_type="gpu"
        )
        self.assertListEqual(
            [x ** 2 for x in range(10)],
            sorted(p_generator(
                generator,
                self.func,
            ))
        )

    @parameterized.expand([
        (1,),
        (2,),
    ])
    def testBatched(self, jobs_per_gpu):
        generator = batched_generator(range(10), 2)
        p_generator = TorchJITGenerator(
            len(self.GPUs),
            use_gpus=self.GPUs,
            jobs_per_gpu=jobs_per_gpu,
            worker_type="gpu"
        )
        self.assertListEqual(
            [[0, 1], [4, 9], [16, 25], [36, 49], [64, 81]],
            sorted(p_generator(
                generator,
                self.func_batched
            ))
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

    @staticmethod
    def gpu_func(x, gpu=None):
        assert gpu is not None
        device = torch.device(f"cuda:{gpu}")
        ret = (torch.tensor([x], device=device) ** 2).item()
        time.sleep(1)
        return ret

    @staticmethod
    def gpu_func_batched(x, gpu=None):
        assert gpu is not None
        device = torch.device(f"cuda:{gpu}")
        time.sleep(1)
        return (torch.tensor(x, device=device) ** 2).tolist()

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

    @skipIf(not torch.cuda.is_available(), "CUDA not available. Skipping...")
    def testSimpleGPU(self):
        generator = (x for x in range(10))
        p_generator = ThreadsJITGenerator(
            len(self.GPUs),
            use_gpus=self.GPUs,
            worker_type="gpu",
            jobs_per_gpu=2
        )
        self.assertListEqual(
            [x ** 2 for x in range(10)],
            sorted(p_generator(
                generator,
                self.gpu_func,
            ))
        )

    @skipIf(not torch.cuda.is_available(), "CUDA not available. Skipping...")
    def testBatchedGPU(self):
        generator = batched_generator(range(10), 2)
        p_generator = ThreadsJITGenerator(
            len(self.GPUs),
            use_gpus=self.GPUs,
            worker_type="gpu"
        )
        self.assertListEqual(
            [[0, 1], [4, 9], [16, 25], [36, 49], [64, 81]],
            sorted(p_generator(
                generator,
                self.gpu_func_batched
            ))
        )
