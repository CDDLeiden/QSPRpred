import time
from unittest import skipIf

import torch
from parameterized import parameterized

from qsprpred.extra.gpu.utils.parallel import TorchJITGenerator
from qsprpred.utils.parallel import batched_generator, ThreadsJITGenerator
from qsprpred.utils.testing.base import QSPRTestCase


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


class TestThreadedGeneratorsGPU(QSPRTestCase):
    """Test processing using a pool of threads."""

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
