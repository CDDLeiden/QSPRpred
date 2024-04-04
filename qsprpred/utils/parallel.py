import multiprocessing
from abc import ABC, abstractmethod
from concurrent import futures
from typing import Iterable, Callable, Literal, Generator

from qsprpred.logs import logger


def batched_generator(iterable: Iterable, batch_size: int) -> Generator:
    """
    A simple helper generator that batches inputs from a supplied `Iterable`.

    Args:
        iterable: An iterable object to batch with the generator.
        batch_size: Number of items to include in each batch.

    Returns:
        A generator that yields batches of the input generator one at a time.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class ParallelGenerator(ABC):
    """
    An abstract class to facilitate parallel processing of an arbitrary generator.
    This is meant for situations where the generator is too large to fit into memory,
    but can also be used for any situation where parallel distribution over a pool
    of workers (GPUs or CPUs) is needed.
    """

    @abstractmethod
    def make(
            self,
            generator: Generator,
            process_func: Callable,
            *args,
            **kwargs
    ) -> Generator:
        """
        This method is used to wrap an input generator or an iterable
        (needs to support `next()`). The method should evaluate one item of
        the generator at a time and apply the `process_func` to it. This
        is done in parallel over a pool of workers. The method should return
        a generator that yields the results of the function applied in parallel
        as soon as the results are available. Therefore, the user can simply
        iterate over the generator to get the results.

        Args:
            generator: An iterable object to apply the function to.
            process_func: The function to apply to each item in the iterable.

        Returns:
            A generator that yields the results of the function applied to each item
            in the iterable in parallel.
        """

    def __call__(
            self,
            generator: Generator,
            process_func: Callable,
            *args,
            **kwargs
    ) -> Generator:
        """
        This method is used to wrap the `make` method and call it with the
        supplied arguments. This is a convenience method to make the class
        callable.

        Args:
            generator: An iterable object to apply the function to.
            process_func: The function to apply to each item in the iterable.

        Returns:
            A generator that yields the results of the function applied to each item
            in the iterable in parallel.
        """
        return self.make(generator, process_func, *args, **kwargs)


class MultiprocessingPoolGenerator(ParallelGenerator):
    """
    A parallel generator that uses the `multiprocessing` module to parallelize
    the processing of an input generator. This is useful when the input generator
    is too large to fit into memory and needs to be processed in parallel over
    a pool of workers.
    """

    def __init__(
            self,
            n_workers: int | None = None,
            pool_type: Literal[
                "pebble", "multiprocessing", "torch", "threads"] = "multiprocessing",
            timeout: int | None = None,
            worker_type: Literal["cpu", "gpu"] = "cpu",
            use_gpus: list[int] | None = None,
            jobs_per_gpu: int = 1
    ):
        """Configures the multiprocessing pool generator.

        Args:
            n_workers(int):
                Number of workers to use.
            pool_type(Literal["pebble", "multiprocessing", "torch", "threads"]):
                The type of pool to use.
            timeout(int | None):
                A timeout threshold in seconds. Processes that exceed this threshold will
                be terminated and a `TimeoutError` will be returned.
            worker_type(Literal["cpu", "gpu"]):
                The type of worker to use.
            use_gpus(list[int] | None):
                A list of GPU indices to use. Only applicable if `worker_type` is 'gpu'.
                If None, all available GPUs will be used.
            jobs_per_gpu(int):
                Number of jobs to run on each GPU.
        """
        self.poolType = pool_type
        self.timeout = timeout
        if self.poolType not in ["pebble", "multiprocessing", "torch", "threads"]:
            raise ValueError(f"The 'pool_type' must be one of 'pebble' "
                             f"or 'multiprocessing', got {self.poolType} instead.")
        if self.poolType not in ["pebble"] and self.timeout is not None:
            raise ValueError(f"The 'timeout' argument is only supported "
                             f"for 'pebble' pool, "
                             f"got {self.poolType} instead.")
        self.workerType = worker_type
        if self.workerType not in ["cpu", "gpu"]:
            raise ValueError(f"The 'worker_type' must be one of 'cpu' "
                             f"or 'gpu', got {self.workerType} instead.")
        if self.workerType == "gpu" and n_workers is not None:
            logger.warning(
                "The 'n_workers' argument is ignored when 'worker_type' is 'gpu'."
                "The number of workers is determined by 'use_gpus' and 'jobs_per_gpu'."
            )
        if self.workerType == "gpu":
            self.jobsPerGpu = jobs_per_gpu
            if use_gpus is not None:
                self.useGpus = sorted(set(use_gpus))
                self.useGpus = self.useGpus * self.jobsPerGpu
            else:
                logger.warning(
                    "No GPUs specified. Using only the first gpu with index 0."
                    "Only one job will be run on this GPU at a time."
                    "If you want to use multiple GPUs, specify them using 'use_gpus' "
                    "or run more jobs per GPU with 'jobs_per_gpu'."
                )
                self.useGpus = [0] * self.jobsPerGpu
            self.nWorkers = len(self.useGpus)
        else:
            self.useGpus = []
            self.jobsPerGpu = None
            self.nWorkers = n_workers or multiprocessing.cpu_count()

    def getPool(self):
        """Get the pool object based on the pool type.


        Returns:
            A process pool object or a thread pool object.
        """
        if self.poolType == "pebble":
            try:
                from pebble import ProcessPool
                return ProcessPool(max_workers=self.nWorkers)
            except ImportError:
                raise ImportError(
                    "Failed to import pool type 'pebble'. Install it first.")
        elif self.poolType == "torch":
            from torch.multiprocessing import Pool, set_start_method
            set_start_method("spawn", force=True)
            return Pool(self.nWorkers)
        elif self.poolType == "threads":
            from concurrent.futures import ThreadPoolExecutor
            return ThreadPoolExecutor(max_workers=self.nWorkers)
        else:
            return multiprocessing.Pool(
                processes=self.nWorkers
            )

    def checkResultAvailable(self, process):
        try:
            if self.poolType in ("pebble", "threads"):
                # get the result, timeout error if not result yet available
                return process.result(timeout=0.1)
            else:
                process.wait(0.1)
                # check if process is done
                if not process.ready():
                    # not finished, return nothing
                    return
                else:
                    # finished, yield the result
                    return process.get()
        except (futures.TimeoutError, TimeoutError):
            # not done yet, return nothing
            return

    def checkProcess(self, process):
        # check if job timed out and return the exception as a result
        if self.poolType == "pebble" \
                and process.done() \
                and type(process._exception) in [
            TimeoutError,
            futures.TimeoutError
        ]:
            raise process._exception

    def handleException(self, process, exception):
        if self.poolType == "pebble":
            return process._exception
        else:
            return exception

    def createJob(self, pool, process_func, args, kwargs):
        if self.poolType == "pebble":
            return pool.schedule(
                process_func,
                args=args,
                kwargs=kwargs,
                timeout=self.timeout
            )
        elif self.poolType == "threads":
            return pool.submit(
                process_func,
                *args,
                **kwargs
            )
        else:
            return pool.apply_async(
                process_func,
                args,
                kwargs
            )

    def make(
            self,
            generator: Generator,
            process_func: Callable,
            *args,
            **kwargs
    ) -> Generator:
        """A parallel 'JIT (Just In Time)' generator that
        yields the results of a function
        applied in parallel to each item of a supplied generator.
        The advantage of this JIT
        implementation is that it only evaluates the next item in the input generator
        as soon as it is needed for the calculation. This means that only one item from
        the iterable is loaded into memory at a time.

        The generator also supports timeout for each job. The 'pebble'
        pool_type can be used to support this feature. In all other cases,
        the 'multiprocessing' pool_type is sufficient.

        Args:
            generator(SupportsNext):
                An iterable object to apply the function to.
            process_func(Callable):
                The function to apply to each item in the iterable.
            args(tuple | None):
                Additional positional arguments to pass to the function.
            kwargs(dict | None):
                Additional keyword arguments to pass to the function.

        Returns:
            A generator that yields the results of the function applied to each item in the
            iterable. If a timeout is specified, a `TimeoutError` will be returned instead.
        """
        args = args or ()
        kwargs = kwargs or {}
        gpu_pool = self.useGpus
        pool = self.getPool()
        with pool as pool:
            queue = []
            gpus_to_jobs = {}
            done = False
            while not done or queue:
                try:
                    # take a slice of the generator
                    input_args = [next(generator), *list(args)]
                    # add our next slice to the pool
                    gpu = None
                    if self.workerType == "gpu":
                        # get the next free GPU
                        gpu = gpu_pool.pop(0)
                    job = self.createJob(
                        pool,
                        process_func,
                        input_args,
                        dict(**kwargs, gpu=gpu) if gpu is not None else kwargs
                    )
                    queue.append(job)
                    if self.workerType == "gpu":
                        gpus_to_jobs[job] = gpu
                except StopIteration:
                    # no more data, clear out the slice generator
                    done = True
                # wait for a free worker or until all remaining workers finish
                logger.debug(f"Waiting for {len(queue)} workers to finish...")
                while queue and ((len(queue) >= self.nWorkers) or done):
                    # grab a process response from the top
                    process = queue.pop(0)
                    try:
                        # check process status
                        self.checkProcess(process)
                        # check if result available
                        result = self.checkResultAvailable(process)
                        if result is not None:
                            logger.debug(
                                f"Yielding result: {result} for process: {process}"
                            )
                            yield result
                            if self.workerType == "gpu":
                                gpu_pool.append(gpus_to_jobs[process])
                                logger.debug(
                                    f"GPU {gpus_to_jobs[process]} is free.")
                                logger.debug(f"Free GPUs now: {gpu_pool}")
                                logger.debug(
                                    f"Deleting job {process} from gpus_to_jobs.")
                                del gpus_to_jobs[process]
                            break  # make sure to pop the next item from the generator
                        else:
                            # result not available yet, put process back in the queue
                            queue.append(process)
                    except Exception as exp:
                        # something went wrong, log and yield the exception
                        logger.exception(repr(exp))
                        yield self.handleException(process, exp)
                        if self.workerType == "gpu":
                            gpu_pool.append(gpus_to_jobs[process])
                            del gpus_to_jobs[process]
                        break  # make sure to pop the next item from the generator
