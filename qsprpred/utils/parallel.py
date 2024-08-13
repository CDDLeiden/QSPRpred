import multiprocessing
import time
from abc import ABC, abstractmethod
from concurrent import futures
from concurrent.futures import Future
from typing import Iterable, Callable, Literal, Generator, Any

from pebble import ProcessFuture

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
    """An abstract class to facilitate parallel processing of an arbitrary generator."""

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


class JITParallelGenerator(ParallelGenerator, ABC):
    """An abstract class to facilitate JIT (Just In Time)
    parallel processing of an arbitrary generator.
    This is meant for situations where the result of the generator
    is too large to fit into memory or not yet known. Parallelization can be done
    over a pool of CPU or GPU workers. The generator will yield the results of the
    function applied in parallel to each item of a supplied generator.
    """

    def __init__(
            self,
            n_workers: int | None = None,
            worker_type: Literal["cpu", "gpu"] = "cpu",
            use_gpus: list[int] | None = None,
            jobs_per_gpu: int = 1
    ):
        """Configures the multiprocessing pool generator.

        Args:
            n_workers(int):
                Number of workers to use.
            worker_type(Literal["cpu", "gpu"]):
                The type of worker to use.
            use_gpus(list[int] | None):
                A list of GPU indices to use. Only applicable if `worker_type` is 'gpu'.
                If None, all available GPUs will be used.
            jobs_per_gpu(int):
                Number of jobs to run on each GPU.
        """
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

    @abstractmethod
    def getPool(self) -> Any:
        """Create the pool of workers consuming the generator.

        Returns:
            A pool object that can be used to apply the function
            to the generator items in parallel.
        """

    @abstractmethod
    def checkResultAvailable(self, process: Any) -> bool:
        """Check if the result of a process is available.

        Args:
            process(Any):
                The process object or a future to check for a result.

        Returns:
            `True` if the result is available, otherwise `False`.
        """

    @abstractmethod
    def getResult(self, process: Any) -> Any:
        """Get the result of a process.

        Args:
            process(Any):
                The process object or a future to get the result from.

        Returns:
            The result of the process.
        """

    @abstractmethod
    def checkProcess(self, process: Any):
        """A simple check of a process or future before a result
        is attempted to be retrieved.

        Args:
            process(Any):
                The process object or a future to check.

        Returns:
            `None` if the process is OK, otherwise raises an exception.

        Raises:
            Exception: If the process has a problem.
        """

    @abstractmethod
    def handleException(self, process: Any, exception: Exception) -> Any:
        """Handle an exception raised by a process. This is executed
        when the process raises an unexpected exception or the `checkProcess`
        method raises an exception.

        Args:
            process(Any):
                The process object or a future that raised the exception.
            exception(Exception):
                The exception raised by the process.

        Returns:
            The result to yield instead of the result of the process.
        """

    @abstractmethod
    def createJob(self, pool: Any, process_func: Callable, *args,
                  **kwargs) -> Any:
        """Submit a job to the pool that applies the function to a generator item.

        Args:
            pool(Any):
                The pool object to submit the job to.
            process_func(Callable):
                The function to apply to the input arguments.
            args(tuple):
                Positional arguments to pass to the function.
                The first argument should be the item from the generator.
            kwargs(dict):
                Additional keyword arguments to pass to the function.

        Returns:
            The process object or future that was submitted to the pool.
        """

    def make(
            self,
            generator: Generator,
            process_func: Callable,
            *args,
            **kwargs
    ) -> Generator:
        """A parallel "JIT (Just In Time)" generator that
        yields the results of a function
        applied in parallel to each item of a supplied generator.
        The advantage of this JIT
        implementation is that it only evaluates the next item in the input generator
        as soon as it is needed for the calculation. This means that only one item from
        the iterable is loaded into memory at a time.

        The generator also supports timeout for each job. The "pebble"
        pool_type can be used to support this feature. In all other cases,
        the "multiprocessing" pool_type is sufficient.

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
                        *input_args,
                        **dict(**kwargs, gpu=gpu) if gpu is not None else kwargs
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
                        is_ready = self.checkResultAvailable(process)
                        if is_ready:
                            result = self.getResult(process)
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


class ThreadsJITGenerator(JITParallelGenerator):
    """This class uses the `concurrent.futures.ThreadPoolExecutor` to parallelize
    the processing of an input generator. Note that threads in Python are not
    truly parallel due to the Global Interpreter Lock (GIL). However, this can
    still be useful for I/O-bound tasks or tasks that are not CPU-bound downstream.
    """

    def getPool(self) -> Any:
        from concurrent.futures import ThreadPoolExecutor
        return ThreadPoolExecutor(max_workers=self.nWorkers)

    def checkResultAvailable(self, process: Future):
        time.sleep(0.1)
        return process.done()

    def getResult(self, process: Any):
        return process.result()

    def checkProcess(self, process: Any):
        pass

    def handleException(self, process: Any, exception: Exception) -> Any:
        return exception

    def createJob(self, pool: Any, process_func: Callable, *args,
                  **kwargs) -> Any:
        return pool.submit(
            process_func,
            *args,
            **kwargs
        )


class MultiprocessingJITGenerator(JITParallelGenerator):
    """
    A parallel generator that uses the `multiprocessing` module to parallelize
    the processing of an input generator. This is useful when the input generator
    is too large to fit into memory and needs to be processed in parallel over
    a pool of workers.
    """

    def getPool(self):
        return multiprocessing.Pool(
            processes=self.nWorkers
        )

    def checkResultAvailable(self, process):
        try:
            process.wait(0.1)
            # check if process is done
            if not process.ready():
                # not finished, return nothing
                return False
            else:
                return True
        except (futures.TimeoutError, TimeoutError):
            # not done yet, return nothing
            return False

    def getResult(self, process):
        return process.get()

    def checkProcess(self, process):
        pass

    def handleException(self, process, exception):
        return exception

    def createJob(self, pool, process_func, *args, **kwargs):
        return pool.apply_async(
            process_func,
            args,
            kwargs
        )


class PebbleJITGenerator(JITParallelGenerator):
    """Uses the `pebble` library to parallelize the processing of an input generator.
    The main benefit of using `pebble` is that it supports timeouts for each job,
    which makes it easy to handle jobs that take too long to process.
    """

    def __init__(
            self,
            n_workers: int | None = None,
            worker_type: Literal["cpu", "gpu"] = "cpu",
            use_gpus: list[int] | None = None,
            jobs_per_gpu: int = 1,
            timeout: int | None = None
    ):
        """Configures the multiprocessing pool generator.

        Args:
            n_workers(int):
                Number of workers to use.
            worker_type(Literal["cpu", "gpu"]):
                The type of worker to use.
            use_gpus(list[int] | None):
                A list of GPU indices to use. Only applicable if `worker_type` is 'gpu'.
                If None, all available GPUs will be used.
            jobs_per_gpu(int):
                Number of jobs to run on each GPU.
            timeout(int | None):
                A timeout threshold in seconds. Processes that exceed this threshold will
                be terminated and a `TimeoutError` will be returned.
        """
        super().__init__(
            n_workers=n_workers,
            worker_type=worker_type,
            use_gpus=use_gpus,
            jobs_per_gpu=jobs_per_gpu
        )
        self.timeout = timeout

    def getPool(self) -> Any:
        try:
            from pebble import ProcessPool
            return ProcessPool(max_workers=self.nWorkers)
        except ImportError:
            raise ImportError(
                "Failed to import pool type 'pebble'. Install it first.")

    def checkResultAvailable(self, process: ProcessFuture):
        time.sleep(0.1)
        return process.done()

    def getResult(self, process: Any):
        return process.result()

    def checkProcess(self, process: Any):
        # check if job timed out
        if process.done() and type(process._exception) in [
            TimeoutError,
            futures.TimeoutError
        ]:
            raise process._exception

    def handleException(self, process: Any, exception: Exception) -> Any:
        return process._exception

    def createJob(self, pool: Any, process_func: Callable, *args, **kwargs) -> Any:
        return pool.schedule(
            process_func,
            args=args,
            kwargs=kwargs,
            timeout=self.timeout
        )
