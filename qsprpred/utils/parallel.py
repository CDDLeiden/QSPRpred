import multiprocessing
from concurrent import futures
from typing import Iterable, Callable, Literal, Generator

from qsprpred.logs import logger


def batched_generator(iterable: Iterable, batch_size: int) -> Generator:
    """
    A simple generator that batches inputs from a supplied iterable.

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


def parallel_jit_generator(
        generator: Generator,
        process_func: Callable,
        n_cpus: int,
        pool_type: Literal["pebble", "multiprocessing"] = "multiprocessing",
        timeout: int | None = None,
        args: tuple | None = None,
        kwargs: dict | None = None
) -> Generator:
    """
    A parallel 'JIT (Just In Time)' generator that yields the results of a function
    applied in parallel to each item of a supplied generator. The advantage of this JIT
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
        n_cpus(int):
            Number of CPUs to use.
        pool_type(Literal["pebble", "multiprocessing"]):
            The type of pool to use.
        timeout(int | None):
            A timeout threshold in seconds. Processes that exceed this threshold will
            be terminated and a `TimeoutError` will be returned.
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
    if pool_type not in ["pebble", "multiprocessing"]:
        raise ValueError(f"The 'pool_type' must be one of 'pebble' "
                         f"or 'multiprocessing', got {pool_type} instead.")
    if pool_type != "pebble" and timeout is not None:
        raise ValueError(f"The 'timeout' argument is only supported "
                         f"for 'pebble' pools, got {pool_type} instead.")
    pool_type_to_pool = {
        "multiprocessing": multiprocessing.Pool(processes=n_cpus)
    }
    if pool_type == "pebble":
        try:
            from pebble import ProcessPool
            pool_type_to_pool["pebble"] = ProcessPool(max_workers=n_cpus)
        except ImportError:
            raise ImportError("Failed to import pool type 'pebble'. Install it first.")
    with pool_type_to_pool[pool_type] as pool:
        queue = []
        done = False
        while not done or queue:
            try:
                # take a slice of the generator
                input_args = [next(generator), *list(args)]
                # add our next slice to the pool
                if pool_type == "pebble":
                    queue.append(pool.schedule(
                        process_func,
                        args=input_args,
                        kwargs=kwargs,
                        timeout=timeout
                    ))
                else:
                    queue.append(pool.apply_async(
                        process_func,
                        input_args,
                        kwargs
                    ))
            except StopIteration:
                # no more data, clear out the slice generator
                done = True
            # wait for a free worker or until all remaining workers finish
            logger.debug(f"Waiting for {len(queue)} workers to finish...")
            while queue and ((len(queue) >= n_cpus) or done):
                # grab a process response from the top
                process = queue.pop(0)
                # check if job timed out and return the exception as a result
                if pool_type == "pebble" \
                        and process.done() \
                        and type(process._exception ) in [
                            TimeoutError,
                            futures.TimeoutError
                        ]:
                    yield process._exception
                    # make sure to pop the next item from the generator
                    break
                # check if result available
                try:
                    if pool_type == "pebble":
                        # get the result, timeout error if not result yet available
                        result = process.result(timeout=0.1)
                        yield result
                    else:
                        process.wait(0.1)
                        # check if process is done
                        if not process.ready():
                            # not finished, add it back to queue
                            queue.append(process)
                        else:
                            # finished, yield the result
                            yield process.get()
                except (futures.TimeoutError, TimeoutError):
                    # not done yet, add it back to the queue
                    queue.append(process)
                except Exception as exp:
                    # something went wrong, log and yield the exception
                    logger.error(type(exp))
                    logger.error(repr(process))
                    if pool_type == "pebble":
                        yield process._exception
                    else:
                        yield exp
                    break  # make sure to pop the next item from the generator
