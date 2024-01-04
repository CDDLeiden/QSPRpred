import multiprocessing
from concurrent import futures
from typing import Iterable, Callable, Literal, Generator


def batched_generator(iterable: Iterable, batch_size: int) -> Generator:
    """
    A simple generator that batches inputs from a supplied iterable.

    Args:
        iterable: An iterable object to batch with the generator.
        batch_size: Number of items to include in each batch.

    Returns:
        A generator that yields batches of the input generator.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parallel_generator(
        generator: Generator,
        process_func: Callable,
        n_cpus: int,
        pool_type: Literal["pebble", "multiprocessing"] = "multiprocessing",
        timeout: int | None = None,
        args: tuple | None = None,
        kwargs: dict | None = None
) -> Generator:
    """
    A generator that yields the results of a function applied in parallel to each item
    in an iterable. The advantage of this implementation is that it yields results as
    soon as they are available and that only one item from the iterable is loaded into
    memory at a time. If a timeout for a long-running process is required, the 'pebble'
    pool_type can be used.

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
                    # make sure to pop the next item in the queue
                    break
                # check if result available
                try:
                    if pool_type == "pebble":
                        result = process.result(timeout=0.1)
                        yield result
                    else:
                        process.wait(0.1)
                        if not process.ready():
                            # if process not finished, add it back to queue
                            queue.append(process)
                        else:
                            # yield the result when available
                            yield process.get()
                except (futures.TimeoutError, TimeoutError):
                    # add it back to the queue
                    queue.append(process)
                except Exception as exp:
                    print(type(exp))
                    print(repr(process))
                    if pool_type == "pebble":
                        yield process._exception
                    else:
                        yield exp
                    break  # make sure to pop the next item in the queue
