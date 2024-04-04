import time
from typing import Any, Callable

from distributed import Client, Future

from qsprpred.utils.parallel import JITParallelGenerator


class DaskJITGenerator(JITParallelGenerator):
    """Uses the `dask` library to parallelize the processing of an input generator.
    The main benefit of using `dask` is that it supports distributed
    computing across multiple machines.
    """

    def getPool(self) -> Any:
        return Client(n_workers=self.nWorkers, threads_per_worker=1)

    def checkResultAvailable(self, process: Future):
        time.sleep(0.1)
        return process.done()

    def getResult(self, process: Future) -> Any:
        return process.result()

    def checkProcess(self, process: Future):
        if process.status == "error":
            raise process.exception

    def handleException(self, process: Future, exception: Exception) -> Any:
        return exception

    def createJob(self, pool: Client, process_func: Callable, *args, **kwargs) -> Any:
        return pool.submit(
            process_func,
            *args,
            **kwargs
        )
