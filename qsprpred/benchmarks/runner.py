import itertools
import logging
import os
import random
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock
from typing import Generator

import pandas as pd

from .settings.benchmark import BenchmarkSettings
from .replica import Replica

lock = Lock()


class BenchmarkRunner:

    class ReplicaException(Exception):

        def __init__(self, replica_id: int, exception: Exception):
            self.replica_id = replica_id
            self.exception = exception

    def __init__(
            self,
            settings: BenchmarkSettings,
            n_proc: int | None = None,
            data_dir: str = "./data",
            results_file: str = "./data/results.tsv"
    ):
        self.settings = settings
        self.n_proc = n_proc or os.cpu_count()
        self.results_file = results_file
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    @property
    def n_runs(self):
        """Returns the total number of benchmarking runs."""
        benchmark_settings = self.settings
        benchmark_settings.checkConsistency()
        ret = (benchmark_settings.n_replicas * len(benchmark_settings.data_sources)
               * len(benchmark_settings.descriptors) * len(
                    benchmark_settings.target_props)
               * len(benchmark_settings.prep_settings) * len(benchmark_settings.models)
               )
        if len(benchmark_settings.optimizers) > 0:
            ret *= len(benchmark_settings.optimizers)
        return ret

    def run(self, raise_errors=False) -> pd.DataFrame:
        # loop over replicas in parallel
        logging.info(f"Performing {self.n_runs} replica runs...")
        self.settings.toFile(f"{self.data_dir}/settings.json")
        with ProcessPoolExecutor(max_workers=self.n_proc) as executor:
            for result in executor.map(
                    self.run_replica,
                    self.iter_replicas()
            ):
                if result is not None and raise_errors:
                    raise result

        logging.info("Finished all replica runs.")
        return pd.read_table(self.results_file)

    def get_seed_list(self, seed: int) -> list[int]:
        """
        Get a list of seeds for the replicas.

        Args:
            seed(int): master seed to generate the list of seeds from

        Returns:
            list[int]: list of seeds for the replicas

        """
        random.seed(seed)
        return random.sample(range(2 ** 32 - 1), self.n_runs)

    def iter_replicas(self) -> Generator[Replica, None, None]:
        benchmark_settings = self.settings
        # generate all combinations of settings with itertools
        benchmark_settings.checkConsistency()
        indices = [x+1 for x in range(benchmark_settings.n_replicas)]
        optimizers = benchmark_settings.optimizers if len(benchmark_settings.optimizers) > 0 else [None]
        product = itertools.product(
            indices,
            [benchmark_settings.name],
            benchmark_settings.data_sources,
            benchmark_settings.descriptors,
            benchmark_settings.target_props,
            benchmark_settings.prep_settings,
            benchmark_settings.models,
            optimizers,
        )
        seeds = self.get_seed_list(benchmark_settings.random_seed)
        for idx, settings in enumerate(product):
            yield Replica(
                *settings,
                random_seed=seeds[idx],
                assessors=benchmark_settings.assessors
            )

    def run_replica(self, replica: Replica):
        try:
            with lock:
                df_results = None
                if os.path.exists(self.results_file):
                    df_results = pd.read_table(self.results_file)
                if df_results is not None and df_results.ReplicaID.isin([replica.id]).any():
                    logging.warning(f"Skipping {replica.id}")
                    return
                replica.create_dataset(reload=False)
            replica.prep_dataset()
            replica.init_model()
            replica.run_assessment()
            with lock:
                df_report = replica.create_report()
                df_report.to_csv(
                    self.results_file,
                    sep="\t",
                    index=False,
                    mode="a",
                    header=not os.path.exists(self.results_file)
                )
                logging.info(f"Finished {replica.id}.")
        except Exception as e:
            # TODO: make a custom exception for this
            traceback.print_exception(type(e), e, e.__traceback__)
            return self.ReplicaException(replica.id, e)
