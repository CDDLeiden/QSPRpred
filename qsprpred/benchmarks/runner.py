import logging
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock

import pandas as pd

from .settings import BenchmarkSettings
from .replica import Replica

lock = Lock()


class BenchmarkRunner:

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

    def run(self):
        # loop over replicas in parallel
        with ProcessPoolExecutor(max_workers=self.n_proc) as executor:
            for result in executor.map(
                    self.run_replica,
                    self.settings.iter_replicas()
            ):
                if result is not None:
                    logging.error(f"Something went wrong for {result[0]}: ", result[1])
                    logging.exception(result[1])

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
            df_report = replica.create_report()
            with lock:
                df_report.to_csv(
                    self.results_file,
                    sep="\t",
                    index=False,
                    mode="a",
                    header=not os.path.exists(self.results_file)
                )
                logging.info(f"Finished {replica.id}.")
        except Exception as e:
            logging.error(f"Error in {replica.id}:")
            logging.exception(e)
            return replica.id, e
