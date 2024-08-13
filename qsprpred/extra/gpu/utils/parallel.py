from qsprpred.utils.parallel import MultiprocessingJITGenerator


class TorchJITGenerator(MultiprocessingJITGenerator):
    """A variant of the `MultiprocessingPoolGenerator`
    that uses the `torch.multiprocessing.Pool`
    instead of the standard `multiprocessing.Pool`.
    This is needed when the parallel
    processing is done with PyTorch tensors or models,
    which require the `torch.multiprocessing` and using the
    `spawn` start method.
    """

    def getPool(self):
        from torch.multiprocessing import Pool, set_start_method
        set_start_method("spawn", force=True)
        return Pool(self.nWorkers)
