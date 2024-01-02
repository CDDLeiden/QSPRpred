try:
    import torch
    # set default number of threads to 1
    torch.set_num_threads(1)
    # set default device to GPU if available
    DEFAULT_TORCH_DEVICE = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    DEFAULT_TORCH_GPUS = (0,)
except ImportError:
    pass
