import os
import random


def _is_module_available(module_call):
    try:
        eval(module_call)
        return True
    except (ModuleNotFoundError, ImportError):
        return False


IS_NUMPY_AVAILABLE = _is_module_available("exec('import numpy as np')")
IS_TORCH_AVAILABLE = _is_module_available("exec('import torch')")


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if IS_NUMPY_AVAILABLE:
        import numpy as np

        np.random.seed(seed)
    if IS_TORCH_AVAILABLE:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
