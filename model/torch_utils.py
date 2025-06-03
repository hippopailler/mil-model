"""PyTorch model utility functions."""

import types
from types import SimpleNamespace
from typing import Dict, Generator, Iterable, List, Tuple, Union, Optional

import torch
import numpy as np
import slideflow as sf
import contextlib
from packaging import version
from pandas.core.frame import DataFrame
from scipy.special import softmax
from slideflow.stats import df_from_pred
from modules.errors import DatasetError
from util._init_ import log, ImgBatchSpeedColumn, no_scope
from rich.progress import Progress, TimeElapsedColumn, SpinnerColumn
from functools import reduce

def get_device(device: Optional[str] = None):
    if device is None and torch.cuda.is_available():
        return torch.device('cuda')
    elif (device is None
          and hasattr(torch.backends, 'mps')
          and torch.backends.mps.is_available()):
        return torch.device('mps')
    elif device is None:
        return torch.device('cpu')
    elif isinstance(device, str):
        return torch.device(device)
    else:
        return device
    
# -----------------------------------------------------------------------------

def xception(*args, **kwargs):
    import pretrainedmodels
    return pretrainedmodels.xception(*args, **kwargs)


def nasnetalarge(*args, **kwargs):
    import pretrainedmodels
    return pretrainedmodels.nasnetalarge(*args, **kwargs)