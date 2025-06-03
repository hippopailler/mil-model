"""Multiple-instance learning (MIL) models"""

from .att_mil import Attention_MIL, MultiModal_Attention_MIL, UQ_MultiModal_Attention_MIL, MultiModal_Mixed_Attention_MIL
from .transmil import TransMIL
from util._init_ import is_torch_model_path, torch_available
from typing import Any, Dict, List


def is_torch_tensor(arg: Any) -> bool:
    """Checks if the given object is a Tensorflow Tensor."""
    if torch_available:
        import torch
        return isinstance(arg, torch.Tensor)
    else:
        return False

def is_torch_model(arg: Any) -> bool:
    """Checks if the object is a PyTorch Module or path to PyTorch model."""
    if isinstance(arg, str):
        return is_torch_model_path(arg)
    elif torch_available:
        import torch
        return isinstance(arg, torch.nn.Module)
    else:
        return False
