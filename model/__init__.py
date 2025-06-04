"""Multiple-instance learning (MIL) models"""

from util import is_torch_model_path, torch_available, tf_available, is_tensorflow_model_path
from typing import Any, Dict, List



def is_tensorflow_tensor(arg: Any) -> bool:
    """Checks if the given object is a Tensorflow Tensor."""
    if tf_available:
        import tensorflow as tf
        return isinstance(arg, tf.Tensor)
    else:
        return False
    
def is_tensorflow_model(arg: Any) -> bool:
    """Checks if the object is a Tensorflow Model or path to Tensorflow model."""
    if isinstance(arg, str):
        return is_tensorflow_model_path(arg)
    elif tf_available:
        import tensorflow as tf
        return isinstance(arg, tf.keras.models.Model)
    else:
        return False

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
