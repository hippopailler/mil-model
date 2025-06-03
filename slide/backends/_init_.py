"""Abstraction to support both Libvips and cuCIM backends."""

#import slideflow as sf
from typing import List
from _backend import *

def wsi_reader(path: str, *args, **kwargs):
    """Get a slide image reader from the current backend."""
    if slide_backend() == 'libvips':
        from .vips import get_libvips_reader
        return get_libvips_reader(path, *args, **kwargs)

    elif slide_backend() == 'cucim':
        from .cucim import get_cucim_reader
        return get_cucim_reader(path, *args, **kwargs)
