'''Functions for slide extraction reports (PDF).'''

from __future__ import absolute_import, division, print_function

import io
import os
import tempfile
import pandas as pd
import numpy as np
import cv2

from modules import errors
from wsi import WSI
from fpdf import FPDF, XPos, YPos
from PIL import Image, UnidentifiedImageError
from datetime import datetime
from os.path import join, exists
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

#import slideflow as sf
#from slideflow.util import log, path_to_name  # noqa F401
from util import log, path_to_name


if TYPE_CHECKING:
    import pandas as pd

# -----------------------------------------------------------------------------

def render_thumbnail(report: "SlideReport") -> Optional["Image.Image"]:
    return report.thumb


def render_image_row(report: "SlideReport") -> Optional[bytes]:
    return report.image_row()

# -----------------------------------------------------------------------------

class SlideReport:
    '''Report to summarize tile extraction from a slide, including
    example images of extracted tiles.
    '''

    def __init__(
        self,
        images: List[bytes],
        path: str,
        tile_px: int,
        tile_um: Union[int, str],
        *,
        thumb: Optional[Image.Image] = None,
        thumb_coords: Optional[np.ndarray] = None,
        data: Optional[Dict[str, Any]] = None,
        compress: bool = True,
        ignore_thumb_errors: bool = False
    ) -> None:
        """Creates a slide report summarizing tile extraction, with some example
        extracted images.

        Args:
            images (list(str)): List of JPEG image strings (example tiles).
            path (str): Path to slide.
            data (dict, optional): Dictionary of slide extraction report
                metadata. Expected keys may include 'blur_burden', 'num_tiles',
                'locations', and 'qc_mask'. Defaults to None.
            compress (bool, optional): Compresses images to reduce image sizes.
                Defaults to True.
            thumb (PIL.Image): Thumbnail of slide. Defaults to None.
            thumb_coords (np.ndarray): Array of (x, y) tile extraction
                coordinates, for display on the thumbnail. Defaults to None.
            ignore_thumb_errors (bool): Ignore errors raised when attempting
                to create a slide thumbnail.


        """
        self.data = data
        self.path = path
        self.tile_px = tile_px
        self.tile_um = tile_um
        if data is not None:
            self.has_rois = 'num_rois' in data and data['num_rois'] > 0
        else:
            self.has_rois = False
        self.timestamp = str(datetime.now())

        # Thumbnail
        self.ignore_thumb_errors = ignore_thumb_errors
        self.thumb_coords = thumb_coords
        if thumb is not None:
            self._thumb = Image.fromarray(np.array(thumb)[:, :, 0:3])
        else:
            self._thumb = None

        if not compress:
            self.images = images  # type: List[bytes]
        else:
            self.images = [self._compress(img) for img in images]

    @property
    def thumb(self):
        if self._thumb is None:
            try:
                self.calc_thumb()
            except Exception:
                if self.ignore_thumb_errors:
                    return None
                else:
                    raise
        return self._thumb

    @property
    def blur_burden(self) -> Optional[float]:
        """Metric defined as the proportion of non-background slide
        with high blur. Only calculated if both Otsu and Blur QC is used.

        Returns:
            float
        """
        if self.data is None:
            return None
        if 'blur_burden' in self.data:
            return self.data['blur_burden']
        else:
            return None

    @property
    def num_tiles(self) -> Optional[int]:
        """Number of tiles extracted.

        Returns:
            int
        """
        if self.data is None:
            return None
        if 'num_tiles' in self.data:
            return self.data['num_tiles']
        else:
            return None

    @property
    def locations(self) -> Optional["pd.DataFrame"]:
        """DataFrame with locations of extracted tiles, with the following
        columns:

        ``loc_x``: Extracted tile x coordinates (as saved in TFRecords).
        Calculated as the full coordinate value / 10.

        ``loc_y``: Extracted tile y coordinates (as saved in TFRecords).
        Calculated as the full coordinate value / 10.

        ``grid_x``: First dimension index of the tile extraction grid.

        ``grid_y``: Second dimension index of the tile extraction grid.

        ``gs_fraction``: Grayspace fraction. Only included if grayspace
        filtering is used.

        ``ws_fraction``: Whitespace fraction. Only included if whitespace
        filtering is used.

        Returns:
            pandas.DataFrame

        """
        if self.data is None:
            return None
        if 'locations' in self.data:
            return self.data['locations']
        else:
            return None

    @property
    def qc_mask(self) -> Optional[np.ndarray]:
        """Numpy array with the QC mask, of shape WSI.grid and type bool
        (True = include tile, False = discard tile)

        Returns:
            np.ndarray
        """
        if self.data is None:
            return None
        if 'qc_mask' in self.data:
            return self.data['qc_mask']
        else:
            return None

    def calc_thumb(self) -> None:
        try:
            wsi = WSI(
                self.path,
                tile_px=self.tile_px,
                tile_um=self.tile_um,
                verbose=False,
            )
        except errors.SlideMissingMPPError:
            wsi = WSI(
                self.path,
                tile_px=self.tile_px,
                tile_um=self.tile_um,
                verbose=False,
                mpp=1   # Force MPP to 1 to add support for slides missing MPP.
                        # The MPP does not need to be accurate for thumbnail generation.
            )
        self._thumb = wsi.thumb(
            coords=self.thumb_coords,
            rois=self.has_rois,
            low_res=True,
            width=512,
            rect_linewidth=1,
        )
        self._thumb = Image.fromarray(np.array(self._thumb)[:, :, 0:3])

    def _compress(self, img: bytes) -> bytes:
        with io.BytesIO() as output:
            pil_img = Image.open(io.BytesIO(img)).convert('RGB')
            if pil_img.height > 256:
                pil_img = Image.fromarray(
                    cv2.resize(np.array(pil_img), [256, 256])
                )
            pil_img.save(output, format="JPEG", quality=75)
            return output.getvalue()

    def image_row(self) -> Optional[bytes]:
        '''Merges images into a single row of images'''
        if not self.images:
            return None
        pil_images = [Image.open(io.BytesIO(i)) for i in self.images]
        widths, heights = zip(*(pi.size for pi in pil_images))
        total_width = sum(widths)
        max_height = max(heights)
        row_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for image in pil_images:
            row_image.paste(image, (x_offset, 0))
            x_offset += image.size[0]
        with io.BytesIO() as output:
            row_image.save(output, format="JPEG", quality=75)
            return output.getvalue()


