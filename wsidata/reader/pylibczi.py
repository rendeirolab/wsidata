import json
import warnings
from pathlib import Path
from typing import Union

import cv2

from .._utils import find_stack_level
from ._reader_registry import register
from .base import ReaderBase, SlideProperties


@register(name="pylibczi")
class PylibCZIReader(ReaderBase):
    """
    Use pylibCZIrw to interface with Zeiss CZI whole-slide images.

    Depends on `pylibCZIrw <https://github.com/ZEISS/pylibczirw>`_, the
    official Python binding to `libCZI` maintained by Zeiss. Unlike
    BioFormats, pylibCZIrw decodes JPEG-XR natively via `libCZI` on every
    platform it ships wheels for, including ``arm64`` macOS - so this
    reader is the recommended backend for ``.czi`` files on Apple Silicon,
    where BioFormats cannot load the ``ome:jxrlib`` native library.

    Parameters
    ----------
    file : str or Path
        Path to the CZI file on disk.

    Notes
    -----
    pylibCZIrw does not expose pre-baked pyramid levels. This reader
    therefore presents a *synthetic* pyramid built by requesting
    downsampled reads via pylibCZIrw's ``zoom`` parameter, which is the
    sanctioned way to obtain lower-resolution views.

    The CZI format stores coordinates in an absolute reference frame
    whose origin can be far from ``(0, 0)``. This reader translates
    between the wsidata zero-origin convention and the CZI absolute
    coordinate space so that ``read_region(0, 0, ...)`` returns the
    top-left of the scene, matching every other wsidata reader.

    pylibCZIrw zero-pads out-of-bounds reads (at any origin, including
    negative), so this reader does not add an extra bounds guard.

    The initial implementation targets ``Bgr24`` (24-bit BGR) CZI files,
    which is the pixel format used by brightfield H&E scans on Zeiss
    microscopes. Other pixel types raise ``NotImplementedError``.
    """

    name = "pylibczi"
    pkg_namespaces = "pylibCZIrw"
    pkgs = ["pylibCZIrw"]

    # Number of synthetic pyramid levels to expose (1x, 2x, 4x, 8x, 16x, 32x).
    _N_LEVELS = 6

    def __init__(self, file: Union[Path, str], **kwargs):
        self.file = str(file)
        self._origin_x = 0
        self._origin_y = 0
        self._ctx = None
        self.create_reader()
        self._process_pylibczi_properties()

    def create_reader(self):
        from pylibCZIrw import czi as pyczi

        # pylibCZIrw exposes its reader only through a context manager, so
        # we keep the manager on the instance and drive __enter__ /
        # __exit__ manually. This is the only reader in wsidata that does
        # this - every other backend has a plain constructor.
        self.detach_reader()
        self._ctx = pyczi.open_czi(self.file)
        self.set_reader(self._ctx.__enter__())

    def detach_reader(self):
        if self._reader is not None:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._ctx = None
            self.set_reader(None)

    def _process_pylibczi_properties(self):
        doc = self.reader

        # Only Bgr24 is currently supported. Fail early and loudly for
        # other pixel types so users know where to open a follow-up issue.
        pixel_type = doc.pixel_types.get(0)
        if pixel_type != "Bgr24":
            raise NotImplementedError(
                f"PylibCZIReader currently only supports Bgr24 pixel type, "
                f"got {pixel_type!r}. Please open an issue on "
                f"https://github.com/rendeirolab/wsidata with a sample "
                f"file if you need support for other pixel types."
            )

        # Use scene 0 if the file has scenes, otherwise fall back to the
        # total bounding rectangle (synthetic and single-scene files).
        rect = doc.scenes_bounding_rectangle.get(0, doc.total_bounding_rectangle)
        width, height = int(rect.w), int(rect.h)

        # Multi-scene CZIs (e.g. multiple tissue sections on a single
        # slide) are read as scene 0 only. Warn loudly so callers notice
        # truncated data and point them at the issue tracker for a
        # principled multi-scene API, which should be designed once
        # across all wsidata readers rather than ad-hoc here.
        n_scenes = len(doc.scenes_bounding_rectangle)
        if n_scenes > 1:
            warnings.warn(
                f"CZI file contains {n_scenes} scenes; only scene 0 is "
                f"read. Multi-scene selection is not yet exposed - please "
                f"open an issue on https://github.com/rendeirolab/wsidata "
                f"with a sample file if you need it.",
                UserWarning,
                stacklevel=find_stack_level(),
            )

        # Record the CZI absolute origin so that get_region can translate
        # back from the zero-origin frame exposed to callers.
        self._origin_x = int(rect.x)
        self._origin_y = int(rect.y)

        # Extract per-axis micron-per-pixel from the scaling metadata.
        # The canonical path is ImageDocument.Metadata.Scaling.Items.Distance,
        # which holds a list of {"@Id": "X"|"Y", "Value": "<metres>"} dicts.
        mpp_x, mpp_y = None, None
        try:
            distances = doc.metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"][
                "Distance"
            ]
            # Some files store a single dict rather than a list.
            if isinstance(distances, dict):
                distances = [distances]
            for d in distances:
                if d.get("@Id") == "X":
                    mpp_x = float(d["Value"]) * 1e6  # metres -> microns
                elif d.get("@Id") == "Y":
                    mpp_y = float(d["Value"]) * 1e6
        except (KeyError, TypeError, ValueError):
            pass

        mpp = None
        if mpp_x is not None and mpp_y is not None:
            mpp = (mpp_x + mpp_y) / 2.0
        elif mpp_x is not None:
            mpp = mpp_x
        elif mpp_y is not None:
            mpp = mpp_y

        # Build a synthetic pyramid: level i has downsample 2**i.
        downsamples = [2**i for i in range(self._N_LEVELS)]
        level_shape = [
            [max(1, height // ds), max(1, width // ds)] for ds in downsamples
        ]

        raw = {
            "pixel_type": pixel_type,
            "origin_x": self._origin_x,
            "origin_y": self._origin_y,
            "width": width,
            "height": height,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "n_scenes": len(doc.scenes_bounding_rectangle),
        }

        self.set_properties(
            SlideProperties(
                shape=[height, width],
                n_level=self._N_LEVELS,
                level_shape=level_shape,
                level_downsample=[float(ds) for ds in downsamples],
                mpp=mpp,
                magnification=None,
                bounds=[0, 0, width, height],
                raw=json.dumps(raw),
            )
        )

    def get_region(self, x, y, width, height, level=0, **kwargs):
        """
        Read a region at the given pyramid level.

        Follows the wsidata convention: ``(x, y)`` is in level-0,
        zero-origin coordinates; ``(width, height)`` is in pixel units at
        the requested level.
        """
        level = self.translate_level(level)
        downsample = self.properties.level_downsample[level]

        # Convert the output size back to level-0 pixel units so that the
        # ``zoom`` argument to pylibCZIrw performs the downsampling.
        src_width = int(width * downsample)
        src_height = int(height * downsample)

        # Translate the zero-origin request back into the CZI absolute
        # coordinate frame.
        czi_x = int(x) + self._origin_x
        czi_y = int(y) + self._origin_y

        zoom = 1.0 / downsample
        img = self.reader.read(
            roi=(czi_x, czi_y, src_width, src_height),
            plane={"C": 0},
            zoom=zoom,
        )

        # pylibCZIrw returns BGR for Bgr24 pixel types, so a direct
        # channel swap is needed here. We cannot reuse the base-class
        # ``convert_image`` helper because that assumes an RGBA input.
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def get_thumbnail(self, size, **kwargs):
        """Get a thumbnail whose longest edge is ``size`` pixels."""
        height, width = self.properties.shape
        zoom = min(size / max(height, width), 1.0)

        img = self.reader.read(
            roi=(self._origin_x, self._origin_y, width, height),
            plane={"C": 0},
            zoom=zoom,
        )

        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img
