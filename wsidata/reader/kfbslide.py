
from pathlib import Path
from typing import Union

from ._reader_registry import register
from .openslide import OpenSlideReader


@register(name="kfbslide")
class KFBSlideReader(OpenSlideReader):
    """Read KFB whole-slide images using kfbslide.

    Reuses the OpenSlide-compatible interface for region
    extraction, thumbnails, and associated images.

    Parameters
    ----------
    file : str or Path
        Path to the KFB whole-slide image.
    """

    name = "kfbslide"
    pkg_namespaces = "kfbslide"
    pkgs = ["kfbslide"]
    extensions = (".kfb",)

    def __init__(
        self,
        file: Union[Path, str],
        **kwargs,
    ):
        self.file = str(file)
        self.create_reader()

        try:
            reader = self.reader
            metadata = dict(reader.properties)

            # Normalize pyramid metadata for wsidata.
            metadata["openslide.level-count"] = str(
                reader.level_count
            )

            for level, (width, height) in enumerate(
                reader.level_dimensions
            ):
                prefix = f"openslide.level[{level}]"

                metadata[f"{prefix}.width"] = str(width)
                metadata[f"{prefix}.height"] = str(height)
                metadata[f"{prefix}.downsample"] = str(
                    reader.level_downsamples[level]
                )

            self.set_properties(metadata)

        except Exception:
            self.detach_reader()
            raise

    def create_reader(self):
        from kfbslide import OpenSlide

        self.set_reader(OpenSlide(self.file))
