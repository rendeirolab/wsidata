# The implementation is highly inspired by
# https://github.com/AllenCellModeling/aicsimageio/blob/main/aicsimageio/readers/bioformats_reader.py
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

from .base import ReaderBase, SlideProperties, convert_image


class BioFormatsReader(ReaderBase):
    """
    Use Bio-Formats to interface with image files.

    Depends on `scyjava <https://github.com/scijava/scyjava>`_

    Parameters
    ----------
    file : str or Path
        Path to image file on disk

    """

    name = "bioformats"

    def __init__(
        self,
        file: Union[Path, str],
        memorize: bool = False,
        **kwargs,
    ):
        self.file = str(file)
        self._memorize = memorize
        # Bio-Formats specific properties
        self._is_interleaved = None  # This is a flag to indicate the storage layout
        self._dtype = None  # This is the pixel type
        self._series = None  # This is the series index point to the used pyramid

        # Create a reader object
        self.create_reader()
        # This should only run once
        self._process_bioformats_properties(self.reader, self.reader.getMetadataStore())

    # TODO: Test if this is the same with openslide
    def get_region(
        self,
        x,
        y,
        width,
        height,
        level: int = 0,
        **kwargs,
    ):
        level = self.translate_level(level)
        open_width, open_height = width, height
        downsample = self.properties.level_downsample[level]
        open_x, open_y = int(x / downsample), int(y / downsample)
        img_height, img_width = self.properties.level_shape[level]

        # If x, y is out of bounds, directly return a black image
        # x, y is always the positive integer
        if open_x >= img_width or open_y >= img_height:
            return np.zeros((height, width, 3), dtype=np.uint8)

        # Move the reader to the correct series
        self.reader.setResolution(level)

        idx = self.reader.getIndex(0, 0, 0)

        clip_width = open_x + width - img_width
        clip_height = open_y + height - img_height

        # Check if the region is out of bounds
        if clip_width > 0:
            open_width -= clip_width
        else:
            clip_width = 0
        if clip_height > 0:
            open_height -= clip_height
        else:
            clip_height = 0

        buffer = self.reader.openBytes(idx, open_x, open_y, open_width, open_height)
        img = np.frombuffer(buffer, dtype=self._dtype)
        if self._is_interleaved:
            img.shape = (open_height, open_width, 3)
        else:
            img.shape = (3, open_height, open_width)
            img = np.transpose(img, (1, 2, 0))

        # Fill the clipped region with black
        if clip_width > 0 or clip_height > 0:
            img = np.pad(
                img,
                ((0, clip_height), (0, clip_width), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # Coerce to uint8 regardless of the original dtype
        img = img.astype(np.uint8)
        return convert_image(img)

    # Not working properly
    def get_thumbnail(self, size, **kwargs):
        sx, sy = self.properties.shape
        if size > sx or size > sy:
            raise ValueError("Requested thumbnail size is larger than the image")
        # Get the level that is closest to the target size, but always smaller
        level_shape = np.asarray(self.properties.level_shape)
        areas = level_shape[:, 0] * level_shape[:, 1]
        ix = np.where(areas < size**2)[0]
        if len(ix) == 0:
            level = 0
        else:
            level = ix[np.argmax(areas[ix])]
        return self.get_level(level)

    def detach_reader(self):
        if self._reader is not None:
            try:
                self._reader.close()
                self._reader = None
            except (AttributeError, ImportError, RuntimeError):
                pass

    def create_reader(self):
        import jpype
        import scyjava

        scyjava.config.endpoints.append("ome:formats-gpl")
        scyjava.start_jvm()

        # Suppress the output from Java
        System = jpype.JPackage("java").lang.System
        System.err.close()

        # Import the loci namespace, which is the Bio-Formats library
        loci = jpype.JPackage("loci")
        loci.common.DebugTools.setRootLevel("ERROR")

        # Create a reader object
        reader = loci.formats.ImageReader()

        # Setup metadata store to hold the metadata
        factory = loci.common.services.ServiceFactory()
        ome_service = factory.getInstance(loci.formats.services.OMEXMLService)
        meta = ome_service.createOMEXMLMetadata()
        # Attach the metadata store to the reader
        # To allow access the metadata
        reader.setMetadataStore(meta)
        reader.setOriginalMetadataPopulated(True)
        # Now each series will be a pyramid level
        reader.setFlattenedResolutions(False)
        # This will cache the reader on java side
        if self._memorize:
            reader = loci.formats.Memoizer(reader, 1)
        reader.setId(self.file)

        # Always point the reader to the Pyramid with the highest resolution
        if self._series is not None:
            reader.setSeries(self._series)

        self._reader = reader

    def _get_loci_namespace(self):
        import jpype

        loci = jpype.JPackage("loci")
        return loci

    def _process_bioformats_properties(self, reader, meta):
        n_series = reader.getSeriesCount()

        multi_pyramids = []
        for series in range(n_series):
            reader.setSeries(series)
            n_res = reader.getResolutionCount()

            mpp = meta.getPixelsPhysicalSizeX(series)
            if mpp is not None:
                mpp = float(mpp.value())
            if (n_res > 1) | (mpp is not None):
                try:
                    mag = meta.getObjectiveNominalMagnification(0, series)
                    mag = float(mag)
                except:  # noqa
                    mag = None
            else:
                mag = None

            level_shape = []
            for res in range(n_res):
                reader.setResolution(res)
                size_x = reader.getSizeX()
                size_y = reader.getSizeY()
                # n_rgb = reader.getRGBChannelCount()
                level_shape.append([int(size_y), int(size_x)])
            multi_pyramids.append(
                Pyramids(
                    series=series,
                    width=level_shape[0][1],
                    height=level_shape[0][0],
                    level_shape=level_shape,
                    mpp=mpp,
                    mag=mag,
                )
            )
        # Identify the pyramid with the highest resolution
        used_pyramids = sorted(multi_pyramids, key=lambda x: x.height * x.width)[-1]
        # Calculate the downsample for each level
        level_downsample = []
        for i, shape in enumerate(used_pyramids.level_shape):
            if i == 0:
                downsample = 1
            else:
                level_0_shape = used_pyramids.level_shape[0]
                downsample = level_0_shape[0] / shape[0]
            level_downsample.append(downsample)
        # Point the reader back to the pyramid that we will use
        self._series = int(used_pyramids.series)
        reader.setSeries(self._series)
        self._is_interleaved = bool(reader.isInterleaved())
        # parse the pixel type
        self._dtype = reader.getPixelType()
        loci = self._get_loci_namespace()
        FT = loci.formats.FormatTools
        fmt2type = {
            FT.INT8: "i1",
            FT.UINT8: "u1",
            FT.INT16: "i2",
            FT.UINT16: "u2",
            FT.INT32: "i4",
            FT.UINT32: "u4",
            FT.FLOAT: "f4",
            FT.DOUBLE: "f8",
        }
        little_endian = reader.isLittleEndian()
        pixel_dtype = reader.getPixelType()
        self._dtype = np.dtype(("<" if little_endian else ">") + fmt2type[pixel_dtype])

        self.properties = SlideProperties(
            shape=[used_pyramids.height, used_pyramids.width],
            level_shape=used_pyramids.level_shape,
            level_downsample=level_downsample,
            n_level=len(used_pyramids.level_shape),
            mpp=used_pyramids.mpp,
            magnification=used_pyramids.mag,
            bounds=[0, 0, used_pyramids.width, used_pyramids.height],
            # TODO: Attach raw metadata
        )


@dataclass
class Pyramids:
    series: int
    width: int
    height: int
    level_shape: List[List[int]]
    mpp: float
    mag: float
