from typing import NamedTuple

import cv2
import numpy as np
from shapely import Polygon

from .._normalizer import ColorNormalizer


class TissueContour(NamedTuple):
    tissue_id: int
    contour: np.ndarray
    holes: list[np.ndarray]


class TissueImage(NamedTuple):
    tissue_id: int
    x: int
    y: int
    image: np.ndarray
    mask: np.ndarray


class TileImage(NamedTuple):
    id: int
    x: int
    y: int
    tissue_id: int
    image: np.ndarray


class IterAccessor(object):
    """An accessor to iterate over the WSI data.

    Usage: `wsidata.iter`

    """

    def __init__(self, obj):
        self._obj = obj

    def tissue_contours(
        self,
        key,
        as_array: bool = False,
        dtype: np.dtype = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> TissueContour:
        """A generator to extract tissue contours from the WSI.

        Parameters
        ----------
        key : str
            The tissue key.
        as_array : bool, default: False
            Return the contour as an array.
            If False, the contour is returned as a shapely geometry.
        dtype : np.dtype, default: None
            The data type of the array if as_array is True.
        shuffle : bool, default: False
            If True, return tissue contour in random order.
        seed : int, default: 0

        Returns
        -------
        TissueContour
            A named tuple with fields:

            - tissue_id : The tissue id.
            - contour : The tissue contour as a shapely geometry or an array.

        """

        contours = self._obj.sdata.shapes[key]

        if shuffle:
            contours = contours.sample(frac=1, random_state=seed)

        for ix, cnt in contours.iterrows():
            tissue_id = cnt["tissue_id"]
            if as_array:
                yield TissueContour(
                    tissue_id=tissue_id,
                    contour=np.array(cnt.geometry.exterior.coords, dtype=dtype),
                    holes=[
                        np.asarray(h.coords, dtype=dtype)
                        for h in cnt.geometry.interiors
                    ],
                )
            else:
                yield TissueContour(
                    tissue_id=tissue_id,
                    contour=cnt.geometry,
                    holes=[Polygon(i) for i in cnt.geometry.interiors],
                )

    def tissue_images(
        self,
        key,
        level=0,
        mask_bg=False,
        tissue_mask=False,
        color_norm: str = None,
        format: str = "xyc",
    ) -> TissueImage:
        """Extract tissue images from the WSI.

        Parameters
        ----------
        key : str
            The tissue key.
        level : int, default: 0
            The level to extract the tissue images.
        mask_bg : bool | int, default: False
            Mask the background with the given value.

            If False, the background is not masked.

            If True, the background is masked with 0.

            If an integer, the background is masked with the given value.
        color_norm : str, {"macenko", "reinhard"}, default: None
            Color normalization method.
        format : str, {"xyc", "cyx"}, default: "xyc"
            The channel format of the image.

        Returns
        -------
        TissueImage
            A named tuple with fields:

            - tissue_id : The tissue id.
            - x : The x-coordinate of the image.
            - y : The y-coordinate of the image.
            - image : The tissue image.
            - mask : The tissue mask.

        """
        import cv2

        level_downsample = self._obj.properties.level_downsample[level]
        if color_norm is not None:
            cn = ColorNormalizer(method=color_norm)
            cn_func = lambda x: cn(x).numpy()  # noqa
        else:
            cn_func = lambda x: x  # noqa

        if isinstance(mask_bg, bool):
            do_mask = mask_bg
            if do_mask:
                mask_bg = 0
        else:
            do_mask = True
            mask_bg = mask_bg
        for tissue_contour in self.tissue_contours(key):
            ix = tissue_contour.tissue_id
            contour = tissue_contour.contour
            holes = [np.asarray(h.coords) for h in contour.interiors]
            minx, miny, maxx, maxy = contour.bounds
            x = int(minx)
            y = int(miny)
            w = int(maxx - minx) / level_downsample
            h = int(maxy - miny) / level_downsample
            img = self._obj.reader.get_region(x, y, w, h, level=level)
            img = cn_func(img)

            mask = None
            if do_mask or tissue_mask:
                mask = np.zeros_like(img[:, :, 0])
                # Offset and scale the contour
                offset_x, offset_y = x / level_downsample, y / level_downsample
                coords = np.array(contour.exterior.coords) - [offset_x, offset_y]
                coords = (coords / level_downsample).astype(np.int32)
                # Fill the contour with 1
                cv2.fillPoly(mask, [coords], 1)

                # Fill the holes with 0
                for hole in holes:
                    hole -= [offset_x, offset_y]
                    hole = (hole / level_downsample).astype(np.int32)
                    cv2.fillPoly(mask, [hole], 0)

            if do_mask:
                # Fill everything that is not the contour
                # (which is background) with 0
                img[mask != 1] = mask_bg
            if not tissue_mask:
                mask = None
            else:
                mask = mask.astype(bool)

            if format == "cyx":
                img = img.transpose(2, 0, 1)

            yield TissueImage(tissue_id=ix, x=x, y=y, image=img, mask=mask)

    def tile_images(
        self,
        key,
        raw=False,
        color_norm: str = None,
        format: str = "xyc",
        shuffle: bool = False,
        sample_n: int = None,
        seed: int = 0,
    ) -> TileImage:
        """Extract tile images from the WSI.

        Parameters
        ----------
        key : str
            The tile key.
        raw : bool, default: True
            Return the raw image without resizing.

            If False, the image is resized to the requested tile size.
        color_norm : str, {"macenko", "reinhard"}, default: None
            Color normalization method.
        format : str, {"xyc", "cyx"}, default: "xyc"
            The channel format of the image.
        shuffle : bool, default: False
            If True, return tile images in random order.
        sample_n : int, default: None
            The number of samples to return.
        seed : int, default: 0
            The random seed.

        Returns
        -------
        TileImage
            A named tuple with fields:

            - id: The tile id.
            - x: The x-coordinate of the tile.
            - y: The y-coordinate of the tile.
            - tissue_id: The tissue id.
            - image: The tile image.

        """
        tile_spec = self._obj.tile_spec(key)
        # Check if the image needs to be transformed
        need_transform = (
            tile_spec.raw_width != tile_spec.width
            or tile_spec.raw_height != tile_spec.height
        )

        if color_norm is not None:
            cn = ColorNormalizer(method=color_norm)
            cn_func = lambda x: cn(x).numpy()  # noqa
        else:
            cn_func = lambda x: x  # noqa

        points = self._obj.sdata[key]
        if sample_n is not None:
            points = points.sample(n=sample_n, random_state=seed)
        elif shuffle:
            points = points.sample(frac=1, random_state=seed)

        for _, row in points.iterrows():
            x = row["x"]
            y = row["y"]
            ix = row["id"]
            tix = row["tissue_id"]
            img = self._obj.reader.get_region(
                x, y, tile_spec.raw_width, tile_spec.raw_height, level=tile_spec.level
            )
            img = cn_func(img)

            if not raw or need_transform:
                img = cv2.resize(img, (tile_spec.width, tile_spec.height))

            if format == "cyx":
                img = img.transpose(2, 0, 1)

            yield TileImage(id=ix, x=x, y=y, tissue_id=tix, image=img)
