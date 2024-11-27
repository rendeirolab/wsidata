from __future__ import annotations

from typing import NamedTuple, Sequence, Dict, List, Tuple

import cv2
import numpy as np
from shapely import MultiPolygon, Polygon, box, clip_by_rect
from shapely.affinity import translate, scale

from .._normalizer import ColorNormalizer


class TissueContour(NamedTuple):
    tissue_id: int
    contour: np.ndarray
    holes: List[np.ndarray | Polygon] | None = None


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
    anno_mask: np.ndarray | None = None
    anno_shapes: List[Tuple[Polygon, str, int]] | None = None

    def __repr__(self):
        image_dtype = self.image.dtype
        if self.anno_mask is not None:
            mask_repr = (
                f"(shape: {self.anno_mask.shape}, dtype: {self.anno_mask.dtype})"
            )
        else:
            mask_repr = None

        return (
            f"TileImage(id={self.id}, "
            f"x={self.x}, y={self.y}, "
            f"tissue_id={self.tissue_id}, "
            f"image=(shape: {self.image.shape}, dtype: {image_dtype}), "
            f"anno_mask={mask_repr}, "
            f"anno_shapes=({len(self.anno_shapes)} shapes))"
        )


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
            - holes : The holes in the tissue contour.

        """

        contours = self._obj.shapes[key]

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
            cn_func = lambda x: cn(x).numpy().astype(np.uint8)  # noqa
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
        annotation_key: str = None,
        annotation_name: str | Sequence[str] = None,
        annotation_label: str | Dict[str, int] = None,
        mask_dtype: np.dtype = None,
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
        annotation_key : str, default: None
            The key to the annotation table in :bdg-danger:`shapes` slot.
        annotation_name : str or array of str, default: None
            The name of the annotation column in the annotation table or a list of names.
        annotation_label : str or dict, default: None
            The name of the label column in the annotation table or a dictionary of label mappings.
        mask_dtype : np.dtype, default: np.uint8
            The data type of the annotation mask, if you have annotation labels more than 255,
            consider using data type with higher bound.
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
            - anno_mask: The annotation mask.
            - anno_shapes: The annotation shapes.

        """
        tile_spec = self._obj.tile_spec(key)

        if color_norm is not None:
            cn = ColorNormalizer(method=color_norm)
            cn_func = lambda x: cn(x).numpy().astype(np.uint8)  # noqa
        else:
            cn_func = lambda x: x  # noqa

        create_anno_mask = False
        anno_tb = None
        mask_size = (
            (tile_spec.raw_height, tile_spec.raw_width)
            if raw
            else (tile_spec.height, tile_spec.width)
        )
        downsample = tile_spec.downsample
        if annotation_key is not None:
            create_anno_mask = True
            if annotation_name is None:
                raise ValueError(
                    "annotation_name must be provided to create annotation mask."
                )
            if annotation_label is None:
                raise ValueError(
                    "annotation_label must be provided to create annotation mask."
                )

            anno_tb = self._obj.shapes[annotation_key]
            if isinstance(annotation_name, str):
                annotation_name = anno_tb[annotation_name]

            if isinstance(annotation_label, str):
                annotation_label = anno_tb[annotation_label]
            else:
                annotation_label = [annotation_label[n] for n in annotation_name]
            annotation_name = np.asarray(annotation_name)
            annotation_label = np.asarray(annotation_label)

            if mask_dtype is None:
                if annotation_label.max() > 255:
                    mask_dtype = np.uint32
                else:
                    mask_dtype = np.uint8

        points = self._obj[key]
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

            if not raw:
                img = cv2.resize(img, (tile_spec.width, tile_spec.height))

            if format == "cyx":
                img = img.transpose(2, 0, 1)

            anno_shapes = None
            anno_mask = None
            if create_anno_mask:
                anno_shapes = []
                bbox = box(x, y, x + tile_spec.raw_width, y + tile_spec.raw_height)
                sel = anno_tb.geometry.intersects(bbox)  # return a boolean mask
                anno_mask = np.zeros(mask_size, dtype=mask_dtype)
                if sel.sum() > 0:
                    sel = sel.values
                    geos = anno_tb.geometry[sel]
                    names = annotation_name[sel]
                    labels = annotation_label[sel]

                    for geo, name, label in zip(geos, names, labels):
                        geo = translate(geo, xoff=-x, yoff=-y)
                        if not raw:
                            geo = scale(
                                geo, 1 / downsample, 1 / downsample, origin=(0, 0)
                            )
                        cnt = np.array(geo.exterior.coords, dtype=np.int32)
                        holes = [
                            np.array(h.coords, dtype=np.int32) for h in geo.interiors
                        ]
                        cv2.fillPoly(anno_mask, [cnt], int(label))  # noqa
                        cv2.fillPoly(anno_mask, holes, 0)  # noqa
                        # Clip the annotation by the tile
                        # May not be valid after clipping
                        output_geo = clip_by_rect(geo, 0, 0, *mask_size)
                        if not output_geo.is_valid:
                            continue
                        elif isinstance(output_geo, MultiPolygon):
                            output_geo = [p for p in output_geo.geoms]
                        else:
                            output_geo = [output_geo]
                        for p in output_geo:
                            anno_shapes.append((p, name, label))

            yield TileImage(
                id=ix,
                x=x,
                y=y,
                tissue_id=tix,
                image=img,
                anno_mask=anno_mask,
                anno_shapes=anno_shapes,
            )
