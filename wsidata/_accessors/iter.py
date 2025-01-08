from __future__ import annotations

from functools import cached_property
from typing import NamedTuple, Sequence, Dict, List, Tuple

import cv2
import numpy as np
from shapely import MultiPolygon, Polygon, box, clip_by_rect
from shapely.affinity import translate, scale


def _get_cn_func(color_norm):
    from .._normalizer import ColorNormalizer

    if color_norm is not None:
        cn = ColorNormalizer(method=color_norm)
        cn_func = lambda x: cn(x).numpy().astype(np.uint8)  # noqa
    else:
        cn_func = lambda x: x  # noqa
    return cn_func


def _normalize_polygon(polygon, bbox):
    """
    Normalize a polygon to the 0-1 range based on a bounding box.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Input polygon.
    bbox : tuple
        Bounding box as (minx, miny, maxx, maxy).

    Returns
    -------
    shapely.geometry.Polygon: Normalized polygon.

    """
    minx, miny, maxx, maxy = bbox
    width = maxx - minx
    height = maxy - miny

    # Translate the polygon to origin
    translated = translate(polygon, xoff=-minx, yoff=-miny)
    # Scale to 0-1 range
    normalized = scale(translated, xfact=1 / width, yfact=1 / height, origin=(0, 0))
    return normalized


PALETTE = (
    "#e60049",
    "#0bb4ff",
    "#50e991",
    "#e6d800",
    "#9b19f5",
    "#ffa300",
    "#dc0ab4",
    "#b3d4ff",
    "#00bfa0",
)


class TissueContour:
    tissue_id: int
    shape: Polygon
    as_array: bool
    dtype: np.dtype

    def __init__(self, tissue_id, shape, as_array=False, dtype=None):
        self.tissue_id = tissue_id
        self.shape = shape
        self._as_array = as_array
        self._dtype = dtype

    @property
    def as_array(self):
        return self._as_array

    @property
    def dtype(self):
        return self._dtype

    @cached_property
    def contour(self) -> np.ndarray | Polygon:
        if self.as_array:
            return np.array(self.shape.exterior.coords, dtype=self.dtype)
        else:
            return Polygon(self.shape.exterior.coords)

    @cached_property
    def holes(self) -> List[np.ndarray | Polygon]:
        if self.as_array:
            return [
                np.asarray(h.coords, dtype=self.dtype) for h in self.shape.interiors
            ]
        else:
            return [Polygon(h) for h in self.shape.interiors]

    def __repr__(self):
        n_holes = len(self.holes)
        hole_text = "holes" if n_holes > 1 else "hole"
        return (
            f"TissueContour(tissue_id={self.tissue_id}, {n_holes} {hole_text}) "
            f"with attributes: \n"
            f"tissue_id, shape, contour, holes"
        )

    def plot(
        self,
        ax=None,
        outline_color="#117554",
        hole_color="#4379F2",
        linewidth=2,
        outline_kwargs=None,
        hole_kwargs=None,
    ):
        """Plot the tissue contour.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, default: None
            The axes to plot the contour.
        outline_color : str, default: "#117554"
            The color of the outline.
        hole_color : str, default: "#4379F2"
            The color of the holes.
        linewidth : int, default: 2
            The width of the line.
        outline_kwargs : dict, default: None
            Additional keyword arguments for the outline.
        hole_kwargs : dict, default: None
            Additional keyword arguments for the holes.

        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        outline_kwargs = outline_kwargs or {}
        outline_kwargs.setdefault("color", outline_color)
        outline_kwargs.setdefault("linewidth", linewidth)
        hole_kwargs = hole_kwargs or {}
        hole_kwargs.setdefault("color", hole_color)
        hole_kwargs.setdefault("linewidth", linewidth)
        contour = self.contour
        holes = self.holes
        if isinstance(contour, Polygon):
            ax.plot(*contour.exterior.xy, **outline_kwargs)
        else:
            ax.plot(contour[:, 0], contour[:, 1], **outline_kwargs)
        if holes is not None:
            for hole in holes:
                if isinstance(hole, Polygon):
                    ax.plot(*hole.exterior.xy, **hole_kwargs)
                else:
                    ax.plot(hole[:, 0], hole[:, 1], **hole_kwargs)
        ax.set_aspect(1)
        ax.invert_yaxis()
        return ax


class TissueImage(TissueContour):
    image: np.ndarray
    format: str
    mask_bg: int | None

    def __init__(
        self,
        tissue_id,
        shape,
        image,
        format="xyc",
        mask_bg=None,
        as_array=False,
        dtype=None,
        downsample=1.0,
    ):
        super().__init__(tissue_id, shape, as_array, dtype)
        self._image = image
        self.format = format
        self.mask_bg = mask_bg
        self.downsample = downsample

    @cached_property
    def x(self):
        return int(self.shape.bounds[0])

    @cached_property
    def y(self):
        return int(self.shape.bounds[1])

    @cached_property
    def width(self):
        return int(self.shape.bounds[2] - self.shape.bounds[0])

    @cached_property
    def height(self):
        return int(self.shape.bounds[3] - self.shape.bounds[1])

    @property
    def image(self):
        if self.format == "cyx":
            return self._image.transpose(1, 2, 0)
        return self._image

    @cached_property
    def mask(self):
        mask = None
        if self.mask_bg is not None:
            mask = np.zeros_like(self.image[:, :, 0])
            # Offset and scale the contour
            offset_x, offset_y = self.shape.bounds[:2]
            coords = np.array(self.shape.exterior.coords) - [offset_x, offset_y]
            coords = coords.astype(np.int32)
            # Fill the contour with 1
            cv2.fillPoly(mask, [coords], 1)

            # Fill the holes with 0
            for hole in self.holes:
                if isinstance(hole, Polygon):
                    hole = np.array(hole.exterior.coords)
                hole -= [offset_x, offset_y]
                hole = hole.astype(np.int32)
                cv2.fillPoly(mask, [hole], 0)
            mask = mask.astype(bool)
        return mask

    @cached_property
    def masked_image(self) -> np.ndarray:
        """A masked image with the background masked will be returned
        if the mask_bg is not None."""
        if self.mask_bg is None:
            return self.image
        mask = self.mask
        image = self.image.copy()
        if mask is not None:
            # Fill everything that is not the contour
            # (which is background) with 0
            image[mask != 1] = self.mask_bg

        if self.format == "cyx":
            image = image.transpose(2, 0, 1)
        return image

    def __repr__(self):
        n_holes = len(self.holes)
        hole_text = "holes" if n_holes > 1 else "hole"
        return (
            f"TissueImage(tissue_id={self.tissue_id}, HW={self._image.shape[:2]}, {n_holes} {hole_text}) "
            f"with attributes: \n"
            f"tissue_id, image, mask, masked_image, shape, contour, holes, x, y, width, height"
        )

    def plot(self, ax=None, masked: bool = False, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        img = self.masked_image if masked else self.image
        if self.format == "cyx":
            img = img.transpose(1, 2, 0)
        extent = (self.x, self.x + self.width, self.y + self.height, self.y)
        ax.imshow(img, extent=extent, **kwargs)
        return ax


class TileImage:
    id: int
    x: int
    y: int
    tissue_id: int
    image: np.ndarray
    annot_mask: np.ndarray | None = None
    annot_shapes: List[Tuple[Polygon, str, int]] | None = None
    annot_labels: Dict[str, int] | None = None

    def __init__(
        self,
        id,
        x,
        y,
        base_width,
        base_height,
        tissue_id,
        image,
        annot_mask=None,
        annot_shapes=None,
        annot_labels=None,
    ):
        self.id = id
        self.x = x
        self.y = y
        self.base_width = base_width
        self.base_height = base_height
        self.tissue_id = tissue_id
        self.image = image
        self.annot_mask = annot_mask
        self.annot_shapes = annot_shapes
        self.annot_labels = annot_labels

    @cached_property
    def norm_annot_shapes(self):
        if self.annot_shapes is not None:
            new_shapes = []
            for shape, name, label in self.annot_shapes:
                norm_shape = _normalize_polygon(
                    shape, (0, 0, self.base_width, self.base_height)
                )
                new_shapes.append((norm_shape, name, self.annot_labels[name]))
            return new_shapes

    @property
    def has_annot(self):
        return self.annot_shapes is not None

    def __repr__(self):
        if self.annot_shapes is not None:
            n_shapes = len(self.annot_shapes)
            shape_text = "annotation" if n_shapes == 1 else "annotations"
            shapes_repr = f"({n_shapes} {shape_text})"
        else:
            shapes_repr = None

        return (
            f"TileImage(id={self.id}, x={self.x}, y={self.y}, "
            f"tissue_id={self.tissue_id}) {shapes_repr} "
            f"with attributes: \n"
            f"image, annot_mask, annot_shapes"
        )

    def plot(
        self,
        ax=None,
        show_annots: bool = True,
        palette: Dict = None,
        legend: bool = True,
        alpha: float = 0.3,
        linewidth=1,
        **kwargs,
    ):
        import matplotlib.pyplot as plt

        from matplotlib.patches import PathPatch
        from matplotlib.path import Path

        if ax is None:
            ax = plt.gca()

        ax.imshow(self.image, **kwargs)

        # Create palette
        if palette is None:
            palette = {k: v for k, v in zip(self.annot_labels.keys(), PALETTE)}

        if show_annots and self.annot_shapes is not None:
            for shape, name, label in self.annot_shapes:
                path = Path.make_compound_path(
                    Path(np.asarray(shape.exterior.coords)[:, :2]),
                    *[Path(np.asarray(ring.coords)[:, :2]) for ring in shape.interiors],
                )
                outline = PathPatch(
                    path,
                    label=name,
                    facecolor="none",
                    edgecolor=palette[name],
                    linewidth=linewidth,
                    alpha=1,
                )
                ax.add_patch(outline)
                fill = PathPatch(
                    path,
                    label=name,
                    facecolor=palette[name],
                    edgecolor="none",
                    alpha=alpha,
                )
                ax.add_patch(fill)
        if legend:
            from legendkit import cat_legend

            cat_legend(
                labels=palette.keys(),
                colors=palette.values(),
                loc="out right center",
                ax=ax,
            )
        ax.set_axis_off()
        return ax


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

            yield TissueContour(
                tissue_id=tissue_id,
                shape=cnt.geometry,
                as_array=as_array,
                dtype=dtype,
            )

    def tissue_images(
        self,
        key,
        level=-1,
        mask_bg=False,
        color_norm: str = None,
        format: str = "xyc",
        as_array: bool = False,
        dtype: np.dtype = None,
        shuffle: bool = False,
        seed: int = 0,
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

        # Determine if we should mask the background
        if isinstance(mask_bg, bool):
            if mask_bg:
                mask_bg = 0
            else:
                mask_bg = None
        else:
            mask_bg = int(mask_bg)

        contours = self._obj.shapes[key]
        if shuffle:
            contours = contours.sample(frac=1, random_state=seed)

        level_downsample = self._obj.properties.level_downsample[level]
        cn_func = _get_cn_func(color_norm)
        for ix, cnt in contours.iterrows():
            tissue_id = cnt["tissue_id"]
            minx, miny, maxx, maxy = cnt.geometry.bounds
            x = int(minx)
            y = int(miny)
            w = int((maxx - minx) // level_downsample)
            h = int((maxy - miny) // level_downsample)
            img = self._obj.reader.get_region(x, y, w, h, level=level)
            img = cn_func(img)

            # scale the shape
            shape = scale(
                cnt.geometry,
                xfact=1 / level_downsample,
                yfact=1 / level_downsample,
                origin=(0, 0),
            )
            yield TissueImage(
                tissue_id=tissue_id,
                shape=shape,
                image=img,
                as_array=as_array,
                dtype=dtype,
                format=format,
                mask_bg=mask_bg,
                downsample=level_downsample,
            )

    def tile_images(
        self,
        key,
        color_norm: str = None,
        format: str = "xyc",
        annot_key: str = None,
        annot_names: str | Sequence[str] = None,
        annot_labels: str | Dict[str, int] = None,
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
        color_norm : str, {"macenko", "reinhard"}, default: None
            Color normalization method.
        format : str, {"xyc", "cyx"}, default: "xyc"
            The channel format of the image.
        annot_key : str, default: None
            The key to the annotation table in :bdg-danger:`shapes` slot.
        annot_names : str or array of str, default: None
            The name of the annotation column in the annotation table or a list of names.
        annot_labels : str or dict, default: None
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

        create_annot_mask = False
        annot_tb = None
        annot_labels_dict = None
        mask_size = tile_spec.height, tile_spec.width

        if annot_key is not None:
            create_annot_mask = True
            if annot_names is None:
                raise ValueError(
                    "annot_name must be provided to create annotation mask."
                )
            if annot_labels is None:
                raise ValueError(
                    "annot_labels must be provided to create annotation mask."
                )

            annot_tb = self._obj.shapes[annot_key]
            if isinstance(annot_names, str):
                annot_names = annot_tb[annot_names]

            if isinstance(annot_labels, str):
                annot_labels = annot_tb[annot_labels]
            else:
                annot_labels = [annot_labels[n] for n in annot_names]
            annot_names = np.asarray(annot_names)
            annot_labels = np.asarray(annot_labels)
            annot_labels_dict = dict(zip(annot_names, annot_labels))

            # If there are more than 255 labels, use uint32
            if mask_dtype is None:
                if annot_labels.max() > 255:
                    mask_dtype = np.uint32
                else:
                    mask_dtype = np.uint8

        points = self._obj[key]
        if sample_n is not None:
            points = points.sample(n=sample_n, random_state=seed)
        elif shuffle:
            points = points.sample(frac=1, random_state=seed)

        cn_func = _get_cn_func(color_norm)
        downsample = tile_spec.base_downsample
        for _, row in points.iterrows():
            x = row["x"]
            y = row["y"]
            ix = row["id"]
            tix = row["tissue_id"]
            tile_bbox = row["geometry"]

            img = self._obj.reader.get_region(
                x,
                y,
                tile_spec.ops_width,
                tile_spec.ops_height,
                level=tile_spec.ops_level,
            )
            img = cv2.resize(img, (tile_spec.width, tile_spec.height))
            img = cn_func(img)

            if format == "cyx":
                img = img.transpose(2, 0, 1)

            annot_shapes = None
            annot_mask = None
            if create_annot_mask:
                annot_shapes = []
                sel = annot_tb.geometry.intersects(tile_bbox)  # return a boolean mask
                anno_mask = np.zeros(mask_size, dtype=mask_dtype)
                if sel.sum() > 0:
                    sel = sel.values
                    geos = annot_tb.geometry[sel]
                    names = annot_names[sel]
                    labels = annot_labels[sel]

                    for geo, name, label in zip(geos, names, labels):
                        geo = translate(geo, xoff=-x, yoff=-y)
                        geo = scale(
                            geo,
                            xfact=1 / downsample,
                            yfact=1 / downsample,
                            origin=(0, 0),
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
                        if (not output_geo.is_valid) or output_geo.is_empty:
                            continue
                        elif isinstance(output_geo, MultiPolygon):
                            output_geo = [p for p in output_geo.geoms]
                        else:
                            output_geo = [output_geo]
                        for p in output_geo:
                            annot_shapes.append((p, name, label))

            yield TileImage(
                id=ix,
                x=x,
                y=y,
                base_width=tile_spec.base_width,
                base_height=tile_spec.base_height,
                tissue_id=tix,
                image=img,
                annot_mask=annot_mask,
                annot_shapes=annot_shapes,
                annot_labels=annot_labels_dict,
            )
