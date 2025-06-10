from __future__ import annotations

import base64
import io
import re
import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Dict, Generator, List, Literal, Sequence, Tuple

import cv2
import numpy as np
from PIL.Image import fromarray
from shapely import MultiPolygon, Polygon, clip_by_rect
from shapely.affinity import scale, translate

if TYPE_CHECKING:
    from .._model import WSIData


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

REPR_BOX_STYLE = (
    'style="border: 2px solid #C68FE6; border-radius: 8px;'
    'padding: 10px 15px; display: inline-block;"'
)


def _html_attributes(attrs):
    """Max 4 attributes per line, show in card style"""
    n = 4
    n_attr = len(attrs)
    n_row = n_attr // n + 1
    rows = []
    for i in range(n_row):
        row = attrs[i * n : (i + 1) * n]
        row = "".join(
            f"<p style='border: 1px solid #C68FE6; padding: 3pt; "
            f"border-radius: 4px; text-align: center; margin-bottom: 2pt;'>"
            f"{attr}</p>"
            for attr in row
        )
        rows.append(row)
    return "".join(f"<div style='display: inline-block;'>{row}</div>" for row in rows)


class TissueContour:
    """The data container return by :meth:`wsidata.iter.tissue_contours <wsidata.IterAccessor.tissue_contours>`

    Attributes
    ----------
    tissue_id : int
        The id of tissue
    shape : :class:`Polygon <shapely.Polygon>`
        The contour of the tissue
    contour : :class:`Polygon <shapely.Polygon>`
        The contour of the tissue
    holes : array of :class:`Polygon <shapely.Polygon>`
        The holes of the tissue
    x : int
        The x-coordinate of the tissue
    y : int
        The y-coordinate of the tissue
    width : int
        The width of the tissue
    height : int
        The height of the tissue

    """

    def __init__(self, tissue_id, shape):
        self.tissue_id = tissue_id
        self.shape = shape

    @cached_property
    def x(self):
        """The x-coordinate of the tissue"""
        return int(self.shape.bounds[0])

    @cached_property
    def y(self):
        """The y-coordinate of the tissue"""
        return int(self.shape.bounds[1])

    @cached_property
    def width(self):
        """The width of the tissue"""
        return int(self.shape.bounds[2] - self.shape.bounds[0])

    @cached_property
    def height(self):
        """The height of the tissue"""
        return int(self.shape.bounds[3] - self.shape.bounds[1])

    @property
    def contour(self) -> Polygon:
        """The contour of the tissue"""
        return Polygon(self.shape.exterior.coords)

    @property
    def holes(self) -> List[Polygon]:
        """The holes of the tissue"""
        return [Polygon(h) for h in self.shape.interiors]

    _attrs = [
        "tissue_id",
        "shape",
        "contour",
        "holes",
        "x",
        "y",
        "width",
        "height",
    ]

    def __repr__(self):
        return f"TissueContour with attributes: \n{', '.join(self._attrs)}"

    def _repr_html_(self):
        svg_shape = self.shape._repr_svg_()
        svg_shape = re.sub(r'(<svg[^>]*?)width="[^"]*"', rf'\1width="{130}"', svg_shape)
        svg_shape = re.sub(
            r'(<svg[^>]*?)height="[^"]*"', rf'\1height="{130}"', svg_shape
        )
        return f"""
                <div {REPR_BOX_STYLE}>
                    <strong style="font-size: 1.1em; color: #C68FE6;">TissueContour</strong>
                    <p style="margin-bottom: 0">Attributes:</p>
                    <div style="display: flex; gap: 10px;">
                        {_html_attributes(self._attrs)}
                        {svg_shape}
                    </div>

                </div>
                """

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
    """
    The data container return by :meth:`wsidata.iter.tissue_images <wsidata.IterAccessor.tissue_images>`

    This container shares the same attributes as :class:`TissueContour <wsi.accessors.iter.TissueContour>`.

    Attributes
    ----------
    image : :class:`np.ndarray <numpy.ndarray>`
        The tissue image.
    mask : :class:`np.ndarray <numpy.ndarray>`
        The tissue mask.
    masked_image : :class:`np.ndarray <numpy.ndarray>`
        The masked image with the background masked.

    """

    def __init__(
        self,
        tissue_id,
        shape,
        image,
        format: Literal["cyx", "yxc"] = "yxc",
        mask_bg: int = None,
        downsample=1.0,
    ):
        super().__init__(tissue_id, shape)
        self._image = image
        self.format = format
        self.mask_bg = mask_bg
        self.downsample = downsample

    @property
    def image(self):
        if self.format == "cyx":
            # "yxc" -> "cyx"
            return self._image.transpose(2, 0, 1)
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
            cv2.fillPoly(mask, [coords], 1)  # noqa

            # Fill the holes with 0
            for hole in self.holes:
                if isinstance(hole, Polygon):
                    hole = np.array(hole.exterior.coords)
                hole -= [offset_x, offset_y]
                hole = hole.astype(np.int32)
                cv2.fillPoly(mask, [hole], 0)  # noqa
            mask = mask.astype(bool)
        return mask

    @cached_property
    def masked_image(self) -> np.ndarray | None:
        """A masked image with the background masked will be returned
        if the mask_bg is not None."""
        if self.mask_bg is None:
            return None
        mask = self.mask
        image = self.image.copy()
        if mask is not None:
            # Fill everything that is not the contour
            # (which is background) with 0
            image[mask != 1] = self.mask_bg

        return image

    _attrs = [
        "tissue_id",
        "shape",
        "contour",
        "holes",
        "x",
        "y",
        "width",
        "height",
        "image",
        "mask",
        "masked_image",
    ]

    def __repr__(self):
        return f"TissueImage with attributes: \n{', '.join(self._attrs)}"

    @cached_property
    def _html_thumbnail(self) -> str:
        buffer = io.BytesIO()
        image = self.image if self.mask_bg is None else self.masked_image
        if self.format == "cyx":
            image = image.transpose(1, 2, 0)
        image = fromarray(image)
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _repr_html_(self):
        return f"""
                <div {REPR_BOX_STYLE}>
                    <strong style="font-size: 1.1em; color: #C68FE6;">TissueImage</strong>
                    <p style="margin-bottom: 0">Attributes:</p>
                    <div style="display: flex; gap: 10px;">
                        {_html_attributes(self._attrs)}
                        <img src="data:image/png;base64,{self._html_thumbnail}" 
                            style="width: 130px; border-radius: 8px;">
                    </div>
                </div>
                """

    def plot(self, ax=None, masked=None, **kwargs):
        """
        Plot the tissue image.

        Parameters
        ----------
        ax : :class:`Axes <matplotlib.axes.Axes>`
            The axes to plot the image.
        masked : bool
            Whether to show the masked image.
        **kwargs
            Additional keyword arguments for :meth:`imshow <matplotlib.axes.Axes.imshow>`.

        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        img = self.image
        if masked is None:
            if self.mask_bg is not None:
                img = self.masked_image
        if self.format == "cyx":
            img = img.transpose(1, 2, 0)
        extent = (self.x, self.x + self.width, self.y + self.height, self.y)
        ax.imshow(img, extent=extent, **kwargs)
        return ax


class TileImage:
    """
    The data container return by :meth:`wsidata.iter.tile_images <wsidata.IterAccessor.tile_images>`

    Attributes
    ----------
    id : int
        The tile id.
    x : int
        The x-coordinate of the tile.
    y : int
        The y-coordinate of the tile.
    width : int
        The width of the tile image.
    height : int
        The height of the tile image.
    tissue_id : int
        The tissue id.
    image : :class:`np.ndarray <numpy.ndarray>`
        The tile image.
    annot_mask : :class:`np.ndarray <numpy.ndarray>`
        The annotation mask.
    annot_shapes : array of (:class:`Polygon <shapely.Polygon>`, name, label)
        The annotation shapes.
    annot_labels : dict
        The annotation labels mapping.
    norm_annot_shapes : array of (:class:`Polygon <shapely.Polygon>`, name, label)
        The normalized annotation shapes, the coordinates are normalized to the 0-1 range.
    has_annot : bool
        Whether the tile has annotation.

    """

    id: int
    x: int
    y: int
    image: np.ndarray
    tissue_id: int | None = None
    annot_mask: np.ndarray | None = None
    annot_shapes: List[Tuple[Polygon, str, int]] | None = None
    annot_labels: Dict[str, int] | None = None

    def __init__(
        self,
        id,
        x,
        y,
        image,
        tissue_id=None,
        annot_mask=None,
        annot_shapes=None,
        annot_labels=None,
    ):
        self.id = id
        self.x = x
        self.y = y
        self.tissue_id = tissue_id
        self.image = image
        self.annot_mask = annot_mask
        self.annot_shapes = annot_shapes
        self.annot_labels = annot_labels

    @property
    def width(self):
        """The width of the tile image."""
        return self.image.shape[1]

    @property
    def height(self):
        """The height of the tile image."""
        return self.image.shape[0]

    @cached_property
    def norm_annot_shapes(self):
        """The normalized annotation shapes, the coordinates are normalized to the 0-1 range."""
        if self.annot_shapes is not None:
            new_shapes = []
            for shape, name, label in self.annot_shapes:
                norm_shape = _normalize_polygon(shape, (0, 0, self.width, self.height))
                new_shapes.append((norm_shape, name, self.annot_labels[name]))
            return new_shapes

    @property
    def has_annot(self):
        """Whether the tile has annotation."""
        return self.annot_shapes is not None

    _attrs = [
        "id",
        "x",
        "y",
        "width",
        "height",
        "tissue_id",
        "image",
        "annot_mask",
        "annot_shapes",
        "annot_labels",
        "norm_annot_shapes",
        "has_annot",
    ]

    def __repr__(self):
        if self.annot_shapes is not None:
            n_shapes = len(self.annot_shapes)
            shape_text = "annotation" if n_shapes == 1 else "annotations"
            shapes_repr = f" ({n_shapes} {shape_text})"
        else:
            shapes_repr = ""

        return (
            f"TileImage(id={self.id}, x={self.x}, y={self.y}, "
            f"tissue_id={self.tissue_id}){shapes_repr} "
            f"with attributes: \n"
            f"{', '.join(self._attrs)}"
        )

    @cached_property
    def _html_thumbnail(self) -> str:
        buffer = io.BytesIO()
        image = self.image
        image = fromarray(image)
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _repr_html_(self):
        return f"""
                <div {REPR_BOX_STYLE}>
                    <strong style="font-size: 1.1em; color: #C68FE6;">TileImage</strong>
                    <p style="margin-bottom: 0">Attributes:</p>
                    <div style="display: flex; gap: 10px;">
                        {_html_attributes(self._attrs)}
                        <img src="data:image/png;base64,{self._html_thumbnail}"
                        style="width: 150px; border-radius: 8px;">
                    </div>

                </div>
                """

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
        """
        Plot the tile image.

        Parameters
        ----------
        ax : :class:`Axes <matplotlib.axes.Axes>`
            The axes to plot the image.
        show_annots : bool
            Whether to show the annotations.
        palette : dict
            The palette for the annotation colors.
        legend : bool
            Whether to show the legend.
        alpha : float
            The transparency of the annotation.
        linewidth : int
            The width of the annotation line.
        **kwargs
            Additional keyword arguments for :meth:`imshow <matplotlib.axes.Axes.imshow>`.

        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path

        if ax is None:
            ax = plt.gca()

        ax.imshow(self.image, **kwargs)

        # Create palette
        if self.has_annot:
            if palette is None:
                palette = {k: v for k, v in zip(self.annot_labels.keys(), PALETTE)}

        if show_annots and self.has_annot:
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
        if legend and self.has_annot:
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

    Usage: :code:`wsidata.iter`

    The iter accessor will return a data container,
    you can access the data within container using attributes.
    All container has :code:`.plot()` method to visualize its content.

    """

    def __init__(self, obj: WSIData):
        self._obj = obj

    def tissue_contours(
        self,
        key,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Generator[TissueContour]:
        """A generator to extract tissue contours from the WSI.

        Parameters
        ----------
        key : str
            The tissue key.
        shuffle : bool, default: False
            If True, return tissue contour in random order.
        seed : int, default: 0

        Returns
        -------
        :class:`TissueContour <wsidata.accessors.iter.TissueContour>`

        """

        contours = self._obj.shapes[key]

        if shuffle:
            contours = contours.sample(frac=1, random_state=seed)

        for ix, cnt in contours.iterrows():
            tissue_id = cnt["tissue_id"]

            yield TissueContour(
                tissue_id=tissue_id,
                shape=cnt.geometry,
            )

    def tissue_images(
        self,
        key,
        level=-1,
        mask_bg=False,
        color_norm: str = None,
        format: str = "yxc",
        shuffle: bool = False,
        seed: int = 0,
    ) -> Generator[TissueImage]:
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
        format : str, {"yxc", "cyx"}, default: "yxc"
            The channel format of the image.

        Returns
        -------
        :class:`TissueImage <wsidata.accessors.iter.TissueImage>`

        """

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
                format=format,
                mask_bg=mask_bg,
                downsample=level_downsample,
            )

    def tile_images(
        self,
        key,
        color_norm: str = None,
        format: str = "yxc",
        annot_key: str = None,
        annot_names: str | Sequence[str] = None,
        annot_labels: str | Dict[str, int] = None,
        mask_dtype: np.dtype = None,
        shuffle: bool = False,
        sample_n: int = None,
        seed: int = 0,
    ) -> Generator[TileImage]:
        """Extract tile images from the WSI.

        Parameters
        ----------
        key : str
            The tile key.
        color_norm : str, {"macenko", "reinhard"}, default: None
            Color normalization method.
        format : str, {"yxc", "cyx"}, default: "yxc"
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
        :class:`TileImage <wsidata.accessors.iter.TileImage>`

        """
        tile_spec = self._obj.tile_spec(key)
        create_annot_mask = False
        annot_tb = None
        annot_labels_dict = None
        mask_size = None, None

        if tile_spec is None:
            warnings.warn(
                f"Tile spec cannot be found for the {key}. It's not valid tiles. "
                f"Tile image can still be extracted but may not be optimized."
            )
        else:
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
        has_tissue_id = "tissue_id" in points.columns
        for _, row in points.iterrows():
            ix = row["tile_id"]
            tile_bbox = row["geometry"]
            x, y = tile_bbox.bounds[:2]
            if has_tissue_id:
                tix = row["tissue_id"]
            else:
                tix = None

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
                annot_mask = np.zeros(mask_size, dtype=mask_dtype)
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
                        cv2.fillPoly(annot_mask, [cnt], int(label))  # noqa
                        cv2.fillPoly(annot_mask, holes, 0)  # noqa
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
                image=img,
                tissue_id=tix,
                annot_mask=annot_mask,
                annot_shapes=annot_shapes,
                annot_labels=annot_labels_dict,
            )
