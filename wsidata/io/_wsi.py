from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from rich.progress import track
from spatialdata import read_zarr, SpatialData
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale

from .._model import WSIData
from ..reader import get_reader, to_datatree


def open_wsi(
    wsi: str | Path,
    store: str = "auto",
    reader: Literal["openslide", "tiffslide", "bioformats"] = None,
    attach_images: bool = False,
    image_key: str = "wsi",
    save_images: bool = False,
    attach_thumbnail: bool = True,
    thumbnail_key: str = "wsi_thumbnail",
    thumbnail_size: int = 2000,
    save_thumbnail: bool = True,
    **kwargs,
):
    """Open a whole slide image.

    You can attach images and thumbnail to the SpatialData object. By default, only the thumbnail is attached,
    the thumbnail is a downsampled version of the whole slide image,
    the original image is not attached as it will make unnecessary copies of the data on disk
    when saving the SpatialData object.

    Parameters
    ----------
    wsi : str or Path
        The URL to whole slide image.
    store : str, optional
        The backed file path, by default will create
        a zarr file with the same name as the slide file.
        You can either supply a file path or a directory.
        If a directory is supplied, the zarr file will be created in that directory.
        This is useful when you want to store all zarr files in a specific location.
    reader : str, optional
        Reader to use, by default "auto", choosing available reader, first openslide, then tifffile.
    attach_images : bool, optional, default: False
        Whether to attach whole slide image to image slot in the spatial data object.
    image_key : str, optional
        The key to store the whole slide image, by default "wsi_thumbnail".
    save_images : bool, optional, default: False
        Whether to save the whole slide image to on the disk.
        Only works for wsi.save() method.
    attach_thumbnail : bool, optional, default: True
        Whether to attach thumbnail to image slot in the spatial data object.
    thumbnail_key : str, optional
        The key to store the thumbnail, by default "wsi_thumbnail".
    thumbnail_size : int, optional, default: 2000
        The size of the thumbnail.
    save_thumbnail : bool, optional, default: True
        Whether to save the thumbnail to on the disk.

        Only works for wsi.save() method.

    Returns
    -------
    :class:`WSIData`
        Whole slide image data.

    Examples
    --------

    .. code-block:: python

        >>> from wsidata import open_wsi
        >>> wsi = open_wsi("slide.svs")

    """
    # Check if the slide is a file or URL
    wsi = Path(wsi)
    if not wsi.exists():
        raise ValueError(f"Slide {wsi} not existed or not accessible.")

    sdata = None

    # Early attempt with reader
    ReaderClass = get_reader(reader, format=wsi.suffix)

    reader_instance = ReaderClass(wsi)
    if store == "auto":
        store = wsi.with_suffix(".zarr")
    else:
        if store is not None:
            store_path = Path(store)
            # We also support write store to a directory
            if store_path.is_dir():
                # If the directory is a zarr directory, we just use it
                if is_zarr_dir(store_path):
                    store = store_path
                # Otherwise, we create a zarr file in that directory
                else:
                    zarr_name = wsi.with_suffix(".zarr").name
                    store = store_path / zarr_name
            # If store is a not a directory, we assume it is a valid zarr file
            # WARNING: No guarantee
            else:
                store = store_path
    if store is not None:
        if store.exists():
            sdata = read_zarr(store)

    if sdata is None:
        sdata = SpatialData()

    exclude_elements = []

    if attach_images and image_key not in sdata:
        images_datatree = to_datatree(reader_instance)
        sdata.images[image_key] = images_datatree
        if not save_images:
            exclude_elements.append(image_key)

    if attach_thumbnail and thumbnail_key not in sdata:
        max_thumbnail_size = min(reader_instance.properties.shape)
        if thumbnail_size > max_thumbnail_size:
            thumbnail_size = max_thumbnail_size
        thumbnail = reader_instance.get_thumbnail(thumbnail_size)
        thumbnail_shape = thumbnail.shape
        origin_shape = reader_instance.properties.shape
        scale_x, scale_y = (
            origin_shape[0] / thumbnail_shape[0],
            origin_shape[1] / thumbnail_shape[1],
        )

        if thumbnail is not None:
            sdata.images[thumbnail_key] = Image2DModel.parse(
                np.asarray(thumbnail).transpose(2, 0, 1),
                dims=("c", "y", "x"),
                transformations={"global": Scale([scale_x, scale_y], axes=("x", "y"))},
            )
            if not save_thumbnail:
                exclude_elements.append(thumbnail_key)

    slide_data = WSIData.from_spatialdata(sdata, reader_instance)
    slide_data.set_exclude_elements(exclude_elements)
    if store is not None:
        slide_data.set_wsi_store(store)
    return slide_data


def agg_wsi(
    slides_table: pd.DataFrame,
    feature_key: str,
    tile_key: str = "tiles",
    agg_key: str = "agg_slide",
    wsi_col: str = None,
    store_col: str = None,
    pbar: bool = False,
    error: Literal["raise", "skip"] = "raise",
):
    """
    Aggregate feature from a whole slide image.

    Parameters
    ----------
    slides_table: pd.DataFrame
        The table of slides, including information of whole slide image and .zarr paths and metadata.

        Backed file path to the SpatialData file is optional, by default the same directory as the
        whole slide image.
    feature_key: str
        The feature key on which aggregation should be run on.
    tile_key: str
        The tile key.
    agg_key: str
        The output aggregation key in the varm slot.
    wsi_col: str
        The column name of the whole slide image paths.
    store_col: str
        The column name of the backed file.
    pbar: bool, default: False
        Whether to show progress bar.
    error: str
        Whether to raise error when file not existed.

    Returns
    -------
    AnnData
        The aggregated feature space.
    """

    if store_col is not None:
        backed_files = slides_table[store_col]
    elif wsi_col is not None:
        backed_files = slides_table[wsi_col].apply(
            lambda x: Path(x).with_suffix(".zarr")
        )
    else:
        raise ValueError("Either wsi_col or store_col must be provided.")

    jobs = []
    with ThreadPoolExecutor() as executor:
        for backed_f in backed_files:
            job = executor.submit(
                _agg_wsi, backed_f, feature_key, tile_key, agg_key, error
            )
            jobs.append(job)

    results = []
    for job in track(
        as_completed(jobs),
        total=len(jobs),
        disable=not pbar,
        description=f"Aggregation of {len(jobs)} slides",
    ):
        results.append(job.result())

    if error != "raise":
        mask = np.asarray([r is not None for r in results])
        X = np.vstack(np.asarray(results, dtype=object)[mask])
        slides_table = slides_table[mask]
    else:
        X = np.vstack(results)

    # Convert index to string
    slides_table.index = slides_table.index.astype(str)
    return AnnData(X, obs=slides_table)


def _agg_wsi(f, feature_key, tile_key, agg_key, error="raise"):
    """
    Aggregate feature from a whole slide image.

    Parameters
    ----------
    f: str
        The backed file path to the anndata file stored as .zarr.
    feature_key: str
        The feature key on which aggregation should be run on.
    tile_key: str
        The tile key.
    agg_key:
        The output aggregation key in the varm slot.
    error: str
        Whether to raise error when file not existed.

    Returns
    -------
    np.ndarray
        The aggregated feature space.

    """
    if not Path(f).exists():
        if error == "raise":
            raise ValueError(f"File {f} not existed.")
        else:
            return None
    try:
        import zarr
        from anndata import read_zarr

        tables = zarr.open(f"{f}/tables")
        available_keys = list(tables.keys())
        if feature_key not in available_keys:
            feature_key = f"{feature_key}_{tile_key}"
        s = read_zarr(f"{f}/tables/{feature_key}")
        if agg_key in s.varm:
            return np.squeeze(s.varm[agg_key])
        else:
            raise ValueError(f"Aggregation key {agg_key} not found.")
    except Exception as e:
        if error == "raise":
            raise e
        else:
            return None


def is_zarr_dir(path):
    """
    Detect if the given directory is a Zarr storage using the Zarr library.

    Parameters:
        path : The path to the directory.

    Returns:
        bool: True if the directory is a Zarr storage, False otherwise.
    """
    import zarr

    try:
        zarr.open_group(str(path), mode="r")
        return True
    except Exception:
        return False
