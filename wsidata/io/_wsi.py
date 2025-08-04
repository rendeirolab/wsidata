from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from rich.progress import track
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale

from .._model import WSIData
from .._utils import find_stack_level
from ..reader import to_datatree, try_reader


def open_wsi(
    wsi: str | Path,
    store: str = "auto",
    reader: Literal["openslide", "tiffslide", "bioformats"] = None,
    attach_images: bool = False,
    image_key: str = "wsi",
    save_images: bool = True,
    attach_thumbnail: bool = True,
    thumbnail_key: str = "wsi_thumbnail",
    thumbnail_size: int = 2000,
    save_thumbnail: bool = True,
    **kwargs,
):
    """Open a whole slide image.

    You can attach images and thumbnails to the SpatialData object. By default, only the thumbnail is attached,
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
    save_images : bool, optional, default: True
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

        Only works for wsi.write() method.

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
    reader_instance = try_reader(wsi, reader=reader)

    # Check if the image is not pyramidal and too large
    if reader_instance.properties.n_level <= 1:
        height, width = reader_instance.properties.shape
        if height > 10000 or width > 10000:
            warnings.warn(
                f"The image is not pyramidal (n_level={reader_instance.properties.n_level}) "
                f"and has a large size ({width}x{height} pixels). "
                "This may cause performance issues. "
                "Consider generating pyramids for this image using vips or bioformats).",
                UserWarning,
                stacklevel=find_stack_level(),
            )

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

    exclude_elements = []
    sdata_images = {}

    if attach_images:
        if sdata is None or image_key not in sdata:
            images_datatree = to_datatree(reader_instance)
            sdata_images[image_key] = images_datatree
            if not save_images:
                exclude_elements.append(image_key)

    if attach_thumbnail:
        if sdata is None or thumbnail_key not in sdata:
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
                sdata_images[thumbnail_key] = Image2DModel.parse(
                    np.asarray(thumbnail).transpose(2, 0, 1),
                    dims=("c", "y", "x"),
                    transformations={
                        "global": Scale([scale_x, scale_y], axes=("x", "y"))
                    },
                )
                if not save_thumbnail:
                    exclude_elements.append(thumbnail_key)

    if sdata is None:
        sdata = SpatialData(images=sdata_images)
    else:
        for key, img in sdata_images.items():
            sdata.images[key] = img

    slide_data = WSIData.from_spatialdata(sdata, reader_instance)
    slide_data.set_exclude_elements(exclude_elements)
    if store is not None:
        slide_data.set_wsi_store(store)
    return slide_data


def agg_wsi(
    slides_table: pd.DataFrame,
    feature_key: str,
    tile_key: str = "tiles",
    agg_key: str = None,
    agg_by: str | Sequence[str] = None,
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
    agg_by: str or array of str
        The keys that have been used to aggregate the features.
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

    if agg_key is None and agg_by is None:
        agg_key = "agg_slide"
    elif agg_key is None and agg_by is not None:
        if isinstance(agg_by, str):
            agg_by = [agg_by]
        agg_key = f"agg_{'_'.join(agg_by)}"

    if store_col is not None:
        backed_files = slides_table[store_col]
    elif wsi_col is not None:
        backed_files = slides_table[wsi_col].apply(
            lambda x: Path(x).with_suffix(".zarr")
        )
    else:
        raise ValueError("Either wsi_col or store_col must be provided.")

    slides_table = slides_table.copy()
    slides_table["_job_id"] = np.arange(len(slides_table))

    jobs = []
    with ThreadPoolExecutor() as executor:
        for job_id, backed_f in enumerate(backed_files):
            job = executor.submit(
                _agg_wsi, backed_f, feature_key, tile_key, agg_key, error
            )
            job.job_id = job_id
            jobs.append(job)

        # Store results with their job_ids to maintain original order
        features_with_ids = []
        features_annos = []
        for job in track(
            as_completed(jobs),
            total=len(jobs),
            disable=not pbar,
            description=f"Aggregation of {len(jobs)} slides",
        ):
            feature, feature_annos = job.result()
            if feature is not None:
                # Store feature with its job_id to preserve original order
                features_with_ids.append((job.job_id, feature))
                # we will have feature_annos only if aggregation not at slide level
                if feature_annos is not None:
                    feature_annos["_job_id"] = job.job_id
                    features_annos.append(feature_annos)

        # Sort features by job_id to restore original order
        features_with_ids.sort(key=lambda x: x[0])
        features = [feature for _, feature in features_with_ids]

    mask = np.asarray([r is not None for r in features])
    X = np.vstack(features)
    slides_table = slides_table.loc[mask]
    if len(features_annos) > 0:
        annos = pd.concat(features_annos, ignore_index=True)
        # Add job_id to the slides table
        slides_table = pd.merge(slides_table, annos, on="_job_id", how="left")

    slides_table = slides_table.drop(columns="_job_id")
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

        agg_ops = s.uns.get("agg_ops")
        if agg_ops is None:
            raise ValueError(
                f"Aggregation operations not found for {f}. Did you run feature aggregation?"
            )
        agg_annos = agg_ops.get(agg_key)
        if agg_annos is None:
            raise ValueError(f"Aggregation key {agg_key} not found.")

        if "features" in agg_annos:
            feature = agg_annos["features"]
        else:
            feature = s.varm[agg_key].T
        feature_annos = None
        if len(agg_annos) > 0:
            if "values" in agg_annos:
                feature_annos = pd.DataFrame(
                    agg_annos["values"], columns=agg_annos["keys"]
                )
        return feature, feature_annos

    except Exception as e:
        if error == "raise":
            raise e
        else:
            return None, None


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
