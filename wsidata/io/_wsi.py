from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from numbers import Integral
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from rich.progress import track
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale

from .._model import WSIData
from .._utils import find_stack_level
from ..reader import READERS, to_datatree


def open_wsi(
    wsi: str | Path | SpatialData,
    store: str = "auto",
    reader: str = None,
    scene: int | None = None,
    attach_images: bool = False,
    image_key: str = None,
    save_images: bool = False,
    attach_thumbnail: bool = False,
    thumbnail_key: str = "wsi_thumbnail",
    thumbnail_size: int = 2000,
    save_thumbnail: bool = False,
    **kwargs,  # Kept for backward compatibility.
):
    """Open a whole slide image.

    You can attach images and thumbnails to the SpatialData object. By default, only the thumbnail is attached,
    the thumbnail is a downsampled version of the whole slide image,
    the original image is not attached as it will make unnecessary copies of the data on disk
    when saving the SpatialData object.

    Parameters
    ----------
    wsi : str or Path or SpatialData
        The path to whole slide image, or an existing SpatialData object.
        When passing a SpatialData object, ``image_key`` must be provided.
    store : str, optional
        The backed file path, by default will create
        a zarr file with the same name as the slide file.
        You can either supply a file path or a directory.
        If a directory is supplied, the zarr file will be created in that directory.
        This is useful when you want to store all zarr files in a specific location.
        Pass ``None`` to skip persistence (no zarr store will be set).
    reader : str, optional
        Reader to use, by default ``None``. Passing ``None`` enables automatic reader
        selection. To check avaiable readers: `print(wsidata.READERS)`
    scene : int, optional
        Zero-based scene to open. By default, each reader selects its primary image.
    attach_images : bool, optional, default: False
        Whether to attach whole slide image to image slot in the spatial data object.
    image_key : str, optional
        The key to store the whole slide image, by default "wsi".
        If the wsi is a SpatialData object, the image from this key will be used as the whole slide image.
    save_images : bool, optional, default: False
        Whether to save the whole slide image to on the disk.
        Only works for wsi.save() method.
    attach_thumbnail : bool, optional, default: False
        Whether to attach thumbnail to image slot in the spatial data object.
    thumbnail_key : str, optional
        The key to store the thumbnail, by default "wsi_thumbnail".
    thumbnail_size : int, optional, default: 2000
        The size of the thumbnail.
    save_thumbnail : bool, optional, default: False
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
    if scene is not None:
        if isinstance(scene, bool) or not isinstance(scene, Integral):
            raise TypeError("scene must be a non-negative integer or None.")
        scene = int(scene)
        if scene < 0:
            raise ValueError("scene must be a non-negative integer or None.")

    # -- SpatialData input path --
    if isinstance(wsi, SpatialData):
        if image_key is None:
            raise ValueError(
                "When reading from SpatialData, image_key must be provided."
            )
        # Warn about ignored parameters
        _ignored = []
        if store != "auto":
            _ignored.append("store")
        if reader is not None:
            _ignored.append("reader")
        if attach_images:
            _ignored.append("attach_images")
        if attach_thumbnail:
            _ignored.append("attach_thumbnail")
        if _ignored:
            warnings.warn(
                f"When wsi is a SpatialData object, the following parameters "
                f"are ignored: {', '.join(_ignored)}.",
                UserWarning,
                stacklevel=find_stack_level(),
            )

        from ..reader import SpatialDataImage2DReader

        reader_instance = SpatialDataImage2DReader(wsi[image_key], key=image_key)
        reader_instance.validate_scene(scene, 1)
        return WSIData.from_spatialdata(wsi, reader_instance)

    # -- File path input --
    sdata = None
    wsi = Path(wsi)
    if not wsi.exists():
        raise ValueError(f"Slide {wsi} does not exist, or is not accessible.")

    reader_instance = READERS.try_open(wsi, reader=reader, scene=scene)
    try:
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
            store = wsi.parent / _default_store_name(wsi, reader_instance)
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
                        zarr_name = _default_store_name(wsi, reader_instance)
                        store = store_path / zarr_name
                # If store is a not a directory, we assume it is a valid zarr file
                # WARNING: No guarantee
                else:
                    store = store_path
        if store is not None:
            if store.exists():
                sdata = read_zarr(store)
                _validate_store_scene(sdata, reader_instance, store)

        exclude_elements = []
        sdata_images = {}

        if attach_images:
            if image_key is None:
                image_key = "wsi"
            if sdata is None or image_key not in sdata:
                images_datatree = to_datatree(reader_instance)
                sdata_images[image_key] = images_datatree
                if not save_images:
                    exclude_elements.append(image_key)

        # Warn if image_key and thumbnail_key would collide
        if attach_images and attach_thumbnail:
            effective_image_key = image_key if image_key is not None else "wsi"
            if effective_image_key == thumbnail_key:
                warnings.warn(
                    f"image_key and thumbnail_key are both '{thumbnail_key}'. "
                    f"The thumbnail will overwrite the full image in the images slot.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )

        if attach_thumbnail:
            if sdata is None or thumbnail_key not in sdata:
                max_thumbnail_size = min(reader_instance.properties.shape)
                if thumbnail_size > max_thumbnail_size:
                    thumbnail_size = max_thumbnail_size
                thumbnail = reader_instance.get_thumbnail(thumbnail_size)

                if thumbnail is not None:
                    thumbnail_shape = thumbnail.shape  # numpy (H, W, C)
                    origin_h, origin_w = reader_instance.properties.shape
                    # x-axis = width, y-axis = height
                    scale_x = origin_w / thumbnail_shape[1]
                    scale_y = origin_h / thumbnail_shape[0]

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
    except Exception:
        reader_instance.detach_reader()
        raise
    return slide_data


def _default_store_name(wsi: Path, reader) -> str:
    """Return a scene-safe default Zarr store name."""
    if reader.n_scenes > 1:
        return f"{wsi.stem}.scene-{reader.scene}.zarr"
    return wsi.with_suffix(".zarr").name


def _validate_store_scene(sdata, reader, store):
    """Ensure an existing store belongs to the selected scene."""
    if reader.n_scenes <= 1:
        return
    stored_properties = sdata.attrs.get(WSIData.SLIDE_PROPERTIES_KEY, {})
    if "scene" not in stored_properties:
        raise ValueError(
            f"Store '{store}' has no scene metadata and cannot be safely used "
            "with a multi-scene image. Choose a new store path or migrate the store."
        )
    stored_scene = int(stored_properties["scene"])
    if stored_scene != reader.scene:
        raise ValueError(
            f"Store '{store}' belongs to scene {stored_scene}, but scene "
            f"{reader.scene} was requested."
        )


def _resolve_backed_files(slides_table, wsi_col, store_col):
    """Resolve backed file paths from slides_table columns."""
    if store_col is not None:
        return slides_table[store_col].astype(str)
    elif wsi_col is not None:
        return slides_table[wsi_col].apply(lambda x: str(Path(x).with_suffix(".zarr")))
    else:
        raise ValueError("Either wsi_col or store_col must be provided.")


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
    from anndata import AnnData

    if agg_key is None and agg_by is None:
        agg_key = "agg_slide"
    elif agg_key is None and agg_by is not None:
        if isinstance(agg_by, str):
            agg_by = [agg_by]
        agg_key = f"agg_{'_'.join(agg_by)}"

    backed_files = _resolve_backed_files(slides_table, wsi_col, store_col)

    slides_table = slides_table.copy()
    slides_table["_job_id"] = np.arange(len(slides_table))

    jobs = []
    with ThreadPoolExecutor() as executor:
        for job_id, backed_f in enumerate(backed_files):
            job = executor.submit(
                _agg_wsi, str(backed_f), feature_key, tile_key, agg_key, error
            )
            job.job_id = job_id
            jobs.append(job)

        # Collect results keyed by job_id
        # Use a dict indexed by job_id so we can reconstruct input order
        results = {}
        for job in track(
            as_completed(jobs),
            total=len(jobs),
            disable=not pbar,
            description=f"Aggregation of {len(jobs)} slides",
        ):
            results[job.job_id] = job.result()

    # Rebuild in original input order, filtering out failed slides
    succeeded_ids = []
    features = []
    features_annos = []
    for job_id in range(len(backed_files)):
        feature, feature_annos = results[job_id]
        if feature is None:
            continue
        succeeded_ids.append(job_id)
        features.append(feature)
        if feature_annos is not None:
            feature_annos = feature_annos.copy()
            feature_annos["_job_id"] = job_id
            features_annos.append(feature_annos)

    if len(features) == 0:
        warnings.warn(
            "No slides produced valid features. Returning empty AnnData.",
            UserWarning,
            stacklevel=find_stack_level(),
        )
        slides_table = slides_table.drop(columns="_job_id")
        return AnnData(obs=slides_table.iloc[:0])

    X = np.vstack(features)
    # Filter slides_table to only succeeded slides (positional, not label-based)
    slides_table = slides_table.iloc[succeeded_ids].copy()

    if len(features_annos) > 0:
        annos = pd.concat(features_annos, ignore_index=True)
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
            return None, None
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


def concat_feature_anndata(
    slides_table: pd.DataFrame,
    feature_key,
    tile_key,
    wsi_col: str = None,
    store_col: str = None,
    pbar: bool = False,
    error: Literal["raise", "skip"] = "raise",
    as_anncollection: bool = False,
):
    """Aggregates features from multiple slides into an AnnData or AnnCollection object.

    This function collects and combines data from slide datasets specified in a given
    table, allowing flexible aggregation by feature and tile keys.

    Parameters
    ----------
    slides_table : pandas.DataFrame
        The dataframe specifying slides to aggregate, their feature
        and tile keys, as well as optional storage or WSI column information.
    feature_key : str
        Key or identifier for the feature within the datasets. Details should
        be specific to the slide data format.
    tile_key : str
        Key or identifier for the tile data within the datasets. Details should
        be specific to the data structure.
    wsi_col : str, optional
        Optional name of the column in the table that contains whole slide image (WSI)
        paths. Either `wsi_col` or `store_col` must be provided.
    store_col : str, optional
        Optional name of the column specifying storage information for slides.
        Either `store_col` or `wsi_col` is required.
    pbar : bool, default False
        A flag to toggle progress bar visibility during processing.
    error : {"raise", "skip"}, default "raise"
        Policy for error handling during individual slide aggregation. Valid options are
        "raise" to terminate on error or "skip" to skip files with errors.
    as_anncollection : bool, default False
        Flag to determine if the return type is an AnnCollection object instead
        of a concatenated AnnData object. If True, returns AnnCollection.

    Returns
    -------
    AnnData or AnnCollection
        Aggregated data from slides either as an AnnData or AnnCollection object,
        depending on the value of `as_anncollection`.
    """
    from anndata import concat

    backed_files = _resolve_backed_files(slides_table, wsi_col, store_col)

    jobs = []
    with ThreadPoolExecutor() as executor:
        for job_id, backed_f in enumerate(backed_files):
            job = executor.submit(
                _concat_feature_anndata, str(backed_f), feature_key, tile_key, error
            )
            job.job_id = job_id
            jobs.append(job)

        # Collect results keyed by job_id
        results = {}
        for job in track(
            as_completed(jobs),
            total=len(jobs),
            disable=not pbar,
            description=f"Concatenation of {len(jobs)} slides",
        ):
            results[job.job_id] = job.result()

    # Rebuild in original input order, filtering out None results
    adatas = {}
    for job_id in range(len(backed_files)):
        adata = results[job_id]
        if adata is not None:
            adatas[job_id] = adata

    if len(adatas) == 0:
        warnings.warn(
            "No slides produced valid features. Returning empty AnnData.",
            UserWarning,
            stacklevel=find_stack_level(),
        )
        from anndata import AnnData

        return AnnData()

    if as_anncollection:
        from anndata.experimental import AnnCollection

        return AnnCollection(
            adatas,
            join_obs=None,
            join_vars=None,
            join_obsm=None,
            label="slide_name",
            index_unique="-",
        )
    else:
        return concat(
            adatas,
            join="outer",
            label="slide_name",
            index_unique="-",
        )


def _concat_feature_anndata(f, feature_key, tile_key, error="raise"):
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
        return s
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
