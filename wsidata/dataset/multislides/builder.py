from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Type

import geopandas as gpd
import numpy as np
import pandas as pd
from zarr.errors import PathNotFoundError

from ..._utils import find_stack_level
from .dataset import CachedFeaturesDataset, IterableFeaturesDataset
from .sampler import BaseTileDatasetSampler


class DatasetBuilder(ABC):
    sampler: BaseTileDatasetSampler

    def set_sampler(self, key):
        self.sampler = BaseTileDatasetSampler.registry[key]

    @abstractmethod
    def split(self, val=0.15, test=0.15):
        pass


class FeaturesDatasetBuilder(DatasetBuilder):
    """
    Build train and test dataset from multiple slides.

    The train/val/test split is guaranteed with no overlap between slides.

    .. note::
        This is still a provisional API, may change in the future without notice.

    Parameters
    ----------

    stores : list
        A list of paths pointing to the WSIData stores, must be .zarr file
    tile_key : str
        The key of the tile table.
    feature_key : str
        The key for the tile features.
    target_key : str
        The key for the Y variable in the observation table.
    skip_class : list
        The classes to skip.
    sampler: {'no-balance', 'undersample'}
        How to sample the tile features for dataset construction.
        The no-balance option will return all the tiles for each class.
        The undersample option will ensure that each class has the same number of tiles.
    n_per_class: int
        The number of tiles to sample for each class.
    in_memory: bool
        If True, load the dataset into memory.
        If False, an IterableDataset will be returned.

    """

    sampler: Type[BaseTileDatasetSampler]

    def __init__(
        self,
        stores,  # Can be wsi path, store path
        tile_key=None,
        feature_key=None,
        target_key=None,
        target_transform=None,
        skip_class=None,
        sampler: str = "undersample",
        n_per_class=None,
        in_memory=True,
        seed=0,
    ):
        self.tile_key = tile_key
        self.feature_key = feature_key
        self.target_key = target_key
        self.target_transform = target_transform
        self.n_per_class = n_per_class
        self.in_memory = in_memory
        self.seed = seed

        tiles = []
        with ThreadPoolExecutor() as executor:
            tasks = []
            for f in stores:
                task = executor.submit(self._get_targets, f)
                tasks.append(task)

            for s, t in zip(stores, tasks):
                targets = t.result()
                if target_transform is not None:
                    targets = target_transform(targets)
                tiles.append(
                    pd.DataFrame(
                        {
                            target_key: targets,
                            "index": np.arange(len(targets)),
                            "table_path": f"{s}/tables/{self.feature_key}",
                        }
                    )
                )

        dataset_table = pd.concat(tiles, ignore_index=True)
        if dataset_table[target_key].isna().any():
            warnings.warn(
                "It looks like the target column contains NaN values. "
                "The NaN rows will be removed.",
                stacklevel=find_stack_level(),
            )
        if skip_class is not None:
            dataset_table = dataset_table[~dataset_table[target_key].isin(skip_class)]
        dataset_table = dataset_table.dropna(subset=[target_key]).reset_index(drop=True)
        dataset_table[target_key] = dataset_table[self.target_key].astype("category")
        dataset_table["table_path"] = dataset_table["table_path"].astype("category")
        self._dataset_table = dataset_table
        self.set_sampler(sampler)

    def _get_targets(self, f, error="raise"):
        if not Path(f).exists():
            if error == "raise":
                raise ValueError(f"File {f} does not existed.")
            else:
                return None
        try:
            import zarr
            from anndata import read_zarr

            tile_table = f"{f}/shapes/{self.tile_key}/shapes.parquet"
            tile_table = gpd.read_parquet(tile_table)

            tables = zarr.open(f"{f}/tables")
            available_keys = list(tables.keys())
            feature_key = self.feature_key
            if self.feature_key not in available_keys:
                feature_key = f"{self.feature_key}_{self.tile_key}"
            try:
                s = read_zarr(f"{f}/tables/{feature_key}")
                self.feature_key = feature_key
            except PathNotFoundError:
                raise KeyError(
                    f"Neither {feature_key} nor {self.feature_key} are found."
                )

            if self.target_key in tile_table:
                targets = tile_table[self.target_key].to_numpy()
            else:
                if self.target_key in s.obs:
                    targets = s.obs[self.target_key].to_numpy()
                else:
                    raise KeyError(
                        "Cannot find target key in either feature table or tile table."
                    )
            return targets
        except Exception as e:
            if error == "raise":
                raise e
            else:
                return None

    def _dataset_from_split(self, split_data):
        D = CachedFeaturesDataset if self.in_memory else IterableFeaturesDataset
        return D(
            split_data["table_path"].values,
            split_data["index"].values,
            split_data[self.target_key].values,
        )

    def split(self, val=0.15, test=0.15):
        s = self.sampler(
            self._dataset_table["table_path"].values,
            self._dataset_table[self.target_key].values,
            self.seed,
        )
        s.split(val_size=val, test_size=test, stratify=True)

        ds = {}
        train_data = self._dataset_table.iloc[s.train_data]
        ds["train"] = self._dataset_from_split(train_data)
        if test > 0:
            test_data = self._dataset_table.iloc[s.test_data]
            ds["test"] = self._dataset_from_split(test_data)
        if val > 0:
            val_data = self._dataset_table.iloc[s.val_data]
            ds["val"] = self._dataset_from_split(val_data)

        return ds

    def class_distribution(self, ax):
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        ax.pie(
            self._dataset_table[self.target_key].value_counts(),
        )
