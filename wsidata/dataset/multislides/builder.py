from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Type

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    from zarr.errors import PathNotFoundError as FeatureTableNotFoundError
except ImportError:
    from zarr.errors import GroupNotFoundError as FeatureTableNotFoundError

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
    targets_mapping: dict[str, int] | None
        Optional explicit mapping from class label to integer code to use across
        train/val/test splits. If not provided, a deterministic mapping will be
        derived from the observed labels (sorted by label name).

    """

    sampler: Type[BaseTileDatasetSampler]

    def __init__(
        self,
        stores,  # Can be wsi path, store path
        splits=None,
        tile_key=None,
        feature_key=None,
        target_key=None,
        target_transform=None,
        skip_class=None,
        sampler: str = "undersample",
        n_per_class=None,
        in_memory=True,
        seed=0,
        targets_mapping=None,
    ):
        self.tile_key = tile_key
        self.feature_key = feature_key
        self.target_key = target_key
        self.target_transform = target_transform
        self.n_per_class = n_per_class
        self.in_memory = in_memory
        self.seed = seed
        self.recent_splits = None
        self._targets_mapping = targets_mapping

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
        # Ensure categorical dtype for targets
        dataset_table[target_key] = dataset_table[self.target_key].astype("category")
        # Build or validate targets mapping (label -> int) and keep it consistent
        if self._targets_mapping is None:
            cats = list(dataset_table[target_key].cat.categories)
            # Deterministic order by category name to avoid randomness
            cats = sorted(cats)
            self._targets_mapping = {t: i for i, t in enumerate(cats)}
            # Reorder categories to follow mapping order
            dataset_table[target_key] = dataset_table[target_key].cat.set_categories(
                list(self._targets_mapping.keys())
            )
        else:
            # Validate provided mapping covers all labels
            missing = set(dataset_table[target_key].unique()) - set(
                self._targets_mapping.keys()
            )
            if len(missing) > 0:
                raise ValueError(
                    f"targets_mapping is missing labels: {sorted(missing)}"
                )
            # Reorder categories to follow mapping keys order for consistency
            ordered_cats = list(self._targets_mapping.keys())
            dataset_table[target_key] = dataset_table[target_key].cat.set_categories(
                ordered_cats
            )
        dataset_table["table_path"] = dataset_table["table_path"].astype("category")
        self._dataset_table = dataset_table
        self.set_sampler(sampler)

        if splits is not None:
            ss = pd.unique(splits)
            assert all([s in {"train", "val", "test"} for s in ss]), (
                "Only train/val/test splits are supported."
            )
            new_splits = {
                "train": [],
                "val": [],
                "test": [],
            }
            for store, split in zip(stores, splits):
                new_splits[split].append(f"{store}/tables/{self.feature_key}")
            self.preset_splits = new_splits
        else:
            self.preset_splits = None

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
            except FeatureTableNotFoundError:
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
            targets_mapping=self._targets_mapping,
        )

    def split(self, val=0.15, test=0.15):
        s = self.sampler(
            self._dataset_table["table_path"].values,
            self._dataset_table[self.target_key].values,
            self.seed,
        )
        splits = s.split(
            val_size=val,
            test_size=test,
            stratify=True,
            preset_splits=self.preset_splits,
        )
        self.recent_splits = splits

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

    def class_distribution(self, ax=None):
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        ax.pie(
            self._dataset_table[self.target_key].value_counts(),
        )

    @property
    def targets_mapping(self):
        """Mapping from class label to integer code used in all datasets."""
        return dict(self._targets_mapping)

    @property
    def slides_splits(self):
        if self.preset_splits is not None:
            return self.preset_splits
        else:
            return self.recent_splits
