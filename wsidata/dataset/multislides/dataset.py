from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset, IterableDataset


class BaseFeaturesDataset:
    """Base class for feature datasets with common functionality"""

    def __init__(
        self,
        table_path: Sequence[str],
        feature_index: Sequence[int],
        targets: Sequence[str],
        targets_mapping: Mapping[str, int] = None,
    ):
        self.table_path = np.asarray(table_path)
        self.feature_index = np.asarray(feature_index)
        self.targets = np.asarray(targets)

        if targets_mapping is None:
            self.unique_targets = np.unique(targets)
            self.targets_mapping = {t: ix for ix, t in enumerate(self.unique_targets)}
        else:
            self.unique_targets = list(targets_mapping.keys())
            self.targets_mapping = targets_mapping
        self.targets_codes = np.array([self.targets_mapping[t] for t in self.targets])

        # The length of three should be the same
        len_table_path = len(table_path)
        len_index = len(feature_index)
        len_targets = len(targets)
        assert len_table_path == len_index == len_targets, (
            "The length of table_path, feature_index and targets must match"
        )

    def _get_remap_ixs(self):
        remap_ixs = {}
        df = pd.DataFrame(
            {
                "table_path": self.table_path,
                "feature_index": self.feature_index,
            }
        )
        for tp, gdf in df.groupby("table_path", observed=True):
            remap_ixs[tp] = {fix: ix for ix, fix in enumerate(gdf["feature_index"])}
        return remap_ixs

    @property
    def target_weights(self):
        c = Counter(self.targets)
        n_targets = list(c.keys())
        n_targets.sort()
        return [c[i] for i in n_targets]


class CachedFeaturesDataset(BaseFeaturesDataset, Dataset):
    """
    A torch dataset that read extracted features from multiple slides.

    Parameters
    ----------
    datasource : pd.DataFrame

        A dataframe that contains at least three columns
            1. 'table_path', path to anndata zarr store
            2. 'index', the index of that feature in anndata
            3. target_key, the target variable
    """

    def __init__(
        self,
        table_path: Sequence[str],
        feature_index: Sequence[int],
        targets: Sequence[str],
        targets_mapping: Mapping[str, int] = None,
    ):
        super().__init__(table_path, feature_index, targets, targets_mapping)
        self._remap_ixs = self._get_remap_ixs()
        # Preload all features into memory during initialization
        self._features = self._load_features()

    def _get_feature(self, table_path: str, index: int):
        """Get feature with caching"""
        real_ix = self._remap_ixs[table_path][index]
        return self._features[table_path][real_ix]

    def _load_features(self):
        """Preload all features into memory"""
        features = {}

        # Group by table_path to minimize zarr store openings
        path_ixs = {}
        for i, (path, idx) in enumerate(zip(self.table_path, self.feature_index)):
            path_ixs.setdefault(path, []).append(idx)

        def _read_feature(table_path: str, idx: int):
            store = zarr.open(f"{table_path}/X", mode="r")
            return store[idx]

        with ThreadPoolExecutor() as executor:
            tasks = {}
            for table_path, ixs in path_ixs.items():
                tasks[table_path] = executor.submit(_read_feature, table_path, ixs)
            for table_path, task in tasks.items():
                features[table_path] = task.result()
        return features

    def memory_usage(self):
        """Compute the memory usage in preloaded features"""
        if self._features is None:
            return 0
        n_bytes = []
        for table_path, features in self._features.items():
            n_bytes.append(features.nbytes)
        return np.sum(n_bytes)

    def __len__(self):
        return len(self.table_path)

    def __getitem__(self, index):
        target = self.targets_codes[index]
        table_path = self.table_path[index]
        feature_idx = self.feature_index[index]

        feature = self._get_feature(table_path, feature_idx)

        return {
            "feature": torch.from_numpy(feature),
            "target": torch.tensor(target, dtype=torch.long),
        }

    @property
    def X(self):  # noqa
        X = []
        for i in range(len(self)):
            table_path = self.table_path[i]
            feature_idx = self.feature_index[i]
            feature = self._get_feature(table_path, feature_idx)
            X.append(feature)

        return np.stack(X)

    @property
    def y(self):
        return self.targets_codes.copy()


class IterableFeaturesDataset(BaseFeaturesDataset, IterableDataset):
    """
    An iterable torch dataset that reads extracted features from multiple slides.

    The dataset splits data by zarr storage so each worker holds complete zarr stores
    that don't belong to other workers, making it suitable for distributed training.

    Parameters
    ----------
    table_path : Sequence[str]
        Paths to zarr stores
    feature_index : Sequence[int]
        Feature indices within each zarr store
    targets : Sequence[str]
        Target labels for each sample
    targets_mapping : Mapping[str, int], optional
        Mapping from target labels to codes
    """

    def __init__(
        self,
        table_path: Sequence[str],
        feature_index: Sequence[int],
        targets: Sequence[str],
        targets_mapping: Mapping[str, int] = None,
    ):
        super().__init__(table_path, feature_index, targets, targets_mapping)

        # Group data by zarr storage for worker distribution
        self._storage_groups = self._group_by_storage()
        # Cache for zarr stores
        self._zarr_cache = {}

    def _get_zarr_store(self, table_path: str):
        """Cached zarr store access"""
        if table_path not in self._zarr_cache:
            self._zarr_cache[table_path] = zarr.open(f"{table_path}/X", mode="r")
        return self._zarr_cache[table_path]

    def _group_by_storage(self):
        """Group samples by their zarr storage paths"""
        storage_groups = {}
        for i, (path, idx, target) in enumerate(
            zip(self.table_path, self.feature_index, self.targets)
        ):
            if path not in storage_groups:
                storage_groups[path] = {
                    "indices": [],
                    "feature_indices": [],
                    "targets": [],
                    "target_codes": [],
                }
            storage_groups[path]["indices"].append(i)
            storage_groups[path]["feature_indices"].append(idx)
            storage_groups[path]["targets"].append(target)
            storage_groups[path]["target_codes"].append(self.targets_codes[i])

        return storage_groups

    def _get_worker_storages(self):
        """Distribute zarr storages among workers to avoid sharing"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single process - return all storages
            return list(self._storage_groups.keys())

        # Distribute storages among workers
        storage_paths = list(self._storage_groups.keys())
        num_workers = worker_info.num_workers
        worker_id = worker_info.id

        # Each worker gets a subset of storage paths
        worker_storages = []
        for i, storage_path in enumerate(storage_paths):
            if i % num_workers == worker_id:
                worker_storages.append(storage_path)

        return worker_storages

    def __iter__(self):
        """Iterate over samples assigned to this worker"""
        worker_storages = self._get_worker_storages()

        # Yield all samples from this worker's assigned storages
        for storage_path in worker_storages:
            storage_data = self._storage_groups[storage_path]

            for idx, feature_idx, target_code in zip(
                storage_data["indices"],
                storage_data["feature_indices"],
                storage_data["target_codes"],
            ):
                # Load feature from zarr store
                store = self._get_zarr_store(storage_path)
                feature = store[feature_idx]

                yield {
                    "feature": torch.from_numpy(feature),
                    "target": torch.tensor(target_code, dtype=torch.long),
                }

    def __len__(self):
        """Return the total number of samples across all workers"""
        return len(self.table_path)

    def clear_cache(self):
        """Clear zarr store cache to free memory"""
        self._zarr_cache.clear()
