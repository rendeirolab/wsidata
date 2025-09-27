import warnings
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd

from wsidata._utils import find_stack_level


def split_integer(N, proportions):
    # Normalize proportions so they sum to 1
    proportions = np.array(proportions, dtype=float)
    proportions = proportions / proportions.sum()
    # Initial allocation by flooring
    raw = proportions * N
    result = np.floor(raw).astype(int)
    # Distribute the remainder (due to flooring)
    remainder = N - result.sum()
    # Find indices with largest fractional parts
    frac = raw - result
    idx = np.argsort(-frac)  # descending order
    # Add 1 to the top `remainder` indices
    result[idx[:remainder]] += 1
    return result


class BaseTileDatasetSampler(ABC):
    registry = {}

    def __init_subclass__(cls, key=None, **kwargs):
        super().__init_subclass__(**kwargs)

        if key is not None:
            if key in cls.registry:
                raise KeyError(f"key {key} already exists in registry.")
            BaseTileDatasetSampler.registry[key] = cls

    def __init__(
        self,
        slides: Sequence[str] = None,
        target: Sequence[str] = None,
        seed=0,
    ):
        self.data = pd.DataFrame(
            {
                "slide": pd.Categorical(slides),
                "target": pd.Categorical(target),
                "index": np.arange(len(slides)),
            }
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._train_data = None
        self._val_data = None
        self._test_data = None

    @abstractmethod
    def balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply balancing to a dataframe and return a balanced dataframe."""
        pass

    @property
    def train_data(self) -> np.ndarray:
        if self._train_data is None:
            raise ValueError("Train data not available, please run split() first.")
        return np.asarray(self._train_data)

    @property
    def test_data(self) -> np.ndarray:
        if self._test_data is None:
            raise ValueError("Test data not available, please run split() first.")
        return np.asarray(self._test_data)

    @property
    def val_data(self) -> np.ndarray:
        if self._val_data is None:
            raise ValueError("Validation data not available, please run split() first.")
        return np.asarray(self._val_data)

    def _stratified_split(self, train_size, val_size, test_size):
        splits = {"train": set()}
        keys = ["train"]
        if val_size > 0:
            splits["val"] = set()
            keys.append("val")
        if test_size > 0:
            splits["test"] = set()
            keys.append("test")
        slides = sorted(self.data["slide"].unique())
        # This part ensures that minority class is presented in train, val and test when possible
        # Get minority class
        minor_cls = self.data["target"].value_counts().sort_index().index[0]
        # Count the number of minority labels in each slide
        self.data.groupby("slide", observed=True).value_counts(subset=["target"])
        # Get slides that contain at least one minority class
        candidate_slides = sorted(
            list(
                self.data[self.data["target"] == minor_cls]
                .groupby("slide", observed=True)
                .size()
                .index
            )
        )

        self.rng.shuffle(candidate_slides)
        for (
            s,
            k,
        ) in zip(candidate_slides, keys):
            splits[k].add(s)

        n_candidates = len(candidate_slides)
        if n_candidates <= 2:
            warnings.warn(
                f"The minority class {minor_cls} appears in only "
                f"{n_candidates} slides → not present in test or val.",
                stacklevel=find_stack_level(),
            )

        # Fill the remaining, according to proportions
        remaining = sorted(
            list(
                set(slides)
                - splits["train"]
                - splits.get("val", set())
                - splits.get("test", set())
            )
        )
        self.rng.shuffle(remaining)
        # Calculate the remaining number that needs to be filled
        n_total = len(slides)
        n_train, n_val, n_test = split_integer(
            n_total, [train_size, val_size, test_size]
        )

        n_train -= len(splits["train"])
        splits["train"].update(remaining[:n_train])

        if val_size > 0:
            n_val -= len(splits["val"])
            splits["val"].update(remaining[n_train : n_train + n_val])
        if test_size > 0:
            n_test -= len(splits["test"])
            splits["test"].update(remaining[n_train + n_val :])

        return splits

    def _random_split(self, train_size, val_size, test_size):
        slides = sorted(list(self.data["slide"].unique()))
        self.rng.shuffle(slides)

        n_train, n_val, n_test = split_integer(
            len(slides), [train_size, val_size, test_size]
        )
        train_slides = slides[:n_train]
        val_slides = slides[n_train : n_train + n_val]
        test_slides = slides[n_train + n_val :]

        splits = {"train": set(train_slides)}
        if len(val_slides) > 0:
            splits["val"] = set(val_slides)
        if len(test_slides) > 0:
            splits["test"] = set(test_slides)

        return splits

    def split(self, val_size=0.15, test_size=0.15, stratify=False):
        train_size = 1 - val_size - test_size
        if train_size <= 0:
            raise ValueError("The validation size and test size are too large")

        if stratify:
            splits = self._stratified_split(train_size, val_size, test_size)
        else:
            splits = self._random_split(train_size, val_size, test_size)

        df_train = self.data[self.data["slide"].isin(splits.get("train", set()))]
        df_val = self.data[self.data["slide"].isin(splits.get("val", set()))]
        df_test = self.data[self.data["slide"].isin(splits.get("test", set()))]

        # Apply balancing
        df_train = self.balance(df_train)
        # Only balance non-empty dataframes
        if len(df_val) > 0:
            df_val = self.balance(df_val)
        if len(df_test) > 0:
            df_test = self.balance(df_test)  # test set is usually kept as-is

        self._train_data = df_train["index"].values
        self._val_data = (
            df_val["index"].values if len(df_val) > 0 else np.array([], dtype=int)
        )
        self._test_data = (
            df_test["index"].values if len(df_test) > 0 else np.array([], dtype=int)
        )


class NoBalance(BaseTileDatasetSampler, key="no-balance"):
    def balance(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class RandomUnderSampler(BaseTileDatasetSampler, key="undersample"):
    def balance(self, df: pd.DataFrame) -> pd.DataFrame:
        min_class_size = df["target"].value_counts().min()
        dfs = []
        for label, group in df.groupby("target", observed=True):
            dfs.append(group.sample(n=min_class_size, random_state=self.rng))
        return (
            pd.concat(dfs).sample(frac=1, random_state=self.rng).reset_index(drop=True)
        )


class InSlideUnderSampler(BaseTileDatasetSampler, key="slide-undersample"):
    def balance(self, df: pd.DataFrame) -> pd.DataFrame:
        dfs = []
        for slide, slide_df in df.groupby("slide", observed=True):
            min_class_size = slide_df["target"].value_counts().min()
            for label, group in slide_df.groupby("target", observed=True):
                dfs.append(group.sample(n=min_class_size, random_state=self.rng))
        return (
            pd.concat(dfs).sample(frac=1, random_state=self.rng).reset_index(drop=True)
        )
