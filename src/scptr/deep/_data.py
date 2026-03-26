"""AnnData to PyTorch DataLoader conversion for DeepPTR."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from anndata import AnnData


def setup_dataloaders(
    adata: AnnData,
    batch_size: int = 256,
    val_frac: float = 0.1,
    stratify_key: str | None = None,
    seed: int = 0,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Build train and validation DataLoaders from an AnnData object.

    Extracts raw integer counts from ``adata.layers['spliced']`` and
    ``adata.layers['unspliced']``, together with per-cell library sizes.

    Parameters
    ----------
    adata
        Annotated data matrix. Must contain ``layers['spliced']`` and
        ``layers['unspliced']``.
    batch_size
        Mini-batch size.
    val_frac
        Fraction of cells held out for validation.
    stratify_key
        Optional obs column for stratified splitting.
    seed
        Random seed for reproducibility.
    num_workers
        DataLoader workers.

    Returns
    -------
    train_dl, val_dl, train_idx, val_idx
    """
    from scipy.sparse import issparse

    from ._utils import get_library_sizes

    def _dense(mat):
        if issparse(mat):
            return np.asarray(mat.todense())
        return np.asarray(mat)

    s = _dense(adata.layers["spliced"]).astype(np.float32)
    u = _dense(adata.layers["unspliced"]).astype(np.float32)
    l_u, l_s = get_library_sizes(adata)

    n = adata.n_obs
    indices = np.arange(n)

    if stratify_key is not None and stratify_key in adata.obs.columns:
        from sklearn.model_selection import StratifiedShuffleSplit

        labels = adata.obs[stratify_key].values
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac, random_state=seed
        )
        train_idx, val_idx = next(splitter.split(indices, labels))
    else:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_frac))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

    def _make_loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(s[idx]),
            torch.from_numpy(u[idx]),
            torch.from_numpy(l_s[idx]),
            torch.from_numpy(l_u[idx]),
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

    train_dl = _make_loader(train_idx, shuffle=True)
    val_dl = _make_loader(val_idx, shuffle=False)
    return train_dl, val_dl, train_idx, val_idx
