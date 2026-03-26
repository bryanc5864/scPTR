"""Tests for DeepPTR training loop."""

import numpy as np
import pytest
import torch

from scptr.deep._model import DeepPTR
from scptr.deep._trainer import Trainer, TrainHistory
from scptr.deep._data import setup_dataloaders


@pytest.fixture
def tiny_adata():
    """Minimal AnnData for training tests."""
    from anndata import AnnData

    rng = np.random.RandomState(0)
    n, g = 80, 30
    s = rng.poisson(5, size=(n, g)).astype(np.float32)
    u = rng.poisson(2, size=(n, g)).astype(np.float32)
    adata = AnnData(X=s)
    adata.layers["spliced"] = s
    adata.layers["unspliced"] = u
    return adata


@pytest.fixture
def tiny_loaders(tiny_adata):
    train_dl, val_dl, _, _ = setup_dataloaders(
        tiny_adata, batch_size=32, val_frac=0.2, seed=0
    )
    return train_dl, val_dl


class TestTrainer:
    def test_fit_runs(self, tiny_loaders):
        train_dl, val_dl = tiny_loaders
        model = DeepPTR(n_genes=30, d_T=3, d_PT=3, d_hidden=16, n_enc_layers=1)
        trainer = Trainer(
            model=model,
            max_epochs=5,
            kl_warmup_epochs=2,
            patience=100,
            device="cpu",
        )
        history = trainer.fit(train_dl, val_dl, verbose=False)
        assert isinstance(history, TrainHistory)
        assert len(history.train_loss) == 5
        assert len(history.val_loss) == 5

    def test_loss_decreases(self, tiny_loaders):
        train_dl, val_dl = tiny_loaders
        model = DeepPTR(n_genes=30, d_T=3, d_PT=3, d_hidden=16, n_enc_layers=1)
        trainer = Trainer(
            model=model,
            max_epochs=30,
            kl_warmup_epochs=5,
            patience=100,
            device="cpu",
        )
        history = trainer.fit(train_dl, val_dl, verbose=False)
        # Training loss should decrease from start to end
        assert history.train_loss[-1] < history.train_loss[0]

    def test_early_stopping(self, tiny_loaders):
        train_dl, val_dl = tiny_loaders
        model = DeepPTR(n_genes=30, d_T=3, d_PT=3, d_hidden=16, n_enc_layers=1)
        trainer = Trainer(
            model=model,
            max_epochs=500,
            kl_warmup_epochs=2,
            patience=3,
            device="cpu",
        )
        history = trainer.fit(train_dl, val_dl, verbose=False)
        # Should stop before max_epochs
        assert len(history.train_loss) < 500

    def test_kl_warmup(self, tiny_loaders):
        train_dl, val_dl = tiny_loaders
        model = DeepPTR(n_genes=30, d_T=3, d_PT=3, d_hidden=16, n_enc_layers=1)
        trainer = Trainer(
            model=model,
            max_epochs=10,
            kl_warmup_epochs=5,
            patience=100,
            device="cpu",
        )
        history = trainer.fit(train_dl, val_dl, verbose=False)
        # KL weight should ramp up
        assert history.kl_weight[0] < history.kl_weight[-1]
        assert history.kl_weight[0] < 1.0

    def test_history_fields(self, tiny_loaders):
        train_dl, val_dl = tiny_loaders
        model = DeepPTR(n_genes=30, d_T=3, d_PT=3, d_hidden=16, n_enc_layers=1)
        trainer = Trainer(
            model=model, max_epochs=3, kl_warmup_epochs=1, patience=100, device="cpu"
        )
        history = trainer.fit(train_dl, val_dl, verbose=False)
        for attr in ("train_loss", "val_loss", "train_recon", "val_recon",
                      "train_kl", "val_kl", "kl_weight", "lr"):
            assert len(getattr(history, attr)) == 3
