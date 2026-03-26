"""DeepPTR: Deep generative model for mRNA degradation rate estimation.

Provides a structured VAE with a kinetic-model decoder that disentangles
transcriptional and post-transcriptional latent factors, outputs calibrated
uncertainty via posterior sampling, and uses a negative binomial likelihood.

Quick start::

    model, history = scptr.deep.fit_deepptr(adata)
    # adata.layers["gamma"]     — posterior mean gamma
    # adata.layers["gamma_var"] — posterior variance
    # adata.obsm["X_z_T"]      — transcription latent
    # adata.obsm["X_z_PT"]     — post-transcription latent
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

    from ._model import DeepPTR
    from ._trainer import TrainHistory, Trainer


def fit_deepptr(
    adata: "AnnData",
    d_T: int = 10,
    d_PT: int = 10,
    d_hidden: int = 128,
    n_enc_layers: int = 3,
    dropout: float = 0.1,
    batch_size: int = 256,
    max_epochs: int = 400,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    kl_warmup_epochs: int = 50,
    patience: int = 30,
    max_grad_norm: float = 5.0,
    val_frac: float = 0.1,
    stratify_key: str | None = None,
    n_posterior_samples: int = 50,
    device: str | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> tuple["DeepPTR", "TrainHistory"]:
    """Fit a DeepPTR model and store results in *adata*.

    After fitting, the following are stored:

    * ``adata.layers["gamma"]`` — posterior mean degradation rate
    * ``adata.layers["gamma_var"]`` — posterior variance
    * ``adata.obsm["X_z_T"]`` — transcription latent (posterior mean)
    * ``adata.obsm["X_z_PT"]`` — post-transcription latent (posterior mean)

    Parameters
    ----------
    adata
        AnnData with ``layers['spliced']`` and ``layers['unspliced']``
        containing raw integer counts.
    d_T
        Dimension of transcription latent space.
    d_PT
        Dimension of post-transcription latent space.
    d_hidden
        Hidden layer width.
    n_enc_layers
        Number of encoder hidden layers.
    dropout
        Dropout rate in encoder.
    batch_size
        Mini-batch size for training.
    max_epochs
        Maximum number of training epochs.
    lr
        Learning rate.
    weight_decay
        L2 regularization.
    kl_warmup_epochs
        Epochs for linear KL annealing.
    patience
        Early-stopping patience (after warmup).
    max_grad_norm
        Gradient clipping threshold.
    val_frac
        Fraction held out for validation.
    stratify_key
        Optional obs column for stratified splitting.
    n_posterior_samples
        Number of MC samples for posterior gamma statistics.
    device
        ``"cuda"`` or ``"cpu"``.  Auto-detected if ``None``.
    seed
        Random seed.
    verbose
        Print training progress.

    Returns
    -------
    model, history
        The trained :class:`DeepPTR` model and :class:`TrainHistory`.
    """
    import torch

    from ._data import setup_dataloaders
    from ._guide import extract_latent, posterior_gamma
    from ._model import DeepPTR as _DeepPTR
    from ._trainer import Trainer
    from ._utils import beta_from_adata

    torch.manual_seed(seed)

    # --- Data ---
    train_dl, val_dl, train_idx, val_idx = setup_dataloaders(
        adata,
        batch_size=batch_size,
        val_frac=val_frac,
        stratify_key=stratify_key,
        seed=seed,
    )

    # --- Model ---
    n_genes = adata.n_vars
    model = _DeepPTR(
        n_genes=n_genes,
        d_T=d_T,
        d_PT=d_PT,
        d_hidden=d_hidden,
        n_enc_layers=n_enc_layers,
        dropout=dropout,
    )

    # Warm-start beta from analytical estimate
    log_beta_init = beta_from_adata(adata)
    model.decoder.log_beta.data.copy_(log_beta_init)

    # --- Train ---
    trainer = Trainer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        kl_warmup_epochs=kl_warmup_epochs,
        patience=patience,
        max_grad_norm=max_grad_norm,
        device=device,
    )
    history = trainer.fit(train_dl, val_dl, verbose=verbose)

    # --- Posterior extraction ---
    if verbose:
        print("Extracting posterior gamma (MC sampling)...")
    gamma_mean, gamma_var = posterior_gamma(
        model, adata, n_samples=n_posterior_samples, device=device
    )
    z_T, z_PT = extract_latent(model, adata, device=device)

    adata.layers["gamma"] = gamma_mean
    adata.layers["gamma_var"] = gamma_var
    adata.obsm["X_z_T"] = z_T
    adata.obsm["X_z_PT"] = z_PT

    # Log parameters
    if "scptr" not in adata.uns:
        adata.uns["scptr"] = {}
    adata.uns["scptr"]["fit_deepptr"] = {
        "d_T": d_T,
        "d_PT": d_PT,
        "d_hidden": d_hidden,
        "n_enc_layers": n_enc_layers,
        "max_epochs": max_epochs,
        "kl_warmup_epochs": kl_warmup_epochs,
        "patience": patience,
        "n_posterior_samples": n_posterior_samples,
        "n_epochs_trained": len(history.train_loss),
        "final_train_loss": history.train_loss[-1],
        "final_val_loss": history.val_loss[-1],
    }

    return model, history
