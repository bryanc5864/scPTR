"""Training loop for DeepPTR with KL warmup and early stopping."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from ._model import DeepPTR


@dataclass
class TrainHistory:
    """Stores per-epoch training metrics."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_recon: list[float] = field(default_factory=list)
    val_recon: list[float] = field(default_factory=list)
    train_kl: list[float] = field(default_factory=list)
    val_kl: list[float] = field(default_factory=list)
    kl_weight: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)


class Trainer:
    """Train a :class:`DeepPTR` model.

    Parameters
    ----------
    model
        A :class:`DeepPTR` instance.
    lr
        Initial learning rate for Adam.
    weight_decay
        L2 regularization.
    max_epochs
        Maximum training epochs.
    kl_warmup_epochs
        Number of epochs for linear KL annealing (0→1).
    patience
        Early-stopping patience (starts counting after warmup).
    max_grad_norm
        Gradient clipping threshold.
    device
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model: DeepPTR,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        max_epochs: int = 400,
        kl_warmup_epochs: int = 50,
        patience: int = 30,
        max_grad_norm: float = 5.0,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.max_epochs = max_epochs
        self.kl_warmup_epochs = kl_warmup_epochs
        self.patience = patience
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        self.history = TrainHistory()

    def _kl_weight(self, epoch: int) -> float:
        if self.kl_warmup_epochs <= 0:
            return 1.0
        return min(1.0, epoch / self.kl_warmup_epochs)

    def _run_epoch(
        self, loader: DataLoader, kl_w: float, train: bool = True
    ) -> tuple[float, float, float]:
        self.model.train(train)
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for s, u, l_s, l_u in loader:
                s = s.to(self.device)
                u = u.to(self.device)
                l_s = l_s.to(self.device)
                l_u = l_u.to(self.device)

                out = self.model(s, u, l_s, l_u, kl_weight=kl_w)
                loss = out["loss"]

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                total_loss += loss.item()
                total_recon += out["recon_loss"].item()
                total_kl += out["kl_loss"].item()
                n_batches += 1

        return (
            total_loss / max(n_batches, 1),
            total_recon / max(n_batches, 1),
            total_kl / max(n_batches, 1),
        )

    def fit(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        verbose: bool = True,
    ) -> TrainHistory:
        """Run the full training loop.

        Parameters
        ----------
        train_dl, val_dl
            Training and validation DataLoaders.
        verbose
            Print progress every 10 epochs.

        Returns
        -------
        TrainHistory
        """
        best_val = math.inf
        wait = 0
        best_state = None

        for epoch in range(1, self.max_epochs + 1):
            kl_w = self._kl_weight(epoch)

            tr_loss, tr_recon, tr_kl = self._run_epoch(train_dl, kl_w, train=True)
            vl_loss, vl_recon, vl_kl = self._run_epoch(val_dl, kl_w, train=False)

            self.scheduler.step(vl_loss)
            cur_lr = self.optimizer.param_groups[0]["lr"]

            self.history.train_loss.append(tr_loss)
            self.history.val_loss.append(vl_loss)
            self.history.train_recon.append(tr_recon)
            self.history.val_recon.append(vl_recon)
            self.history.train_kl.append(tr_kl)
            self.history.val_kl.append(vl_kl)
            self.history.kl_weight.append(kl_w)
            self.history.lr.append(cur_lr)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(
                    f"Epoch {epoch:4d} | "
                    f"train {tr_loss:.2f} (recon {tr_recon:.2f}, kl {tr_kl:.2f}) | "
                    f"val {vl_loss:.2f} | kl_w {kl_w:.3f} | lr {cur_lr:.1e}"
                )

            # Early stopping (only after warmup)
            if epoch >= self.kl_warmup_epochs:
                if vl_loss < best_val:
                    best_val = vl_loss
                    wait = 0
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    wait += 1
                    if wait >= self.patience:
                        if verbose:
                            print(
                                f"Early stopping at epoch {epoch} "
                                f"(best val={best_val:.2f})"
                            )
                        break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return self.history
