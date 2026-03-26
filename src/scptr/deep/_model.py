"""Structured VAE with kinetic model decoder for DeepPTR.

The generative model factorises latent space into transcriptional (z_T) and
post-transcriptional (z_PT) factors.  The decoder uses RNA kinetic equations
to map these to expected unspliced / spliced counts observed through a
negative binomial likelihood.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._distributions import log_nb_positive
from ._utils import init_weights


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """Amortised inference network q(z_T, z_PT | x).

    Takes concatenated log1p(spliced, unspliced) and outputs mean and
    log-variance for two independent Gaussian posteriors.
    """

    def __init__(
        self,
        n_genes: int,
        d_hidden: int = 128,
        d_T: int = 10,
        d_PT: int = 10,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        n_input = 2 * n_genes  # [log1p(s); log1p(u)]

        layers: list[nn.Module] = []
        in_dim = n_input
        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, d_hidden),
                    nn.LayerNorm(d_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = d_hidden
        self.shared = nn.Sequential(*layers)

        # Heads for z_T
        self.mu_T = nn.Linear(d_hidden, d_T)
        self.logvar_T = nn.Linear(d_hidden, d_T)

        # Heads for z_PT
        self.mu_PT = nn.Linear(d_hidden, d_PT)
        self.logvar_PT = nn.Linear(d_hidden, d_PT)

        self.apply(init_weights)

    def forward(
        self, s: Tensor, u: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode observations to posterior parameters.

        Returns
        -------
        mu_T, logvar_T, mu_PT, logvar_PT : Tensor
            Each shape ``(N, d_*)``.
        """
        x = torch.cat([torch.log1p(s), torch.log1p(u)], dim=-1)
        h = self.shared(x)
        return (
            self.mu_T(h),
            self.logvar_T(h),
            self.mu_PT(h),
            self.logvar_PT(h),
        )


# ---------------------------------------------------------------------------
# Kinetic Decoder
# ---------------------------------------------------------------------------


class KineticDecoder(nn.Module):
    r"""Decoder mapping (z_T, z_PT) → expected counts via kinetic model.

    .. math::
        \alpha_g &= \text{softplus}(f_\alpha(z_T))_g \\
        \gamma_g &= \text{softplus}(f_\gamma(z_{PT}))_g \\
        \beta_g  &= \exp(\log\beta_g) \quad (\text{gene-specific parameter}) \\
        \mu^u_g  &= \frac{\alpha_g / \beta_g}{\sum_g \alpha_g / \beta_g}
                     \cdot l_u \\
        \mu^s_g  &= \frac{\alpha_g / \gamma_g}{\sum_g \alpha_g / \gamma_g}
                     \cdot l_s
    """

    def __init__(
        self,
        n_genes: int,
        d_T: int = 10,
        d_PT: int = 10,
        d_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes

        # alpha network: z_T → alpha (transcription rate)
        self.f_alpha = nn.Sequential(
            nn.Linear(d_T, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_genes),
        )

        # gamma network: z_PT → gamma (degradation rate)
        self.f_gamma = nn.Sequential(
            nn.Linear(d_PT, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_genes),
        )

        # Gene-specific splicing rate (not cell-specific)
        self.log_beta = nn.Parameter(torch.zeros(n_genes))

        # Inverse dispersion parameters (learnable, per-gene)
        self.log_theta_s = nn.Parameter(torch.zeros(n_genes))
        self.log_theta_u = nn.Parameter(torch.zeros(n_genes))

        self.apply(init_weights)
        # Re-init log_beta/theta after apply
        nn.init.zeros_(self.log_beta)
        nn.init.constant_(self.log_theta_s, 2.0)  # ~exp(2)≈7.4
        nn.init.constant_(self.log_theta_u, 2.0)

    def forward(
        self,
        z_T: Tensor,
        z_PT: Tensor,
        l_s: Tensor,
        l_u: Tensor,
    ) -> dict[str, Tensor]:
        """Decode latent variables to NB distribution parameters.

        Parameters
        ----------
        z_T : (N, d_T)
        z_PT : (N, d_PT)
        l_s : (N,)  library size for spliced
        l_u : (N,)  library size for unspliced

        Returns
        -------
        dict with keys: mu_s, mu_u, theta_s, theta_u, alpha, gamma, beta
        """
        alpha = F.softplus(self.f_alpha(z_T))  # (N, G), positive
        gamma = F.softplus(self.f_gamma(z_PT))  # (N, G), positive
        beta = self.log_beta.exp()  # (G,), positive

        eps = 1e-8

        # Expected proportions from kinetic model
        rho_u = alpha / (beta.unsqueeze(0) + eps)  # (N, G)
        rho_s = alpha / (gamma + eps)  # (N, G)

        # Normalize to proportions (sum to 1 across genes)
        rho_u = rho_u / (rho_u.sum(dim=-1, keepdim=True) + eps)
        rho_s = rho_s / (rho_s.sum(dim=-1, keepdim=True) + eps)

        # Scale by library size
        mu_u = rho_u * l_u.unsqueeze(-1)  # (N, G)
        mu_s = rho_s * l_s.unsqueeze(-1)  # (N, G)

        theta_s = self.log_theta_s.exp()
        theta_u = self.log_theta_u.exp()

        return {
            "mu_s": mu_s,
            "mu_u": mu_u,
            "theta_s": theta_s,
            "theta_u": theta_u,
            "alpha": alpha,
            "gamma": gamma,
            "beta": beta,
        }


# ---------------------------------------------------------------------------
# Full VAE
# ---------------------------------------------------------------------------


class DeepPTR(nn.Module):
    """Structured VAE for mRNA degradation rate estimation.

    Combines an amortised encoder with a kinetic-model-constrained decoder
    and a negative binomial observation model.
    """

    def __init__(
        self,
        n_genes: int,
        d_T: int = 10,
        d_PT: int = 10,
        d_hidden: int = 128,
        n_enc_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.d_T = d_T
        self.d_PT = d_PT

        self.encoder = Encoder(
            n_genes=n_genes,
            d_hidden=d_hidden,
            d_T=d_T,
            d_PT=d_PT,
            n_layers=n_enc_layers,
            dropout=dropout,
        )
        self.decoder = KineticDecoder(
            n_genes=n_genes,
            d_T=d_T,
            d_PT=d_PT,
            d_hidden=d_hidden,
        )

    # -- helpers --

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z = mu + eps * std with reparameterization trick."""
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    @staticmethod
    def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
        """KL(q(z) || N(0,I)), summed over latent dims, mean over batch."""
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

    # -- forward --

    def forward(
        self,
        s: Tensor,
        u: Tensor,
        l_s: Tensor,
        l_u: Tensor,
        kl_weight: float = 1.0,
    ) -> dict[str, Tensor]:
        """Full forward pass: encode → sample → decode → loss.

        Parameters
        ----------
        s : (N, G)  spliced counts
        u : (N, G)  unspliced counts
        l_s : (N,)  spliced library size
        l_u : (N,)  unspliced library size
        kl_weight : float  annealing coefficient for KL term

        Returns
        -------
        dict
            ``loss``, ``recon_loss``, ``kl_loss``, and decoder outputs.
        """
        mu_T, logvar_T, mu_PT, logvar_PT = self.encoder(s, u)

        z_T = self.reparameterize(mu_T, logvar_T)
        z_PT = self.reparameterize(mu_PT, logvar_PT)

        dec = self.decoder(z_T, z_PT, l_s, l_u)

        # Reconstruction: sum NB log-likelihood over genes, mean over batch
        ll_s = log_nb_positive(s, dec["mu_s"], dec["theta_s"]).sum(dim=-1).mean()
        ll_u = log_nb_positive(u, dec["mu_u"], dec["theta_u"]).sum(dim=-1).mean()
        recon_loss = -(ll_s + ll_u)

        kl_T = self.kl_divergence(mu_T, logvar_T)
        kl_PT = self.kl_divergence(mu_PT, logvar_PT)
        kl_loss = kl_T + kl_PT

        loss = recon_loss + kl_weight * kl_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "kl_T": kl_T,
            "kl_PT": kl_PT,
            "mu_T": mu_T,
            "logvar_T": logvar_T,
            "mu_PT": mu_PT,
            "logvar_PT": logvar_PT,
            **dec,
        }

    @torch.no_grad()
    def get_latent(
        self, s: Tensor, u: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return posterior means for z_T and z_PT (no sampling)."""
        mu_T, logvar_T, mu_PT, logvar_PT = self.encoder(s, u)
        return mu_T, logvar_T, mu_PT, logvar_PT
