from collections.abc import Iterable

import torch
import torch.nn.functional as F
from scvi.nn import FCLayers
from torch import nn
from torch.distributions import Normal

from .utils import one_hot


# Encoder without covariates
class Encoder(nn.Module):
    """Encoder for spVIPES"""

    def __init__(
        self,
        n_input: int,  # n_in
        n_topics_shared: int,  # n_out
        n_topics_private: int,
        hidden: int = 100,
        dropout: float = 0.1,
        n_cat_list: Iterable[int] = None,
        species: str = None,
    ):
        super().__init__()
        self.n_topics_shared = n_topics_shared
        self.n_topics_private = n_topics_private
        self.species = species

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)
        # input -> hidden 128
        self.fc1 = nn.Linear(n_input + cat_dim * True, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # hidden 128 -> topics
        self.mu_encoder = nn.Sequential(
            nn.Linear(hidden, n_topics_shared + n_topics_private, bias=True),
            nn.BatchNorm1d(n_topics_shared + n_topics_private),
        )

        # hidden 128 -> topics
        self.lvar_encoder = nn.Sequential(
            nn.Linear(hidden, n_topics_shared + n_topics_private, bias=True),
            nn.BatchNorm1d(n_topics_shared + n_topics_private),
        )

    def forward(self, data: torch.Tensor, specie: int, *cat_list: int):
        """Forward pass."""
        one_hot_cat_list = []
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat > 1:  # only proceed if there's more than one batch, if no batch key is specified this value == 1.
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        data = torch.cat((data, *one_hot_cat_list), dim=-1)
        data = self.relu(self.fc1(data))
        data = self.relu(self.fc2(data))
        data = self.drop(data)

        logtheta_loc = self.mu_encoder(data)
        logtheta_loc_shared = logtheta_loc[:, : self.n_topics_shared]
        logtheta_loc_private = logtheta_loc[:, self.n_topics_shared : self.n_topics_private + self.n_topics_shared]
        logtheta_logvar = self.lvar_encoder(data)
        logtheta_logvar_shared = logtheta_logvar[:, : self.n_topics_shared]
        logtheta_logvar_private = logtheta_logvar[
            :, self.n_topics_shared : self.n_topics_private + self.n_topics_shared
        ]
        logtheta_scale_shared = (0.5 * logtheta_logvar_shared).exp()  # Enforces positivity
        logtheta_scale_private = (0.5 * logtheta_logvar_private).exp()  # Enforces positivity
        logtheta_scale = (0.5 * logtheta_logvar).exp()

        qz_private = Normal(logtheta_loc_private, logtheta_scale_private)
        qz_shared = Normal(logtheta_loc_shared, logtheta_scale_shared)
        qz_private_shared = Normal(logtheta_loc, logtheta_scale)

        log_z_shared = qz_shared.rsample().to(data.device)
        log_z_private = qz_private.rsample().to(data.device)
        log_z_private_shared = qz_private_shared.rsample().to(data.device)
        # we sample from normal but by applying softmax we go from logz to z. We are originally sampling the logarithm of the random variable, so we transform it here
        theta_shared = F.softmax(log_z_shared, -1)
        theta_private = F.softmax(log_z_private, -1)
        theta_private_shared = F.softmax(log_z_private_shared, -1)

        private_stats = {
            "logtheta_loc": logtheta_loc_private,
            "logtheta_logvar": logtheta_logvar_private,
            "logtheta_scale": logtheta_scale_private,
            "log_z": log_z_private,
            "theta": theta_private,
            "qz": qz_private,
        }
        shared_stats = {
            "logtheta_loc": logtheta_loc_shared,
            "logtheta_logvar": logtheta_logvar_shared,
            "logtheta_scale": logtheta_scale_shared,
            "log_z": log_z_shared,
            "theta": theta_shared,
            "qz": qz_shared,
        }

        private_shared_stats = {
            "logtheta_loc": logtheta_loc,
            "logtheta_logvar": logtheta_logvar,
            "logtheta_scale": logtheta_scale,
            "log_z": log_z_private_shared,
            "theta": theta_private_shared,
            "qz": qz_private_shared,
        }
        return {"private": private_stats, "shared": shared_stats, "ps": private_shared_stats}


class LinearDecoderSPVIPE(nn.Module):
    """Linear decoder for scVI."""

    def __init__(
        self,
        n_input_private: int,
        n_input_shared: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        bias: bool = False,
        n_hidden: int = 256,
        **kwargs,
    ):
        super().__init__()

        # mean gamma private
        self.factor_regressor_private = FCLayers(
            n_in=n_input_private,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=False,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0,
            **kwargs,
        )

        # mean gamma shared
        self.factor_regressor_shared = FCLayers(
            n_in=n_input_shared,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=False,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0,
            **kwargs,
        )

        # self.factor_regressor_private_shared = FCLayers(
        #     n_in=n_input_private + n_input_shared,
        #     n_out=n_output,
        #     n_cat_list=n_cat_list,
        #     n_layers=1,
        #     use_activation=False,
        #     use_batch_norm=use_batch_norm,
        #     use_layer_norm=use_layer_norm,
        #     bias=bias,
        #     dropout_rate=0,
        #     **kwargs,
        # )

        # mixture component for private-shared contribution in MixtureNB

        self.sigmoid_decoder = FCLayers(
            n_in=n_input_shared + n_input_private,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=0,
            use_batch_norm=True,
            use_layer_norm=False,
        )

        self.mixture = FCLayers(
            n_in=n_hidden + n_input_shared + n_input_private,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=False,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=0,
        )

    def forward(
        self, dispersion: str, z_private: torch.Tensor, z_shared: torch.Tensor, library: torch.Tensor, *cat_list: int
    ):
        """Forward pass."""
        #         # The decoder returns values for the parameters of the ZINB distribution
        #         raw_px_scale_private = self.factor_regressor_private(z_private, *cat_list)
        #         px_scale_private = torch.softmax(raw_px_scale_private, dim=-1)
        #         raw_px_scale_shared = self.factor_regressor_shared(z_shared, *cat_list)
        #         px_scale_shared = torch.softmax(raw_px_scale_shared, dim=-1)

        #         # z_private_shared = torch.cat([z_private, z_shared], dim=1)
        #         # p_mixing = self.sigmoid_decoder(z_private_shared, *cat_list)
        #         # p_mixing_cat_z = torch.cat([p_mixing, z_private_shared], dim=-1)
        #         # px_mixing = self.mixture(p_mixing_cat_z, *cat_list)

        #         px_rate_private = torch.exp(library) * px_scale_private
        #         px_rate_shared = torch.exp(library) * px_scale_shared

        #         px_r = None

        #        return px_scale_private, px_scale_shared, px_r, px_rate_private, px_rate_shared,
        # px_mixing

        raw_px_scale_private = self.factor_regressor_private(z_private, *cat_list)
        px_scale_private = torch.softmax(raw_px_scale_private, dim=-1)
        px_rate_private = torch.exp(library) * px_scale_private

        raw_px_scale_shared = self.factor_regressor_shared(z_shared, *cat_list)
        px_scale_shared = torch.softmax(raw_px_scale_shared, dim=-1)
        px_rate_shared = torch.exp(library) * px_scale_shared

        z_private_shared = torch.cat([z_private, z_shared], dim=1)
        p_mixing = self.sigmoid_decoder(z_private_shared, *cat_list)
        p_mixing_cat_z = torch.cat([p_mixing, z_private_shared], dim=-1)
        px_mixing = self.mixture(p_mixing_cat_z, *cat_list)

        mixing = 1 / (1 + torch.exp(-px_mixing))
        px_scale = torch.nn.functional.normalize((1 - mixing) * px_rate_shared, p=1, dim=-1)

        # raw_px_scale_private_shared = self.factor_regressor_private_shared(z_private_shared, *cat_list)
        # px_scale_private_shared = torch.softmax(raw_px_scale_private_shared, dim=-1)
        # px_rate_private_shared = torch.exp(library) * px_scale_private_shared

        # return px_scale_private_shared, px_rate_private_shared
        return px_scale_private, px_scale_shared, px_rate_private, px_rate_shared, px_mixing, px_scale
