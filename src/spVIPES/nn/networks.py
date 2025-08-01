from collections.abc import Iterable

import torch
import torch.nn.functional as F
from scvi.nn import FCLayers
from torch import nn
from torch.distributions import Normal

from .utils import one_hot


# Encoder without covariates
class Encoder(nn.Module):
    """
    Variational encoder network for spVIPES.
    
    This encoder maps input gene expression data to latent representations using
    a variational approach. It outputs both mean and variance parameters for the
    latent distribution, enabling sampling during training and inference.

    Parameters
    ----------
    n_input : int
        Number of input features (genes) in the expression data.
    n_topics : int  
        Number of output dimensions in the latent space (topics/factors).
    hidden : int, default=100
        Number of hidden units in the fully connected layers.
    dropout : float, default=0.1
        Dropout rate applied to hidden layers for regularization.
    n_cat_list : Iterable[int], optional
        List of categorical covariate dimensions. Each element represents
        the number of categories for a categorical covariate (e.g., batch).
    groups : str, optional
        Group identifier for this encoder instance.

    Notes
    -----
    The encoder uses a two-layer fully connected architecture with ReLU activations
    and batch normalization on the output layers. It outputs parameters for a
    normal distribution in latent space, following the variational autoencoder framework.
    
    The forward pass returns both the latent representation (theta) and intermediate
    statistics needed for the variational objective.
    """

    def __init__(
        self,
        n_input: int,  # n_in
        n_topics: int,  # n_out
        hidden: int = 100,
        dropout: float = 0.1,
        n_cat_list: Iterable[int] = None,
        groups: str = None,
    ):
        super().__init__()
        self.n_topics = n_topics
        self.groups = groups

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
            nn.Linear(hidden, n_topics, bias=True),
            nn.BatchNorm1d(n_topics),
        )

        # hidden 128 -> topics
        self.lvar_encoder = nn.Sequential(
            nn.Linear(hidden, n_topics, bias=True),
            nn.BatchNorm1d(n_topics),
        )

    def forward(self, data: torch.Tensor, specie: int, *cat_list: int):
        """
        Forward pass through the variational encoder.

        Parameters
        ----------
        data : torch.Tensor
            Input gene expression data with shape (batch_size, n_input).
        specie : int
            Species or group identifier (currently unused but kept for compatibility).
        *cat_list : int
            Variable length list of categorical covariate indices for each sample.

        Returns
        -------
        dict
            Dictionary containing encoder outputs:
            
            - **logtheta_loc** : torch.Tensor - Mean of latent distribution
            - **logtheta_logvar** : torch.Tensor - Log variance of latent distribution  
            - **logtheta_scale** : torch.Tensor - Standard deviation of latent distribution
            - **log_z** : torch.Tensor - Sampled latent variable (log space)
            - **theta** : torch.Tensor - Normalized latent representation (simplex)
            - **qz** : torch.distributions.Normal - Latent distribution object
        """
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
        logtheta_logvar = self.lvar_encoder(data)
        logtheta_scale = (0.5 * logtheta_logvar).exp()

        qz = Normal(logtheta_loc, logtheta_scale)
        log_z = qz.rsample().to(data.device)
        theta = F.softmax(log_z, -1)

        stats = {
            "logtheta_loc": logtheta_loc,
            "logtheta_logvar": logtheta_logvar,
            "logtheta_scale": logtheta_scale,
            "log_z": log_z,
            "theta": theta,
            "qz": qz,
        }

        return stats


class LinearDecoderSPVIPE(nn.Module):
    """
    Linear decoder for spVIPES with shared-private latent space decomposition.
    
    This decoder takes separate shared and private latent representations and
    decodes them into gene expression parameters. It implements a mixture model
    that combines shared and private contributions to generate the final output
    distribution parameters for the negative binomial likelihood.

    Parameters
    ----------
    n_input_private : int
        Dimensionality of the private latent space input.
    n_input_shared : int  
        Dimensionality of the shared latent space input.
    n_output : int
        Number of output features (genes) to reconstruct.
    n_cat_list : Iterable[int], optional
        List of categorical covariate dimensions for batch correction.
    use_batch_norm : bool, default=False
        Whether to use batch normalization in the decoder layers.
    use_layer_norm : bool, default=False
        Whether to use layer normalization in the decoder layers.
    bias : bool, default=False
        Whether to include bias terms in linear layers.
    n_hidden : int, default=256
        Number of hidden units in the mixing network.
    **kwargs
        Additional keyword arguments passed to FCLayers.

    Notes
    -----
    The decoder consists of three main components:
    
    1. **Private factor regressor**: Maps private latent space to gene-specific factors
    2. **Shared factor regressor**: Maps shared latent space to gene-specific factors  
    3. **Mixing network**: Learns how to combine shared and private contributions
    
    The output includes both separate private/shared reconstructions and a mixed
    reconstruction that combines both components according to learned mixing weights.
    """

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
        """
        Forward pass through the decoder network.

        Parameters
        ----------
        dispersion : str
            Dispersion parameter identifier (currently unused but kept for compatibility).
        z_private : torch.Tensor
            Private latent representation with shape (batch_size, n_input_private).
        z_shared : torch.Tensor
            Shared latent representation with shape (batch_size, n_input_shared).
        library : torch.Tensor
            Library size factors with shape (batch_size, 1) for scaling output rates.
        *cat_list : int
            Variable length list of categorical covariate indices.

        Returns
        -------
        tuple
            Tuple of decoder outputs:
            
            - **px_scale_private** : torch.Tensor - Normalized expression rates from private space
            - **px_scale_shared** : torch.Tensor - Normalized expression rates from shared space  
            - **px_rate_private** : torch.Tensor - Library-scaled rates from private space
            - **px_rate_shared** : torch.Tensor - Library-scaled rates from shared space
            - **px_mixing** : torch.Tensor - Learned mixing weights (logits)
            - **px_scale** : torch.Tensor - Final mixed expression rates
        """
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
