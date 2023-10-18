import logging
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.model.base import BaseModelClass
from scvi.utils import setup_anndata_dsp

from spVIPES.data import AnnDataManager
from spVIPES.dataloaders._concat_dataloader import ConcatDataLoader
from spVIPES.model.base.training_mixin import MultiGroupTrainingMixin

# from spVIPES.dataloader import DataSplitter
from spVIPES.module.spVIPESmodule import spVIPESmodule

logger = logging.getLogger(__name__)


class spVIPES(MultiGroupTrainingMixin, BaseModelClass):
    """
    Implementation of the spVIPES model
    ----------
    adata
        AnnData object that has been registered via :math:`~mypackage.MyModel.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_topics
        Number of topics to infer. Dimensionality of the latent space.
    **model_kwargs
        Keyword args for :class:`~mypackage.MyModule`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> mypackage.MyModel.setup_anndata(adata, batch_key="batch")
    >>> vae = mypackage.MyModel(adata)
    >>> vae.train()
    >>> adata.obsm["X_mymodel"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_topics_shared: int = 10,
        n_topics_private: int = 10,
        dropout_rate: float = 0.1,
        **model_kwargs,
    ):
        super().__init__(adata)
        self.adata = adata
        self.n_topics_private = n_topics_private
        self.n_topics_shared = n_topics_shared

        # self.summary_stats provides information about anndata dimensions and other tensor info
        n_batch = self.summary_stats.n_batch

        species_lengths = adata.uns["species_lengths"]
        species_obs_names = adata.uns["species_obs_names"]
        species_var_names = adata.uns["species_var_names"]
        species_obs_indices = adata.uns["species_obs_indices"]
        species_var_indices = adata.uns["species_var_indices"]

        ## pass this information to the module
        self.module = spVIPESmodule(
            species_lengths=species_lengths,
            species_obs_names=species_obs_names,
            species_var_names=species_var_names,
            species_var_indices=species_var_indices,
            species_obs_indices=species_obs_indices,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_topics_shared=n_topics_shared,
            n_topics_private=n_topics_private,
            **model_kwargs,
        )

        self._model_summary_string = (
            "spVIPES Model with the following params: \nn_hidden: {}, n_topics_shared: {}, n_topics_private: {}, dropout_rate: {}"
        ).format(n_hidden, n_topics_shared, n_topics_private, dropout_rate)
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        species_key: str,
        labels_key: str,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        **kwargs,
    ) -> Optional[AnnData]:
        """
        %(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_batch_key)s
        %(param_layer)s

        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(
                "species", species_key
            ),  # CAN'T UPDATE REGISTRY KEYS WITH NEW COLUMN, AS IT IS A NAMED TUPLE DEFINED IN _constants.py
            CategoricalObsField("labels", labels_key),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_latent_representation(
        self,
        group_indices_list: list[list[int]],
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        normalized: bool = False,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return the latent representation for each cell.
        ######This is denoted as :math:`z_n` in our manuscripts.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        normalized
            Whether to return the normalized cell embedding (softmaxed) or not
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        Low-dimensional topic for each cell.
        """
        adata = self._validate_anndata(adata)
        n_species_1, n_species_2 = (len(group) for group in group_indices_list)
        # group_indices_list = [l.tolist() for l in group_indices_list]
        scdl = ConcatDataLoader(
            self.adata_manager,
            indices_list=group_indices_list,
            shuffle=False,
            weighted=False,
            drop_last=False,
            batch_size=batch_size,
        )
        species_1_latent_shared = []
        species_2_latent_shared = []
        species_1_latent = []
        species_2_latent = []
        for tensors_by_group in scdl:
            inference_inputs = self.module._get_inference_input(tensors_by_group)
            outputs = self.module.inference(**inference_inputs)
            _, _, _, poe_qz_species_1, poe_log_z_species_1, poe_theta_species_1 = outputs["poe_stats"][0].values()
            _, _, _, poe_qz_species_2, poe_log_z_species_2, poe_theta_species_2 = outputs["poe_stats"][1].values()

            if not normalized:
                species_1_latent_shared += [poe_log_z_species_1.cpu()]
                species_2_latent_shared += [poe_log_z_species_2.cpu()]
            # else:
            #     if give_mean:
            #         samples = poe_qz.sample([mc_samples])
            #         theta = torch.nn.functional.softmax(samples, dim=-1)
            #         theta = theta.mean(dim=0)
            #     latent_shared += [theta.cpu()]

            _, _, _, species_1_private_log_z, species_1_private_theta, species_1_private_qz = outputs["private_stats"][
                0
            ].values()
            _, _, _, species_2_private_log_z, species_2_private_theta, species_2_private_qz = outputs["private_stats"][
                1
            ].values()
            if not normalized:
                species_1_latent += [species_1_private_log_z.cpu()]
                species_2_latent += [species_2_private_log_z.cpu()]
            else:
                if give_mean:
                    species_1_samples = species_1_private_qz.sample([mc_samples])
                    species_2_samples = species_2_private_qz.sample([mc_samples])
                    theta_species_1 = torch.nn.functional.softmax(species_1_samples, dim=-1)
                    theta_species_1 = theta_species_1.mean(dim=0)
                    theta_species_2 = torch.nn.functional.softmax(species_2_samples, dim=-1)
                    theta_species_2 = theta_species_2.mean(dim=0)
                species_1_latent += [theta_species_1.cpu()]
                species_2_latent += [theta_species_2.cpu()]
        species_1_latent = torch.cat(species_1_latent).numpy()
        species_2_latent = torch.cat(species_2_latent).numpy()
        species_1_latent_shared = torch.cat(species_1_latent_shared).numpy()
        species_2_latent_shared = torch.cat(species_2_latent_shared).numpy()

        latent_private = {0: species_1_latent[:n_species_1], 1: species_2_latent[:n_species_2]}
        latent_shared = {0: species_1_latent_shared[:n_species_1], 1: species_2_latent_shared[:n_species_2]}

        return {"shared": latent_shared, "private": latent_private}
        # return {'species_1': latent[0], 'species_2': latent[1]}

    def get_loadings(self) -> dict:
        """Extract per-gene weights in the linear decoder.

        Shape is genes by `n_latent`.

        """
        num_datasets = len(self.module.input_dims)
        datasets_obs_indices = self.module.species_obs_indices
        datasets_var_indices = self.module.species_var_indices
        adata = self.adata
        loadings_dict = {}
        for i in range(num_datasets):
            dataset_obs_indices = datasets_obs_indices[i]
            s_adata = adata[dataset_obs_indices, :].copy()
            cols_private = [f"Z_private_{n}" for n in range(self.module.n_topics_private)]
            cols_shared = [f"Z_shared_{n}" for n in range(self.module.n_topics_shared)]
            var_names = s_adata[:, datasets_var_indices[i]].var_names
            loadings_private = pd.DataFrame(
                self.module.get_loadings(i, "private"), index=var_names, columns=cols_private
            )
            loadings_shared = pd.DataFrame(self.module.get_loadings(i, "shared"), index=var_names, columns=cols_shared)

            loadings_dict[(i, "private")] = loadings_private
            loadings_dict[(i, "shared")] = loadings_shared

        return loadings_dict
