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
    n_dimensions_shared
        Dimensionality of the shared latent space.
    n_dimensions_private
        Dimensionalites of the private latent spaces.
    **model_kwargs
        Keyword args for :class:`~mypackage.MyModule`

    Examples
    --------
    >>> adata = spVIPES.data.prepare_adatas({"dataset1": dataset1, "dataset2": dataset2})
    >>> spVIPES.model.setup_anndata(adata, groups_key="groups", label_key="labels")
    >>> spvipes = spVIPES.model.spVIPES(adata)
    >>> group_indices_list = [np.where(adata.obs['groups'] == group)[0] for group in adata.obs['groups'].unique()]
    >>> spvipes.train(group_indices_list)
    >>> latents = spvipes.get_latent_representation(group_indices_list)

    Notes
    -----
    We recommend n_dimensions_private < n_dimensions_shared
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_dimensions_shared: int = 25,
        n_dimensions_private: int = 10,
        dropout_rate: float = 0.1,
        **model_kwargs,
    ):
        super().__init__(adata)
        self.adata = adata
        self.n_dimensions_private = n_dimensions_private
        self.n_dimensions_shared = n_dimensions_shared

        # self.summary_stats provides information about anndata dimensions and other tensor info
        n_batch = self.summary_stats.n_batch

        groups_lengths = adata.uns["groups_lengths"]
        groups_obs_names = adata.uns["groups_obs_names"]
        groups_var_names = adata.uns["groups_var_names"]
        groups_obs_indices = adata.uns["groups_obs_indices"]
        groups_var_indices = adata.uns["groups_var_indices"]

        ## pass this information to the module
        self.module = spVIPESmodule(
            groups_lengths=groups_lengths,
            groups_obs_names=groups_obs_names,
            groups_var_names=groups_var_names,
            groups_var_indices=groups_var_indices,
            groups_obs_indices=groups_obs_indices,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_dimensions_shared=n_dimensions_shared,
            n_dimensions_private=n_dimensions_private,
            **model_kwargs,
        )

        self._model_summary_string = (
            "spVIPES Model with the following params: \nn_hidden: {}, n_dimensions_shared: {}, n_dimensions_private: {}, dropout_rate: {}"
        ).format(n_hidden, n_dimensions_shared, n_dimensions_private, dropout_rate)
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        groups_key: str,
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
                "groups", groups_key
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

        Parameters
        ----------
        group_indices_list
            List of lists containing the indices of cells in each of the groups used as input for spVIPES.
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
        n_groups_1, n_groups_2 = (len(group) for group in group_indices_list)
        # group_indices_list = [l.tolist() for l in group_indices_list]
        scdl = ConcatDataLoader(
            self.adata_manager,
            indices_list=group_indices_list,
            shuffle=False,
            weighted=False,
            drop_last=False,
            batch_size=batch_size,
        )
        groups_1_latent_shared = []
        groups_2_latent_shared = []
        groups_1_latent = []
        groups_2_latent = []
        for tensors_by_group in scdl:
            inference_inputs = self.module._get_inference_input(tensors_by_group)
            outputs = self.module.inference(**inference_inputs)
            _, _, _, poe_qz_groups_1, poe_log_z_groups_1, poe_theta_groups_1 = outputs["poe_stats"][0].values()
            _, _, _, poe_qz_groups_2, poe_log_z_groups_2, poe_theta_groups_2 = outputs["poe_stats"][1].values()

            if not normalized:
                groups_1_latent_shared += [poe_log_z_groups_1.cpu()]
                groups_2_latent_shared += [poe_log_z_groups_2.cpu()]
            # else:
            #     if give_mean:
            #         samples = poe_qz.sample([mc_samples])
            #         theta = torch.nn.functional.softmax(samples, dim=-1)
            #         theta = theta.mean(dim=0)
            #     latent_shared += [theta.cpu()]

            _, _, _, groups_1_private_log_z, groups_1_private_theta, groups_1_private_qz = outputs["private_stats"][
                0
            ].values()
            _, _, _, groups_2_private_log_z, groups_2_private_theta, groups_2_private_qz = outputs["private_stats"][
                1
            ].values()
            if not normalized:
                groups_1_latent += [groups_1_private_log_z.cpu()]
                groups_2_latent += [groups_2_private_log_z.cpu()]
            else:
                if give_mean:
                    groups_1_samples = groups_1_private_qz.sample([mc_samples])
                    groups_2_samples = groups_2_private_qz.sample([mc_samples])
                    theta_groups_1 = torch.nn.functional.softmax(groups_1_samples, dim=-1)
                    theta_groups_1 = theta_groups_1.mean(dim=0)
                    theta_groups_2 = torch.nn.functional.softmax(groups_2_samples, dim=-1)
                    theta_groups_2 = theta_groups_2.mean(dim=0)
                groups_1_latent += [theta_groups_1.cpu()]
                groups_2_latent += [theta_groups_2.cpu()]
        groups_1_latent = torch.cat(groups_1_latent).numpy()
        groups_2_latent = torch.cat(groups_2_latent).numpy()
        groups_1_latent_shared = torch.cat(groups_1_latent_shared).numpy()
        groups_2_latent_shared = torch.cat(groups_2_latent_shared).numpy()

        latent_private = {0: groups_1_latent[:n_groups_1], 1: groups_2_latent[:n_groups_2]}
        latent_shared = {0: groups_1_latent_shared[:n_groups_1], 1: groups_2_latent_shared[:n_groups_2]}

        return {"shared": latent_shared, "private": latent_private}
        # return {'groups_1': latent[0], 'groups_2': latent[1]}

    def get_loadings(self) -> dict:
        """Extract per-gene weights in the linear decoder.

        Shape is genes by `n_latent`.

        """
        num_datasets = len(self.module.input_dims)
        datasets_obs_indices = self.module.groups_obs_indices
        datasets_var_indices = self.module.groups_var_indices
        adata = self.adata
        loadings_dict = {}
        for i in range(num_datasets):
            dataset_obs_indices = datasets_obs_indices[i]
            s_adata = adata[dataset_obs_indices, :].copy()
            cols_private = [f"Z_private_{n}" for n in range(self.module.n_dimensions_private)]
            cols_shared = [f"Z_shared_{n}" for n in range(self.module.n_dimensions_shared)]
            var_names = s_adata[:, datasets_var_indices[i]].var_names
            loadings_private = pd.DataFrame(
                self.module.get_loadings(i, "private"), index=var_names, columns=cols_private
            )
            loadings_shared = pd.DataFrame(self.module.get_loadings(i, "shared"), index=var_names, columns=cols_shared)

            loadings_dict[(i, "private")] = loadings_private
            loadings_dict[(i, "shared")] = loadings_shared

        return loadings_dict
