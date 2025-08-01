import logging
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from scvi import REGISTRY_KEYS
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.model.base import BaseModelClass
from scvi.utils import setup_anndata_dsp
from tqdm import tqdm

from spVIPES.data import AnnDataManager
from spVIPES.dataloaders._concat_dataloader import ConcatDataLoader
from spVIPES.model.base.training_mixin import MultiGroupTrainingMixin
from spVIPES.module.spVIPESmodule import spVIPESmodule

logger = logging.getLogger(__name__)


def process_transport_plan(transport_plan, adata, groups_key):
    """
    Process the transport plan using cluster labels to create a common set of clusters between datasets.

    Parameters
    ----------
    transport_plan : np.ndarray
        The original transport plan matrix (shape: cells1 x cells2).
    adata : AnnData
        The AnnData object containing the combined datasets.
    groups_key : str
        Key for grouping of cells in `adata.obs`.

    Returns
    -------
    processed_labels : np.ndarray
        Array of processed cluster labels for all cells.
    """
    transport_plan = np.nan_to_num(transport_plan, nan=0.0)

    # Extract the groups
    groups = adata.obs[groups_key].unique()
    cluster_labels = []

    def optimize_resolution(group_adata, group_transport_plan, other_group_size):
        resolutions = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        scores = []
        for res in tqdm(resolutions, desc="Optimizing resolution"):
            sc.tl.leiden(group_adata, resolution=res, key_added=f"leiden_{res}")
            cluster_transport = np.zeros((len(group_adata.obs[f"leiden_{res}"].unique()), other_group_size))
            for i, cluster in enumerate(group_adata.obs[f"leiden_{res}"].unique()):
                mask = group_adata.obs[f"leiden_{res}"] == cluster
                cluster_transport[i] = group_transport_plan[mask].sum(axis=0)

            # Normalize the cluster transport
            cluster_transport /= cluster_transport.sum(axis=1, keepdims=True)

            # Calculate the entropy of the transport distribution for each cluster
            cluster_entropies = entropy(cluster_transport, axis=1)

            # Use the negative mean entropy as the score (higher is better)
            scores.append(-np.mean(cluster_entropies))

        optimal_res = resolutions[np.argmax(scores)]
        return optimal_res

    optimal_resolutions = {}

    for i, group in enumerate(groups):
        group_mask = adata.obs[groups_key] == group
        group_adata = adata[group_mask].copy()
        # Filter out .var indices that don't correspond to this group
        group_var_names = adata.uns["groups_var_names"][group]
        group_adata = group_adata[:, group_adata.var_names.isin(group_var_names)].copy()

        # Normalize the data
        sc.pp.normalize_total(group_adata)
        sc.pp.log1p(group_adata)
        sc.pp.pca(group_adata)

        # Compute neighborhood graph
        sc.pp.neighbors(group_adata)

        # Optimize resolution
        other_group_size = adata[adata.obs[groups_key] != group].shape[0]
        group_transport_plan = transport_plan if i == 0 else transport_plan.T
        optimal_res = optimize_resolution(group_adata, group_transport_plan, other_group_size)
        optimal_resolutions[group] = optimal_res

        # Perform Leiden clustering with optimal resolution
        sc.tl.leiden(group_adata, resolution=optimal_res)
        group_clusters = group_adata.obs["leiden"].astype(str)
        group_clusters = group + "_" + group_clusters
        cluster_labels.extend(group_clusters)

    # Add the cluster labels to adata.obs
    adata.obs["group_cluster_labels"] = pd.Categorical(cluster_labels)

    # Create a DataFrame of transport values between clusters
    clusters1 = adata[adata.obs[groups_key] == groups[0]].obs["group_cluster_labels"]
    clusters2 = adata[adata.obs[groups_key] == groups[1]].obs["group_cluster_labels"]

    transport_df = pd.DataFrame(
        {
            "source_cluster": np.repeat(clusters1, len(clusters2)),
            "target_cluster": np.tile(clusters2, len(clusters1)),
            "transport_value": transport_plan.flatten(),
        }
    )

    # Create a pivot table of median transport values between clusters
    pivot_df = transport_df.pivot_table(
        values="transport_value", index="source_cluster", columns="target_cluster", aggfunc="median"
    )

    def rename_clusters(pivot_df):
        # Convert the pivot_df to a cost matrix (negative because we want to maximize)
        cost_matrix = -pivot_df.values

        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create the rename dictionary
        rename_dict = {}
        for i, (source_idx, target_idx) in enumerate(zip(row_ind, col_ind)):
            source_cluster = pivot_df.index[source_idx]
            target_cluster = pivot_df.columns[target_idx]
            new_name = f"Cluster_{i}"
            rename_dict[source_cluster] = new_name
            rename_dict[target_cluster] = new_name

        # Handle any unmatched clusters
        all_clusters = set(pivot_df.index) | set(pivot_df.columns)
        matched_clusters = set(rename_dict.keys())
        unmatched_clusters = all_clusters - matched_clusters

        for cluster in unmatched_clusters:
            new_name = f"Cluster_{len(rename_dict) // 2}"
            rename_dict[cluster] = new_name

        return rename_dict

    rename_dict = rename_clusters(pivot_df)

    # Apply renaming to the AnnData object
    adata.obs["processed_transport_labels"] = adata.obs["group_cluster_labels"].map(rename_dict)

    # Ensure the categories are in the correct order and format
    categories = np.array(sorted(set(rename_dict.values()), key=lambda x: int(x.split("_")[1])))
    adata.obs["processed_transport_labels"] = pd.Categorical(
        adata.obs["processed_transport_labels"], categories=categories, ordered=True
    )

    # Store the optimal resolutions in adata.uns
    adata.uns["optimal_resolutions"] = optimal_resolutions

    return adata.obs["processed_transport_labels"].values


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
    >>> spVIPES.model.setup_anndata(adata, groups_key="groups", transport_plan_key="transport_plan")
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

        n_batch = self.summary_stats.n_batch

        groups_lengths = adata.uns["groups_lengths"]
        groups_obs_names = adata.uns["groups_obs_names"]
        groups_var_names = adata.uns["groups_var_names"]
        groups_obs_indices = adata.uns["groups_obs_indices"]
        groups_var_indices = adata.uns["groups_var_indices"]

        setup_args = self.adata_manager._get_setup_method_args()["setup_args"]
        transport_plan_key = setup_args.get("transport_plan_key")

        if transport_plan_key:
            transport_plan_data = adata.uns.get(transport_plan_key)
            if transport_plan_data is None:
                raise ValueError(f"Transport plan not found in adata.uns['{transport_plan_key}']")
            transport_plan = torch.tensor(transport_plan_data, dtype=torch.float32)
        else:
            transport_plan = None

        pair_data = "processed_transport_labels" not in adata.obs.columns

        use_labels = "labels" in self.adata_manager.data_registry
        n_labels = self.summary_stats.n_labels if use_labels else None

        self.module = spVIPESmodule(
            groups_lengths=groups_lengths,
            groups_obs_names=groups_obs_names,
            groups_var_names=groups_var_names,
            groups_var_indices=groups_var_indices,
            groups_obs_indices=groups_obs_indices,
            transport_plan=transport_plan,
            pair_data=pair_data,
            use_labels=use_labels,
            n_labels=n_labels,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_dimensions_shared=n_dimensions_shared,
            n_dimensions_private=n_dimensions_private,
            dropout_rate=dropout_rate,
            **model_kwargs,
        )

        self._model_summary_string = (
            "spVIPES Model with the following params: \nn_hidden: {}, n_dimensions_shared: {}, n_dimensions_private: {}, dropout_rate: {}, transport_plan: {}"
        ).format(
            n_hidden,
            n_dimensions_shared,
            n_dimensions_private,
            dropout_rate,
            "Provided" if transport_plan is not None else "Not provided",
        )
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        groups_key: str,
        match_clusters: bool = False,
        transport_plan_key: Optional[str] = None,
        label_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        %(summary)s.

        Parameters
        ----------
        %(param_adata)s
        groups_key
            Key for grouping of cells in `adata.obs`.
        transport_plan_key
            Key for transport plan in `adata.uns`. Optional.
        label_key
            Key for cell labels in `adata.obs`. Optional.
        threshold
            Threshold for sparsifying the transport plan. Default is 1e-9.
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
            CategoricalObsField("groups", groups_key),
        ]

        print("=== spVIPES AnnData Setup ===")
        print(f"Setting up with groups_key: '{groups_key}'")

        transport_plan_configured = False
        labels_configured = False

        if transport_plan_key is not None:
            if transport_plan_key not in adata.uns:
                raise ValueError(f"Transport plan key '{transport_plan_key}' not found in adata.uns")
            adata.uns["transport_plan"] = adata.uns[transport_plan_key]
            transport_plan_configured = True

            print(f"✓ Transport plan: Using '{transport_plan_key}' from adata.uns")

            # Process the transport plan
            transport_plan = adata.uns[transport_plan_key]

            if match_clusters:
                print("✓ Cluster matching: Enabled - will create processed transport labels")
                # Process the transport plan using the cluster labels
                processed_labels = process_transport_plan(
                    transport_plan,
                    adata,
                    groups_key,
                )
                adata.obs["processed_transport_labels"] = pd.Categorical(processed_labels)
                anndata_fields.append(CategoricalObsField("processed_transport_labels", "processed_transport_labels"))
            else:
                print("✓ Cluster matching: Disabled - will use direct cell pairing")

            # Add indices field if using transport plan
            anndata_fields.append(CategoricalObsField("indices", "indices"))

            if "indices" not in adata.obs:
                raise ValueError("'indices' must be present in adata.obs when using a transport plan")

        if label_key is not None:
            labels_configured = True
            print(f"✓ Labels: Using '{label_key}' from adata.obs")
            anndata_fields.append(CategoricalObsField("labels", label_key))
            anndata_fields.append(CategoricalObsField("indices", "indices"))

        # Inform user about the PoE method that will be used
        print("\n--- Product of Experts (PoE) Configuration ---")
        if labels_configured and transport_plan_configured:
            print("🎯 Will use: Label-based PoE (labels take priority over transport plan)")
        elif labels_configured:
            print("🎯 Will use: Label-based PoE")
        elif transport_plan_configured:
            if match_clusters:
                print("🎯 Will use: Cluster-based PoE (transport plan with cluster matching)")
            else:
                print("🎯 Will use: Paired PoE (direct cell-to-cell transport plan)")
        else:
            print("⚠️  No transport plan or labels configured - you may need one for integration")

        print("=" * 45)

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
        drop_last: Optional[bool] = None,
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
        drop_last
            Whether to drop the last incomplete batch. If None, automatically determined based on
            whether using paired PoE (True for paired, False for others).

        Returns
        -------
        Low-dimensional topic for each cell.
        """
        adata = self._validate_anndata(adata)
        n_groups_1, n_groups_2 = (len(group) for group in group_indices_list)

        # Automatically determine drop_last based on PoE type if not specified
        if drop_last is None:
            # Prioritize label-based PoE (drop_last=False) when labels are available
            if self.module.use_labels and "labels" in self.adata_manager.data_registry:
                print("Using label-based PoE with drop_last=False")
                drop_last = False  # Label-based PoE can handle unequal batches
            elif self.module.use_transport_plan and self.module.pair_data:
                print("Using paired PoE with drop_last=False (will use special handling)")
                drop_last = False  # Use special handling to preserve all cells
            else:
                print("Using cluster-based PoE with drop_last=False")
                drop_last = False  # Cluster-based PoE can handle unequal batches
        else:
            print(f"User specified drop_last={drop_last}")

        print(f"Input cells: Group 1: {n_groups_1}, Group 2: {n_groups_2}")
        print(f"Using pair_data: {self.module.pair_data}")
        print(f"Using transport plan: {self.module.use_transport_plan}")
        print(f"Using labels: {self.module.use_labels}")

        # Determine which PoE will actually be used
        if self.module.use_labels and "labels" in self.adata_manager.data_registry:
            print("Will use: Label-based PoE")
        elif self.module.use_transport_plan:
            if self.module.pair_data:
                print("Will use: Paired PoE")
            else:
                print("Will use: Cluster-based PoE")

        # For paired PoE with drop_last=False, use cycling to handle unequal group sizes
        use_cycling = (
            self.module.use_transport_plan
            and self.module.pair_data
            and not drop_last
            and not (self.module.use_labels and "labels" in self.adata_manager.data_registry)
        )
        print(f"Use cycling approach: {use_cycling}")

        if use_cycling:
            print("Using cycling approach for paired PoE with drop_last=False")
            results = self._process_all_cells_with_cycling(
                group_indices_list, normalized, give_mean, mc_samples, batch_size
            )
            return self._format_results(results, n_groups_1, n_groups_2)

        # Standard processing
        scdl = ConcatDataLoader(
            self.adata_manager,
            indices_list=group_indices_list,
            shuffle=False,
            drop_last=drop_last,
            batch_size=batch_size,
        )

        results = self._process_batches(scdl, normalized, give_mean, mc_samples)
        final_results = self._format_results(results, n_groups_1, n_groups_2)

        return final_results

    def _process_batches(self, dataloader, normalized, give_mean, mc_samples):
        """Process batches and return intermediate results."""
        groups_1_latent_shared = []
        groups_2_latent_shared = []
        groups_1_latent = []
        groups_2_latent = []
        groups_1_original_indices = []
        groups_2_original_indices = []

        for tensors_by_group in dataloader:
            inference_inputs = self.module._get_inference_input(tensors_by_group)
            outputs = self.module.inference(**inference_inputs)
            _, _, _, poe_qz_groups_1, poe_log_z_groups_1, poe_theta_groups_1 = outputs["poe_stats"][0].values()
            _, _, _, poe_qz_groups_2, poe_log_z_groups_2, poe_theta_groups_2 = outputs["poe_stats"][1].values()

            if not normalized:
                groups_1_latent_shared += [poe_log_z_groups_1.cpu()]
                groups_2_latent_shared += [poe_log_z_groups_2.cpu()]

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

            groups_1_original_indices += [tensors_by_group[0]["indices"].cpu()]
            groups_2_original_indices += [tensors_by_group[1]["indices"].cpu()]

        return {
            "groups_1_latent_shared": groups_1_latent_shared,
            "groups_2_latent_shared": groups_2_latent_shared,
            "groups_1_latent": groups_1_latent,
            "groups_2_latent": groups_2_latent,
            "groups_1_original_indices": groups_1_original_indices,
            "groups_2_original_indices": groups_2_original_indices,
        }

    def _process_all_cells_with_cycling(self, group_indices_list, normalized, give_mean, mc_samples, batch_size):
        """Process all cells using cycling approach to handle unequal group sizes."""
        # Find minimum and maximum group sizes
        min_group_size = min(len(group_indices_list[0]), len(group_indices_list[1]))
        max_group_size = max(len(group_indices_list[0]), len(group_indices_list[1]))

        if min_group_size == 0:
            raise ValueError("One of the groups is empty")

        # Initialize results
        results = {
            "groups_1_latent_shared": [],
            "groups_2_latent_shared": [],
            "groups_1_latent": [],
            "groups_2_latent": [],
            "groups_1_original_indices": [],
            "groups_2_original_indices": [],
        }

        # Process all cells by cycling through in chunks of min_group_size
        for start_idx in range(0, max_group_size, min_group_size):
            # Get chunk indices, cycling through the smaller group as needed
            chunk_indices_1 = []
            chunk_indices_2 = []

            for i in range(min_group_size):
                # Use modulo to cycle through indices if one group is smaller
                idx1 = (start_idx + i) % len(group_indices_list[0])
                idx2 = (start_idx + i) % len(group_indices_list[1])
                chunk_indices_1.append(group_indices_list[0][idx1])
                chunk_indices_2.append(group_indices_list[1][idx2])

            # Create dataloader for this chunk
            chunk_scdl = ConcatDataLoader(
                self.adata_manager,
                indices_list=[chunk_indices_1, chunk_indices_2],
                shuffle=False,
                drop_last=False,
                batch_size=batch_size,
            )

            # Process this chunk
            chunk_results = self._process_batches(chunk_scdl, normalized, give_mean, mc_samples)

            # Add chunk results to overall results
            for key in results:
                results[key].extend(chunk_results[key])

        return results

    def _format_results(self, results, n_groups_1, n_groups_2):
        """Format the final results dictionary."""
        groups_2_original_indices = torch.cat(results["groups_2_original_indices"]).numpy().flatten()[:n_groups_2]

        groups_1_latent = torch.cat(results["groups_1_latent"]).numpy()[:n_groups_1]
        groups_2_latent = torch.cat(results["groups_2_latent"]).numpy()[:n_groups_2]
        groups_1_latent_shared = torch.cat(results["groups_1_latent_shared"]).numpy()[:n_groups_1]
        groups_2_latent_shared = torch.cat(results["groups_2_latent_shared"]).numpy()[:n_groups_2]

        latent_private = {0: groups_1_latent, 1: groups_2_latent}
        latent_shared = {0: groups_1_latent_shared, 1: groups_2_latent_shared}
        latent_private_reordered = {0: groups_1_latent, 1: groups_2_latent[np.argsort(groups_2_original_indices)]}
        latent_shared_reordered = {
            0: groups_1_latent_shared,
            1: groups_2_latent_shared[np.argsort(groups_2_original_indices)],
        }

        return {
            "shared": latent_shared,
            "private": latent_private,
            "shared_reordered": latent_shared_reordered,
            "private_reordered": latent_private_reordered,
        }

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
