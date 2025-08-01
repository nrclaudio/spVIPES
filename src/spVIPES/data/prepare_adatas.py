from typing import Optional, Union

import anndata as ad
import numpy as np


def prepare_adatas(
    adatas: dict[str, ad.AnnData],
    layers: Optional[list[list[Union[str, None]]]] = None,
):
    """
    Prepare and concatenate multiple AnnData objects for spVIPES integration.

    This function takes multiple single-cell datasets and prepares them for 
    multi-group integration by concatenating them into a single AnnData object
    while preserving group-specific metadata. It sets up all the necessary
    data structures for spVIPES to perform shared-private latent space learning.

    Parameters
    ----------
    adatas : dict[str, AnnData]
        Dictionary mapping group names (strings) to their corresponding AnnData objects.
        Each AnnData contains single-cell expression data for one group/dataset.
        Currently supports exactly 2 groups.
    layers : list[list[str or None]], optional
        Specification of which layers to use from each AnnData object. Currently
        not implemented in the function body.

    Returns
    -------
    AnnData
        Concatenated AnnData object containing all groups with additional metadata:
        
        - **groups** : Added to `.obs` indicating which group each cell belongs to
        - **indices** : Added to `.obs` with within-group cell indices  
        - **groups_var_indices** : In `.uns`, indices of variables for each group
        - **groups_obs_indices** : In `.uns`, indices of observations for each group
        - **groups_obs_names** : In `.uns`, observation names for each group
        - **groups_obs** : In `.uns`, observation metadata for each group
        - **groups_lengths** : In `.uns`, number of features per group
        - **groups_var_names** : In `.uns`, variable names for each group
        - **groups_mapping** : In `.uns`, mapping from indices to group names

    Raises
    ------
    ValueError
        If more or fewer than 2 groups are provided (current limitation).

    Notes
    -----
    The function performs several important preprocessing steps:
    
    1. **Variable name prefixing**: Adds group prefixes to avoid name conflicts
    2. **Metadata harmonization**: Combines observation metadata across groups
    3. **Index tracking**: Creates mappings to track group-specific indices
    4. **Outer join concatenation**: Preserves all variables from all groups
    
    This prepared data structure enables spVIPES to handle datasets with different
    feature sets (genes) while maintaining the ability to separate shared and
    private latent representations.

    Examples
    --------
    Basic usage with two datasets:
    
    >>> import spVIPES
    >>> import scanpy as sc
    >>> 
    >>> # Load your datasets
    >>> adata1 = sc.read_h5ad("dataset1.h5ad")
    >>> adata2 = sc.read_h5ad("dataset2.h5ad")
    >>> 
    >>> # Prepare for spVIPES
    >>> adatas_dict = {"treatment": adata1, "control": adata2}
    >>> combined_adata = spVIPES.data.prepare_adatas(adatas_dict)
    >>> 
    >>> # Now ready for spVIPES setup
    >>> spVIPES.model.spVIPES.setup_anndata(combined_adata, groups_key="groups")

    Integration with different feature sets:
    
    >>> # Datasets can have different genes
    >>> print(f"Dataset 1: {adata1.n_vars} genes")
    >>> print(f"Dataset 2: {adata2.n_vars} genes") 
    >>> 
    >>> combined = spVIPES.data.prepare_adatas({"batch1": adata1, "batch2": adata2})
    >>> print(f"Combined: {combined.n_vars} genes")  # Union of all genes
    """
    groups_obs_names = []
    groups_obs = {}
    groups_lengths = {}
    groups_var_names = {}  # Changed to dictionary
    groups_mapping = {}
    if len(adatas) != 2:
        raise ValueError("Currently only 2 groups are supported")

    for i, (groups, adata) in enumerate(adatas.items()):
        if adata is not None:
            groups_lengths[i] = adata.shape[1]
            groups_obs_names.append(adata.obs_names)
            if groups_obs.get(groups, None) is None:
                groups_obs[groups] = adata.obs.copy()
                groups_obs[groups].loc[:, "group"] = groups

            else:
                cols_to_use = adata.obs.columns.difference(groups_obs[groups].columns)
                groups_obs[groups] = groups_obs[groups].join(adata.obs[cols_to_use])
            # Store var_names for each group
            adata.obs["groups"] = groups
            adata.var_names = f"{groups}_" + adata.var_names
            groups_var_names[groups] = adata.var_names
            groups_mapping[i] = groups

    multigroups_adata = ad.concat(adatas, join="outer", label="groups", index_unique="-")
    multigroups_adata.uns["groups_var_indices"] = [
        np.where(multigroups_adata.var_names.str.startswith(k))[0] for k in adatas.keys()
    ]
    multigroups_adata.uns["groups_obs_indices"] = [
        np.where(multigroups_adata.obs["groups"].str.startswith(k))[0] for k in adatas.keys()
    ]
    multigroups_adata.uns["groups_obs_names"] = groups_obs_names
    multigroups_adata.uns["groups_obs"] = groups_obs
    multigroups_adata.uns["groups_lengths"] = groups_lengths
    multigroups_adata.uns["groups_var_names"] = groups_var_names
    multigroups_adata.uns["groups_mapping"] = groups_mapping

    # Create indices column
    indices = []
    for _, group_indices in zip(adatas.keys(), multigroups_adata.uns["groups_obs_indices"]):
        group_size = len(group_indices)
        indices.extend(np.arange(group_size, dtype=np.int32))
    multigroups_adata.obs["indices"] = indices

    return multigroups_adata
