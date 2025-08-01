from typing import Optional, Union

import anndata as ad
import numpy as np


def prepare_adatas(
    adatas: dict[str, ad.AnnData],
    layers: Optional[list[list[Union[str, None]]]] = None,
):
    """Concatenate all the input anndata objects.
    :param adatas:
        Dict of AnnData objects
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