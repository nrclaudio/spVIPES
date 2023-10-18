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
    # TOOD: add checks for layers

    # needed for scArches operation setup
    species_obs_names = []
    species_obs = {}
    species_lengths = {}
    species_var_names = []  # we store for filtering out other species' var_names later
    species_mapping = {}
    if len(adatas) != 2:
        raise ValueError("Currently only 2 species are supported")

    for i, (species, adata) in enumerate(adatas.items()):
        if adata is not None:
            species_lengths[i] = adata.shape[1]
            species_obs_names.append(adata.obs_names)
            if species_obs.get(species, None) is None:
                species_obs[species] = adata.obs
                species_obs[species].loc[:, "group"] = species
            else:
                cols_to_use = adata.obs.columns.difference(species_obs[species].columns)
                species_obs[species] = species_obs[species].join(adata.obs[cols_to_use])

            species_var_names.append(adata.var_names)
            adata.obs["species"] = species
            adata.var_names = f"{species}_" + adata.var_names
            species_mapping[i] = species

    multispecies_adata = ad.concat(adatas, join="outer", label="species", index_unique="-")
    multispecies_adata.uns["species_var_indices"] = [
        np.where(multispecies_adata.var_names.str.startswith(k))[0] for k in adatas.keys()
    ]
    multispecies_adata.uns["species_obs_indices"] = [
        np.where(multispecies_adata.obs["species"].str.startswith(k))[0] for k in adatas.keys()
    ]
    multispecies_adata.uns["species_obs_names"] = species_obs_names
    multispecies_adata.uns["species_obs"] = species_obs
    multispecies_adata.uns["species_lengths"] = species_lengths
    multispecies_adata.uns["species_var_names"] = species_var_names
    multispecies_adata.uns["species_mapping"] = species_mapping

    return multispecies_adata
