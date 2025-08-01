from itertools import cycle
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from spVIPES.data import AnnDataManager
from spVIPES.dataloaders._ann_dataloader import AnnDataLoader


class ConcatDataLoader(DataLoader):
    """DataLoader that supports a list of list of indices to load.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    indices_list
        List where each element is a list of indices in the adata to load
    shuffle
        Whether the data should be shuffled
    use_labels
        Whether to use labels for sampling
    batch_size
        minibatch size to load each iteration
    data_and_attributes
        Dictionary with keys representing keys in data registry (``adata_manager.data_registry``)
        and value equal to desired numpy loading type (later made into torch tensor).
        If ``None``, defaults to all registered data.
    data_loader_kwargs
        Keyword arguments for :class:`~torch.utils.data.DataLoader`
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        indices_list: list[list[int]],
        shuffle: bool = True,
        use_labels: bool = False,
        batch_size: int = 128,
        data_and_attributes: Optional[dict] = None,
        drop_last: Union[bool, int] = False,
        **data_loader_kwargs,
    ):
        self.adata_manager = adata_manager
        self.groups_obs_indices = adata_manager.adata.uns['groups_obs_indices']
        self.dataloader_kwargs = data_loader_kwargs
        self.data_and_attributes = data_and_attributes
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._shuffle = shuffle

        # Extract transport plan
        transport_plan = adata_manager.adata.uns.get('transport_plan')
        
        self.dataloaders = []
        largest_species = max([len(indices) for indices in indices_list])
        for indices in indices_list:
            self.dataloaders.append(
                    AnnDataLoader(
                        adata_manager,
                        indices=indices,
                        shuffle=shuffle,
                        use_labels=use_labels,
                        batch_size=batch_size,
                        data_and_attributes=data_and_attributes,
                        drop_last=drop_last,
                        **self.dataloader_kwargs,
                    )
                )
    

        # if transport_plan is not None:
        #     # Use transport plan for pairing
        #     paired_indices = self._create_paired_indices(indices_list, transport_plan)
        #     for indices in paired_indices:
        #         self.dataloaders.append(
        #             AnnDataLoader(
        #                 adata_manager,
        #                 indices=indices,
        #                 shuffle=False,  # We don't shuffle here as the indices are already paired
        #                 use_labels=use_labels,
        #                 batch_size=batch_size,
        #                 data_and_attributes=data_and_attributes,
        #                 drop_last=drop_last,
        #                 **self.dataloader_kwargs,
        #             )
        #         )
        # else:
        #     # Original behavior when no transport plan is available
        #     largest_species = max([len(indices) for indices in indices_list])
        #     for indices in indices_list:
        #         self.dataloaders.append(
        #             AnnDataLoader(
        #                 adata_manager,
        #                 indices=indices,
        #                 shuffle=shuffle,
        #                 use_labels=use_labels,
        #                 batch_size=batch_size,
        #                 data_and_attributes=data_and_attributes,
        #                 drop_last=drop_last,
        #                 **self.dataloader_kwargs,
        #             )
        #         )

        lens = [len(dl) for dl in self.dataloaders]
        self.largest_dl = self.dataloaders[np.argmax(lens)]
        super().__init__(self.largest_dl, **data_loader_kwargs)

    def __len__(self):
        return len(self.largest_dl)

    def __iter__(self):
        iter_list = [cycle(dl) if dl != self.largest_dl else dl for dl in self.dataloaders]
        return zip(*iter_list)

    def _create_paired_indices(self, indices_list, transport_plan, top_k=5):
        dataset1_indices, dataset2_indices = indices_list
        paired_indices = [[], []]
        
        if len(dataset1_indices) <= len(dataset2_indices):
            smaller_indices, larger_indices = dataset1_indices, dataset2_indices
            transpose_plan = False
        else:
            smaller_indices, larger_indices = dataset2_indices, dataset1_indices
            transpose_plan = True
        
        if transpose_plan:
            transport_plan = transport_plan.T
        
        # Normalize transport plan
        transport_plan = transport_plan / transport_plan.sum(axis=1, keepdims=True)
        
        # First, ensure each cell from the smaller dataset is matched
        for i, idx_small in enumerate(smaller_indices):
            top_k_matches = np.argsort(transport_plan[i])[-top_k:][::-1]
            best_match = larger_indices[top_k_matches[0]]
            
            if transpose_plan:
                paired_indices[0].append(best_match)
                paired_indices[1].append(idx_small)
            else:
                paired_indices[0].append(idx_small)
                paired_indices[1].append(best_match)
        
        # Then, match the remaining cells from the larger dataset
        for i in range(len(smaller_indices), len(larger_indices)):
            idx_small = smaller_indices[i % len(smaller_indices)]
            top_k_matches = np.argsort(transport_plan[i % len(smaller_indices)])[-top_k:][::-1]
            best_match = larger_indices[top_k_matches[0]]
            
            if transpose_plan:
                paired_indices[0].append(best_match)
                paired_indices[1].append(idx_small)
            else:
                paired_indices[0].append(idx_small)
                paired_indices[1].append(best_match)
        
        return paired_indices