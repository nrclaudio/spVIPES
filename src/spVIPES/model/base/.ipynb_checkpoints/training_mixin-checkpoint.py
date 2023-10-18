"""
Mixin classes for pre-coded features.
For more details on Mixin classes, see
https://docs.scvi-tools.org/en/0.9.0/user_guide/notebooks/model_user_guide.html#Mixing-in-pre-coded-features
"""


from typing import List, Optional, Union

import numpy as np
from scvi.train import TrainingPlan, TrainRunner

from spVIPES.data._multi_datasplitter import MultiGroupDataSplitter


class MultiGroupTrainingMixin:
    """General methods for multigroup learning."""

    def train(
        self,
        group_indices_list: List[List[int]],
        batch_size: Optional[int] = 128,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = 400,
        **trainer_kwargs,
    ) -> None:
        """
        Train a multigroup model.
        Args:
        ----
            group_indices_list: Indices corresponding to each group of samples.
            max_epochs: Number of passes through the dataset. If `None`, default to
                `np.min([round((20000 / n_cells) * 400), 400])`.
            use_gpu: Use default GPU if available (if `None` or `True`), or index of
                GPU to use (if `int`), or name of GPU (if `str`, e.g., `"cuda:0"`),
                or use CPU (if `False`).
            train_size: Size of training set in the range [0.0, 1.0].
            validation_size: Size of the validation set. If `None`, default to
                `1 - train_size`. If `train_size + validation_size < 1`, the remaining
                cells belong to the test set.
            batch_size: Mini-batch size to use during training.
            early_stopping: Perform early stopping. Additional arguments can be passed
                in `**kwargs`. See :class:`~scvi.train.Trainer` for further options.
            plan_kwargs: Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword
                arguments passed to `train()` will overwrite values present
                in `plan_kwargs`, when appropriate.
            **trainer_kwargs: Other keyword args for :class:`~scvi.train.Trainer`.
        Returns
        -------
            None. The model is trained.
        """

        # if batch_sizes is None:
        #     n_cells_per_group = [len(group) for group in group_indices_list]
        #     biggest_species = n_cells_per_group.index(max(n_cells_per_group)) #get index, species0 or species1
        #     n_iters = n_cells_per_group[biggest_species] / 128 #default to 128 batch size for biggest species
        #     batch_sizes = [math.floor(i/n_iters) for i in n_cells_per_group]
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400]).item()

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        update_dict = {
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        data_splitter = MultiGroupDataSplitter(
            self.adata_manager,
            group_indices_list=group_indices_list,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = TrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()
