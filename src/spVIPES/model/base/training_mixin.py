"""
Mixin classes for pre-coded features.
For more details on Mixin classes, see
https://docs.scvi-tools.org/en/0.9.0/user_guide/notebooks/model_user_guide.html#Mixing-in-pre-coded-features
"""


from typing import Optional, Union

import numpy as np
from scvi.train import TrainingPlan, TrainRunner

from spVIPES.data._multi_datasplitter import MultiGroupDataSplitter


class MultiGroupTrainingMixin:
    """General methods for multigroup learning."""

    def train(
        self,
        group_indices_list: list[list[int]],
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
        Train a multigroup spVIPES model.

        This method trains the model using a custom data splitter that handles
        multiple groups of cells separately while maintaining the shared-private
        latent space learning objective.

        Parameters
        ----------
        group_indices_list : list[list[int]]
            List of indices corresponding to each group of samples. Each inner list
            contains the indices for cells belonging to that specific group.
        max_epochs : int, optional
            Number of passes through the dataset. If None, defaults to
            ``np.min([round((20000 / n_cells) * 400), 400])``.
        use_gpu : str, int, bool, optional
            GPU usage specification. Use default GPU if available (if None or True),
            or index of GPU to use (if int), or name of GPU (if str, e.g., "cuda:0"),
            or use CPU (if False).
        train_size : float, default=0.9
            Size of training set in the range [0.0, 1.0].
        validation_size : float, optional
            Size of the validation set. If None, defaults to ``1 - train_size``.
            If ``train_size + validation_size < 1``, the remaining cells belong
            to the test set.
        batch_size : int, default=128
            Mini-batch size to use during training.
        early_stopping : bool, default=False
            Whether to perform early stopping. Additional arguments can be passed
            in ``**trainer_kwargs``.
        plan_kwargs : dict, optional
            Keyword arguments for the training plan. Arguments passed to ``train()``
            will overwrite values present in ``plan_kwargs``, when appropriate.
        n_steps_kl_warmup : int, optional
            Number of training steps for KL warmup. Takes precedence over n_epochs_kl_warmup.
        n_epochs_kl_warmup : int, default=400
            Number of epochs for KL divergence warmup.
        **trainer_kwargs
            Additional keyword arguments for the trainer.

        Returns
        -------
        None
            The model is trained in-place.

        Notes
        -----
        This method uses a specialized MultiGroupDataSplitter that ensures proper
        handling of multiple cell groups during training, maintaining the integrity
        of the shared-private latent space learning.
        """
        # if batch_sizes is None:
        #     n_cells_per_group = [len(group) for group in group_indices_list]
        #     biggest_groups = n_cells_per_group.index(max(n_cells_per_group)) #get index, groups0 or groups1
        #     n_iters = n_cells_per_group[biggest_groups] / 128 #default to 128 batch size for biggest groups
        #     batch_sizes = [math.floor(i/n_iters) for i in n_cells_per_group]
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400]).item()

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}
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
