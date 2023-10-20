"""Main module."""
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomialMixture
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from spVIPES.nn.networks import Encoder, LinearDecoderSPVIPE

torch.backends.cudnn.benchmark = True


class spVIPESmodule(BaseModuleClass):
    """
    Pytorch Implementation of Product of Experts LDA with batch-correction.
    We construct the model around a basic version of scVI's underlying VAE.

    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer
    n_dimensions
        Number of dimensions in the latent space.
    dropout_rate
        Dropout rate for neural networks
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    """

    def __init__(
        self,
        groups_lengths,
        groups_obs_names,
        groups_var_names,
        groups_obs_indices,
        groups_var_indices,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_dimensions_shared: int = 25,
        n_dimensions_private: int = 10,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        log_variational_inference: bool = True,
        log_variational_generative: bool = True,
        dispersion: Literal["gene", "gene-batch", "gene-cell"] = "gene",
    ):
        """Variational auto-encoder model.

        This is an implementation of the scVI model described in :cite:p:`Lopez18`.

        Parameters
        ----------
        n_batch
            Number of batches, if 0, no batch correction is performed.
        n_hidden
            Number of nodes per hidden layer
        n_dimensions_shared
            Dimensionalities of the private spaces
        n_dimensions_private
            Dimensionality of the shared space
        dropout_rate
            Dropout rate for neural networks
        log_variational_inference
            Log(data+1) prior to encoding for numerical stability. Not normalization.
        log_variational_generative
            Log(data+1) prior to reconstruction for numerical stability. Not normalization.
        use_batch_norm
            Whether to use batch norm in layers.
        use_layer_norm
            Whether to use layer norm in layers.
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        """
        super().__init__()
        self.n_dimensions_shared = n_dimensions_shared
        self.n_dimensions_private = n_dimensions_private
        self.n_batch = n_batch
        self.px_r = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(length)) for length in groups_lengths.values()]
        )
        self.input_dims = groups_lengths
        self.groups_barcodes = groups_obs_names
        self.groups_genes = groups_var_names
        self.groups_obs_indices = groups_obs_indices
        self.groups_var_indices = groups_var_indices
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.label_per_batch = []
        self.dispersion = dispersion
        self.log_variational_inference = log_variational_inference
        self.log_variational_generative = log_variational_generative
        # cat_list includes both batch ids and cat_covs
        cat_list = [n_batch] if n_batch > 0 else None
        self.encoders = {
            groups: Encoder(
                x_dim,
                n_dimensions_shared,
                n_dimensions_private,
                hidden=n_hidden,
                dropout=dropout_rate,
                n_cat_list=cat_list,
                groups=groups,
            )
            for groups, x_dim in self.input_dims.items()
        }

        # n_input_decoder = n_dimensions_shared + n_dimensions_private
        self.decoders = {
            groups: LinearDecoderSPVIPE(
                n_dimensions_private,
                n_dimensions_shared,
                x_dim,
                # hidden=n_hidden,
                n_cat_list=cat_list,
                use_batch_norm=True,
                use_layer_norm=False,
                bias=False,
            )
            for groups, x_dim in self.input_dims.items()
        }

        # register sub-modules
        for (groups, values_encoder), (_, values_decoder) in zip(self.encoders.items(), self.decoders.items()):
            self.add_module(f"encoder_{groups}", values_encoder)
            self.add_module(f"decoder_{groups}", values_decoder)

    def _supervised_poe(self, shared_stats: dict, label_group: dict):
        groups_1_stats, groups_2_stats = shared_stats.values()
        groups_1_stats = {
            k: groups_1_stats[k] for k in ["logtheta_loc", "logtheta_logvar", "logtheta_scale"] if k in groups_1_stats
        }
        groups_2_stats = {
            k: groups_2_stats[k] for k in ["logtheta_loc", "logtheta_logvar", "logtheta_scale"] if k in groups_2_stats
        }

        groups_1_labels, groups_2_labels = label_group.values()

        groups_1_labels_list = groups_1_labels.flatten().tolist()  # to keep track of order
        groups_2_labels_list = groups_2_labels.flatten().tolist()  # to keep track of order
        # Convert label tensors to sets
        set1 = set(groups_1_labels_list)
        set2 = set(groups_2_labels_list)

        # Find the intersection of labels
        common_labels = list(set1.intersection(set2))

        poe_stats_per_label = {}
        for label in common_labels:
            mask1 = (groups_1_labels == label).squeeze()
            mask2 = (groups_2_labels == label).squeeze()
            groups_1_stats_label = {key: value[mask1] for key, value in groups_1_stats.items()}
            groups_2_stats_label = {key: value[mask2] for key, value in groups_2_stats.items()}
            poe_stats_label = self._poe2({0: groups_1_stats_label, 1: groups_2_stats_label})
            poe_stats_per_label[label] = poe_stats_label

        poe_stats = {}
        for label, value in poe_stats_per_label.items():
            dataset_tensors = {}
            for group, tensors in value.items():
                tensor_dict = {}
                for tensor_key, tensor in tensors.items():
                    if tensor_key in tensor_dict:
                        tensor_dict[tensor_key] = torch.cat([tensor_dict[tensor_key], tensor], dim=0)
                    else:
                        tensor_dict[tensor_key] = tensor
                dataset_tensors[group] = tensor_dict
            poe_stats[label] = dataset_tensors

        # Find the unique labels in each tensor
        unique_labels1 = torch.unique(groups_1_labels)
        unique_labels2 = torch.unique(groups_2_labels)

        # Find the non-common labels for tensor1
        non_common_labels1 = unique_labels1[~torch.isin(unique_labels1, unique_labels2)]

        # Find the non-common labels for tensor2
        non_common_labels2 = unique_labels2[~torch.isin(unique_labels2, unique_labels1)]

        for label in non_common_labels1:
            poe_stats[label.item()] = {}
            mask1 = (groups_1_labels == label).squeeze()
            groups_1_stats_label = {key: value[mask1] for key, value in groups_1_stats.items()}
            groups_2_stats_label = {
                "logtheta_loc": torch.zeros_like(groups_1_stats_label["logtheta_loc"]),
                "logtheta_logvar": torch.ones_like(groups_1_stats_label["logtheta_logvar"]),
            }
            poe_stats_label = self._poe2({0: groups_1_stats_label, 1: groups_2_stats_label})
            poe_stats_label[1] = {
                "logtheta_loc": torch.empty((0, poe_stats_label[0]["logtheta_loc"].shape[1])),
                "logtheta_logvar": torch.empty((0, poe_stats_label[0]["logtheta_logvar"].shape[1])),
                "logtheta_scale": torch.empty((0, poe_stats_label[0]["logtheta_scale"].shape[1])),
            }
            poe_stats[label.item()] = poe_stats_label  #  we try without poe with unmatched cell types

        for label in non_common_labels2:
            poe_stats[label.item()] = {}
            mask2 = (groups_2_labels == label).squeeze()
            groups_2_stats_label = {key: value[mask2] for key, value in groups_2_stats.items()}
            groups_1_stats_label = {
                "logtheta_loc": torch.zeros_like(groups_2_stats_label["logtheta_loc"]),
                "logtheta_logvar": torch.ones_like(groups_2_stats_label["logtheta_logvar"]),
            }
            poe_stats_label = self._poe2({0: groups_1_stats_label, 1: groups_2_stats_label})

            poe_stats_label[0] = {
                "logtheta_loc": torch.empty((0, poe_stats_label[1]["logtheta_loc"].shape[1])),
                "logtheta_logvar": torch.empty((0, poe_stats_label[1]["logtheta_logvar"].shape[1])),
                "logtheta_scale": torch.empty((0, poe_stats_label[1]["logtheta_scale"].shape[1])),
            }
            poe_stats[label.item()] = poe_stats_label  # we try without poe with unmatched cell types

        # Initialize the output tensors
        groups_1_output = {
            "logtheta_loc": torch.empty(
                groups_1_stats["logtheta_loc"].shape[0], groups_1_stats["logtheta_loc"].shape[1], dtype=torch.float32
            ),
            "logtheta_logvar": torch.empty(
                groups_1_stats["logtheta_loc"].shape[0], groups_1_stats["logtheta_loc"].shape[1], dtype=torch.float32
            ),
            "logtheta_scale": torch.empty(
                groups_1_stats["logtheta_loc"].shape[0], groups_1_stats["logtheta_loc"].shape[1], dtype=torch.float32
            ),
        }

        groups_2_output = {
            "logtheta_loc": torch.empty(
                groups_2_stats["logtheta_loc"].shape[0], groups_2_stats["logtheta_loc"].shape[1], dtype=torch.float32
            ),
            "logtheta_logvar": torch.empty(
                groups_2_stats["logtheta_loc"].shape[0], groups_2_stats["logtheta_loc"].shape[1], dtype=torch.float32
            ),
            "logtheta_scale": torch.empty(
                groups_2_stats["logtheta_loc"].shape[0], groups_2_stats["logtheta_loc"].shape[1], dtype=torch.float32
            ),
        }

        # Initialize a dictionary to store the count of occurrences for each label
        label_count = {}
        # Iterate over the labels and assign values to the output tensor
        for i, label in enumerate(groups_1_labels):
            # Get the count of occurrences for the current label
            count = label_count.get(label.item(), 0)

            # Update the count for the current label
            label_count[label.item()] = count + 1

            # Calculate the tensor index based on the count of occurrences
            tensor_index = count % poe_stats[label.item()][0]["logtheta_loc"].size(0)

            groups_1_output["logtheta_loc"][i] = poe_stats[label.item()][0]["logtheta_loc"][tensor_index, :]
            groups_1_output["logtheta_logvar"][i] = poe_stats[label.item()][0]["logtheta_logvar"][tensor_index, :]
            groups_1_output["logtheta_scale"][i] = poe_stats[label.item()][0]["logtheta_scale"][tensor_index, :]

        # Initialize a dictionary to store the count of occurrences for each label
        label_count = {}
        # Iterate over the labels and assign values to the output tensor
        for i, label in enumerate(groups_2_labels):
            # Get the count of occurrences for the current label
            count = label_count.get(label.item(), 0)

            # Update the count for the current label
            label_count[label.item()] = count + 1

            # Calculate the tensor index based on the count of occurrences
            tensor_index = count % poe_stats[label.item()][1]["logtheta_loc"].size(0)

            groups_2_output["logtheta_loc"][i] = poe_stats[label.item()][1]["logtheta_loc"][tensor_index, :]
            groups_2_output["logtheta_logvar"][i] = poe_stats[label.item()][1]["logtheta_logvar"][tensor_index, :]
            groups_2_output["logtheta_scale"][i] = poe_stats[label.item()][1]["logtheta_scale"][tensor_index, :]

        concat_poe_stats = {0: groups_1_output, 1: groups_2_output}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose the appropriate device

        # Transfer tensors to the desired device
        for key, value in concat_poe_stats.items():
            for sub_key, tensor in value.items():
                concat_poe_stats[key][sub_key] = tensor.to(device)

        concat_poe_stats[0]["logtheta_qz"] = Normal(
            concat_poe_stats[0]["logtheta_loc"], concat_poe_stats[0]["logtheta_scale"]
        )
        concat_poe_stats[0]["logtheta_log_z"] = (
            concat_poe_stats[0]["logtheta_qz"].rsample().to("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        concat_poe_stats[0]["logtheta_theta"] = F.softmax(concat_poe_stats[0]["logtheta_log_z"], -1)

        concat_poe_stats[1]["logtheta_qz"] = Normal(
            concat_poe_stats[1]["logtheta_loc"], concat_poe_stats[1]["logtheta_scale"]
        )
        concat_poe_stats[1]["logtheta_log_z"] = (
            concat_poe_stats[1]["logtheta_qz"].rsample().to("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        concat_poe_stats[1]["logtheta_theta"] = F.softmax(concat_poe_stats[1]["logtheta_log_z"], -1)

        return concat_poe_stats

    def _poe2(self, shared_stats: dict):
        if len(shared_stats.keys()) > 2:
            raise ValueError(
                f"Number of groups passed to `_poe` is {len(shared_stats.keys())}, the only supported value is 2, make sure you passed only 2 groups to `prepare_adatas`"
            )

        groups_1, groups_2 = shared_stats.values()
        groups_1_size = groups_1["logtheta_logvar"].shape[0]
        groups_2_size = groups_2["logtheta_logvar"].shape[0]

        vars_groups_1 = torch.exp(groups_1["logtheta_logvar"])
        vars_groups_2 = torch.exp(groups_2["logtheta_logvar"])
        inverse_vars_groups_1 = 1.0 / vars_groups_1
        inverse_vars_groups_2 = 1.0 / vars_groups_2

        if inverse_vars_groups_1.shape != inverse_vars_groups_2.shape:
            if inverse_vars_groups_1.shape[0] < inverse_vars_groups_2.shape[0]:
                inverse_vars_groups_1_zeros = torch.ones_like(inverse_vars_groups_2)
                inverse_vars_groups_1_zeros[:groups_1_size] = inverse_vars_groups_1
                inverse_vars_groups_1 = inverse_vars_groups_1_zeros
                del inverse_vars_groups_1_zeros

            else:
                inverse_vars_groups_2_zeros = torch.ones_like(inverse_vars_groups_1)
                inverse_vars_groups_2_zeros[:groups_2_size] = inverse_vars_groups_2
                inverse_vars_groups_2 = inverse_vars_groups_2_zeros
                del inverse_vars_groups_2_zeros

        inverse_vars = torch.stack([inverse_vars_groups_1, inverse_vars_groups_2], dim=1)

        mus_vars_div_groups_1 = groups_1["logtheta_loc"] / vars_groups_1
        mus_vars_div_groups_2 = groups_2["logtheta_loc"] / vars_groups_2

        if mus_vars_div_groups_1.shape != mus_vars_div_groups_2.shape:
            if mus_vars_div_groups_1.shape[0] < mus_vars_div_groups_2.shape[0]:
                mus_vars_div_groups_1_zeros = torch.zeros_like(mus_vars_div_groups_2)
                mus_vars_div_groups_1_zeros[:groups_1_size] = mus_vars_div_groups_1
                mus_vars_div_groups_1 = mus_vars_div_groups_1_zeros
                del mus_vars_div_groups_1_zeros

            else:
                mus_vars_div_groups_2_zeros = torch.zeros_like(mus_vars_div_groups_1)
                mus_vars_div_groups_2_zeros[:groups_2_size] = mus_vars_div_groups_2
                mus_vars_div_groups_2 = mus_vars_div_groups_2_zeros
                del mus_vars_div_groups_2_zeros

        mus_vars = torch.stack([mus_vars_div_groups_1, mus_vars_div_groups_2], dim=1)

        if vars_groups_1.shape != vars_groups_2.shape:
            if vars_groups_1.shape[0] < vars_groups_2.shape[0]:
                vars_groups_1_zeros = torch.zeros_like(vars_groups_2)
                vars_groups_1_zeros[:groups_1_size] = vars_groups_1
                vars_groups_1 = vars_groups_1_zeros
                del vars_groups_1_zeros

            else:
                vars_groups_2_zeros = torch.zeros_like(vars_groups_1)
                vars_groups_2_zeros[:groups_2_size] = vars_groups_2
                vars_groups_2 = vars_groups_2_zeros
                del vars_groups_2_zeros

        _vars = torch.stack([vars_groups_1, vars_groups_2], dim=1)

        mus_joint = torch.sum(mus_vars, dim=1)
        logvars_joint = torch.ones_like(mus_joint)
        logvars_joint += torch.sum(inverse_vars, dim=1)
        logvars_joint = 1.0 / logvars_joint
        mus_joint *= logvars_joint
        logvars_joint = torch.log(logvars_joint)

        mus_joint_groups_1 = mus_joint[:groups_1_size]
        mus_joint_groups_2 = mus_joint[:groups_2_size]
        logvars_joint_groups_1 = logvars_joint[:groups_1_size]
        logvars_joint_groups_2 = logvars_joint[:groups_2_size]

        # groups_1
        logtheta_scale_groups_1 = torch.sqrt(torch.exp(logvars_joint_groups_1))
        qz_shared_groups_1 = Normal(mus_joint_groups_1, logtheta_scale_groups_1)
        log_z_shared_groups_1 = qz_shared_groups_1.rsample().to("cuda:0" if torch.cuda.is_available() else "cpu")
        F.softmax(log_z_shared_groups_1, -1)
        # groups_2
        logtheta_scale_groups_2 = torch.sqrt(torch.exp(logvars_joint_groups_2))
        qz_shared_groups_2 = Normal(mus_joint_groups_2, logtheta_scale_groups_2)
        log_z_shared_groups_2 = qz_shared_groups_2.rsample().to("cuda:0" if torch.cuda.is_available() else "cpu")
        F.softmax(log_z_shared_groups_2, -1)

        return {
            0: {
                "logtheta_loc": mus_joint_groups_1,
                "logtheta_logvar": logvars_joint_groups_1,
                "logtheta_scale": logtheta_scale_groups_1,
            },
            1: {
                "logtheta_loc": mus_joint_groups_2,
                "logtheta_logvar": logvars_joint_groups_2,
                "logtheta_scale": logtheta_scale_groups_2,
            },
        }

    def _get_inference_input(
        self,
        tensors_by_group,
    ):
        batch_index = [group[REGISTRY_KEYS.BATCH_KEY] for group in tensors_by_group]
        x = {int(k): group[REGISTRY_KEYS.X_KEY] for group in tensors_by_group for k in np.unique(group["groups"].cpu())}
        groups = [group["groups"] for group in tensors_by_group]

        labels = [group["labels"].flatten() for group in tensors_by_group]

        input_dict = {"x": x, "batch_index": batch_index, "groups": groups, "labels": labels}
        return input_dict

    def _get_generative_input(self, tensors_by_group, inference_outputs):
        private_stats = inference_outputs["private_stats"]
        shared_stats = inference_outputs["shared_stats"]
        private_shared_stats = inference_outputs["private_shared_stats"]
        poe_stats = inference_outputs["poe_stats"]
        library = inference_outputs["library"]
        groups = [group["groups"] for group in tensors_by_group]
        batch_index = [group[REGISTRY_KEYS.BATCH_KEY] for group in tensors_by_group]

        input_dict = {
            "private_stats": private_stats,
            "shared_stats": shared_stats,
            "private_shared_stats": private_shared_stats,
            "poe_stats": poe_stats,
            "library": library,
            "groups": groups,
            "batch_index": batch_index,
        }
        return input_dict

    @auto_move_data
    def inference(self, x, batch_index, groups, labels):
        """Runs the encoder model."""
        # encoder_input = x
        # [int(np.unique(group.cpu()).item()) for group in groups]
        x = {
            i: xs[:, self.groups_var_indices[i]] for i, xs in x.items()
        }  # update each groups minibatch with its own gene indices
        if self.log_variational_inference:
            x = {i: torch.log(1 + xs) for i, xs in x.items()}  # logvariational
        label_group = dict(enumerate(labels))
        library = {i: torch.log(xs.sum(1)).unsqueeze(1) for i, xs in x.items()}  # observed library size

        stats = {}
        for (group, item), batch in zip(x.items(), batch_index):
            encoder = self.encoders[group]
            values = encoder(item, group, batch)
            stats[group] = values
        private_stats = {}
        shared_stats = {}
        private_shared_stats = {}

        for key, value in stats.items():
            private_stats[key] = value["private"]
            shared_stats[key] = value["shared"]
            private_shared_stats[key] = value["ps"]

        poe_stats = self._supervised_poe(shared_stats, label_group)

        outputs = {
            "private_stats": private_stats,
            "shared_stats": shared_stats,
            "poe_stats": poe_stats,
            "private_shared_stats": private_shared_stats,
            "library": library,
        }

        return outputs

    @auto_move_data
    def generative(self, private_stats, shared_stats, private_shared_stats, poe_stats, library, groups, batch_index):
        """Runs the generative model."""
        if (len(private_stats.items()) > 2) or (len(shared_stats.items()) > 2):
            raise ValueError(
                f"Number of groups passed to `generative` is shared:{len(shared_stats.keys())}, private:{len(private_stats.keys())}, the only supported value is 2, make sure you passed only 2 groups to `prepare_adatas`"
            )
        _, _, _, groups_1_private_log_z, groups_1_private_theta, _ = private_stats[0].values()
        _, _, _, groups_2_private_log_z, groups_2_private_theta, _ = private_stats[1].values()
        _, _, _, _, groups_1_poe_log_z, groups_1_poe_theta = poe_stats[0].values()
        _, _, _, _, groups_2_poe_log_z, groups_2_poe_theta = poe_stats[1].values()

        # private1-poe groups_1 -> reconstruct data from groups_1 (Decoder_0)
        groups_1_private_poe_log_z = torch.cat((groups_1_private_log_z, groups_1_poe_log_z), dim=-1)
        groups_1_private_poe_theta = torch.cat((groups_1_private_theta, groups_1_poe_theta), dim=-1)

        # private1-poe groups_1 -> reconstruct data from groups_1 (Decoder_0)
        groups_2_private_poe_log_z = torch.cat((groups_2_private_log_z, groups_2_poe_log_z), dim=-1)
        groups_2_private_poe_theta = torch.cat((groups_2_private_theta, groups_2_poe_theta), dim=-1)

        private_poe = {
            0: {"log_z": groups_1_private_poe_log_z, "theta": groups_1_private_poe_theta},
            1: {"log_z": groups_2_private_poe_log_z, "theta": groups_2_private_poe_theta},
        }

        shared_stats = {}

        poe_stats = {}
        for (group, stats), batch in zip(private_poe.items(), batch_index):
            key = str(group)
            decoder = self.decoders[group]
            px_scale_private, px_scale_shared, px_rate_private, px_rate_shared, px_mixing, px_scale = decoder(
                self.dispersion,
                stats["log_z"][:, self.n_dimensions_shared : self.n_dimensions_private + self.n_dimensions_shared],
                stats["log_z"][:, : self.n_dimensions_shared],
                library[group],
                batch,
            )
            px_r = torch.exp(self.px_r[group])  # TO-DO specify px_r per groups
            px = NegativeBinomialMixture(mu1=px_rate_private, mu2=px_rate_shared, theta1=px_r, mixture_logits=px_mixing)
            pz = Normal(torch.zeros_like(stats["log_z"]), torch.ones_like(stats["log_z"]))
            poe_stats[key] = {
                "px_scale_private": px_scale_private,
                "px_scale_shared": px_scale_shared,
                "px_rate_private": px_rate_private,
                "px_rate_shared": px_rate_shared,
                "px": px,
                "pz": pz,
            }

        outputs = {"private_shared": shared_stats, "private_poe": poe_stats}
        return outputs

    @torch.inference_mode()
    def get_loadings(self, dataset: int, type_latent: str) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the linear decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        self.use_batch_norm = True  # REMOVE LATER
        if type_latent not in ["shared", "private"]:
            raise ValueError(f"Invalid value for type_latent: {type_latent}. It can only be 'shared' or 'private'")
        if self.use_batch_norm is True:
            w = (
                self.decoders[dataset].factor_regressor_private.fc_layers[0][0].weight
                if type_latent == "private"
                else self.decoders[dataset].factor_regressor_shared.fc_layers[0][0].weight
            )
            bn = (
                self.decoders[dataset].factor_regressor_private.fc_layers[0][1]
                if type_latent == "private"
                else self.decoders[dataset].factor_regressor_shared.fc_layers[0][1]
            )
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            b_identity = torch.diag(b)
            loadings = torch.matmul(b_identity, w)
        else:
            loadings = (
                self.decoders[dataset].factor_regressor_private.fc_layers[0][0].weight
                if type_latent == "private"
                else self.decoders[dataset].factor_regressor_shared.fc_layers[0][0].weight
            )

        loadings = loadings.detach().cpu().numpy()
        if self.n_batch > 1:
            loadings = loadings[:, : -self.n_batch]

        return loadings

    def loss(
        self,
        tensors_by_group,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """Loss function."""
        x = {int(k): group[REGISTRY_KEYS.X_KEY] for group in tensors_by_group for k in np.unique(group["groups"].cpu())}
        x = {i: xs[:, self.groups_var_indices[i]] for i, xs in x.items()}

        if self.log_variational_generative:
            x = {i: torch.log(1 + xs) for i, xs in x.items()}  # logvariational

        reconstruction_loss_groups_1_poe = -generative_outputs["private_poe"]["0"]["px"].log_prob(x[0]).sum(-1)
        reconstruction_loss_groups_2_poe = -generative_outputs["private_poe"]["1"]["px"].log_prob(x[1]).sum(-1)

        # distributions approx
        qz_private_groups_1 = inference_outputs["private_stats"][0][
            "qz"
        ]  # (batch_size, shared_dimensions + private_dimensions)
        qz_private_groups_2 = inference_outputs["private_stats"][1][
            "qz"
        ]  # (batch_size, shared_dimensions + private_dimensions)
        qz_poe_groups_1 = inference_outputs["poe_stats"][0][
            "logtheta_qz"
        ]  # (batch_size, shared_dimensions + private_dimensions)
        qz_poe_groups_2 = inference_outputs["poe_stats"][1][
            "logtheta_qz"
        ]  # (batch_size, shared_dimensions + private_dimensions)

        # kl
        kl_divergence_private_groups_1 = kl(
            qz_private_groups_1,
            Normal(
                torch.zeros_like(inference_outputs["private_stats"][0]["log_z"]),
                torch.ones_like(inference_outputs["private_stats"][0]["log_z"]),
            ),
        ).sum(dim=1)
        kl_divergence_private_groups_2 = kl(
            qz_private_groups_2,
            Normal(
                torch.zeros_like(inference_outputs["private_stats"][1]["log_z"]),
                torch.ones_like(inference_outputs["private_stats"][1]["log_z"]),
            ),
        ).sum(dim=1)
        kl_divergence_poe_groups_1 = kl(
            qz_poe_groups_1,
            Normal(
                torch.zeros_like(inference_outputs["poe_stats"][0]["logtheta_log_z"]),
                torch.ones_like(inference_outputs["poe_stats"][0]["logtheta_log_z"]),
            ),
        ).sum(dim=1)
        kl_divergence_poe_groups_2 = kl(
            qz_poe_groups_2,
            Normal(
                torch.zeros_like(inference_outputs["poe_stats"][1]["logtheta_log_z"]),
                torch.ones_like(inference_outputs["poe_stats"][1]["logtheta_log_z"]),
            ),
        ).sum(dim=1)

        extra_metrics = {
            "kl_divergence_private_groups_1": kl_divergence_private_groups_1.mean(),
            "kl_divergence_poe_groups_1": kl_divergence_poe_groups_1.mean(),
            "kl_divergence_private_groups_2": kl_divergence_private_groups_2.mean(),
            "kl_divergence_poe_groups_2": kl_divergence_poe_groups_2.mean(),
        }
        reconst_losses = {
            "reconst_loss_groups_1_poe": reconstruction_loss_groups_1_poe,
            "reconst_loss_groups_2_poe": reconstruction_loss_groups_2_poe,
        }
        kl_local = {
            "kl_divergence_groups_1_private": kl_divergence_private_groups_1,
            "kl_divergence_groups_1_poe": kl_divergence_poe_groups_1,
            "kl_divergence_groups_2_private": kl_divergence_private_groups_2,
            "kl_divergence_groups_2_poe": kl_divergence_poe_groups_2,
        }
        loss = torch.mean(
            reconstruction_loss_groups_1_poe
            + reconstruction_loss_groups_2_poe
            + kl_weight * (kl_divergence_private_groups_1)
            + kl_weight * kl_divergence_poe_groups_1
            + kl_weight * (kl_divergence_private_groups_2)
            + kl_weight * kl_divergence_poe_groups_2
        )

        output = LossOutput(
            loss=loss, reconstruction_loss=reconst_losses, kl_local=kl_local, extra_metrics=extra_metrics
        )

        return output
