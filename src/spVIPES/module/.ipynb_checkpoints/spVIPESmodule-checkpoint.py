"""Main module."""
import math
from typing import Dict, Literal

import numpy as np
import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, NegativeBinomialMixture
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import LinearDecoderSCVI
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
import time

from spVIPES.nn.networks import Encoder, LinearDecoderSPVIPE

from .utils import logsumexp, mutual_information

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
    n_topics
        Number of topics in the latent space.
    dropout_rate
        Dropout rate for neural networks
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    """

    def __init__(
        self,
        species_lengths,
        species_obs_names,
        species_var_names,
        species_obs_indices,
        species_var_indices,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_topics_shared: int = 25,
        n_topics_private: int = 25,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        log_variational_inference: bool = True,
        log_variational_generative: bool = True,
        dispersion: Literal["gene", "gene-batch", "gene-cell"] = "gene"
        # encode_covariates: bool = False, # by default scvi doesnt include covariates in encoder, only in decoder....
    ):
        super().__init__()
        self.n_topics_shared = n_topics_shared
        self.n_topics_private = n_topics_private
        self.n_batch = n_batch
        self.px_r = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(length)) for length in species_lengths.values()]
        )
        self.input_dims = species_lengths
        self.species_barcodes = species_obs_names
        self.species_genes = species_var_names
        self.species_obs_indices = species_obs_indices
        self.species_var_indices = species_var_indices
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.label_per_batch = []
        self.dispersion = dispersion
        self.log_variational_inference = log_variational_inference
        self.log_variational_generative = log_variational_generative
        # cat_list includes both batch ids and cat_covs
        cat_list = [n_batch] if n_batch > 0 else None
        self.encoders = {
            species: Encoder(
                x_dim,
                n_topics_shared,
                n_topics_private,
                hidden=n_hidden,
                dropout=dropout_rate,
                n_cat_list=cat_list,
                species=species,
            )
            for species, x_dim in self.input_dims.items()
        }

        # n_input_decoder = n_topics_shared + n_topics_private
        self.decoders = {
            species: LinearDecoderSPVIPE(
                n_topics_private,
                n_topics_shared,
                x_dim,
                # hidden=n_hidden,
                n_cat_list=cat_list,
                use_batch_norm=True,
                use_layer_norm=False,
                bias=False,
            )
            for species, x_dim in self.input_dims.items()
        }

        # register sub-modules
        for (species, values_encoder), (_, values_decoder) in zip(self.encoders.items(), self.decoders.items()):
            self.add_module(f"encoder_{species}", values_encoder)
            self.add_module(f"decoder_{species}", values_decoder)

    #     def _supervised_poe(self, shared_stats: Dict, label_specie: Dict):
    #         stats_keys = ["logtheta_loc", "logtheta_logvar", "logtheta_scale"]
    #         species_1_stats, species_2_stats = ({k: v[k] for k in stats_keys if k in v} for v in shared_stats.values())
    #         species_1_labels, species_2_labels = (v.flatten().tolist() for v in label_specie.values())
    #         common_labels = list(set(species_1_labels).intersection(species_2_labels))

    #         unique_labels1 = torch.unique(torch.tensor(species_1_labels))
    #         unique_labels2 = torch.unique(torch.tensor(species_2_labels))
    #         non_common_labels1 = unique_labels1[~torch.isin(unique_labels1, unique_labels2)]
    #         non_common_labels2 = unique_labels2[~torch.isin(unique_labels2, unique_labels1)]

    #         poe_stats = {}
    #         for species_stats, species_labels, non_common_labels, species_index in zip(
    #                 [species_1_stats, species_2_stats], [species_1_labels, species_2_labels],
    #                 [non_common_labels1, non_common_labels2], [0, 1]):
    #             for label in common_labels + non_common_labels.tolist():
    #                 mask = (torch.tensor(species_labels) == label).squeeze()
    #                 species_stats_label = {key: value[mask] for key, value in species_stats.items()}
    #                 if label in common_labels:
    #                     other_species_index = 1 - species_index
    #                     other_species_stats_label = {key: value[mask] for key, value in species_stats.items()}
    #                     poe_stats_label = self._poe2({species_index: species_stats_label, other_species_index: other_species_stats_label})
    #                 else:
    #                     poe_stats_label = {
    #                         "logtheta_loc": torch.zeros_like(species_stats_label["logtheta_loc"]),
    #                         "logtheta_logvar": torch.ones_like(species_stats_label["logtheta_logvar"]),
    #                     }
    #                     poe_stats_label = self._poe2({0: species_stats_label, 1: poe_stats_label})
    #                     poe_stats_label[1 - species_index] = {
    #                         "logtheta_loc": torch.empty((0, poe_stats_label[species_index]["logtheta_loc"].shape[1])),
    #                         "logtheta_logvar": torch.empty((0, poe_stats_label[species_index]["logtheta_logvar"].shape[1])),
    #                         "logtheta_scale": torch.empty((0, poe_stats_label[species_index]["logtheta_scale"].shape[1])),
    #                     }
    #                 poe_stats[label] = poe_stats_label

    #         def initialize_output(stats):
    #             return {k: torch.empty(stats["logtheta_loc"].shape[0], stats["logtheta_loc"].shape[1], dtype=torch.float32) for k in stats_keys}

    #         def update_output(species_labels, output, stats):
    #             label_count = {}
    #             for i, label in enumerate(species_labels):
    #                 count = label_count.get(label, 0)
    #                 label_count[label] = count + 1
    #                 tensor_index = count % stats[label][0]["logtheta_loc"].size(0)
    #                 for k in stats_keys:
    #                     output[k][i] = stats[label][0][k][tensor_index, :]

    #         species_1_output = initialize_output(species_1_stats)
    #         species_2_output = initialize_output(species_2_stats)
    #         update_output(species_1_labels, species_1_output, poe_stats)
    #         update_output(species_2_labels, species_2_output, poe_stats)

    #         concat_poe_stats = {0: species_1_output, 1: species_2_output}

    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         for _, value in concat_poe_stats.items():
    #             for _, tensor in value.items():
    #                 tensor.to(device)

    #         for i in range(2):
    #             concat_poe_stats[i]["logtheta_qz"] = Normal(concat_poe_stats[i]["logtheta_loc"], concat_poe_stats[i]["logtheta_scale"])
    #             concat_poe_stats[i]["logtheta_log_z"] = concat_poe_stats[i]["logtheta_qz"].rsample().to(device)
    #             concat_poe_stats[i]["logtheta_theta"] = F.softmax(concat_poe_stats[i]["logtheta_log_z"], -1)

    #         return concat_poe_stats

    def _supervised_poe(self, shared_stats: Dict, label_specie: Dict):
        species_1_stats, species_2_stats = shared_stats.values()
        species_1_stats = {
            k: species_1_stats[k] for k in ["logtheta_loc", "logtheta_logvar", "logtheta_scale"] if k in species_1_stats
        }
        species_2_stats = {
            k: species_2_stats[k] for k in ["logtheta_loc", "logtheta_logvar", "logtheta_scale"] if k in species_2_stats
        }

        species_1_labels, species_2_labels = label_specie.values()

        species_1_labels_list = species_1_labels.flatten().tolist()  # to keep track of order
        species_2_labels_list = species_2_labels.flatten().tolist()  # to keep track of order
        # Convert label tensors to sets
        set1 = set(species_1_labels_list)
        set2 = set(species_2_labels_list)

        # Find the intersection of labels
        common_labels = list(set1.intersection(set2))

        poe_stats_per_label = {}
        for label in common_labels:
            mask1 = (species_1_labels == label).squeeze()
            mask2 = (species_2_labels == label).squeeze()
            species_1_stats_label = {key: value[mask1] for key, value in species_1_stats.items()}
            species_2_stats_label = {key: value[mask2] for key, value in species_2_stats.items()}
            poe_stats_label = self._poe2({0: species_1_stats_label, 1: species_2_stats_label})
            poe_stats_per_label[label] = poe_stats_label

        poe_stats = {}
        for label, value in poe_stats_per_label.items():
            dataset_tensors = {}
            for specie, tensors in value.items():
                tensor_dict = {}
                for tensor_key, tensor in tensors.items():
                    if tensor_key in tensor_dict:
                        tensor_dict[tensor_key] = torch.cat([tensor_dict[tensor_key], tensor], dim=0)
                    else:
                        tensor_dict[tensor_key] = tensor
                dataset_tensors[specie] = tensor_dict
            poe_stats[label] = dataset_tensors

        # Find the unique labels in each tensor
        unique_labels1 = torch.unique(species_1_labels)
        unique_labels2 = torch.unique(species_2_labels)

        # Find the non-common labels for tensor1
        non_common_labels1 = unique_labels1[~torch.isin(unique_labels1, unique_labels2)]

        # Find the non-common labels for tensor2
        non_common_labels2 = unique_labels2[~torch.isin(unique_labels2, unique_labels1)]

        for label in non_common_labels1:
            poe_stats[label.item()] = {}
            mask1 = (species_1_labels == label).squeeze()
            species_1_stats_label = {key: value[mask1] for key, value in species_1_stats.items()}
            species_2_stats_label = {
                "logtheta_loc": torch.zeros_like(species_1_stats_label["logtheta_loc"]),
                "logtheta_logvar": torch.ones_like(species_1_stats_label["logtheta_logvar"]),
            }
            poe_stats_label = self._poe2({0: species_1_stats_label, 1: species_2_stats_label})
            poe_stats_label[1] = {
                "logtheta_loc": torch.empty((0, poe_stats_label[0]["logtheta_loc"].shape[1])),
                "logtheta_logvar": torch.empty((0, poe_stats_label[0]["logtheta_logvar"].shape[1])),
                "logtheta_scale": torch.empty((0, poe_stats_label[0]["logtheta_scale"].shape[1])),
            }
            poe_stats[label.item()] = poe_stats_label  #  we try without poe with unmatched cell types

        for label in non_common_labels2:
            poe_stats[label.item()] = {}
            mask2 = (species_2_labels == label).squeeze()
            species_2_stats_label = {key: value[mask2] for key, value in species_2_stats.items()}
            species_1_stats_label = {
                "logtheta_loc": torch.zeros_like(species_2_stats_label["logtheta_loc"]),
                "logtheta_logvar": torch.ones_like(species_2_stats_label["logtheta_logvar"]),
            }
            poe_stats_label = self._poe2({0: species_1_stats_label, 1: species_2_stats_label})

            poe_stats_label[0] = {
                "logtheta_loc": torch.empty((0, poe_stats_label[1]["logtheta_loc"].shape[1])),
                "logtheta_logvar": torch.empty((0, poe_stats_label[1]["logtheta_logvar"].shape[1])),
                "logtheta_scale": torch.empty((0, poe_stats_label[1]["logtheta_scale"].shape[1])),
            }
            poe_stats[label.item()] = poe_stats_label  # we try without poe with unmatched cell types

        # Initialize the output tensors
        species_1_output = {
            "logtheta_loc": torch.empty(
                species_1_stats["logtheta_loc"].shape[0], species_1_stats["logtheta_loc"].shape[1], dtype=torch.float32
            ),
            "logtheta_logvar": torch.empty(
                species_1_stats["logtheta_loc"].shape[0], species_1_stats["logtheta_loc"].shape[1], dtype=torch.float32
            ),
            "logtheta_scale": torch.empty(
                species_1_stats["logtheta_loc"].shape[0], species_1_stats["logtheta_loc"].shape[1], dtype=torch.float32
            ),
        }

        species_2_output = {
            "logtheta_loc": torch.empty(
                species_2_stats["logtheta_loc"].shape[0], species_2_stats["logtheta_loc"].shape[1], dtype=torch.float32
            ),
            "logtheta_logvar": torch.empty(
                species_2_stats["logtheta_loc"].shape[0], species_2_stats["logtheta_loc"].shape[1], dtype=torch.float32
            ),
            "logtheta_scale": torch.empty(
                species_2_stats["logtheta_loc"].shape[0], species_2_stats["logtheta_loc"].shape[1], dtype=torch.float32
            ),
        }

        # Initialize a dictionary to store the count of occurrences for each label
        label_count = {}
        # Iterate over the labels and assign values to the output tensor
        for i, label in enumerate(species_1_labels):
            # Get the count of occurrences for the current label
            count = label_count.get(label.item(), 0)

            # Update the count for the current label
            label_count[label.item()] = count + 1

            # Calculate the tensor index based on the count of occurrences
            tensor_index = count % poe_stats[label.item()][0]["logtheta_loc"].size(0)

            species_1_output["logtheta_loc"][i] = poe_stats[label.item()][0]["logtheta_loc"][tensor_index, :]
            species_1_output["logtheta_logvar"][i] = poe_stats[label.item()][0]["logtheta_logvar"][tensor_index, :]
            species_1_output["logtheta_scale"][i] = poe_stats[label.item()][0]["logtheta_scale"][tensor_index, :]

        # Initialize a dictionary to store the count of occurrences for each label
        label_count = {}
        # Iterate over the labels and assign values to the output tensor
        for i, label in enumerate(species_2_labels):
            # Get the count of occurrences for the current label
            count = label_count.get(label.item(), 0)

            # Update the count for the current label
            label_count[label.item()] = count + 1

            # Calculate the tensor index based on the count of occurrences
            tensor_index = count % poe_stats[label.item()][1]["logtheta_loc"].size(0)

            species_2_output["logtheta_loc"][i] = poe_stats[label.item()][1]["logtheta_loc"][tensor_index, :]
            species_2_output["logtheta_logvar"][i] = poe_stats[label.item()][1]["logtheta_logvar"][tensor_index, :]
            species_2_output["logtheta_scale"][i] = poe_stats[label.item()][1]["logtheta_scale"][tensor_index, :]

        concat_poe_stats = {0: species_1_output, 1: species_2_output}

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

    def _poe2(self, shared_stats: Dict):
        if len(shared_stats.keys()) > 2:
            raise ValueError(
                f"Number of species passed to `_poe` is {len(shared_stats.keys())}, the only supported value is 2, make sure you passed only 2 species to `prepare_adatas`"
            )

        #         species_1, species_2 = shared_stats.values()
        #         logvars = torch.stack([species_1["logtheta_logvar"], species_2["logtheta_logvar"]], dim=1)
        #         _vars = torch.exp(logvars)
        #         mus = torch.stack([species_1["logtheta_loc"], species_2["logtheta_loc"]], dim=1)
        #         mus_joint = torch.sum(mus / _vars, dim=1)
        #         logvars_joint = torch.ones_like(mus_joint)  # batch size
        #         logvars_joint += torch.sum(1 / _vars, dim=1)
        #         logvars_joint = 1.0 / logvars_joint  # inverse
        #         mus_joint *= logvars_joint
        #         logvars_joint = torch.log(logvars_joint)
        #         scales_joint = torch.sqrt(torch.exp(logvars_joint))

        #         qz_joint = Normal(mus_joint, scales_joint)
        #         log_z_joint = qz_joint.rsample().to("cuda:0" if torch.cuda.is_available() else "cpu")
        #         theta_joint = F.softmax(log_z_joint, -1)

        #         return {
        #             "logtheta_loc": mus_joint,
        #             "logtheta_logvar": logvars_joint,
        #             "logtheta_scale": scales_joint,
        #             "log_z": log_z_joint,
        #             "theta": theta_joint,
        #             "qz": qz_joint,
        #         }
        species_1, species_2 = shared_stats.values()
        species_1_size = species_1["logtheta_logvar"].shape[0]
        species_2_size = species_2["logtheta_logvar"].shape[0]

        vars_species_1 = torch.exp(species_1["logtheta_logvar"])
        vars_species_2 = torch.exp(species_2["logtheta_logvar"])
        inverse_vars_species_1 = 1.0 / vars_species_1
        inverse_vars_species_2 = 1.0 / vars_species_2

        if inverse_vars_species_1.shape != inverse_vars_species_2.shape:
            if inverse_vars_species_1.shape[0] < inverse_vars_species_2.shape[0]:
                inverse_vars_species_1_zeros = torch.ones_like(inverse_vars_species_2)
                inverse_vars_species_1_zeros[:species_1_size] = inverse_vars_species_1
                inverse_vars_species_1 = inverse_vars_species_1_zeros
                del inverse_vars_species_1_zeros

            else:
                inverse_vars_species_2_zeros = torch.ones_like(inverse_vars_species_1)
                inverse_vars_species_2_zeros[:species_2_size] = inverse_vars_species_2
                inverse_vars_species_2 = inverse_vars_species_2_zeros
                del inverse_vars_species_2_zeros

        inverse_vars = torch.stack([inverse_vars_species_1, inverse_vars_species_2], dim=1)

        mus_vars_div_species_1 = species_1["logtheta_loc"] / vars_species_1
        mus_vars_div_species_2 = species_2["logtheta_loc"] / vars_species_2

        if mus_vars_div_species_1.shape != mus_vars_div_species_2.shape:
            if mus_vars_div_species_1.shape[0] < mus_vars_div_species_2.shape[0]:
                mus_vars_div_species_1_zeros = torch.zeros_like(mus_vars_div_species_2)
                mus_vars_div_species_1_zeros[:species_1_size] = mus_vars_div_species_1
                mus_vars_div_species_1 = mus_vars_div_species_1_zeros
                del mus_vars_div_species_1_zeros

            else:
                mus_vars_div_species_2_zeros = torch.zeros_like(mus_vars_div_species_1)
                mus_vars_div_species_2_zeros[:species_2_size] = mus_vars_div_species_2
                mus_vars_div_species_2 = mus_vars_div_species_2_zeros
                del mus_vars_div_species_2_zeros

        mus_vars = torch.stack([mus_vars_div_species_1, mus_vars_div_species_2], dim=1)

        if vars_species_1.shape != vars_species_2.shape:
            if vars_species_1.shape[0] < vars_species_2.shape[0]:
                vars_species_1_zeros = torch.zeros_like(vars_species_2)
                vars_species_1_zeros[:species_1_size] = vars_species_1
                vars_species_1 = vars_species_1_zeros
                del vars_species_1_zeros

            else:
                vars_species_2_zeros = torch.zeros_like(vars_species_1)
                vars_species_2_zeros[:species_2_size] = vars_species_2
                vars_species_2 = vars_species_2_zeros
                del vars_species_2_zeros

        _vars = torch.stack([vars_species_1, vars_species_2], dim=1)

        mus_joint = torch.sum(mus_vars, dim=1)
        logvars_joint = torch.ones_like(mus_joint)
        logvars_joint += torch.sum(inverse_vars, dim=1)
        logvars_joint = 1.0 / logvars_joint
        mus_joint *= logvars_joint
        logvars_joint = torch.log(logvars_joint)

        mus_joint_species_1 = mus_joint[:species_1_size]
        mus_joint_species_2 = mus_joint[:species_2_size]
        logvars_joint_species_1 = logvars_joint[:species_1_size]
        logvars_joint_species_2 = logvars_joint[:species_2_size]

        # species_1
        logtheta_scale_species_1 = torch.sqrt(torch.exp(logvars_joint_species_1))
        qz_shared_species_1 = Normal(mus_joint_species_1, logtheta_scale_species_1)
        log_z_shared_species_1 = qz_shared_species_1.rsample().to("cuda:0" if torch.cuda.is_available() else "cpu")
        F.softmax(log_z_shared_species_1, -1)
        # species_2
        logtheta_scale_species_2 = torch.sqrt(torch.exp(logvars_joint_species_2))
        qz_shared_species_2 = Normal(mus_joint_species_2, logtheta_scale_species_2)
        log_z_shared_species_2 = qz_shared_species_2.rsample().to("cuda:0" if torch.cuda.is_available() else "cpu")
        F.softmax(log_z_shared_species_2, -1)

        return {
            0: {
                "logtheta_loc": mus_joint_species_1,
                "logtheta_logvar": logvars_joint_species_1,
                "logtheta_scale": logtheta_scale_species_1,
                # "log_z": log_z_shared_species_1,
                # "theta": theta_shared_species_1,
                # "qz": qz_shared_species_1,
            },
            1: {
                "logtheta_loc": mus_joint_species_2,
                "logtheta_logvar": logvars_joint_species_2,
                "logtheta_scale": logtheta_scale_species_2,
                # "log_z": log_z_shared_species_2,
                # "theta": theta_shared_species_2,
                # "qz": qz_shared_species_2,
            },
        }

    def _get_inference_input(
        self,
        tensors_by_group,
    ):
        batch_index = [group[REGISTRY_KEYS.BATCH_KEY] for group in tensors_by_group]
        x = {
            int(k): group[REGISTRY_KEYS.X_KEY] for group in tensors_by_group for k in np.unique(group["species"].cpu())
        }
        species = [group["species"] for group in tensors_by_group]

        labels = [group["labels"].flatten() for group in tensors_by_group]

        input_dict = dict(x=x, batch_index=batch_index, species=species, labels=labels)  # cat_covs=cat_covs

        return input_dict

    def _get_generative_input(self, tensors_by_group, inference_outputs):
        private_stats = inference_outputs["private_stats"]
        shared_stats = inference_outputs["shared_stats"]
        private_shared_stats = inference_outputs["private_shared_stats"]
        poe_stats = inference_outputs["poe_stats"]
        library = inference_outputs["library"]
        species = [group["species"] for group in tensors_by_group]
        batch_index = [group[REGISTRY_KEYS.BATCH_KEY] for group in tensors_by_group]

        input_dict = dict(
            private_stats=private_stats,
            shared_stats=shared_stats,
            private_shared_stats=private_shared_stats,
            poe_stats=poe_stats,
            library=library,
            species=species,
            batch_index=batch_index,
        )
        return input_dict

    @auto_move_data
    def inference(self, x, batch_index, species, labels):
        """Runs the encoder model."""
        # encoder_input = x
        # [int(np.unique(specie.cpu()).item()) for specie in species]
        x = {
            i: xs[:, self.species_var_indices[i]] for i, xs in x.items()
        }  # update each species minibatch with its own gene indices
        if self.log_variational_inference:
            x = {i: torch.log(1 + xs) for i, xs in x.items()}  # logvariational
        label_specie = {i: label for i, label in enumerate(labels)}

        library = {i: torch.log(xs.sum(1)).unsqueeze(1) for i, xs in x.items()}  # observed library size

        stats = {}
        for (specie, x), batch in zip(x.items(), batch_index):
            encoder = self.encoders[specie]
            values = encoder(x, specie, batch)
            stats[specie] = values

        private_stats = {}
        shared_stats = {}
        private_shared_stats = {}

        for key, value in stats.items():
            private_stats[key] = value["private"]
            shared_stats[key] = value["shared"]
            private_shared_stats[key] = value["ps"]

        poe_stats = self._supervised_poe(shared_stats, label_specie)

        outputs = dict(
            private_stats=private_stats,
            shared_stats=shared_stats,
            poe_stats=poe_stats,
            private_shared_stats=private_shared_stats,
            library=library,
        )

        return outputs

    @auto_move_data
    def generative(self, private_stats, shared_stats, private_shared_stats, poe_stats, library, species, batch_index):
        """Runs the generative model."""
        if (len(private_stats.items()) > 2) or (len(shared_stats.items()) > 2):
            raise ValueError(
                f"Number of species passed to `generative` is shared:{len(shared_stats.keys())}, private:{len(private_stats.keys())}, the only supported value is 2, make sure you passed only 2 species to `prepare_adatas`"
            )
        _, _, _, species_1_private_log_z, species_1_private_theta, _ = private_stats[0].values()
        _, _, _, species_2_private_log_z, species_2_private_theta, _ = private_stats[1].values()
        _, _, _, _, species_1_poe_log_z, species_1_poe_theta = poe_stats[0].values()
        _, _, _, _, species_2_poe_log_z, species_2_poe_theta = poe_stats[1].values()

        # private1-poe Species_1 -> reconstruct data from Species_1 (Decoder_0)
        species_1_private_poe_log_z = torch.cat((species_1_private_log_z, species_1_poe_log_z), dim=-1)
        species_1_private_poe_theta = torch.cat((species_1_private_theta, species_1_poe_theta), dim=-1)

        # private1-poe Species_1 -> reconstruct data from Species_1 (Decoder_0)
        species_2_private_poe_log_z = torch.cat((species_2_private_log_z, species_2_poe_log_z), dim=-1)
        species_2_private_poe_theta = torch.cat((species_2_private_theta, species_2_poe_theta), dim=-1)

        private_poe = {
            0: {"log_z": species_1_private_poe_log_z, "theta": species_1_private_poe_theta},
            1: {"log_z": species_2_private_poe_log_z, "theta": species_2_private_poe_theta},
        }

        shared_stats = {}
        # for (specie, stats), batch in zip(private_shared_stats.items(), batch_index):
        #     key = str(specie)
        #     decoder = self.decoders[specie]
        #     px_scale_private_shared, px_rate_private_shared  = decoder(self.dispersion, stats["log_z"][:, self.n_topics_shared : self.n_topics_private + self.n_topics_shared], stats["log_z"][:, : self.n_topics_shared], library[specie], batch)
        #     # px_scale, _, px_rate, px_dropout = decoder(self.dispersion, stats["log_z"], library[specie], batch)
        #     px_r = torch.exp(self.px_r[specie])  # TO-DO specify px_r per species
        #     px = NegativeBinomial(mu=px_rate_private_shared, theta=px_r, scale=px_scale_private_shared)
        #     # elif self.distribution == "mixturenegativebinomial":
        #     #     px = NegativeBinomialMixture(mu1=px_rate_private, mu2=px_rate_shared, theta1=px_r, mixture_logits=px_mixing)
        #     # else:
        #     #     px=None
        #     pz = Normal(torch.zeros_like(stats["log_z"]), torch.ones_like(stats["log_z"]))
        #     shared_stats[key] = {
        #         # "px_scale_private": px_scale_private,
        #         # "px_scale_shared": px_scale_shared,
        #         # "px_rate_private": px_rate_private,
        #         # "px_rate_shared": px_rate_shared,
        #         "px_scale":px_scale_private_shared,
        #         "px_rate": px_rate_private_shared,
        #         # "px_dropout": px_dropout,
        #         "px": px,
        #        "pz": pz,
        #     }

        poe_stats = {}
        for (specie, stats), batch in zip(private_poe.items(), batch_index):
            key = str(specie)
            decoder = self.decoders[specie]
            # px_scale_private, px_scale_shared, px_rate_private, px_rate_shared, px_mixing, px_scale, _, _   = decoder(self.dispersion, stats["log_z"][:, self.n_topics_shared : self.n_topics_private + self.n_topics_shared], stats["log_z"][:, : self.n_topics_shared], library[specie], batch)
            # # px_scale, _, px_rate, px_dropout = decoder(self.dispersion, stats["log_z"], library[specie], batch)
            # px_r = torch.exp(self.px_r[specie])  # TO-DO specify px_r per species
            # # px = NegativeBinomial(mu=(px_rate_private + px_rate_shared) / 2, theta=px_r, scale=px_scale)
            # # elif self.distribution == "mixturenegativebinomial":
            # px = NegativeBinomialMixture(mu1=px_rate_private, mu2=px_rate_shared, theta1=px_r, mixture_logits=px_mixing)
            px_scale_private, px_scale_shared, px_rate_private, px_rate_shared, px_mixing, px_scale = decoder(
                self.dispersion,
                stats["log_z"][:, self.n_topics_shared : self.n_topics_private + self.n_topics_shared],
                stats["log_z"][:, : self.n_topics_shared],
                library[specie],
                batch,
            )
            # px_scale, _, px_rate, px_dropout = decoder(self.dispersion, stats["log_z"], library[specie], batch)
            px_r = torch.exp(self.px_r[specie])  # TO-DO specify px_r per species
            # px = NegativeBinomial(mu=px_rate_private_shared, theta=px_r, scale=px_scale_private_shared)
            px = NegativeBinomialMixture(mu1=px_rate_private, mu2=px_rate_shared, theta1=px_r, mixture_logits=px_mixing)
            # else:
            #     px = None
            pz = Normal(torch.zeros_like(stats["log_z"]), torch.ones_like(stats["log_z"]))
            poe_stats[key] = {
                "px_scale_private": px_scale_private,
                "px_scale_shared": px_scale_shared,
                "px_rate_private": px_rate_private,
                "px_rate_shared": px_rate_shared,
                # "px_scale":px_scale_private_shared,
                # "px_rate": px_rate_private_shared,
                # "px_dropout": px_dropout,
                "px": px,
                "pz": pz,
            }

        outputs = dict(private_shared=shared_stats, private_poe=poe_stats)
        return outputs

    def _log_density_gaussian(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = -0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def _decomposed_kl(self, mu, logvar, p, z):
        """Decomposed KLD: MI | TC | DW-KL"""
        # q: estimated posterior dist
        # z: sampled latent
        alpha = 1.0
        beta = 1
        gamma = 1.0
        weight = 1  # kwargs['M_N']  # Account for the minibatch samples from the dataset
        log_q_zx = self._log_density_gaussian(z, q.loc, torch.log(q.scale**2)).sum(dim=1)
        log_p_z = p.log_prob(z).sum(dim=1)

        batch_size, latent_dim = z.shape
        mat_log_q_z = self._log_density_gaussian(
            z.view(batch_size, 1, latent_dim),
            mu.view(1, batch_size, latent_dim),
            logvar.view(1, batch_size, latent_dim),
        )
        dataset_size = (1 / 0.001) * batch_size  # dataset size
        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
        importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size - 1)).to("cuda")
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss = (log_q_zx - log_q_z) * alpha
        tc_loss = (log_q_z - log_prod_q_z) * beta
        kld_loss = (log_prod_q_z - log_p_z) * gamma

        return mi_loss + tc_loss + kld_loss

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

        # loadings = loadings / torch.max(torch.abs(loadings))
        # loadings = (loadings - torch.mean(loadings)) / torch.std(loadings)

        # loadings = softmax_temperature(loadings, 2)
        loadings = loadings.detach().cpu().numpy()
        if self.n_batch > 1:
            loadings = loadings[:, : -self.n_batch]

        # if type_latent == "shared":
        #     dimensions = self.n_topics_shared
        #     loadings = loadings[:, -dimensions:]
        # else:
        #     dimensions = self.n_topics_private
        #     loadings = loadings[:, :dimensions]

        return loadings

    def loss(
        self,
        tensors_by_group,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """Loss function."""
        x = {
            int(k): group[REGISTRY_KEYS.X_KEY] for group in tensors_by_group for k in np.unique(group["species"].cpu())
        }
        x = {i: xs[:, self.species_var_indices[i]] for i, xs in x.items()}

        if self.log_variational_generative:
            x = {i: torch.log(1 + xs) for i, xs in x.items()}  # logvariational

        # reconstruction_loss_species_1_shared = -generative_outputs["private_shared"]["0"]["px"].log_prob(x[0]).sum(-1)
        reconstruction_loss_species_1_poe = -generative_outputs["private_poe"]["0"]["px"].log_prob(x[0]).sum(-1)
        # reconstruction_loss_species_2_shared = -generative_outputs["private_shared"]["1"]["px"].log_prob(x[1]).sum(-1)
        reconstruction_loss_species_2_poe = -generative_outputs["private_poe"]["1"]["px"].log_prob(x[1]).sum(-1)

        # distributions approx
        qz_private_species_1 = inference_outputs["private_stats"][0][
            "qz"
        ]  # (batch_size, shared_topics + private_topics)
        qz_private_species_2 = inference_outputs["private_stats"][1][
            "qz"
        ]  # (batch_size, shared_topics + private_topics)
        qz_poe_species_1 = inference_outputs["poe_stats"][0][
            "logtheta_qz"
        ]  # (batch_size, shared_topics + private_topics)
        qz_poe_species_2 = inference_outputs["poe_stats"][1][
            "logtheta_qz"
        ]  # (batch_size, shared_topics + private_topics)
        qz_shared_species_1 = inference_outputs["shared_stats"][0]["qz"]  # (batch_size, shared_topics + private_topics)
        qz_shared_species_2 = inference_outputs["shared_stats"][1]["qz"]  #

        # kl
        kl_divergence_private_species_1 = kl(
            qz_private_species_1,
            Normal(
                torch.zeros_like(inference_outputs["private_stats"][0]["log_z"]),
                torch.ones_like(inference_outputs["private_stats"][0]["log_z"]),
            ),
        ).sum(dim=1)
        kl_divergence_private_species_2 = kl(
            qz_private_species_2,
            Normal(
                torch.zeros_like(inference_outputs["private_stats"][1]["log_z"]),
                torch.ones_like(inference_outputs["private_stats"][1]["log_z"]),
            ),
        ).sum(dim=1)
        kl_divergence_poe_species_1 = kl(
            qz_poe_species_1,
            Normal(
                torch.zeros_like(inference_outputs["poe_stats"][0]["logtheta_log_z"]),
                torch.ones_like(inference_outputs["poe_stats"][0]["logtheta_log_z"]),
            ),
        ).sum(dim=1)
        kl_divergence_poe_species_2 = kl(
            qz_poe_species_2,
            Normal(
                torch.zeros_like(inference_outputs["poe_stats"][1]["logtheta_log_z"]),
                torch.ones_like(inference_outputs["poe_stats"][1]["logtheta_log_z"]),
            ),
        ).sum(dim=1)

        #         kl_divergence_shared_species_2 = kl(
        #             qz_shared_species_2,
        #             Normal(
        #                 torch.zeros_like(inference_outputs["shared_stats"][1]["log_z"]),
        #                 torch.ones_like(inference_outputs["shared_stats"][1]["log_z"]),
        #             ),
        #         ).sum(dim=1)

        #         kl_divergence_shared_species_1 = kl(
        #             qz_shared_species_1,
        #             Normal(
        #                 torch.zeros_like(inference_outputs["shared_stats"][0]["log_z"]),
        #                 torch.ones_like(inference_outputs["shared_stats"][0]["log_z"]),
        #             ),
        #         ).sum(dim=1)

        extra_metrics = dict(
            kl_divergence_private_species_1=kl_divergence_private_species_1.mean(),
            kl_divergence_poe_species_1=kl_divergence_poe_species_1.mean(),
            kl_divergence_private_species_2=kl_divergence_private_species_2.mean(),
            kl_divergence_poe_species_2=kl_divergence_poe_species_2.mean(),
            # kl_divergence_shared_species_1=kl_divergence_shared_species_1.mean(),
            # kl_divergence_shared_species_2=kl_divergence_shared_species_2.mean()
        )
        reconst_losses = {
            # "reconst_loss_species_1_shared": reconstruction_loss_species_1_shared,
            "reconst_loss_species_1_poe": reconstruction_loss_species_1_poe,
            # "reconst_loss_species_2_shared": reconstruction_loss_species_2_shared,
            "reconst_loss_species_2_poe": reconstruction_loss_species_2_poe,
        }
        kl_local = {
            "kl_divergence_species_1_private": kl_divergence_private_species_1,
            "kl_divergence_species_1_poe": kl_divergence_poe_species_1,
            "kl_divergence_species_2_private": kl_divergence_private_species_2,
            "kl_divergence_species_2_poe": kl_divergence_poe_species_2,
            # "kl_divergence_species_1_shared": kl_divergence_shared_species_1,
            # "kl_divergence_species_2_shared": kl_divergence_shared_species_2,
        }
        loss = torch.mean(
            # reconstruction_loss_species_1_shared +
            reconstruction_loss_species_1_poe
            # + reconstruction_loss_species_2_shared
            + reconstruction_loss_species_2_poe
            + kl_weight * (kl_divergence_private_species_1)
            + kl_weight * kl_divergence_poe_species_1
            + kl_weight * (kl_divergence_private_species_2)
            + kl_weight * kl_divergence_poe_species_2
            # + kl_weight * (kl_divergence_shared_species_1)
            # + kl_weight * (kl_divergence_shared_species_2)
        )

        output = LossOutput(
            loss=loss, reconstruction_loss=reconst_losses, kl_local=kl_local, extra_metrics=extra_metrics
        )

        return output
