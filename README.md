# spVIPES

[![Tests][badge-tests]][link-tests]

<!-- [![Documentation][badge-docs]][link-docs] -->

[badge-tests]: https://img.shields.io/github/actions/workflow/status/nrclaudio/spVIPES/test.yaml?branch=main
[link-tests]: https://github.com/nrclaudio/spVIPES/actions/workflows/test.yml

<!-- [badge-docs]: https://img.shields.io/readthedocs/spVIPES -->

Shared-private Variational Inference with Product of Experts and Supervision

## Overview

spVIPES is a method for integrating multi-group single-cell datasets using a shared-private latent space approach. The model learns both shared representations (common across groups) and private representations (group-specific) through a Product of Experts (PoE) framework.

### Key Features

- **Multi-modal integration**: Integrate datasets with different feature sets (e.g., different gene panels)
- **Shared-private decomposition**: Learn both common and group-specific latent representations
- **Multiple PoE variants**: Support for different data alignment strategies
- **Batch correction**: Built-in batch effect correction capabilities
- **GPU acceleration**: Optimized for CUDA-enabled training

### Product of Experts Variants

spVIPES supports three different PoE strategies for data integration:

1. **Label-based PoE** 🎯
   - Uses explicit cell type labels for alignment
   - Ideal when ground truth cell type annotations are available
   - Provides the most direct supervision signal

2. **Optimal Transport (OT) - Paired PoE** 🔄
   - Uses direct cell-to-cell transport plans from optimal transport algorithms
   - Best for datasets with known cell correspondences
   - Enables precise cell-level alignment

3. **Optimal Transport (OT) - Cluster-based PoE** 🧩
   - Combines transport plans with automated cluster matching
   - Uses entropy-based resolution optimization for clustering
   - Applies Hungarian algorithm for optimal cluster correspondence
   - Ideal for datasets without direct cell correspondences but with similar cell type distributions

The method automatically selects the appropriate PoE variant based on the provided inputs, with label-based PoE taking priority when both labels and transport plans are available.

### When to Use Each Variant

- **Use Label-based PoE** when you have reliable cell type annotations for both datasets
- **Use Paired PoE** when you have computed cell-to-cell correspondences (e.g., from trajectory analysis or temporal data)
- **Use Cluster-based PoE** when you have transport plans but no direct cell correspondences (most common OT scenario)


## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.9 or newer installed on your system.

### PyTorch installation

We strongly recommend using spVIPES with GPU acceleration. In Linux, check your NVIDIA drivers running:

```bash
nvidia-smi
```

You can then install a compatible PyTorch version from https://pytorch.org/get-started/previous-versions/. For example if your CUDA drivers are version 11.3, you should install PyTorch v1.12.1 with the following command:

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

Make sure your torch installation can see your CUDA device to take full advantage of GPU acceleration by running:

```python
import torch

torch.cuda.is_available()
```

This should return `True` if everything is installed correctly.

### spVIPES installation

To install spVIPES:

1. Install the latest release of `spVIPES` from `PyPI <https://pypi.org/project/spVIPES/>`\_:

```bash
pip install spVIPES
```

2. Install the latest development version:

```bash
pip install git+https://github.com/nrclaudio/spVIPES.git@main
```

## Usage

### Basic Integration with Labels

```python
import spVIPES

# Setup with cell type labels
spVIPES.model.setup_anndata(
    adata, 
    groups_key="dataset", 
    label_key="cell_type"
)

# Train the model
model = spVIPES.model(adata)
model.train()
```

### Optimal Transport Integration

#### Paired PoE (Direct Cell Correspondences)

```python
# For datasets with known cell-to-cell correspondences
spVIPES.model.setup_anndata(
    adata, 
    groups_key="dataset", 
    transport_plan_key="transport_matrix",
    match_clusters=False  # Use direct cell pairing
)
```

#### Cluster-based PoE (Automatic Cluster Matching)

```python
# For datasets without direct correspondences
spVIPES.model.setup_anndata(
    adata, 
    groups_key="dataset", 
    transport_plan_key="transport_matrix",
    match_clusters=True  # Enable cluster-based matching
)
```

The optimal transport variants are particularly useful when:
- You have computed transport plans using external OT algorithms (e.g., from `scot`, `moscot`, or custom implementations)
- You want to leverage cellular similarity information for integration
- You need to integrate datasets with complex correspondence patterns

## Tutorials

To get started, please refer to the [basic spVIPES tutorial](docs/notebooks/Tutorial.ipynb).

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/nrclaudio/spVIPES/issues
[changelog]: https://spVIPES.readthedocs.io/latest/changelog.html
[link-docs]: https://spVIPES.readthedocs.io
[link-api]: https://spVIPES.readthedocs.io/latest/api.html
