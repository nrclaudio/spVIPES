<div align="center">

# spVIPES

**Shared-private Variational Inference with Product of Experts and Supervision**

[![Tests][badge-tests]][link-tests]
[![Python][badge-python]][link-python]
[![PyPI][badge-pypi]][link-pypi]
[![Documentation][badge-docs]][link-docs]

</div>

---

## About

spVIPES enables robust integration of multi-group single-cell datasets through a principled shared-private latent space decomposition. The method leverages a Product of Experts (PoE) framework to learn both shared biological signals common across datasets and private representations capturing group-specific variations.

### Integration Strategies

spVIPES provides three complementary approaches for dataset alignment:

| Method                   | Description                                               | Best Use Case                                       |
| ------------------------ | --------------------------------------------------------- | --------------------------------------------------- |
| **Label-based PoE**      | Uses cell type annotations for direct supervision         | High-quality cell type labels available             |
| **OT Paired PoE**        | Direct cell-to-cell correspondences via optimal transport | Known cellular correspondences (e.g., time series)  |
| **OT Cluster-based PoE** | Automated cluster matching with transport plans           | Similar cell populations, no direct correspondences |

> **Note:** The method automatically selects the most appropriate strategy based on available annotations and transport information.

## Installation

### Requirements

-   Python 3.9+
-   PyTorch (GPU support strongly recommended)

### Quick Install

Install the latest stable release from PyPI:

```bash
pip install spVIPES
```

For the development version:

```bash
pip install git+https://github.com/nrclaudio/spVIPES.git@main
```

### GPU Setup (Recommended)

For optimal performance, ensure CUDA-compatible PyTorch is installed:

```bash
# Check GPU availability
nvidia-smi

# Install PyTorch with CUDA support (example for CUDA 11.3)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

> See [PyTorch installation guide](https://pytorch.org/get-started/locally/) for version-specific instructions.

## Quick Start

### Basic Workflow

```python
import spVIPES
import scanpy as sc

# Load your multi-group dataset
adata = sc.read_h5ad("data.h5ad")

# Configure integration strategy
spVIPES.model.setup_anndata(
    adata,
    groups_key="dataset",
    label_key="cell_type",  # Optional: for supervised integration
)

# Initialize and train model
model = spVIPES.model(adata)
model.train(max_epochs=200)

# Extract integrated representations
latent = model.get_latent_representation()
adata.obsm["X_spVIPES"] = latent
```

### Integration Strategies

<details>
<summary><b>📋 Label-based Integration</b></summary>

Use when high-quality cell type annotations are available:

```python
spVIPES.model.setup_anndata(
    adata,
    groups_key="dataset",
    label_key="cell_type",
    batch_key="batch",  # Optional batch correction
)
```

</details>

<details>
<summary><b>🔄 Optimal Transport: Paired Cells</b></summary>

For datasets with known cell-to-cell correspondences:

```python
# Assumes transport plan stored in adata.uns["transport_plan"]
spVIPES.model.setup_anndata(
    adata,
    groups_key="dataset",
    transport_plan_key="transport_plan",
    match_clusters=False,
)
```

</details>

<details>
<summary><b>🧩 Optimal Transport: Cluster Matching</b></summary>

For automatic cluster-based alignment:

```python
spVIPES.model.setup_anndata(
    adata,
    groups_key="dataset",
    transport_plan_key="transport_plan",
    match_clusters=True,  # Enables automated cluster matching
)
```

</details>

### Advanced Configuration

```python
# Custom model parameters
model = spVIPES.model(
    adata,
    n_dimensions_shared=25,  # Shared latent dimensions
    n_dimensions_private=10,  # Private latent dimensions
    n_hidden=128,  # Hidden layer size
    dropout_rate=0.1,  # Regularization
)

# Training with custom settings
model.train(
    max_epochs=300, batch_size=512, early_stopping=True, check_val_every_n_epoch=10
)
```

## Documentation & Tutorials

📚 **Getting Started**

-   [Basic Tutorial](docs/notebooks/Tutorial.ipynb) — Complete walkthrough of spVIPES functionality
-   [API Documentation][link-api] — Comprehensive API reference

## Support & Community

💬 **Get Help**

-   [Issue Tracker][issue-tracker] — Report bugs and request features

## Citation

If you use spVIPES in your research, please cite:

```bibtex
@article{spVIPES2023,
  title={Integrative learning of disentangled representations},
  author={C. Novella-Rausell, D.J.M Peters and A. Mahfouz},
  journal={bioRxiv},
  year={2023},
  doi={10.1101/2023.11.07.565957},
  url={https://www.biorxiv.org/content/10.1101/2023.11.07.565957v1}
}
```

**Paper**: [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.11.07.565957v1)

---

<!-- Badge references -->

[badge-tests]: https://img.shields.io/github/actions/workflow/status/nrclaudio/spVIPES/test.yaml?branch=main
[badge-python]: https://img.shields.io/pypi/pyversions/spVIPES
[badge-pypi]: https://img.shields.io/pypi/v/spVIPES
[badge-docs]: https://readthedocs.org/projects/spvipes/badge/?version=latest
[link-tests]: https://github.com/nrclaudio/spVIPES/actions/workflows/test.yml
[link-python]: https://pypi.org/project/spVIPES
[link-pypi]: https://pypi.org/project/spVIPES
[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/nrclaudio/spVIPES/issues
[changelog]: https://spVIPES.readthedocs.io/latest/changelog.html
[link-docs]: https://spvipes.readthedocs.io/en/latest/
[link-api]: https://spvipes.readthedocs.io/en/latest/api.html
