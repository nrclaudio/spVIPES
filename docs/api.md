# API Reference

## Core Classes

### spVIPES Model

The main model class for shared-private variational inference.

```{eval-rst}
.. currentmodule:: spVIPES

.. autosummary::
    :toctree: generated
    :template: class.rst

    model.spvipes.spVIPES

.. autoclass:: spVIPES.model.spvipes.spVIPES
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

### spVIPES Module

The PyTorch Lightning module implementing the variational autoencoder.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: class.rst

    module.spVIPESmodule.spVIPESmodule

.. autoclass:: spVIPES.module.spVIPESmodule.spVIPESmodule
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

## Neural Network Components

### Encoder

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: class.rst

    nn.networks.Encoder

.. autoclass:: spVIPES.nn.networks.Encoder
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

### Decoder

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: class.rst

    nn.networks.LinearDecoderSPVIPE

.. autoclass:: spVIPES.nn.networks.LinearDecoderSPVIPE
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

## Data Management

### AnnData Manager

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: class.rst

    data._manager.AnnDataManager

.. autoclass:: spVIPES.data._manager.AnnDataManager
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

### Data Preparation

```{eval-rst}
.. autosummary::
    :toctree: generated

    data.prepare_adatas.prepare_adatas

.. autofunction:: spVIPES.data.prepare_adatas.prepare_adatas
```

## Data Loaders

### Concatenated Data Loader

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: class.rst

    dataloaders._concat_dataloader.ConcatDataLoader

.. autoclass:: spVIPES.dataloaders._concat_dataloader.ConcatDataLoader
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

### AnnData Loader

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: class.rst

    dataloaders._ann_dataloader.AnnDataLoader

.. autoclass:: spVIPES.dataloaders._ann_dataloader.AnnDataLoader
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```
