# spVIPES

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/nrclaudio/spVIPES/test.yaml?branch=main
[link-tests]: https://github.com/nrclaudio/spVIPES/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/spVIPES

Shared-private Variational Inference with Product of Experts and Supervision

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.9 or newer installed on your system.

To install spVIPES:

<!--
1) Install the latest release of `spVIPES` from `PyPI <https://pypi.org/project/spVIPES/>`_:

```bash
pip install spVIPES
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/nrclaudio/spVIPES.git@main
```

Sometimes your CUDA drivers might be too old for the latest release of PyTorch. Make sure your torch installation can see your CUDA device to take full advantage of GPU acceleration by running:

```python
import torch

torch.cuda.is_available()
```

This should return `True` if everything is fine. To solve the error with CUDA drivers being too old you can search for a compatible release here: https://pytorch.org/get-started/previous-versions/.

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/nrclaudio/spVIPES/issues
[changelog]: https://spVIPES.readthedocs.io/latest/changelog.html
[link-docs]: https://spVIPES.readthedocs.io
[link-api]: https://spVIPES.readthedocs.io/latest/api.html
