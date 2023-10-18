"""spVIPES setup file. setup file adapter from the scvi-tools-skeleton repo."""

import logging

from rich.console import Console
from rich.logging import RichHandler

# this import needs to come after prior imports to prevent circular import
from . import data, model, module, nn

__all__ = ["model", "module", "nn", "data"]

# https://github.com/python-poetry/poetry/pull/2366#issuecomment-652418094
# https://github.com/python-poetry/poetry/issues/144#issuecomment-623927302
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "spVIPES"
__version__ = importlib_metadata.version(package_name)

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("spVIPES: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False
