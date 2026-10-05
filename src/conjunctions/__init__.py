from importlib.metadata import version

from .cli import main
from .core import find_conjunctions

__version__ = version("conjunctions")

__all__ = ["find_conjunctions", "main", "__version__"]
