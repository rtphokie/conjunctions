from importlib.metadata import version

from .cli import main
from .core import find_conjunctions, find_oppositions

__version__ = version("conjunctions")

__all__ = ["find_conjunctions", "find_oppositions", "main", "__version__"]
