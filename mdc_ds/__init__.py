from pathlib import Path
from typing import Final

from .get_dataset import get_dataset

__version__: Final[str] = "0.2.0"

MDC_API_KEY_NAME: Final[str] = "MDC_API_KEY"
MDC_CACHE_NAME: Final[str] = "MDC_CACHE"
MDC_DATASETS_CACHE_NAME: Final[str] = "MDC_DATASETS_CACHE"
DEFAULT_MDC_DOWNLOADS_CACHE: Final[Path] = Path(
    "~/.cache/huggingface/datasets/downloads"
).expanduser()
DEFAULT_MDC_DATASETS_CACHE: Final[Path] = Path(
    "~/.cache/huggingface/datasets"
).expanduser()

__all__ = ["get_dataset"]
