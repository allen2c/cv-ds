import functools
import json
import logging
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from mdc_ds.types.dataset_details import DatasetDetails


logger = logging.getLogger(__name__)


def get_datasets_details() -> Dict[str, "DatasetDetails"]:
    from pathlib import Path
    from typing import List

    from pydantic import TypeAdapter

    from mdc_ds.types.dataset_details import DatasetDetails

    try:
        data: list = json.loads(
            Path(__file__).parent.joinpath("cached_datasets.json").read_bytes()
        )
    except FileNotFoundError:
        logger.error("cached_datasets.json not found")
        return {}

    datasets = TypeAdapter(List[DatasetDetails]).validate_python(data)
    return {dataset.id: dataset for dataset in datasets}


def get_dataset_details(id: str) -> Optional["DatasetDetails"]:
    ds_map = get_datasets_details()
    if id in ds_map:
        return ds_map[id]

    for ds in ds_map.values():
        if ds.slug == id:
            return ds
        elif ds.name == id:
            return ds
        elif ds.name.replace(" ", "_").lower() == id.replace(" ", "_").lower():
            return ds

    return None


@functools.cache
def retrieve_dataset_details(id: str) -> "DatasetDetails":
    if ds := get_dataset_details(id):
        return ds
    else:
        raise ValueError(f"Dataset with input {id} not found")
