import math
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from datasets import Dataset


def shard_ds(ds: "Dataset", max_rows_per_shard: int = 20000) -> List["Dataset"]:
    num_shards = math.ceil(len(ds) / max_rows_per_shard)

    return [
        ds.shard(num_shards=num_shards, index=i, contiguous=True)
        for i in range(num_shards)
    ]
