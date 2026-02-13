import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import List, Literal

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from google_language_support import LanguageCodes

from mdc_ds.types.feature import feature
from mdc_ds.types.manifest_item import ManifestItemWithClientId
from mdc_ds.utils.audio_processor import AudioProcessor
from mdc_ds.utils.shard_ds import shard_ds
from mdc_ds.utils.split_dataset_balanced_by_speaker import (
    split_dataset_balanced_by_speaker,
)

logger = logging.getLogger(__name__)

slug_name = "common-voice-v24-english-en-au-subset-fo-0447e8a6"


def get_metadata(downloaded_filepath: Path | str) -> List[ManifestItemWithClientId]:
    tar_root = Path("commonvoice-v24_en-AU")
    train_manifests: List[ManifestItemWithClientId] = []

    # 1. Parse TSV files to build metadata lists (Lightweight)
    # We open tar only to get the TSV files initially
    logger.debug("Parsing CSV files to build metadata lists")
    with tarfile.open(downloaded_filepath, "r:gz") as tar:
        might_manifest_filepaths = [
            m.name
            for m in tar.getmembers()
            if m.name.lower().endswith((".csv", ".json", ".tsv"))
        ]
        try:
            train_tar_filepath = tar.extractfile(
                str(tar_root.joinpath("commonvoice-v24_en-AU.csv"))
            )
            if not train_tar_filepath:
                raise ValueError(
                    "Train tar file not found, "
                    f"manifest files might be in {might_manifest_filepaths}"
                )
        except KeyError as e:
            logger.error(
                f"Train tar file not found: {e}, "
                f"manifest files might be in {might_manifest_filepaths}"
            )
            raise e

        train_df = pd.read_csv(train_tar_filepath)

        # Convert to list of dicts immediately for Dataset.from_list
        for train_row in train_df.itertuples(index=False):
            if train_row.sentence is None or train_row.path is None:  # type: ignore
                logger.error(f"Skipping row with empty sentence or path: {train_row}")
                continue

            full_audio_path = str(
                tar_root.joinpath("audio_files").joinpath(train_row.path)  # type: ignore  # noqa: E501
            )
            train_manifests.append(
                {
                    "client_id": train_row.client_id,  # type: ignore
                    "audio_path": full_audio_path,
                    "text": train_row.sentence,  # type: ignore
                    "language": LanguageCodes.ENGLISH,
                }
            )

    return train_manifests


def get_dataset(
    name: Literal["common-voice-v24-english-en-au-subset-fo-0447e8a6"],
    split: Literal["train", "test", "validation"] = "train",
) -> "Dataset":
    from mdc_ds import DEFAULT_MDC_DATASETS_CACHE, MDC_DATASETS_CACHE_NAME
    from mdc_ds.client import MozillaDataCollectiveClient

    cache_path = Path(
        os.getenv(MDC_DATASETS_CACHE_NAME, None) or DEFAULT_MDC_DATASETS_CACHE
    ).joinpath(slug_name)
    cache_path.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and cache_path.is_dir() and any(cache_path.glob("*.json")):
        return DatasetDict.load_from_disk(str(cache_path))[split]

    client = MozillaDataCollectiveClient()
    ds_details = client.get_dataset_details(slug_name)
    downloaded_filepath = client.download_dataset(ds_details.id)

    # 1. Get metadata
    train_manifests = get_metadata(downloaded_filepath)

    # 2. Create initial Datasets (Metadata only, no heavy processing yet)
    # Using from_list is faster than generator for in-memory data
    all_dataset = Dataset.from_list(train_manifests)  # type: ignore

    # Split speakers into train, dev, and test sets in 90/5/5 ratio
    train_ds, dev_ds, test_ds = split_dataset_balanced_by_speaker(
        all_dataset, ratios=(0.9, 0.05, 0.05), client_col="client_id"
    )

    # 3. Apply Audio Processing in Parallel using .map()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        processed_shard_paths: List[Path] = []
        sharded_ds_items = shard_ds(train_ds)
        for idx, sharded_ds in enumerate(sharded_ds_items):
            sharded_ds = sharded_ds.map(
                AudioProcessor(str(downloaded_filepath)),
                num_proc=16,
                remove_columns=[
                    "client_id",
                    "audio_path",
                ],  # We don't need the path anymore after extraction
                desc=f"Decoding train audio shard {idx}/{len(sharded_ds_items)}",
            )
            shared_path = temp_path / f"shard_train_{idx}"
            sharded_ds.save_to_disk(str(shared_path))
            processed_shard_paths.append(shared_path)

            del sharded_ds

        processed_shards: List[Dataset] = [
            Dataset.load_from_disk(str(p)) for p in processed_shard_paths
        ]
        train_ds = concatenate_datasets(processed_shards)

        logger.debug("Processing dev dataset audio in parallel...")
        dev_ds = dev_ds.map(
            AudioProcessor(str(downloaded_filepath)),
            num_proc=16,
            remove_columns=["client_id", "audio_path"],
            desc="Decoding dev audio",
        )

        logger.debug("Processing test dataset audio in parallel...")
        test_ds = test_ds.map(
            AudioProcessor(str(downloaded_filepath)),
            num_proc=16,
            remove_columns=["client_id", "audio_path"],
            desc="Decoding test audio",
        )

        # 4. Cast to target features (Optional but recommended to match original intent)
        # This ensures the 'audio' column matches the type defined in 'feature'
        train_ds = train_ds.cast(feature)
        dev_ds = dev_ds.cast(feature)
        test_ds = test_ds.cast(feature)

        dataset_dict = DatasetDict(
            {
                "train": train_ds,
                "dev": dev_ds,
                "test": test_ds,
            }
        )

        dataset_dict.save_to_disk(str(cache_path))

    return DatasetDict.load_from_disk(str(cache_path))[split]
