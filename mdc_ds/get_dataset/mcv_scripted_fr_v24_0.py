import csv
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import List, Literal, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from google_language_support import LanguageCodes

from mdc_ds.types.feature import feature
from mdc_ds.types.manifest_item import ManifestItem
from mdc_ds.utils.audio_processor import AudioProcessor
from mdc_ds.utils.shard_ds import shard_ds

logger = logging.getLogger(__name__)

slug_name = "mcv-scripted-fr-v24.0"


def get_metadata(
    downloaded_filepath: Path | str,
) -> Tuple[List[ManifestItem], List[ManifestItem], List[ManifestItem]]:

    tar_root = Path("cv-corpus-24.0-2025-12-05/fr")
    train_manifests: List[ManifestItem] = []
    dev_manifests: List[ManifestItem] = []
    test_manifests: List[ManifestItem] = []

    # 1. Parse TSV files to build metadata lists (Lightweight)
    # We open tar only to get the TSV files initially
    logger.debug("Parsing TSV files to build metadata lists")
    with tarfile.open(downloaded_filepath, "r:gz") as tar:
        train_tar_filepath = tar.extractfile(str(tar_root.joinpath("train.tsv")))
        dev_tar_filepath = tar.extractfile(str(tar_root.joinpath("dev.tsv")))
        test_tar_filepath = tar.extractfile(str(tar_root.joinpath("test.tsv")))
        if not train_tar_filepath or not dev_tar_filepath or not test_tar_filepath:
            raise ValueError("Train, dev, or test tar file not found")

        train_df = pd.read_csv(
            train_tar_filepath, sep="\t", dtype=str, quoting=csv.QUOTE_NONE
        )
        dev_df = pd.read_csv(
            dev_tar_filepath, sep="\t", dtype=str, quoting=csv.QUOTE_NONE
        )
        test_df = pd.read_csv(
            test_tar_filepath, sep="\t", dtype=str, quoting=csv.QUOTE_NONE
        )

        # Convert to list of dicts immediately for Dataset.from_list
        for train_row in train_df.itertuples(index=False):
            sentence = str(train_row.sentence) if train_row.sentence is not None else ""  # type: ignore  # noqa: E501
            path = str(train_row.path) if train_row.path is not None else ""  # type: ignore  # noqa: E501

            if (
                not sentence
                or sentence.lower() == "nan"
                or not path
                or path.lower() == "nan"
            ):
                logger.warning(f"Skipping dirty row: {train_row}")
                continue

            full_audio_path = str(tar_root.joinpath("clips").joinpath(path))
            train_manifests.append(
                {
                    "audio_path": full_audio_path,
                    "text": sentence,
                    "language": LanguageCodes.FRENCH,
                }
            )

        for dev_row in dev_df.itertuples(index=False):
            sentence = str(dev_row.sentence) if dev_row.sentence is not None else ""  # type: ignore  # noqa: E501
            path = str(dev_row.path) if dev_row.path is not None else ""  # type: ignore  # noqa: E501
            if (
                not sentence
                or sentence.lower() == "nan"
                or not path
                or path.lower() == "nan"
            ):
                logger.warning(f"Skipping dirty row: {dev_row}")
                continue

            full_audio_path = str(tar_root.joinpath("clips").joinpath(path))
            dev_manifests.append(
                {
                    "audio_path": full_audio_path,
                    "text": sentence,
                    "language": LanguageCodes.FRENCH,
                }
            )

        for test_row in test_df.itertuples(index=False):
            sentence = str(test_row.sentence) if test_row.sentence is not None else ""  # type: ignore  # noqa: E501
            path = str(test_row.path) if test_row.path is not None else ""  # type: ignore  # noqa: E501
            if (
                not sentence
                or sentence.lower() == "nan"
                or not path
                or path.lower() == "nan"
            ):
                logger.warning(f"Skipping dirty row: {test_row}")
                continue

            full_audio_path = str(tar_root.joinpath("clips").joinpath(path))
            test_manifests.append(
                {
                    "audio_path": full_audio_path,
                    "text": sentence,
                    "language": LanguageCodes.FRENCH,
                }
            )

    return train_manifests, dev_manifests, test_manifests


def get_dataset(
    name: Literal["mcv-scripted-fr-v24.0"],
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

    train_manifests, dev_manifests, test_manifests = get_metadata(downloaded_filepath)

    # 2. Create initial Datasets (Metadata only, no heavy processing yet)
    # Using from_list is faster than generator for in-memory data
    train_dataset = Dataset.from_list(train_manifests)  # type: ignore
    dev_dataset = Dataset.from_list(dev_manifests)  # type: ignore
    test_dataset = Dataset.from_list(test_manifests)  # type: ignore

    # 3. Apply Audio Processing in Parallel using .map()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        processed_shard_paths: List[Path] = []
        sharded_ds_items = shard_ds(train_dataset)
        for idx, sharded_ds in enumerate(sharded_ds_items):
            sharded_ds = sharded_ds.map(
                AudioProcessor(str(downloaded_filepath)),
                num_proc=16,
                remove_columns=[
                    "audio_path"
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
        train_dataset = concatenate_datasets(processed_shards)

        logger.debug("Processing dev dataset audio in parallel...")
        dev_dataset = dev_dataset.map(
            AudioProcessor(str(downloaded_filepath)),
            num_proc=16,
            remove_columns=["audio_path"],
            desc="Decoding dev audio",
        )

        logger.debug("Processing test dataset audio in parallel...")
        test_dataset = test_dataset.map(
            AudioProcessor(str(downloaded_filepath)),
            num_proc=16,
            remove_columns=["audio_path"],
            desc="Decoding test audio",
        )

        # 4. Cast to target features (Optional but recommended to match original intent)
        # This ensures the 'audio' column matches the type defined in 'feature'
        train_dataset = train_dataset.cast(feature)
        dev_dataset = dev_dataset.cast(feature)
        test_dataset = test_dataset.cast(feature)

        dataset_dict = DatasetDict(
            {
                "train": train_dataset,
                "dev": dev_dataset,
                "test": test_dataset,
            }
        )

        dataset_dict.save_to_disk(str(cache_path))

    return DatasetDict.load_from_disk(str(cache_path))[split]
