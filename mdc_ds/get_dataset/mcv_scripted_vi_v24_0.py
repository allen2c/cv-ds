import logging
import os
import tarfile
from pathlib import Path
from typing import List, Literal, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict
from google_language_support import LanguageCodes

from mdc_ds.types.feature import feature
from mdc_ds.types.manifest_item import ManifestItem
from mdc_ds.utils.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

slug_name = "mcv-scripted-vi-v24.0"


def get_metadata(
    downloaded_filepath: Path | str,
) -> Tuple[List[ManifestItem], List[ManifestItem], List[ManifestItem]]:

    tar_root = Path("cv-corpus-24.0-2025-12-05/vi")
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

        train_df = pd.read_csv(train_tar_filepath, sep="\t")
        dev_df = pd.read_csv(dev_tar_filepath, sep="\t")
        test_df = pd.read_csv(test_tar_filepath, sep="\t")

        # Convert to list of dicts immediately for Dataset.from_list
        for train_row in train_df.itertuples(index=False):
            if train_row.sentence is None or train_row.path is None:  # type: ignore
                logger.error(f"Skipping row with empty sentence or path: {train_row}")
                continue

            full_audio_path = str(
                tar_root.joinpath("clips").joinpath(train_row.path)  # type: ignore
            )
            train_manifests.append(
                {
                    "audio_path": full_audio_path,
                    "text": train_row.sentence,  # type: ignore
                    "language": LanguageCodes.VIETNAMESE,
                }
            )

        for dev_row in dev_df.itertuples(index=False):
            if dev_row.sentence is None or dev_row.path is None:  # type: ignore
                logger.error(f"Skipping row with empty sentence or path: {dev_row}")
                continue

            full_audio_path = str(
                tar_root.joinpath("clips").joinpath(dev_row.path)  # type: ignore
            )
            dev_manifests.append(
                {
                    "audio_path": full_audio_path,
                    "text": dev_row.sentence,  # type: ignore
                    "language": LanguageCodes.VIETNAMESE,
                }
            )

        for test_row in test_df.itertuples(index=False):
            if test_row.sentence is None or test_row.path is None:  # type: ignore
                logger.error(f"Skipping row with empty sentence or path: {test_row}")
                continue

            full_audio_path = str(
                tar_root.joinpath("clips").joinpath(test_row.path)  # type: ignore
            )
            test_manifests.append(
                {
                    "audio_path": full_audio_path,
                    "text": test_row.sentence,  # type: ignore
                    "language": LanguageCodes.VIETNAMESE,
                }
            )

    return train_manifests, dev_manifests, test_manifests


def get_dataset(
    name: Literal["mcv-scripted-vi-v24.0"],
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
    # Initialize the processor with the path to the tar file
    audio_processor = AudioProcessor(str(downloaded_filepath))

    logger.debug("Processing train dataset audio in parallel...")
    train_dataset = train_dataset.map(
        audio_processor,
        num_proc=16,
        remove_columns=[
            "audio_path"
        ],  # We don't need the path anymore after extraction
        desc="Decoding train audio",
    )

    logger.debug("Processing dev dataset audio in parallel...")
    dev_dataset = dev_dataset.map(
        audio_processor,
        num_proc=16,
        remove_columns=["audio_path"],
        desc="Decoding dev audio",
    )

    logger.debug("Processing test dataset audio in parallel...")
    test_dataset = test_dataset.map(
        audio_processor,
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
