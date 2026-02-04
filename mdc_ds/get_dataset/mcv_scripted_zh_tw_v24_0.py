import io
import logging
import os
import tarfile
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import pandas as pd
from datasets import Dataset, DatasetDict
from google_language_support import LanguageCodes
from pydub import AudioSegment

from mdc_ds.types.feature import feature

logger = logging.getLogger(__name__)

slug_name = "mcv-scripted-zh-TW-v24.0"


class AudioProcessor:
    """
    Helper class to handle tarfile opening within worker processes.
    Ensures that the tar file is opened once per process to avoid overhead.
    """

    def __init__(self, tar_path: str):
        self.tar_path = tar_path
        # The tar handle will be initialized lazily in the worker process
        self._tar: Optional[tarfile.TarFile] = None

    def __call__(self, example):
        if self._tar is None:
            self._tar = tarfile.open(self.tar_path, "r:gz")

        audio_path = example["audio_path"]

        # Extract file from tar
        audio_tar_obj = self._tar.extractfile(audio_path)
        if not audio_tar_obj:
            raise ValueError(f"Audio tar object not found: {audio_path}")

        # Audio processing logic (same as original)
        audio_seg: AudioSegment = AudioSegment.from_file(
            io.BytesIO(audio_tar_obj.read())
        )
        audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)
        mp3_io = io.BytesIO()
        audio_seg.export(mp3_io, format="mp3", bitrate="128k")
        audio_bytes = mp3_io.getvalue()

        # Return the processed audio bytes
        return {"audio": audio_bytes}

    def __del__(self):
        if self._tar:
            self._tar.close()


class ManifestItem(TypedDict):
    audio_path: str
    text: str
    language: LanguageCodes


def get_metadata(
    downloaded_filepath: Path | str,
) -> Tuple[List[ManifestItem], List[ManifestItem], List[ManifestItem]]:

    tar_root = Path("cv-corpus-24.0-2025-12-05/zh-TW")
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
            full_audio_path = str(
                tar_root.joinpath("clips").joinpath(train_row.path)  # type: ignore
            )
            train_manifests.append(
                {
                    "audio_path": full_audio_path,
                    "text": train_row.sentence,  # type: ignore
                    "language": LanguageCodes.CHINESE_TRADITIONAL,
                }
            )

        for dev_row in dev_df.itertuples(index=False):
            full_audio_path = str(
                tar_root.joinpath("clips").joinpath(dev_row.path)  # type: ignore
            )
            dev_manifests.append(
                {
                    "audio_path": full_audio_path,
                    "text": dev_row.sentence,  # type: ignore
                    "language": LanguageCodes.CHINESE_TRADITIONAL,
                }
            )

        for test_row in test_df.itertuples(index=False):
            full_audio_path = str(
                tar_root.joinpath("clips").joinpath(test_row.path)  # type: ignore
            )
            test_manifests.append(
                {
                    "audio_path": full_audio_path,
                    "text": test_row.sentence,  # type: ignore
                    "language": LanguageCodes.CHINESE_TRADITIONAL,
                }
            )

    return train_manifests, dev_manifests, test_manifests


def get_dataset(
    name: Literal["mcv-scripted-zh-TW-v24.0"],
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
        num_proc=4,
        remove_columns=[
            "audio_path"
        ],  # We don't need the path anymore after extraction
        desc="Decoding train audio",
    )

    logger.debug("Processing dev dataset audio in parallel...")
    dev_dataset = dev_dataset.map(
        audio_processor,
        num_proc=4,
        remove_columns=["audio_path"],
        desc="Decoding dev audio",
    )

    logger.debug("Processing test dataset audio in parallel...")
    test_dataset = test_dataset.map(
        audio_processor,
        num_proc=4,
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
