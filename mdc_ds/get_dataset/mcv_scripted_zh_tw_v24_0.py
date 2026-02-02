import io
import os
import tarfile
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from google_language_support import LanguageCodes
from pydub import AudioSegment

from mdc_ds.types.feature import feature

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

    if cache_path.exists() and cache_path.is_dir():
        return DatasetDict.load_from_disk(str(cache_path))[split]

    client = MozillaDataCollectiveClient()
    ds_details = client.get_dataset_details(slug_name)
    downloaded_filepath = client.download_dataset(ds_details.id)

    tar_root = Path("cv-corpus-24.0-2025-12-05/zh-TW")
    train_manifests: List[dict] = []
    test_manifests: List[dict] = []

    # 1. Parse TSV files to build metadata lists (Lightweight)
    # We open tar only to get the TSV files initially
    print("Parsing TSV files to build metadata lists")
    with tarfile.open(downloaded_filepath, "r:gz") as tar:
        train_tar_filepath = tar.extractfile(str(tar_root.joinpath("train.tsv")))
        test_tar_filepath = tar.extractfile(str(tar_root.joinpath("test.tsv")))
        if not train_tar_filepath:
            raise ValueError("Train tar file not found")
        if not test_tar_filepath:
            raise ValueError("Test tar file not found")

        train_df = pd.read_csv(train_tar_filepath, sep="\t")
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

    # 2. Create initial Datasets (Metadata only, no heavy processing yet)
    # Using from_list is faster than generator for in-memory data
    train_dataset = Dataset.from_list(train_manifests)
    test_dataset = Dataset.from_list(test_manifests)

    # 3. Apply Audio Processing in Parallel using .map()
    # Initialize the processor with the path to the tar file
    audio_processor = AudioProcessor(str(downloaded_filepath))

    print("Processing train dataset audio in parallel...")
    train_dataset = train_dataset.map(
        audio_processor,
        num_proc=4,
        remove_columns=[
            "audio_path"
        ],  # We don't need the path anymore after extraction
        desc="Decoding train audio",
    )

    print("Processing test dataset audio in parallel...")
    test_dataset = test_dataset.map(
        audio_processor,
        num_proc=4,
        remove_columns=["audio_path"],
        desc="Decoding test audio",
    )

    # 4. Cast to target features (Optional but recommended to match original intent)
    # This ensures the 'audio' column matches the type defined in 'feature'
    train_dataset = train_dataset.cast(feature)
    test_dataset = test_dataset.cast(feature)

    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )

    dataset_dict.save_to_disk(str(cache_path))
    return DatasetDict.load_from_disk(str(cache_path))[split]
