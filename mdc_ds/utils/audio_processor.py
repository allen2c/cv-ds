import io
import tarfile
from typing import Optional

from pydub import AudioSegment


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
