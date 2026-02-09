import json
import logging
import os
from pathlib import Path
from typing import Any, Generator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mdc_ds import DEFAULT_MDC_DOWNLOADS_CACHE, MDC_API_KEY_NAME, MDC_CACHE_NAME
from mdc_ds.types.dataset_details import DatasetDetails
from mdc_ds.types.download_session import DownloadSession

logger = logging.getLogger(__name__)

error_api_key_missing_msg = (
    "The API key for Mozilla Data Collective is not set. "
    + "Please provide it as an argument or "
    + f"set the `{MDC_API_KEY_NAME}` environment variable."
)


class BearerAuth(httpx.Auth):
    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, None, None]:
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


class MozillaDataCollectiveClient(httpx.Client):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: httpx.URL | str = "https://datacollective.mozillafoundation.org/api",
        cache_dir: Path | str | None = None,
        **kwargs: Any,
    ):
        if api_key is None:
            if not (api_key := os.getenv(MDC_API_KEY_NAME)):
                raise ValueError(error_api_key_missing_msg)
        if cache_dir is None:
            cache_dir = os.getenv(MDC_CACHE_NAME, None) or DEFAULT_MDC_DOWNLOADS_CACHE
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        auth = BearerAuth(api_key)
        headers = json.loads(json.dumps(kwargs.pop("headers", None) or {}))

        # Set default timeout for downloads (5 minutes total, 30s connect, 60s read)
        default_timeout = httpx.Timeout(timeout=300.0, connect=30.0, read=60.0)
        timeout = kwargs.pop("timeout", default_timeout)

        super().__init__(
            base_url=base_url, auth=auth, headers=headers, timeout=timeout, **kwargs
        )

    def get_dataset_details(self, dataset_id: str) -> DatasetDetails:
        """Retrieves the details of a specific dataset."""

        from mdc_ds.registry import get_dataset_details

        if cached_ds := get_dataset_details(dataset_id):
            logger.debug(f"Dataset input {dataset_id} found in cache")
            return cached_ds

        response = self.get(f"/datasets/{dataset_id}")
        response.raise_for_status()
        return DatasetDetails.model_validate(response.json())

    def get_dataset_download_session(self, dataset_id: str) -> DownloadSession:
        """Creates a download session and returns the dataset's download URL for direct download from storage. The user must have previously agreed to the dataset's terms of use through the web interface."""  # noqa: E501

        from mdc_ds.registry import get_dataset_details

        if not (ds := get_dataset_details(dataset_id)):
            ds = self.get_dataset_details(dataset_id)

        logger.debug(f"Getting download session for dataset '{ds.slug}'")

        response = self.post(f"/datasets/{ds.id}/download")
        response.raise_for_status()
        return DownloadSession.model_validate(response.json())

    def download_dataset(self, dataset_id: str) -> Path:
        ds_details = self.get_dataset_details(dataset_id)
        cache_filepath = self.cache_dir.joinpath(f"{ds_details.id}")

        if cache_filepath.is_file():
            logger.debug(f"Dataset {ds_details.slug} found in cache")
            return cache_filepath
        else:
            download_session = self.get_dataset_download_session(ds_details.id)
            downloaded_filepath = self.download_dataset_session(
                download_session, cache_filepath
            )
            alias_filepath = self.cache_dir.joinpath(download_session.filename)
            alias_filepath.unlink(missing_ok=True)
            os.symlink(downloaded_filepath, alias_filepath)

            return cache_filepath

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(
            (
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            )
        ),
        reraise=True,
    )
    def download_dataset_session(
        self, download_session: DownloadSession, output: Path | str
    ) -> Path:
        """Downloads a dataset file from the provided URL and returns the path to the downloaded file.

        Supports resumable downloads by checking for existing temporary files and using HTTP Range requests.
        """  # noqa: E501

        from tqdm import tqdm

        url = download_session.downloadUrl
        output = Path(output)
        temp_output = output.with_suffix(".tmp")

        # Check if we can resume from existing download
        resume_from_byte = 0
        if temp_output.exists():
            resume_from_byte = temp_output.stat().st_size
            logger.debug(
                f"Found existing temp file, resuming from byte {resume_from_byte}"
            )

        # Get file size and check if server supports range requests
        try:
            head_response = self.head(url)
            total_size = int(head_response.headers.get("content-length", 0))
            supports_range = head_response.headers.get("accept-ranges") == "bytes"
        except Exception:
            # If HEAD request fails, fall back to GET request info
            total_size = download_session.sizeBytes or 0
            supports_range = False
            logger.debug("HEAD request failed, assuming no range support")

        # If we have a partial file but server doesn't support range, start over
        if resume_from_byte > 0 and not supports_range:
            logger.warning("Server doesn't support range requests, restarting download")
            temp_output.unlink(missing_ok=True)
            resume_from_byte = 0

        # If we're resuming, validate the partial file isn't larger than total
        if resume_from_byte >= total_size and total_size > 0:
            logger.warning(
                "Partial file seems complete or corrupted, restarting download"
            )
            temp_output.unlink(missing_ok=True)
            resume_from_byte = 0

        logger.debug(f"Downloading '{url}' to '{output}' (temp: '{temp_output}')")
        logger.debug(f"Resume from byte: {resume_from_byte}, Total size: {total_size}")

        try:
            # Prepare headers for range request if resuming
            headers = {}
            if resume_from_byte > 0:
                headers["Range"] = f"bytes={resume_from_byte}-"

            with self.stream("GET", url, auth=None, headers=headers) as response:
                if resume_from_byte > 0:
                    # Expect 206 Partial Content for range requests
                    if response.status_code == 206:
                        logger.debug("Successfully resumed partial download")
                    elif response.status_code == 200:
                        # Server ignored range header, restart download
                        logger.warning(
                            "Server ignored range request, restarting download"
                        )
                        temp_output.unlink(missing_ok=True)
                        resume_from_byte = 0
                    else:
                        response.raise_for_status()
                else:
                    response.raise_for_status()

                # Get actual total size from response if not known
                if total_size == 0:
                    content_range = response.headers.get("content-range")
                    if content_range:
                        # Parse "bytes START-END/TOTAL"
                        total_part = content_range.split("/")[-1]
                        if total_part != "*":
                            total_size = int(total_part)
                    else:
                        total_size = int(response.headers.get("content-length", 0))

                # Open file in append mode if resuming, write mode if starting fresh
                file_mode = "ab" if resume_from_byte > 0 else "wb"

                with (
                    open(temp_output, file_mode) as f,
                    tqdm(
                        desc=f"Downloading {download_session.filename}",
                        total=total_size,
                        initial=resume_from_byte,  # Start progress bar from resume point  # noqa: E501
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as pbar,
                ):
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Verify download completed
            final_size = temp_output.stat().st_size
            if total_size > 0 and final_size != total_size:
                raise ValueError(
                    f"Download incomplete: got {final_size} bytes, "
                    + f"expected {total_size}"
                )

            # Atomic move: only replace final file after successful download
            temp_output.replace(output)
            logger.info(f"Downloaded dataset to {download_session.filename}")
            return output

        except Exception:
            # Clean up temporary file on failure
            if temp_output.exists():
                temp_output.unlink(missing_ok=True)
            raise


class MozillaDataCollectiveAsyncClient(httpx.AsyncClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: httpx.URL | str = "https://datacollective.mozillafoundation.org/api",
        cache_dir: Path | str | None = None,
        **kwargs: Any,
    ):
        if api_key is None:
            if not (api_key := os.getenv(MDC_API_KEY_NAME)):
                raise ValueError(error_api_key_missing_msg)
        if cache_dir is None:
            cache_dir = os.getenv(MDC_CACHE_NAME, None) or DEFAULT_MDC_DOWNLOADS_CACHE
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        auth = BearerAuth(api_key)
        headers = json.loads(json.dumps(kwargs.pop("headers", None) or {}))

        # Set default timeout for downloads (5 minutes total, 30s connect, 60s read)
        default_timeout = httpx.Timeout(timeout=300.0, connect=30.0, read=60.0)
        timeout = kwargs.pop("timeout", default_timeout)

        super().__init__(
            base_url=base_url, auth=auth, headers=headers, timeout=timeout, **kwargs
        )

    async def get_dataset_details(self, dataset_id: str) -> DatasetDetails:
        """Retrieves the details of a specific dataset."""

        from mdc_ds.registry import get_dataset_details

        if cached_ds := get_dataset_details(dataset_id):
            logger.debug(f"Dataset input {dataset_id} found in cache")
            return cached_ds

        response = await self.get(f"/datasets/{dataset_id}")
        response.raise_for_status()
        return DatasetDetails.model_validate(response.json())

    async def get_dataset_download_session(self, dataset_id: str) -> DownloadSession:
        """Creates a download session and returns the dataset's download URL for direct download from storage. The user must have previously agreed to the dataset's terms of use through the web interface."""  # noqa: E501

        from mdc_ds.registry import get_dataset_details

        if not (ds := get_dataset_details(dataset_id)):
            ds = await self.get_dataset_details(dataset_id)

        response = await self.post(f"/datasets/{ds.id}/download")
        response.raise_for_status()
        return DownloadSession.model_validate(await response.json())

    async def download_dataset(self, dataset_id: str) -> Path:
        ds_details = await self.get_dataset_details(dataset_id)
        cache_filepath = self.cache_dir.joinpath(f"{ds_details.id}")

        if cache_filepath.is_file():
            logger.debug(f"Dataset {ds_details.slug} found in cache")
            return cache_filepath
        else:
            download_session = await self.get_dataset_download_session(ds_details.id)
            downloaded_filepath = await self.download_dataset_session(
                download_session, cache_filepath
            )
            alias_filepath = self.cache_dir.joinpath(download_session.filename)
            alias_filepath.unlink(missing_ok=True)
            os.symlink(downloaded_filepath, alias_filepath)

            return cache_filepath

    async def download_dataset_session(
        self, download_session: DownloadSession, output: Path | str
    ) -> Path:
        """Downloads a dataset file from the provided URL and returns the path to the downloaded file.

        Supports resumable downloads by checking for existing temporary files and using HTTP Range requests.
        """  # noqa: E501
        import aiofiles
        from tqdm.asyncio import tqdm as async_tqdm

        url = download_session.downloadUrl
        download_filepath = Path(output)
        temp_filepath = download_filepath.with_suffix(".tmp")

        # Check if we can resume from existing download
        resume_from_byte = 0
        if temp_filepath.exists():
            resume_from_byte = temp_filepath.stat().st_size
            logger.debug(
                f"Found existing temp file, resuming from byte {resume_from_byte}"
            )

        # Get file size and check if server supports range requests
        try:
            head_response = await self.head(url)
            total_size = int(head_response.headers.get("content-length", 0))
            supports_range = head_response.headers.get("accept-ranges") == "bytes"
        except Exception:
            # If HEAD request fails, fall back to GET request info
            total_size = download_session.sizeBytes or 0
            supports_range = False
            logger.debug("HEAD request failed, assuming no range support")

        # If we have a partial file but server doesn't support range, start over
        if resume_from_byte > 0 and not supports_range:
            logger.warning("Server doesn't support range requests, restarting download")
            temp_filepath.unlink(missing_ok=True)
            resume_from_byte = 0

        # If we're resuming, validate the partial file isn't larger than total
        if resume_from_byte >= total_size and total_size > 0:
            logger.warning(
                "Partial file seems complete or corrupted, restarting download"
            )
            temp_filepath.unlink(missing_ok=True)
            resume_from_byte = 0

        logger.info(f"Downloading dataset {download_session.filename}")
        logger.debug(f"Resume from byte: {resume_from_byte}, Total size: {total_size}")

        try:
            # Prepare headers for range request if resuming
            headers = {}
            if resume_from_byte > 0:
                headers["Range"] = f"bytes={resume_from_byte}-"

            async with self.stream("GET", url, auth=None, headers=headers) as response:
                if resume_from_byte > 0:
                    # Expect 206 Partial Content for range requests
                    if response.status_code == 206:
                        logger.debug("Successfully resumed partial download")
                    elif response.status_code == 200:
                        # Server ignored range header, restart download
                        logger.warning(
                            "Server ignored range request, restarting download"
                        )
                        temp_filepath.unlink(missing_ok=True)
                        resume_from_byte = 0
                    else:
                        response.raise_for_status()
                else:
                    response.raise_for_status()

                # Get actual total size from response if not known
                if total_size == 0:
                    content_range = response.headers.get("content-range")
                    if content_range:
                        # Parse "bytes START-END/TOTAL"
                        total_part = content_range.split("/")[-1]
                        if total_part != "*":
                            total_size = int(total_part)
                    else:
                        total_size = int(response.headers.get("content-length", 0))

                # Open file in append mode if resuming, write mode if starting fresh
                file_mode = "ab" if resume_from_byte > 0 else "wb"

                # Async file writing with progress tracking
                async with aiofiles.open(temp_filepath, file_mode) as f:
                    with async_tqdm(
                        desc=f"Downloading {download_session.filename}",
                        total=total_size,
                        initial=resume_from_byte,  # Start progress bar from resume point  # noqa: E501
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as pbar:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            await f.write(chunk)
                            pbar.update(len(chunk))

            # Verify download completed
            final_size = temp_filepath.stat().st_size
            if total_size > 0 and final_size != total_size:
                raise ValueError(
                    f"Download incomplete: got {final_size} bytes, "
                    + f"expected {total_size}"
                )

            # Atomic move: only replace final file after successful download
            temp_filepath.replace(download_filepath)

            logger.info(f"Downloaded dataset to {download_filepath}")
            return download_filepath

        except Exception:
            # Clean up temporary file on failure
            if temp_filepath.exists():
                temp_filepath.unlink(missing_ok=True)
            raise
