import json
import logging
import os
from pathlib import Path
from typing import Any, Generator

import httpx

from mdc_ds import DEFAULT_MDC_CACHE, MDC_API_KEY_NAME, MDC_CACHE_NAME
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
            cache_dir = os.getenv(MDC_CACHE_NAME, None) or DEFAULT_MDC_CACHE
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        auth = BearerAuth(api_key)
        headers = json.loads(json.dumps(kwargs.pop("headers", None) or {}))

        super().__init__(base_url=base_url, auth=auth, headers=headers, **kwargs)

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

        response = self.post(f"/datasets/{ds.id}/download")
        response.raise_for_status()
        return DownloadSession.model_validate(response.json())

    def download_dataset(
        self, dataset_id: str, *, output: Path | str | None = None
    ) -> Path:
        import shutil

        from mdc_ds.utils.ensure_output import ensure_output

        ds_details = self.get_dataset_details(dataset_id)
        cache_filepath = self.cache_dir.joinpath(f"{ds_details.id}.tar.gz")

        if cache_filepath.is_file():
            logger.debug(f"Dataset {dataset_id} found in cache")
        else:
            download_session = self.get_dataset_download_session(ds_details.id)
            cache_filepath = self.download_dataset_by_url(
                download_session, output=cache_filepath
            )

        if output is None:
            return cache_filepath
        else:
            output = ensure_output(output=output)
            shutil.copy(cache_filepath, output)
            return output

    def download_dataset_by_url(
        self, download_url: DownloadSession | str, *, output: Path | str
    ) -> Path:
        """Downloads a dataset file from the provided URL and returns the path to the downloaded file."""  # noqa: E501

        from tqdm import tqdm

        from mdc_ds.utils.ensure_output import ensure_output

        # Extract URL and determine output path (existing logic)
        url = (
            download_url if isinstance(download_url, str) else download_url.downloadUrl
        )
        output = ensure_output(output=output)

        # Get total size for progress tracking
        total_size = (
            download_url.sizeBytes
            if isinstance(download_url, DownloadSession)
            else None
        )

        # Stream download with progress tracking
        with self.stream("GET", url, auth=None) as response:
            response.raise_for_status()

            # Use response content-length if sizeBytes not available
            if total_size is None:
                total_size = int(response.headers.get("content-length", 0))

            with (
                open(output, "wb") as f,
                tqdm(
                    desc=f"Downloading {output.name}",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info(f"Downloaded dataset to {output}")
        return output


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
            cache_dir = os.getenv(MDC_CACHE_NAME, None) or DEFAULT_MDC_CACHE
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        auth = BearerAuth(api_key)
        headers = json.loads(json.dumps(kwargs.pop("headers", None) or {}))

        super().__init__(base_url=base_url, auth=auth, headers=headers, **kwargs)

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

    async def download_dataset(
        self, dataset_id: str, *, output: Path | str | None = None
    ) -> Path:
        import shutil

        from mdc_ds.utils.ensure_output import ensure_output

        ds_details = await self.get_dataset_details(dataset_id)
        cache_filepath = self.cache_dir.joinpath(f"{ds_details.id}.tar.gz")

        if cache_filepath.is_file():
            logger.debug(f"Dataset {dataset_id} found in cache")
        else:
            download_session = await self.get_dataset_download_session(ds_details.id)
            cache_filepath = await self.download_dataset_by_url(
                download_session, output=cache_filepath
            )

        if output is None:
            return cache_filepath
        else:
            output = ensure_output(output=output)
            shutil.copy(cache_filepath, output)
            return output

    async def download_dataset_by_url(
        self, download_url: DownloadSession | str, *, output: Path | str
    ) -> Path:
        """Downloads a dataset file from the provided URL and returns the path to the downloaded file."""  # noqa: E501
        import aiofiles
        from tqdm.asyncio import tqdm as async_tqdm

        from mdc_ds.utils.ensure_output import ensure_output

        # Extract URL and determine output path
        url = (
            download_url if isinstance(download_url, str) else download_url.downloadUrl
        )
        output = ensure_output(output=output)

        # Get total size for progress tracking
        total_size = (
            download_url.sizeBytes
            if isinstance(download_url, DownloadSession)
            else None
        )

        # Stream download with async progress tracking
        logger.info(f"Downloading dataset from {url} to {output}")
        async with self.stream("GET", url) as response:
            response.raise_for_status()

            # Use response content-length if sizeBytes not available
            if total_size is None:
                total_size = int(response.headers.get("content-length", 0))

            # Async file writing with progress tracking
            async with aiofiles.open(output, "wb") as f:
                with async_tqdm(
                    desc=f"Downloading {output.name}",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        await f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Downloaded dataset to {output}")
        return output
