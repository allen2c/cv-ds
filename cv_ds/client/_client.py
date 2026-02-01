import json
import logging
import os
from typing import Any, Generator

import httpx

from cv_ds import MDC_API_KEY_NAME
from cv_ds.types.dataset_details import DatasetDetails

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
        api_key: str | None = None,
        base_url: str = "https://datacollective.mozillafoundation.org/api",
        **kwargs: Any,
    ):
        if api_key is None:
            if not (api_key := os.getenv(MDC_API_KEY_NAME)):
                raise ValueError(error_api_key_missing_msg)

        auth = BearerAuth(api_key)

        headers = json.loads(json.dumps(kwargs.pop("headers", None) or {}))

        super().__init__(base_url=base_url, auth=auth, headers=headers, **kwargs)

    def get_dataset_details(self, dataset_id: str) -> DatasetDetails:
        from cv_ds.registry import get_dataset_details

        if cached_ds := get_dataset_details(dataset_id):
            logger.debug(f"Dataset input {dataset_id} found in cache")
            return cached_ds

        response = self.get(f"/datasets/{dataset_id}")
        response.raise_for_status()
        return DatasetDetails.model_validate(response.json())


class MozillaDataCollectiveAsyncClient(httpx.AsyncClient):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://datacollective.mozillafoundation.org/api",
        **kwargs: Any,
    ):
        if api_key is None:
            if not (api_key := os.getenv(MDC_API_KEY_NAME)):
                raise ValueError(error_api_key_missing_msg)

        auth = BearerAuth(api_key)
        headers = json.loads(json.dumps(kwargs.pop("headers", None) or {}))

        super().__init__(base_url=base_url, auth=auth, headers=headers, **kwargs)

    async def get_dataset_details(self, dataset_id: str) -> DatasetDetails:
        from cv_ds.registry import get_dataset_details

        if cached_ds := get_dataset_details(dataset_id):
            logger.debug(f"Dataset input {dataset_id} found in cache")
            return cached_ds

        response = await self.get(f"/datasets/{dataset_id}")
        response.raise_for_status()
        return DatasetDetails.model_validate(response.json())
