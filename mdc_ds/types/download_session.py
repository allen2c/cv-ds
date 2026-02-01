from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DownloadSession(BaseModel):
    model_config = ConfigDict(extra="allow")
    accessRecordId: str = Field(..., description="The access record ID")
    downloadToken: str = Field(..., description="The download token")
    downloadUrl: str = Field(..., description="The download URL")
    expiresAt: datetime = Field(..., description="The expiration date")
    sizeBytes: int = Field(..., description="The size of the dataset in bytes")
    contentType: str = Field(..., description="The content type")
    filename: str = Field(..., description="The filename")
    checksum: str = Field(..., description="The checksum")
