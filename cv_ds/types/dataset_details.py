from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .orgnazation import Organization


class DatasetDetails(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(..., description="The ID of the dataset")
    slug: str = Field(..., description="The slug of the dataset")
    name: str = Field(..., description="The name of the dataset")
    shortDescription: str | None = Field(
        None, description="The short description of the dataset"
    )
    longDescription: str = Field(..., description="The long description of the dataset")
    locale: str = Field(..., description="The locale of the dataset")
    sizeBytes: int = Field(..., description="The size of the dataset in bytes")
    createdAt: datetime = Field(..., description="The creation date of the dataset")
    organization: Organization = Field(
        ..., description="The organization of the dataset"
    )
    license: str = Field(..., description="The license of the dataset")
    licenseAbbreviation: str = Field(
        ..., description="The abbreviation of the license of the dataset"
    )
    task: str = Field(..., description="The task of the dataset")
    format: str = Field(..., description="The format of the dataset")
    datasetUrl: str = Field(..., description="The URL of the dataset")
