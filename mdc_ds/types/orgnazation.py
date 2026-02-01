# cv_ds/types/orgnazation.py
from pydantic import BaseModel, ConfigDict, Field


class Organization(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(..., description="The name of the organization")
    slug: str = Field(..., description="The slug of the organization")
