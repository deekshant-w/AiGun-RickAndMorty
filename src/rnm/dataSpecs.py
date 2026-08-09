from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class FilePathInput(BaseModel):
    path: Path = Field(..., description="Absolute path to the file")

    @field_validator("path")
    @classmethod
    def must_be_absolute_and_exist(cls, v):
        p = Path(v)
        if not p.is_absolute():
            raise ValueError(f"Path must be absolute, got: {v}")
        if not p.exists():
            raise ValueError(f"Path does not exist: {v}")
        return p.resolve()
