import pandera as pa
from pandera.typing import Series
from typing import Optional

class RawMatchSchema(pa.DataFrameModel):
    """Schema for data entering the pipeline (after scraping)."""
    date: Series[pa.DateTime] = pa.Field(coerce=True)
    team: Series[str]
    opponent: Series[str]
    result: Optional[Series[str]] = pa.Field(nullable=True)
    
    class Config:
        coerce = True
        strict = False  # Allow extra columns

class TrainingSchema(pa.DataFrameModel):
    """Schema for data ready for the model."""
    venue_code: Series[int]
    opp_code: Series[int]
    hour: Series[int]
    day_code: Series[int]
    target: Series[int] = pa.Field(ge=0, le=2) # 0, 1, or 2
    # Check for critical rolling features
    gf_rolling: Series[float] = pa.Field(nullable=True)
    ga_rolling: Series[float] = pa.Field(nullable=True)
    
    class Config:
        strict = False