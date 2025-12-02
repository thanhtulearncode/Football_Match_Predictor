from pydantic import BaseModel, Field

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    day_code: int = Field(5, ge=0, le=6, description="0=Mon, 5=Sat, 6=Sun")
    hour: int = Field(20, ge=0, le=23, description="Match hour (24h format)")