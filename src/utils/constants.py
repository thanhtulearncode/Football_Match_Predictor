from config.settings import settings

# Re-export for backward compatibility
PREDICTORS = settings.PREDICTORS
TEAM_NAME_MAPPINGS = settings.TEAM_NAME_MAPPINGS

__all__ = ["PREDICTORS", "TEAM_NAME_MAPPINGS"]