from functools import lru_cache

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Configuration settings for PLEQUE application.

    """
@lru_cache
def get_settings():
    return Settings()