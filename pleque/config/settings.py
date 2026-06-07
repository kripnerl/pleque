from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configuration settings for PLEQUE application.

    """
    npsi_grid: int = Field(200, description="Default number of flux labels used for evaluation of flux surfaces.")
    nr_grid: int | None = Field(None,
                                   description=
                                   """Default number of radial points of rectangular grid used for various calculations.
                                    If None, the input grid is used."""
                                   )
    nz_grid: int | None = Field(None,
                                   description=
                                   """Default number of vertical points of rectangular grid used for various calculations.
                                   If None, the input grid is used."""
    )
    psin0: float = Field(0.01, description="Minimum flux surface psi_n used for flux surface averaging and volume integration.")


@lru_cache
def get_settings():
    return Settings()
