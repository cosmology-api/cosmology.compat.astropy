"""The Cosmology API compatability wrapper for Astropy."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable, Union, cast

from astropy.cosmology import Cosmology as AstropyCosmology  # noqa: TCH002
from astropy.units import Quantity
from numpy import floating
from numpy.typing import NDArray
from typing_extensions import TypeAlias  # noqa: TCH002

from cosmology.api import CosmologyNamespace
from cosmology.api import CosmologyWrapper as CosmologyWrapperAPI

__all__: list[str] = []


NDFloating: TypeAlias = NDArray[floating[Any]]
InputT: TypeAlias = Union[Quantity, NDFloating, float]

################################################################################


@dataclass(frozen=True)
class CosmologyWrapper(CosmologyWrapperAPI[Quantity, InputT]):
    """The Cosmology API wrapper for :mod:`astropy.cosmology.Cosmology`."""

    cosmo: AstropyCosmology

    def __post_init__(self) -> None:
        self._cosmo_fn: dict[str, Callable[..., Any]]
        object.__setattr__(self, "_cosmo_fn", {})

        with suppress(AttributeError):
            super().__post_init__()

    @property
    def __cosmology_namespace__(self) -> CosmologyNamespace:
        """Returns an object that has all the cosmology API functions on it.

        Returns
        -------
        `cosmology.api.CosmologyNamespace`
            An object representing the Astropy cosmology API namespace.
        """
        import cosmology.compat.astropy as namespace

        return cast(CosmologyNamespace, namespace)

    @property
    def name(self) -> str | None:
        """The name of the cosmology instance."""
        return cast(Union[str, None], self.cosmo.name)

    # ========================================================================
    # Convenience methods

    # @property constants is inherited from CosmologyWrapperAPI
