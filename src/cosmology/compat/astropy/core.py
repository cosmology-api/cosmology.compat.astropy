"""The Cosmology API compatability wrapper for Astropy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union, cast

from cosmology.api import CosmologyNamespace, CosmologyWrapper

__all__: list[str] = []


if TYPE_CHECKING:
    import astropy.cosmology as astropy_cosmology
    from numpy import floating
    from numpy.typing import NDArray
    from typing_extensions import TypeAlias

    NDFloating: TypeAlias = NDArray[floating[Any]]


################################################################################


@dataclass(frozen=True)
class AstropyCosmology(CosmologyWrapper["NDFloating", "NDFloating | float"]):
    """The Cosmology API wrapper for :mod:`astropy.cosmology.Cosmology`."""

    cosmo: astropy_cosmology.Cosmology

    @property
    def __cosmology_namespace__(self) -> CosmologyNamespace:
        """Returns an object that has all the cosmology API functions on it.

        Parameters
        ----------
        api_version: Optional[str]
            string representing the version of the cosmology API specification
            to be returned, in ``'YYYY.MM'`` form, for example, ``'2020.10'``.
            If ``None``, it return the namespace corresponding to latest version
            of the cosmology API specification.  If the given version is invalid
            or not implemented for the given module, an error is raised.
            Default: ``None``.

            .. note:: currently only `None` is supported.

        Returns
        -------
        `CosmologyNamespace`
            An object representing the Astropy cosmology API namespace.
        """
        import cosmology.compat.astropy

        return cast(CosmologyNamespace, cosmology.compat.astropy)

    @property
    def name(self) -> str | None:
        """The name of the cosmology instance."""
        return cast(Union[str, None], self.cosmo.name)
