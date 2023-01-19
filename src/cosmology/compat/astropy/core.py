"""The Cosmology API compatability wrapper for Astropy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from cosmology.api import CosmologyAPIConformant, CosmologyAPINamespace

import astropy.cosmology as astropy_cosmology

__all__: list[str] = []


@dataclass(frozen=True)
class AstropyCosmology(CosmologyAPIConformant):
    """The Cosmology API wrapper for :mod:`astropy.cosmology.Cosmology`."""

    cosmo: astropy_cosmology.Cosmology

    def __cosmology_namespace__(
        self,
        /,
        *,
        api_version: str | None = None,
    ) -> CosmologyAPINamespace:
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
        `CosmologyAPINamespace`
            An object representing the Astropy cosmology API namespace.
        """
        import cosmology.compat.astropy  # type: ignore[import]

        return cast("CosmologyAPINamespace", cosmology.compat.astropy)

    @property
    def name(self) -> str | None:
        """The name of the cosmology instance."""
        return cast("str | None", self.cosmo.name)
