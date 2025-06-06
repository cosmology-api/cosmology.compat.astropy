"""Custom Sphinx extension for documenting Cosmology API."""

import sys
from inspect import getmembers, getmro
from types import ModuleType
from typing import Any

from sphinx.util.inspect import signature, stringify_signature

# get_overloads is a Python 3.11 feature
try:
    from typing import get_overloads
except ImportError:
    get_overloads = None

import cosmology.compat.astropy

DOCS_MODULE_NAME = "cosmology.compat.astropy.docs"


class cosmo(cosmology.compat.astropy.StandardCosmologyWrapper):  # noqa: N801
    """Mock cosmology class used for documenting all protocols."""


COSMOLOGY_BASES = [cls for cls in getmro(cosmo) if cls is not cosmo]


def context_callback(app, name, obj, parent, context):
    """Callback function to provide extra context for reference."""
    # add protocols to context if obj is a method or property of Cosmology
    extra_context = None
    if parent is cosmo:
        _, _, membername = name.rpartition(".")
        protocols = [
            f"{cls.__qualname__}.{membername}"
            for cls in COSMOLOGY_BASES
            if hasattr(cls, membername)
        ]
        extra_context = {"protocols": protocols}
    return extra_context


def signature_callback(app, what, name, obj, options, sig, return_annotation):
    """Callback function to provide overloaded signatures."""
    if what in ("function", "method") and callable(obj):
        overloads = get_overloads(obj)
        if overloads:
            kwargs = {}
            if app.config.autodoc_typehints in ("none", "description"):
                kwargs["show_annotation"] = False
            if app.config.autodoc_typehints_format == "short":
                kwargs["unqualified_typehints"] = True
            type_aliases = app.config.autodoc_type_aliases
            bound_method = what == "method"
            sigs = []
            for overload in overloads:
                overload_sig = signature(
                    overload, bound_method=bound_method, type_aliases=type_aliases
                )
                sigs.append(stringify_signature(overload_sig, **kwargs))
            return "\n".join(sigs), None
    return None


def setup(app: Any) -> None:
    """Initialise the cosmology.compat.astropy documentation extension."""
    # create a mock `cosmology.compat.astropy.docs` module containing `Cosmology`
    docs_module = ModuleType(DOCS_MODULE_NAME)
    docs_module.cosmo = cosmo
    cosmo.__module__ = DOCS_MODULE_NAME
    sys.modules[DOCS_MODULE_NAME] = docs_module

    # change the autosummary_filename_map for all members of Cosmology
    filename_map = {}
    filename_prefix = f"{cosmo.__module__}.{cosmo.__name__}"
    for name, _obj in getmembers(cosmo):
        filename_map[f"{filename_prefix}.{name}"] = name
    app.config.autosummary_filename_map.update(filename_map)

    # register a callback for extra autosummary context
    app.connect("autosummary-gather-context", context_callback)

    # register a callback to show overloaded signatures
    if get_overloads:
        app.connect("autodoc-process-signature", signature_callback)
