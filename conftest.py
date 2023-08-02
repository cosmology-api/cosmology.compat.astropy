"""Doctest configuration."""

from __future__ import annotations

from doctest import ELLIPSIS
from typing import TYPE_CHECKING

from sybil import Sybil
from sybil.evaluators.python import PythonEvaluator, pad
from sybil.parsers.abstract import AbstractCodeBlockParser
from sybil.parsers.rest import DocTestParser, SkipParser
from sybil.parsers.rest.lexers import DirectiveInCommentLexer, DirectiveLexer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sybil.typing import Lexer


class PythonCodeBlockParser(AbstractCodeBlockParser):
    """An abstract parser for use when evaluating blocks of code.

    :param lexers:
        A sequence of :any:`Lexer` objects that will be applied in turn to each
        :class:`~sybil.Document` that is parsed. The :class:`~sybil.LexedRegion`
        objects returned by these lexers must have both an ``arguments`` string,
        containing the language of the lexed region, and a ``source``
        :class:`~sybil.Lexeme` containing the source code of the lexed region.

    :param future_imports:
        An optional list of strings that will be turned into
        ``from __future__ import ...`` statements and prepended to the code
        in each of the examples found by this parser.
    """

    def __init__(
        self, lexers: Sequence[Lexer], future_imports: tuple[str, ...] = ()
    ) -> None:
        super().__init__(lexers, "python", PythonEvaluator(future_imports))

    pad = staticmethod(pad)


pytest_collect_file = Sybil(
    parsers=[
        DocTestParser(optionflags=ELLIPSIS),
        PythonCodeBlockParser(
            lexers=[
                DirectiveLexer(directive=r"code-block"),
                DirectiveInCommentLexer(
                    directive=r"(invisible-)?code(-block)?"
                ),  # sybil
                DirectiveLexer(directive=r"testsetup"),  # sphinx.ext.doctest
                DirectiveLexer(directive=r"testcleanup"),  # sphinx.ext.doctest
            ],
            future_imports=("annotations",),
        ),
        SkipParser(),
    ],
    patterns=["*.rst", "*.py"],
).pytest()
