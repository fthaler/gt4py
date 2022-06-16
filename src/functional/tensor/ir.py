from typing import ClassVar

import eve
from eve import Coerced, SymbolName, SymbolRef
from eve.traits import SymbolTableTrait, ValidatedSymbolTableTrait
from eve.utils import noninstantiable


@noninstantiable
class Node(eve.FrozenNode):
    def __str__(self):
        from functional.tensor.pretty_printer import pformat

        return pformat(self)


@noninstantiable
class Expr(Node):
    ...


class Sym(Node):
    id: Coerced[SymbolName]  # noqa: A003


class SymRef(Expr):
    id: Coerced[SymbolRef]  # noqa: A003


class Literal(Expr):
    value: str
    type: str  # noqa: A003


class Lambda(Expr, SymbolTableTrait):
    params: tuple[Sym, ...]
    expr: Expr


class Slice(Expr):
    expr: Expr
    dims: tuple[str, ...]
    starts: tuple[int, ...]
    stops: tuple[int, ...]


class FunCall(Expr):
    fun: Expr
    args: tuple[Expr, ...]


class StencilClosure(Node):
    stencil: Expr
    output: Expr
    inputs: tuple[Expr, ...]


BUILTINS = {
    "shift",
    "plus",
    "if_",
    "multiplies",
    "greater",
    "minus",
}


class Fencil(Node, ValidatedSymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003
    params: tuple[Sym, ...]
    closures: tuple[StencilClosure, ...]

    _NODE_SYMBOLS_: ClassVar = [Sym(id=name) for name in BUILTINS]
