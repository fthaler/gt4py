from typing import Union

import eve
from eve import Coerced, SymbolName, SymbolRef
from eve.traits import SymbolTableTrait, ValidatedSymbolTableTrait
from eve.utils import noninstantiable


@noninstantiable
class Node(eve.FrozenNode):
    def __str__(self) -> str:
        from functional.tensor.pretty_printer import pformat

        return pformat(self)


@noninstantiable
class Type(Node):
    ...


class FunctionType(Type):
    args: tuple[Type, ...]
    ret: Type


class Dim(Node):
    name: str
    start: int
    stop: int


@noninstantiable
class DType(Node):
    ...


class ScalarDType(DType):
    name: str
    bits: int


class TupleDType(DType):
    elems: tuple[DType, ...]


class TensorType(Type):
    dims: tuple[Dim, ...]
    dtype: DType


class OffsetType(Type):
    ...


@noninstantiable
class Typed(Node):
    type: Type  # noqa: A003


class Sym(Typed):
    id: Coerced[SymbolName]  # noqa: A003


@noninstantiable
class Expr(Typed):
    ...


class Literal(Expr):
    value: str


class OffsetLiteral(Expr):
    value: Union[int, str]


class Builtin(Expr):
    name: str


class SymRef(Expr):
    id: Coerced[SymbolRef]  # noqa: A003


class Lambda(Expr, SymbolTableTrait):
    params: tuple[Sym, ...]
    expr: Expr


class FunCall(Expr):
    fun: Expr
    args: tuple[Expr, ...]

    def __post_init__(self):
        assert isinstance(self.fun.type, FunctionType)
        assert self.fun.type.args == tuple(arg.type for arg in self.args)
        assert self.fun.type.ret == self.type


class StencilClosure(Node):
    stencil: Expr
    output: Union[SymRef, FunCall]
    inputs: tuple[Union[SymRef, FunCall], ...]

    def __post_init__(self):
        assert isinstance(self.stencil.type, FunctionType)
        assert self.stencil.type.args == tuple(inp.type for inp in self.inputs)
        assert self.stencil.type.ret == self.output.type


BUILTINS = {
    "make_tuple",
    "tuple_get",
    "reduce",
    "can_deref",
    "shift",
    "scan",
    "plus",
    "minus",
    "multiplies",
    "divides",
    "eq",
    "less",
    "greater",
    "if_",
    "not_",
    "and_",
    "or_",
}


class Fencil(Node, ValidatedSymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003
    params: tuple[Sym, ...]
    closures: tuple[StencilClosure, ...]
