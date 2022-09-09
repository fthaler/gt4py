import enum
from typing import Union

import eve
from eve import Coerced, SymbolName, SymbolRef
from eve.traits import SymbolTableTrait, ValidatedSymbolTableTrait
from eve.type_definitions import StrEnum
from eve.utils import noninstantiable


@enum.unique
class BuiltinFun(StrEnum):
    # tuple fields
    MAKE_TUPLE = "make_tuple"
    TUPLE_GET = "tuple_get"

    # special tensor ops
    SUBSET = "subset"

    # domain-specific builtins
    REDUCE = "reduce"
    SCAN = "scan"
    SHIFT = "shift"

    # element-wise unary tensor ops
    ABS = "abs"
    ARCCOS = "arccos"
    ARCCOSH = "arccosh"
    ARCSIN = "arcsin"
    ARCSINH = "arcsinh"
    ARCTAN2 = "arctan2"
    ARCTAN = "arctan"
    ARCTANH = "arctanh"
    CAN_DEREF = "can_deref"
    CBRT = "cbrt"
    CEIL = "ceil"
    COS = "cos"
    COSH = "cosh"
    EXP = "exp"
    FLOOR = "floor"
    GAMMA = "gamma"
    ISFINITE = "isfinite"
    ISINF = "isinf"
    ISNAN = "isnan"
    LOG = "log"
    NOT = "not"
    SINH = "sinh"
    SIN = "sin"
    SQRT = "sqrt"
    TANH = "tanh"
    TAN = "tan"
    TRUNC = "trunc"

    # element-wise binary tensor ops
    AND = "and"
    DIVIDES = "divides"
    EQ = "eq"
    FMOD = "fmod"
    GREATER = "greater"
    LESS = "less"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    MINUS = "minus"
    MULTIPLIES = "multiplies"
    OR = "or"
    PLUS = "plus"
    POWER = "power"

    # element-wise ternary tensor ops
    IF = "if"


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
    name: BuiltinFun


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


class Fencil(Node, ValidatedSymbolTableTrait):
    id: Coerced[SymbolName]  # noqa: A003
    params: tuple[Sym, ...]
    closures: tuple[StencilClosure, ...]
