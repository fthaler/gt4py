import operator
from dataclasses import dataclass

import jax
import numpy as np
from jax import numpy as jnp, tree_util

from functional.iterator.embedded import LocatedField
from functional.tensor import ir
from functional.tensor.evaluator import Evaluator


@dataclass(frozen=True)
class DimArray:
    array: jax.core.Jaxpr
    dims: tuple[str, ...]

    def all_dims(self, *others):
        return tuple(
            sorted(set(self.dims).union(*(o.dims for o in others if isinstance(o, DimArray))))
        )

    def transpose(self, dims):
        assert set(self.dims) == set(dims)
        return DimArray(array=self.array.transpose([self.dims.index(d) for d in dims]), dims=dims)

    def slice(self, **kwargs):  # noqa: A003
        slices = [slice(None) for _ in self.dims]
        for dim, (start, stop) in kwargs.items():
            slices[self.dims.index(dim)] = slice(start, stop)
        return DimArray(array=self.array[tuple(slices)], dims=self.dims)

    def extend_dims(self, dims):
        transposed = self.transpose([d for d in dims if d in self.dims])
        slices = tuple(slice(None) if d in transposed.dims else jnp.newaxis for d in dims)
        return DimArray(array=transposed.array[slices], dims=dims)

    def binary_op(self, other, op):
        if not isinstance(other, DimArray):
            return DimArray(array=op(self.array, other), dims=self.dims)
        dims = self.all_dims(other)
        self_extended = self.extend_dims(dims)
        other_extended = other.extend_dims(dims)
        return DimArray(array=op(self_extended.array, other_extended.array), dims=dims)

    def rbinary_op(self, other, op):
        assert not isinstance(other, DimArray)
        return DimArray(array=op(other, self.array), dims=self.dims)

    def unary_op(self, op):
        return DimArray(array=op(self.array), dims=self.dims)

    def __add__(self, other):
        return self.binary_op(other, operator.add)

    def __radd__(self, other):
        return self.rbinary_op(other, operator.add)

    def __sub__(self, other):
        return self.binary_op(other, operator.sub)

    def __rsub__(self, other):
        return self.rbinary_op(other, operator.sub)

    def __mul__(self, other):
        return self.binary_op(other, operator.mul)

    def __rmul__(self, other):
        return self.binary_op(other, operator.mul)

    def __truediv__(self, other):
        return self.binary_op(other, operator.truediv)

    def __rtruediv__(self, other):
        return self.binary_op(other, operator.truediv)

    def __gt__(self, other):
        return self.binary_op(other, operator.gt)

    def __lt__(self, other):
        return self.binary_op(other, operator.lt)

    def __eq__(self, other):
        return self.binary_op(other, operator.eq)

    def __and__(self, other):
        return self.binary_op(other, operator.and_)

    def __or__(self, other):
        return self.binary_op(other, operator.or_)

    def __invert__(self):
        return self.unary_op(operator.invert)


tree_util.register_pytree_node(
    DimArray,
    lambda x: ((x.array,), {"dims": x.dims}),
    lambda data, children: DimArray(array=children[0], dims=data["dims"]),
)


def _if_(c, t, f):
    if not isinstance(c, DimArray):
        c = DimArray(array=jnp.asarray(c), dims=())
    if not isinstance(t, DimArray):
        t = DimArray(array=jnp.asarray(t), dims=())
    if not isinstance(f, DimArray):
        f = DimArray(array=jnp.asarray(f), dims=())
    dims = c.all_dims(t, f)
    t = t.extend_dims(dims)
    f = f.extend_dims(dims)
    return DimArray(array=jnp.where(c.array, t.array, f.array), dims=dims)


BUILTIN_SYMS = {
    "plus": operator.add,
    "minus": operator.sub,
    "multiplies": operator.mul,
    "divides": operator.truediv,
    "greater": operator.gt,
    "less": operator.lt,
    "eq": operator.eq,
    "not_": operator.invert,
    "and_": operator.and_,
    "or_": operator.or_,
    "make_tuple": lambda *args: args,
    "tuple_get": lambda idx, tup: tup[idx],
    "if_": _if_,
}


def _to_jax(inp, dims=None):
    if isinstance(inp, tuple):
        return tuple(_to_jax(i) for i in inp)
    if isinstance(inp, np.ndarray):
        assert dims is not None
        return DimArray(array=jnp.asarray(inp), dims=dims)
    dims = _get_dims(inp)
    array = inp.array()
    if array.dtype.fields:
        return tuple(_to_jax(array[f], dims) for f in array.dtype.fields)
    return DimArray(array=jnp.asarray(array), dims=dims)


def _get_dims(output):
    if isinstance(output, tuple):
        assert all(_get_dims(o) == _get_dims(output[0]) for o in output[1:])
        return _get_dims(output[0])
    return tuple(a.value for a in output.axises)


def _sliced_assign(output, evaluated, slices):
    if isinstance(output, LocatedField):
        return _sliced_assign(output.array(), evaluated, slices)
    elif isinstance(output, tuple) and isinstance(evaluated, tuple):
        assert len(output) == len(evaluated)
        for o, e in zip(output, evaluated):
            _sliced_assign(o, e, slices)
    elif isinstance(evaluated, tuple):
        assert len(output.dtype.fields) == len(evaluated)
        for f, e in zip(output.dtype.fields, evaluated):
            _sliced_assign(output[f], e, slices)
        return
    else:
        output[slices] = evaluated.array


class JaxEvaluator(Evaluator):
    def visit_Slice(self, node, *, syms):
        expr = self.visit(node.expr, syms=syms)
        slices = {
            dim: (start, stop) for dim, start, stop in zip(node.dims, node.starts, node.stops)
        }

        def apply_slice(expr):
            if isinstance(expr, tuple):
                return tuple(apply_slice(e) for e in expr)
            return expr.slice(**slices)

        return apply_slice(expr)

    def visit_Lambda(self, node, **kwargs):
        return jax.jit(super().visit_Lambda(node, **kwargs))

    def visit_FunCall(self, node, **kwargs):
        fun = self.visit(node.fun, **kwargs)
        args = self.visit(node.args, **kwargs)
        if hasattr(fun, "lower"):
            print(fun.lower(*args).compiler_ir())
        return fun(*args)

    def visit_StencilClosure(self, node, *, argmap):
        wrapped = ir.FunCall(fun=node.stencil, args=node.inputs)
        args = dict()
        for arg in node.inputs:
            args[arg.id] = _to_jax(argmap[arg.id])
        evaluated = self.visit(wrapped, syms=BUILTIN_SYMS | args)
        if isinstance(node.output, ir.SymRef):
            output = argmap[node.output.id]
            slices = (Ellipsis,)
        else:
            output = argmap[node.output.expr.id]
            assert isinstance(node.output, ir.Slice)
            dims = _get_dims(output)
            slices = [slice(None) for _ in dims]
            for dim, start, stop in zip(node.output.dims, node.output.starts, node.output.stops):
                slices[dims.index(dim)] = slice(start, stop)
            slices = tuple(slices)
        _sliced_assign(output, evaluated, slices)

    def visit_Fencil(self, node, *, args):
        argmap = {param.id: arg for param, arg in zip(node.params, args)}

        for closure in node.closures:
            self.visit(closure, argmap=argmap)
