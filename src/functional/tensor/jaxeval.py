import operator
from dataclasses import dataclass

import jax
from jax import numpy as jnp, tree_util

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

    def __gt__(self, other):
        return self.binary_op(other, operator.gt)


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
    "if_": _if_,
}


class JaxEvaluator(Evaluator):
    def visit_Slice(self, node, *, syms):
        expr = self.visit(node.expr, syms=syms)
        return expr.slice(
            **{dim: (start, stop) for dim, start, stop in zip(node.dims, node.starts, node.stops)}
        )

    def visit_Lambda(self, node, **kwargs):
        return jax.jit(super().visit_Lambda(node, **kwargs))

    def visit_FunCall(self, node, **kwargs):
        fun = self.visit(node.fun, **kwargs)
        args = self.visit(node.args, **kwargs)
        if hasattr(fun, "lower"):
            print(fun.lower(*args).compiler_ir())
        return fun(*args)

    def visit_StencilClosure(self, node, *, argmap, offset_provider):
        wrapped = ir.FunCall(fun=node.stencil, args=node.inputs)
        args = dict()
        dim_map = {v.value: k for k, v in offset_provider.items()}
        for arg in node.inputs:
            argval = argmap[arg.id]
            args[arg.id] = DimArray(
                array=jnp.asarray(argval.array()),
                dims=tuple(dim_map[a.value] for a in argval.axises),
            )
        evaluated = self.visit(wrapped, syms=BUILTIN_SYMS | args)
        if isinstance(node.output, ir.SymRef):
            output = argmap[node.output.id]
            slices = [Ellipsis]
        else:
            output = argmap[node.output.expr.id]
            assert isinstance(node.output, ir.Slice)
            dims = tuple(dim_map[a.value] for a in output.axises)
            slices = [slice(None) for _ in dims]
            for dim, start, stop in zip(node.output.dims, node.output.starts, node.output.stops):
                slices[dims.index(dim)] = slice(start, stop)
        output.array()[tuple(slices)] = evaluated.array

    def visit_Fencil(self, node, *, args, offset_provider):
        argmap = {param.id: arg for param, arg in zip(node.params, args)}

        for closure in node.closures:
            self.visit(closure, argmap=argmap, offset_provider=offset_provider)
