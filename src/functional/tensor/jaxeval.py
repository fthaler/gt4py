import operator

import jax
import numpy as np
from jax import numpy as jnp

import eve
from functional.tensor import ir


class JaxEvaluator(eve.NodeTranslator):
    def visit_SymRef(self, node, *, syms, **kwargs):
        return syms[node.id], node.type

    def visit_Literal(self, node, **kwargs):
        assert isinstance(node.type, ir.TensorType)
        dtype = node.type.dtype
        assert isinstance(dtype, ir.ScalarDType)
        if dtype.name in ("float", "int"):
            return jnp.dtype(f"{dtype.name}{dtype.bits}").type(node.value), node.type
        if dtype.name == "bool":
            return jnp.bool_(node.value == "True"), node.type
        return NotImplementedError()

    def visit_OffsetLiteral(self, node, **kwargs):
        return node.value, node.type

    @staticmethod
    def _slice_transpose(array, src_dims, dst_dims):
        dst_dims_dict = {d.name: (d.start, d.stop) for d in dst_dims}
        slices = tuple(
            slice((d := dst_dims_dict[s.name])[0] - s.start, d[1] - s.start) for s in src_dims
        )
        src_names = [s.name for s in src_dims]
        axes = []
        expanded_slices = []
        for d in dst_dims:
            if d.name in src_names:
                axes.append(src_names.index(d.name))
                expanded_slices.append(slice(None))
            else:
                expanded_slices.append(jnp.newaxis)

        sliced = array[slices]
        transposed = sliced.transpose(axes)
        res = transposed[tuple(expanded_slices)]
        return res

    def visit_Builtin(self, node, **kwargs):
        ops = {
            "minus": operator.sub,
            "plus": operator.add,
            "multiplies": operator.mul,
            "divides": operator.truediv,
            "greater": operator.gt,
            "less": operator.lt,
            "eq": operator.eq,
            "and_": operator.and_,
            "or_": operator.or_,
            "not_": operator.not_,
        }
        if node.name in (
            "minus",
            "plus",
            "multiplies",
            "divides",
            "greater",
            "less",
            "eq",
            "and_",
            "or_",
        ):

            def fun(x, y):
                x = self._slice_transpose(x, node.type.args[0].dims, node.type.ret.dims)
                y = self._slice_transpose(y, node.type.args[1].dims, node.type.ret.dims)
                return ops[node.name](x, y)

            return fun, node.type
        if node.name == "not_":

            def fun(x):  # type: ignore
                x = self._slice_transpose(x, node.type.args[0].dims, node.type.ret.dims)
                return ops[node.name](x)

            return fun, node.type
        if node.name == "if_":

            def fun(x, y, z):  # type: ignore
                x = self._slice_transpose(x, node.type.args[0].dims, node.type.ret.dims)
                y = self._slice_transpose(y, node.type.args[1].dims, node.type.ret.dims)
                z = self._slice_transpose(z, node.type.args[2].dims, node.type.ret.dims)
                return jnp.where(x, y, z)

            return fun, node.type
        if node.name == "shift":

            def fun(*offsets):  # type: ignore
                def apply(x):
                    return x

                return apply

            return fun, node.type

        raise NotImplementedError()

    def visit_FunCall(self, node, *, jit=False, **kwargs):
        fun, _ = self.visit(node.fun, **kwargs)
        args = [arg[0] for arg in self.visit(node.args, **kwargs)]
        if jit:
            fun = jax.jit(fun)
            print(fun.lower(*args).compiler_ir())
        return fun(*args), node.type

    def visit_Lambda(self, node, *, syms, **kwargs):
        def fun(*args):
            assert len(args) == len(node.params)
            return self.visit(
                node.expr, syms=syms | {p.id: a for p, a in zip(node.params, args)}, **kwargs
            )[0]

        return fun, node.type

    def visit_StencilClosure(self, node, argmap, **kwargs):
        fun = ir.FunCall(fun=node.stencil, args=node.inputs, type=node.output.type)
        out, outtype = self.visit(fun, jit=True, **kwargs)
        argmap[node.output.id].array()[...] = np.asarray(out)

    def visit_Fencil(self, node, *, args, offset_provider):
        argmap = {p.id: a for p, a in zip(node.params, args)}
        syms = {k: jnp.asarray(v) for k, v in argmap.items()}

        self.visit(node.closures, argmap=argmap, syms=syms, offset_provider=offset_provider)
