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
            value = np.dtype(f"{dtype.name}{dtype.bits}").type(node.value)
        elif dtype.name == "bool":
            value = node.value == "True"
        else:
            return NotImplementedError()
        if node.type.dims:
            value = jnp.full([d.stop - d.start for d in node.type.dims], value)
        return value, node.type

    def visit_OffsetLiteral(self, node, **kwargs):
        return node.value, node.type

    @classmethod
    def _slice_transpose(cls, array, src_dims, dst_dims):
        if isinstance(array, tuple):
            return tuple(cls._slice_transpose(a, src_dims, dst_dims) for a in array)
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

    def visit_Builtin(self, node, **kwargs):  # noqa: C901
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
            "not_": operator.inv,
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
        if node.name == "make_tuple":

            def fun(*args):  # type: ignore
                return args

            return fun, node.type
        if node.name == "tuple_get":

            def fun(idx, tup):  # type: ignore
                return tup[idx]

            return fun, node.type
        if node.name == "subset":

            def fun(x):  # type: ignore
                return self._slice_transpose(x, node.type.args[0].dims, node.type.ret.dims)

            return fun, node.type

        if node.name == "scan":
            full_dims = set().union(*({d.name for d in a.dims} for a in node.type.ret.args))
            noncol_dims = set().union(*({d.name for d in a.dims} for a in node.type.args[0].args))
            column_dims = full_dims - noncol_dims
            assert len(column_dims) == 1
            column_dim = next(iter(column_dims))

            def fun(f, forward, init):  # type: ignore
                def wrapped_f(carry, args):
                    res = f(carry, *args)
                    return (res, res)

                def apply(*args):
                    transposed_dims = []
                    for dim in node.type.ret.ret.dims:
                        if dim.name == column_dim:
                            transposed_dims.insert(0, dim)
                        else:
                            transposed_dims.append(dim)

                    args = tuple(
                        self._slice_transpose(a, t.dims, transposed_dims)
                        for a, t in zip(args, node.type.ret.args)
                    )

                    res = jax.lax.scan(wrapped_f, init, args, reverse=not forward)[1]

                    return self._slice_transpose(res, transposed_dims, node.type.ret.ret.dims)

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

    @classmethod
    def _to_numpy(cls, expr):
        if isinstance(expr, tuple):
            return tuple(cls._to_numpy(e) for e in expr)
        return np.asarray(expr)

    @classmethod
    def _assign(cls, dst, src):
        if isinstance(dst, tuple):
            assert isinstance(src, tuple)
            for d, s in zip(dst, src):
                cls._assign(d, s)
        else:
            dst_array = np.asarray(dst)
            if dst_array.dtype.names:
                cls._assign(tuple(dst_array[n] for n in dst_array.dtype.names), src)
            else:
                dst_array[...] = src

    def visit_StencilClosure(self, node, argmap, **kwargs):
        fun = ir.FunCall(fun=node.stencil, args=node.inputs, type=node.output.type)
        out, outtype = self.visit(fun, jit=True, **kwargs)

        self._assign(argmap[node.output.id], self._to_numpy(out))

    @classmethod
    def _to_jax(cls, expr):
        if isinstance(expr, tuple):
            return tuple(cls._to_jax(e) for e in expr)
        expr = np.asarray(expr)
        if expr.dtype.names:
            return tuple(cls._to_jax(expr[n]) for n in expr.dtype.names)
        return jnp.asarray(expr)

    def visit_Fencil(self, node, *, args, offset_provider):
        argmap = {p.id: a for p, a in zip(node.params, args)}
        syms = {k: self._to_jax(v) for k, v in argmap.items()}

        self.visit(node.closures, argmap=argmap, syms=syms, offset_provider=offset_provider)
