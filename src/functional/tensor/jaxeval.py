import operator
from dataclasses import dataclass

import jax
import numpy as np
from jax import numpy as jnp
from jax.tree_util import register_pytree_node

import eve
from functional.iterator import embedded
from functional.tensor import ir


@dataclass
class IndexField:
    axis: str

    @property
    def dtype(self):
        return np.dtype("int64")

    def __getitem__(self, indices):
        (indices,) = indices
        if isinstance(indices, slice):
            if indices.start is indices.stop is None:
                return self
            return jnp.arange(indices.start, indices.stop)
        return indices


register_pytree_node(
    IndexField,
    lambda x: ((), x.axis),
    lambda axis, _: IndexField(axis=axis),
)


def masked_value(dtype):
    if dtype.kind == "f":
        return np.nan
    if dtype.kind == "i":
        return -(2 ** (dtype.itemsize * 8 - 1))
    raise NotImplementedError()


def mask(x):
    if x.dtype.kind == "f":
        return ~jnp.isnan(x)
    return x != masked_value(x.dtype)


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

                    offset_stack = list(zip(offsets, node.type.args))
                    dims = node.type.ret.args[0].dims
                    while offset_stack:
                        offset, offset_type = offset_stack.pop()
                        if isinstance(offset, int):
                            dim, dim_type = offset_stack.pop()
                            if isinstance(dim, str):
                                # cartesian shift is a no-op
                                continue
                            assert not isinstance(dim, int)

                            # full unstructured shift
                            nb_dim = dim_type.dims[1].name.removeprefix("NB_")
                            indices = dim[:, offset]
                            slices = tuple(
                                indices if d.name == nb_dim else slice(None) for d in dims
                            )
                            mask_slices = tuple(
                                slice(None) if d.name == nb_dim else np.newaxis for d in dims
                            )
                            mask = (indices == -1)[mask_slices]
                            x = jnp.where(mask, masked_value(x.dtype), x[slices])
                            dims = tuple(dim_type.dims[0] if d.name == nb_dim else d for d in dims)
                        else:
                            # partial unstructured shift
                            dim, dim_type = offset, offset_type
                            nb_dim = dim_type.dims[1].name.removeprefix("NB_")
                            indices = dim
                            slices = tuple(
                                indices if d.name == nb_dim else slice(None) for d in dims
                            )
                            idx = 0
                            while f"_NB_{idx}" in {d.name for d in dims}:
                                idx += 1
                            new_dims = []
                            mask_slices = []
                            for dim in dims:
                                if dim.name == nb_dim:
                                    new_dims += [
                                        dim_type.dims[0],
                                        ir.Dim(
                                            name=f"_NB_{idx}",
                                            start=dim_type.dims[1].start,
                                            stop=dim_type.dims[1].stop,
                                        ),
                                    ]
                                    mask_slices += [slice(None), slice(None)]
                                else:
                                    new_dims.append(dim)
                                    mask_slices.append(np.newaxis)
                            mask = (indices == -1)[tuple(mask_slices)]
                            x = jnp.where(mask, masked_value(x.dtype), x[slices])
                            dims = tuple(new_dims)
                    assert [d.name for d in dims] == [d.name for d in node.type.ret.ret.dims]

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

        if node.name == "can_deref":

            def fun(x):  # type: ignore
                return mask(x)

            return fun, node.type

        if node.name == "scan":
            full_dims = set().union(*({d.name for d in a.dims} for a in node.type.ret.args))
            noncol_dims = set().union(*({d.name for d in a.dims} for a in node.type.args[0].args))
            column_dims = full_dims - noncol_dims
            assert len(column_dims) == 1
            column_dim = next(iter(column_dims))
            transposed_dims = []
            for dim in node.type.ret.ret.dims:
                if dim.name == column_dim:
                    transposed_dims.insert(0, dim)
                else:
                    transposed_dims.append(dim)

            def fun(f, forward, init):  # type: ignore
                def wrapped_f(carry, args):
                    res = f(carry, *args)
                    return (res, res)

                def apply(*args):
                    args = tuple(
                        self._slice_transpose(a, t.dims, transposed_dims)
                        for a, t in zip(args, node.type.ret.args)
                    )

                    res = jax.lax.scan(wrapped_f, init, args, reverse=not forward)[1]
                    return self._slice_transpose(res, transposed_dims, node.type.ret.ret.dims)

                return apply

            return fun, node.type

        if node.name == "reduce":
            transposed_arg_dims = []
            ret_dims = {d.name for d in node.type.ret.ret.dims}
            for arg in node.type.ret.args:
                transposed_dims = []
                red_dims = {d.name for d in arg.dims} - ret_dims
                assert len(red_dims) == 1
                red_dim = next(iter(red_dims))
                for dim in arg.dims:
                    if dim.name == red_dim:
                        transposed_dims.insert(0, dim)
                    else:
                        transposed_dims.append(dim)
                transposed_arg_dims.append(transposed_dims)

            def fun(f, init):  # type: ignore
                def wrapped_f(carry, args):
                    res = jnp.where(mask(args[0]), f(carry, *args), carry)
                    return (res, res)

                def apply(*args):
                    args = tuple(
                        self._slice_transpose(a, t.dims, d)
                        for a, t, d in zip(args, node.type.ret.args, transposed_arg_dims)
                    )
                    return jax.lax.scan(wrapped_f, init, args)[0]

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
        if isinstance(expr, embedded.IndexField):
            return IndexField(axis=expr.axis.value)
        expr = np.asarray(expr)
        if expr.dtype.names:
            return tuple(cls._to_jax(expr[n]) for n in expr.dtype.names)
        return jnp.asarray(expr)

    def visit_Fencil(self, node, *, args, offset_provider):
        argmap = {p.id: a for p, a in zip(node.params, args)}
        syms = {k: self._to_jax(v) for k, v in argmap.items()}

        for k, v in offset_provider.items():
            if isinstance(v, embedded.NeighborTableOffsetProvider):
                assert k not in syms
                syms[k] = self._to_jax(v.tbl)

        self.visit(node.closures, argmap=argmap, syms=syms, offset_provider=offset_provider)
