import operator

import eve
from functional.iterator import embedded, ir as itir
from functional.tensor import ir as teir


def _common_dims(dims_iter):
    dims_dict = dict()
    for dims in dims_iter:
        for d in dims:
            if d.name not in dims_dict:
                dims_dict[d.name] = (d.start, d.stop)
            else:
                start, stop = dims_dict[d.name]
                start, stop = max(start, d.start), min(stop, d.stop)
                assert start < stop
                dims_dict[d.name] = start, stop
    return tuple(
        teir.Dim(name=name, start=start, stop=stop) for name, (start, stop) in dims_dict.items()
    )


def _common_tensor_type(types):
    types = list(types)
    assert all(t.dtype == types[0].dtype for t in types)
    dims = _common_dims(t.dims for t in types)
    return teir.TensorType(dims=dims, dtype=types[0].dtype)


def _neighbor_table_type(provider):
    assert isinstance(provider, embedded.NeighborTableOffsetProvider)
    dims = (
        teir.Dim(name=provider.origin_axis.value, start=0, stop=provider.tbl.shape[0]),
        teir.Dim(name=f"_NB_{provider.neighbor_axis.value}", start=0, stop=provider.tbl.shape[1]),
    )
    return teir.TensorType(
        dims=dims, dtype=teir.ScalarDType(name="int", bits=provider.tbl.dtype.itemsize * 8)
    )


def _broadcast(e, t):
    if (
        isinstance(e, teir.FunCall)
        and isinstance(e.fun, teir.Builtin)
        and e.fun.name == "make_tuple"
    ):
        arg_types = [teir.TensorType(dims=t.dims, dtype=e) for e in t.dtype.elems]
        args = [_broadcast(a, at) for a, at in zip(e.args, arg_types)]
        return teir.FunCall(
            type=t,
            fun=teir.Builtin(
                type=teir.FunctionType(args=tuple(a.type for a in args), ret=t),
                name="make_tuple",
            ),
            args=tuple(args),
        )
    assert isinstance(e, teir.Literal)
    return teir.Literal(type=t, value=e.value)


def highest_nb_dim(dim_names, add_one=False):
    dim_names = set(dim_names)
    idx = 0
    while f"_NB_{idx}" in dim_names:
        idx += 1
    if not add_one:
        idx -= 1
    return f"_NB_{idx}"


def _domain_dims(domain):
    assert isinstance(domain, itir.FunCall)
    dims = []
    for named_range in domain.args:
        assert isinstance(named_range, itir.FunCall) and named_range.fun == itir.SymRef(
            id="named_range"
        )
        axis, start, stop = named_range.args
        assert isinstance(axis, itir.AxisLiteral)
        assert isinstance(start, itir.Literal)
        assert isinstance(stop, itir.Literal)
        dims.append(teir.Dim(name=axis.value, start=int(start.value), stop=int(stop.value)))
    return tuple(dims)


def _dim_offsets(dims, reference_dims):
    offsets = dict()
    assert len(dims) == len(reference_dims)
    key = operator.attrgetter("name")
    for dd, rd in zip(sorted(dims, key=key), sorted(reference_dims, key=key)):
        assert dd.name == rd.name
        start_offset = dd.start - rd.start
        stop_offset = dd.stop - rd.stop
        assert start_offset >= 0 and stop_offset <= 0
        if start_offset != 0 or stop_offset != 0:
            offsets[dd.name] = start_offset, stop_offset
    return offsets


def _offset_dims(dims, offsets):
    new_dims = []
    for dim in dims:
        if dim.name not in offsets:
            new_dims.append(dim)
        else:
            start, stop = offsets[dim.name]
            new_dims.append(teir.Dim(name=dim.name, start=dim.start + start, stop=dim.stop + stop))
    return tuple(new_dims)


def _subset(node, offsets):
    assert isinstance(node.type, teir.TensorType)
    new_dims = _offset_dims(node.type.dims, offsets)
    if new_dims == node.type.dims:
        return node
    if isinstance(node, teir.FunCall) and node.fun == teir.Builtin(name="subset"):
        arg = node.args[0]
    else:
        arg = node
    ret = teir.TensorType(dims=new_dims, dtype=node.type.dtype)
    return teir.FunCall(
        fun=teir.Builtin(name="subset", type=teir.FunctionType(args=(arg.type,), ret=ret)),
        args=(arg,),
        type=ret,
    )


def _subset_closure(closure, stencil_offsets, output_offsets):
    class Subsetter(eve.NodeTranslator):
        def visit_TensorType(self, node: teir.TensorType):
            dims = _offset_dims(node.dims, stencil_offsets)
            if dims == node.dims:
                return node
            return teir.TensorType(dims=dims, dtype=node.dtype)

    return teir.StencilClosure(
        stencil=Subsetter().visit(closure.stencil),
        output=_subset(closure.output, output_offsets),
        inputs=tuple(_subset(inp, stencil_offsets) for inp in closure.inputs),
    )


class Lifter(eve.NodeTranslator):
    def visit_Sym(self, node, *, symtypes, **kwargs):
        return teir.Sym(type=symtypes[node.id], id=node.id)

    def visit_SymRef(self, node, *, symtypes, **kwargs):
        return teir.SymRef(type=symtypes[node.id], id=node.id)

    def visit_Literal(self, node, **kwargs):
        if node.type.startswith("int"):
            name = node.type[:3]
            bitstr = node.type[3:]
        elif node.type.startswith("float"):
            name = node.type[:5]
            bitstr = node.type[5:]
        elif node.type == "bool":
            name = node.type
            bitstr = "8"
        else:
            raise NotImplementedError()
        bits = int(bitstr) if bitstr else 64
        return teir.Literal(
            type=teir.TensorType(dims=(), dtype=teir.ScalarDType(name=name, bits=bits)),
            value=node.value,
        )

    def visit_OffsetLiteral(self, node, *, offset_provider, **kwargs):
        if isinstance(
            (provider := offset_provider.get(node.value)), embedded.NeighborTableOffsetProvider
        ):
            return teir.SymRef(
                type=_neighbor_table_type(provider),
                id=node.value,
            )
        return teir.OffsetLiteral(type=teir.OffsetType(), value=node.value)

    def visit_FunCall(self, node, *, symtypes, **kwargs):  # noqa: C901
        args = tuple(self.visit(node.args, symtypes=symtypes, **kwargs))
        if isinstance(node.fun, itir.Lambda):
            argtypes = {p.id: a.type for p, a in zip(node.fun.params, args)}
            fun = self.visit(node.fun, symtypes=symtypes | argtypes, **kwargs)
            return teir.FunCall(type=fun.type.ret, fun=fun, args=args)
        if isinstance(node.fun, itir.SymRef):
            if node.fun.id == "deref":
                assert len(args) == 1
                return args[0]
            if node.fun.id == "can_deref":
                assert len(args) == 1
                ret = teir.TensorType(
                    dims=args[0].type.dims, dtype=teir.ScalarDType(name="bool", bits=8)
                )
                fun = teir.Builtin(
                    name=node.fun.id,
                    type=teir.FunctionType(args=(args[0].type,), ret=ret),
                )
                return teir.FunCall(type=fun.type.ret, fun=fun, args=args)
            if node.fun.id in (
                "multiplies",
                "plus",
                "minus",
                "divides",
                "and_",
                "or_",
                "minimum",
                "maximum",
                "fmod",
                "power",
            ):
                ret = _common_tensor_type(arg.type for arg in args)
                fun = teir.Builtin(
                    name=node.fun.id,
                    type=teir.FunctionType(args=tuple(arg.type for arg in args), ret=ret),
                )
                return teir.FunCall(type=fun.type.ret, fun=fun, args=args)
            if node.fun.id in (
                "greater",
                "less",
                "eq",
                "isfinite",
                "isinf",
                "isnan",
            ):
                common = _common_tensor_type(arg.type for arg in args)
                ret = teir.TensorType(dims=common.dims, dtype=teir.ScalarDType(name="bool", bits=8))
                fun = teir.Builtin(
                    name=node.fun.id,
                    type=teir.FunctionType(args=tuple(arg.type for arg in args), ret=ret),
                )
                return teir.FunCall(type=fun.type.ret, fun=fun, args=args)
            if node.fun.id in (
                "not_",
                "abs",
                "sin",
                "cos",
                "tan",
                "arcsin",
                "arccos",
                "arctan",
                "arctan2",
                "sinh",
                "cosh",
                "tanh",
                "arcsinh",
                "arccosh",
                "arctanh",
                "sqrt",
                "exp",
                "log",
                "gamma",
                "cbrt",
                "floor",
                "ceil",
                "trunc",
            ):
                fun = teir.Builtin(
                    name=node.fun.id,
                    type=teir.FunctionType(args=tuple(arg.type for arg in args), ret=args[0].type),
                )
                return teir.FunCall(type=fun.type.ret, fun=fun, args=args)
            if node.fun.id == "if_":
                dims = _common_dims(arg.type.dims for arg in args)
                common = _common_tensor_type(arg.type for arg in args[1:])
                ret = teir.TensorType(dims=dims, dtype=common.dtype)
                fun = teir.Builtin(
                    name=node.fun.id,
                    type=teir.FunctionType(args=tuple(arg.type for arg in args), ret=ret),
                )
                return teir.FunCall(type=fun.type.ret, fun=fun, args=args)
            if node.fun.id == "make_tuple":
                dims = _common_dims(arg.type.dims for arg in args)
                dtype = teir.TupleDType(elems=tuple(arg.type.dtype for arg in args))
                ret = teir.TensorType(dims=dims, dtype=dtype)
                fun = teir.Builtin(
                    name=node.fun.id,
                    type=teir.FunctionType(args=tuple(arg.type for arg in args), ret=ret),
                )
                return teir.FunCall(type=fun.type.ret, fun=fun, args=args)
            if node.fun.id == "tuple_get":
                index = int(args[0].value)
                ret = teir.TensorType(dims=args[1].type.dims, dtype=args[1].type.dtype.elems[index])
                fun = teir.Builtin(
                    name=node.fun.id,
                    type=teir.FunctionType(args=tuple(arg.type for arg in args), ret=ret),
                )
                return teir.FunCall(type=fun.type.ret, fun=fun, args=args)
        if isinstance(node.fun, itir.FunCall) and node.fun.fun == itir.SymRef(id="shift"):
            assert len(args) == 1
            shifts = tuple(self.visit(node.fun.args, symtypes=symtypes, **kwargs))
            dims_dict = {d.name: (d.start, d.stop) for d in args[0].type.dims}
            offset_provider = kwargs["offset_provider"]

            cartesian_shifts = []
            unstructured_shifts = []

            offset_stack = list(reversed(node.fun.args))
            while offset_stack:
                offset = offset_stack.pop()
                assert isinstance(offset, itir.OffsetLiteral)
                if isinstance(offset.value, int):
                    # applying a partial (unstructured) shift
                    unstructured_shifts.insert(0, offset.value)
                else:
                    provider = offset_provider[offset.value]
                    if isinstance(provider, embedded.NeighborTableOffsetProvider):
                        if offset_stack and isinstance(offset_stack[-1].value, int):
                            # full unstructured shift
                            unstructured_shifts.append(offset_stack.pop().value)
                        # starting a partial unstructured shift
                        unstructured_shifts.append(offset.value)
                    else:
                        # full Cartesian shift (no support for partial shifts here)
                        distance = offset_stack.pop().value
                        assert isinstance(distance, int)
                        cartesian_shifts.append((offset.value, distance))

            # handle Cartesian shifts
            for dim, offset in cartesian_shifts:
                d = offset_provider[dim].value
                if d in dims_dict:
                    start, stop = dims_dict[d]
                    dims_dict[d] = (start - offset, stop - offset)

            # handle unstructured shifts
            for offset in reversed(unstructured_shifts):
                if isinstance(offset, int):
                    # apply a partial shift
                    nb_dim = highest_nb_dim(dims_dict)
                    del dims_dict[nb_dim]
                else:
                    # start a partial shift
                    provider = offset_provider[offset]
                    del dims_dict[provider.neighbor_axis.value]
                    dims_dict[provider.origin_axis.value] = (0, provider.tbl.shape[0])
                    nb_dim = highest_nb_dim(dims_dict, add_one=True)
                    dims_dict[nb_dim] = (0, provider.max_neighbors)

            dims = tuple(
                teir.Dim(name=name, start=start, stop=stop)
                for name, (start, stop) in dims_dict.items()
            )
            funtype = teir.FunctionType(
                args=(args[0].type,), ret=teir.TensorType(dtype=args[0].type.dtype, dims=dims)
            )
            shift = teir.Builtin(
                name="shift",
                type=teir.FunctionType(args=tuple(s.type for s in shifts), ret=funtype),
            )
            shift_call = teir.FunCall(type=shift.type.ret, fun=shift, args=shifts)
            return teir.FunCall(type=shift_call.type.ret, fun=shift_call, args=args)
        if isinstance(node.fun, itir.FunCall) and node.fun.fun == itir.SymRef(id="scan"):
            forward = self.visit(node.fun.args[1], symtypes=symtypes, **kwargs)
            init = self.visit(node.fun.args[2], symtypes=symtypes, **kwargs)
            dims = _common_dims([init.type.dims] + [arg.type.dims for arg in args])
            column_axis = kwargs["column_axis"].value

            def type_wo_column(t):
                return teir.TensorType(
                    dims=tuple(d for d in t.dims if d.name != column_axis), dtype=t.dtype
                )

            ret = teir.TensorType(dims=dims, dtype=init.type.dtype)
            init = _broadcast(init, type_wo_column(ret))
            scan_fun = node.fun.args[0]
            scan_fun_argtypes = [init.type] + [type_wo_column(a.type) for a in args]
            scan_fun_symtypes = symtypes | {
                p.id: t for p, t in zip(scan_fun.params, scan_fun_argtypes)
            }
            scan_fun = self.visit(scan_fun, symtypes=symtypes | scan_fun_symtypes, **kwargs)

            funtype = teir.FunctionType(args=tuple(a.type for a in args), ret=ret)
            scan_args = (scan_fun, forward, init)
            scan = teir.Builtin(
                name="scan",
                type=teir.FunctionType(args=tuple(a.type for a in scan_args), ret=funtype),
            )
            scan_call = teir.FunCall(type=scan.type.ret, fun=scan, args=scan_args)
            return teir.FunCall(type=scan_call.type.ret, fun=scan_call, args=args)
        if isinstance(node.fun, itir.FunCall) and node.fun.fun == itir.SymRef(id="lift"):
            unlifted_call = itir.FunCall(fun=node.fun.args[0], args=node.args)
            return self.visit(unlifted_call, symtypes=symtypes, **kwargs)
        if isinstance(node.fun, itir.FunCall) and node.fun.fun == itir.SymRef(id="reduce"):

            def remove_reduction_dim(dims):
                rdim = highest_nb_dim(d.name for d in dims)
                return tuple(d for d in dims if d.name != rdim)

            init = self.visit(node.fun.args[1], symtypes=symtypes, **kwargs)
            dims = _common_dims(
                [init.type.dims] + [remove_reduction_dim(a.type.dims) for a in args]
            )
            ret = teir.TensorType(dims=dims, dtype=init.type.dtype)
            init = _broadcast(init, ret)
            red_fun = node.fun.args[0]
            red_fun_argtypes = [init.type] + [
                teir.TensorType(dims=remove_reduction_dim(a.type.dims), dtype=a.type.dtype)
                for a in args
            ]
            if isinstance(red_fun, itir.Lambda):
                red_fun_symtypes = symtypes | {
                    p.id: t for p, t in zip(red_fun.params, red_fun_argtypes)
                }
                red_fun = self.visit(red_fun, symtypes=symtypes | red_fun_symtypes, **kwargs)
            else:
                assert isinstance(red_fun, itir.SymRef) and red_fun.id in ("plus", "multiplies")
                red_fun = teir.Builtin(
                    name=red_fun.id,
                    type=teir.FunctionType(args=tuple(red_fun_argtypes), ret=init.type),
                )
            funtype = teir.FunctionType(args=tuple(a.type for a in args), ret=ret)
            red_args = (red_fun, init)
            red = teir.Builtin(
                name="reduce",
                type=teir.FunctionType(args=tuple(a.type for a in red_args), ret=funtype),
            )
            red_call = teir.FunCall(type=red.type.ret, fun=red, args=red_args)
            return teir.FunCall(type=red_call.type.ret, fun=red_call, args=args)

        raise NotImplementedError()

    def visit_Lambda(self, node, **kwargs):
        params = tuple(self.visit(node.params, **kwargs))
        expr = self.visit(node.expr, **kwargs)
        return teir.Lambda(
            type=teir.FunctionType(args=tuple(p.type for p in params), ret=expr.type),
            params=params,
            expr=expr,
        )

    def visit_StencilClosure(self, node, **kwargs):
        output = self.visit(node.output, **kwargs)
        inputs = tuple(self.visit(node.inputs, **kwargs))
        call = self.visit(itir.FunCall(fun=node.stencil, args=node.inputs), **kwargs)
        if isinstance(call, teir.SymRef):
            # handling of direct deref
            call = teir.FunCall(
                type=call.type,
                args=(call,),
                fun=teir.Lambda(
                    type=teir.FunctionType(args=(call.type,), ret=call.type),
                    params=(teir.Sym(type=call.type, id="x"),),
                    expr=teir.SymRef(type=call.type, id="x"),
                ),
            )
        stencil = call.fun
        closure = teir.StencilClosure(stencil=stencil, output=output, inputs=inputs)
        domain_dims = _domain_dims(node.domain)
        stencil_offsets = _dim_offsets(domain_dims, stencil.type.ret.dims)
        output_offsets = _dim_offsets(domain_dims, output.type.dims)
        if stencil_offsets or output_offsets:
            closure = _subset_closure(closure, stencil_offsets, output_offsets)
        return closure

    def visit_FencilDefinition(self, node, *, args, offset_provider, column_axis):
        assert not node.function_definitions

        def get_dtype(x):
            if x.fields:
                return teir.TupleDType(elems=tuple(get_dtype(f[0]) for f in x.fields.values()))
            if x.kind == "f":
                name = "float"
            elif x.kind == "i":
                name = "int"
            elif x.kind == "b":
                name = "bool"
            else:
                raise NotImplementedError()
            return teir.ScalarDType(name=name, bits=x.itemsize * 8)

        def get_type(x):
            if isinstance(x, embedded.LocatedFieldImpl):
                shape = x.array().shape
                preshift_idx = -1

                def preshift_dim():
                    nonlocal preshift_idx
                    preshift_idx += 1
                    return f"_NB_{preshift_idx}"

                dims = tuple(
                    teir.Dim(
                        name=preshift_dim() if d.value in offset_provider else d.value,
                        start=-o,
                        stop=s + -o,
                    )
                    for d, o, s in zip(x.axes, x.offsets, shape)
                    if d is not None
                )
                dtype = get_dtype(x.array().dtype)
                tuple_dims = [s for d, s in zip(x.axes, shape) if d is None]
                for s in tuple_dims:
                    dtype = teir.TupleDType(elems=(dtype,) * s)
                return teir.TensorType(dims=dims, dtype=dtype)
            if isinstance(x, embedded.IndexField):
                return teir.TensorType(
                    dims=(teir.Dim(name=x.axis.value, start=-1000000000, stop=1000000000),),
                    dtype=teir.ScalarDType(name="int", bits=64),
                )
            if isinstance(x, (int, float)):
                return teir.TensorType(
                    dims=(), dtype=teir.ScalarDType(name=type(x).__name__, bits=64)
                )
            if isinstance(x, tuple):
                types = [get_type(v) for v in x]
                assert all(t.dims == types[0].dims for t in types)
                return teir.TensorType(
                    dims=types[0].dims, dtype=teir.TupleDType(elems=tuple(t.dtype for t in types))
                )
            raise NotImplementedError()

        argtypes = {str(p.id): get_type(a) for p, a in zip(node.params, args)}

        params = tuple(self.visit(node.params, symtypes=argtypes))
        closures = tuple(
            self.visit(
                node.closures,
                symtypes=argtypes,
                offset_provider=offset_provider,
                column_axis=column_axis,
            )
        )

        neighbor_tables = tuple(
            teir.Sym(type=_neighbor_table_type(v), id=k)
            for k, v in offset_provider.items()
            if isinstance(v, embedded.NeighborTableOffsetProvider)
        )

        return teir.Fencil(id=node.id, params=params + neighbor_tables, closures=closures)
