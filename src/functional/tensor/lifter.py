import eve
from functional.iterator import ir as itir
from functional.iterator.embedded import LocatedFieldImpl
from functional.tensor import ir as teir


def common_dims(dims_iter):
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


def common_tensor_type(types):
    types = list(types)
    assert all(t.dtype == types[0].dtype for t in types)
    dims = common_dims(t.dims for t in types)
    return teir.TensorType(dims=dims, dtype=types[0].dtype)


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
        else:
            raise NotImplementedError()
        bits = int(bitstr) if bitstr else 64
        return teir.Literal(
            type=teir.TensorType(dims=(), dtype=teir.ScalarDType(name=name, bits=bits)),
            value=node.value,
        )

    def visit_OffsetLiteral(self, node, **kwargs):
        return teir.OffsetLiteral(type=teir.OffsetType(), value=node.value)

    def visit_FunCall(self, node, *, symtypes, **kwargs):
        args = tuple(self.visit(node.args, symtypes=symtypes, **kwargs))
        if isinstance(node.fun, itir.Lambda):
            argtypes = {p.id: a.type for p, a in zip(node.fun.params, args)}
            fun = self.visit(node.fun, symtypes=symtypes | argtypes, **kwargs)
            return teir.FunCall(type=fun.type.ret, fun=fun, args=args)
        if isinstance(node.fun, itir.SymRef):
            if node.fun.id == "deref":
                assert len(args) == 1
                fun = teir.Lambda(
                    type=teir.FunctionType(args=(args[0].type,), ret=args[0].type),
                    params=(teir.Sym(id="x", type=args[0].type),),
                    expr=teir.SymRef(id="x", type=args[0].type),
                )
                return teir.FunCall(type=fun.type.ret, fun=fun, args=args)
            if node.fun.id in ("multiplies", "plus", "minus", "divides", "and_", "or_"):
                ret = common_tensor_type(arg.type for arg in args)
                fun = teir.Builtin(
                    name=node.fun.id,
                    type=teir.FunctionType(args=tuple(arg.type for arg in args), ret=ret),
                )
                return teir.FunCall(type=fun.type.ret, fun=fun, args=args)
            if node.fun.id in ("greater", "less", "eq"):
                common = common_tensor_type(arg.type for arg in args)
                ret = teir.TensorType(dims=common.dims, dtype=teir.ScalarDType(name="bool", bits=8))
                fun = teir.Builtin(
                    name=node.fun.id,
                    type=teir.FunctionType(args=tuple(arg.type for arg in args), ret=ret),
                )
                return teir.FunCall(type=fun.type.ret, fun=fun, args=args)
            if node.fun.id == "not_":
                fun = teir.Builtin(
                    name=node.fun.id,
                    type=teir.FunctionType(args=tuple(arg.type for arg in args), ret=args[0].type),
                )
                return teir.FunCall(type=fun.type.ret, fun=fun, args=args)
            if node.fun.id == "if_":
                dims = common_dims(arg.type.dims for arg in args)
                common = common_tensor_type(arg.type for arg in args[1:])
                ret = teir.TensorType(dims=dims, dtype=common.dtype)
                fun = teir.Builtin(
                    name=node.fun.id,
                    type=teir.FunctionType(args=tuple(arg.type for arg in args), ret=ret),
                )
                return teir.FunCall(type=fun.type.ret, fun=fun, args=args)
            if node.fun.id == "make_tuple":
                dims = common_dims(arg.type.dims for arg in args)
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
            assert len(shifts) % 2 == 0
            dims_dict = {d.name: (d.start, d.stop) for d in args[0].type.dims}
            offset_provider = kwargs["offset_provider"]
            for dim, offset in zip(shifts[::2], shifts[1::2]):
                d = offset_provider[dim.value].value
                if d in dims_dict:
                    start, stop = dims_dict[d]
                    dims_dict[d] = (start - offset.value, stop - offset.value)
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
        call = itir.FunCall(fun=node.stencil, args=node.inputs)
        stencil = self.visit(call, **kwargs).fun
        if output.type != stencil.type.ret:
            raise NotImplementedError()
        return teir.StencilClosure(stencil=stencil, output=output, inputs=inputs)

    def visit_FencilDefinition(self, node, *, args, offset_provider):
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
            if isinstance(x, LocatedFieldImpl):
                shape = x.array().shape
                dims = tuple(
                    teir.Dim(name=d.value, start=-o, stop=s + -o)
                    for d, o, s in zip(x.axes, x.offsets, shape)
                )
                return teir.TensorType(dims=dims, dtype=get_dtype(x.array().dtype))
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
            self.visit(node.closures, symtypes=argtypes, offset_provider=offset_provider)
        )

        return teir.Fencil(id=node.id, params=params, closures=closures)
