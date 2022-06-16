import functools

import eve
from functional.tensor import ir
from functional.tensor.collect_shifts import CollectShifts, TensorTracer


def _offset_sum(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


class ShiftsToSlices(eve.NodeTranslator):
    def visit_SymRef(self, node, *, offsets, shape, offset, do_slice):
        if do_slice:
            offs = _offset_sum(offsets[node.id], offset)
            dims = tuple(set(offs) | set(shape))
            starts = tuple(offs.get(dim, 0) for dim in dims)
            stops = tuple(shape[dim] + start for dim, start in zip(dims, starts))
            return ir.Slice(expr=node, dims=dims, starts=starts, stops=stops)
        return node

    def visit_FunCall(self, node, *, offsets, shape, offset, do_slice):
        if isinstance(node.fun, ir.FunCall) and node.fun.fun == ir.SymRef(id="shift"):
            shift_offsets = node.fun.args
            offset = offset.copy()
            for dim, off in zip(shift_offsets[::2], shift_offsets[1::2]):
                offset[dim.value] = offset.get(dim.value, 0) + int(off.value)
            return self.visit(
                node.args[0], offsets=offsets, shape=shape, offset=offset, do_slice=do_slice
            )
        if isinstance(node.fun, ir.Lambda):
            inner_offsets = {
                k: v for k, v in offsets.items() if k not in {p.id for p in node.fun.params}
            } | {
                k.id: offsets[v.id]
                for k, v in zip(node.fun.params, node.args)
                if isinstance(v, ir.SymRef) and v.id in offsets
            }
            expr = self.visit(
                node.fun.expr, offsets=inner_offsets, shape=shape, offset=offset, do_slice=do_slice
            )
            args = self.visit(
                node.args, offsets=offsets, shape=shape, offset=offset, do_slice=False
            )
            return ir.FunCall(fun=ir.Lambda(expr=expr, params=node.fun.params), args=args)
        if isinstance(node.fun, ir.SymRef) and node.fun.id in (
            "plus",
            "minus",
            "multiplies",
            "divides",
            "if_",
            "greater",
        ):
            return ir.FunCall(
                fun=node.fun,
                args=self.visit(
                    node.args, offsets=offsets, shape=shape, offset=offset, do_slice=do_slice
                ),
            )
        raise NotImplementedError()

    def visit_StencilClosure(self, node, *, offsets, shape, offset):
        wrapped = self.visit(
            ir.FunCall(fun=node.stencil, args=node.inputs),
            offsets=offsets,
            shape=shape,
            offset=offset,
            do_slice=True,
        )
        return ir.StencilClosure(
            stencil=wrapped.fun,
            inputs=wrapped.args,
            output=self.visit(
                node.output, offsets=offsets, shape=shape, offset=offset, do_slice=True
            ),
        )

    def visit_Fencil(self, node):
        closure_shifts = []
        for closure in node.closures:
            args = {str(p.id): TensorTracer() for p in node.params}
            CollectShifts().visit(closure, syms=args)
            closure_shifts.append({k: v.minmax for k, v in args.items()})

        minmax_shifts = functools.reduce(
            lambda x, y: {
                k: (min((xk := x.get(k, 0)), (yk := y.get(k, 0))), max(xk, yk))
                for k in set(x) | set(y)
            },
            closure_shifts,
        )

        global_offsets = {
            sym: {k: -v[0] for k, v in shifts.items()} for sym, shifts in minmax_shifts.items()
        }

        closures = []
        for closure in node.closures:
            domain = closure.annex.domain
            domain_offset = {k: v[0] for k, v in domain.items()}
            domain_shape = {k: v[1] for k, v in domain.items()}
            offsets = {
                inp.id: _offset_sum(global_offsets[inp.id], domain_offset) for inp in closure.inputs
            }
            offsets[closure.output.id] = _offset_sum(
                global_offsets[closure.output.id], domain_offset
            )
            offset = dict()
            closures.append(self.visit(closure, offsets=offsets, shape=domain_shape, offset=offset))

        return ir.Fencil(id=node.id, params=node.params, closures=tuple(closures))
