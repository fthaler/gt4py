from dataclasses import dataclass, field

from functional.tensor import evaluator, ir


@dataclass
class Tracer:
    ...


@dataclass
class TensorTracer(Tracer):
    pos: dict[str, int] = field(default_factory=dict)
    minmax: dict[str, tuple[int, int]] = field(default_factory=dict)

    def shift(self, offsets):
        pos = self.pos.copy()
        for dim, offset in zip(offsets[::2], offsets[1::2]):
            pos[dim] = pos.get(dim, 0) + offset
        for dim in set(self.minmax) | set(self.pos):
            cmin, cmax = self.minmax.get(dim, (0, 0))
            cpos = pos.get(dim, 0)
            self.minmax[dim] = min(cmin, cpos), max(cmax, cpos)
        return TensorTracer(pos=pos, minmax=self.minmax)


@dataclass
class MergedTracer(Tracer):
    tracers: list[Tracer]

    def shift(self, offsets):
        return MergedTracer(tracers=[t.shift(offsets) for t in self.tracers])


class CollectShifts(evaluator.Evaluator):
    def visit_SymRef(self, node, *, syms):
        if node.id in ("greater", "if_", "minus", "multiplies", "plus"):

            def fun(*args):
                return MergedTracer([arg for arg in args if isinstance(arg, Tracer)])

            return fun
        if node.id == "shift":

            def shift(*offsets):
                def res(x):
                    return x.shift(offsets)

                return res

            return shift

        return syms[node.id]

    def visit_StencilClosure(self, node, *, syms):
        wrapped = ir.FunCall(fun=node.stencil, args=node.inputs)
        self.visit(node.output, syms=syms)
        self.visit(wrapped, syms=syms)

    def visit_Fencil(self, node):
        args = {str(p.id): TensorTracer() for p in node.params}
        for closure in node.closures:
            self.visit(closure, syms=args)

        return {k: v.minmax for k, v in args.items()}
