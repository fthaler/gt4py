import eve
from functional.iterator import ir as itir
from functional.tensor import ir as teir


class Lifter(eve.NodeTranslator):
    def visit_Sym(self, node):
        return teir.Sym(id=node.id)

    def visit_SymRef(self, node):
        if node.id in ("lift", "deref"):
            return teir.Lambda(params=(teir.Sym(id="x"),), expr=teir.SymRef(id="x"))
        return teir.SymRef(id=node.id)

    def visit_Literal(self, node):
        return teir.Literal(value=node.value, type=node.type)

    def visit_OffsetLiteral(self, node):
        return teir.Literal(value=str(node.value), type=type(node.value).__name__)

    def visit_Lambda(self, node):
        return teir.Lambda(params=tuple(self.visit(node.params)), expr=self.visit(node.expr))

    def visit_FunCall(self, node):
        if isinstance(node.fun, itir.SymRef) and node.fun.id in ("lift", "deref"):
            assert len(node.args) == 1
            return self.visit(node.args[0])
        return teir.FunCall(fun=self.visit(node.fun), args=tuple(self.visit(node.args)))

    def visit_FunctionDefinition(self, node):
        return teir.FunDef(
            id=node.id,
            fun=teir.Lambda(params=tuple(self.visit(node.params)), expr=self.visit(node.expr)),
        )

    def visit_StencilClosure(self, node, *, offset_provider):
        res = teir.StencilClosure(
            stencil=self.visit(node.stencil),
            output=self.visit(node.output),
            inputs=tuple(self.visit(node.inputs)),
        )
        if not isinstance(node.domain, itir.FunCall) or node.domain.fun != itir.SymRef(id="domain"):
            return NotImplementedError()
        domain_ranges = node.domain.args
        domain = dict()
        dim_map = {v.value: k for k, v in offset_provider.items()}
        for dr in domain_ranges:
            if not isinstance(dr, itir.FunCall) or dr.fun != itir.SymRef(id="named_range"):
                raise NotImplementedError()
            dim, start, stop = dr.args
            if not isinstance(start, itir.Literal) or not isinstance(stop, itir.Literal):
                raise NotImplementedError()
            domain[dim_map[dim.value]] = (int(start.value), int(stop.value))
        res.annex.domain = domain
        return res

    def visit_FencilDefinition(self, node, *, offset_provider):
        assert not node.function_definitions
        return teir.Fencil(
            id=node.id,
            params=tuple(self.visit(node.params)),
            closures=tuple(self.visit(node.closures, offset_provider=offset_provider)),
        )
