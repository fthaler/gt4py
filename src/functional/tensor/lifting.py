import eve
from functional.iterator import ir as itir
from functional.tensor import ir as teir


class Lifter(eve.NodeTranslator):
    def visit_Sym(self, node, **kwargs):
        return teir.Sym(id=node.id)

    def visit_SymRef(self, node, **kwargs):
        if node.id in ("lift", "deref"):
            return teir.Lambda(params=(teir.Sym(id="x"),), expr=teir.SymRef(id="x"))
        return teir.SymRef(id=node.id)

    def visit_Literal(self, node, **kwargs):
        return teir.Literal(value=node.value, type=node.type)

    def visit_OffsetLiteral(self, node, *, offset_provider, **kwargs):
        if node.value in offset_provider:
            value = offset_provider[node.value].value
        else:
            value = node.value
        return teir.Literal(value=str(value), type=type(value).__name__)

    def visit_Lambda(self, node, **kwargs):
        return teir.Lambda(
            params=tuple(self.visit(node.params, **kwargs)), expr=self.visit(node.expr, **kwargs)
        )

    def visit_FunCall(self, node, **kwargs):
        if isinstance(node.fun, itir.SymRef) and node.fun.id in ("lift", "deref"):
            assert len(node.args) == 1
            return self.visit(node.args[0], **kwargs)
        return teir.FunCall(
            fun=self.visit(node.fun, **kwargs), args=tuple(self.visit(node.args, **kwargs))
        )

    def visit_FunctionDefinition(self, node, **kwargs):
        return teir.FunDef(
            id=node.id,
            fun=teir.Lambda(
                params=tuple(self.visit(node.params, **kwargs)),
                expr=self.visit(node.expr, **kwargs),
            ),
        )

    def visit_StencilClosure(self, node, **kwargs):
        res = teir.StencilClosure(
            stencil=self.visit(node.stencil, **kwargs),
            output=self.visit(node.output, **kwargs),
            inputs=tuple(self.visit(node.inputs, **kwargs)),
        )
        if isinstance(node.domain, itir.FunCall) and node.domain.fun == itir.SymRef(id="domain"):
            domain_ranges = node.domain.args
            domain = dict()
            for dr in domain_ranges:
                if not isinstance(dr, itir.FunCall) or dr.fun != itir.SymRef(id="named_range"):
                    raise NotImplementedError()
                dim, start, stop = dr.args
                if not isinstance(start, itir.Literal) or not isinstance(stop, itir.Literal):
                    raise NotImplementedError()
                domain[dim.value] = (int(start.value), int(stop.value))
        elif isinstance(node.domain, itir.Literal):
            import functional.common

            domain_dict = eval(node.domain.value, {"Dimension": functional.common.Dimension})
            domain = {k.value: (v.start, v.stop) for k, v in domain_dict.items()}
        else:
            raise NotImplementedError()
        res.annex.domain = domain
        return res

    def visit_FencilDefinition(self, node, **kwargs):
        assert not node.function_definitions
        return teir.Fencil(
            id=node.id,
            params=tuple(self.visit(node.params, **kwargs)),
            closures=tuple(self.visit(node.closures, **kwargs)),
        )
