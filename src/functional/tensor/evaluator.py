import eve


class Evaluator(eve.NodeTranslator):
    def visit_SymRef(self, node, *, syms, **kwargs):
        return syms[node.id]

    def visit_Literal(self, node, **kwargs):
        if node.type == "str":
            return node.value
        if node.type == "int":
            return int(node.value)
        if node.type == "float":
            return float(node.value)
        raise NotImplementedError()

    def visit_FunCall(self, node, **kwargs):
        fun = self.visit(node.fun, **kwargs)
        args = self.visit(node.args, **kwargs)
        return fun(*args)

    def visit_Lambda(self, node, *, syms, **kwargs):
        def fun(*args):
            assert len(args) == len(node.params)
            return self.visit(
                node.expr,
                syms=syms | {param.id: arg for param, arg in zip(node.params, args)},
                **kwargs,
            )

        return fun
