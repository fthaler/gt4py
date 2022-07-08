import eve
from functional.tensor import ir


class NormalizeShifts(eve.NodeTranslator):
    def visit_FunCall(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.fun, ir.FunCall)
            and node.fun.fun == ir.SymRef(id="shift")
            and len(node.args) == 1
            and isinstance(node.args[0], ir.FunCall)
            and isinstance(node.args[0].fun, ir.FunCall)
            and node.args[0].fun.fun == ir.SymRef(id="shift")
        ):
            return ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="shift"), args=node.args[0].fun.args + node.fun.args
                ),
                args=node.args[0].args,
            )
        return node


class PushShifts(eve.NodeTranslator):
    def visit_FunCall(self, node):
        if not isinstance(node, ir.Lambda):
            return self.generic_visit(node)
