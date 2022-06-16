import eve
from functional.iterator import ir


def _lit(value):
    assert isinstance(value, (int, float, str))
    return ir.Literal(value=str(value), type=type(value).__name__)


def _unlit(node):
    assert isinstance(node, ir.Literal)
    if node.type == "int":
        return int(node.value)
    if node.type == "float":
        return float(node.value)
    if node.type == "str":
        return node.value
    if node.type == "bool":
        assert node.value in ("True", "False")
        return node.value == "True"
    raise NotImplementedError()


class ConstantPropagation(eve.NodeTranslator):
    def visit_SymRef(self, node, *, constants):
        return constants.get(node.id, node)

    def visit_FunCall(self, node, *, constants):
        if isinstance(node.fun, ir.Lambda):
            args = self.visit(node.args, constants=constants)
            inner_constants = {
                k: v for k, v in constants.items() if k not in {p.id for p in node.fun.params}
            }
            non_const_params = []
            non_const_args = []
            for param, arg in zip(node.fun.params, args):
                if isinstance(arg, (ir.Literal, ir.AxisLiteral, ir.OffsetLiteral)):
                    inner_constants[param.id] = arg
                else:
                    non_const_params.append(param)
                    non_const_args.append(arg)
            expr = self.visit(node.fun.expr, constants=inner_constants)
            if isinstance(expr, ir.Literal) or not non_const_params:
                return expr
            if len(non_const_params) == 1 and expr == ir.SymRef(id=non_const_params[0].id):
                return non_const_args[0]
            return ir.FunCall(
                fun=ir.Lambda(params=non_const_params, expr=expr), args=non_const_args
            )

        node = self.generic_visit(node, constants=constants)
        if not all(isinstance(arg, ir.Literal) for arg in node.args):
            return node
        if node.fun == ir.SymRef(id="shift"):
            return node
        if node.fun == ir.SymRef(id="plus"):
            return _lit(_unlit(node.args[0]) + _unlit(node.args[1]))
        if node.fun == ir.SymRef(id="minus"):
            return _lit(_unlit(node.args[0]) - _unlit(node.args[1]))
        if node.fun == ir.SymRef(id="multiplies"):
            return _lit(_unlit(node.args[0]) * _unlit(node.args[1]))
        if node.fun == ir.SymRef(id="divides"):
            return _lit(_unlit(node.args[0]) / _unlit(node.args[1]))
        if node.fun == ir.SymRef(id="greater"):
            return _lit(_unlit(node.args[0]) > _unlit(node.args[1]))
        if node.fun == ir.SymRef(id="less"):
            return _lit(_unlit(node.args[0]) < _unlit(node.args[1]))
        if node.fun == ir.SymRef(id="if_"):
            return _lit(_unlit(node.args[1]) if _unlit(node.args[0]) else _unlit(node.args[2]))
        raise NotImplementedError()
