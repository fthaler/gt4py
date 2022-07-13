import jax
import numpy as np
from jax import numpy as jnp

import eve
from functional.tensor import ir


class JaxEvaluator(eve.NodeTranslator):
    def visit_SymRef(self, node, *, syms, **kwargs):
        return syms[node.id], node.type

    def visit_FunCall(self, node, **kwargs):
        fun, _ = self.visit(node.fun, **kwargs)
        args = [arg[0] for arg in self.visit(node.args, **kwargs)]
        if hasattr(fun, "lower"):
            print(fun.lower(*args).compiler_ir())
        return fun(*args), node.type

    def visit_Builtin(self, node, **kwargs):
        raise NotImplementedError()

    def visit_Lambda(self, node, *, syms, **kwargs):
        @jax.jit
        def fun(*args):
            assert len(args) == len(node.params)
            return self.visit(
                node.expr, syms=syms | {p.id: a for p, a in zip(node.params, args)}, **kwargs
            )[0]

        return fun, node.type

    def visit_StencilClosure(self, node, argmap, **kwargs):
        fun = ir.FunCall(fun=node.stencil, args=node.inputs, type=node.output.type)
        out, outtype = self.visit(fun, **kwargs)
        argmap[node.output.id].array()[...] = np.asarray(out)

    def visit_Fencil(self, node, *, args, offset_provider):
        argmap = {p.id: a for p, a in zip(node.params, args)}
        syms = {k: jnp.asarray(v) for k, v in argmap.items()}

        self.visit(node.closures, argmap=argmap, syms=syms, offset_provider=offset_provider)
