from functional.iterator import ir as itir
from functional.iterator.backends import backend
from functional.iterator.embedded import LocatedField
from functional.iterator.transforms.constant_propagation import ConstantPropagation
from functional.iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from functional.iterator.transforms.inline_lambdas import InlineLambdas
from functional.tensor import jaxeval, lifting, shifts_to_slices


def lift(prog, *args, offset_provider, **kwargs):
    prog = InlineFundefs().visit(prog)
    prog = PruneUnreferencedFundefs().visit(prog)
    prog = InlineLambdas().visit(prog)

    def is_constant(arg):
        if isinstance(arg, LocatedField):
            return False
        if isinstance(arg, tuple):
            return all(is_constant(a) for a in arg)
        return True

    constants = {
        param.id: itir.Literal(value=str(arg), type=type(arg).__name__)
        for param, arg in zip(prog.params, args)
        if is_constant(arg)
    }
    prog = ConstantPropagation().visit(prog, constants=constants)
    lifted = lifting.Lifter().visit(prog, offset_provider=offset_provider)

    sliced = shifts_to_slices.ShiftsToSlices().visit(lifted)
    jaxeval.JaxEvaluator().visit(sliced, args=args)


backend.register_backend("tensor", lift)
