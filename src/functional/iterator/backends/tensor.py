from functional.iterator import ir as itir
from functional.iterator.backends import backend
from functional.iterator.embedded import LocatedField
from functional.iterator.transforms.constant_propagation import ConstantPropagation
from functional.iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from functional.tensor import jaxeval, lifting, shifts_to_slices, typing


def argtype(arg):
    assert isinstance(arg, LocatedField)
    return typing.Tensor(
        element=arg.dtype.name,
        sizes=tuple((a.value, s) for a, s in zip(arg.axises, arg.array().shape)),
    )


def lift(prog, *args, offset_provider, **kwargs):
    prog = InlineFundefs().visit(prog)
    prog = PruneUnreferencedFundefs().visit(prog)

    constants = {
        param.id: itir.Literal(value=str(arg), type=type(arg).__name__)
        for param, arg in zip(prog.params, args)
        if not isinstance(arg, LocatedField)
    }
    prog = ConstantPropagation().visit(prog, constants=constants)
    lifted = lifting.Lifter().visit(prog, offset_provider=offset_provider)

    sliced = shifts_to_slices.ShiftsToSlices().visit(lifted)
    jaxeval.JaxEvaluator().visit(sliced, args=args, offset_provider=offset_provider)


backend.register_backend("tensor", lift)
