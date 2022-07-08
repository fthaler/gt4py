from functional.common import Dimension
from functional.iterator import ir as itir
from functional.iterator.processor_interface import fencil_executor
from functional.iterator.transforms.constant_propagation import ConstantPropagation
from functional.iterator.transforms.pass_manager import apply_common_transforms
from functional.iterator.transforms.remap_symbols import RemapSymbolRefs
from functional.tensor.lifter import Lifter


def _inline_domain(root, args):
    new_params = []
    new_args = []
    symbol_map = dict()
    for param, arg in zip(root.params, args):
        if isinstance(arg, dict) and all(isinstance(k, Dimension) for k in arg):
            symbol_map[param.id] = itir.FunCall(
                fun=itir.SymRef(id="cartesian_domain"),
                args=[
                    itir.FunCall(
                        fun=itir.SymRef(id="named_range"),
                        args=[
                            itir.AxisLiteral(value=k.value),
                            itir.Literal(value=str(v.start), type="int"),
                            itir.Literal(value=str(v.stop), type="int"),
                        ],
                    )
                    for k, v in arg.items()
                ],
            )
        else:
            new_params.append(param)
            new_args.append(arg)
    assert not root.function_definitions
    closures = RemapSymbolRefs().visit(root.closures, symbol_map=symbol_map)
    return (
        itir.FencilDefinition(
            id=root.id,
            function_definitions=[],
            params=new_params,
            closures=closures,
        ),
        new_args,
    )


@fencil_executor
def run(root, *args, **kwargs):
    root = apply_common_transforms(root)

    constants = {
        p.id: itir.Literal(value=str(a), type=type(a).__name__)
        for p, a in zip(root.params, args)
        if isinstance(a, (int, float))
    }
    root = ConstantPropagation().visit(root, constants=constants)
    root, args = _inline_domain(root, args)

    lifted = Lifter().visit(root, args=args, offset_provider=kwargs["offset_provider"])
    print(lifted)
