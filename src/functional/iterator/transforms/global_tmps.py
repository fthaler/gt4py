from typing import Any, Optional

from eve import Node, NodeTranslator
from eve.traits import SymbolTableTrait
from functional.iterator import ir, type_inference
from functional.iterator.runtime import CartesianAxis
from functional.iterator.transforms.collect_shifts import CollectShifts
from functional.iterator.transforms.eta_reduction import EtaReduction
from functional.iterator.transforms.popup_tmps import PopupTmps
from functional.iterator.transforms.prune_closure_inputs import PruneClosureInputs


AUTO_DOMAIN = ir.SymRef(id="_gtmp_auto_domain")


class Temporary(Node):
    id: ir.SymbolName  # noqa: A003
    domain: Optional[ir.Expr] = None
    dtype: Optional[Any] = None


class FencilWithTemporaries(Node, SymbolTableTrait):
    fencil: ir.FencilDefinition
    params: list[ir.Sym]
    tmps: list[Temporary]


def split_closures(node: ir.FencilDefinition) -> FencilWithTemporaries:
    tmps: list[ir.SymRef] = []

    def handle_arg(arg):
        if isinstance(arg, ir.SymRef):
            return arg
        if (
            isinstance(arg, ir.FunCall)
            and isinstance(arg.fun, ir.FunCall)
            and isinstance(arg.fun.fun, ir.SymRef)
            and arg.fun.fun.id == "lift"
        ):
            assert len(arg.fun.args) == 1
            ref = ir.SymRef(id=f"_gtmp_{len(tmps)}")
            tmps.append(ir.Sym(id=ref.id))
            unlifted = ir.FunCall(fun=arg.fun.args[0], args=arg.args)
            todos.append((ref, unlifted))
            return ref
        raise AssertionError()

    closures = []
    for closure in reversed(node.closures):
        wrapped_stencil = ir.FunCall(fun=closure.stencil, args=closure.inputs)
        popped_stencil = PopupTmps().visit(wrapped_stencil)

        todos = [(closure.output, popped_stencil)]
        domain = closure.domain
        while todos:
            output, call = todos.pop()
            closure = ir.StencilClosure(
                domain=domain,
                stencil=call.fun,
                output=output,
                inputs=[handle_arg(arg) for arg in call.args],
            )
            closures.append(closure)
            domain = AUTO_DOMAIN

    return FencilWithTemporaries(
        fencil=ir.FencilDefinition(
            id=node.id,
            function_definitions=node.function_definitions,
            params=node.params + tmps + [ir.Sym(id="_gtmp_auto_domain")],
            closures=list(reversed(closures)),
        ),
        params=node.params,
        tmps=[Temporary(id=tmp.id) for tmp in tmps],
    )


def _collect_stencil_shifts(stencil, return_params=False):
    if isinstance(stencil, ir.FunCall) and stencil.fun == ir.SymRef(id="scan"):
        # get params of scan function, but ignore accumulator
        fun = stencil.args[0]
        params = fun.params[1:]
    else:
        assert isinstance(stencil, ir.Lambda)
        fun = stencil
        params = fun.params
    shifts: dict[str, list[tuple]] = dict()
    CollectShifts().visit(fun, shifts=shifts)
    if return_params:
        return shifts, params
    return shifts


def update_cartesian_domains(node: FencilWithTemporaries, offset_provider) -> FencilWithTemporaries:
    def extend(domain, shifts):
        assert isinstance(domain, ir.FunCall) and domain.fun == ir.SymRef(id="domain")
        if not shifts:
            return domain
        assert all(isinstance(axis, CartesianAxis) for axis in offset_provider.values())

        offset_limits = {k: (0, 0) for k in offset_provider.keys()}
        for shift in shifts:
            offsets = {k: 0 for k in offset_provider.keys()}
            for k, v in zip(shift[0::2], shift[1::2]):
                offsets[k.value] += v.value
            for k, v in offsets.items():
                old_min, old_max = offset_limits[k]
                offset_limits[k] = (min(old_min, v), max(old_max, v))

        offset_limits = {v.value: offset_limits[k] for k, v in offset_provider.items()}

        named_ranges = []
        for named_range in domain.args:
            assert (
                isinstance(named_range, ir.FunCall)
                and isinstance(named_range.fun, ir.SymRef)
                and named_range.fun.id == "named_range"
            )
            axis_literal, lower_bound, upper_bound = named_range.args
            assert isinstance(axis_literal, ir.AxisLiteral)

            lower_offset, upper_offset = offset_limits.get(axis_literal.value, (0, 0))
            named_ranges.append(
                ir.FunCall(
                    fun=named_range.fun,
                    args=[
                        axis_literal,
                        ir.FunCall(
                            fun=ir.SymRef(id="plus"),
                            args=[lower_bound, ir.IntLiteral(value=lower_offset)],
                        )
                        if lower_offset
                        else lower_bound,
                        ir.FunCall(
                            fun=ir.SymRef(id="plus"),
                            args=[upper_bound, ir.IntLiteral(value=upper_offset)],
                        )
                        if upper_offset
                        else upper_bound,
                    ],
                )
            )

        return ir.FunCall(fun=domain.fun, args=named_ranges)

    closures = []
    shifts: dict[str, list[tuple]] = dict()
    domain = None
    for closure in reversed(node.fencil.closures):
        if closure.domain == AUTO_DOMAIN:
            output_shifts = shifts.get(closure.output.id, [])
            domain = extend(domain, output_shifts)
            closure = ir.StencilClosure(
                domain=domain, stencil=closure.stencil, output=closure.output, inputs=closure.inputs
            )
        else:
            domain = closure.domain
            shifts = dict()

        closures.append(closure)

        if closure.stencil == ir.SymRef(id="deref"):
            continue

        local_shifts, params = _collect_stencil_shifts(closure.stencil, return_params=True)
        input_map = {param.id: inp.id for param, inp in zip(params, closure.inputs)}
        for param, shift in local_shifts.items():
            shifts.setdefault(input_map[param], []).extend(shift)

    return FencilWithTemporaries(
        fencil=ir.FencilDefinition(
            id=node.fencil.id,
            function_definitions=node.fencil.function_definitions,
            params=node.fencil.params[:-1],
            closures=list(reversed(closures)),
        ),
        params=node.params,
        tmps=node.tmps,
    )


def update_unstructured_domains(node: FencilWithTemporaries, offset_provider):
    known_domains = {
        closure.output.id: closure.domain
        for closure in node.fencil.closures
        if closure.domain != AUTO_DOMAIN
    }

    closures = []
    for closure in node.fencil.closures:
        if closure.domain == AUTO_DOMAIN:
            if closure.stencil == ir.SymRef(id="deref"):
                domain = known_domains[closure.inputs[0].id]
            else:
                shifts = _collect_stencil_shifts(closure.stencil)
                first_connectivity = {c[0].value for s in shifts.values() for c in s if c}
                if not first_connectivity:
                    # for now we assume that all inputs have the same domain
                    assert all(
                        known_domains[inp.id] == known_domains[closure.inputs[0].id]
                        for inp in closure.inputs
                    )
                    domain = known_domains[closure.inputs[0].id]
                else:
                    assert len(first_connectivity) == 1
                    conn = offset_provider[next(iter(first_connectivity))]
                    axis = conn.origin_axis
                    size = conn.tbl.shape[0]
                    domain = ir.FunCall(
                        fun=ir.SymRef(id="domain"),
                        args=[
                            ir.FunCall(
                                fun=ir.SymRef(id="named_range"),
                                args=[
                                    ir.AxisLiteral(value=axis.value),
                                    ir.IntLiteral(value=0),
                                    ir.IntLiteral(value=size),
                                ],
                            )
                        ],
                    )

            known_domains[closure.output.id] = domain
            closure = ir.StencilClosure(
                domain=domain, stencil=closure.stencil, output=closure.output, inputs=closure.inputs
            )

        closures.append(closure)
    return FencilWithTemporaries(
        fencil=ir.FencilDefinition(
            id=node.fencil.id,
            function_definitions=node.fencil.function_definitions,
            params=node.fencil.params[:-1],
            closures=closures,
        ),
        params=node.params,
        tmps=node.tmps,
    )


def collect_tmps_info(node: FencilWithTemporaries):
    tmps = {tmp.id for tmp in node.tmps}
    domains = {
        closure.output.id: closure.domain
        for closure in node.fencil.closures
        if closure.output.id in tmps
    }

    def convert_type(dtype):
        if isinstance(dtype, type_inference.Primitive):
            return dtype.name
        if isinstance(dtype, type_inference.Var):
            return dtype.idx
        if isinstance(dtype, type_inference.PartialTupleVar):
            elems_dict = dict(dtype.elems)
            assert len(elems_dict) == max(elems_dict) + 1
            return tuple(convert_type(elems_dict[i]) for i in range(len(elems_dict)))
        assert isinstance(dtype, type_inference.Tuple)
        return tuple(convert_type(e) for e in dtype.elems)

    fencil_type = type_inference.infer(node.fencil)
    assert isinstance(fencil_type, type_inference.Fencil)
    assert isinstance(fencil_type.params, type_inference.Tuple)
    all_types = []
    types = dict()
    for param, dtype in zip(node.fencil.params, fencil_type.params.elems):
        assert isinstance(dtype, type_inference.Val)
        all_types.append(convert_type(dtype.dtype))
        if param.id in tmps:
            assert param.id not in types
            t = all_types[-1]
            types[param.id] = all_types.index(t) if isinstance(t, int) else t

    return FencilWithTemporaries(
        fencil=node.fencil,
        params=node.params,
        tmps=[
            Temporary(id=tmp.id, domain=domains[tmp.id], dtype=types[tmp.id]) for tmp in node.tmps
        ],
    )


class CreateGlobalTmps(NodeTranslator):
    def visit_FencilDefinition(
        self, node: ir.FencilDefinition, *, offset_provider
    ) -> FencilWithTemporaries:
        node = split_closures(node)
        node = PruneClosureInputs().visit(node)
        node = EtaReduction().visit(node)
        if all(isinstance(o, CartesianAxis) for o in offset_provider.values()):
            node = update_cartesian_domains(node, offset_provider)
        else:
            node = update_unstructured_domains(node, offset_provider)
        return collect_tmps_info(node)