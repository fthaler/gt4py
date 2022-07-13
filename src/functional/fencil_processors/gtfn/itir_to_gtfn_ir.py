from typing import Any, Iterable, Optional, Type, Union

import eve
from eve.concepts import SymbolName
from eve.utils import UIDs
from functional.common import Connectivity, Dimension
from functional.fencil_processors.gtfn.gtfn_ir import (
    Backend,
    BinaryExpr,
    CartesianDomain,
    Expr,
    FencilDefinition,
    FunCall,
    FunctionDefinition,
    GridType,
    Lambda,
    Literal,
    Node,
    OffsetLiteral,
    Scan,
    ScanExecution,
    ScanPassDefinition,
    StencilExecution,
    Sym,
    SymRef,
    TaggedValues,
    TemporaryAllocation,
    TernaryExpr,
    UnaryExpr,
    UnstructuredDomain,
)
from functional.iterator import ir as itir


def pytype_to_cpptype(t: str):
    try:
        return {
            "float": "double",
            "float32": "float",
            "float64": "double",
            "int": "int",
            "int32": "std::int32_t",
            "int64": "std::int64_t",
            "bool": "bool",
            "axis_literal": None,  # TODO: domain?
        }[t]
    except KeyError:
        raise TypeError(f"Unsupported type '{t}'") from None


class GTFN_lowering(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    _binary_op_map = {
        "plus": "+",
        "minus": "-",
        "multiplies": "*",
        "divides": "/",
        "eq": "==",
        "less": "<",
        "greater": ">",
        "and_": "&&",
        "or_": "||",
    }
    _unary_op_map = {"not_": "!"}

    def visit_Sym(self, node: itir.Sym, **kwargs: Any) -> Sym:
        return Sym(id=node.id)

    def visit_SymRef(
        self,
        node: itir.SymRef,
        force_function_extraction: bool = False,
        extracted_functions: Optional[list] = None,
        **kwargs: Any,
    ) -> SymRef:
        if force_function_extraction:
            assert extracted_functions is not None
            assert node.id == "deref"
            fun_id = UIDs.sequential_id(prefix="_fun")
            fun_def = FunctionDefinition(
                id=fun_id,
                params=[Sym(id="x")],
                expr=FunCall(fun=SymRef(id="deref"), args=[SymRef(id="x")]),
            )
            extracted_functions.append(fun_def)
            return SymRef(id=fun_id)
        return SymRef(id=node.id)

    def visit_Lambda(
        self,
        node: itir.Lambda,
        *,
        force_function_extraction: bool = False,
        extracted_functions: Optional[list] = None,
        **kwargs: Any,
    ) -> Union[SymRef, Lambda]:
        if force_function_extraction:
            assert extracted_functions is not None
            fun_id = UIDs.sequential_id(prefix="_fun")
            fun_def = FunctionDefinition(
                id=fun_id,
                params=self.visit(node.params, **kwargs),
                expr=self.visit(node.expr, **kwargs),
            )
            extracted_functions.append(fun_def)
            return SymRef(id=fun_id)
        return Lambda(
            params=self.visit(node.params, **kwargs), expr=self.visit(node.expr, **kwargs)
        )

    def visit_Literal(self, node: itir.Literal, **kwargs: Any) -> Literal:
        return Literal(value=node.value, type=node.type)

    def visit_OffsetLiteral(self, node: itir.OffsetLiteral, **kwargs: Any) -> OffsetLiteral:
        if node.value in self.offset_provider:
            if isinstance(
                self.offset_provider[node.value], Dimension
            ):  # replace offset tag by dimension tag
                return OffsetLiteral(value=self.offset_provider[node.value].value)
        return OffsetLiteral(value=node.value)

    def visit_AxisLiteral(self, node: itir.AxisLiteral, **kwargs: Any) -> Literal:
        return Literal(value=node.value, type="axis_literal")

    @staticmethod
    def _is_sparse_deref_shift(node: itir.FunCall) -> bool:
        return (
            node.fun == itir.SymRef(id="deref")
            and isinstance(node.args[0], itir.FunCall)
            and isinstance(node.args[0].fun, itir.FunCall)
            and node.args[0].fun.fun == itir.SymRef(id="shift")
            and bool(len(node.args[0].fun.args) % 2)
        )

    def _sparse_deref_shift_to_tuple_get(self, node: itir.FunCall) -> Expr:
        # deref(shift(i)(sparse)) -> tuple_get(i, deref(sparse))
        # TODO: remove once ‘real’ sparse field handling is available
        assert isinstance(node.args[0], itir.FunCall)
        assert isinstance(node.args[0].fun, itir.FunCall)
        offsets = node.args[0].fun.args
        deref_arg = node.args[0].args[0]
        if len(offsets) > 1:
            deref_arg = itir.FunCall(
                fun=itir.FunCall(fun=itir.SymRef(id="shift"), args=offsets[:-1]),
                args=[deref_arg],
            )
        derefed = itir.FunCall(fun=itir.SymRef(id="deref"), args=[deref_arg])
        sparse_access = itir.FunCall(fun=itir.SymRef(id="tuple_get"), args=[offsets[-1], derefed])
        return self.visit(sparse_access)

    def _make_domain(self, node: itir.FunCall):
        tags = []
        sizes = []
        offsets = []
        for named_range in node.args:
            if not (
                isinstance(named_range, itir.FunCall)
                and named_range.fun == itir.SymRef(id="named_range")
            ):
                raise ValueError("Arguments to `domain` need to be calls to `named_range`.")
            tags.append(self.visit(named_range.args[0]))
            sizes.append(
                BinaryExpr(
                    op="-", lhs=self.visit(named_range.args[2]), rhs=self.visit(named_range.args[1])
                )
            )
            offsets.append(self.visit(named_range.args[1]))
        return TaggedValues(tags=tags, values=sizes), TaggedValues(tags=tags, values=offsets)

    @staticmethod
    def _collect_offset_or_axis_node(
        node_type: Type, tree: eve.Node | Iterable[eve.Node]
    ) -> set[str]:
        if not isinstance(tree, Iterable):
            tree = [tree]
        result = set()
        for n in tree:
            result.update(
                n.pre_walk_values()
                .if_isinstance(node_type)
                .getattr("value")
                .if_isinstance(str)
                .to_set()
            )
        return result

    def visit_FunCall(self, node: itir.FunCall, **kwargs: Any) -> Node:
        if isinstance(node.fun, itir.SymRef):
            if node.fun.id in self._unary_op_map:
                assert len(node.args) == 1
                return UnaryExpr(op=self._unary_op_map[node.fun.id], expr=self.visit(node.args[0]))
            elif node.fun.id in self._binary_op_map:
                assert len(node.args) == 2
                return BinaryExpr(
                    op=self._binary_op_map[node.fun.id],
                    lhs=self.visit(node.args[0], **kwargs),
                    rhs=self.visit(node.args[1], **kwargs),
                )
            elif node.fun.id == "if_":
                assert len(node.args) == 3
                return TernaryExpr(
                    cond=self.visit(node.args[0], **kwargs),
                    true_expr=self.visit(node.args[1], **kwargs),
                    false_expr=self.visit(node.args[2], **kwargs),
                )
            elif self._is_sparse_deref_shift(node):
                return self._sparse_deref_shift_to_tuple_get(node)
            elif node.fun.id == "shift":
                raise ValueError("unapplied shift call not supported: {node}")
            elif node.fun.id == "scan":
                raise ValueError("scans are only supported at the top level of a stencil closure")
            elif node.fun.id == "cartesian_domain":
                sizes, domain_offsets = self._make_domain(node)
                return CartesianDomain(tagged_sizes=sizes, tagged_offsets=domain_offsets)
            elif node.fun.id == "unstructured_domain":
                sizes, domain_offsets = self._make_domain(node)
                connectivities = []
                if "stencil" in kwargs:
                    shift_offsets = self._collect_offset_or_axis_node(
                        itir.OffsetLiteral, kwargs["stencil"]
                    )
                    for o in shift_offsets:
                        if o in self.offset_provider and isinstance(
                            self.offset_provider[o], Connectivity
                        ):
                            connectivities.append(SymRef(id=o))
                return UnstructuredDomain(
                    tagged_sizes=sizes,
                    tagged_offsets=domain_offsets,
                    connectivities=connectivities,
                )
        elif isinstance(node.fun, itir.FunCall) and node.fun.fun == itir.SymRef(id="shift"):
            assert len(node.args) == 1
            return FunCall(
                fun=self.visit(node.fun.fun, **kwargs),
                args=self.visit(node.args, **kwargs) + self.visit(node.fun.args, **kwargs),
            )
        elif isinstance(node.fun, itir.FunCall) and node.fun == itir.SymRef(id="shift"):
            raise ValueError("unapplied shift call not supported: {node}")
        return FunCall(fun=self.visit(node.fun, **kwargs), args=self.visit(node.args, **kwargs))

    def visit_FunctionDefinition(
        self, node: itir.FunctionDefinition, **kwargs: Any
    ) -> FunctionDefinition:
        return FunctionDefinition(
            id=node.id,
            params=self.visit(node.params, **kwargs),
            expr=self.visit(node.expr, **kwargs),
        )

    @staticmethod
    def _is_scan(node: itir.Node):
        return isinstance(node, itir.FunCall) and node.fun == itir.SymRef(id="scan")

    @staticmethod
    def _bool_from_literal(node: itir.Node):
        assert isinstance(node, itir.Literal)
        assert node.type == "bool" and node.value in ("True", "False")
        return node.value == "True"

    def visit_StencilClosure(
        self, node: itir.StencilClosure, extracted_functions: list, **kwargs: Any
    ) -> Union[ScanExecution, StencilExecution]:
        backend = Backend(domain=self.visit(node.domain, stencil=node.stencil, **kwargs))
        if self._is_scan(node.stencil):
            scan_id = UIDs.sequential_id(prefix="_scan")
            assert isinstance(node.stencil, itir.FunCall)
            scan_lambda = self.visit(node.stencil.args[0], **kwargs)
            forward = self._bool_from_literal(node.stencil.args[1])
            scan_def = ScanPassDefinition(
                id=scan_id, params=scan_lambda.params, expr=scan_lambda.expr, forward=forward
            )
            extracted_functions.append(scan_def)
            scan = Scan(
                function=SymRef(id=scan_id),
                output=Literal(value="0", type="int"),
                inputs=[Literal(value=str(i + 1), type="int") for i, _ in enumerate(node.inputs)],
                init=self.visit(node.stencil.args[2], **kwargs),
            )
            return ScanExecution(
                backend=backend,
                scans=[scan],
                args=[self.visit(node.output, **kwargs)] + self.visit(node.inputs),
            )
        return StencilExecution(
            stencil=self.visit(
                node.stencil,
                force_function_extraction=True,
                extracted_functions=extracted_functions,
                **kwargs,
            ),
            output=self.visit(node.output, **kwargs),
            inputs=self.visit(node.inputs, **kwargs),
            backend=backend,
        )

    @staticmethod
    def _merge_scans(
        executions: list[Union[StencilExecution, ScanExecution]]
    ) -> list[Union[StencilExecution, ScanExecution]]:
        def merge(a: ScanExecution, b: ScanExecution) -> ScanExecution:
            assert a.backend == b.backend

            index_map = dict[int, int]()
            compacted_b_args = list[SymRef]()
            for b_idx, b_arg in enumerate(b.args):
                try:
                    a_idx = a.args.index(b_arg)
                    index_map[b_idx] = a_idx
                except ValueError:
                    index_map[b_idx] = len(a.args) + len(compacted_b_args)
                    compacted_b_args.append(b_arg)

            def remap_args(s: Scan) -> Scan:
                def remap_literal(x: Literal) -> Literal:
                    return Literal(value=str(index_map[int(x.value)]), type=x.type)

                return Scan(
                    function=s.function,
                    output=remap_literal(s.output),
                    inputs=[remap_literal(i) for i in s.inputs],
                    init=s.init,
                )

            return ScanExecution(
                backend=a.backend,
                scans=a.scans + [remap_args(s) for s in b.scans],
                args=a.args + compacted_b_args,
            )

        res = executions[:1]
        for execution in executions[1:]:
            if (
                isinstance(execution, ScanExecution)
                and isinstance(res[-1], ScanExecution)
                and execution.backend == res[-1].backend
            ):
                res[-1] = merge(res[-1], execution)
            else:
                res.append(execution)
        return res

    @staticmethod
    def _collect_offsets(node: itir.FencilDefinition) -> list[str]:
        return list(
            node.pre_walk_values()
            .if_isinstance(itir.OffsetLiteral)
            .getattr("value")
            .if_isinstance(str)
            .to_set()
        )

    def visit_FencilDefinition(
        self, node: itir.FencilDefinition, *, grid_type: str, **kwargs: Any
    ) -> FencilDefinition:
        grid_type = getattr(GridType, grid_type.upper())
        extracted_functions: list[Union[FunctionDefinition, ScanPassDefinition]] = []
        self.offset_provider = kwargs["offset_provider"]
        executions = self.visit(
            node.closures, grid_type=grid_type, extracted_functions=extracted_functions
        )
        executions = self._merge_scans(executions)
        function_definitions = self.visit(node.function_definitions) + extracted_functions
        axes = self._collect_offset_or_axis_node(itir.AxisLiteral, node)
        offsets = self._collect_offset_or_axis_node(
            OffsetLiteral, executions + function_definitions
        )  # collect offsets from gtfn nodes as some might have been dropped
        offset_declarations = list(map(lambda x: Sym(id=x), axes | offsets))
        return FencilDefinition(
            id=SymbolName(node.id),
            params=self.visit(node.params),
            executions=executions,
            offset_declarations=offset_declarations,
            function_definitions=function_definitions,
            grid_type=grid_type,
            temporaries=[],
        )

    def visit_Temporary(self, node, *, params: list, **kwargs) -> TemporaryAllocation:
        def dtype_to_cpp(x):
            if isinstance(x, int):
                return f"std::remove_const_t<sid::element_type<decltype({params[x]})>>"
            if isinstance(x, tuple):
                return "tuple<" + ", ".join(dtype_to_cpp(i) for i in x) + ">"
            assert isinstance(x, str)
            return pytype_to_cpptype(x)

        return TemporaryAllocation(
            id=node.id, dtype=dtype_to_cpp(node.dtype), domain=self.visit(node.domain, **kwargs)
        )

    def visit_FencilWithTemporaries(self, node, **kwargs) -> FencilDefinition:
        fencil = self.visit(node.fencil, **kwargs)
        return FencilDefinition(
            id=fencil.id,
            params=self.visit(node.params),
            executions=fencil.executions,
            grid_type=fencil.grid_type,
            offset_declarations=fencil.offset_declarations,
            function_definitions=fencil.function_definitions,
            temporaries=self.visit(node.tmps, params=[p.id for p in node.params]),
        )
