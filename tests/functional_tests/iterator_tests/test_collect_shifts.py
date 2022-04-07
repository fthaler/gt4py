from functional.iterator import ir
from functional.iterator.transforms.collect_shifts import CollectShifts


def test_trivial():
    testee = ir.FunCall(
        fun=ir.SymRef(id="deref"),
        args=[
            ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="shift"),
                    args=[ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1)],
                ),
                args=[ir.SymRef(id="x")],
            )
        ],
    )
    expected = {"x": [(ir.OffsetLiteral(value="I"), ir.OffsetLiteral(value=1))]}

    actual = dict()
    CollectShifts().visit(testee, shifts=actual)
    assert actual == expected


def test_reduce():
    testee = ir.FunCall(
        fun=ir.FunCall(
            fun=ir.SymRef(id="reduce"), args=[ir.SymRef(id="plus"), ir.FloatLiteral(value=0.0)]
        ),
        args=[
            ir.FunCall(
                fun=ir.FunCall(fun=ir.SymRef(id="shift"), args=[ir.OffsetLiteral(value="V2E")]),
                args=[ir.SymRef(id="x")],
            )
        ],
    )

    expected = {"x": [(ir.OffsetLiteral(value="V2E"), CollectShifts.ALL_NEIGHBORS)]}

    actual = dict()
    CollectShifts().visit(testee, shifts=actual)
    assert actual == expected
