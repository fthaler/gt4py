import numpy as np
import pytest

from functional.common import Dimension
from functional.fencil_processors.formatters.gtfn import format_sourcecode as gtfn_format_sourcecode
from functional.fencil_processors.runners.gtfn_cpu import run_gtfn
from functional.iterator.builtins import *
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import closure, fendef, fundef, offset

from .conftest import run_processor


I = offset("I")
K = offset("K")


@fundef
def multiply_stencil(inp):
    return deref(shift(K, 1, I, 1)(inp))


KDim = Dimension("KDim")
IDim = Dimension("IDim")


@fendef(column_axis=KDim)
def fencil(i_size, k_size, inp, out):
    closure(
        cartesian_domain(named_range(IDim, 0, i_size), named_range(KDim, 0, k_size)),
        multiply_stencil,
        out,
        [inp],
    )


def test_column_stencil(fencil_processor, lift_mode):
    fencil_processor, validate = fencil_processor
    shape = [5, 7]
    inp = np_as_located_field(IDim, KDim)(
        np.fromfunction(lambda i, k: i * 10 + k, [shape[0] + 1, shape[1] + 1])
    )
    out = np_as_located_field(IDim, KDim)(np.zeros(shape))

    ref = np.asarray(inp)[1:, 1:]

    run_processor(
        fencil,
        fencil_processor,
        shape[0],
        shape[1],
        inp,
        out,
        offset_provider={"I": IDim, "K": KDim},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(ref, out)


def test_column_stencil_with_k_origin(fencil_processor, lift_mode):
    fencil_processor, validate = fencil_processor
    if fencil_processor == run_gtfn:
        pytest.xfail("origin not yet supported in gtfn")

    shape = [5, 7]
    raw_inp = np.fromfunction(lambda i, k: i * 10 + k, [shape[0] + 1, shape[1] + 2])
    inp = np_as_located_field(IDim, KDim, origin={IDim: 0, KDim: 1})(raw_inp)
    out = np_as_located_field(IDim, KDim)(np.zeros(shape))

    ref = np.asarray(inp)[1:, 2:]

    run_processor(
        fencil,
        fencil_processor,
        shape[0],
        shape[1],
        inp,
        out,
        offset_provider={"I": IDim, "K": KDim},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(ref, out)


@fundef
def sum_scanpass(state, inp):
    return state + deref(inp)


@fundef
def ksum(inp):
    return scan(sum_scanpass, True, 0.0)(inp)


@fendef(column_axis=KDim)
def ksum_fencil(i_size, k_start, k_end, inp, out):
    closure(
        cartesian_domain(named_range(IDim, 0, i_size), named_range(KDim, k_start, k_end)),
        ksum,
        out,
        [inp],
    )


@pytest.mark.parametrize(
    "kstart, reference",
    [
        (0, np.asarray([[0, 1, 3, 6, 10, 15, 21]])),
        (2, np.asarray([[0, 0, 2, 5, 9, 14, 20]])),
    ],
)
def test_ksum_scan(fencil_processor, lift_mode, kstart, reference):
    fencil_processor, validate = fencil_processor
    shape = [1, 7]
    inp = np_as_located_field(IDim, KDim)(np.asarray([list(range(7))], dtype="float64"))
    out = np_as_located_field(IDim, KDim)(np.zeros(shape))

    run_processor(
        ksum_fencil,
        fencil_processor,
        shape[0],
        kstart,
        shape[1],
        inp,
        out,
        offset_provider={"I": IDim, "K": KDim},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(reference, np.asarray(out))


@fundef
def ksum_back(inp):
    return scan(sum_scanpass, False, 0.0)(inp)


@fendef(column_axis=KDim)
def ksum_back_fencil(i_size, k_size, inp, out):
    closure(
        cartesian_domain(named_range(IDim, 0, i_size), named_range(KDim, 0, k_size)),
        ksum_back,
        out,
        [inp],
    )


def test_ksum_back_scan(fencil_processor, lift_mode):
    fencil_processor, validate = fencil_processor
    shape = [1, 7]
    inp = np_as_located_field(IDim, KDim)(np.asarray([list(range(7))], dtype="float64"))
    out = np_as_located_field(IDim, KDim)(np.zeros(shape))

    ref = np.asarray([[21, 21, 20, 18, 15, 11, 6]])

    run_processor(
        ksum_back_fencil,
        fencil_processor,
        shape[0],
        shape[1],
        inp,
        out,
        offset_provider={"I": IDim, "K": KDim},
        lift_mode=lift_mode,
    )

    if validate:
        assert np.allclose(ref, np.asarray(out))


@fundef
def doublesum_scanpass(state, inp0, inp1):
    return make_tuple(tuple_get(0, state) + deref(inp0), tuple_get(1, state) + deref(inp1))


@fundef
def kdoublesum(inp0, inp1):
    return scan(doublesum_scanpass, True, make_tuple(0.0, 0))(inp0, inp1)


@fendef(column_axis=KDim)
def kdoublesum_fencil(i_size, k_start, k_end, inp0, inp1, out):
    closure(
        cartesian_domain(named_range(IDim, 0, i_size), named_range(KDim, k_start, k_end)),
        kdoublesum,
        out,
        [inp0, inp1],
    )


@pytest.mark.parametrize(
    "kstart, reference",
    [
        (
            0,
            np.asarray(
                [[(0, 0), (1, 1), (3, 3), (6, 6), (10, 10), (15, 15), (21, 21)]],
                dtype=np.dtype([("foo", np.float64), ("bar", np.int64)]),
            ),
        ),
        (
            2,
            np.asarray(
                [[(0, 0), (0, 0), (2, 2), (5, 5), (9, 9), (14, 14), (20, 20)]],
                dtype=np.dtype([("foo", np.float64), ("bar", np.int64)]),
            ),
        ),
    ],
)
def test_kdoublesum_scan(fencil_processor, lift_mode, kstart, reference):
    fencil_processor, validate = fencil_processor
    if fencil_processor == run_gtfn or fencil_processor == gtfn_format_sourcecode:
        pytest.xfail("structured dtype input/output currently unsupported")
    shape = [1, 7]
    inp0 = np_as_located_field(IDim, KDim)(np.asarray([list(range(7))], dtype=np.float64))
    inp1 = np_as_located_field(IDim, KDim)(np.asarray([list(range(7))], dtype=np.int64))
    out = np_as_located_field(IDim, KDim)(np.zeros(shape, dtype=reference.dtype))

    run_processor(
        kdoublesum_fencil,
        fencil_processor,
        shape[0],
        kstart,
        shape[1],
        inp0,
        inp1,
        out,
        offset_provider={"I": IDim, "K": KDim},
        lift_mode=lift_mode,
    )

    if validate:
        for n in reference.dtype.names:
            assert np.allclose(reference[n], np.asarray(out)[n])


# TODO(havogt) test tuple_get builtin on a Column
