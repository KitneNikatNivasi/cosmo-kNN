import pytest

import numpy as np

import MAS_library as MASL

import os
import sys

import warnings

warnings.filterwarnings('ignore')

module_path = os.path.abspath(os.path.join('../../'))           # '../' is needed because the parent directory is one directories upstream of the tutorials directory
if module_path not in sys.path:
    sys.path.append(module_path)

from kNNpy import kNN_3D                                # the main module

####################################################################################################

def build_query_grid(boxsize, n_query):
    """Lay out ~n_query query points on a regular 3D grid strictly inside the box."""
    n_side = max(int(round(n_query ** (1.0/3.0))), 1)
    step = boxsize / n_side
    coords = np.arange(n_side) * step + 0.5 * step   # cell-centred, always in (0, boxsize)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)


def build_field(NGRID, BoxSize, rng):
    """Synthetic overdensity field: CIC-paint random particles then normalise.
    Swap this out for a real Quijote/HalfDome density field via
    kNNpy.Data.Datasets if a matching 3D box loader is available -- kept
    synthetic here so the benchmark has no dependency on bundled simulation
    data."""
    n_field_particles = 20 * NGRID**2
    field_pos = rng.uniform(0, BoxSize, (n_field_particles, 3)).astype(np.float32)
    field = np.zeros((NGRID, NGRID, NGRID), dtype=np.float32)
    MASL.MA(field_pos, field, BoxSize, 'CIC')
    field /= np.mean(field, dtype=np.float64)
    field -= 1.0
    return field

####################################################################################################

#-----------------------------------------  TracerAuto3D  ------------------------------------------

def compute_TracerAuto3D(boxsize, kList, BinsRad, QueryPos, TracerPos, n_threads, ReturnNNdist, Verbose):

    results = kNN_3D.TracerAuto3D(boxsize, kList, BinsRad, QueryPos, TracerPos, n_threads, ReturnNNdist, Verbose)
    return results

def test_compute_TracerAuto3D(benchmark, BoxSize, n_tracer, n_query, k_List, threads, rounds, warmup):

    print(BoxSize, n_tracer, n_query, k_List, threads, rounds, warmup)

    def setup():

        rng = np.random.default_rng()
        TracerPos = rng.uniform(0, BoxSize, (n_tracer, 3))
        QueryPos = build_query_grid(BoxSize, n_query)
        BinsRad = [np.geomspace(2.0, 40.0, 8) for _ in k_List]

        return (BoxSize, k_List, BinsRad, QueryPos, TracerPos, threads, False, False), {}

    result = benchmark.pedantic(
        compute_TracerAuto3D,
        setup=setup,
        rounds=rounds,
        iterations=1,
        warmup_rounds=warmup,
    )

    p_gtr_k_list, vol = result
    assert len(p_gtr_k_list) == len(k_List)

####################################################################################################

#-------------------------------------  TracerTracerCross3D  ---------------------------------------

def compute_TracerTracerCross3D(boxsize, kA_kB_list, BinsRad, QueryPos, TracerPos_A, TracerPos_B, n_threads, Verbose):

    results = kNN_3D.TracerTracerCross3D(boxsize, kA_kB_list, BinsRad, QueryPos, TracerPos_A, TracerPos_B, n_threads, Verbose)
    return results

def test_compute_TracerTracerCross3D(benchmark, BoxSize, n_tracer, n_query, kA_kB_List, threads, rounds, warmup):

    print(BoxSize, n_tracer, n_query, kA_kB_List, threads, rounds, warmup)

    def setup():

        rng = np.random.default_rng()
        TracerPos_A = rng.uniform(0, BoxSize, (n_tracer, 3))
        TracerPos_B = rng.uniform(0, BoxSize, (n_tracer, 3))
        QueryPos = build_query_grid(BoxSize, n_query)
        BinsRad = [np.geomspace(2.0, 40.0, 8) for _ in kA_kB_List]

        return (BoxSize, kA_kB_List, BinsRad, QueryPos, TracerPos_A, TracerPos_B, threads, False), {}

    result = benchmark.pedantic(
        compute_TracerTracerCross3D,
        setup=setup,
        rounds=rounds,
        iterations=1,
        warmup_rounds=warmup,
    )

    p_gtr_kA_list, p_gtr_kB_list, p_gtr_kA_kB_list = result
    assert len(p_gtr_kA_kB_list) == len(kA_kB_List)

####################################################################################################

#----------------------------------  TracerTracerCross3D_DataVector  -------------------------------

def compute_TracerTracerCross3D_DataVector(boxsize, kA_kB_list, BinsRad, QueryPos, TracerPos_A_dict, TracerPos_B, n_threads, Verbose):

    results = kNN_3D.TracerTracerCross3D_DataVector(boxsize, kA_kB_list, BinsRad, QueryPos, TracerPos_A_dict, TracerPos_B, n_threads, Verbose)
    return results

def test_compute_TracerTracerCross3D_DataVector(benchmark, BoxSize, n_tracer, n_query, kA_kB_List, n_reals, threads, rounds, warmup):

    print(BoxSize, n_tracer, n_query, kA_kB_List, n_reals, threads, rounds, warmup)

    def setup():

        rng = np.random.default_rng()
        TracerPos_A_dict = {i: rng.uniform(0, BoxSize, (n_tracer, 3)) for i in range(n_reals)}
        TracerPos_B = rng.uniform(0, BoxSize, (n_tracer, 3))
        QueryPos = build_query_grid(BoxSize, n_query)
        BinsRad = [np.geomspace(2.0, 40.0, 8) for _ in kA_kB_List]

        return (BoxSize, kA_kB_List, BinsRad, QueryPos, TracerPos_A_dict, TracerPos_B, threads, False), {}

    result = benchmark.pedantic(
        compute_TracerTracerCross3D_DataVector,
        setup=setup,
        rounds=rounds,
        iterations=1,
        warmup_rounds=warmup,
    )

    assert result.shape[0] == n_reals

####################################################################################################

#---------------------------------------  TracerFieldCross3D  --------------------------------------

def compute_TracerFieldCross3D(kList, RBins, BoxSize, QueryPos, TracerPos, Field3D, FieldConstPercThreshold, n_threads, ReturnSmoothedFieldDict, Verbose):

    results = kNN_3D.TracerFieldCross3D(kList, RBins, BoxSize, QueryPos, TracerPos, Field3D, FieldConstPercThreshold, n_threads, ReturnSmoothedFieldDict, Verbose)
    return results

def test_compute_TracerFieldCross3D(benchmark, NGRID, BoxSize, n_tracer, n_query, k_List, FieldConstPercThreshold, threads, rounds, warmup):

    print(NGRID, BoxSize, n_tracer, n_query, k_List, FieldConstPercThreshold, threads, rounds, warmup)

    def setup():

        rng = np.random.default_rng()
        Field3D = build_field(NGRID, BoxSize, rng)
        TracerPos = rng.uniform(0, BoxSize, (n_tracer, 3))
        QueryPos = build_query_grid(BoxSize, n_query)
        RBins = [np.geomspace(2.0, 40.0, 8) for _ in k_List]

        return (k_List, RBins, BoxSize, QueryPos, TracerPos, Field3D, FieldConstPercThreshold, threads, False, False), {}

    result = benchmark.pedantic(
        compute_TracerFieldCross3D,
        setup=setup,
        rounds=rounds,
        iterations=1,
        warmup_rounds=warmup,
    )

    p_gtr_k_list, p_gtr_dt_list, p_gtr_k_dt_list = result
    assert len(p_gtr_k_list) == len(k_List)

####################################################################################################

#------------------------------------  TracerFieldCross3D_DataVector  ------------------------------

def compute_TracerFieldCross3D_DataVector(kList, RBins, BoxSize, QueryPos, TracerPosVector, Field, FieldConstPercThreshold, n_threads, ReturnSmoothedDict, Verbose):

    results = kNN_3D.TracerFieldCross3D_DataVector(kList, RBins, BoxSize, QueryPos, TracerPosVector, Field, FieldConstPercThreshold, n_threads, ReturnSmoothedDict, Verbose)
    return results

def test_compute_TracerFieldCross3D_DataVector(benchmark, NGRID, BoxSize, n_tracer, n_query, k_List, FieldConstPercThreshold, n_reals, threads, rounds, warmup):

    print(NGRID, BoxSize, n_tracer, n_query, k_List, FieldConstPercThreshold, n_reals, threads, rounds, warmup)

    def setup():

        rng = np.random.default_rng()
        Field = build_field(NGRID, BoxSize, rng)

        # Note: this function's own input validation requires 0 < position <= BoxSize
        # (the closed-upper/open-lower convention here differs from the other four
        # kNN_3D functions), so the lower bound is nudged strictly above zero.
        TracerPosVector = rng.uniform(1e-6, BoxSize, (n_reals, n_tracer, 3))
        QueryPos = build_query_grid(BoxSize, n_query)
        RBins = [np.geomspace(2.0, 40.0, 8) for _ in k_List]

        return (k_List, RBins, BoxSize, QueryPos, TracerPosVector, Field, FieldConstPercThreshold, threads, False, False), {}

    result = benchmark.pedantic(
        compute_TracerFieldCross3D_DataVector,
        setup=setup,
        rounds=rounds,
        iterations=1,
        warmup_rounds=warmup,
    )

    p_gtr_k_veclist, p_gtr_dt_list, p_gtr_k_dt_veclist = result
    assert len(p_gtr_k_veclist) == len(k_List)
