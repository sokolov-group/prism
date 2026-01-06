#!/usr/bin/env python
'''
Extension to numpy
'''
import numpy
from prism.interface import PYSCF
from functools import lru_cache

dot = numpy.dot
asarray = numpy.asarray

##TODO: create attribute for user defined flop threshold
FLOP_THRESHOLD = 1e4
#FLOP_THRESHOLD = 5e6

# Interface flags
PYTBLIS = getattr(PYSCF, 'pytblis', False)
OPT_EINSUM = getattr(PYSCF, 'opt_einsum', False)

# Detect backends
def _has_module(name):
    try:
        __import__(name)
        return True
    except (ImportError, OSError):
        return False

_HAS_PYTBLIS = _has_module('pytblis')
_HAS_OPT_EINSUM = _has_module('opt_einsum')

# Backend priority: provided flags > pytblis > opt_einsum > numpy fallback
if OPT_EINSUM and _HAS_OPT_EINSUM:
    EINSUM_BACKEND = "opt_einsum"
elif PYTBLIS or _HAS_PYTBLIS:
    EINSUM_BACKEND = "pytblis"
elif _HAS_OPT_EINSUM:
    EINSUM_BACKEND = "opt_einsum"
else:
    EINSUM_BACKEND = "numpy"

setattr(PYSCF, 'einsum_backend', EINSUM_BACKEND)

# Import backend
try:
    if EINSUM_BACKEND == "pytblis":
        import pytblis
    elif EINSUM_BACKEND == "opt_einsum":
        import opt_einsum
except Exception:
    EINSUM_BACKEND = "numpy"

_numpy_einsum = numpy.einsum

try:
    import opt_einsum as _opt_einsum
    _einsum_path = getattr(_opt_einsum, "contract_path", numpy.einsum_path)
except ImportError:
    _einsum_path = getattr(numpy, "einsum_path", None)

def contract(subscripts, A, B, **kwargs):
    '''
    Perform tensor contraction using einsum notation
    C = alpha * einsum(subscripts, A, B) + beta * C

    Kwargs:
        alpha : scalar, optional
            Default value is 1.
        beta : scalar, optional
            Default value is 0.
        out : ndarray, optional
            Output tensor to store the result.
    '''
    idx_str = subscripts.replace(" ","")

    if EINSUM_BACKEND == "opt_einsum":
        return opt_einsum.contract(idx_str, A, B, **kwargs)

    if EINSUM_BACKEND == "pytblis":
        return pytblis.einsum(idx_str, A, B, **kwargs)

    # Call numpy.asarray because A or B may be HDF5 Datasets
    A = asarray(A)
    B = asarray(B)

    # Linear algebra kwargs for GEMM branch
    alpha = kwargs.pop('alpha', 1)
    beta = kwargs.pop('beta', 0)
    out = kwargs.pop('out', None)

    # binary einsum with an explicit output?
    if "->" not in idx_str or idx_str.count(",") != 1:
        return _numpy_einsum(idx_str, A, B, **kwargs)

    # valid GEMM contraction?
    analysis = _analyze_indices(idx_str)
    if analysis is None:
        return _numpy_einsum(idx_str, A, B, **kwargs)

    idxA, idxB, idxC, orderA, orderB, permC, shared = analysis

    if len(idxA) != A.ndim or len(idxB) != B.ndim:
        return _numpy_einsum(idx_str, A, B, **kwargs)

    # GEMM based on FLOP heuristic
    shapeA = dict(zip(idxA, A.shape))
    shapeB = dict(zip(idxB, B.shape))

    inner_dim = 1
    for s in shared:
        if shapeA[s] != shapeB[s]:
            raise ValueError(
                f"Index '{s}' has incompatible dimensions: "
                f"{shapeA[s]} vs {shapeB[s]}"
            )
        inner_dim *= shapeA[s]

    outer_dim_A = A.size // inner_dim
    outer_dim_B = B.size // inner_dim
    flops = outer_dim_A * inner_dim * outer_dim_B

    if flops < FLOP_THRESHOLD:
        return _numpy_einsum(idx_str, A, B, **kwargs)

    if A.size == 0 or B.size == 0:
        out_shape = (
            [shapeA[i] for i in idxA if i not in shared]
            + [shapeB[i] for i in idxB if i not in shared]
        )
        out_shape = [out_shape[i] for i in permC]
        return numpy.zeros(out_shape, dtype=numpy.result_type(A, B))

    At = A.transpose(orderA)
    Bt = B.transpose(orderB)

    At = asarray(At.reshape(-1, inner_dim), order="C")
    Bt = asarray(Bt.reshape(inner_dim, -1), order="C")

    Cmat = dot(At, Bt)

    out_shape = (
        [shapeA[i] for i in idxA if i not in shared]
        + [shapeB[i] for i in idxB if i not in shared]
    )

    Cres = Cmat.reshape(out_shape)
    #return C.reshape(out_shape, order="A").transpose(permC)

    # only apply permutation if perm is not identity
    if permC != list(range(len(permC))):
        Cres = Cres.transpose(permC)

    # alpha/beta/out semantics
    if out is None:
        if alpha != 1:
            Cres = Cres * alpha
        return Cres
    else:
        out_arr = asarray(out)
        # compute into temporary to avoid aliasing issues
        result = Cres * alpha
        if beta != 0:
            result = result + out_arr * beta
        # cast/assign back to out with broadcasting checks
        out[...] = numpy.asarray(result, dtype=numpy.result_type(A, B, out))
        return out

def einsum(scripts, *tensors, **kwargs):
    '''Perform a more efficient einsum via reshaping to a matrix multiply.

    Current differences compared to numpy.einsum:
    This assumes that each repeated index is actually summed (i.e. no 'i,i->i')
    and appears only twice (i.e. no 'ij,ik,il->jkl'). The output indices must
    be explicitly specified (i.e. 'ij,j->i' and not 'ij,j').
    '''
    subscripts = scripts.replace(' ','')

    if EINSUM_BACKEND == "pytblis":
        if kwargs.get("optimize") is True:
           kwargs["optimize"] = "optimal"
        return pytblis.einsum(subscripts, *tensors, **kwargs)

    if len(tensors) <= 1 or '...' in subscripts:
        return _numpy_einsum(subscripts, *tensors, **kwargs)

    optimize = kwargs.pop('optimize', True)
    if EINSUM_BACKEND == "opt_einsum":
        return opt_einsum.contract(subscripts, *tensors, optimize=optimize)

    _contract = kwargs.pop('_contract', contract)
    if len(tensors) <= 2:
        return _contract(subscripts, *tensors, **kwargs)

    _, contraction_list = _einsum_path(subscripts, *tensors, optimize=optimize, einsum_call=True)

    operands = list(tensors)
    for inds, _, einsum_str, _, _ in contraction_list:
        ops = [operands[i] for i in inds]
        fn = _contract if len(ops) == 2 else _numpy_einsum
        result = fn(einsum_str, *ops, **kwargs)
        for i in sorted(inds, reverse=True):
            del operands[i]
        operands.append(result)

    return operands[0]

@lru_cache(maxsize=256)
def _analyze_indices(idx_str):
    '''
    Analyze an einsum index string of the form 'ab,bc->ac'.

    Returns:
        idxA, idxB, idxC
        orderA, orderB
        permC
        shared
    '''

    # Split the strings into a list of idx char's
    try:
        idxA, idxBC = idx_str.split(",")
        idxB, idxC = idxBC.split("->")
    except ValueError:
        return None

    uniqA = set(idxA)
    uniqB = set(idxB)
    # reject when indices repeat
    if len(idxA) != len(uniqA) or len(idxB) != len(uniqB):
        return None

    shared = sorted(uniqA & uniqB)
    # reject when no indices overlap
    if not shared:
        return None

    internal_ind = [i for i in shared if i not in idxC]
    external_ind = [i for i in shared if i in idxC]

    # reject when no contraction exists
    if not internal_ind:
        return None

    external_A = [i for i in idxA if i not in internal_ind]
    external_B = [i for i in idxB if i not in internal_ind]

    # reject when external index is in both GEMM axes or when multiple external indices cannot be flattened
    if (external_ind and external_A and external_B) or ((len(external_ind) > 1) and (external_A or external_B)):
        return None

    # reorder A: [..., internal]
    idxA_t = external_ind + external_A + internal_ind
    orderA = [idxA.index(i) for i in idxA_t]

    # reorder B: [internal, ...]
    idxB_t = external_ind + internal_ind + external_B
    orderB = [idxB.index(i) for i in idxB_t]

    # output permutation
    out_idx = external_ind + external_A + external_B
    if set(out_idx) != set(idxC):
        return None

    permC = [out_idx.index(i) for i in idxC]

    return idxA, idxB, idxC, orderA, orderB, permC, shared