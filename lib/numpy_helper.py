#!/usr/bin/env python
'''
Extension to numpy
'''
import numpy
from functools import lru_cache

asarray = numpy.asarray

##TODO: create attribute for user defined flop threshold
FLOP_THRESHOLD = 5e6

# Detect backends
def _has_module(name):
    try:
        __import__(name)
        return True
    except (ImportError, OSError):
        return False

def einsum_backend(opt_einsum, pytblis, log):

    has_pytblis = _has_module("pytblis")
    has_opt_einsum = _has_module("opt_einsum")

    # Backend priority: provided flags > pytblis > opt_einsum > numpy fallback
    if opt_einsum and has_opt_einsum:
        backend = "opt_einsum"
    elif has_pytblis:
        backend = "pytblis"
    elif has_opt_einsum:
        backend = "opt_einsum"
    else:
        backend = "numpy"

    if opt_einsum and backend != "opt_einsum":
        log.warn(
            "opt_einsum was requested (OPT_EINSUM=True) but is not available. "
            "Falling back to %s.", backend
        )
    if pytblis and backend != "pytblis":
        log.warn(
            "pytblis was requested (PYTBLIS=True) but is not available. "
            "Falling back to %s.", backend
        )

    return backend

# Import backend
try:
    import pytblis as _pytblis
except ImportError:
    _pytblis = None

try:
    import opt_einsum as _opt_einsum
    _einsum_path = getattr(_opt_einsum, "contract_path", numpy.einsum_path)
except ImportError:
    _opt_einsum = None
    _einsum_path = getattr(numpy, "einsum_path", None)

@lru_cache(maxsize=None)
def _compiled_opt_expr(subscripts, shapes, optimize):
    return _opt_einsum.contract_expression(subscripts, *shapes, optimize=optimize)

def contract(subscripts, A, B, backend, **kwargs):
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

    # Call numpy.asarray because A or B may be HDF5 Datasets
    A, B = asarray(A), asarray(B)

    if backend == "opt_einsum":
        optimize = kwargs.pop('optimize', True)
        shapes = (tuple(A.shape), tuple(B.shape))
        expr = _compiled_opt_expr(idx_str, shapes, optimize)
        return expr(A, B, **kwargs)

    if backend == "pytblis":
       return _pytblis.contract(idx_str, A, B, **kwargs)

    # Linear algebra kwargs
    alpha = kwargs.pop('alpha', 1)
    beta = kwargs.pop('beta', 0)
    out = kwargs.pop('out', None)

    # check: binary einsum with an explicit output?
    if "->" not in idx_str or idx_str.count(",") != 1:
        return numpy.einsum(idx_str, A, B, **kwargs)

    # check: valid GEMM contraction?
    analysis = _analyze_indices(idx_str)
    if analysis is None:
        return numpy.einsum(idx_str, A, B, **kwargs)

    idxA, idxB, idxC, orderA, orderB, permC, shared = analysis

    if A.size == 0 or B.size == 0:
        shapeA = dict(zip(idxA, A.shape))
        shapeB = dict(zip(idxB, B.shape))
        out_shape = (
            [shapeA[i] for i in idxA if i not in shared] +
            [shapeB[i] for i in idxB if i not in shared])
        out_shape = [out_shape[i] for i in permC]
        return numpy.zeros(out_shape, dtype=numpy.result_type(A, B))

    # check: dimensions valid?
    if len(idxA) != A.ndim or len(idxB) != B.ndim:
        return numpy.einsum(idx_str, A, B, **kwargs)

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

    # check: FLOP threshold met?
    if flops < FLOP_THRESHOLD:
        return numpy.einsum(idx_str, A, B, **kwargs)

    At = A.transpose(orderA).reshape(-1, inner_dim)
    Bt = B.transpose(orderB).reshape(inner_dim, -1)

    if not At.flags.c_contiguous:
        At = asarray(At, order="C")
    if not Bt.flags.c_contiguous:
        Bt = asarray(Bt, order="C")

    Cmat = numpy.dot(At, Bt)

    out_shape = (
        [shapeA[i] for i in idxA if i not in shared] +
        [shapeB[i] for i in idxB if i not in shared])

    Cres = Cmat.reshape(out_shape)

    # check if perm is identity
    if permC != list(range(len(permC))):
        Cres = Cres.transpose(permC)

    # alpha/beta/out semantics
    if out is None:
        return Cres * alpha if alpha != 1 else Cres.copy()

    # out provided
    out_arr = asarray(out)
    numpy.multiply(Cres, alpha, out=Cres)
    if beta != 0:
        numpy.add(Cres, out_arr * beta, out=Cres)
    # assign to output
    out_arr[...] = Cres
    return out_arr

def einsum(scripts, *tensors, backend, **kwargs):
    '''
    Perform a more efficient einsum via reshaping to a matrix multiply.

    Current differences compared to numpy.einsum:
    This assumes that each repeated index is actually summed (i.e. no 'i,i->i')
    and appears only twice (i.e. no 'ij,ik,il->jkl'). The output indices must
    be explicitly specified (i.e. 'ij,j->i' and not 'ij,j').
    '''
    subscripts = scripts.replace(" ","")
    tensors = tuple(asarray(t) for t in tensors)

    if len(tensors) <= 1 or '...' in subscripts:
        return numpy.einsum(subscripts, *tensors, **kwargs)

    if backend == "pytblis":
        if kwargs.get("optimize") is True:
           kwargs["optimize"] = "optimal"
        return _pytblis.einsum(subscripts, *tensors, **kwargs)

    optimize = kwargs.pop('optimize', True)
    if backend == "opt_einsum":
        shapes = tuple(tuple(t.shape) for t in tensors)
        expr = _compiled_opt_expr(subscripts, shapes, optimize)
        return expr(*tensors)

    _contract = kwargs.pop('_contract', contract)
    if len(tensors) <= 2:
        return _contract(subscripts, *tensors, backend=backend, **kwargs)

    _, contraction_list = _einsum_path(subscripts, *tensors, optimize=optimize, einsum_call=True)

    ##TODO: add in handling of in-place contraction (see numpy/numpy/blob/v2.4.0/numpy/_core/einsumfunc.py)
    operands = list(tensors)
    for num, contraction in enumerate(contraction_list):
        # NumPy 2.4.0 >= returns 3 values instead of 5 for einsum path
        if len(contraction) == 3:
            inds, einsum_str, _ = contraction
        else:
            inds, _, einsum_str, _, _ = contraction
        ops = [operands[i] for i in inds]
        if len(ops)==2:
            result = _contract(einsum_str, *ops, backend=backend, **kwargs)
        else:
            result = numpy.einsum(einsum_str, *ops, **kwargs)
        operands = [op for i, op in enumerate(operands) if i not in inds]
        operands.append(result)
        del ops, result

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

    # reject when external index is in both GEMM axes or 
    # when multiple external indices cannot be flattened
    if ((external_ind and external_A and external_B) or
        ((len(external_ind) > 1) and (external_A or external_B))):
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
