import numpy as np
from functools import reduce

def compute_S12_p1(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas
    rdm_ca = mr_adc.rdm.ca

    s_thresh = mr_adc.s_thresh_doubles

    # Compute S matrix: < Psi_0 | a_X a^{\dag}_Y | Psi_0 >
    S_p1  = einsum('XY->XY', np.identity(ncas), optimize = einsum_type).copy()
    S_p1 -= 1/2 * einsum('YX->XY', rdm_ca, optimize = einsum_type).copy()

    # Compute S^{-1/2} matrix
    S_eval, S_evec = np.linalg.eigh(S_p1)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])
    S_evec = S_evec[:, S_ind_nonzero]

    S_p1_12_inv = np.dot(S_evec, np.diag(S_inv_eval))

    if not ignore_print:
        print("Dimension of the [+1] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print("Smallest eigenvalue of the [+1] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_p1_12_inv

def compute_S12_m1(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca

    s_thresh = mr_adc.s_thresh_doubles

    # Compute S matrix: < Psi_0 | a^{\dag}_X a_Y | Psi_0 >
    S_m1  = 1/2 * einsum('XY->XY', rdm_ca, optimize = einsum_type).copy()

    # Compute S^{-1/2} matrix
    S_eval, S_evec = np.linalg.eigh(S_m1)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])
    S_evec = S_evec[:, S_ind_nonzero]

    S_m1_12_inv = np.dot(S_evec, np.diag(S_inv_eval))

    if not ignore_print:
        print("Dimension of the [-1] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print("Smallest eigenvalue of the [-1] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_m1_12_inv

def compute_S12_p2(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    s_thresh = mr_adc.s_thresh_doubles

    # Compute S matrix: < Psi_0 | a_Z a_W a^{\dag}_X a^{\dag}_Y | Psi_0 >
    S_p2  = 1/3 * einsum('XYWZ->ZWYX', rdm_ccaa, optimize = einsum_type).copy()
    S_p2 += 1/6 * einsum('XYZW->ZWYX', rdm_ccaa, optimize = einsum_type).copy()
    S_p2 += einsum('WX,YZ->ZWYX', np.identity(ncas), np.identity(ncas), optimize = einsum_type)
    S_p2 -= 1/2 * einsum('WX,YZ->ZWYX', np.identity(ncas), rdm_ca, optimize = einsum_type)
    S_p2 -= 1/2 * einsum('YZ,XW->ZWYX', np.identity(ncas), rdm_ca, optimize = einsum_type)
    S_p2 = S_p2.reshape(ncas**2, ncas**2)

    # Compute S^{-1/2} matrix
    S_eval, S_evec = np.linalg.eigh(S_p2)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])
    S_evec = S_evec[:, S_ind_nonzero]

    S_p2_12_inv = np.dot(S_evec, np.diag(S_inv_eval))

    if not ignore_print:
        print("Dimension of the [+2] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print("Smallest eigenvalue of the [+2] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_p2_12_inv

def compute_S12_m2(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    rdm_ccaa = mr_adc.rdm.ccaa

    S_m2  = 1/6 * einsum('XYWZ->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S_m2 += 1/3 * einsum('XYZW->XYZW', rdm_ccaa, optimize = einsum_type).copy()

    S_m2 = S_m2.reshape(ncas**2, ncas**2)

    s_thresh = mr_adc.s_thresh_doubles
    S_eval, S_evec = np.linalg.eigh(S_m2)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])
    S_evec = S_evec[:, S_ind_nonzero]

    S_m2_12_inv = np.dot(S_evec, np.diag(S_inv_eval))

    if not ignore_print:
        print("Dimension of the [-2] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print("Smallest eigenvalue of the [-2] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_m2_12_inv

def compute_S12_0p_gno_projector(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Damping for singles
    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    # Compute S matrix
    ## S11 block: < Psi_0 | a^{\dag}_X a_Y | Psi_0 >
    S12_a_a = 1/2 * einsum('XY->XY', rdm_ca, optimize = einsum_type).copy()

    ## S22 block: < Psi_0 | a^{\dag}_X a_Y a^{\dag}_W a_Z | Psi_0 >
    S22_aa_aa =- 1/6 * einsum('WXYZ->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_aa += 1/6 * einsum('WXZY->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_aa += 1/2 * einsum('WY,XZ->XYZW', np.identity(ncas), rdm_ca, optimize = einsum_type)

    S22_aa_bb  = 1/6 * einsum('WXYZ->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_bb += 1/3 * einsum('WXZY->XYZW', rdm_ccaa, optimize = einsum_type).copy()

    ## Reshape tensors to matrix form
    dim_ZW = ncas * ncas
    dim_S_0p = 1 + 2 * dim_ZW

    S12_a_a = S12_a_a.reshape(-1)

    S22_aa_aa = S22_aa_aa.reshape(dim_ZW, dim_ZW)
    S22_aa_bb = S22_aa_bb.reshape(dim_ZW, dim_ZW)

    # Building S_0p matrix
    S_aa_i = 1
    S_aa_f = S_aa_i + dim_ZW
    S_bb_i = S_aa_f
    S_bb_f = S_bb_i + dim_ZW

    S_0p = np.zeros((dim_S_0p, dim_S_0p))

    S_0p[0,0] = 1.0

    S_0p[0, S_aa_i:S_aa_f] = S12_a_a.copy()
    S_0p[0, S_bb_i:S_bb_f] = S12_a_a.copy()
    S_0p[S_aa_i:S_aa_f, 0] = S12_a_a.T.copy()
    S_0p[S_bb_i:S_bb_f, 0] = S12_a_a.T.copy()

    S_0p[S_aa_i:S_aa_f, S_aa_i:S_aa_f] = S22_aa_aa.copy()
    S_0p[S_bb_i:S_bb_f, S_bb_i:S_bb_f] = S22_aa_aa.copy()

    S_0p[S_aa_i:S_aa_f, S_bb_i:S_bb_f] = S22_aa_bb.copy()
    S_0p[S_bb_i:S_bb_f, S_aa_i:S_aa_f] = S22_aa_bb.T.copy()

    # Compute projector to the GNO operator basis
    Y = np.identity(S_0p.shape[0])

    Y[0, S_aa_i:S_aa_f] =- 0.5 * rdm_ca.reshape(-1).copy()
    Y[0, S_bb_i:S_bb_f] =- 0.5 * rdm_ca.reshape(-1).copy()

    # Compute S^{-1/2} matrix
    St_0p = reduce(np.dot, (Y.T, S_0p, Y))

    S_eval, S_evec = np.linalg.eigh(St_0p)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    ## Apply damping
    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], mr_adc.s_thresh_singles, mr_adc.s_damping_strength)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_0p_12_inv = reduce(np.dot, (Y, S_evec, np.diag(S_inv_eval)))

    if not ignore_print:
        print("Dimension of the [0'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print("Smallest eigenvalue of the [0'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_0p_12_inv

def compute_S12_p1p_gno_projector(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Damping for singles
    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    # Compute S matrix
    ## S11 block: < Psi_0 | a_X a^{\dag}_Z | Psi_0 >
    S11_a_a  = einsum('XZ->XZ', np.identity(ncas), optimize = einsum_type).copy()
    S11_a_a -= 1/2 * einsum('ZX->XZ', rdm_ca, optimize = einsum_type).copy()

    ## S12 block: < Psi_0 | a_X a^{\dag}_Z a^{\dag}_W a_Y | Psi_0 >
    S12_a_bba =- 1/6 * einsum('WZXY->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_bba -= 1/3 * einsum('WZYX->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_bba += 1/2 * einsum('XZ,WY->XYWZ', np.identity(ncas), rdm_ca, optimize = einsum_type)

    S12_a_aaa = np.ascontiguousarray(S12_a_bba - S12_a_bba.transpose(0,1,3,2))

    ## S22 block: < Psi_0 | a^{\dag}_U a_V a_X a^{\dag}_Z a^{\dag}_W a_Y | Psi_0 >
    S22_aaa_aaa =- 1/12 * einsum('UWZVYX->UVXYWZ', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa -= 1/12 * einsum('UWZXVY->UVXYWZ', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa -= 1/12 * einsum('UWZYXV->UVXYWZ', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa += 1/6 * einsum('VW,UZXY->UVXYWZ', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa -= 1/6 * einsum('VW,UZYX->UVXYWZ', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa -= 1/6 * einsum('VZ,UWXY->UVXYWZ', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa += 1/6 * einsum('VZ,UWYX->UVXYWZ', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa -= 1/6 * einsum('WX,UZVY->UVXYWZ', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa += 1/6 * einsum('WX,UZYV->UVXYWZ', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa += 1/6 * einsum('XZ,UWVY->UVXYWZ', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa -= 1/6 * einsum('XZ,UWYV->UVXYWZ', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa += 1/2 * einsum('VW,XZ,UY->UVXYWZ', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    S22_aaa_aaa -= 1/2 * einsum('VZ,WX,UY->UVXYWZ', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)

    S22_bba_bba =- 1/12 * einsum('UWZVYX->UVXYWZ', rdm_cccaaa, optimize = einsum_type).copy()
    S22_bba_bba += 1/12 * einsum('UWZXVY->UVXYWZ', rdm_cccaaa, optimize = einsum_type).copy()
    S22_bba_bba += 1/6 * einsum('UWZYVX->UVXYWZ', rdm_cccaaa, optimize = einsum_type).copy()
    S22_bba_bba += 1/12 * einsum('UWZYXV->UVXYWZ', rdm_cccaaa, optimize = einsum_type).copy()
    S22_bba_bba -= 1/6 * einsum('VW,UZXY->UVXYWZ', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_bba_bba -= 1/3 * einsum('VW,UZYX->UVXYWZ', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_bba_bba += 1/6 * einsum('XZ,UWVY->UVXYWZ', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_bba_bba -= 1/6 * einsum('XZ,UWYV->UVXYWZ', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_bba_bba += 1/2 * einsum('VW,XZ,UY->UVXYWZ', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)

    S22_aaa_bba = np.ascontiguousarray(S22_aaa_aaa -
                                       S22_bba_bba.transpose(0,2,1,3,5,4) +
                                       S22_bba_bba.transpose(0,1,2,3,5,4))

    S22_bba_aaa = np.ascontiguousarray(S22_aaa_bba.transpose(3,4,5,0,1,2))

    ## Reshape tensors to matrix form
    dim_X = ncas
    dim_YWZ = ncas * ncas * ncas
    dim_tril_YWZ = ncas * ncas * (ncas - 1) // 2

    dim_act = dim_X + dim_tril_YWZ + dim_YWZ

    tril_ind = np.tril_indices(ncas, k=-1)

    S12_a_aaa = S12_a_aaa[:,:,tril_ind[0],tril_ind[1]]

    S22_aaa_aaa = S22_aaa_aaa[:,:,:,:,tril_ind[0],tril_ind[1]]
    S22_aaa_aaa = S22_aaa_aaa[:,tril_ind[0],tril_ind[1]]

    S22_aaa_bba = S22_aaa_bba[:,tril_ind[0],tril_ind[1]]
    S22_bba_aaa = S22_bba_aaa[:,:,:,:,tril_ind[0],tril_ind[1]]

    S12_a_aaa = S12_a_aaa.reshape(dim_X, dim_tril_YWZ)
    S12_a_bba = S12_a_bba.reshape(dim_X, dim_YWZ)

    S22_aaa_aaa = S22_aaa_aaa.reshape(dim_tril_YWZ, dim_tril_YWZ)
    S22_aaa_bba = S22_aaa_bba.reshape(dim_tril_YWZ, dim_YWZ)

    S22_bba_aaa = S22_bba_aaa.reshape(dim_YWZ, dim_tril_YWZ)
    S22_bba_bba = S22_bba_bba.reshape(dim_YWZ, dim_YWZ)

    # Build S_p1p_act matrix
    S_a_i = 0
    S_a_f = dim_X
    S_aaa_i = S_a_f
    S_aaa_f = S_aaa_i + dim_tril_YWZ
    S_bba_i = S_aaa_f
    S_bba_f = S_bba_i + dim_YWZ

    S_p1p_act = np.zeros((dim_act, dim_act))

    S_p1p_act[S_a_i:S_a_f, S_a_i:S_a_f] = S11_a_a

    S_p1p_act[S_a_i:S_a_f, S_aaa_i:S_aaa_f] = S12_a_aaa
    S_p1p_act[S_a_i:S_a_f, S_bba_i:S_bba_f] = S12_a_bba

    S_p1p_act[S_aaa_i:S_aaa_f, S_a_i:S_a_f] = S12_a_aaa.T
    S_p1p_act[S_bba_i:S_bba_f, S_a_i:S_a_f] = S12_a_bba.T

    S_p1p_act[S_aaa_i:S_aaa_f, S_aaa_i:S_aaa_f] = S22_aaa_aaa
    S_p1p_act[S_aaa_i:S_aaa_f, S_bba_i:S_bba_f] = S22_aaa_bba

    S_p1p_act[S_bba_i:S_bba_f, S_aaa_i:S_aaa_f] = S22_bba_aaa
    S_p1p_act[S_bba_i:S_bba_f, S_bba_i:S_bba_f] = S22_bba_bba

    # Compute projector to the GNO operator basis
    Y = np.identity(S_p1p_act.shape[0])

    Y_a_aaa =- 0.5 * np.einsum("XZ,YW->XYWZ", np.identity(ncas), rdm_ca)
    Y_a_aaa += 0.5 * np.einsum("XW,YZ->XYWZ", np.identity(ncas), rdm_ca)

    Y_a_bba =- 0.5 * np.einsum("XZ,YW->XYWZ", np.identity(ncas), rdm_ca)

    Y_a_aaa = Y_a_aaa[:,:,tril_ind[0],tril_ind[1]]

    Y_a_aaa = Y_a_aaa.reshape(dim_X, dim_tril_YWZ)
    Y_a_bba = Y_a_bba.reshape(dim_X, dim_YWZ)

    Y[S_a_i:S_a_f, S_aaa_i:S_aaa_f] = Y_a_aaa
    Y[S_a_i:S_a_f, S_bba_i:S_bba_f] = Y_a_bba

    # Compute S^{-1/2} matrix
    St_p1p = reduce(np.dot, (Y.T, S_p1p_act, Y))

    S_eval, S_evec = np.linalg.eigh(St_p1p)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    ## Apply damping
    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], mr_adc.s_thresh_singles, mr_adc.s_damping_strength)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_p1p_12_inv_act = reduce(np.dot, (Y, S_evec, np.diag(S_inv_eval)))

    if not ignore_print:
        print("Dimension of the [+1'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print("Smallest eigenvalue of the [+1'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_p1p_12_inv_act

def compute_S12_m1p_gno_projector(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Damping for singles
    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    # Compute S matrix
    ## S11 block: < Psi_0 | a^{\dag}_X a_Z | Psi_0 >
    S11_a_a  = 1/2 * einsum('XY->XY', rdm_ca, optimize = einsum_type).copy()

    ## S12 block: < Psi_0 | a^{\dag}_X a^{\dag}_W a_Z a_Y | Psi_0 >
    S12_a_abb  = 1/6 * einsum('WXYZ->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_abb += 1/3 * einsum('WXZY->XYZW', rdm_ccaa, optimize = einsum_type).copy()

    S12_a_aaa = np.ascontiguousarray(S12_a_abb - S12_a_abb.transpose(0,2,1,3))

    ## S22 block: < Psi_0 | a^{\dag}_X a^{\dag}_U a_V a^{\dag}_W a_Z a_Y | Psi_0 >
    S22_aaa_aaa  = 1/12 * einsum('UWXVZY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa += 1/12 * einsum('UWXYVZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa += 1/12 * einsum('UWXZYV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa -= 1/6 * einsum('VW,UXYZ->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa += 1/6 * einsum('VW,UXZY->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    S22_abb_abb  = 1/12 * einsum('UWXVZY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_abb_abb -= 1/12 * einsum('UWXYVZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_abb_abb -= 1/6 * einsum('UWXZVY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_abb_abb -= 1/12 * einsum('UWXZYV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_abb_abb += 1/6 * einsum('VW,UXYZ->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_abb_abb += 1/3 * einsum('VW,UXZY->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    S22_aaa_abb = np.ascontiguousarray(S22_aaa_aaa -
                                       S22_abb_abb.transpose(1,0,2,4,3,5) +
                                       S22_abb_abb.transpose(0,1,2,4,3,5))

    S22_abb_aaa = np.ascontiguousarray(S22_aaa_abb.transpose(3,4,5,0,1,2))

    ## Reshape tensors to matrix form
    dim_X = ncas
    dim_YWZ = ncas * ncas * ncas
    dim_tril_YWZ = ncas * ncas * (ncas - 1) // 2

    dim_act = dim_X + dim_tril_YWZ + dim_YWZ

    tril_ind = np.tril_indices(ncas, k=-1)

    S12_a_aaa = S12_a_aaa[:,tril_ind[0],tril_ind[1]]

    S22_aaa_aaa = S22_aaa_aaa[:,:,:,tril_ind[0],tril_ind[1]]
    S22_aaa_aaa = S22_aaa_aaa[tril_ind[0],tril_ind[1]]

    S22_aaa_abb = S22_aaa_abb[tril_ind[0],tril_ind[1]]
    S22_abb_aaa = S22_abb_aaa[:,:,:,tril_ind[0],tril_ind[1]]

    S12_a_aaa = S12_a_aaa.reshape(dim_X, dim_tril_YWZ)
    S12_a_abb = S12_a_abb.reshape(dim_X, dim_YWZ)

    S22_aaa_aaa = S22_aaa_aaa.reshape(dim_tril_YWZ, dim_tril_YWZ)
    S22_aaa_abb = S22_aaa_abb.reshape(dim_tril_YWZ, dim_YWZ)

    S22_abb_aaa = S22_abb_aaa.reshape(dim_YWZ, dim_tril_YWZ)
    S22_abb_abb = S22_abb_abb.reshape(dim_YWZ, dim_YWZ)

    # Build S_p1p_act matrix
    S_a_i = 0
    S_a_f = dim_X
    S_aaa_i = S_a_f
    S_aaa_f = S_aaa_i + dim_tril_YWZ
    S_abb_i = S_aaa_f
    S_abb_f = S_abb_i + dim_YWZ

    S_m1p_act = np.zeros((dim_act, dim_act))

    S_m1p_act[S_a_i:S_a_f, S_a_i:S_a_f] = S11_a_a

    S_m1p_act[S_a_i:S_a_f, S_aaa_i:S_aaa_f] = S12_a_aaa
    S_m1p_act[S_a_i:S_a_f, S_abb_i:S_abb_f] = S12_a_abb

    S_m1p_act[S_aaa_i:S_aaa_f, S_a_i:S_a_f] = S12_a_aaa.T
    S_m1p_act[S_abb_i:S_abb_f, S_a_i:S_a_f] = S12_a_abb.T

    S_m1p_act[S_aaa_i:S_aaa_f, S_aaa_i:S_aaa_f] = S22_aaa_aaa
    S_m1p_act[S_aaa_i:S_aaa_f, S_abb_i:S_abb_f] = S22_aaa_abb

    S_m1p_act[S_abb_i:S_abb_f, S_aaa_i:S_aaa_f] = S22_abb_aaa
    S_m1p_act[S_abb_i:S_abb_f, S_abb_i:S_abb_f] = S22_abb_abb

    # Compute projector to the GNO operator basis
    Y = np.identity(S_m1p_act.shape[0])

    Y_a_aaa =- 0.5 * np.einsum("XY,ZW->XYZW", np.identity(ncas), rdm_ca)
    Y_a_aaa += 0.5 * np.einsum("XZ,YW->XYZW", np.identity(ncas), rdm_ca)

    Y_a_abb =- 0.5 * np.einsum("XY,ZW->XYZW", np.identity(ncas), rdm_ca)

    Y_a_aaa = Y_a_aaa[:,tril_ind[0],tril_ind[1]]

    Y_a_aaa = Y_a_aaa.reshape(dim_X, dim_tril_YWZ)
    Y_a_abb = Y_a_abb.reshape(dim_X, dim_YWZ)

    Y[S_a_i:S_a_f, S_aaa_i:S_aaa_f] = Y_a_aaa
    Y[S_a_i:S_a_f, S_abb_i:S_abb_f] = Y_a_abb

    # Compute S^{-1/2} matrix
    St_m1p = reduce(np.dot, (Y.T, S_m1p_act, Y))

    S_eval, S_evec = np.linalg.eigh(St_m1p)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    ## Apply damping
    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], mr_adc.s_thresh_singles, mr_adc.s_damping_strength)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_m1p_12_inv_act = reduce(np.dot, (Y, S_evec, np.diag(S_inv_eval)))

    if not ignore_print:
        print("Dimension of the [-1'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print("Smallest eigenvalue of the [-1'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_m1p_12_inv_act

# Spin-Orbital Sanity Check Functions
def compute_S12_0p_sanity_check_gno_projector(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    dim = 1 + ncas * 2 * ncas * 2

    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    S_0p = np.zeros((dim, dim))

    S_0p[0,0] = 1.0

    S12 = np.zeros((ncas * 2, ncas * 2))

    S12_a_a  = 1/2 * einsum('XY->XY', rdm_ca, optimize = einsum_type).copy()
    S12[::2,::2] = S12_a_a.copy()
    S12[1::2,1::2] = S12_a_a.copy()

    S22 = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2))

    S22_ab_ab =- 1/3 * einsum('XZWY->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S22_ab_ab -= 1/6 * einsum('XZYW->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S22_ab_ab += 1/2 * einsum('YZ,XW->XYZW', np.identity(ncas), rdm_ca, optimize = einsum_type)
    S22_ab_ab = S22_ab_ab.transpose(0,1,3,2)

    S22_aa_bb  = 1/6 * einsum('XZWY->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_bb += 1/3 * einsum('XZYW->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_bb = S22_aa_bb.transpose(0,1,3,2)

    S22[::2,1::2,::2,1::2] = S22_ab_ab.copy()
    S22[1::2,::2,1::2,::2] = S22_ab_ab.copy()

    S22[::2,::2,1::2,1::2] = S22_aa_bb.copy()
    S22[1::2,1::2,::2,::2] = S22_aa_bb.copy()

    S22[::2,::2,::2,::2]  = S22_ab_ab.copy()
    S22[::2,::2,::2,::2] += S22_aa_bb.copy()
    S22[1::2,1::2,1::2,1::2] = S22[::2,::2,::2,::2].copy()

    S_0p[0,1:] = S12.reshape(-1).copy()
    S_0p[1:,0] = S12.T.reshape(-1).copy()
    S_0p[1:,1:] = S22.reshape(ncas * 2 * ncas * 2, ncas * 2 * ncas * 2).copy()

    # Compute projector to the GNO operator basis
    rdm_ca_so = np.zeros((ncas * 2, ncas * 2))
    rdm_ca_so[::2,::2] = 0.5 * rdm_ca
    rdm_ca_so[1::2,1::2] = 0.5 * rdm_ca

    Y = np.identity(S_0p.shape[0])
    Y[0,1:] =- rdm_ca_so.reshape(-1).copy()

    St = reduce(np.dot, (Y.T, S_0p, Y))

    S_eval, S_evec = np.linalg.eigh(St)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], mr_adc.s_thresh_singles, mr_adc.s_damping_strength)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_0p_12_inv = reduce(np.dot, (Y, S_evec, np.diag(S_inv_eval)))

    if not ignore_print:
        print("Dimension of the [0'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print("Smallest eigenvalue of the [0'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_0p_12_inv

def compute_S12_p1p_sanity_check_gno_projector(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    ncas = mr_adc.ncas

    n_x = ncas * 2
    n_xzw = ncas * 2 * ncas * 2 * (ncas * 2 - 1) // 2
    dim_act = n_x + n_xzw
    xy_ind = np.tril_indices(ncas * 2, k=-1)

    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    S_act = np.zeros((dim_act, dim_act))

    S11 = np.zeros((ncas * 2, ncas * 2))

    S11_a_a  = einsum('XY->XY', np.identity(ncas), optimize = einsum_type).copy()
    S11_a_a -= 1/2 * einsum('YX->XY', rdm_ca, optimize = einsum_type).copy()

    S11[::2,::2] = S11_a_a.copy()
    S11[1::2,1::2] = S11_a_a.copy()

    S12 = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2))

    S12_a_bba =- 1/6 * einsum('WYXZ->XZWY', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_bba -= 1/3 * einsum('WYZX->XZWY', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_bba += 1/2 * einsum('XY,WZ->XZWY', np.identity(ncas), rdm_ca, optimize = einsum_type)

    S12[::2,1::2,1::2,::2] = S12_a_bba.copy()
    S12[1::2,::2,::2,1::2] = S12_a_bba.copy()

    S12[::2,1::2,::2,1::2] -= S12_a_bba.transpose(0,1,3,2).copy()
    S12[1::2,::2,1::2,::2]  = S12[::2,1::2,::2,1::2].copy()

    S12[::2,::2,::2,::2]  = S12_a_bba.copy()
    S12[::2,::2,::2,::2] += S12[::2,1::2,::2,1::2].copy()
    S12[1::2,1::2,1::2,1::2] = S12[::2,::2,::2,::2].copy()

    S22 = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2, ncas * 2, ncas * 2))

    S22_aab_aab =- 1/12 * einsum('UWYVZX->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab += 1/12 * einsum('UWYXVZ->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab += 1/6 * einsum('UWYZVX->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab += 1/12 * einsum('UWYZXV->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab -= 1/6 * einsum('VW,UYXZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aab_aab -= 1/3 * einsum('VW,UYZX->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aab_aab += 1/6 * einsum('XY,UWVZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aab_aab -= 1/6 * einsum('XY,UWZV->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aab_aab += 1/2 * einsum('VW,XY,UZ->UVXZWY', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)

    S22_baa_baa  = 1/12 * einsum('UWYVZX->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa += 1/12 * einsum('UWYXVZ->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa += 1/6 * einsum('UWYZVX->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa -= 1/12 * einsum('UWYZXV->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa -= 1/6 * einsum('VW,UYXZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_baa_baa -= 1/3 * einsum('VW,UYZX->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_baa_baa += 1/6 * einsum('VY,UWXZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_baa_baa += 1/3 * einsum('VY,UWZX->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_baa_baa += 1/6 * einsum('WX,UYVZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_baa_baa += 1/3 * einsum('WX,UYZV->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_baa_baa -= 1/6 * einsum('XY,UWVZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_baa_baa -= 1/3 * einsum('XY,UWZV->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_baa_baa += 1/2 * einsum('VW,XY,UZ->UVXZWY', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    S22_baa_baa -= 1/2 * einsum('VY,WX,UZ->UVXZWY', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)

    S22[::2,::2,1::2,::2,::2,1::2] = S22_aab_aab.copy()
    S22[1::2,1::2,::2,1::2,1::2,::2] = S22_aab_aab.copy()

    S22[1::2,::2,::2,1::2,::2,::2] = S22_baa_baa.copy()
    S22[::2,1::2,1::2,::2,1::2,1::2] = S22_baa_baa.copy()

    S22[::2,1::2,::2,::2,1::2,::2] = S22_aab_aab.transpose(0,2,1,3,5,4).copy()
    S22[1::2,::2,1::2,1::2,::2,1::2] = S22[::2,1::2,::2,::2,1::2,::2].copy()

    S22[::2,1::2,::2,::2,::2,1::2] -= S22_aab_aab.transpose(0,2,1,3,4,5).copy()
    S22[1::2,::2,1::2,1::2,1::2,::2] = S22[::2,1::2,::2,::2,::2,1::2].copy()

    S22[::2,::2,1::2,::2,1::2,::2] -= S22_aab_aab.transpose(0,1,2,3,5,4).copy()
    S22[1::2,1::2,::2,1::2,::2,1::2] = S22[::2,::2,1::2,::2,1::2,::2].copy()

    S22[::2,::2,::2,1::2,1::2,::2]  = S22_aab_aab.copy()
    S22[::2,::2,::2,1::2,1::2,::2] -= S22_baa_baa.copy()
    S22[::2,::2,::2,1::2,1::2,::2] += S22[::2,1::2,::2,::2,::2,1::2].copy()
    S22[1::2,1::2,1::2,::2,::2,1::2] = S22[::2,::2,::2,1::2,1::2,::2].copy()

    S22[1::2,1::2,::2,::2,::2,::2] = S22[::2,::2,::2,1::2,1::2,::2].transpose(3,4,5,0,1,2).copy()
    S22[::2,::2,1::2,1::2,1::2,1::2] = S22[1::2,1::2,::2,::2,::2,::2].copy()

    S22[1::2,::2,1::2,::2,::2,::2] -= S22[1::2,1::2,::2,::2,::2,::2].transpose(0,2,1,3,4,5).copy()
    S22[::2,1::2,::2,1::2,1::2,1::2] = S22[1::2,::2,1::2,::2,::2,::2].copy()

    S22[::2,::2,::2,1::2,::2,1::2] -= S22[::2,::2,::2,1::2,1::2,::2].transpose(0,1,2,3,5,4).copy()
    S22[1::2,1::2,1::2,::2,1::2,::2] = S22[::2,::2,::2,1::2,::2,1::2].copy()

    S22[::2,::2,::2,::2,::2,::2]  = S22[::2,::2,::2,1::2,1::2,::2].copy()
    S22[::2,::2,::2,::2,::2,::2] += S22[1::2,1::2,::2,1::2,::2,1::2].copy()
    S22[::2,::2,::2,::2,::2,::2] += S22[1::2,::2,1::2,1::2,::2,1::2].copy()
    S22[1::2,1::2,1::2,1::2,1::2,1::2] = S22[::2,::2,::2,::2,::2,::2].copy()

    # ###
    # # S22[::2,::2,1::2,::2,::2,1::2] = S22_aab_aab.copy()
    # S22[1::2,1::2,::2,1::2,1::2,::2] = S22_aab_aab.copy()

    # # S22[1::2,::2,::2,1::2,::2,::2] = S22_baa_baa.copy()
    # # S22[::2,1::2,1::2,::2,1::2,1::2] = S22_baa_baa.copy()

    # # S22[::2,1::2,::2,::2,1::2,::2] = S22_aab_aab.transpose(0,2,1,3,5,4).copy()
    # S22[1::2,::2,1::2,1::2,::2,1::2] = S22_aab_aab.transpose(0,2,1,3,5,4).copy()

    # # S22[::2,1::2,::2,::2,::2,1::2] -= S22_aab_aab.transpose(0,2,1,3,4,5).copy()
    # S22[1::2,::2,1::2,1::2,1::2,::2] -= S22_aab_aab.transpose(0,2,1,3,4,5).copy()

    # # S22[::2,::2,1::2,::2,1::2,::2] -= S22_aab_aab.transpose(0,1,2,3,5,4).copy()
    # S22[1::2,1::2,::2,1::2,::2,1::2] -= S22_aab_aab.transpose(0,1,2,3,5,4).copy()

    # S22[::2,::2,::2,1::2,1::2,::2]  = S22_aab_aab.copy()
    # S22[::2,::2,::2,1::2,1::2,::2] -= S22_baa_baa.copy()
    # S22[::2,::2,::2,1::2,1::2,::2] -= S22_aab_aab.transpose(0,2,1,3,4,5).copy()

    # # S22[1::2,1::2,1::2,::2,::2,1::2]  = S22_aab_aab.copy()
    # # S22[1::2,1::2,1::2,::2,::2,1::2] -= S22_baa_baa.copy()
    # # S22[1::2,1::2,1::2,::2,::2,1::2] -= S22_aab_aab.transpose(0,2,1,3,4,5).copy()

    # S22[1::2,1::2,::2,::2,::2,::2]  = S22_aab_aab.copy()
    # S22[1::2,1::2,::2,::2,::2,::2] -= S22_baa_baa.copy()
    # S22[1::2,1::2,::2,::2,::2,::2] -= S22_aab_aab.transpose(0,1,2,3,5,4).copy()

    # # S22[::2,::2,1::2,1::2,1::2,1::2]  = S22_aab_aab.copy()
    # # S22[::2,::2,1::2,1::2,1::2,1::2] -= S22_baa_baa.copy()
    # # S22[::2,::2,1::2,1::2,1::2,1::2] -= S22_aab_aab.transpose(0,1,2,3,5,4).copy()

    # S22[1::2,::2,1::2,::2,::2,::2] -= S22_aab_aab.transpose(0,2,1,3,4,5).copy()
    # S22[1::2,::2,1::2,::2,::2,::2] -= S22_baa_baa.copy()
    # S22[1::2,::2,1::2,::2,::2,::2] += S22_aab_aab.transpose(0,2,1,3,5,4).copy()

    # # S22[::2,1::2,::2,1::2,1::2,1::2] -= S22_aab_aab.transpose(0,2,1,3,4,5).copy()
    # # S22[::2,1::2,::2,1::2,1::2,1::2] -= S22_baa_baa.copy()
    # # S22[::2,1::2,::2,1::2,1::2,1::2] += S22_aab_aab.transpose(0,2,1,3,5,4).copy()

    # S22[::2,::2,::2,1::2,::2,1::2] -= S22_aab_aab.transpose(0,1,2,3,5,4).copy()
    # S22[::2,::2,::2,1::2,::2,1::2] -= S22_baa_baa.copy()
    # S22[::2,::2,::2,1::2,::2,1::2] += S22_aab_aab.transpose(0,2,1,3,5,4).copy()

    # # S22[1::2,1::2,1::2,::2,1::2,::2] -= S22_aab_aab.transpose(0,1,2,3,5,4).copy()
    # # S22[1::2,1::2,1::2,::2,1::2,::2] -= S22_baa_baa.copy()
    # # S22[1::2,1::2,1::2,::2,1::2,::2] += S22_aab_aab.transpose(0,2,1,3,5,4).copy()

    # S22[::2,::2,::2,::2,::2,::2]  = S22_aab_aab.copy()
    # S22[::2,::2,::2,::2,::2,::2] -= S22_baa_baa.copy()
    # S22[::2,::2,::2,::2,::2,::2] -= S22_aab_aab.transpose(0,2,1,3,4,5).copy()
    # S22[::2,::2,::2,::2,::2,::2] -= S22_aab_aab.transpose(0,1,2,3,5,4).copy()
    # S22[::2,::2,::2,::2,::2,::2] += S22_aab_aab.transpose(0,2,1,3,5,4).copy()

    # # S22[1::2,1::2,1::2,1::2,1::2,1::2]  = S22_aab_aab.copy()
    # # S22[1::2,1::2,1::2,1::2,1::2,1::2] -= S22_baa_baa.copy()
    # # S22[1::2,1::2,1::2,1::2,1::2,1::2] -= S22_aab_aab.transpose(0,2,1,3,4,5).copy()
    # # S22[1::2,1::2,1::2,1::2,1::2,1::2] -= S22_aab_aab.transpose(0,1,2,3,5,4).copy()
    # # S22[1::2,1::2,1::2,1::2,1::2,1::2] += S22_aab_aab.transpose(0,2,1,3,5,4).copy()
    # ###

    # print(">>> SA (sanity) S12 (a-aaa) norm: {:}".format(np.linalg.norm(S12[::2,::2,::2,::2])))
    # print(">>> SA (sanity) S12 (a-bba) norm: {:}".format(np.linalg.norm(S12[::2,1::2,1::2,::2])))
    # print(">>> SA (sanity) S12 (a-bab) norm: {:}".format(np.linalg.norm(S12[::2,1::2,::2,1::2])))

    # print(">>> SA (sanity) S11 norm: {:}".format(np.linalg(S11)))
    # print(">>> SA (sanity) S12 norm: {:}".format(np.linalg.norm(S12)))
    # print(">>> SA (sanity) S22 norm: {:}".format(np.linalg.norm(S22)))
    # print(">>> SA (sanity) S_act norm: {:}".format(np.sum(S11) + 2*np.sum(S12) + np.sum(S22)))

    # TESTS
    # S_act_test = np.zeros((ncas * 2 + ncas * 2 * ncas * 2 * ncas * 2, ncas * 2 + ncas * 2 * ncas * 2 * ncas * 2))
    # S_act_test[:(ncas*2),:(ncas*2)] = S11
    # S_act_test[:(ncas*2),(ncas*2):] = S12.reshape(ncas*2,ncas*2*ncas*2*ncas*2)
    # S_act_test[(ncas*2):,:(ncas*2)] = S12.reshape(ncas*2,ncas*2*ncas*2*ncas*2).T
    # S_act_test[(ncas*2):,(ncas*2):] = S22.reshape(ncas*2*ncas*2*ncas*2,ncas*2*ncas*2*ncas*2)

    # print(">>> SA (sanity) S_act_test norm: {:}".format(np.linalg.norm(S_act_test)))
    # if mr_adc.s_damping_strength is None:
    #     s_thresh = mr_adc.s_thresh_singles
    # else:
    #     s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    # Y = np.identity(S_act_test.shape[0])

    # rdm_ca_so = np.zeros((ncas * 2, ncas * 2))
    # rdm_ca_so[::2,::2] = 0.5 * rdm_ca
    # rdm_ca_so[1::2,1::2] = 0.5 * rdm_ca

    # Y_ten = -np.einsum("uw,vx->uvxw", np.identity(ncas * 2), rdm_ca_so)
    # Y_ten += np.einsum("ux,vw->uvxw", np.identity(ncas * 2), rdm_ca_so)

    # S_act_test[:(ncas*2),(ncas*2):] = Y_ten.reshape(ncas*2,ncas*2*ncas*2*ncas*2)

    # St = reduce(np.dot, (Y.T, S_act_test, Y))

    # S_eval, S_evec = np.linalg.eigh(St)

    # np.save('S_eval_test', S_eval)
    # np.save('S_evec_test', S_evec)
    # TESTS

    # np.save('SA_S22', S22)
    S12 = S12[:,:,xy_ind[0],xy_ind[1]]
    S22 = S22[:,:,:,:,xy_ind[0],xy_ind[1]]
    S22 = S22[:,xy_ind[0],xy_ind[1]]

    S_act[:n_x,:n_x] = S11.copy()
    S_act[:n_x,n_x:] = S12.reshape(n_x, n_xzw)
    S_act[n_x:,:n_x] = S12.reshape(n_x, n_xzw).T
    S_act[n_x:,n_x:] = S22.reshape(n_xzw, n_xzw)

    # np.save('SA_S_act', S_act)
    # print(">>> SA (sanity) S_act norm: {:}".format(np.linalg.norm(S_act)))

    ### DEBUG: Turning off
    # Compute projector to the GNO operator basis
    Y = np.identity(S_act.shape[0])

    rdm_ca_so = np.zeros((ncas * 2, ncas * 2))
    rdm_ca_so[::2,::2] = 0.5 * rdm_ca
    rdm_ca_so[1::2,1::2] = 0.5 * rdm_ca

    Y_ten = -np.einsum("uw,vx->uvxw", np.identity(ncas * 2), rdm_ca_so)
    Y_ten += np.einsum("ux,vw->uvxw", np.identity(ncas * 2), rdm_ca_so)

    Y_ten[1::2,::2,::2,1::2] = 0.0
    Y_ten[1::2,::2,1::2,::2] = 0.0
    Y_ten[1::2,1::2,1::2,1::2] = 0.0

    Y[:n_x,n_x:] = Y_ten[:,:,xy_ind[0],xy_ind[1]].reshape(n_x, n_xzw)

    St = reduce(np.dot, (Y.T, S_act, Y))

    S_eval, S_evec = np.linalg.eigh(St)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]
    ### DEBUG: Turning off

    # S_eval, S_evec = np.linalg.eigh(S_act)
    # S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], mr_adc.s_thresh_singles, mr_adc.s_damping_strength)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_p1p_12_inv_act = reduce(np.dot, (Y, S_evec, np.diag(S_inv_eval)))
    # S_p1p_12_inv_act = reduce(np.dot, (S_evec, np.diag(S_inv_eval)))

    if not ignore_print:
        print("Dimension of the [+1'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print("Smallest eigenvalue of the [+1'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_p1p_12_inv_act

def compute_S12_m1p_sanity_check_gno_projector(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    ncas = mr_adc.ncas

    n_x = ncas * 2
    n_xzw = ncas * 2 * ncas * 2 * (ncas * 2 - 1) // 2
    dim_act = n_x + n_xzw
    xy_ind = np.tril_indices(ncas * 2, k=-1)

    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    S_act = np.zeros((dim_act, dim_act))

    S11 = np.zeros((ncas * 2, ncas * 2))

    S11_a_a  = 1/2 * einsum('XY->XY', rdm_ca, optimize = einsum_type).copy()

    S11[::2,::2] = S11_a_a.copy()
    S11[1::2,1::2] = S11_a_a.copy()

    S12 = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2))

    S12_a_abb  = 1/6 * einsum('WXYZ->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_abb += 1/3 * einsum('WXZY->XYZW', rdm_ccaa, optimize = einsum_type).copy()

    S12[::2,::2,1::2,1::2] = S12_a_abb.copy()
    S12[1::2,1::2,::2,::2] = S12_a_abb.copy()

    S12[::2,1::2,::2,1::2] -= S12[::2,::2,1::2,1::2].transpose(0,2,1,3).copy()
    S12[1::2,::2,1::2,::2]  = S12[::2,1::2,::2,1::2].copy()

    S12[::2,::2,::2,::2]  = S12_a_abb.copy()
    S12[::2,::2,::2,::2] += S12[::2,1::2,::2,1::2].copy()
    S12[1::2,1::2,1::2,1::2] = S12[::2,::2,::2,::2].copy()

    S22 = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2, ncas * 2, ncas * 2))

    S22_aab_aab =- 1/12 * einsum('UWXVZY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab += 1/12 * einsum('UWXYVZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab -= 1/6 * einsum('UWXZVY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab -= 1/12 * einsum('UWXZYV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab -= 1/6 * einsum('VW,UXYZ->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aab_aab += 1/6 * einsum('VW,UXZY->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    S22_baa_baa  = 1/12 * einsum('UWXVZY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa -= 1/12 * einsum('UWXYVZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa -= 1/6 * einsum('UWXZVY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa -= 1/12 * einsum('UWXZYV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa += 1/6 * einsum('VW,UXYZ->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_baa_baa += 1/3 * einsum('VW,UXZY->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    S22[::2,::2,1::2,::2,::2,1::2] = S22_aab_aab.copy()
    S22[1::2,1::2,::2,1::2,1::2,::2] = S22_aab_aab.copy()

    S22[1::2,::2,::2,1::2,::2,::2] = S22_baa_baa.copy()
    S22[::2,1::2,1::2,::2,1::2,1::2] = S22_baa_baa.copy()

    S22[1::2,::2,::2,::2,1::2,::2] -= S22_baa_baa.transpose(0,1,2,4,3,5).copy()
    S22[::2,1::2,1::2,1::2,::2,1::2] = S22[1::2,::2,::2,::2,1::2,::2].copy()

    S22[::2,1::2,::2,1::2,::2,::2] -= S22_baa_baa.transpose(1,0,2,3,4,5).copy()
    S22[1::2,::2,1::2,::2,1::2,1::2] = S22[::2,1::2,::2,1::2,::2,::2].copy()

    S22[::2,1::2,::2,::2,1::2,::2] = S22_baa_baa.transpose(1,0,2,4,3,5).copy()
    S22[1::2,::2,1::2,1::2,::2,1::2] = S22[::2,1::2,::2,::2,1::2,::2].copy()

    S22[::2,::2,::2,::2,1::2,1::2]  = S22_baa_baa.copy()
    S22[::2,::2,::2,::2,1::2,1::2] -= S22_aab_aab.copy()
    S22[::2,::2,::2,::2,1::2,1::2] += S22[::2,1::2,::2,1::2,::2,::2].copy()
    S22[1::2,1::2,1::2,1::2,::2,::2] = S22[::2,::2,::2,::2,1::2,1::2].copy()

    S22[::2,::2,::2,1::2,::2,1::2] -= S22[::2,::2,::2,::2,1::2,1::2].transpose(0,1,2,4,3,5).copy()
    S22[1::2,1::2,1::2,::2,1::2,::2] = S22[::2,::2,::2,1::2,::2,1::2].copy()

    S22[::2,1::2,1::2,::2,::2,::2] = S22[::2,::2,::2,::2,1::2,1::2].transpose(3,4,5,0,1,2).copy()
    S22[1::2,::2,::2,1::2,1::2,1::2] = S22[::2,1::2,1::2,::2,::2,::2].copy()

    S22[1::2,::2,1::2,::2,::2,::2] -= S22[::2,1::2,1::2,::2,::2,::2].transpose(1,0,2,3,4,5).copy()
    S22[::2,1::2,::2,1::2,1::2,1::2] = S22[1::2,::2,1::2,::2,::2,::2].copy()

    S22[::2,::2,::2,::2,::2,::2]  = S22[::2,::2,::2,::2,1::2,1::2].copy()
    S22[::2,::2,::2,::2,::2,::2] += S22[::2,1::2,1::2,1::2,::2,1::2].copy()
    S22[::2,::2,::2,::2,::2,::2] += S22[1::2,::2,1::2,1::2,::2,1::2].copy()
    S22[1::2,1::2,1::2,1::2,1::2,1::2] = S22[::2,::2,::2,::2,::2,::2].copy()

    S12 = S12[:,xy_ind[0],xy_ind[1]]
    S22 = S22[:,:,:,xy_ind[0],xy_ind[1]]
    S22 = S22[xy_ind[0],xy_ind[1]]

    S_act[:n_x,:n_x] = S11.copy()
    S_act[:n_x,n_x:] = S12.reshape(n_x, n_xzw)
    S_act[n_x:,:n_x] = S12.reshape(n_x, n_xzw).T
    S_act[n_x:,n_x:] = S22.reshape(n_xzw, n_xzw)

    # Compute projector to the GNO operator basis
    rdm_ca_so = np.zeros((ncas * 2, ncas * 2))
    rdm_ca_so[::2,::2] = 0.5 * rdm_ca
    rdm_ca_so[1::2,1::2] = 0.5 * rdm_ca

    Y_ten = -np.einsum("uv,wx->uvwx", np.identity(ncas * 2), rdm_ca_so)
    Y_ten += np.einsum("uw,vx->uvwx", np.identity(ncas * 2), rdm_ca_so)

    Y = np.identity(S_act.shape[0])
    Y[:n_x,n_x:] = Y_ten[:,xy_ind[0],xy_ind[1]].reshape(n_x, n_xzw)

    St = reduce(np.dot, (Y.T, S_act, Y))

    S_eval, S_evec = np.linalg.eigh(St)

    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], mr_adc.s_thresh_singles, mr_adc.s_damping_strength)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_m1p_12_inv_act = reduce(np.dot, (Y, S_evec, np.diag(S_inv_eval)))

    if not ignore_print:
        print("Dimension of the [-1'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print("Smallest eigenvalue of the [-1'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_m1p_12_inv_act

# Damping Function
## Calculate logarithmic sigmoid damping prefactors for overlap eigenvalues in the range (damping_max, damping_min)
def compute_damping(s_evals, damping_center, damping_strength):

    def sigmoid(x, shift, scale):
        x = np.log10(x)
        return 1 / (1 + np.exp(scale * (shift - x)))

    center = np.log10(damping_center)
    scale_sigmoid = 10 / damping_strength

    return sigmoid(s_evals, center, scale_sigmoid)

# Under development
def compute_S12_0p_projector(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    nelecas = sum(mr_adc.nelecas)

    if nelecas == 0:
        nelecas = 1

    # dim = 1 + (ncas * 2) * (ncas * 2)
    dim = 1 + ncas * ncas

    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    rdm_caca  = einsum('WYZX->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    rdm_caca += einsum('YZ,XW->XYZW', np.identity(ncas), rdm_ca, optimize = einsum_type)

    # Compute projector to remove linear dependencies
    Q  = np.einsum("xw,yz->xywz", np.identity(ncas), np.identity(ncas)).copy()
    Q -= np.einsum("xy,uuzw->xywz", np.identity(ncas), rdm_caca) / (nelecas ** 2)

    Q = Q.reshape(ncas**2, ncas**2).copy()

    S22 = rdm_caca.reshape(ncas**2, ncas**2).copy()
    S22 = np.dot(S22, Q)

    S_eval, S_evec = np.linalg.eigh(S22)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    S_evec = S_evec[:, S_ind_nonzero]

    S22_12_inv = reduce(np.dot, (Q, S_evec, np.diag(S_inv_eval)))

    S_0p_12_inv = np.zeros((dim, 1 + S22_12_inv.shape[1]))
    S_0p_12_inv[0,0] = 1.0
    S_0p_12_inv[1:,1:] = S22_12_inv.copy()

    if not ignore_print:
        print("Dimension of the [0'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print("Smallest eigenvalue of the [0'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_0p_12_inv

