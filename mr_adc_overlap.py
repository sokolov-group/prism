import numpy as np
from functools import reduce

def compute_S12_p1(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas
    rdm_ca = mr_adc.rdm.ca

    S_p1  = einsum('XY->XY', np.identity(ncas), optimize = einsum_type).copy()
    S_p1 -= 1/2 * einsum('YX->XY', rdm_ca, optimize = einsum_type).copy()

    s_thresh = mr_adc.s_thresh_doubles
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

    S_m1  = 1/2 * einsum('XY->XY', rdm_ca, optimize = einsum_type).copy()

    s_thresh = mr_adc.s_thresh_doubles
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

    S_p2  = 1/3 * einsum('XYWZ->ZWYX', rdm_ccaa, optimize = einsum_type).copy()
    S_p2 += 1/6 * einsum('XYZW->ZWYX', rdm_ccaa, optimize = einsum_type).copy()
    S_p2 += einsum('WX,YZ->ZWYX', np.identity(ncas), np.identity(ncas), optimize = einsum_type)
    S_p2 -= 1/2 * einsum('WX,YZ->ZWYX', np.identity(ncas), rdm_ca, optimize = einsum_type)
    S_p2 -= 1/2 * einsum('YZ,XW->ZWYX', np.identity(ncas), rdm_ca, optimize = einsum_type)

    S_p2 = S_p2.reshape(ncas**2, ncas**2)

    s_thresh = mr_adc.s_thresh_doubles
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
    S12[1::2,::2,::2,1::2] = S12[::2,1::2,1::2,::2].copy()

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

    S12 = S12[:,:,xy_ind[0],xy_ind[1]]
    S22 = S22[:,:,:,:,xy_ind[0],xy_ind[1]]
    S22 = S22[:,xy_ind[0],xy_ind[1]]

    S_act[:n_x,:n_x] = S11.copy()
    S_act[:n_x,n_x:] = S12.reshape(n_x, n_xzw)
    S_act[n_x:,:n_x] = S12.reshape(n_x, n_xzw).T
    S_act[n_x:,n_x:] = S22.reshape(n_xzw, n_xzw)

    # Compute projector to the GNO operator basis
    Y = np.identity(S_act.shape[0])

    rdm_ca_so = np.zeros((ncas * 2, ncas * 2))
    rdm_ca_so[::2,::2] = 0.5 * rdm_ca
    rdm_ca_so[1::2,1::2] = 0.5 * rdm_ca

    Y_ten = -np.einsum("uw,vx->uvxw", np.identity(ncas * 2), rdm_ca_so)
    Y_ten += np.einsum("ux,vw->uvxw", np.identity(ncas * 2), rdm_ca_so)

    Y[:n_x,n_x:] = Y_ten[:,:,xy_ind[0],xy_ind[1]].reshape(n_x, n_xzw)

    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    St = reduce(np.dot, (Y.T, S_act, Y))

    S_eval, S_evec = np.linalg.eigh(St)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

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

    S22_ab_ab =- 1/3 * einsum('XZWY->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_ab_ab -= 1/6 * einsum('XZYW->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_ab_ab += 1/2 * einsum('YZ,XW->XYWZ', np.identity(ncas), rdm_ca, optimize = einsum_type)

    S22_aa_bb  = 1/6 * einsum('XZWY->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_bb += 1/3 * einsum('XZYW->XYWZ', rdm_ccaa, optimize = einsum_type).copy()

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

# This function calculates logarithmic sigmoid damping prefactors for overlap eigenvalues in the range (damping_max, damping_min)
def compute_damping(s_evals, damping_center, damping_strength):

    def sigmoid(x, shift, scale):
        x = np.log10(x)
        return 1 / (1 + np.exp(scale * (shift - x)))

    center = np.log10(damping_center)
    scale_sigmoid = 10 / damping_strength

    return sigmoid(s_evals, center, scale_sigmoid)

# Under development
def compute_S12_0p_test_gno_projector(mr_adc, ignore_print = True):

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

    S22_ab_ab =- 1/3 * einsum('XZWY->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_ab_ab -= 1/6 * einsum('XZYW->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_ab_ab += 1/2 * einsum('YZ,XW->XYWZ', np.identity(ncas), rdm_ca, optimize = einsum_type)

    S22_aa_bb  = 1/6 * einsum('XZWY->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_bb += 1/3 * einsum('XZYW->XYWZ', rdm_ccaa, optimize = einsum_type).copy()

    # S22[::2,1::2,::2,1::2] = S22_ab_ab.copy()
    # S22[1::2,::2,1::2,::2] = S22_ab_ab.copy()

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

def compute_S12_p1p_test_gno_projector(mr_adc, ignore_print = True):

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
    S12[1::2,::2,::2,1::2] = S12[::2,1::2,1::2,::2].copy()

    S12[::2,1::2,::2,1::2] -= S12_a_bba.transpose(0,1,3,2).copy()
    S12[1::2,::2,1::2,::2]  = S12[::2,1::2,::2,1::2].copy()

    S12[::2,::2,::2,::2]  = S12_a_bba.copy()
    S12[::2,::2,::2,::2] -= S12_a_bba.transpose(0,1,3,2).copy()
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

    # decoupled # S22[1::2,::2,::2,1::2,::2,::2] = S22_baa_baa.copy()
    # decoupled # S22[::2,1::2,1::2,::2,1::2,1::2] = S22_baa_baa.copy()

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
    S22[::2,::2,1::2,1::2,1::2,1::2] = S22[::2,::2,::2,1::2,1::2,::2].transpose(3,4,5,0,1,2).copy()

    S22[1::2,::2,1::2,::2,::2,::2] -= S22[1::2,1::2,::2,::2,::2,::2].transpose(0,2,1,3,4,5).copy()
    S22[::2,1::2,::2,1::2,1::2,1::2] = S22[1::2,::2,1::2,::2,::2,::2].copy()

    S22[::2,::2,::2,1::2,::2,1::2] -= S22[::2,::2,::2,1::2,1::2,::2].transpose(0,1,2,3,5,4).copy()
    S22[1::2,1::2,1::2,::2,1::2,::2] = S22[::2,::2,::2,1::2,::2,1::2].copy()

    S22[::2,::2,::2,::2,::2,::2]  = S22[::2,::2,::2,1::2,1::2,::2].copy()
    S22[::2,::2,::2,::2,::2,::2] -= S22_aab_aab.transpose(0,1,2,3,5,4).copy()
    S22[::2,::2,::2,::2,::2,::2] += S22_aab_aab.transpose(0,2,1,3,5,4).copy()
    S22[1::2,1::2,1::2,1::2,1::2,1::2] = S22[::2,::2,::2,::2,::2,::2].copy()

    S12 = S12[:,:,xy_ind[0],xy_ind[1]]
    S22 = S22[:,:,:,:,xy_ind[0],xy_ind[1]]
    S22 = S22[:,xy_ind[0],xy_ind[1]]

    S_act[:n_x,:n_x] = S11.copy()
    S_act[:n_x,n_x:] = S12.reshape(n_x, n_xzw)
    S_act[n_x:,:n_x] = S12.reshape(n_x, n_xzw).T
    S_act[n_x:,n_x:] = S22.reshape(n_xzw, n_xzw)

    # Compute projector to the GNO operator basis
    Y = np.identity(S_act.shape[0])

    rdm_ca_so = np.zeros((ncas * 2, ncas * 2))
    rdm_ca_so[::2,::2] = 0.5 * rdm_ca
    rdm_ca_so[1::2,1::2] = 0.5 * rdm_ca

    Y_ten = -np.einsum("uw,vx->uvxw", np.identity(ncas * 2), rdm_ca_so)
    Y_ten += np.einsum("ux,vw->uvxw", np.identity(ncas * 2), rdm_ca_so)

    Y[:n_x,n_x:] = Y_ten[:,:,xy_ind[0],xy_ind[1]].reshape(n_x, n_xzw)

    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    St = reduce(np.dot, (Y.T, S_act, Y))

    S_eval, S_evec = np.linalg.eigh(St)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

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

def compute_S12_0p_gno_projector(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Damping for Singles
    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    # Defining dimensions and S_0p tensor
    S_dim = 1 + ncas * ncas * 2
    S12_dim = 1 + ncas * ncas
    S22_dim = ncas * ncas

    S_0p = np.zeros((S_dim, S_dim))

    # Computing S12 and S22 blocks
    S12_a_a = 1/2 * einsum('XY->XY', rdm_ca, optimize = einsum_type).copy()
    S12_a_a = S12_a_a.reshape(-1)

    # Check this indices.
    S22_aa_aa =- 1/6 * einsum('WXYZ->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_aa += 1/6 * einsum('WXZY->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_aa += 1/2 * einsum('WY,XZ->XYZW', np.identity(ncas), rdm_ca, optimize = einsum_type)
    S22_aa_aa = S22_aa_aa.reshape(S22_dim, S22_dim)

    S22_aa_bb  = 1/6 * einsum('WXYZ->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_bb += 1/3 * einsum('WXZY->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_bb = S22_aa_bb.reshape(S22_dim, S22_dim)

    # Building S_0p tensor
    S_0p[0,0] = 1.0

    S_0p[0,1:S12_dim] = S12_a_a.copy()
    S_0p[0,S12_dim:]  = S12_a_a.copy()
    S_0p[1:S12_dim,0] = S12_a_a.T.copy()
    S_0p[S12_dim:,0]  = S12_a_a.T.copy()

    S_0p[1:S12_dim,1:S12_dim] = S22_aa_aa.copy()
    S_0p[S12_dim:,S12_dim:]   = S22_aa_aa.copy()

    S_0p[1:S12_dim,S12_dim:] = S22_aa_bb.T.copy()
    S_0p[S12_dim:,1:S12_dim] = S22_aa_bb.copy()

    # Compute projector to the GNO operator basis
    Y = np.identity(S_0p.shape[0])
    Y[0,1:(ncas * ncas + 1)] =- 0.5 * rdm_ca.reshape(-1).copy()
    Y[0,(ncas * ncas + 1):] =- 0.5 * rdm_ca.reshape(-1).copy()

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
