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
        print ("Dimension of the [+1] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print ("Smallest eigenvalue of the [+1] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

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
        print ("Dimension of the [-1] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print ("Smallest eigenvalue of the [-1] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

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
        print ("Dimension of the [+2] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print ("Smallest eigenvalue of the [+2] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

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
        print ("Dimension of the [-2] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print ("Smallest eigenvalue of the [-2] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_m2_12_inv

def compute_S12_p1p(mr_adc, ignore_print = True, half_transform = False, s_thresh = None, conv_order = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    ncas = mr_adc.ncas

    if s_thresh is None:
        if mr_adc.s_damping_strength is None:
            s_thresh = mr_adc.s_thresh_singles
        else:
            s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    n_x = ncas
    n_zwy = ncas**3
    dim_act = n_x + n_zwy

    S_act = np.zeros((dim_act, dim_act))

    S11  = 2 * einsum('XY->XY', np.identity(ncas), optimize = einsum_type).copy()
    S11 -= einsum('YX->XY', rdm_ca, optimize = einsum_type).copy()
    # if mr_adc.debug_mode:
    #     print (">>> SA S11 norm: {:}".format(np.linalg.norm(S11)))
    #     print (">>> SA S11 trace: {:}".format(np.einsum('ii', S11)))
    #     with open('SA_S11.out', 'w') as outfile:
    #         outfile.write(repr(S11))

    S12 =- einsum('WYZX->XZWY', rdm_ccaa, optimize = einsum_type).copy()
    S12 -= einsum('WX,YZ->XZWY', np.identity(ncas), rdm_ca, optimize = einsum_type)
    S12 += 2 * einsum('XY,WZ->XZWY', np.identity(ncas), rdm_ca, optimize = einsum_type)
    # if mr_adc.debug_mode:
    #     print (">>> SA S12 norm: {:}".format(np.linalg.norm(S12)))
    #     print (">>> SA S12 trace: {:}".format(np.einsum('iiii', S12)))
    #     with open('SA_S12.out', 'w') as outfile:
    #         outfile.write(repr(S12))

    S22  = 1/3 * einsum('UWYVXZ->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22 -= 2/3 * einsum('UWYVZX->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22 += 1/3 * einsum('UWYXVZ->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22 += 1/3 * einsum('UWYXZV->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22 += 1/3 * einsum('UWYZVX->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22 += 1/3 * einsum('UWYZXV->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22 -= einsum('VW,UYZX->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22 -= einsum('VY,UWXZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22 -= einsum('WX,UYVZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22 += 2 * einsum('XY,UWVZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22 += 2 * einsum('VW,XY,UZ->UVXZWY', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    S22 -= einsum('VY,WX,UZ->UVXZWY', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)

    S_act[:n_x,:n_x] = S11.copy()
    S_act[:n_x,n_x:] = S12.reshape(n_x, n_zwy)
    S_act[n_x:,:n_x] = S12.reshape(n_x, n_zwy).T
    S_act[n_x:,n_x:] = S22.reshape(n_zwy, n_zwy)
    # if mr_adc.debug_mode:
    #     with open('SA_S_act.out', 'w') as outfile:
    #         outfile.write(repr(S_act))

    # print(">>> SA S11 is Hermitian?: {:}".format(np.allclose(S11, np.asmatrix(S11).H, rtol=1e-05, atol=1e-08)))
    # print(">>> SA S22 is Hermitian?: {:}".format(np.allclose(S22.reshape(ncas**3, ncas**3), np.asmatrix(S22.reshape(ncas**3, ncas**3)).H, rtol=1e-05, atol=1e-08)))
    # print(">>> SA S_act is Hermitian?: {:}".format(np.allclose(S_act, np.asmatrix(S_act).H, rtol=1e-05, atol=1e-08)))

    # print (">>> SA S11 norm: {:}".format(np.linalg.norm(S11)))
    # print (">>> SA S12 norm: {:}".format(np.linalg.norm(S12)))
    # print (">>> SA S22 norm: {:}".format(np.linalg.norm(S22)))

    S_eval, S_evec = np.linalg.eigh(S_act)

    # if mr_adc.debug_mode:
    #     with open('SA_S_evec.out', 'w') as outfile:
    #         outfile.write(repr(S_evec))
    #     with open('SA_S_eval.out', 'w') as outfile:
    #         outfile.write(repr(S_eval))

    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], mr_adc.s_thresh_singles, mr_adc.s_damping_strength)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    if half_transform:
        S_p1p_12_inv_act = np.dot(S_evec, np.diag(S_inv_eval))
    else:
        S_p1p_12_inv_act = reduce(np.dot, (S_evec, np.diag(S_inv_eval), S_evec.T))

    if not ignore_print:
        print ("Dimension of the [+1'] orthonormalized subspace:  %d" % (S_inv_eval.shape[0]))
        if len(S_ind_nonzero) > 0:
            print ("Smallest eigenvalue of the [+1'] overlap metric:  %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_p1p_12_inv_act

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

    S12_a_aaa  = 1/6 * einsum('WYXZ->XZWY', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_aaa -= 1/6 * einsum('WYZX->XZWY', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_aaa -= 1/2 * einsum('WX,YZ->XZWY', np.identity(ncas), rdm_ca, optimize = einsum_type)
    S12_a_aaa += 1/2 * einsum('XY,WZ->XZWY', np.identity(ncas), rdm_ca, optimize = einsum_type)

    S12_a_bba =- 1/6 * einsum('WYXZ->XZWY', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_bba -= 1/3 * einsum('WYZX->XZWY', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_bba += 1/2 * einsum('XY,WZ->XZWY', np.identity(ncas), rdm_ca, optimize = einsum_type)

    S12_a_bab  = 1/3 * einsum('WYXZ->XZWY', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_bab += 1/6 * einsum('WYZX->XZWY', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_bab -= 1/2 * einsum('WX,YZ->XZWY', np.identity(ncas), rdm_ca, optimize = einsum_type)

    S12[::2,::2,::2,::2] = S12_a_aaa.copy()
    S12[1::2,1::2,1::2,1::2] = S12_a_aaa.copy()

    S12[::2,1::2,1::2,::2] = S12_a_bba.copy()
    S12[1::2,::2,::2,1::2] = S12_a_bba.copy()

    S12[::2,1::2,::2,1::2] = S12_a_bab.copy()
    S12[1::2,::2,1::2,::2] = S12_a_bab.copy()

    S22 = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2, ncas * 2, ncas * 2))

    S22_aaa_aaa =- 1/12 * einsum('UWYVZX->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa -= 1/12 * einsum('UWYXVZ->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa -= 1/12 * einsum('UWYZXV->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa += 1/6 * einsum('VW,UYXZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa -= 1/6 * einsum('VW,UYZX->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa -= 1/6 * einsum('VY,UWXZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa += 1/6 * einsum('VY,UWZX->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa -= 1/6 * einsum('WX,UYVZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa += 1/6 * einsum('WX,UYZV->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa += 1/6 * einsum('XY,UWVZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa -= 1/6 * einsum('XY,UWZV->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa += 1/2 * einsum('VW,XY,UZ->UVXZWY', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    S22_aaa_aaa -= 1/2 * einsum('VY,WX,UZ->UVXZWY', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)

    S22_aaa_bba =- 1/12 * einsum('UWYVZX->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_bba += 1/12 * einsum('UWYXVZ->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_bba += 1/6 * einsum('UWYXZV->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_bba += 1/12 * einsum('UWYZXV->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_bba -= 1/3 * einsum('VY,UWXZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_bba -= 1/6 * einsum('VY,UWZX->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_bba += 1/3 * einsum('XY,UWVZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_bba += 1/6 * einsum('XY,UWZV->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)

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

    S22_bba_aaa  = 1/6 * einsum('UWYVXZ->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_bba_aaa -= 1/12 * einsum('UWYVZX->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_bba_aaa += 1/12 * einsum('UWYXVZ->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_bba_aaa += 1/12 * einsum('UWYZXV->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_bba_aaa -= 1/3 * einsum('WX,UYVZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_bba_aaa -= 1/6 * einsum('WX,UYZV->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_bba_aaa += 1/3 * einsum('XY,UWVZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_bba_aaa += 1/6 * einsum('XY,UWZV->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    S22[::2,::2,::2,::2,::2,::2] = S22_aaa_aaa.copy()
    S22[1::2,1::2,1::2,1::2,1::2,1::2] = S22_aaa_aaa.copy()

    S22[::2,::2,::2,1::2,1::2,::2] = S22_aaa_bba.copy()
    S22[1::2,1::2,1::2,::2,::2,1::2] = S22_aaa_bba.copy()

    S22[::2,::2,::2,1::2,::2,1::2] = - S22_aaa_bba.transpose(0,1,2,3,5,4).copy()
    S22[1::2,1::2,1::2,::2,1::2,::2] = - S22_aaa_bba.transpose(0,1,2,3,5,4).copy()

    S22[::2,::2,1::2,::2,::2,1::2] = S22_aab_aab.copy()
    S22[1::2,1::2,::2,1::2,1::2,::2] = S22_aab_aab.copy()

    S22[::2,1::2,::2,::2,1::2,::2] = S22_aab_aab.transpose(0,2,1,3,5,4).copy()
    S22[1::2,::2,1::2,1::2,::2,1::2] = S22_aab_aab.transpose(0,2,1,3,5,4).copy()

    S22[::2,1::2,::2,::2,::2,1::2] = - S22_aab_aab.transpose(0,2,1,3,4,5).copy()
    S22[1::2,::2,1::2,1::2,1::2,::2] = - S22_aab_aab.transpose(0,2,1,3,4,5).copy()

    S22[::2,::2,1::2,::2,1::2,::2] = - S22_aab_aab.transpose(0,1,2,3,5,4).copy()
    S22[1::2,1::2,::2,1::2,::2,1::2] = - S22_aab_aab.transpose(0,1,2,3,5,4).copy()

    S22[1::2,1::2,::2,::2,::2,::2] = S22_bba_aaa.copy()
    S22[::2,::2,1::2,1::2,1::2,1::2] = S22_bba_aaa.copy()

    S22[1::2,::2,1::2,::2,::2,::2] = - S22_bba_aaa.transpose(0,2,1,3,4,5).copy()
    S22[::2,1::2,::2,1::2,1::2,1::2] = - S22_bba_aaa.transpose(0,2,1,3,4,5).copy()

    S22[1::2,::2,::2,1::2,::2,::2] = S22_baa_baa.copy()
    S22[::2,1::2,1::2,::2,1::2,1::2] = S22_baa_baa.copy()

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
        print ("Dimension of the [+1'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print ("Smallest eigenvalue of the [+1'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_p1p_12_inv_act

def compute_S12_p1p_singles(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca

    ncas = mr_adc.ncas

    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    S11  = 2 * einsum('XY->XY', np.identity(ncas), optimize = einsum_type).copy()
    S11 -= einsum('YX->XY', rdm_ca, optimize = einsum_type).copy()

    S_eval, S_evec = np.linalg.eigh(S11)

    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], mr_adc.s_thresh_singles, mr_adc.s_damping_strength)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_p1p_inv_act = np.dot(S_evec, np.diag(S_inv_eval))

    if not ignore_print:
        print ("Dimension of the [+1'] orthonormalized subspace:  %d" % (S_inv_eval.shape[0]))
        if len(S_ind_nonzero) > 0:
            print ("Smallest eigenvalue of the [+1'] overlap metric:  %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_p1p_inv_act

def compute_S12_p1p_singles_sanity_check_gno_projector(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca

    ncas = mr_adc.ncas

    S11 = np.zeros((ncas * 2, ncas * 2))

    S11_a_a  = einsum('XY->XY', np.identity(ncas), optimize = einsum_type).copy()
    S11_a_a -= 1/2 * einsum('YX->XY', rdm_ca, optimize = einsum_type).copy()

    S11[::2,::2] = S11_a_a.copy()
    S11[1::2,1::2] = S11_a_a.copy()

    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    S_eval, S_evec = np.linalg.eigh(S11)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], mr_adc.s_thresh_singles, mr_adc.s_damping_strength)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_p1p_12_inv_act = reduce(np.dot, (S_evec, np.diag(S_inv_eval)))

    if not ignore_print:
        print ("Dimension of the [+1'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print ("Smallest eigenvalue of the [+1'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_p1p_12_inv_act

def compute_S12_p1p_definition(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    ncas = mr_adc.ncas

    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_thresh = mr_adc.s_thresh_singles * 10**(-mr_adc.s_damping_strength / 2)

    S22  = 1/3 * einsum('UWYVXZ->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22 -= 2/3 * einsum('UWYVZX->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22 += 1/3 * einsum('UWYXVZ->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22 += 1/3 * einsum('UWYXZV->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22 += 1/3 * einsum('UWYZVX->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22 += 1/3 * einsum('UWYZXV->UVXZWY', rdm_cccaaa, optimize = einsum_type).copy()
    S22 -= einsum('VW,UYZX->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22 -= einsum('VY,UWXZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22 -= einsum('WX,UYVZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22 += 2 * einsum('XY,UWVZ->UVXZWY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22 += 2 * einsum('VW,XY,UZ->UVXZWY', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    S22 -= einsum('VY,WX,UZ->UVXZWY', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)

    S_act = S22.reshape(ncas**3, ncas**3)

    S_eval, S_evec = np.linalg.eigh(S_act)

    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/S_eval[S_ind_nonzero]

    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], mr_adc.s_thresh_singles, mr_adc.s_damping_strength)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_p1p_inv_act = np.dot(S_evec, np.diag(S_inv_eval))

    if not ignore_print:
        print ("Dimension of the [+1'] orthonormalized subspace:  %d" % (S_inv_eval.shape[0]))
        if len(S_ind_nonzero) > 0:
            print ("Smallest eigenvalue of the [+1'] overlap metric:  %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_p1p_inv_act

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

    S12_a_aaa =- 1/6 * einsum('WXYZ->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_aaa += 1/6 * einsum('WXZY->XYZW', rdm_ccaa, optimize = einsum_type).copy()

    S12_a_abb  = 1/6 * einsum('WXYZ->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_abb += 1/3 * einsum('WXZY->XYZW', rdm_ccaa, optimize = einsum_type).copy()

    S12_a_bab =- 1/3 * einsum('WXYZ->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_bab -= 1/6 * einsum('WXZY->XYZW', rdm_ccaa, optimize = einsum_type).copy()

    S12[::2,::2,::2,::2] = S12_a_aaa.copy()
    S12[1::2,1::2,1::2,1::2] = S12_a_aaa.copy()

    S12[::2,::2,1::2,1::2] = S12_a_abb.copy()
    S12[1::2,1::2,::2,::2] = S12_a_abb.copy()

    S12[::2,1::2,::2,1::2] = S12_a_bab.copy()
    S12[1::2,::2,1::2,::2] = S12_a_bab.copy()

    S22 = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2, ncas * 2, ncas * 2))

    S22_aaa_aaa  = 1/12 * einsum('UWXVZY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa += 1/12 * einsum('UWXYVZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa += 1/12 * einsum('UWXZYV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa -= 1/6 * einsum('VW,UXYZ->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa += 1/6 * einsum('VW,UXZY->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    S22_aaa_abb  = 1/12 * einsum('UWXVZY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_abb -= 1/12 * einsum('UWXYVZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_abb -= 1/6 * einsum('UWXYZV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_abb -= 1/12 * einsum('UWXZYV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()

    S22_aab_aab =- 1/12 * einsum('UWXVZY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab += 1/12 * einsum('UWXYVZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab -= 1/6 * einsum('UWXZVY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab -= 1/12 * einsum('UWXZYV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab -= 1/6 * einsum('VW,UXYZ->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aab_aab += 1/6 * einsum('VW,UXZY->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    S22_abb_aaa =- 1/6 * einsum('UWXVYZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_abb_aaa += 1/12 * einsum('UWXVZY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_abb_aaa -= 1/12 * einsum('UWXYVZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_abb_aaa -= 1/12 * einsum('UWXZYV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()

    S22_baa_baa  = 1/12 * einsum('UWXVZY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa -= 1/12 * einsum('UWXYVZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa -= 1/6 * einsum('UWXZVY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa -= 1/12 * einsum('UWXZYV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa += 1/6 * einsum('VW,UXYZ->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_baa_baa += 1/3 * einsum('VW,UXZY->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    S22[::2,::2,::2,::2,::2,::2] = S22_aaa_aaa.copy()
    S22[1::2,1::2,1::2,1::2,1::2,1::2] = S22_aaa_aaa.copy()

    S22[::2,::2,::2,::2,1::2,1::2] = S22_aaa_abb.copy()
    S22[1::2,1::2,1::2,1::2,::2,::2] = S22_aaa_abb.copy()

    S22[::2,::2,::2,1::2,::2,1::2] = - S22_aaa_abb.transpose(0,1,2,4,3,5).copy()
    S22[1::2,1::2,1::2,::2,1::2,::2] = - S22_aaa_abb.transpose(0,1,2,4,3,5).copy()

    S22[1::2,::2,::2,1::2,::2,::2] = S22_baa_baa.copy()
    S22[::2,1::2,1::2,::2,1::2,1::2] = S22_baa_baa.copy()

    S22[1::2,::2,::2,::2,1::2,::2] = - S22_baa_baa.transpose(0,1,2,4,3,5).copy()
    S22[::2,1::2,1::2,1::2,::2,1::2] = - S22_baa_baa.transpose(0,1,2,4,3,5).copy()

    S22[::2,1::2,::2,1::2,::2,::2] = - S22_baa_baa.transpose(1,0,2,3,4,5).copy()
    S22[1::2,::2,1::2,::2,1::2,1::2] = - S22_baa_baa.transpose(1,0,2,3,4,5).copy()

    S22[::2,1::2,::2,::2,1::2,::2] = S22_baa_baa.transpose(1,0,2,4,3,5).copy()
    S22[1::2,::2,1::2,1::2,::2,1::2] = S22_baa_baa.transpose(1,0,2,4,3,5).copy()

    S22[::2,1::2,1::2,::2,::2,::2] = S22_abb_aaa.copy()
    S22[1::2,::2,::2,1::2,1::2,1::2] = S22_abb_aaa.copy()

    S22[1::2,::2,1::2,::2,::2,::2] = - S22_abb_aaa.transpose(1,0,2,3,4,5).copy()
    S22[::2,1::2,::2,1::2,1::2,1::2] = - S22_abb_aaa.transpose(1,0,2,3,4,5).copy()

    S22[::2,::2,1::2,::2,::2,1::2] = S22_aab_aab.copy()
    S22[1::2,1::2,::2,1::2,1::2,::2] = S22_aab_aab.copy()

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
        print ("Dimension of the [-1'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print ("Smallest eigenvalue of the [-1'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_m1p_12_inv_act

def compute_S12_m1p_sanity_check(mr_adc, ignore_print = True):

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

    S12_a_aaa =- 1/6 * einsum('WXYZ->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_aaa += 1/6 * einsum('WXZY->XYZW', rdm_ccaa, optimize = einsum_type).copy()

    S12_a_abb  = 1/6 * einsum('WXYZ->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_abb += 1/3 * einsum('WXZY->XYZW', rdm_ccaa, optimize = einsum_type).copy()

    S12_a_bab =- 1/3 * einsum('WXYZ->XYZW', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_bab -= 1/6 * einsum('WXZY->XYZW', rdm_ccaa, optimize = einsum_type).copy()

    S12[::2,::2,::2,::2] = S12_a_aaa.copy()
    S12[1::2,1::2,1::2,1::2] = S12_a_aaa.copy()

    S12[::2,::2,1::2,1::2] = S12_a_abb.copy()
    S12[1::2,1::2,::2,::2] = S12_a_abb.copy()

    S12[::2,1::2,::2,1::2] = S12_a_bab.copy()
    S12[1::2,::2,1::2,::2] = S12_a_bab.copy()

    S22 = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2, ncas * 2, ncas * 2))

    S22_aaa_aaa  = 1/12 * einsum('UWXVZY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa += 1/12 * einsum('UWXYVZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa += 1/12 * einsum('UWXZYV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa -= 1/6 * einsum('VW,UXYZ->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa += 1/6 * einsum('VW,UXZY->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    S22_aaa_abb  = 1/12 * einsum('UWXVZY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_abb -= 1/12 * einsum('UWXYVZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_abb -= 1/6 * einsum('UWXYZV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_abb -= 1/12 * einsum('UWXZYV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()

    S22_aab_aab =- 1/12 * einsum('UWXVZY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab += 1/12 * einsum('UWXYVZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab -= 1/6 * einsum('UWXZVY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab -= 1/12 * einsum('UWXZYV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aab_aab -= 1/6 * einsum('VW,UXYZ->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aab_aab += 1/6 * einsum('VW,UXZY->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    S22_abb_aaa =- 1/6 * einsum('UWXVYZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_abb_aaa += 1/12 * einsum('UWXVZY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_abb_aaa -= 1/12 * einsum('UWXYVZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_abb_aaa -= 1/12 * einsum('UWXZYV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()

    S22_baa_baa  = 1/12 * einsum('UWXVZY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa -= 1/12 * einsum('UWXYVZ->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa -= 1/6 * einsum('UWXZVY->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa -= 1/12 * einsum('UWXZYV->XUVYZW', rdm_cccaaa, optimize = einsum_type).copy()
    S22_baa_baa += 1/6 * einsum('VW,UXYZ->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_baa_baa += 1/3 * einsum('VW,UXZY->XUVYZW', np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    S22[::2,::2,::2,::2,::2,::2] = S22_aaa_aaa.copy()
    S22[1::2,1::2,1::2,1::2,1::2,1::2] = S22_aaa_aaa.copy()

    S22[::2,::2,::2,::2,1::2,1::2] = S22_aaa_abb.copy()
    S22[1::2,1::2,1::2,1::2,::2,::2] = S22_aaa_abb.copy()

    S22[::2,::2,::2,1::2,::2,1::2] = - S22_aaa_abb.transpose(0,1,2,4,3,5).copy()
    S22[1::2,1::2,1::2,::2,1::2,::2] = - S22_aaa_abb.transpose(0,1,2,4,3,5).copy()

    S22[1::2,::2,::2,1::2,::2,::2] = S22_baa_baa.copy()
    S22[::2,1::2,1::2,::2,1::2,1::2] = S22_baa_baa.copy()

    S22[1::2,::2,::2,::2,1::2,::2] = - S22_baa_baa.transpose(0,1,2,4,3,5).copy()
    S22[::2,1::2,1::2,1::2,::2,1::2] = - S22_baa_baa.transpose(0,1,2,4,3,5).copy()

    S22[::2,1::2,::2,1::2,::2,::2] = - S22_baa_baa.transpose(1,0,2,3,4,5).copy()
    S22[1::2,::2,1::2,::2,1::2,1::2] = - S22_baa_baa.transpose(1,0,2,3,4,5).copy()

    S22[::2,1::2,::2,::2,1::2,::2] = S22_baa_baa.transpose(1,0,2,4,3,5).copy()
    S22[1::2,::2,1::2,1::2,::2,1::2] = S22_baa_baa.transpose(1,0,2,4,3,5).copy()

    S22[::2,1::2,1::2,::2,::2,::2] = S22_abb_aaa.copy()
    S22[1::2,::2,::2,1::2,1::2,1::2] = S22_abb_aaa.copy()

    S22[1::2,::2,1::2,::2,::2,::2] = - S22_abb_aaa.transpose(1,0,2,3,4,5).copy()
    S22[::2,1::2,::2,1::2,1::2,1::2] = - S22_abb_aaa.transpose(1,0,2,3,4,5).copy()

    S22[::2,::2,1::2,::2,::2,1::2] = S22_aab_aab.copy()
    S22[1::2,1::2,::2,1::2,1::2,::2] = S22_aab_aab.copy()

    S12 = S12[:,xy_ind[0],xy_ind[1]]
    S22 = S22[:,:,:,xy_ind[0],xy_ind[1]]
    S22 = S22[xy_ind[0],xy_ind[1]]

    S_act[:n_x,:n_x] = S11.copy()
    S_act[:n_x,n_x:] = S12.reshape(n_x, n_xzw)
    S_act[n_x:,:n_x] = S12.reshape(n_x, n_xzw).T
    S_act[n_x:,n_x:] = S22.reshape(n_xzw, n_xzw)

    S_eval, S_evec = np.linalg.eigh(S_act)

    s_cut = 0
    s_trace = np.sum(S_eval)
    s_thresh = mr_adc.s_thresh_singles
    for p in range(S_eval.shape[0]):
        s_sum = np.sum(S_eval[:p])
        if (s_sum / s_trace > s_thresh):
            s_cut = p - 1
            break
    S_inv_eval = 1.0/np.sqrt(S_eval[s_cut:])

    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[s_cut:], mr_adc.s_thresh_singles, mr_adc.s_damping_strength)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, s_cut:]

    S_m1p_12_inv_act = reduce(np.dot, (S_evec, np.diag(S_inv_eval)))

    if not ignore_print:
        print ("Dimension of the [-1'] orthonormalized subspace:  %d" % S_eval[s_cut:].shape[0])
        print ("Smallest eigenvalue of the [-1'] overlap metric:  %e" % np.amin(S_eval[s_cut:]))

    return S_m1p_12_inv_act

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

    S22_aa_aa =- 1/6 * einsum('XZWY->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_aa += 1/6 * einsum('XZYW->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_aa += 1/2 * einsum('YZ,XW->XYWZ', np.identity(ncas), rdm_ca, optimize = einsum_type)

    S22_ab_ab =- 1/3 * einsum('XZWY->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_ab_ab -= 1/6 * einsum('XZYW->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_ab_ab += 1/2 * einsum('YZ,XW->XYWZ', np.identity(ncas), rdm_ca, optimize = einsum_type)

    S22_aa_bb  = 1/6 * einsum('XZWY->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_bb += 1/3 * einsum('XZYW->XYWZ', rdm_ccaa, optimize = einsum_type).copy()

    S22[::2,::2,::2,::2] = S22_aa_aa.copy()
    S22[1::2,1::2,1::2,1::2] = S22_aa_aa.copy()

    S22[::2,1::2,::2,1::2] = S22_ab_ab.copy()
    S22[1::2,::2,1::2,::2] = S22_ab_ab.copy()

    S22[::2,::2,1::2,1::2] = S22_aa_bb.copy()
    S22[1::2,1::2,::2,::2] = S22_aa_bb.copy()

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
        print ("Dimension of the [0'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print ("Smallest eigenvalue of the [0'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_0p_12_inv

# This function calculates logarithmic sigmoid damping prefactors for overlap eigenvalues in the range (damping_max, damping_min)
def compute_damping(s_evals, damping_center, damping_strength):

    def sigmoid(x, shift, scale):
        x = np.log10(x)
        return 1 / (1 + np.exp(scale * (shift - x)))

    center = np.log10(damping_center)
    scale_sigmoid = 10 / damping_strength

    return sigmoid(s_evals, center, scale_sigmoid)