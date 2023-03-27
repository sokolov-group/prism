# Copyright 2023 Prism Developers. All Rights Reserved.
#
# Licensed under the GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.
#
# See the License file for the specific language governing
# permissions and limitations.
#
# Available at https://github.com/sokolov-group/prism
#
# Authors: Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#          Carlos E. V. de Moura <carlosevmoura@gmail.com>
#

import numpy as np
from functools import reduce

def compute_S12_p1(mr_adc, ignore_print = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    ## Reduced density matrices
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

    if (not ignore_print) or (mr_adc.print_level > 4):
        print("Dimension of the [+1] orthonormalized subspace:    %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print("Smallest eigenvalue of the [+1] overlap metric:    %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_p1_12_inv

def compute_S12_m1(mr_adc, ignore_print = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ## Reduced density matrices
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

    if (not ignore_print) or (mr_adc.print_level > 4):
        print("Dimension of the [-1] orthonormalized subspace:    %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print("Smallest eigenvalue of the [-1] overlap metric:    %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_m1_12_inv

def compute_S12_p2(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    s_thresh = mr_adc.s_thresh_doubles

    # Compute S matrix: < Psi_0 | a_X a_Y a^{\dag}_Z a^{\dag}_W | Psi_0 >
    S_p2  = 1/3 * einsum('WZXY->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S_p2 += 1/6 * einsum('WZYX->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S_p2 += einsum('WX,YZ->XYWZ', np.identity(ncas), np.identity(ncas), optimize = einsum_type)
    S_p2 -= 1/2 * einsum('WX,ZY->XYWZ', np.identity(ncas), rdm_ca, optimize = einsum_type)
    S_p2 -= 1/2 * einsum('YZ,WX->XYWZ', np.identity(ncas), rdm_ca, optimize = einsum_type)
    S_p2 = S_p2.reshape(ncas**2, ncas**2)

    # Compute S^{-1/2} matrix
    S_eval, S_evec = np.linalg.eigh(S_p2)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])
    S_evec = S_evec[:, S_ind_nonzero]

    S_p2_12_inv = np.dot(S_evec, np.diag(S_inv_eval))

    print("Dimension of the [+2] orthonormalized subspace:    %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        print("Smallest eigenvalue of the [+2] overlap metric:    %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_p2_12_inv

def compute_S12_m2(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    ## Reduced density matrices
    rdm_ccaa = mr_adc.rdm.ccaa

    # Compute S matrix: < Psi_0 | a^{\dag}_X a^{\dag}_Y a_Z a_W | Psi_0 >
    S_m2  = 1/3 * einsum('WZXY->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S_m2 += 1/6 * einsum('WZYX->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S_m2 = S_m2.reshape(ncas**2, ncas**2)

    s_thresh = mr_adc.s_thresh_doubles
    S_eval, S_evec = np.linalg.eigh(S_m2)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])
    S_evec = S_evec[:, S_ind_nonzero]

    S_m2_12_inv = np.dot(S_evec, np.diag(S_inv_eval))

    print("Dimension of the [-2] orthonormalized subspace:    %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        print("Smallest eigenvalue of the [-2] overlap metric:    %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_m2_12_inv

def compute_S12_0p_projector(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas
    nelecas = sum(mr_adc.nelecas)

    if nelecas == 0:
        nelecas = 1

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Damping for singles
    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_damping = mr_adc.s_damping_strength
        s_damping_thresh = mr_adc.s_thresh_singles
        s_thresh = s_damping_thresh * 10**(-s_damping / 2)

    # Compute S matrix
    ## S22 block: < Psi_0 | a^{\dag}_X a_Y a^{\dag}_Z a_W | Psi_0 >
    S22_aa_aa =- 1/6 * einsum('WYXZ->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_aa += 1/6 * einsum('WYZX->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_aa += 1/2 * einsum('YZ,XW->XYWZ', np.identity(ncas), rdm_ca, optimize = einsum_type)

    S22_aa_bb  = 1/6 * einsum('WYXZ->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_bb += 1/3 * einsum('WYZX->XYWZ', rdm_ccaa, optimize = einsum_type).copy()

    S22_ba_ba =- 1/3 * einsum('WYXZ->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_ba_ba -= 1/6 * einsum('WYZX->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_ba_ba += 1/2 * einsum('YZ,XW->XYWZ', np.identity(ncas), rdm_ca, optimize = einsum_type)

    ## Reshape tensors to matrix form
    dim_wz = ncas * ncas
    dim_S22 = 3 * dim_wz

    S22_aa_aa = S22_aa_aa.reshape(dim_wz, dim_wz)
    S22_aa_bb = S22_aa_bb.reshape(dim_wz, dim_wz)
    S22_ba_ba = S22_ba_ba.reshape(dim_wz, dim_wz)

    # Building S_0p matrix
    s_aa = 0
    f_aa = s_aa + dim_wz
    s_bb = f_aa
    f_bb = s_bb + dim_wz
    s_ba = f_bb
    f_ba = s_ba + dim_wz

    S22 = np.zeros((dim_S22, dim_S22))

    S22[s_aa:f_aa, s_aa:f_aa] = S22_aa_aa.copy()
    S22[s_bb:f_bb, s_bb:f_bb] = S22_aa_aa.copy()

    S22[s_aa:f_aa, s_bb:f_bb] = S22_aa_bb.copy()
    S22[s_bb:f_bb, s_aa:f_aa] = S22_aa_bb.T.copy()

    S22[s_ba:f_ba, s_ba:f_ba] = S22_ba_ba.copy()

    # Compute projector to remove linear dependencies
    S22_aa_aa = S22_aa_aa.reshape(ncas, ncas, ncas, ncas)
    S22_aa_bb = S22_aa_bb.reshape(ncas, ncas, ncas, ncas)

    Q_aa_aa  = np.einsum("XW,YZ->XYWZ", np.identity(ncas), np.identity(ncas)).copy()
    Q_aa_aa -= 2.0 * np.einsum("XY,uuWZ->XYWZ", np.identity(ncas), S22_aa_aa) / (nelecas ** 2)

    Q_aa_bb =- 2.0 * np.einsum("XY,uuWZ->XYWZ", np.identity(ncas), S22_aa_bb) / (nelecas ** 2)
    Q_bb_aa =- 2.0 * np.einsum("XY,uuWZ->XYWZ", np.identity(ncas), S22_aa_bb.T) / (nelecas ** 2)

    Q_aa_aa = Q_aa_aa.reshape(dim_wz, dim_wz)
    Q_aa_bb = Q_aa_bb.reshape(dim_wz, dim_wz)
    Q_bb_aa = Q_bb_aa.reshape(dim_wz, dim_wz)

    ## Building Q matrix
    Q = np.identity(dim_S22)

    Q[s_aa:f_aa, s_aa:f_aa] = Q_aa_aa.copy()
    Q[s_bb:f_bb, s_bb:f_bb] = Q_aa_aa.copy()

    Q[s_aa:f_aa, s_bb:f_bb] = Q_aa_bb.copy()
    Q[s_bb:f_bb, s_aa:f_aa] = Q_bb_aa.copy()

    # Compute S^{-1/2} matrix
    S22 = np.dot(S22, Q)

    S_eval, S_evec = np.linalg.eigh(S22)

    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    S_evec = S_evec[:, S_ind_nonzero]

    S22_12_inv = reduce(np.dot, (Q, S_evec, np.diag(S_inv_eval)))

    S_0p_12_inv = np.zeros((1 + dim_S22, 1 + S22_12_inv.shape[1]))
    S_0p_12_inv[0,0] = 1.0
    S_0p_12_inv[1:,1:] = S22_12_inv.copy()

    if mr_adc.print_level > 4:
        print("Dimension of the [0'] orthonormalized subspace:    %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print("Smallest eigenvalue of the [0'] overlap metric:    %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_0p_12_inv

def compute_S12_0p_gno_projector(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Damping for singles
    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_damping = mr_adc.s_damping_strength
        s_damping_thresh = mr_adc.s_thresh_singles
        s_thresh = s_damping_thresh * 10**(-s_damping / 2)

    # Compute S matrix
    ## S11 block: < Psi_0 | a^{\dag}_X a_Y | Psi_0 >
    S12_a_a = 1/2 * einsum('XY->XY', rdm_ca, optimize = einsum_type).copy()

    ## S22 block: < Psi_0 | a^{\dag}_X a_Y a^{\dag}_Z a_W | Psi_0 >
    S22_aa_aa =- 1/6 * einsum('WYXZ->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_aa += 1/6 * einsum('WYZX->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_aa += 1/2 * einsum('YZ,XW->XYWZ', np.identity(ncas), rdm_ca, optimize = einsum_type)

    S22_aa_bb  = 1/6 * einsum('WYXZ->XYWZ', rdm_ccaa, optimize = einsum_type).copy()
    S22_aa_bb += 1/3 * einsum('WYZX->XYWZ', rdm_ccaa, optimize = einsum_type).copy()

    ## Reshape tensors to matrix form
    dim_wz = ncas * ncas
    dim_S_0p = 1 + 2 * dim_wz

    S12_a_a = S12_a_a.reshape(-1)

    S22_aa_aa = S22_aa_aa.reshape(dim_wz, dim_wz)
    S22_aa_bb = S22_aa_bb.reshape(dim_wz, dim_wz)

    # Building S_0p matrix
    s_aa = 1
    f_aa = s_aa + dim_wz
    s_bb = f_aa
    f_bb = s_bb + dim_wz

    S_0p = np.zeros((dim_S_0p, dim_S_0p))

    S_0p[0,0] = 1.0

    S_0p[0, s_aa:f_aa] = S12_a_a.copy()
    S_0p[0, s_bb:f_bb] = S12_a_a.copy()
    S_0p[s_aa:f_aa, 0] = S12_a_a.T.copy()
    S_0p[s_bb:f_bb, 0] = S12_a_a.T.copy()

    S_0p[s_aa:f_aa, s_aa:f_aa] = S22_aa_aa.copy()
    S_0p[s_bb:f_bb, s_bb:f_bb] = S22_aa_aa.copy()

    S_0p[s_aa:f_aa, s_bb:f_bb] = S22_aa_bb.copy()
    S_0p[s_bb:f_bb, s_aa:f_aa] = S22_aa_bb.T.copy()

    # Compute projector to the GNO operator basis
    Y = np.identity(S_0p.shape[0])

    Y[0, s_aa:f_aa] =- 0.5 * rdm_ca.reshape(-1).copy()
    Y[0, s_bb:f_bb] =- 0.5 * rdm_ca.reshape(-1).copy()

    # Compute S^{-1/2} matrix
    St_0p = reduce(np.dot, (Y.T, S_0p, Y))

    S_eval, S_evec = np.linalg.eigh(St_0p)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    ## Apply damping
    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], s_damping_thresh, s_damping)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_0p_12_inv = reduce(np.dot, (Y, S_evec, np.diag(S_inv_eval)))

    print("Dimension of the [0'] orthonormalized subspace:    %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        print("Smallest eigenvalue of the [0'] overlap metric:    %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_0p_12_inv

def compute_S12_p1p_gno_projector(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Damping for singles
    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_damping = mr_adc.s_damping_strength
        s_damping_thresh = mr_adc.s_thresh_singles
        s_thresh = s_damping_thresh * 10**(-s_damping / 2)

    # Compute S matrix
    ## S11 block: < Psi_0 | a_X a^{\dag}_Y | Psi_0 >
    S11_a_a  = einsum('XY->XY', np.identity(ncas), optimize = einsum_type).copy()
    S11_a_a -= 1/2 * einsum('YX->XY', rdm_ca, optimize = einsum_type).copy()

    ## S12 block: < Psi_0 | a_X a^{\dag}_W a^{\dag}_Z a_Y | Psi_0 >
    S12_a_bba =- 1/6 * einsum('WXYZ->XWZY', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_bba -= 1/3 * einsum('WXZY->XWZY', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_bba += 1/2 * einsum('XY,ZW->XWZY', np.identity(ncas), rdm_ca, optimize = einsum_type)

    S12_a_aaa = np.ascontiguousarray(S12_a_bba - S12_a_bba.transpose(0,1,3,2))

    ## S22 block: < Psi_0 | a^{\dag}_U a_V a_X a^{\dag}_Y a^{\dag}_Z a_W | Psi_0 >
    S22_aaa_aaa =- 1/12 * einsum('UYZVXW->UVXWZY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa -= 1/12 * einsum('UYZWVX->UVXWZY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa -= 1/12 * einsum('UYZXWV->UVXWZY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa += 1/6 * einsum('VY,UZWX->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa -= 1/6 * einsum('VY,UZXW->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa -= 1/6 * einsum('VZ,UYWX->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa += 1/6 * einsum('VZ,UYXW->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa += 1/6 * einsum('XY,UZVW->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa -= 1/6 * einsum('XY,UZWV->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa -= 1/6 * einsum('XZ,UYVW->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa += 1/6 * einsum('XZ,UYWV->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa -= 1/2 * einsum('VY,XZ,UW->UVXWZY', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    S22_aaa_aaa += 1/2 * einsum('VZ,XY,UW->UVXWZY', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)

    S22_bba_bba =- 1/12 * einsum('UYZVXW->UVXWZY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_bba_bba += 1/12 * einsum('UYZWVX->UVXWZY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_bba_bba += 1/6 * einsum('UYZWXV->UVXWZY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_bba_bba += 1/12 * einsum('UYZXWV->UVXWZY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_bba_bba -= 1/3 * einsum('VZ,UYWX->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_bba_bba -= 1/6 * einsum('VZ,UYXW->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_bba_bba += 1/6 * einsum('XY,UZVW->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_bba_bba -= 1/6 * einsum('XY,UZWV->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_bba_bba += 1/2 * einsum('VZ,XY,UW->UVXWZY', np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)

    S22_aaa_bba = np.ascontiguousarray(S22_aaa_aaa -
                                       S22_bba_bba.transpose(0,2,1,3,5,4) +
                                       S22_bba_bba.transpose(0,1,2,3,5,4))

    S22_bba_aaa = np.ascontiguousarray(S22_aaa_bba.transpose(3,4,5,0,1,2))

    ## Reshape tensors to matrix form
    dim_x = ncas
    dim_wzy = ncas * ncas * ncas
    dim_tril_wzy = ncas * ncas * (ncas - 1) // 2

    dim_act = dim_x + dim_wzy + dim_tril_wzy

    tril_ind = np.tril_indices(ncas, k=-1)

    S12_a_aaa = S12_a_aaa[:, :, tril_ind[0], tril_ind[1]]

    S22_aaa_aaa = S22_aaa_aaa[:, :, :, :, tril_ind[0], tril_ind[1]]
    S22_aaa_aaa = S22_aaa_aaa[:, tril_ind[0], tril_ind[1]]

    S22_aaa_bba = S22_aaa_bba[:, tril_ind[0], tril_ind[1]]
    S22_bba_aaa = S22_bba_aaa[:, :, :, :, tril_ind[0], tril_ind[1]]

    S12_a_aaa = S12_a_aaa.reshape(dim_x, dim_tril_wzy)
    S12_a_bba = S12_a_bba.reshape(dim_x, dim_wzy)

    S22_aaa_aaa = S22_aaa_aaa.reshape(dim_tril_wzy, dim_tril_wzy)
    S22_aaa_bba = S22_aaa_bba.reshape(dim_tril_wzy, dim_wzy)

    S22_bba_aaa = S22_bba_aaa.reshape(dim_wzy, dim_tril_wzy)
    S22_bba_bba = S22_bba_bba.reshape(dim_wzy, dim_wzy)

    # Build S_p1p_act matrix
    s_a = 0
    f_a = dim_x
    s_aaa = f_a
    f_aaa = s_aaa + dim_tril_wzy
    s_bba = f_aaa
    f_bba = s_bba + dim_wzy

    S_p1p_act = np.zeros((dim_act, dim_act))

    S_p1p_act[s_a:f_a, s_a:f_a] = S11_a_a

    S_p1p_act[s_a:f_a, s_aaa:f_aaa] = S12_a_aaa
    S_p1p_act[s_a:f_a, s_bba:f_bba] = S12_a_bba

    S_p1p_act[s_aaa:f_aaa, s_a:f_a] = S12_a_aaa.T
    S_p1p_act[s_bba:f_bba, s_a:f_a] = S12_a_bba.T

    S_p1p_act[s_aaa:f_aaa, s_aaa:f_aaa] = S22_aaa_aaa
    S_p1p_act[s_aaa:f_aaa, s_bba:f_bba] = S22_aaa_bba

    S_p1p_act[s_bba:f_bba, s_aaa:f_aaa] = S22_bba_aaa
    S_p1p_act[s_bba:f_bba, s_bba:f_bba] = S22_bba_bba

    # Compute projector to the GNO operator basis
    Y = np.identity(S_p1p_act.shape[0])

    Y_a_aaa =- 0.5 * np.einsum("XZ,YW->XYWZ", np.identity(ncas), rdm_ca)
    Y_a_aaa += 0.5 * np.einsum("XW,YZ->XYWZ", np.identity(ncas), rdm_ca)

    Y_a_bba =- 0.5 * np.einsum("XZ,YW->XYWZ", np.identity(ncas), rdm_ca)

    Y_a_aaa = Y_a_aaa[:, :, tril_ind[0], tril_ind[1]]

    Y_a_aaa = Y_a_aaa.reshape(dim_x, dim_tril_wzy)
    Y_a_bba = Y_a_bba.reshape(dim_x, dim_wzy)

    Y[s_a:f_a, s_aaa:f_aaa] = Y_a_aaa
    Y[s_a:f_a, s_bba:f_bba] = Y_a_bba

    # Compute S^{-1/2} matrix
    St_p1p = reduce(np.dot, (Y.T, S_p1p_act, Y))

    S_eval, S_evec = np.linalg.eigh(St_p1p)

    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    ## Apply damping
    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], s_damping_thresh, s_damping)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_p1p_12_inv_act = reduce(np.dot, (Y, S_evec, np.diag(S_inv_eval)))

    print("Dimension of the [+1'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        print("Smallest eigenvalue of the [+1'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_p1p_12_inv_act

def compute_S12_m1p_gno_projector(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Damping for singles
    if mr_adc.s_damping_strength is None:
        s_thresh = mr_adc.s_thresh_singles
    else:
        s_damping = mr_adc.s_damping_strength
        s_damping_thresh = mr_adc.s_thresh_singles
        s_thresh = s_damping_thresh * 10**(-s_damping / 2)

    # Compute S matrix
    ## S11 block: < Psi_0 | a^{\dag}_X a_Y | Psi_0 >
    S11_a_a  = 1/2 * einsum('XY->XY', rdm_ca, optimize = einsum_type).copy()

    ## S12 block: < Psi_0 | a^{\dag}_X a^{\dag}_Y a_Z a_W | Psi_0 >
    S12_a_abb  = 1/3 * einsum('WZXY->XWZY', rdm_ccaa, optimize = einsum_type).copy()
    S12_a_abb += 1/6 * einsum('WZYX->XWZY', rdm_ccaa, optimize = einsum_type).copy()

    S12_a_aaa = np.ascontiguousarray(S12_a_abb - S12_a_abb.transpose(0,2,1,3))

    ## S22 block: < Psi_0 | a^{\dag}_U a^{\dag}_V a_X a^{\dag}_W a_Z a_Y | Psi_0 >
    S22_aaa_aaa =- 1/12 * einsum('UVYWZX->UVXWZY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa -= 1/12 * einsum('UVYXWZ->UVXWZY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa -= 1/12 * einsum('UVYZXW->UVXWZY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_aaa_aaa += 1/6 * einsum('XY,UVWZ->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_aaa_aaa -= 1/6 * einsum('XY,UVZW->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    S22_abb_abb  = 1/6 * einsum('UVYWXZ->UVXWZY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_abb_abb -= 1/12 * einsum('UVYWZX->UVXWZY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_abb_abb += 1/12 * einsum('UVYXWZ->UVXWZY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_abb_abb += 1/12 * einsum('UVYZXW->UVXWZY', rdm_cccaaa, optimize = einsum_type).copy()
    S22_abb_abb += 1/3 * einsum('XY,UVWZ->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    S22_abb_abb += 1/6 * einsum('XY,UVZW->UVXWZY', np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    S22_aaa_abb = np.ascontiguousarray(S22_aaa_aaa -
                                       S22_abb_abb.transpose(1,0,2,4,3,5) +
                                       S22_abb_abb.transpose(0,1,2,4,3,5))

    S22_abb_aaa = np.ascontiguousarray(S22_aaa_abb.transpose(3,4,5,0,1,2))

    ## Reshape tensors to matrix form
    dim_x = ncas
    dim_wzy = ncas * ncas * ncas
    dim_tril_wzy = ncas * ncas * (ncas - 1) // 2

    dim_act = dim_x + dim_wzy + dim_tril_wzy

    tril_ind = np.tril_indices(ncas, k=-1)

    S12_a_aaa = S12_a_aaa[:, tril_ind[0], tril_ind[1]]

    S22_aaa_aaa = S22_aaa_aaa[:, :, :, tril_ind[0], tril_ind[1]]
    S22_aaa_aaa = S22_aaa_aaa[tril_ind[0], tril_ind[1]]

    S22_aaa_abb = S22_aaa_abb[tril_ind[0], tril_ind[1]]
    S22_abb_aaa = S22_abb_aaa[:, :, :, tril_ind[0], tril_ind[1]]

    S12_a_aaa = S12_a_aaa.reshape(dim_x, dim_tril_wzy)
    S12_a_abb = S12_a_abb.reshape(dim_x, dim_wzy)

    S22_aaa_aaa = S22_aaa_aaa.reshape(dim_tril_wzy, dim_tril_wzy)
    S22_aaa_abb = S22_aaa_abb.reshape(dim_tril_wzy, dim_wzy)

    S22_abb_aaa = S22_abb_aaa.reshape(dim_wzy, dim_tril_wzy)
    S22_abb_abb = S22_abb_abb.reshape(dim_wzy, dim_wzy)

    # Build S_p1p_act matrix
    s_a = 0
    f_a = dim_x
    s_aaa = f_a
    f_aaa = s_aaa + dim_tril_wzy
    s_abb = f_aaa
    f_abb = s_abb + dim_wzy

    S_m1p_act = np.zeros((dim_act, dim_act))

    S_m1p_act[s_a:f_a, s_a:f_a] = S11_a_a

    S_m1p_act[s_a:f_a, s_aaa:f_aaa] = S12_a_aaa
    S_m1p_act[s_a:f_a, s_abb:f_abb] = S12_a_abb

    S_m1p_act[s_aaa:f_aaa, s_a:f_a] = S12_a_aaa.T
    S_m1p_act[s_abb:f_abb, s_a:f_a] = S12_a_abb.T

    S_m1p_act[s_aaa:f_aaa, s_aaa:f_aaa] = S22_aaa_aaa
    S_m1p_act[s_aaa:f_aaa, s_abb:f_abb] = S22_aaa_abb

    S_m1p_act[s_abb:f_abb, s_aaa:f_aaa] = S22_abb_aaa
    S_m1p_act[s_abb:f_abb, s_abb:f_abb] = S22_abb_abb

    # Compute projector to the GNO operator basis
    Y = np.identity(S_m1p_act.shape[0])

    Y_a_aaa =- 0.5 * np.einsum("XY,ZW->XYZW", np.identity(ncas), rdm_ca)
    Y_a_aaa += 0.5 * np.einsum("XZ,YW->XYZW", np.identity(ncas), rdm_ca)

    Y_a_abb =- 0.5 * np.einsum("XY,ZW->XYZW", np.identity(ncas), rdm_ca)

    Y_a_aaa = Y_a_aaa[:,tril_ind[0],tril_ind[1]]

    Y_a_aaa = Y_a_aaa.reshape(dim_x, dim_tril_wzy)
    Y_a_abb = Y_a_abb.reshape(dim_x, dim_wzy)

    Y[s_a:f_a, s_aaa:f_aaa] = Y_a_aaa
    Y[s_a:f_a, s_abb:f_abb] = Y_a_abb

    # Compute S^{-1/2} matrix
    St_m1p = reduce(np.dot, (Y.T, S_m1p_act, Y))

    S_eval, S_evec = np.linalg.eigh(St_m1p)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    ## Apply damping
    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], s_damping_thresh, s_damping)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_m1p_12_inv_act = reduce(np.dot, (Y, S_evec, np.diag(S_inv_eval)))

    print("Dimension of the [-1'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        print("Smallest eigenvalue of the [-1'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_m1p_12_inv_act

# Spin-Orbital Sanity Check Functions
def compute_S12_0p_sanity_check_gno_projector(mr_adc):

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
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], mr_adc.s_damping_strength, mr_adc.s_thresh_singles)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_0p_12_inv = reduce(np.dot, (Y, S_evec, np.diag(S_inv_eval)))

    print("Dimension of the [0'] orthonormalized subspace:    %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        print("Smallest eigenvalue of the [0'] overlap metric:    %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_0p_12_inv

def compute_S12_p1p_sanity_check_gno_projector(mr_adc):

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

    Y_ten[1::2,::2,::2,1::2] = 0.0
    Y_ten[1::2,::2,1::2,::2] = 0.0
    Y_ten[1::2,1::2,1::2,1::2] = 0.0

    Y[:n_x,n_x:] = Y_ten[:,:,xy_ind[0],xy_ind[1]].reshape(n_x, n_xzw)

    St = reduce(np.dot, (Y.T, S_act, Y))

    S_eval, S_evec = np.linalg.eigh(St)

    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])

    if mr_adc.s_damping_strength is not None:
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], mr_adc.s_damping_strength, mr_adc.s_thresh_singles)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_p1p_12_inv_act = reduce(np.dot, (Y, S_evec, np.diag(S_inv_eval)))

    print("Dimension of the [+1'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        print("Smallest eigenvalue of the [+1'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_p1p_12_inv_act

def compute_S12_m1p_sanity_check_gno_projector(mr_adc):

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
        damping_prefactor = compute_damping(S_eval[S_ind_nonzero], mr_adc.s_damping_strength, mr_adc.s_thresh_singles)
        S_inv_eval *= damping_prefactor

    S_evec = S_evec[:, S_ind_nonzero]

    S_m1p_12_inv_act = reduce(np.dot, (Y, S_evec, np.diag(S_inv_eval)))

    print("Dimension of the [-1'] orthonormalized subspace:    %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        print("Smallest eigenvalue of the [-1'] overlap metric:    %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_m1p_12_inv_act

# Damping Function
def compute_damping(s_evals, damping_center, damping_strength):
    'Calculate logarithmic sigmoid damping prefactors for overlap eigenvalues in the range (damping_max, damping_min)'

    def sigmoid(x, shift, scale):
        x = np.log10(x)
        return 1 / (1 + np.exp(scale * (shift - x)))

    center = np.log10(damping_center)
    scale_sigmoid = 10 / damping_strength

    return sigmoid(s_evals, center, scale_sigmoid)
