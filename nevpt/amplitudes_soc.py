# Copyright 2026 Prism Developers. All Rights Reserved.
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
# Authors: Carlos E. V. de Moura <carlosevmoura@gmail.com>
#          Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#          James D. Serna <jserna456@gmail.com>
#          Nicholas Y. Chiang <nicholas.yiching.chiang@gmail.com>
#

import numpy as np
from functools import reduce

from prism.nevpt import intermediates
from prism.nevpt import overlap

import prism.lib.logger as logger
import prism.lib.tools as tools


def compute_t1_0p_soc(nevpt, rdms, rdms_1s):

    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.extra("\nComputing T[0']^(1) amplitudes...")

    # Einsum definition from kernel
    einsum = nevpt.interface.einsum
    einsum_type = nevpt.interface.einsum_type

    # Variables from kernel
    ncore = nevpt.ncore - nevpt.nfrozen
    ncas = nevpt.ncas
    nextern = nevpt.nextern

    ## Molecular Orbitals Energies
    e_core = nevpt.mo_energy.c
    e_extern = nevpt.mo_energy.e

    ## One-electron integrals
    h_ce = nevpt.h1eff.ce

    ## Two-electron integrals
    #v_caae = nevpt.v2e.caae
    #v_ceaa = nevpt.v2e.ceaa

    ## Reduced density matrices
    rdm_ca = rdms.ca
    rdm_ccaa = rdms.ccaa

    # Compute K_caca matrix
    K_caca = intermediates.compute_K_caca(nevpt, rdms)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_0p_12_inv_act = overlap.compute_S12_0p_gno_projector(nevpt, rdms)

    # Compute K^{-1} matrix
    SKS = reduce(np.dot, (S_0p_12_inv_act[1:,:].T, K_caca, S_0p_12_inv_act[1:,:]))
    evals, evecs = np.linalg.eigh(SKS)
    del(SKS)

    # Compute R.H.S. of the equation
    ## V1 block: - < Psi_0 | a^{\dag}_I a_A V | Psi_0>
    V1_a_a =- einsum('IA->IA', h_ce, optimize = einsum_type).copy()
    V1_a_a -= einsum('IAxy,yx->IA', v_ceaa, rdm_ca, optimize = einsum_type)
    V1_a_a += 1/2 * einsum('IxyA,xy->IA', v_caae, rdm_ca, optimize = einsum_type)

    ## V2 block: - < Psi_0 | a^{\dag}_I a^{\dag}_X a_Y a_A V | Psi_0>
    V2_aa_aa =- 1/2 * einsum('IA,XY->IAXY', h_ce, rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('IAxY,Xx->IAXY', v_ceaa, rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('IAxy,XyYx->IAXY', v_ceaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('IYxA,Xx->IAXY', v_caae, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('IxyA,XxYy->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('IxyA,XxyY->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)

    V2_aa_bb =- 1/2 * einsum('IA,XY->IAXY', h_ce, rdm_ca, optimize = einsum_type)
    V2_aa_bb -= 1/2 * einsum('IAxY,Xx->IAXY', v_ceaa, rdm_ca, optimize = einsum_type)
    V2_aa_bb -= 1/2 * einsum('IAxy,XyYx->IAXY', v_ceaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_bb += 1/3 * einsum('IxyA,XxYy->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)
    V2_aa_bb += 1/6 * einsum('IxyA,XxyY->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)

    V2_aa_aa = V2_aa_aa.reshape(ncore, nextern, -1)
    V2_aa_bb = V2_aa_bb.reshape(ncore, nextern, -1)

    ## Build V tensor
    dim_XY = ncas * ncas
    dim_act = 2 * dim_XY + 1

    V_aa_aa_i = 1
    V_aa_aa_f = V_aa_aa_i + dim_XY
    V_aa_bb_i = V_aa_aa_f
    V_aa_bb_f = V_aa_bb_i + dim_XY

    V_0p = np.zeros((ncore, nextern, dim_act))

    V_0p[:,:,0] = V1_a_a.copy()
    del(V1_a_a)

    V_0p[:,:,V_aa_aa_i:V_aa_aa_f] = V2_aa_aa.copy()
    V_0p[:,:,V_aa_bb_i:V_aa_bb_f] = V2_aa_bb.copy()
    del(V2_aa_aa, V2_aa_bb)

    ## Compute denominators
    d_ai = (e_extern[:,None] - e_core).reshape(-1)
    d_aip = (d_ai[:,None] + evals).reshape(nextern, ncore, -1)
    d_aip = d_aip**(-1)

    # Compute T[0'] amplitudes
    S_12_V_0p = einsum("iaP,Pm->iam", V_0p, S_0p_12_inv_act, optimize = einsum_type)
    S_12_V_0p = einsum("mp,iam->iap", evecs, S_12_V_0p, optimize = einsum_type)
    S_12_V_0p = einsum("aip,iap->iap", d_aip, S_12_V_0p, optimize = einsum_type)
    S_12_V_0p = einsum("mp,iap->iam", evecs, S_12_V_0p, optimize = einsum_type)
    del(V_0p, d_ai, d_aip, evals, evecs)

    ## Compute T[0'] t1_ce, t1_caea and t1_caae tensors
    t_0p = einsum("iam,Pm->iaP", S_12_V_0p, S_0p_12_inv_act, optimize = einsum_type)
    del(S_12_V_0p, S_0p_12_inv_act)

    ## Build T[0'] tensors
    t1_ce = t_0p[:,:,0].copy()
    t1_caea_aaaa = t_0p[:,:,V_aa_aa_i:V_aa_aa_f].reshape(ncore, nextern, ncas, ncas).transpose(0,2,1,3)
    t1_caea_abab = t_0p[:,:,V_aa_bb_i:V_aa_bb_f].reshape(ncore, nextern, ncas, ncas).transpose(0,2,1,3)

    t1_caea = t1_caea_abab
    t1_caae = (t1_caea_abab - t1_caea_aaaa).transpose(0,1,3,2).copy()
    del(t_0p, t1_caea_aaaa, t1_caea_abab)

    # Compute electronic correlation energy for T[0']
    e_0p  = 2 * einsum('ia,ia', h_ce, t1_ce, optimize = einsum_type)
    e_0p += 2 * einsum('ia,ixay,yx', h_ce, t1_caea, rdm_ca, optimize = einsum_type)
    e_0p -= einsum('ia,ixya,yx', h_ce, t1_caae, rdm_ca, optimize = einsum_type)
    e_0p += 2 * einsum('ia,iaxy,yx', t1_ce, v_ceaa, rdm_ca, optimize = einsum_type)
    e_0p -= einsum('ia,ixya,xy', t1_ce, v_caae, rdm_ca, optimize = einsum_type)
    e_0p += 2 * einsum('ixay,iazw,yzxw', t1_caea, v_ceaa, rdm_ccaa, optimize = einsum_type)
    e_0p += 2 * einsum('ixay,iazy,xz', t1_caea, v_ceaa, rdm_ca, optimize = einsum_type)
    e_0p -= einsum('ixay,izwa,ywxz', t1_caea, v_caae, rdm_ccaa, optimize = einsum_type)
    e_0p -= einsum('ixay,iyza,xz', t1_caea, v_caae, rdm_ca, optimize = einsum_type)
    e_0p -= einsum('ixya,iazw,yzxw', t1_caae, v_ceaa, rdm_ccaa, optimize = einsum_type)
    e_0p -= einsum('ixya,iazy,xz', t1_caae, v_ceaa, rdm_ca, optimize = einsum_type)
    e_0p -= einsum('ixya,izwa,ywzx', t1_caae, v_caae, rdm_ccaa, optimize = einsum_type)
    e_0p += 2 * einsum('ixya,iyza,xz', t1_caae, v_caae, rdm_ca, optimize = einsum_type)

    nevpt.log.extra("Norm of T[0']^(1):                           %20.12f" % (np.linalg.norm(t1_ce) +
                                                                              np.linalg.norm(t1_caea)))
    nevpt.log.info("Correlation energy [0']:                     %20.12f" % e_0p)
    nevpt.log.timer("computing T[0']^(1) amplitudes", *cput0)

    return e_0p, t1_ce, t1_caea, t1_caae

def compute_t1_p1p_soc(nevpt, rdms, rdms_1s):

    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.extra("\nComputing T[+1']^(1) amplitudes...")

    # Einsum definition from kernel
    einsum = nevpt.interface.einsum
    einsum_type = nevpt.interface.einsum_type

    # Variables from kernel
    ncore = nevpt.ncore - nevpt.nfrozen
    ncas = nevpt.ncas

    ## Molecular Orbitals Energies
    e_core = nevpt.mo_energy.c

    ## One-electron integrals
    h_ca = nevpt.h1eff.ca

    ## Two-electron integrals
    v_caaa = nevpt.v2e.caaa

    ## Reduced density matrices
    rdm_ca = rdms.ca
    rdm_ccaa = rdms.ccaa
    rdm_cccaaa = rdms.cccaaa

    # Compute K_p1p matrix
    K_p1p = intermediates.compute_K_p1p(nevpt, rdms)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    if nevpt.semi_internal_projector == "gno":
        S_p1p_12_inv_act = overlap.compute_S12_p1p_gno_projector(nevpt, rdms)
    else:
        S_p1p_12_inv_act = overlap.compute_S12_p1p_gs_projector(nevpt, rdms)

    # Compute K^{-1} matrix
    SKS = reduce(np.dot, (S_p1p_12_inv_act.T, K_p1p, S_p1p_12_inv_act))
    evals, evecs = np.linalg.eigh(SKS)
    del(SKS)

    # Compute R.H.S. of the equation
    ## V1 block: - < Psi_0 | a^{\dag}_I a_X V | Psi_0>
    V1_a_a =- einsum('IX->IX', h_ca, optimize = einsum_type).copy()
    V1_a_a += 1/2 * einsum('Ix,Xx->IX', h_ca, rdm_ca, optimize = einsum_type)
    V1_a_a -= einsum('IXxy,xy->IX', v_caaa, rdm_ca, optimize = einsum_type)
    V1_a_a += 1/2 * einsum('IxyX,yx->IX', v_caaa, rdm_ca, optimize = einsum_type)
    V1_a_a += 1/2 * einsum('Ixyz,Xyxz->IX', v_caaa, rdm_ccaa, optimize = einsum_type)

    ## V2 block: - < Psi_0 | a^{\dag}_I a^{\dag}_X a_Y a_Z V | Psi_0>
    V2_aa_aa  = 1/2 * einsum('IY,XZ->IXYZ', h_ca, rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('IZ,XY->IXYZ', h_ca, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('Ix,XxYZ->IXYZ', h_ca, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('Ix,XxZY->IXYZ', h_ca, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('IYxZ,Xx->IXYZ', v_caaa, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('IYxy,XyZx->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('IZxY,Xx->IXYZ', v_caaa, rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('IZxy,XyYx->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('IxyY,XxZy->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('IxyY,XxyZ->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('IxyZ,XxYy->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('IxyZ,XxyY->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('Ixyz,XxzYZy->IXYZ', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('Ixyz,XxzZYy->IXYZ', v_caaa, rdm_cccaaa, optimize = einsum_type)

    V2_ab_ba =- 1/2 * einsum('IZ,XY->IXYZ', h_ca, rdm_ca, optimize = einsum_type)
    V2_ab_ba += 1/3 * einsum('Ix,XxYZ->IXYZ', h_ca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/6 * einsum('Ix,XxZY->IXYZ', h_ca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/2 * einsum('IZxY,Xx->IXYZ', v_caaa, rdm_ca, optimize = einsum_type)
    V2_ab_ba -= 1/2 * einsum('IZxy,XyYx->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/6 * einsum('IxyY,XxZy->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/3 * einsum('IxyY,XxyZ->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/3 * einsum('IxyZ,XxYy->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/6 * einsum('IxyZ,XxyY->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/3 * einsum('Ixyz,XxzYZy->IXYZ', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba += 1/6 * einsum('Ixyz,XxzZYy->IXYZ', v_caaa, rdm_cccaaa, optimize = einsum_type)

    ## Reshape tensors to matrix form
    tril_ind = np.tril_indices(ncas, k=-1)

    V2_aa_aa = V2_aa_aa[:,:,tril_ind[0],tril_ind[1]]

    V2_aa_aa = V2_aa_aa.reshape(ncore, -1)
    V2_ab_ba = V2_ab_ba.reshape(ncore, -1)

    ## Build V matrix
    dim_X = ncas
    dim_YWZ = ncas * ncas * ncas
    dim_tril_YWZ = ncas * ncas * (ncas - 1) // 2

    dim_act = dim_X + dim_tril_YWZ + dim_YWZ

    V_a_i = 0
    V_a_f = dim_X
    V_aaa_i = V_a_f
    V_aaa_f = V_aaa_i + dim_tril_YWZ
    V_bba_i = V_aaa_f
    V_bba_f = V_bba_i + dim_YWZ

    V_p1p = np.zeros((ncore, dim_act))

    V_p1p[:,V_a_i:V_a_f] = V1_a_a.copy()
    del(V1_a_a)

    V_p1p[:,V_aaa_i:V_aaa_f] = V2_aa_aa.copy()
    V_p1p[:,V_bba_i:V_bba_f] = V2_ab_ba.copy()
    del(V2_aa_aa, V2_ab_ba)

    ## Compute denominators
    d_ip = (-e_core[:,None] + evals)
    d_ip = d_ip**(-1)

    # Compute T[+1'] amplitudes
    S_12_V_p1p = einsum("iP,Pm->im", V_p1p, S_p1p_12_inv_act, optimize = einsum_type)
    S_12_V_p1p = einsum("mp,im->ip", evecs, S_12_V_p1p, optimize = einsum_type)
    S_12_V_p1p = einsum("ip,ip->ip", d_ip, S_12_V_p1p, optimize = einsum_type)
    S_12_V_p1p = einsum("mp,ip->im", evecs, S_12_V_p1p, optimize = einsum_type)
    del(V_p1p, d_ip, evals, evecs)

    ## Compute T[+1'] t1_ca and t1_caaa tensors
    t_p1p = einsum("Pm,im->iP", S_p1p_12_inv_act, S_12_V_p1p, optimize = einsum_type)
    del(S_p1p_12_inv_act, S_12_V_p1p)

    ## Build T[+1'] tensors
    t1_ca = t_p1p[:, V_a_i:V_a_f].copy()
    t1_caaa = t_p1p[:,V_bba_i: V_bba_f].reshape(ncore, ncas, ncas, ncas).copy()

    ## Transpose indices to the conventional order
    t1_caaa = t1_caaa.transpose(0,1,3,2).copy()
    del(t_p1p)

    # Compute electronic correlation energy for T[+1']
    e_p1p  = 2 * einsum('ix,ix', h_ca, t1_ca, optimize = einsum_type)
    e_p1p -= einsum('ix,iy,xy', h_ca, t1_ca, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('ix,iyzw,xyzw', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= einsum('ix,iyzx,zy', h_ca, t1_caaa, rdm_ca, optimize = einsum_type)
    e_p1p += 2 * einsum('ix,iyxz,zy', h_ca, t1_caaa, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('ix,iyzw,xzyw', t1_ca, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= einsum('ix,iyzx,yz', t1_ca, v_caaa, rdm_ca, optimize = einsum_type)
    e_p1p += 2 * einsum('ix,ixyz,zy', t1_ca, v_caaa, rdm_ca, optimize = einsum_type)
    e_p1p += 1/3 * einsum('ixyz,iwuv,xwvyzu', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= 2/3 * einsum('ixyz,iwuv,xwvzyu', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= einsum('ixyz,iwuz,yuwx', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= einsum('ixyz,izwu,ywxu', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 1/3 * einsum('ixyz,iwuv,xwvuzy', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/3 * einsum('ixyz,iwuv,xwvuyz', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/3 * einsum('ixyz,iwuv,xwvzuy', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/3 * einsum('ixyz,iwuv,xwvyuz', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= einsum('ixyz,iwuy,zuxw', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= einsum('ixyz,izwy,xw', t1_caaa, v_caaa, rdm_ca, optimize = einsum_type)
    e_p1p += 2 * einsum('ixyz,iywu,zwxu', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 2 * einsum('ixyz,iywz,xw', t1_caaa, v_caaa, rdm_ca, optimize = einsum_type)

    nevpt.log.extra("Norm of T[+1']^(1):                          %20.12f" % (np.linalg.norm(t1_ca) +
                                                                              np.linalg.norm(t1_caaa)))
    nevpt.log.info("Correlation energy [+1']:                    %20.12f" % e_p1p)
    nevpt.log.timer("computing T[+1']^(1) amplitudes", *cput0)

    return e_p1p, t1_ca, t1_caaa

def compute_t1_m1p_soc(nevpt, rdms, rdms_1s):

    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.extra("\nComputing T[-1']^(1) amplitudes...")

    # Einsum definition from kernel
    einsum = nevpt.interface.einsum
    einsum_type = nevpt.interface.einsum_type

    # Variables from kernel
    ncas = nevpt.ncas
    nextern = nevpt.nextern

    ## Molecular Orbitals Energies
    e_extern = nevpt.mo_energy.e

    ## One-electron integrals
    h_ae = nevpt.h1eff.ae

    ## Two-electron integrals
    v_aaae = nevpt.v2e.aaae

    ## Reduced density matrices
    rdm_ca = rdms.ca
    rdm_ccaa = rdms.ccaa
    rdm_cccaaa = rdms.cccaaa

    # Compute K_m1p matrix
    K_m1p = intermediates.compute_K_m1p(nevpt, rdms)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    if nevpt.semi_internal_projector == "gno":
        S_m1p_12_inv_act = overlap.compute_S12_m1p_gno_projector(nevpt, rdms)
    else:
        S_m1p_12_inv_act = overlap.compute_S12_m1p_gs_projector(nevpt, rdms)

    # Compute K^{-1} matrix
    SKS = reduce(np.dot, (S_m1p_12_inv_act.T, K_m1p, S_m1p_12_inv_act))
    evals, evecs = np.linalg.eigh(SKS)
    del(SKS)

    # Compute R.H.S. of the equation
    ## V1 block: - < Psi_0 | a^{\dag}_X a_A V | Psi_0>
    V1_a_a =- 1/2 * einsum('xA,Xx->XA', h_ae, rdm_ca, optimize = einsum_type)
    V1_a_a -= 1/2 * einsum('xyzA,Xyzx->XA', v_aaae, rdm_ccaa, optimize = einsum_type)

    ## V2 block: - < Psi_0 | a^{\dag}_X a^{\dag}_Y a_Z a_A V | Psi_0>
    V2_aa_aa  = 1/6 * einsum('xA,XYZx->XYZA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('xA,XYxZ->XYZA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('xZyA,XYxy->XYZA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('xZyA,XYyx->XYZA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('xyzA,XYyZzx->XYZA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('xyzA,XYyzZx->XYZA', v_aaae, rdm_cccaaa, optimize = einsum_type)

    V2_ab_ba =- 1/6 * einsum('xA,XYZx->XYZA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/3 * einsum('xA,XYxZ->XYZA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/6 * einsum('xZyA,XYxy->XYZA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/3 * einsum('xZyA,XYyx->XYZA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/6 * einsum('xyzA,XYyZxz->XYZA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba += 1/6 * einsum('xyzA,XYyxZz->XYZA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba += 1/6 * einsum('xyzA,XYyxzZ->XYZA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba -= 1/6 * einsum('xyzA,XYyzZx->XYZA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba += 1/6 * einsum('xyzA,XYyzxZ->XYZA', v_aaae, rdm_cccaaa, optimize = einsum_type)

    ## Reshape tensors to matrix form
    tril_ind = np.tril_indices(ncas, k=-1)

    V2_aa_aa = V2_aa_aa[tril_ind[0], tril_ind[1]]

    V2_aa_aa = V2_aa_aa.reshape(-1, nextern)
    V2_ab_ba = V2_ab_ba.reshape(-1, nextern)

    ## Build V matrix
    dim_X = ncas
    dim_YWZ = ncas * ncas * ncas
    dim_tril_YWZ = ncas * ncas * (ncas - 1) // 2

    dim_act = dim_X + dim_tril_YWZ + dim_YWZ

    V_a_i = 0
    V_a_f = dim_X
    V_aaa_i = V_a_f
    V_aaa_f = V_aaa_i + dim_tril_YWZ
    V_abb_i = V_aaa_f
    V_abb_f = V_abb_i + dim_YWZ

    V_m1p = np.zeros((dim_act, nextern))

    V_m1p[V_a_i:V_a_f, :] = V1_a_a.copy()
    del(V1_a_a)

    V_m1p[V_aaa_i:V_aaa_f, :] = V2_aa_aa.copy()
    V_m1p[V_abb_i:V_abb_f, :] = V2_ab_ba.copy()
    del(V2_aa_aa, V2_ab_ba)

    ## Compute denominators
    d_pa = (evals[:,None] + e_extern)
    d_pa = d_pa**(-1)

    # Compute T[-1'] amplitudes
    S_12_V_m1p = einsum("Pa,Pm->ma", V_m1p, S_m1p_12_inv_act, optimize = einsum_type)
    S_12_V_m1p = einsum("mp,ma->pa", evecs, S_12_V_m1p, optimize = einsum_type)
    S_12_V_m1p = einsum("pa,pa->pa", d_pa, S_12_V_m1p, optimize = einsum_type)
    S_12_V_m1p = einsum("mp,pa->ma", evecs, S_12_V_m1p, optimize = einsum_type)
    del(V_m1p, d_pa, evals, evecs)

    ## Compute T[-1'] t1_ae and t1_aaea tensors
    t_m1p = einsum("Pm,ma->Pa", S_m1p_12_inv_act, S_12_V_m1p, optimize = einsum_type)
    del(S_m1p_12_inv_act, S_12_V_m1p)

    ## Build T[-1'] tensors
    t1_ae = t_m1p[V_a_i:V_a_f, :].copy()
    t1_aaae = t_m1p[V_abb_i:V_abb_f, :].reshape(ncas, ncas, ncas, nextern).copy()
    t1_aaae = t1_aaae.transpose(1,0,2,3)
    del(t_m1p)

    # Compute electronic correlation energy for T[-1']
    e_m1p  = einsum('xa,ya,xy', h_ae, t1_ae, rdm_ca, optimize = einsum_type)
    e_m1p += einsum('xa,yzwa,xwzy', h_ae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    e_m1p += einsum('xa,yzwa,xzwy', t1_ae, v_aaae, rdm_ccaa, optimize = einsum_type)
    e_m1p -= 1/3 * einsum('xyza,wuva,zvwuyx', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p -= 1/3 * einsum('xyza,wuva,zvwxuy', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p += 2/3 * einsum('xyza,wuva,zvwxyu', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p -= 1/3 * einsum('xyza,wuva,zvwuxy', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p -= 1/3 * einsum('xyza,wuva,zvwyux', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p -= 1/3 * einsum('xyza,wuva,zvwyxu', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p += einsum('xyza,wzua,xywu', t1_aaae, v_aaae, rdm_ccaa, optimize = einsum_type)

    nevpt.log.extra("Norm of T[-1']^(1):                          %20.12f" % (np.linalg.norm(t1_ae) +
                                                                              np.linalg.norm(t1_aaae)))
    nevpt.log.info("Correlation energy [-1']:                    %20.12f" % e_m1p)
    nevpt.log.timer("computing T[-1']^(1) amplitudes", *cput0)

    return e_m1p, t1_ae, t1_aaae
