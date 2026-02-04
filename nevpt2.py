# Copyright 2025 Prism Developers. All Rights Reserved.
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

import numpy as np
from functools import reduce

import prism.nevpt_intermediates as nevpt_intermediates
import prism.nevpt_overlap as nevpt_overlap
import prism.nevpt_amplitudes as nevpt_amplitudes

import prism.lib.logger as logger
import prism.lib.tools as tools

def compute_energy(nevpt, rdms, e_0 = None):

    ncore = nevpt.ncore - nevpt.nfrozen
    ncas = nevpt.ncas
    nelecas = nevpt.ref_nelecas
    nextern = nevpt.nextern

    e_0p, e_p1p, e_m1p, e_p1, e_m1, e_p2, e_m2 = (0.0,) * 7

    t1 = lambda:None

    # First-order amplitudes
    # With singles
    if nevpt.compute_singles_amplitudes:
        if ncore > 0 and nextern > 0 and ncas > 0:
            e_0p, t1.ce, t1.caea, t1.caae = nevpt_amplitudes.compute_t1_0p(nevpt, rdms)
        else:
            t1.ce = np.zeros((ncore, nextern))
            t1.caea = np.zeros((ncore, ncas, nextern, ncas))
            t1.caae = np.zeros((ncore, ncas, ncas, nextern))

        if ncore > 0 and ncas > 0:
            e_p1p, t1.ca, t1.caaa = nevpt_amplitudes.compute_t1_p1p(nevpt, rdms)
        else:
            t1.ca = np.zeros((ncore, ncas))
            t1.caaa = np.zeros((ncore, ncas, ncas, ncas))

        if nextern > 0 and ncas > 0:
            e_m1p, t1.ae, t1.aaae = nevpt_amplitudes.compute_t1_m1p(nevpt, rdms)
        else:
            t1.ae = np.zeros((ncas, nextern))
            t1.aaae = np.zeros((ncas, ncas, ncas, nextern))
    # Without singles
    else:
        if ncore > 0 and nextern > 0 and ncas > 0:
            e_0p, t1.caea, t1.caae = nevpt_amplitudes.compute_t1_0p_no_singles(nevpt, rdms)
        else:
            t1.caea = np.zeros((ncore, ncas, nextern, ncas))
            t1.caae = np.zeros((ncore, ncas, ncas, nextern))

        if ncore > 0 and ncas > 0:
            e_p1p, t1.caaa = nevpt_amplitudes.compute_t1_p1p_no_singles(nevpt, rdms)
        else:
            t1.caaa = np.zeros((ncore, ncas, ncas, ncas))

        if nextern > 0 and ncas > 0:
            e_m1p, t1.aaae = nevpt_amplitudes.compute_t1_m1p_no_singles(nevpt, rdms)
        else:
            t1.aaae = np.zeros((ncas, ncas, ncas, nextern))

    nelecas_total = 0
    if isinstance(nelecas, (list)):
        nelecas_total = sum(nelecas[0])
    else:
        nelecas_total = sum(nelecas)

    if ncore > 0 and nextern > 0 and ncas > 0:
        e_p1, t1.ccae = nevpt_amplitudes.compute_t1_p1(nevpt, rdms)
    else:
        t1.ccae = np.zeros((ncore, ncore, ncas, nextern))

    if ncore > 0 and nextern > 0 and ncas > 0 and nelecas_total > 0:
        e_m1, t1.caee = nevpt_amplitudes.compute_t1_m1(nevpt, rdms)
    else:
        t1.caee = np.zeros((ncore, ncas, nextern, nextern))

    if ncore > 0 and ncas > 0:
        e_p2, t1.ccaa = nevpt_amplitudes.compute_t1_p2(nevpt, rdms)
    else:
        t1.ccaa = np.zeros((ncore, ncore, ncas, ncas))

    if nextern > 0 and ncas > 0 and nelecas_total > 1:
        e_m2, t1.aaee = nevpt_amplitudes.compute_t1_m2(nevpt, rdms)
    else:
        t1.aaee = np.zeros((ncas, ncas, nextern, nextern))

    if e_0 is None:
        if ncore > 0 and nextern > 0:
            e_0, t1.ccee = nevpt_amplitudes.compute_t1_0(nevpt)
        else:
            t1.ccee = np.zeros((ncore, ncore, nextern, nextern))
    else:
        nevpt.log.info("Correlation energy [0]:                      %20.12f" % e_0)

    e_corr = e_0p + e_p1p + e_m1p + e_0 + e_p1 + e_m1 + e_p2 + e_m2

    return e_corr, t1


def osc_strength(nevpt, en, t1, t1_0, gs_index = 0):

    ncore = nevpt.ncore 
    n_micro_states = sum(nevpt.ref_wfn_deg)
    dip_mom_ao = nevpt.interface.dip_mom_ao
    mo_coeff = nevpt.mo
    nmo = nevpt.nmo
    ncas = nevpt.ncas

    dip_mom_mo = np.zeros_like(dip_mom_ao)

    # Transform dipole moments from AO to MO basis
    for d in range(dip_mom_ao.shape[0]):
        dip_mom_mo[d] = mo_coeff.T @ dip_mom_ao[d] @ mo_coeff

    # List to store Osc. Strength Values
    osc_total = []
    osc_total_corr = []

    for state in range(gs_index + 1, n_micro_states):
        rdm_mo = np.zeros((nmo, nmo))
        rdm_ca = nevpt.interface.compute_rdm1(nevpt.ref_wfn[gs_index], nevpt.ref_wfn[state], nevpt.ref_nelecas[gs_index])
        rdm_mo[ncore:ncore + ncas ,ncore:ncore + ncas] = rdm_ca

        rdm_mo_corr = compute_corr_1rdm(nevpt, t1, t1_0, gs_idx = gs_index, es_idx = state)

        # Create Dipole Moment Operator with RDM
        dip_evec_x = np.einsum('pq,pq', dip_mom_mo[0], rdm_mo)
        dip_evec_y = np.einsum('pq,pq', dip_mom_mo[1], rdm_mo)
        dip_evec_z = np.einsum('pq,pq', dip_mom_mo[2], rdm_mo)
        
        # Create Dipole Moment Operator with Correlated RDM
        dip_evec_x_corr = np.einsum('pq,pq', dip_mom_mo[0], rdm_mo_corr)
        dip_evec_y_corr = np.einsum('pq,pq', dip_mom_mo[1], rdm_mo_corr)
        dip_evec_z_corr = np.einsum('pq,pq', dip_mom_mo[2], rdm_mo_corr)

        # Uncorrelated
        osc_x = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_x)*dip_evec_x)
        osc_y = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_y)*dip_evec_y)
        osc_z = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_z)*dip_evec_z)

        # Add Dipole Moment Components
        osc_total.append((osc_x + osc_y + osc_z).real)
        
        # Correlated
        osc_x = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_x_corr)*dip_evec_x_corr)
        osc_y = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_y_corr)*dip_evec_y_corr)
        osc_z = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_z_corr)*dip_evec_z_corr)

        # Add Dipole Moment Components
        osc_total_corr.append((osc_x + osc_y + osc_z).real)
        
    return (osc_total, osc_total_corr)


def compute_corr_1rdm(nevpt, t1, t1_0, gs_idx = None, es_idx = None):
    ncore = nevpt.ncore 
    n_micro_states = sum(nevpt.ref_wfn_deg)
    ncas = nevpt.ncas
    mo_coeff = nevpt.mo
    einsum = nevpt.interface.einsum
    einsum_type = nevpt.interface.einsum_type
    nmo = nevpt.nmo
    ncas = nevpt.ncas
    nextern = nevpt.nextern
    
    if gs_idx is None:
        gs_idx = 0
        
    if es_idx is None:
        es_idx = 0
    
    if n_micro_states > 1:
        t1_caea = t1[gs_idx].caea 
        t1_caae = t1[gs_idx].caae  
        t1_caaa = t1[gs_idx].caaa 
        t1_aaae = t1[gs_idx].aaae 
        t1_ccae = t1[gs_idx].ccae 
        t1_ccaa = t1[gs_idx].ccaa 
        t1_caee = t1[gs_idx].caee 
        t1_aaee = t1[gs_idx].aaee 
        
        t1_caea_ = t1[es_idx].caea 
        t1_caae_ = t1[es_idx].caae  
        t1_caaa_ = t1[es_idx].caaa 
        t1_aaae_ = t1[es_idx].aaae 
        
    else: 
        t1_caea = t1.caea 
        t1_caae = t1.caae  
        t1_caaa = t1.caaa 
        t1_aaae = t1.aaae 
        t1_ccae = t1.ccae 
        t1_ccaa = t1.ccaa 
        t1_caee = t1.caee 
        t1_aaee = t1.aaee 
    
    t1_ccee = t1_0
    
    # Initialize 1rdm
    rdm = np.zeros((nmo, nmo)) 
    rdm_ca, rdm_ccaa, rdm_cccaaa, rdm_ccccaaaa = nevpt.interface.compute_rdm1234(nevpt.ref_wfn[gs_idx], nevpt.ref_wfn[es_idx], nevpt.ref_nelecas[gs_idx])
    rdm[ncore:ncore + ncas, ncore:ncore + ncas] = rdm_ca
    
    if gs_idx == es_idx:
        rdm[:ncore, :ncore] = 2 * np.eye(ncore)

        # CORE-CORE #
        rdm[:ncore, :ncore] -= 4 * einsum('Iiab,Jiab->IJ', t1_ccee, t1_ccee, optimize = einsum_type)
        rdm[:ncore, :ncore] += 2 * einsum('Iiab,Jiba->IJ', t1_ccee, t1_ccee, optimize = einsum_type)
        rdm[:ncore, :ncore] -= 4 * einsum('Iixa,Jixa->IJ', t1_ccae, t1_ccae, optimize = einsum_type)
        rdm[:ncore, :ncore] += 2 * einsum('Iixa,iJxa->IJ', t1_ccae, t1_ccae, optimize = einsum_type)
        rdm[:ncore, :ncore] -= 4 * einsum('Iixy,Jixy->IJ', t1_ccaa, t1_ccaa, optimize = einsum_type)
        rdm[:ncore, :ncore] += 2 * einsum('Iixy,Jiyx->IJ', t1_ccaa, t1_ccaa, optimize = einsum_type)
        rdm[:ncore, :ncore] += 2 * einsum('iIxa,Jixa->IJ', t1_ccae, t1_ccae, optimize = einsum_type)
        rdm[:ncore, :ncore] -= 4 * einsum('iIxa,iJxa->IJ', t1_ccae, t1_ccae, optimize = einsum_type)
        rdm[:ncore, :ncore] += 2 * einsum('Iixa,Jiya,xy->IJ', t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] -= einsum('Iixa,iJya,xy->IJ', t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] += 2 * einsum('Iixy,Jixz,yz->IJ', t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] -= einsum('Iixy,Jiyz,xz->IJ', t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] -= einsum('Iixy,Jizw,xyzw->IJ', t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        rdm[:ncore, :ncore] -= einsum('Iixy,Jizx,yz->IJ', t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] += 2 * einsum('Iixy,Jizy,xz->IJ', t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] -= 2 * einsum('Ixab,Jyab,xy->IJ', t1_caee, t1_caee, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] += einsum('Ixab,Jyba,xy->IJ', t1_caee, t1_caee, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] -= 2 * einsum('Ixay,Jzaw,xwyz->IJ', t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        rdm[:ncore, :ncore] -= 2 * einsum('Ixay,Jzay,xz->IJ', t1_caea, t1_caea, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] += einsum('Ixay,Jzwa,xwyz->IJ', t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        rdm[:ncore, :ncore] += einsum('Ixay,Jzya,xz->IJ', t1_caea, t1_caae, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] += einsum('Ixya,Jzaw,xwyz->IJ', t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        rdm[:ncore, :ncore] += einsum('Ixya,Jzay,xz->IJ', t1_caae, t1_caea, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] += einsum('Ixya,Jzwa,xwzy->IJ', t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        rdm[:ncore, :ncore] -= 2 * einsum('Ixya,Jzya,xz->IJ', t1_caae, t1_caae, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvwyz->IJ', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvwzy->IJ', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvywz->IJ', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvyzw->IJ', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvzwy->IJ', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[:ncore, :ncore] += 2/3 * einsum('Ixyz,Jwuv,xuvzyw->IJ', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[:ncore, :ncore] += einsum('Ixyz,Jwuy,xuzw->IJ', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[:ncore, :ncore] += einsum('Ixyz,Jwuz,xuwy->IJ', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[:ncore, :ncore] -= 2 * einsum('Ixyz,Jwyu,xuzw->IJ', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[:ncore, :ncore] -= 2 * einsum('Ixyz,Jwyz,xw->IJ', t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] += einsum('Ixyz,Jwzu,xuyw->IJ', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[:ncore, :ncore] += einsum('Ixyz,Jwzy,xw->IJ', t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] -= einsum('iIxa,Jiya,xy->IJ', t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        rdm[:ncore, :ncore] += 2 * einsum('iIxa,iJya,xy->IJ', t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        
        # ACT-ACT # 
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 4 * einsum('ijXa,ijYa->XY', t1_ccae, t1_ccae, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 2 * einsum('ijXa,jiYa->XY', t1_ccae, t1_ccae, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 4 * einsum('ijXx,ijYx->XY', t1_ccaa, t1_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 2 * einsum('ijXx,jiYx->XY', t1_ccaa, t1_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Xxab,yzab,Yxyz->XY', t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Xxya,zwua,Yxuywz->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Xxya,zwya,Yxzw->XY', t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Yxab,yzab,Xxyz->XY', t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Yxya,zwua,Xxuywz->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Yxya,zwya,Xxzw->XY', t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXab,ixab,Yx->XY', t1_caee, t1_caee, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXab,ixba,Yx->XY', t1_caee, t1_caee, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXax,iyax,Yy->XY', t1_caea, t1_caea, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXax,iyaz,Yzxy->XY', t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXax,iyxa,Yy->XY', t1_caea, t1_caae, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXax,iyza,Yzxy->XY', t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxa,iyax,Yy->XY', t1_caae, t1_caea, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxa,iyaz,Yzxy->XY', t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXxa,iyxa,Yy->XY', t1_caae, t1_caae, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxa,iyza,Yzyx->XY', t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuxyz->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuxzy->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('iXxy,izwu,Ywuyxz->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuyzx->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuzxy->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuzyx->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izwx,Ywyz->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izwy,Ywzx->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXxy,izxw,Ywyz->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXxy,izxy,Yz->XY', t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izyw,Ywxz->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izyx,Yz->XY', t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYab,ixab,Xx->XY', t1_caee, t1_caee, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYab,ixba,Xx->XY', t1_caee, t1_caee, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYax,iyax,Xy->XY', t1_caea, t1_caea, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYax,iyaz,Xzxy->XY', t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYax,iyxa,Xy->XY', t1_caea, t1_caae, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYax,iyza,Xzxy->XY', t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxa,iyax,Xy->XY', t1_caae, t1_caea, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxa,iyaz,Xzxy->XY', t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYxa,iyxa,Xy->XY', t1_caae, t1_caae, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxa,iyza,Xzyx->XY', t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuxyz->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuxzy->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('iYxy,izwu,Xwuyxz->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuyzx->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuzxy->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuzyx->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izwx,Xwyz->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izwy,Xwzx->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYxy,izxw,Xwyz->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYxy,izxy,Xz->XY', t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izyw,Xwxz->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izyx,Xz->XY', t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijXa,ijxa,Yx->XY', t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijXa,jixa,Yx->XY', t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 2 * einsum('ijXx,ijYy,xy->XY', t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijXx,ijxy,Yy->XY', t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijXx,ijyz,Yxyz->XY', t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ijXx,jiYy,xy->XY', t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijXx,jixy,Yy->XY', t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijYa,ijxa,Xx->XY', t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijYa,jixa,Xx->XY', t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijYx,ijxy,Xy->XY', t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijYx,ijyz,Xxyz->XY', t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijYx,jixy,Xy->XY', t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixXa,iyYa,xy->XY', t1_caae, t1_caae, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixXa,iyaY,xy->XY', t1_caae, t1_caea, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXa,iyaz,Yyxz->XY', t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXa,iyza,Yyzx->XY', t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixXy,izYw,yzxw->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixXy,izYy,xz->XY', t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixXy,izwY,yzxw->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzuwx->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzuxw->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzwux->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('ixXy,izwu,Yyzwxu->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzxuw->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzxwu->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXy,izwy,Yzwx->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixXy,izyY,xz->XY', t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXy,izyw,Yzxw->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYa,iyaz,Xyxz->XY', t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYa,iyza,Xyzx->XY', t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzuwx->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzuxw->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzwux->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('ixYy,izwu,Xyzwxu->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzxuw->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzxwu->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYy,izwy,Xzwx->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYy,izyw,Xzxw->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixaX,iyYa,xy->XY', t1_caea, t1_caae, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixaX,iyaY,xy->XY', t1_caea, t1_caea, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixaX,iyaz,Yyxz->XY', t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixaX,iyza,Yyxz->XY', t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixaY,iyaz,Xyxz->XY', t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixaY,iyza,Xyxz->XY', t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixyX,izYw,yzxw->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixyX,izYy,xz->XY', t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixyX,izwY,yzwx->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyX,izwu,Yyzxwu->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyX,izwy,Yzxw->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixyX,izyY,xz->XY', t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixyX,izyw,Yzxw->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyY,izwu,Xyzxwu->XY', t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyY,izwy,Xzxw->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixyY,izyw,Xzxw->XY', t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('xXya,zwua,Yxuwyz->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuwzy->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuywz->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuyzw->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuzwy->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuzyw->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('xXya,zwya,Yxwz->XY', t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('xYya,zwua,Xxuwyz->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuwzy->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuywz->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuyzw->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuzwy->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuzyw->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('xYya,zwya,Xxwz->XY', t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('xyXa,zwYa,xyzw->XY', t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzuxy->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzuyx->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzxuy->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('xyXa,zwua,Ywzxyu->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzyux->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzyxu->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzuxy->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzuyx->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzxuy->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('xyYa,zwua,Xwzxyu->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzyux->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzyxu->XY', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        
        # EXT-EXT #
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 4 * einsum('ijAa,ijBa->AB', t1_ccee, t1_ccee, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 2 * einsum('ijAa,jiBa->AB', t1_ccee, t1_ccee, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 4 * einsum('ijxA,ijxB->AB', t1_ccae, t1_ccae, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 2 * einsum('ijxA,jixB->AB', t1_ccae, t1_ccae, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 2 * einsum('ijxA,ijyB,xy->AB', t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += einsum('ijxA,jiyB,xy->AB', t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixAa,iyBa,xy->AB', t1_caee, t1_caee, rdm_ca, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixAa,iyaB,xy->AB', t1_caee, t1_caee, rdm_ca, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixAy,izBw,yzxw->AB', t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixAy,izBy,xz->AB', t1_caea, t1_caea, rdm_ca, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixAy,izwB,yzxw->AB', t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixAy,izyB,xz->AB', t1_caea, t1_caae, rdm_ca, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixaA,iyBa,xy->AB', t1_caee, t1_caee, rdm_ca, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixaA,iyaB,xy->AB', t1_caee, t1_caee, rdm_ca, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixyA,izBw,yzxw->AB', t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixyA,izBy,xz->AB', t1_caae, t1_caea, rdm_ca, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixyA,izwB,yzwx->AB', t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixyA,izyB,xz->AB', t1_caae, t1_caae, rdm_ca, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += einsum('xyAa,zwBa,xyzw->AB', t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwvxy->AB', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwvyx->AB', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwxvy->AB', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2/3 * einsum('xyzA,wuvB,zuwxyv->AB', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwyvx->AB', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwyxv->AB', t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += einsum('xyzA,wuzB,yxuw->AB', t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    
    elif gs_idx != es_idx: 
        # OFF-DIAGS #
        # COR-ACT #
        rdm[:ncore, ncore:ncore + ncas] += einsum('IxXy,yx->IX', t1_caaa, rdm_ca, optimize = einsum_type)
        rdm[:ncore, ncore:ncore + ncas] -= 1/2 * einsum('IxyX,yx->IX', t1_caaa, rdm_ca, optimize = einsum_type)
        rdm[:ncore, ncore:ncore + ncas] -= 1/2 * einsum('Ixyz,yzXx->IX', t1_caaa, rdm_ccaa, optimize = einsum_type)
        
        # ACT-COR #
        rdm[ncore:ncore + ncas, :ncore] += einsum('IxXy,xy->XI', t1_caaa_, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, :ncore] -= 1/2 * einsum('IxyX,xy->XI', t1_caaa_, rdm_ca, optimize = einsum_type)
        rdm[ncore:ncore + ncas, :ncore] -= 1/2 * einsum('Ixyz,Xxyz->XI', t1_caaa_, rdm_ccaa, optimize = einsum_type)
        
        # COR-EXT #
        rdm[:ncore, ncore + ncas:ncore + ncas + nextern] += einsum('IxAy,yx->IA', t1_caea_, rdm_ca, optimize = einsum_type)
        rdm[:ncore, ncore + ncas:ncore + ncas + nextern] -= 1/2 * einsum('IxyA,yx->IA', t1_caae_, rdm_ca, optimize = einsum_type)
        
        # EXT-COR #
        rdm[ncore + ncas:ncore + ncas + nextern, :ncore] += einsum('IxAy,xy->AI', t1_caea, rdm_ca, optimize = einsum_type)
        rdm[ncore + ncas:ncore + ncas + nextern, :ncore] -= 1/2 * einsum('IxyA,xy->AI', t1_caae, rdm_ca, optimize = einsum_type)
        
        # ACT-EXT #
        rdm[ncore:ncore + ncas, ncore + ncas:ncore + ncas + nextern] += 1/2 * einsum('xyzA,Xzyx->XA', t1_aaae_, rdm_ccaa, optimize = einsum_type)
        
        # EXT-ACT #
        rdm[ncore + ncas:ncore + ncas + nextern, ncore:ncore + ncas] += 1/2 * einsum('xyzA,Xzyx->AX', t1_aaae, rdm_ccaa, optimize = einsum_type)

    return rdm