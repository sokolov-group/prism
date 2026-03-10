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

from prism.nevpt import rdms
from prism.nevpt import amplitudes
from prism.tools import transition

import prism.lib.logger as logger
import prism.lib.tools as tools


def compute_energy(method):

    n_states = len(method.ref_wfn_deg)
    n_micro_states = sum(method.ref_wfn_deg)

    e_tot = []
    e_corr = []
    mstate = 0

    e_0 = 0.0
    t1_0 = None
    t1 = []

    ncore = method.ncore - method.nfrozen

    if ncore > 0 and method.nextern > 0:
        e_0, t1_0 = amplitudes.compute_t1_0(method)
    else:
        t1_0 = np.zeros((ncore, ncore, method.nextern, method.nextern))
    
    for state in range(n_states):
        deg = method.ref_wfn_deg[state]

        method.log.info("\nComputing energy of state #%d..." % (state + 1))
        method.log.info("Reference state active-space energy:         %20.12f" % method.e_ref_cas[state])
        method.log.info("Reference state spin multiplicity:                 %d" % method.ref_wfn_spin_mult[state])
        method.log.info("Number of active electrons:                        %s" % str(method.ref_nelecas[mstate:(mstate+deg)]))

        # Compute reduced density matrices for a specific state
        rdms_ref = rdms.compute_reference_rdms(method, method.ref_wfn[mstate:(mstate+deg)], method.ref_nelecas[mstate:(mstate+deg)])

        # Compute amplitudes and correlation energy
        e_corr_state, t1_state = compute_energy_state(method, rdms_ref, e_0)
        e_tot_state = method.e_ref[state] + e_corr_state

        ref_name = method.interface.reference.upper()
        method_name = method.method.upper()
        method.log.info("%s reference state total energy: %s  %20.12f" % (ref_name.upper(), (12-len(ref_name)) * " ", method.e_ref[state]))
        method.log.info("%s correlation energy:           %s  %20.12f" % (method_name, (12-len(method_name)) * " ", e_corr_state))
        method.log.info("Total %s energy:                 %s  %20.12f" % (method_name, (12-len(method_name)) * " ", e_tot_state))

        e_corr.append(e_corr_state)
        e_tot.append(e_tot_state)

        t1.append(t1_state)

        del (rdms_ref)

        mstate += deg

    # Store amplitudes and energies in method class
    method.e_corr = e_corr
    method.e_tot = e_tot

    method.t1 = t1
    method.t1_0 = t1_0

    del(t1)
    del(t1_state)
    del(t1_0)


def compute_energy_state(method, rdms, e_0 = None):

    ncore = method.ncore - method.nfrozen
    ncas = method.ncas
    nelecas = method.ref_nelecas
    nextern = method.nextern

    e_0p, e_p1p, e_m1p, e_p1, e_m1, e_p2, e_m2 = (0.0,) * 7

    t1 = lambda:None

    # First-order amplitudes
    # With singles
    if method.compute_singles_amplitudes:
        if ncore > 0 and nextern > 0 and ncas > 0:
            e_0p, t1.ce, t1.caea, t1.caae = amplitudes.compute_t1_0p(method, rdms)
        else:
            t1.ce = np.zeros((ncore, nextern))
            t1.caea = np.zeros((ncore, ncas, nextern, ncas))
            t1.caae = np.zeros((ncore, ncas, ncas, nextern))

        if ncore > 0 and ncas > 0:
            e_p1p, t1.ca, t1.caaa = amplitudes.compute_t1_p1p(method, rdms)
        else:
            t1.ca = np.zeros((ncore, ncas))
            t1.caaa = np.zeros((ncore, ncas, ncas, ncas))

        if nextern > 0 and ncas > 0:
            e_m1p, t1.ae, t1.aaae = amplitudes.compute_t1_m1p(method, rdms)
        else:
            t1.ae = np.zeros((ncas, nextern))
            t1.aaae = np.zeros((ncas, ncas, ncas, nextern))
    # Without singles
    else:
        if ncore > 0 and nextern > 0 and ncas > 0:
            e_0p, t1.caea, t1.caae = amplitudes.compute_t1_0p_no_singles(method, rdms)
        else:
            t1.caea = np.zeros((ncore, ncas, nextern, ncas))
            t1.caae = np.zeros((ncore, ncas, ncas, nextern))

        if ncore > 0 and ncas > 0:
            e_p1p, t1.caaa = amplitudes.compute_t1_p1p_no_singles(method, rdms)
        else:
            t1.caaa = np.zeros((ncore, ncas, ncas, ncas))

        if nextern > 0 and ncas > 0:
            e_m1p, t1.aaae = amplitudes.compute_t1_m1p_no_singles(method, rdms)
        else:
            t1.aaae = np.zeros((ncas, ncas, ncas, nextern))

    nelecas_total = 0
    if isinstance(nelecas, (list)):
        nelecas_total = sum(nelecas[0])
    else:
        nelecas_total = sum(nelecas)

    if ncore > 0 and nextern > 0 and ncas > 0:
        e_p1, t1.ccae = amplitudes.compute_t1_p1(method, rdms)
    else:
        t1.ccae = np.zeros((ncore, ncore, ncas, nextern))

    if ncore > 0 and nextern > 0 and ncas > 0 and nelecas_total > 0:
        e_m1, t1.caee = amplitudes.compute_t1_m1(method, rdms)
    else:
        t1.caee = np.zeros((ncore, ncas, nextern, nextern))

    if ncore > 0 and ncas > 0:
        e_p2, t1.ccaa = amplitudes.compute_t1_p2(method, rdms)
    else:
        t1.ccaa = np.zeros((ncore, ncore, ncas, ncas))

    if nextern > 0 and ncas > 0 and nelecas_total > 1:
        e_m2, t1.aaee = amplitudes.compute_t1_m2(method, rdms)
    else:
        t1.aaee = np.zeros((ncas, ncas, nextern, nextern))

    if e_0 is None:
        if ncore > 0 and nextern > 0:
            e_0, t1.ccee = amplitudes.compute_t1_0(method)
        else:
            t1.ccee = np.zeros((ncore, ncore, nextern, nextern))
    else:
        method.log.info("Correlation energy [0]:                      %20.12f" % e_0)

    e_corr = e_0p + e_p1p + e_m1p + e_0 + e_p1 + e_m1 + e_p2 + e_m2

    return e_corr, t1


def compute_properties(method):

    # Determine spin multiplicity
    spin_mult = method.ref_wfn_spin_mult

    # Get Oscillator Strengths for transitions from ground state
    e_diff = method.e_tot - method.e_tot[0]
    e_diff = e_diff[1:]

    rdm_mo = make_rdm1(method, L = 0)
    osc_str = transition.osc_strength(method.interface, e_diff, rdm_mo[1:])

    # Compute all transitions starting from each state
    if method.verbose >= 5:
        osc_str_full = [osc_str.tolist()]
        for gs_index in range(1, len(method.e_tot)):  
            e_diff = method.e_tot - method.e_tot[gs_index]
            e_diff = e_diff[gs_index+1:]
            rdm_mo = make_rdm1(method, L = gs_index)
            osc_str_full.append(transition.osc_strength(method.interface, e_diff, rdm_mo[gs_index+1:]))

        transition.print_osc_strength(method.interface, osc_str_full)

    return osc_str, spin_mult


# Compute 1-RDM for either all CASCI states or a specific states
# L is initial, R is final
def make_rdm1(method, L = None, R = None, type = 'all', t1 = None, t1_0 = None):

    ncore = method.ncore
    ncas = method.ncas
    nextern = method.nextern
    nmo = method.nmo
    n_micro_states = sum(method.ref_wfn_deg)

    einsum = method.interface.einsum
    einsum_type = method.interface.einsum_type

    if t1 is None:
        t1 = method.t1
    
    if t1_0 is None:
        t1_0 = method.t1_0

    L_states = 0
    R_states = 0
    L_list = None
    R_list = None

    if L is None:
        L_states = n_micro_states
        L_list = np.arange(L_states)
    elif isinstance(L, int):
        L_list = np.array([L])
        if L > n_micro_states:
            raise ValueError(f"Invalid indices: L={L}. "f"Maximum allowed index is {n_micro_states - 1}.")
    else:
         raise ValueError(f"Value L={L} not supported")

    if R is None:
        R_states = n_micro_states
        R_list = np.arange(R_states)
    elif isinstance(R, int):
        R_list = np.array([R])
        if R > n_micro_states:
            raise ValueError(f"Invalid indices: R={R}. "f"Maximum allowed index is {n_micro_states - 1}.")
    else:
         raise ValueError(f"Value R={R} not supported")

    error_msg = (
        f"Instability detected in correlated 1RDM. "
        "Consider loosening truncation thresholds."
    )

    avail_types = ["all", "ss", "state-specific"]
    if type not in avail_types:
        raise ValueError(f"Invalid type: {type}. "f"Allowed types are {avail_types}.")
        
    # Initial rdm array
    rdm_final = np.zeros((L_list.shape[0], R_list.shape[0], nmo, nmo))
    
    t1_ccee = t1_0
    
    # Looping over states I,J
    for ind_I, I in enumerate(L_list):
        L_t1_caea = t1[I].caea
        L_t1_caae = t1[I].caae
        L_t1_caaa = t1[I].caaa
        L_t1_aaae = t1[I].aaae
        L_t1_ccae = t1[I].ccae
        L_t1_ccaa = t1[I].ccaa
        L_t1_caee = t1[I].caee 
        L_t1_aaee = t1[I].aaee
            
        for ind_J, J in enumerate(R_list): 
            
            if type in ("ss", "state-specific") and I != J:
                continue

            R_t1_caea = t1[J].caea
            R_t1_caae = t1[J].caae
            R_t1_caaa = t1[J].caaa
            R_t1_aaae = t1[J].aaae
            R_t1_ccae = t1[J].ccae
            R_t1_ccaa = t1[J].ccaa
            R_t1_caee = t1[J].caee 
            R_t1_aaee = t1[J].aaee
            
            # Zeroth-order contributions
            trdm_ca, trdm_ccaa, trdm_cccaaa = method.interface.compute_rdm123(method.ref_wfn[I], method.ref_wfn[J], method.ref_nelecas[I])
            rdm_final[ind_I, ind_J, ncore:ncore + ncas, ncore:ncore + ncas] = trdm_ca

            if I == J:
                #uncorrelated diagonal terms
                rdm_final[ind_I, ind_J, :ncore, :ncore] = 2 * np.identity(ncore)    

            if method.rdm_order == 2:
                
                # Initial rdm array for correlated contributions
                rdm_corr = np.zeros((nmo, nmo))
                
                # DIAGS #
                if I == J:
                    # CORE-CORE #
                    rdm_corr[:ncore, :ncore] -= 4 * einsum('Iiab,Jiab->IJ', t1_ccee, t1_ccee, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += 2 * einsum('Iiab,Jiba->IJ', t1_ccee, t1_ccee, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= 4 * einsum('Iixa,Jixa->IJ', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += 2 * einsum('Iixa,iJxa->IJ', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= 4 * einsum('Iixy,Jixy->IJ', L_t1_ccaa, R_t1_ccaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += 2 * einsum('Iixy,Jiyx->IJ', L_t1_ccaa, R_t1_ccaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += 2 * einsum('iIxa,Jixa->IJ', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= 4 * einsum('iIxa,iJxa->IJ', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += 2 * einsum('Iixa,Jiya,xy->IJ', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= einsum('Iixa,iJya,xy->IJ', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += 2 * einsum('Iixy,Jixz,yz->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= einsum('Iixy,Jiyz,xz->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= einsum('Iixy,Jizw,xyzw->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= einsum('Iixy,Jizx,yz->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += 2 * einsum('Iixy,Jizy,xz->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= 2 * einsum('Ixab,Jyab,xy->IJ', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += einsum('Ixab,Jyba,xy->IJ', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= 2 * einsum('Ixay,Jzaw,xwyz->IJ', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= 2 * einsum('Ixay,Jzay,xz->IJ', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += einsum('Ixay,Jzwa,xwyz->IJ', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += einsum('Ixay,Jzya,xz->IJ', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += einsum('Ixya,Jzaw,xwyz->IJ', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += einsum('Ixya,Jzay,xz->IJ', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += einsum('Ixya,Jzwa,xwzy->IJ', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= 2 * einsum('Ixya,Jzya,xz->IJ', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvwyz->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvwzy->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvywz->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvyzw->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvzwy->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += 2/3 * einsum('Ixyz,Jwuv,xuvzyw->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += einsum('Ixyz,Jwuy,xuzw->IJ', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += einsum('Ixyz,Jwuz,xuwy->IJ', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= 2 * einsum('Ixyz,Jwyu,xuzw->IJ', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= 2 * einsum('Ixyz,Jwyz,xw->IJ', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += einsum('Ixyz,Jwzu,xuyw->IJ', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += einsum('Ixyz,Jwzy,xw->IJ', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] -= einsum('iIxa,Jiya,xy->IJ', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, :ncore] += 2 * einsum('iIxa,iJya,xy->IJ', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                    
                    # ACT-ACT # 
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 4 * einsum('ijXa,ijYa->XY', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 2 * einsum('ijXa,jiYa->XY', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 4 * einsum('ijXx,ijYx->XY', L_t1_ccaa, R_t1_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 2 * einsum('ijXx,jiYx->XY', L_t1_ccaa, R_t1_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Xxab,yzab,Yxyz->XY', L_t1_aaee, R_t1_aaee, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Xxya,zwua,Yxuywz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Xxya,zwya,Yxzw->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Yxab,yzab,Xxyz->XY', L_t1_aaee, R_t1_aaee, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Yxya,zwua,Xxuywz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Yxya,zwya,Xxzw->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXab,ixab,Yx->XY', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXab,ixba,Yx->XY', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXax,iyax,Yy->XY', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXax,iyaz,Yzxy->XY', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXax,iyxa,Yy->XY', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXax,iyza,Yzxy->XY', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxa,iyax,Yy->XY', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxa,iyaz,Yzxy->XY', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXxa,iyxa,Yy->XY', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxa,iyza,Yzyx->XY', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuxyz->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuxzy->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('iXxy,izwu,Ywuyxz->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuyzx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuzxy->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuzyx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izwx,Ywyz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izwy,Ywzx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXxy,izxw,Ywyz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXxy,izxy,Yz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izyw,Ywxz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izyx,Yz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYab,ixab,Xx->XY', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYab,ixba,Xx->XY', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYax,iyax,Xy->XY', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYax,iyaz,Xzxy->XY', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYax,iyxa,Xy->XY', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYax,iyza,Xzxy->XY', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxa,iyax,Xy->XY', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxa,iyaz,Xzxy->XY', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYxa,iyxa,Xy->XY', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxa,iyza,Xzyx->XY', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuxyz->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuxzy->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('iYxy,izwu,Xwuyxz->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuyzx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuzxy->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuzyx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izwx,Xwyz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izwy,Xwzx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYxy,izxw,Xwyz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYxy,izxy,Xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izyw,Xwxz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izyx,Xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijXa,ijxa,Yx->XY', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijXa,jixa,Yx->XY', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 2 * einsum('ijXx,ijYy,xy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijXx,ijxy,Yy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijXx,ijyz,Yxyz->XY', L_t1_ccaa, R_t1_ccaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ijXx,jiYy,xy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijXx,jixy,Yy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijYa,ijxa,Xx->XY', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijYa,jixa,Xx->XY', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijYx,ijxy,Xy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijYx,ijyz,Xxyz->XY', L_t1_ccaa, R_t1_ccaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijYx,jixy,Xy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixXa,iyYa,xy->XY', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixXa,iyaY,xy->XY', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXa,iyaz,Yyxz->XY', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXa,iyza,Yyzx->XY', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixXy,izYw,yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixXy,izYy,xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixXy,izwY,yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzuwx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzuxw->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzwux->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('ixXy,izwu,Yyzwxu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzxuw->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzxwu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXy,izwy,Yzwx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixXy,izyY,xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXy,izyw,Yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYa,iyaz,Xyxz->XY', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYa,iyza,Xyzx->XY', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzuwx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzuxw->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzwux->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('ixYy,izwu,Xyzwxu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzxuw->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzxwu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYy,izwy,Xzwx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYy,izyw,Xzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixaX,iyYa,xy->XY', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixaX,iyaY,xy->XY', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixaX,iyaz,Yyxz->XY', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixaX,iyza,Yyxz->XY', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixaY,iyaz,Xyxz->XY', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixaY,iyza,Xyxz->XY', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixyX,izYw,yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixyX,izYy,xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixyX,izwY,yzwx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyX,izwu,Yyzxwu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyX,izwy,Yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixyX,izyY,xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixyX,izyw,Yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyY,izwu,Xyzxwu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyY,izwy,Xzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixyY,izyw,Xzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('xXya,zwua,Yxuwyz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuwzy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuywz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuyzw->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuzwy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuzyw->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('xXya,zwya,Yxwz->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('xYya,zwua,Xxuwyz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuwzy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuywz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuyzw->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuzwy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuzyw->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('xYya,zwya,Xxwz->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('xyXa,zwYa,xyzw->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzuxy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzuyx->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzxuy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('xyXa,zwua,Ywzxyu->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzyux->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzyxu->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzuxy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzuyx->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzxuy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('xyYa,zwua,Xwzxyu->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzyux->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzyxu->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    
                    # EXT-EXT #
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 4 * einsum('ijAa,ijBa->AB', t1_ccee, t1_ccee, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 2 * einsum('ijAa,jiBa->AB', t1_ccee, t1_ccee, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 4 * einsum('ijxA,ijxB->AB', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 2 * einsum('ijxA,jixB->AB', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 2 * einsum('ijxA,ijyB,xy->AB', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += einsum('ijxA,jiyB,xy->AB', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixAa,iyBa,xy->AB', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixAa,iyaB,xy->AB', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixAy,izBw,yzxw->AB', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixAy,izBy,xz->AB', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixAy,izwB,yzxw->AB', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixAy,izyB,xz->AB', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixaA,iyBa,xy->AB', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixaA,iyaB,xy->AB', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixyA,izBw,yzxw->AB', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixyA,izBy,xz->AB', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixyA,izwB,yzwx->AB', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixyA,izyB,xz->AB', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += einsum('xyAa,zwBa,xyzw->AB', L_t1_aaee, R_t1_aaee, trdm_ccaa, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwvxy->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwvyx->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwxvy->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2/3 * einsum('xyzA,wuvB,zuwxyv->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwyvx->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwyxv->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += einsum('xyzA,wuzB,yxuw->AB', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                # OFF-DIAGS #
                else:
                    # COR-ACT #
                    rdm_corr[:ncore, ncore:ncore + ncas] += einsum('IxXy,yx->IX', R_t1_caaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, ncore:ncore + ncas] -= 1/2 * einsum('IxyX,yx->IX', R_t1_caaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, ncore:ncore + ncas] -= 1/2 * einsum('Ixyz,yzXx->IX', R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    
                    # ACT-COR #
                    rdm_corr[ncore:ncore + ncas, :ncore] += einsum('IxXy,xy->XI', L_t1_caaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, :ncore] -= 1/2 * einsum('IxyX,xy->XI', L_t1_caaa, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore:ncore + ncas, :ncore] -= 1/2 * einsum('Ixyz,Xxyz->XI', L_t1_caaa, trdm_ccaa, optimize = einsum_type)
                    
                    # COR-EXT #
                    rdm_corr[:ncore, ncore + ncas:ncore + ncas + nextern] += einsum('IxAy,yx->IA', R_t1_caea, trdm_ca, optimize = einsum_type)
                    rdm_corr[:ncore, ncore + ncas:ncore + ncas + nextern] -= 1/2 * einsum('IxyA,yx->IA', R_t1_caae, trdm_ca, optimize = einsum_type)
                    
                    # EXT-COR #
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, :ncore] += einsum('IxAy,xy->AI', L_t1_caea, trdm_ca, optimize = einsum_type)
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, :ncore] -= 1/2 * einsum('IxyA,xy->AI', L_t1_caae, trdm_ca, optimize = einsum_type)
                    
                    # ACT-EXT #
                    rdm_corr[ncore:ncore + ncas, ncore + ncas:ncore + ncas + nextern] += 1/2 * einsum('xyzA,Xzyx->XA', R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                    
                    # EXT-ACT #
                    rdm_corr[ncore + ncas:ncore + ncas + nextern, ncore:ncore + ncas] += 1/2 * einsum('xyzA,Xzyx->AX', L_t1_aaae, trdm_ccaa, optimize = einsum_type)

                # Add the correlated contribution
                rdm_final[ind_I, ind_J, :, :] += rdm_corr
                    
                # RDM warning
                norm_check = np.linalg.norm(rdm_corr) / method.nelec

                if norm_check > 0.1:
                    method.log.info(f"WARNING: {error_msg}")
        
    # Single pair of states
    if L is not None and R is not None:
        rdm_final = rdm_final[0,0]

    # One state on the left or right
    if L_list.shape[0] == 1 and R_list.shape[0] > 1:
        rdm_final = rdm_final[0]

    if L_list.shape[0] > 1 and R_list.shape[0] == 1:
        rdm_final = rdm_final[:, 0, :, :]

    # State-specific
    if type in ("ss", "state-specific"):
        rdm_final = np.diagonal(rdm_final, axis1=0, axis2=1)
        rdm_final = np.moveaxis(rdm_final, -1, 0)
        
    return rdm_final


def make_rdm1s(method, L = None, R = None, type = 'all', t1 = None, t1_0 = None):
    ncore = method.ncore
    ncas = method.ncas
    nextern = method.nextern
    n_micro_states = sum(method.ref_wfn_deg)
    einsum = method.interface.einsum   
    einsum_type = method.interface.einsum_type
    nmo = method.nmo

    L_states = 0
    R_states = 0
    L_list = None
    R_list = None

    if L is None:
        L_states = n_micro_states
        L_list = np.arange(L_states)
    else:
        L_list = np.array([L])
        if L > n_micro_states:
            raise ValueError(f"Invalid indices: L={L}. "f"Maximum allowed index is {n_micro_states - 1}.")

    if R is None:
        R_states = n_micro_states
        R_list = np.arange(R_states)
        
    else:
        R_list = np.array([R])
        if R > n_micro_states:
            raise ValueError(f"Invalid indices: R={R}. "f"Maximum allowed index is {n_micro_states - 1}.")

    error_msg = (
        f"Instability detected in correlated 1RDM. "
        "Consider loosening truncation thresholds."
    )

    avail_types = ["all", "ss", "state-specific"]
    if type not in avail_types:
        raise ValueError(f"Invalid type: {type}. "f"Allowed types are {avail_types}.")
        
    # Initial rdm array
    rdm_final = np.zeros((2, L_list.shape[0], R_list.shape[0], nmo, nmo))
    
    #method's wfn
    if method.method == "qd-nevpt2":
        wfn = np.einsum('ij,iab->jab',method.h_evec,method.ref_wfn)
        wfn = list(wfn)
    else:
        wfn = list(method.ref_wfn)
    
    # Looping over states I,J
    for ind_I, I in enumerate(L_list):
        for ind_J, J in enumerate(R_list): 
            
            if type in ("ss", "state-specific") and I != J:
                continue

            from pyscf.fci.direct_spin1 import trans_rdm1s
            tmprdm_aabb = trans_rdm1s(wfn[J], wfn[I], method.ncas, method.ref_nelecas[ind_I])
            rdm_final[0, ind_I, ind_J, method.ncore:method.ncore+method.ncas, method.ncore:method.ncore+method.ncas] = tmprdm_aabb[0]
            rdm_final[1, ind_I, ind_J, method.ncore:method.ncore+method.ncas, method.ncore:method.ncore+method.ncas] = tmprdm_aabb[1]            

            if I == J:
                #uncorrelated diagonal terms
                rdm_final[:,ind_I, ind_J, :ncore, :ncore] =   np.identity(ncore)    

            if method.rdm_order == 2:
                raise ValueError(f"Invalid type: corelation not implement in spin-orbital RDM. ")
                
    # Single pair of states
    if L is not None and R is not None:
        rdm_final = rdm_final[:,0,0]
        
    # State-specific
    if type in ("ss", "state-specific"):
        rdm_final[0] = np.diagonal(rdm_final[0], axis1=0, axis2=1)
        rdm_final[0] = np.moveaxis(rdm_final[0], -1, 0)

        rdm_final[1] = np.diagonal(rdm_final[1], axis1=0, axis2=1)
        rdm_final[1] = np.moveaxis(rdm_final[1], -1, 0)
        
    return rdm_final


