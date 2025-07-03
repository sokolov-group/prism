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
#

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


def osc_strength(nevpt, en, gs_index = 0):

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

    for state in range(gs_index + 1, n_micro_states):
        rdm_mo = np.zeros((nmo, nmo))
        trdm_ca = nevpt.interface.compute_rdm1(nevpt.ref_wfn[gs_index], nevpt.ref_wfn[state], nevpt.ref_nelecas[gs_index])
        rdm_mo[ncore:ncore + ncas ,ncore:ncore + ncas] = trdm_ca

        # Create Dipole Moment Operator with RDM
        dip_evec_x = np.einsum('pq,pq', dip_mom_mo[0], rdm_mo)
        dip_evec_y = np.einsum('pq,pq', dip_mom_mo[1], rdm_mo)
        dip_evec_z = np.einsum('pq,pq', dip_mom_mo[2], rdm_mo)
    
        osc_x = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_x)*dip_evec_x)
        osc_y = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_y)*dip_evec_y)
        osc_z = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_z)*dip_evec_z)

        # Add Dipole Moment Components
        osc_total.append((osc_x + osc_y + osc_z).real)

    return osc_total