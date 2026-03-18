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
#          Nicholas Y. Chiang <nicholas.yiching.chiang@gmail.com>
#

import numpy as np

from prism.libsoc import general_somf
from prism.libsoc import magnetic

def state_interaction_soc(method):

    method.log.info("\nInitializing SOC program...")
    method.interface.x2c_setup()

    # Rotate CAS Wavefunction:
    if method.__class__.__name__ == "NEVPT":
        wfn = list(method.ref_wfn)
    elif method.__class__.__name__ == "QDNEVPT":
        wfn = np.einsum('ij,iab->jab',method.h_evec,method.ref_wfn)
        wfn = list(wfn)
    
    # Calculate method's S, Ms:
    S  = []
    ms = []
    nstate = len(method.e_tot)
    for I in range(nstate):
        sz = method.interface.apply_S_z(wfn[I],method.ncas,method.ref_nelecas[I])
        ms.append(np.dot(wfn[I].ravel(), sz.ravel()))
        SS = method.interface.compute_spin_square(wfn[I], method.ncas, method.ref_nelecas[I])
        S.append((-1+np.sqrt(1+4*SS))/2)
    
    ms = [round(elem,2) for elem in ms]
    S  = [round(elem,2) for elem in S]

    # Make sure ms is the same across all states:
    for I in range(nstate):
        if (np.abs(ms[I]-ms[0])>1e-8):
            raise Exception("Each state's ms should be same for state_interaction_SOC function")

    #If Ms=0 , CG coefficent vanish...
    wfn_ref_nelecas = method.ref_nelecas.copy()
    if ms[0] == 0:
        method.log.info("Apply S_plus due to Ms=0...")
        for I in range(nstate):
            if S[I] > 0 :
                wfn[I], Sp_ne = method.interface.apply_S_plus(wfn[I],method.ncas,method.ref_nelecas[I])
                # Upadate ref_nelecas
                wfn_ref_nelecas[I] = Sp_ne
                # Normalize the wfn
                wfn[I] = wfn[I]/(np.sqrt( S[I]**2 + S[I] ))
                # Upadate ms
                ms[I] = 1


    # Make sure that S is consistent with spin_mult
    for I in range(nstate):
        if (np.abs(2 * S[I] + 1  - float(method.spin_mult[I])) > 1e-6):
            raise Exception("Spin value and multiplicity are not consistent")

    # Calculate RDM_aabb
    rdm_aabb = method.make_rdm1s(wfn, wfn_ref_nelecas)

    en_soc, evec_soc = general_somf.state_interaction_soc(method.interface, method.e_tot, rdm_aabb, S, ms, method.soc, method.verbose)

    method.e_tot = en_soc
    method.h_evec_soc = evec_soc

    #Calculate SOC e_corr respect with CASSCF energy
    e_ref_spinstate = []
    for i in range(nstate):
        n = int(S[i]*2 + 1)
        for j in range(n):
            e_ref_spinstate.append(method.e_ref[i])
    
    e_corr_soc = en_soc - e_ref_spinstate

    return en_soc, e_corr_soc
    

def transform_rdm1(method, rdm_sf, L = None, R = None, type = 'all'):

    evec_soc = method.h_evec_soc
    nstate = len(method.spin_mult)
    nstate_total = method.h_evec_soc.shape[0]
    nmo = method.nmo

    L_states = 0
    R_states = 0
    L_list = None
    R_list = None

    if L is None:
        L_states = nstate_total
        L_list = np.arange(L_states)
    elif isinstance(L, int):
        L_list = np.array([L])
        if L > nstate_total:
            raise ValueError(f"Invalid indices: L={L}. "f"Maximum allowed index is {n_micro_states - 1}.")
    else:
         raise ValueError(f"Value L={L} not supported")

    if R is None:
        R_states = nstate_total
        R_list = np.arange(R_states)
    elif isinstance(R, int):
        R_list = np.array([R])
        if R > nstate_total:
            raise ValueError(f"Invalid indices: R={R}. "f"Maximum allowed index is {n_micro_states - 1}.")
    else:
        raise ValueError(f"Value R={R} not supported")

    # Calculate indexing array
    I_total = []
    S_total = []
    ms_total = []
    for i in range(nstate):
        n = method.spin_mult[i]
        s = (n-1)/2
        for j in range(n):
            I_total.append(i)
            S_total.append(s)
            m = s-j
            ms_total.append(m)

    rdm_mo = np.zeros((nstate_total, nstate_total, nmo, nmo),dtype='complex')
    for I in range(nstate_total):
        for J in  range(nstate_total):
            if (np.abs(S_total[I]-S_total[J])<1e-8) and (np.abs(ms_total[I]-ms_total[J])<1e-8):
                rdm_mo[I,J] = rdm_sf[I_total[I], I_total[J]]
    rdm_all = np.einsum('ai,ibIJ,bj->ajIJ',np.conj(evec_soc).T , rdm_mo , evec_soc)

    # Initial rdm array
    rdm_final = np.zeros((L_list.shape[0], R_list.shape[0], nmo, nmo, ),dtype='complex')

    # Loop structure
    for ind_I, I in enumerate(L_list):
        for ind_J, J in enumerate(R_list):
            rdm_final[ind_I, ind_J] = rdm_all[I, J]

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


def compute_magnetic_properties(method, rdm_sf):

    nstate = len(method.spin_mult)

    S = []
    for i in range(nstate):
        S.append(float((method.spin_mult[i] - 1) / 2))

    method.g_factor, g_evec = magnetic.gtensor(method.interface, method.h_evec_soc, rdm_sf, S, target_state = method.target_state, origin_type=method.origin_type)

