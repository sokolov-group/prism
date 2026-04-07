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
# Authors: Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#          James D. Serna <jserna456@gmail.com>
#

import numpy as np 

def get_pe_corrections(method, state = 0, rdms = None):
    '''
    Compute perturbative energy corrections from
    polarizable embedding (current support)
    ptSS = perturbative state-specific type
    ptLR = perturbative linear-response type
    '''
    n_micro_states = sum(method.ref_wfn_deg)
    nmo = method.nmo
    ncas = method.ncas
    ### TODO: add type flag for 'all', 'gs-only' or something ###
     
    # Compute all 1rdm
    if rdms is None:
        rdms = method.make_rdm1(type = 'all')
         
    # Empty lists for energy corrections
    ptss = []
    ptlr = []
    
    # Warnings
    if method.pe is None:
        raise ValueError("Polarizable embedding (pe) object must be defined before calling this method.")
    
    for m in range(n_micro_states):
        for n in range(n_micro_states):
            if m == state and n == state:
                gs_rdm_mo = rdms[state, state]
                gs_rdm_ao = np.dot(method.mo, np.dot(gs_rdm_mo, method.mo.T))
            
            elif m == n and m > state and n > state:
                es_rdm_mo = rdms[m,m]
                es_rdm_ao = np.dot(method.mo, np.dot(es_rdm_mo, method.mo.T))

                dif = np.subtract(es_rdm_ao, gs_rdm_ao)
                
                e_ptss, v = method.pe.kernel(dm = dif, elec_only = True)
                
                ptss.append(e_ptss * method.interface.hartree_to_ev)
        
            elif m == state and n > state:
                tr_rdm_mo = rdms[m,n]
                tr_rdm_ao = np.dot(method.mo, np.dot(tr_rdm_mo, method.mo.T))
                
                e_ptlr, v = method.pe.kernel(dm = tr_rdm_ao, elec_only = True)
                
                ptlr.append(e_ptlr * method.interface.hartree_to_ev * 2)
    
    return ptss, ptlr