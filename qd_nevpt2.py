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
# Authors: Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#          James D. Serna <jserna456@gmail.com>

import numpy as np
from functools import reduce

import prism.lib.logger as logger
import prism.lib.tools as tools
import prism.nevpt_amplitudes as nevpt_amplitudes

def compute_energy(nevpt, e_diag, t1, t1_0):

    # Einsum definition from kernel
    einsum = nevpt.interface.einsum
    einsum_type = nevpt.interface.einsum_type

    ncore = nevpt.ncore - nevpt.nfrozen
    ncas = nevpt.ncas
    nelecas = nevpt.ref_nelecas
    nextern = nevpt.nextern

    h_eff = np.diag(e_diag)
    dim = h_eff.shape[0]

    t1_ccee = t1_0

    ## One-electron integrals
    h_ca = nevpt.h1eff.ca
    h_ce = nevpt.h1eff.ce
    h_ae = nevpt.h1eff.ae

    ## Two-electron integrals
    v_cace = nevpt.v2e.cace
    v_caca = nevpt.v2e.caca
    v_ceaa = nevpt.v2e.ceaa
    v_caae = nevpt.v2e.caae
    v_caaa = nevpt.v2e.caaa
    v_aaae = nevpt.v2e.aaae

    # Compute the effective Hamiltonian matrix elements
    for I in range(dim):
        for J in range(I):
            # Compute transition density matrices
            trdm_ca, trdm_ccaa, trdm_cccaaa = nevpt.interface.compute_rdm123(nevpt.ref_wfn[I], nevpt.ref_wfn[J], nevpt.ref_nelecas[I])

            # 0.5 * < Psi_I | V * T | Psi_J >
            t1_caea = t1[J].caea
            t1_caae = t1[J].caae
            t1_caaa = t1[J].caaa
            t1_aaae = t1[J].aaae
            t1_ccae = t1[J].ccae
            t1_ccaa = t1[J].ccaa

            H_IJ  = einsum('ia,ixay,yx', h_ce, t1_caea, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ia,ixya,yx', h_ce, t1_caae, trdm_ca, optimize = einsum_type)
            H_IJ += einsum('ix,iyxz,zy', h_ca, t1_caaa, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ix,iyzw,zwxy', h_ca, t1_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ix,iyzx,zy', h_ca, t1_caaa, trdm_ca, optimize = einsum_type)
            H_IJ += 1/2 * einsum('xa,yzwa,xwzy', h_ae, t1_aaae, trdm_ccaa, optimize = einsum_type)
            H_IJ -= einsum('ijxa,iyja,xy', t1_ccae, v_cace, trdm_ca, optimize = einsum_type)
            H_IJ += 1/2 * einsum('ijxa,jyia,xy', t1_ccae, v_cace, trdm_ca, optimize = einsum_type)
            H_IJ -= einsum('ijxy,ixjz,yz', t1_ccaa, v_caca, trdm_ca, optimize = einsum_type)
            H_IJ += 1/2 * einsum('ijxy,iyjz,xz', t1_ccaa, v_caca, trdm_ca, optimize = einsum_type)
            H_IJ += 1/4 * einsum('ijxy,izjw,xyzw', t1_ccaa, v_caca, trdm_ccaa, optimize = einsum_type)
            H_IJ += einsum('ixay,iazw,yzxw', t1_caea, v_ceaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += einsum('ixay,iazy,zx', t1_caea, v_ceaa, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixay,iyza,zx', t1_caea, v_caae, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixay,izwa,ywxz', t1_caea, v_caae, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixya,iazw,yzxw', t1_caae, v_ceaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixya,iazy,zx', t1_caae, v_ceaa, trdm_ca, optimize = einsum_type)
            H_IJ += einsum('ixya,iyza,zx', t1_caae, v_caae, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixya,izwa,ywzx', t1_caae, v_caae, trdm_ccaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,yzuvwx', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,yzuvxw', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,yzuwvx', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/3 * einsum('ixyz,iwuv,yzuwxv', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,yzuxvw', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,yzuxwv', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixyz,iwuy,zuxw', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixyz,iwuz,yuwx', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += einsum('ixyz,iywu,zwxu', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += einsum('ixyz,iywz,wx', t1_caaa, v_caaa, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixyz,izwu,ywxu', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixyz,izwy,wx', t1_caaa, v_caaa, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,zvwuxy', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,zvwuyx', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,zvwxuy', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/3 * einsum('xyza,wuva,zvwxyu', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,zvwyux', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,zvwyxu', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/2 * einsum('xyza,wzua,wuxy', t1_aaae, v_aaae, trdm_ccaa, optimize = einsum_type)

            if ncore > 0 and nextern > 0 and ncas > 0:
                chunks = tools.calculate_chunks(nevpt, nextern, [ncore, ncas, nextern], ntensors = 3)
                for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                    cput1 = (logger.process_clock(), logger.perf_counter())
                    nevpt.log.debug("t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                    t1_caee = t1[J].caee[:,:,s_chunk:f_chunk,:]
                    v_ceae = nevpt.v2e.ceae[:,s_chunk:f_chunk,:,:]

                    H_IJ += einsum('ixab,iayb,yx', t1_caee, v_ceae, trdm_ca, optimize = einsum_type)

                    v_ceae = nevpt.v2e.ceae[:,:,:,s_chunk:f_chunk]

                    H_IJ -= 1/2 * einsum('ixab,ibya,yx', t1_caee, v_ceae, trdm_ca, optimize = einsum_type)

                    nevpt.log.timer_debug("contracting t1.xaee", *cput1)
                    del (t1_caee, v_ceae)

            if nextern > 0 and ncas > 0:
                chunks = tools.calculate_chunks(nevpt, nextern, [ncas, ncas, nextern], ntensors = 3)
                for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                    cput1 = (logger.process_clock(), logger.perf_counter())
                    nevpt.log.debug("t1.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                    t1_aaee = t1[J].aaee[:,:,s_chunk:f_chunk,:]
                    v_aeae = nevpt.v2e.aeae[:,s_chunk:f_chunk,:,:]

                    H_IJ += 1/4 * einsum('xyab,zawb,zwxy', t1_aaee, v_aeae, trdm_ccaa, optimize = einsum_type)

                    nevpt.log.timer_debug("contracting t1.aeae", *cput1)
                    del (t1_aaee, v_aeae)

            t1_caea = t1[I].caea
            t1_caae = t1[I].caae
            t1_caaa = t1[I].caaa
            t1_aaae = t1[I].aaae
            t1_ccae = t1[I].ccae
            t1_ccaa = t1[I].ccaa

            # 0.5 * < Psi_I | T+ * V | Psi_J >
            H_IJ += einsum('ia,ixay,xy', h_ce, t1_caea, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ia,ixya,xy', h_ce, t1_caae, trdm_ca, optimize = einsum_type)
            H_IJ += einsum('ix,iyxz,yz', h_ca, t1_caaa, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ix,iyzw,xyzw', h_ca, t1_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ix,iyzx,yz', h_ca, t1_caaa, trdm_ca, optimize = einsum_type)
            H_IJ += 1/2 * einsum('xa,yzwa,zyxw', h_ae, t1_aaae, trdm_ccaa, optimize = einsum_type)
            H_IJ -= einsum('ijxa,iyja,yx', t1_ccae, v_cace, trdm_ca, optimize = einsum_type)
            H_IJ += 1/2 * einsum('ijxa,jyia,yx', t1_ccae, v_cace, trdm_ca, optimize = einsum_type)
            H_IJ -= einsum('ijxy,ixjz,zy', t1_ccaa, v_caca, trdm_ca, optimize = einsum_type)
            H_IJ += 1/2 * einsum('ijxy,iyjz,zx', t1_ccaa, v_caca, trdm_ca, optimize = einsum_type)
            H_IJ += 1/4 * einsum('ijxy,izjw,zwxy', t1_ccaa, v_caca, trdm_ccaa, optimize = einsum_type)
            H_IJ += einsum('ixay,iazw,xwyz', t1_caea, v_ceaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += einsum('ixay,iazy,xz', t1_caea, v_ceaa, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixay,iyza,xz', t1_caea, v_caae, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixay,izwa,xzyw', t1_caea, v_caae, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixya,iazw,xwyz', t1_caae, v_ceaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixya,iazy,xz', t1_caae, v_ceaa, trdm_ca, optimize = einsum_type)
            H_IJ += einsum('ixya,iyza,xz', t1_caae, v_caae, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixya,izwa,xzwy', t1_caae, v_caae, trdm_ccaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvuyz', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvuzy', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvyuz', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvyzu', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvzuy', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/3 * einsum('ixyz,iwuv,xwvzyu', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixyz,iwuy,xwzu', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixyz,iwuz,xwuy', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += einsum('ixyz,iywu,xuzw', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += einsum('ixyz,iywz,xw', t1_caaa, v_caaa, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixyz,izwu,xuyw', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixyz,izwy,xw', t1_caaa, v_caaa, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,yxuvwz', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/3 * einsum('xyza,wuva,yxuvzw', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,yxuwvz', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,yxuwzv', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,yxuzvw', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,yxuzwv', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/2 * einsum('xyza,wzua,xywu', t1_aaae, v_aaae, trdm_ccaa, optimize = einsum_type)

            if ncore > 0 and nextern > 0 and ncas > 0:
                chunks = tools.calculate_chunks(nevpt, nextern, [ncore, ncas, nextern], ntensors = 3)
                for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                    cput1 = (logger.process_clock(), logger.perf_counter())
                    nevpt.log.debug("t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                    t1_caee = t1[I].caee[:,:,s_chunk:f_chunk,:]
                    v_ceae = nevpt.v2e.ceae[:,s_chunk:f_chunk,:,:]

                    H_IJ += einsum('ixab,iayb,xy', t1_caee, v_ceae, trdm_ca, optimize = einsum_type)

                    v_ceae = nevpt.v2e.ceae[:,:,:,s_chunk:f_chunk]

                    H_IJ -= 1/2 * einsum('ixab,ibya,xy', t1_caee, v_ceae, trdm_ca, optimize = einsum_type)

                    del (t1_caee, v_ceae)

            if nextern > 0 and ncas > 0:
                chunks = tools.calculate_chunks(nevpt, nextern, [ncas, ncas, nextern], ntensors = 3)
                for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                    cput1 = (logger.process_clock(), logger.perf_counter())
                    nevpt.log.debug("t1.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                    t1_aaee = t1[I].aaee[:,:,s_chunk:f_chunk,:]
                    v_aeae = nevpt.v2e.aeae[:,s_chunk:f_chunk,:,:]

                    H_IJ += 1/4 * einsum('xyab,zawb,xyzw', t1_aaee, v_aeae, trdm_ccaa, optimize = einsum_type)

                    nevpt.log.timer_debug("contracting t1.aeae", *cput1)
                    del (t1_aaee, v_aeae)

            h_eff[I, J] = H_IJ
            h_eff[J, I] = H_IJ

    h_eval, h_evec = np.linalg.eigh(h_eff)

    return h_eval, h_evec


def osc_strength(nevpt, en, evec, t1, t1_0, gs_index = 0):

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

    # Looping over CAS States
    for state in range(gs_index + 1, n_micro_states):
        # Reset final transformed RDM
        rdm_qd = np.zeros((nmo, nmo))

        # Looping over states I,J
        for I in range(n_micro_states):
            for J in range(n_micro_states):
                rdm_mo = np.zeros((nmo, nmo))  # Reset RDM in MO Basis   
                trdm_ca = nevpt.interface.compute_rdm1(nevpt.ref_wfn[I], nevpt.ref_wfn[J], nevpt.ref_nelecas[I])
                rdm_mo[ncore:ncore + ncas ,ncore:ncore + ncas] = trdm_ca

                if I == J:
                    rdm_mo[:ncore, :ncore] = 2 * np.eye(nevpt.ncore)
                    rdm_qd += np.conj(evec)[I, state] * rdm_mo * evec[J, gs_index]
                else:
                    rdm_qd += np.conj(evec)[I, state] * rdm_mo * evec[J, gs_index]
        
        rdm_qd_corr = compute_corr_1rdm(nevpt, evec, t1, t1_0, gs_index, state)

        # Create Dipole Moment Operator with RDM
        dip_evec_x = np.einsum('pq,pq', dip_mom_mo[0], rdm_qd)
        dip_evec_y = np.einsum('pq,pq', dip_mom_mo[1], rdm_qd)
        dip_evec_z = np.einsum('pq,pq', dip_mom_mo[2], rdm_qd)
        
        osc_x = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_x)*dip_evec_x)
        osc_y = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_y)*dip_evec_y)
        osc_z = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_z)*dip_evec_z)
        
        # Add Dipole Moment Components
        osc_total.append((osc_x + osc_y + osc_z).real)
        
        ### 
        # Create Dipole Moment Operator with Correlated RDM
        dip_evec_x = np.einsum('pq,pq', dip_mom_mo[0], rdm_qd_corr)
        dip_evec_y = np.einsum('pq,pq', dip_mom_mo[1], rdm_qd_corr)
        dip_evec_z = np.einsum('pq,pq', dip_mom_mo[2], rdm_qd_corr)
 
        osc_x = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_x)*dip_evec_x)
        osc_y = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_y)*dip_evec_y)
        osc_z = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_z)*dip_evec_z)
        
        # Add Dipole Moment Components
        osc_total_corr.append((osc_x + osc_y + osc_z).real)
    
    return osc_total, osc_total_corr


def determine_spin_mult(nevpt, evec):

    spin_mult_old = nevpt.ref_wfn_spin_mult
    spin_mult_new = []

    for root in range(evec.shape[1]):
        index = np.argmax(np.abs(evec[:, root]))
        spin_mult_new.append(spin_mult_old[index])

    return spin_mult_new

def compute_corr_1rdm(nevpt, evec, t1, t1_0, m = None, n = None):
    
    if m is None:
        m = 0
    if n is None:
        n = 0
 
    ncore = nevpt.ncore 
    n_micro_states = sum(nevpt.ref_wfn_deg)
    ncas = nevpt.ncas
    mo_coeff = nevpt.mo
    einsum = nevpt.interface.einsum
    einsum_type = nevpt.interface.einsum_type
    nmo = nevpt.nmo
    ncas = nevpt.ncas
    nextern = nevpt.nextern

    rdm_qd = np.zeros((nmo, nmo))

    # Looping over states I,J
    for I in range(n_micro_states):
        L_t1_caea = t1[I].caea
        L_t1_caae = t1[I].caae
        L_t1_caaa = t1[I].caaa
        L_t1_aaae = t1[I].aaae
        L_t1_ccae = t1[I].ccae
        L_t1_ccaa = t1[I].ccaa
        L_t1_caee = t1[I].caee 
        L_t1_aaee = t1[I].aaee
        L_t1_aaea = t1[I].aaae.transpose(1,0,3,2)

        for J in range(n_micro_states): 
            R_t1_caea = t1[J].caea
            R_t1_caae = t1[J].caae
            R_t1_caaa = t1[J].caaa
            R_t1_aaae = t1[J].aaae
            R_t1_ccae = t1[J].ccae
            R_t1_ccaa = t1[J].ccaa
            R_t1_caee = t1[J].caee 
            R_t1_aaee = t1[J].aaee
            R_t1_aaea = t1[J].aaae.transpose(0,1,3,2)
            
            t1_ccee = t1_0
            
            rdm_mo = np.zeros((nmo, nmo)) 
            trdm_ca, trdm_ccaa, trdm_cccaaa, trdm_ccccaaaa = nevpt.interface.compute_rdm1234(nevpt.ref_wfn[I], nevpt.ref_wfn[J], nevpt.ref_nelecas[I])
            rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] = trdm_ca
            
            if I == J:
    
                #uncorrelated diagonal terms
                rdm_mo[:ncore, :ncore] = 2 * np.eye(nevpt.ncore)
                 
                ## core-core
                # CORE-CORE #
                rdm_mo[:ncore, :ncore] -= 4 * einsum('Iiab,Jiab->IJ', t1_ccee, t1_ccee, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += 2 * einsum('Iiab,Jiba->IJ', t1_ccee, t1_ccee, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= 4 * einsum('Iixa,Jixa->IJ', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += 2 * einsum('Iixa,iJxa->IJ', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= 4 * einsum('Iixy,Jixy->IJ', L_t1_ccaa, R_t1_ccaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += 2 * einsum('Iixy,Jiyx->IJ', L_t1_ccaa, R_t1_ccaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += 2 * einsum('iIxa,Jixa->IJ', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= 4 * einsum('iIxa,iJxa->IJ', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += 2 * einsum('Iixa,Jiya,xy->IJ', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= einsum('Iixa,iJya,xy->IJ', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += 2 * einsum('Iixy,Jixz,yz->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= einsum('Iixy,Jiyz,xz->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= einsum('Iixy,Jizw,xyzw->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= einsum('Iixy,Jizx,yz->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += 2 * einsum('Iixy,Jizy,xz->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= 2 * einsum('Ixab,Jyab,xy->IJ', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += einsum('Ixab,Jyba,xy->IJ', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= 2 * einsum('Ixay,Jzaw,xwyz->IJ', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= 2 * einsum('Ixay,Jzay,xz->IJ', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += einsum('Ixay,Jzwa,xwyz->IJ', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += einsum('Ixay,Jzya,xz->IJ', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += einsum('Ixya,Jzaw,xwyz->IJ', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += einsum('Ixya,Jzay,xz->IJ', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += einsum('Ixya,Jzwa,xwzy->IJ', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= 2 * einsum('Ixya,Jzya,xz->IJ', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvwyz->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvwzy->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvywz->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvyzw->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvzwy->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += 2/3 * einsum('Ixyz,Jwuv,xuvzyw->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += einsum('Ixyz,Jwuy,xuzw->IJ', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += einsum('Ixyz,Jwuz,xuwy->IJ', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= 2 * einsum('Ixyz,Jwyu,xuzw->IJ', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= 2 * einsum('Ixyz,Jwyz,xw->IJ', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += einsum('Ixyz,Jwzu,xuyw->IJ', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += einsum('Ixyz,Jwzy,xw->IJ', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] -= einsum('iIxa,Jiya,xy->IJ', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, :ncore] += 2 * einsum('iIxa,iJya,xy->IJ', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                
                # ACT-ACT # 
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 4 * einsum('ijXa,ijYa->XY', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 2 * einsum('ijXa,jiYa->XY', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 4 * einsum('ijXx,ijYx->XY', L_t1_ccaa, R_t1_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 2 * einsum('ijXx,jiYx->XY', L_t1_ccaa, R_t1_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Xxab,yzab,Yxyz->XY', L_t1_aaee, R_t1_aaee, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Xxya,zwua,Yxuywz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Xxya,zwya,Yxzw->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Yxab,yzab,Xxyz->XY', L_t1_aaee, R_t1_aaee, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Yxya,zwua,Xxuywz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Yxya,zwya,Xxzw->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXab,ixab,Yx->XY', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXab,ixba,Yx->XY', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXax,iyax,Yy->XY', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXax,iyaz,Yzxy->XY', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXax,iyxa,Yy->XY', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXax,iyza,Yzxy->XY', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxa,iyax,Yy->XY', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxa,iyaz,Yzxy->XY', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXxa,iyxa,Yy->XY', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxa,iyza,Yzyx->XY', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuxyz->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuxzy->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('iXxy,izwu,Ywuyxz->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuyzx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuzxy->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuzyx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izwx,Ywyz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izwy,Ywzx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXxy,izxw,Ywyz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXxy,izxy,Yz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izyw,Ywxz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izyx,Yz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYab,ixab,Xx->XY', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYab,ixba,Xx->XY', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYax,iyax,Xy->XY', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYax,iyaz,Xzxy->XY', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYax,iyxa,Xy->XY', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYax,iyza,Xzxy->XY', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxa,iyax,Xy->XY', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxa,iyaz,Xzxy->XY', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYxa,iyxa,Xy->XY', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxa,iyza,Xzyx->XY', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuxyz->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuxzy->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('iYxy,izwu,Xwuyxz->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuyzx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuzxy->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuzyx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izwx,Xwyz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izwy,Xwzx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYxy,izxw,Xwyz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYxy,izxy,Xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izyw,Xwxz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izyx,Xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijXa,ijxa,Yx->XY', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijXa,jixa,Yx->XY', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 2 * einsum('ijXx,ijYy,xy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijXx,ijxy,Yy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijXx,ijyz,Yxyz->XY', L_t1_ccaa, R_t1_ccaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ijXx,jiYy,xy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijXx,jixy,Yy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijYa,ijxa,Xx->XY', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijYa,jixa,Xx->XY', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijYx,ijxy,Xy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijYx,ijyz,Xxyz->XY', L_t1_ccaa, R_t1_ccaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijYx,jixy,Xy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixXa,iyYa,xy->XY', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixXa,iyaY,xy->XY', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXa,iyaz,Yyxz->XY', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXa,iyza,Yyzx->XY', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixXy,izYw,yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixXy,izYy,xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixXy,izwY,yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzuwx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzuxw->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzwux->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('ixXy,izwu,Yyzwxu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzxuw->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzxwu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXy,izwy,Yzwx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixXy,izyY,xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXy,izyw,Yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYa,iyaz,Xyxz->XY', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYa,iyza,Xyzx->XY', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzuwx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzuxw->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzwux->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('ixYy,izwu,Xyzwxu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzxuw->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzxwu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYy,izwy,Xzwx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYy,izyw,Xzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixaX,iyYa,xy->XY', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixaX,iyaY,xy->XY', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixaX,iyaz,Yyxz->XY', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixaX,iyza,Yyxz->XY', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixaY,iyaz,Xyxz->XY', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixaY,iyza,Xyxz->XY', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixyX,izYw,yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixyX,izYy,xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixyX,izwY,yzwx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyX,izwu,Yyzxwu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyX,izwy,Yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixyX,izyY,xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixyX,izyw,Yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyY,izwu,Xyzxwu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyY,izwy,Xzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixyY,izyw,Xzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('xXya,zwua,Yxuwyz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuwzy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuywz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuyzw->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuzwy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuzyw->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('xXya,zwya,Yxwz->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('xYya,zwua,Xxuwyz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuwzy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuywz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuyzw->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuzwy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuzyw->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('xYya,zwya,Xxwz->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('xyXa,zwYa,xyzw->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzuxy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzuyx->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzxuy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('xyXa,zwua,Ywzxyu->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzyux->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzyxu->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzuxy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzuyx->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzxuy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('xyYa,zwua,Xwzxyu->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzyux->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzyxu->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                
                # EXT-EXT #
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 4 * einsum('ijAa,ijBa->AB', t1_ccee, t1_ccee, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 2 * einsum('ijAa,jiBa->AB', t1_ccee, t1_ccee, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 4 * einsum('ijxA,ijxB->AB', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 2 * einsum('ijxA,jixB->AB', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 2 * einsum('ijxA,ijyB,xy->AB', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += einsum('ijxA,jiyB,xy->AB', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixAa,iyBa,xy->AB', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixAa,iyaB,xy->AB', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixAy,izBw,yzxw->AB', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixAy,izBy,xz->AB', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixAy,izwB,yzxw->AB', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixAy,izyB,xz->AB', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixaA,iyBa,xy->AB', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixaA,iyaB,xy->AB', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixyA,izBw,yzxw->AB', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixyA,izBy,xz->AB', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixyA,izwB,yzwx->AB', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixyA,izyB,xz->AB', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += einsum('xyAa,zwBa,xyzw->AB', L_t1_aaee, R_t1_aaee, trdm_ccaa, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwvxy->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwvyx->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwxvy->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2/3 * einsum('xyzA,wuvB,zuwxyv->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwyvx->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwyxv->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += einsum('xyzA,wuzB,yxuw->AB', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                
                rdm_qd += np.conj(evec)[I, n] * rdm_mo * evec[J, m]
                
            else:
                # OFF-DIAGS #
                # COR-ACT #
                rdm_mo[:ncore, ncore:ncore + ncas] += einsum('IxXy,yx->IX', R_t1_caaa, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, ncore:ncore + ncas] -= 1/2 * einsum('IxyX,yx->IX', R_t1_caaa, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, ncore:ncore + ncas] -= 1/2 * einsum('Ixyz,yzXx->IX', R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                
                # ACT-COR #
                rdm_mo[ncore:ncore + ncas, :ncore] += einsum('IxXy,xy->XI', L_t1_caaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, :ncore] -= 1/2 * einsum('IxyX,xy->XI', L_t1_caaa, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore:ncore + ncas, :ncore] -= 1/2 * einsum('Ixyz,Xxyz->XI', L_t1_caaa, trdm_ccaa, optimize = einsum_type)
                
                # COR-EXT #
                rdm_mo[:ncore, ncore + ncas:ncore + ncas + nextern] += einsum('IxAy,yx->IA', R_t1_caea, trdm_ca, optimize = einsum_type)
                rdm_mo[:ncore, ncore + ncas:ncore + ncas + nextern] -= 1/2 * einsum('IxyA,yx->IA', R_t1_caae, trdm_ca, optimize = einsum_type)
                
                # EXT-COR #
                rdm_mo[ncore + ncas:ncore + ncas + nextern, :ncore] += einsum('IxAy,xy->AI', L_t1_caea, trdm_ca, optimize = einsum_type)
                rdm_mo[ncore + ncas:ncore + ncas + nextern, :ncore] -= 1/2 * einsum('IxyA,xy->AI', L_t1_caae, trdm_ca, optimize = einsum_type)
                
                # ACT-EXT #
                rdm_mo[ncore:ncore + ncas, ncore + ncas:ncore + ncas + nextern] += 1/2 * einsum('xyzA,Xzyx->XA', R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                
                # EXT-ACT #
                rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore:ncore + ncas] += 1/2 * einsum('xyzA,Xzyx->AX', L_t1_aaae, trdm_ccaa, optimize = einsum_type)
                
                rdm_qd += np.conj(evec)[I, n] * rdm_mo * evec[J, m]

    return rdm_qd    


