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


def osc_strength(nevpt, en, evec, gs_index = 0):

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

        # Create Dipole Moment Operator with RDM
        dip_evec_x = np.einsum('pq,pq', dip_mom_mo[0], rdm_qd)
        dip_evec_y = np.einsum('pq,pq', dip_mom_mo[1], rdm_qd)
        dip_evec_z = np.einsum('pq,pq', dip_mom_mo[2], rdm_qd)
 
        osc_x = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_x)*dip_evec_x)
        osc_y = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_y)*dip_evec_y)
        osc_z = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_z)*dip_evec_z)

        # Add Dipole Moment Components
        osc_total.append((osc_x + osc_y + osc_z).real)

    return osc_total


def determine_spin_mult(nevpt, evec):

    spin_mult_old = nevpt.ref_wfn_spin_mult
    spin_mult_new = []

    for root in range(evec.shape[1]):
        index = np.argmax(np.abs(evec[:, root]))
        spin_mult_new.append(spin_mult_old[index])

    return spin_mult_new

def Initialize_SOC(method):
    print("\n \n \n Prisim_beta")
    print("Consider spin-orbit coupling effect...")
    ref_wfn =method.ref_wfn  
    ncas = method.ncas 
    ref_nelecas = method.ref_nelecas 
    ref_wfn_spin_mult = method.ref_wfn_spin_mult
    S =  [round((spin_mult-1)/2,2) for spin_mult in ref_wfn_spin_mult]     
    nstate = len(ref_wfn)

    if method.evec_qdnevpt2 is None:
        print("It is NEVPT2")
        method.evec_qdnevpt2 = np.diag(np.ones(len(ref_wfn)))
    else:
        print("It is QDNEVPT2")

    ##test by using CASSCF######
    #print("This is SOC-CASSCF")
    #method.evec_qdnevpt2 = np.diag(np.ones(len(ref_wfn)))
    #method.en_qdnevpt2 = method.e_ref_cas 
    ##test by using CASSCF######

    evec = method.evec_qdnevpt2
    en = method.en_qdnevpt2

    #Get target state psi (wfn)
    wfn = np.einsum('ij,iab->jab',evec,ref_wfn)
    
    #Get ms 
    ms = []
    for I in range(nstate):
        sz = method.interface.apply_S_z(wfn[I],ncas,ref_nelecas[I])
        ms.append(np.dot(wfn[I].ravel(), sz.ravel()))

    ms = [round(elem,2) for elem in ms]

    # Generate all ms states:
    print("Generate all ms states...")
    e_cas_roots_spinstate = []
    wfn_spinstate = []
    ms_spinstate = []
    nele_spinstate = []
    E_spinstate = []

    for root in range(nstate):
        n_plus = int(S[root] - ms[root])
        n_minus =  int(ms[root] + S[root])
        wfn_S = []
        ms_S = []
        nele_S = []
        E_S = []

        wfn_S.append(wfn[root])
        ms_S.append(ms[root])
        nele_S.append(ref_nelecas[root])
        E_S.append(en[root])

        spin_wf_plus = wfn[root]
        spin_wf_minus = wfn[root]
        spin_nelec_plus = ref_nelecas[root]
        spin_nelec_minus = ref_nelecas[root]

        #Operate plus
        for I in range(n_plus):
            # Apply spin operators for finding ms values
            sz_plus = method.interface.apply_S_z(spin_wf_plus, ncas, spin_nelec_plus)
            msz_plus = np.dot(spin_wf_plus.ravel(), sz_plus.ravel())
            # Apply Raising operator:
            spin_wf_plus, spin_nelec_plus = method.interface.apply_S_plus(spin_wf_plus, ncas, spin_nelec_plus)
            # Normalize the wfn
            spin_wf_plus = spin_wf_plus/(np.sqrt(S[root]*(S[root]+1) - msz_plus*(msz_plus + 1)))
            # Add spin states to list
            wfn_S.append(spin_wf_plus)
            ms_S.append(msz_plus+1)
            E_S.append(en[root])
            nele_S.append(spin_nelec_plus)

        #Operate minus
        for I in range(n_minus):
            # Apply spin operators for finding ms values
            sz_minus = method.interface.apply_S_z(spin_wf_minus, ncas, spin_nelec_minus)
            msz_minus = np.dot(spin_wf_minus.ravel(), sz_minus.ravel()) 
            # Apply lowering operator:
            spin_wf_minus, spin_nelec_minus = method.interface.apply_S_minus(spin_wf_minus, ncas, spin_nelec_minus)
            # Normalize the wfn
            spin_wf_minus = spin_wf_minus/(np.sqrt(S[root]*(S[root]+1) - msz_minus*(msz_minus - 1)))
            # Add spin states to list
            wfn_S.append(spin_wf_minus)
            ms_S.append(msz_minus-1)
            E_S.append(en[root])
            nele_S.append(spin_nelec_minus)

        ms_S, nele_S,wfn_S,E_S = zip(*sorted(zip(ms_S, nele_S,wfn_S,E_S), reverse=True))
        ms_S= [round(elem,2) for elem in ms_S]
        
        #saving...
        for i in range(len(wfn_S)):
            ms_spinstate.append(ms_S[i])
            nele_spinstate.append(nele_S[i])
            wfn_spinstate.append(wfn_S[i])
            E_spinstate.append(E_S[i])

    nstate_spinstate = len(ms_spinstate)
    ncas_so = ncas *2

    #calculate rdm_so_ca:
    print("calculate rdm...")
    rdm_so_ca = np.zeros((nstate_spinstate, ncas_so, ncas_so))    
    for ind, wfn in enumerate(wfn_spinstate):
        rdm_so_ca[ind] = compute_rdm_ca_so(method.interface, wfn_spinstate[ind], wfn_spinstate[ind], nele_spinstate[ind], nele_spinstate[ind])
    
    # Compute transition RDMs
    print("calculate trdm...")
    rdm_so_tca = compute_rdm_tcat_so(method.interface, wfn_spinstate, nele_spinstate, offset = -1)

    method.ncasci = nstate_spinstate
    method.rdm_so.ca = rdm_so_ca
    method.rdm_so.tca = rdm_so_tca
       
    print("construct HSOC matrix...")
    #method.h_soc = bpsomf(method)
    print("Use general_somf...")
    from prism import general_somf
    method.h_soc = general_somf.getSOC_integrals(method)
    
    #exit()
    HSOC = compute_h_somf1(method)
    H_sf = np.diag(E_spinstate).astype('complex')
    
    en_soc, evec_soc = np.linalg.eigh(HSOC+H_sf)

    print("\n Absolute energies in a.u |||| Excitation energies in a.u ||  eV ||  cm-1\n*****************************")
    for e in en_soc:
        print("%14.6f ||||  %14.6f  ||  %14.6f  ||   %8.2f"%((e), (e-en_soc[0]),((e-en_soc[0])*27.2114),((e-en_soc[0])*219474.63)))



    


 





#from rdms.py in prisim_beta
def compute_rdm_ca_so(interface, bra = None, ket = None, nelecas_bra = None, nelecas_ket = None):

    ncas = interface.ncas

    #if bra is None:
    #    bra = interface.wfn_casscf
    #if ket is None:
    #    ket = interface.wfn_casscf
    #if nelecas_bra is None:
    #    nelecas_bra = interface.nelecas
    #if nelecas_ket is None:
    #    nelecas_ket = interface.nelecas

    rdm = interface.compute_rdm_ca_si(bra, ket, nelecas_bra, nelecas_ket)

    rdm_so = np.zeros((2 * ncas, 2 * ncas))
    if rdm[0] is not None:
        rdm_so[::2,::2] = rdm[0].copy()
    if rdm[1] is not None:
        rdm_so[::2,1::2] = rdm[1].copy()
    if rdm[2] is not None:
        rdm_so[1::2,::2] = rdm[2].copy()
    if rdm[3] is not None:
        rdm_so[1::2,1::2] = rdm[3].copy()

    return rdm_so

#from rdms.py in prisim_beta
def compute_rdm_tcat_so(interface, wfns, nelecasci, offset = 0):

    if (not isinstance(wfns, list) or not isinstance(nelecasci, list)):
        raise Exception("Provided wavefunctions and electron counts need to be stored as lists!")

    if len(wfns) != len(nelecasci):
        raise Exception("Number of wavefunctions and electron counts does not match!")

    ncas = interface.ncas

    nroots = len(wfns)

    dim = (nroots + offset) * (nroots + 1 + offset) // 2

    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas))
    
    # Loop over unique combination of states
    for I in range(nroots):
        for J in range(I + 1 + offset):
            P = (I + offset) * (I + 1 + offset) // 2 + J
            rdm_so[P] = compute_rdm_ca_so(interface, wfns[I], wfns[J], nelecasci[I], nelecasci[J])
    
    return rdm_so


#from qd_nevpt_h_eff in prisim_beta 
def compute_h_somf1(qd_nevpt):

    # Print the parameters for SOC:
    print("Building BP-SOC Hamiltonian: Two-electron approximation: SOMF.......\n")
    #Define dimensions
    ncasci = qd_nevpt.ncasci
    ncas = qd_nevpt.ncas
    ncore = qd_nevpt.ncore
    #dim = len(qd_nevpt.wfn_casscf)

    #Import trdms and rdms for SOC in QDNEVPT2.
    rdm_ca_so = qd_nevpt.rdm_so.ca
    trdm_soc_ca = qd_nevpt.rdm_so.tca
    #Import the X2c/BP somf integrals:
    h_soc = np.zeros((3,ncas,ncas),dtype='complex')
    for comp in range(3):
        h_soc[comp] = qd_nevpt.h_soc[comp,ncore:ncore+ncas,ncore:ncore+ncas]

    # Evaluation of Matrix elements in hamiltonian:
    h1_plus = -(h_soc[0] + (1j*h_soc[1]))/np.sqrt(2)
    h1_minus = (h_soc[0] - (1j*h_soc[1]))/np.sqrt(2)
    h1_zero = h_soc[2]

    # Construct Matrix elements for one-electron part.
    H_SOC = np.zeros((ncasci, ncasci), dtype = complex)
    for I in range(ncasci):
        T_plus = -(rdm_ca_so[I, ::2, 1::2].copy())/np.sqrt(2)
        T_minus = (rdm_ca_so[I, 1::2, ::2].copy())/np.sqrt(2)
        T_0 = 0.5*(rdm_ca_so[I,::2,::2]-rdm_ca_so[I,1::2,1::2])
            
        H_SOC[I,I] += np.einsum('pq,pq', h1_zero, T_0)
        H_SOC[I,I] -= np.einsum('pq,pq', h1_plus, T_minus)
        H_SOC[I,I] -= np.einsum('pq,pq', h1_minus, T_plus)

        for J in range(I):
                
            if (I>J):
                P = (((I-1)*I)//2) + J
                T_plus = -(trdm_soc_ca[P, ::2, 1::2].copy())/np.sqrt(2)
                T_minus = (trdm_soc_ca[P, 1::2, ::2].copy())/np.sqrt(2)
                T_0 = 0.5*(trdm_soc_ca[P,::2,::2]-trdm_soc_ca[P,1::2,1::2])
                    
                H_SOC[I,J] += np.einsum('pq,pq', h1_zero, T_0)
                H_SOC[I,J] -= np.einsum('pq,pq', h1_plus, T_minus)
                H_SOC[I,J] -= np.einsum('pq,pq', h1_minus, T_plus)

                H_SOC[J,I] = np.conj(H_SOC[I,J]).T

    return H_SOC


def bpsomf(method):
    '''BP SOMF and full integrals given by:
        BP h1 =  0.5 alpha^2  W_sd 
        BP h2 =  (2g_soo + g_sso) ; refer to Majumder 2023'''

    from scipy import constants

    print("Get the BP-SO integrals in MO basis: BP-SOMF")
    nmo = method.nmo
    ncas = method.ncas
    ncore = method.ncore
    nextern = method.nextern
    nocc = method.nocc
    mf = method.interface.mf
    ncasci = method.ncasci
    if ncas > 0:
        if method.method == 'mr-adc(0)' or method.method == 'mr-adc(1)' or method.method == 'mr-adc(2)' or method.method == 'mr-adc(2)-x':
            rdm_ca_so = method.rdm_so.tcat
        else:
            rdm_ca_so = method.rdm_so.ca
    
    mf = method.interface.mf
    mo = method.mo
    mo_e = mo[:, method.nocc:].copy()
    mo_c = mo[:, :ncore].copy()
    mo_a = mo[:, ncore:ncore+ncas].copy()

    # 1e-SOC integrals in AO.
    h1_soc_ao = 0

    for atm in range(mf.mol.natm):
        mf.mol.set_rinv_origin(mf.mol.atom_coord(atm))
        h1_soc_ao += (mf.mol.intor('int1e_prinvxp_sph')*mf.mol.atom_charge(atm))
    # 1e-SOC integrals in MO.
    h1_soc = np.einsum('xpq,pi,qj->xij',h1_soc_ao,mo,mo)
    h1_soc = 0.5j*(h1_soc.astype('complex')*((constants.alpha)**2))
    
    # 2e-SOC integrals in MO representation:
    v_soc_aaaa = transform_2e_integrals_soc(mf.mol, mo_a,mo_a,mo_a,mo_a)
    v_soc_acac = transform_2e_integrals_soc(mf.mol, mo_a,mo_a,mo_c,mo_c)
    v_soc_caac = transform_2e_integrals_soc(mf.mol, mo_c,mo_a,mo_a,mo_c)
    v_soc_caca = transform_2e_integrals_soc(mf.mol, mo_c,mo_c,mo_a,mo_a)
    v_soc_acca = transform_2e_integrals_soc(mf.mol, mo_a,mo_c,mo_c,mo_a)
    v_soc_cccc = transform_2e_integrals_soc(mf.mol, mo_c,mo_c,mo_c,mo_c)
    v_soc_ecec = transform_2e_integrals_soc(mf.mol, mo_e,mo_e,mo_c,mo_c)
    v_soc_ceec = transform_2e_integrals_soc(mf.mol, mo_c,mo_e,mo_e,mo_c)
    v_soc_eaea = transform_2e_integrals_soc(mf.mol, mo_e,mo_e,mo_a,mo_a)
    v_soc_aeea = transform_2e_integrals_soc(mf.mol, mo_a,mo_e,mo_e,mo_a)
    v_soc_ccac = transform_2e_integrals_soc(mf.mol, mo_c,mo_a,mo_c,mo_c)
    v_soc_cacc = transform_2e_integrals_soc(mf.mol, mo_c,mo_c,mo_a,mo_c)
    v_soc_caaa = transform_2e_integrals_soc(mf.mol, mo_c,mo_a,mo_a,mo_a)
    v_soc_acaa = transform_2e_integrals_soc(mf.mol, mo_a,mo_a,mo_c,mo_a)
    v_soc_aaca = transform_2e_integrals_soc(mf.mol, mo_a,mo_c,mo_a,mo_a)
    v_soc_caec = transform_2e_integrals_soc(mf.mol, mo_c,mo_e,mo_a,mo_c)
    v_soc_ceac = transform_2e_integrals_soc(mf.mol, mo_c,mo_a,mo_e,mo_c)
    v_soc_aaea = transform_2e_integrals_soc(mf.mol, mo_a,mo_e,mo_a,mo_a)
    v_soc_aeaa = transform_2e_integrals_soc(mf.mol, mo_a,mo_a,mo_e,mo_a)
    v_soc_ecac = transform_2e_integrals_soc(mf.mol, mo_e,mo_a,mo_c,mo_c)
    v_soc_eaaa = transform_2e_integrals_soc(mf.mol, mo_e,mo_a,mo_a,mo_a)
    v_soc_ccec = transform_2e_integrals_soc(mf.mol, mo_c,mo_e,mo_c,mo_c)
    v_soc_cecc = transform_2e_integrals_soc(mf.mol, mo_c,mo_c,mo_e,mo_c)
    v_soc_caea = transform_2e_integrals_soc(mf.mol, mo_c,mo_e,mo_a,mo_a)
    v_soc_acea = transform_2e_integrals_soc(mf.mol, mo_a,mo_e,mo_c,mo_a)
    v_soc_aeca = transform_2e_integrals_soc(mf.mol, mo_a,mo_c,mo_e,mo_a)

    ##### Build density matrix:
    dm_c = 2*(np.diag(np.ones(ncore)))
    dm_a = np.zeros((ncas,ncas))
    if ncas > 0:
        if method.method == 'qd-nevpt2':
            for I in range(ncasci):
                dm_a += (1/ncasci)*(rdm_ca_so[I,::2,::2] + rdm_ca_so[I,1::2,1::2])
        else:
            for I in range(ncasci):
                P = (I*(I+1))//2 + I
                dm_a += (1/ncasci)*(rdm_ca_so[P,::2,::2] + rdm_ca_so[P,1::2,1::2])
    
    ##### Build SOMF effective integrals:
    h2_somf = np.zeros((3, nmo,nmo), dtype = 'complex')
    if ncas > 0:
        # h2-somf- aa terms
        h2_somf[:,ncore:ncore+ncas,ncore:ncore+ncas] += np.einsum('rs, xprqs->xpq', dm_c, v_soc_acac)         
        h2_somf[:,ncore:ncore+ncas,ncore:ncore+ncas] -= 1.5*np.einsum('rs, xspqr->xpq', dm_c, v_soc_caac)     
        h2_somf[:,ncore:ncore+ncas,ncore:ncore+ncas] += 1.5*np.einsum('rs, xsqpr->xpq', dm_c, v_soc_caac)     
        h2_somf[:,ncore:ncore+ncas,ncore:ncore+ncas] += np.einsum('rs, xprqs->xpq', dm_a, v_soc_aaaa)      
        h2_somf[:,ncore:ncore+ncas,ncore:ncore+ncas] -= 1.5*np.einsum('rs, xspqr->xpq', dm_a, v_soc_aaaa)
        h2_somf[:,ncore:ncore+ncas,ncore:ncore+ncas] += 1.5*np.einsum('rs, xsqpr->xpq', dm_a, v_soc_aaaa)  

    # h2-somf- cc terms
    h2_somf[:,:ncore,:ncore] += np.einsum('rs, xprqs->xpq', dm_c, v_soc_cccc)         
    h2_somf[:,:ncore,:ncore] -= 1.5*np.einsum('rs, xspqr->xpq', dm_c, v_soc_cccc)     
    h2_somf[:,:ncore,:ncore] += 1.5*np.einsum('rs, xsqpr->xpq', dm_c, v_soc_cccc)     
    h2_somf[:,:ncore,:ncore] += np.einsum('rs, xprqs->xpq', dm_a, v_soc_caca)      
    h2_somf[:,:ncore,:ncore] -= 1.5*np.einsum('rs, xspqr->xpq', dm_a, v_soc_acca)
    h2_somf[:,:ncore,:ncore] += 1.5*np.einsum('rs, xsqpr->xpq', dm_a, v_soc_acca)

    # h2-somf- ee terms
    h2_somf[:,nocc:,nocc:] += np.einsum('rs, xprqs->xpq', dm_c, v_soc_ecec)         
    h2_somf[:,nocc:,nocc:] -= 1.5*np.einsum('rs, xspqr->xpq', dm_c, v_soc_ceec)     
    h2_somf[:,nocc:,nocc:] += 1.5*np.einsum('rs, xsqpr->xpq', dm_c, v_soc_ceec)     
    h2_somf[:,nocc:,nocc:] += np.einsum('rs, xprqs->xpq', dm_a, v_soc_eaea)      
    h2_somf[:,nocc:,nocc:] -= 1.5*np.einsum('rs, xspqr->xpq', dm_a, v_soc_aeea)
    h2_somf[:,nocc:,nocc:] += 1.5*np.einsum('rs, xsqpr->xpq', dm_a, v_soc_aeea)

    if ncas > 0:
        #h2-somf- ca terms
        h2_somf[:,:ncore,ncore:ncore+ncas] += np.einsum('rs, xprqs->xpq', dm_c, v_soc_ccac)         
        h2_somf[:,:ncore,ncore:ncore+ncas] -= 1.5*np.einsum('rs, xspqr->xpq', dm_c, v_soc_ccac)     
        h2_somf[:,:ncore,ncore:ncore+ncas] += 1.5*np.einsum('rs, xsqpr->xpq', dm_c, v_soc_cacc)     
        h2_somf[:,:ncore,ncore:ncore+ncas] += np.einsum('rs, xprqs->xpq', dm_a, v_soc_caaa)      
        h2_somf[:,:ncore,ncore:ncore+ncas] -= 1.5*np.einsum('rs, xspqr->xpq', dm_a, v_soc_acaa)
        h2_somf[:,:ncore,ncore:ncore+ncas] += 1.5*np.einsum('rs, xsqpr->xpq', dm_a, v_soc_aaca)

        #h2-somf- ea terms
        h2_somf[:,nocc:,ncore:ncore+ncas] += np.einsum('rs, xprqs->xpq', dm_c, v_soc_ecac)         
        h2_somf[:,nocc:,ncore:ncore+ncas] -= 1.5*np.einsum('rs, xspqr->xpq', dm_c, v_soc_ceac)     
        h2_somf[:,nocc:,ncore:ncore+ncas] += 1.5*np.einsum('rs, xsqpr->xpq', dm_c, v_soc_caec)     
        h2_somf[:,nocc:,ncore:ncore+ncas] += np.einsum('rs, xprqs->xpq', dm_a, v_soc_eaaa)      
        h2_somf[:,nocc:,ncore:ncore+ncas] -= 1.5*np.einsum('rs, xspqr->xpq', dm_a, v_soc_aeaa)
        h2_somf[:,nocc:,ncore:ncore+ncas] += 1.5*np.einsum('rs, xsqpr->xpq', dm_a, v_soc_aaea)

    #h2-somf- ce terms
    h2_somf[:,:ncore,nocc:] += np.einsum('rs, xprqs->xpq', dm_c, v_soc_ccec)         
    h2_somf[:,:ncore,nocc:] -= 1.5*np.einsum('rs, xspqr->xpq', dm_c, v_soc_ccec)     
    h2_somf[:,:ncore,nocc:] += 1.5*np.einsum('rs, xsqpr->xpq', dm_c, v_soc_cecc)     
    h2_somf[:,:ncore,nocc:] += np.einsum('rs, xprqs->xpq', dm_a, v_soc_caea)      
    h2_somf[:,:ncore,nocc:] -= 1.5*np.einsum('rs, xspqr->xpq', dm_a, v_soc_acea)
    h2_somf[:,:ncore,nocc:] += 1.5*np.einsum('rs, xsqpr->xpq', dm_a, v_soc_aeca)

    #h2-somf - ac,ae,ec
    for comp in range(3):
        h2_somf[comp,nocc:,:ncore] = np.conj(h2_somf[comp,:ncore,nocc:]).T
        h2_somf[comp,ncore:ncore+ncas,:ncore] = np.conj(h2_somf[comp,:ncore,ncore:ncore+ncas]).T
        if ncas > 0:
            h2_somf[comp,ncore:ncore+ncas,nocc:] = np.conj(h2_somf[comp,nocc:,ncore:ncore+ncas]).T
    
    del v_soc_aaaa,v_soc_acac,v_soc_caac,v_soc_caca,v_soc_acca,v_soc_cccc,v_soc_ecec,v_soc_ceec,v_soc_eaea,v_soc_aeea,v_soc_ccac,v_soc_cacc,v_soc_caaa,v_soc_acaa,v_soc_aaca,v_soc_caec,v_soc_ceac,v_soc_aaea,v_soc_aeaa,v_soc_ecac,v_soc_eaaa,v_soc_ccec,v_soc_cecc,v_soc_caea,v_soc_acea,v_soc_aeca
    import gc
    gc.collect()
    
    # Build complete h_somf with all subspaces:
    method.h_somf = np.zeros((3,nmo,nmo),dtype='complex')
    for i in range(3):
        method.h_somf[i] = -h1_soc[i] + h2_somf[i]

    return method.h_somf

def transform_2e_integrals_soc(mol, mo1, mo2, mo3, mo4):

        from pyscf import ao2mo
        from scipy import constants
        # Build 2e-SOC SSO integrals: Physicist notation
        int_2e_sso = ao2mo.kernel(mol, (mo1, mo2, mo3, mo4), intor='int2e_p1vxp1_sph', comp=3, aosym="s1")
        int_2e_sso = int_2e_sso.reshape(3, mo1.shape[1],mo2.shape[1],mo3.shape[1],mo4.shape[1])
        int_2e_sso = int_2e_sso.transpose(0,1,3,2,4)
        int_2e_sso = (0.5j*int_2e_sso)*((constants.alpha)**2)
        return int_2e_sso



 