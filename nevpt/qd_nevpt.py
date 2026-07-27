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
#          Nicholas Y. Chiang <nicholas.yiching.chiang@gmail.com>

import numpy as np

from prism.nevpt import nevpt
from prism.tools import trans_prop

import prism.lib.logger as logger
import prism.lib.tools as tools


def compute_energy(method):

    # Run state-specific NEVPT first
    nevpt.compute_energy(method)

    method.log.info("\nComputing the QD-NEVPT2 effective Hamiltonian...")

    # Compute and diagonalize the QD-NEVPT2 effective Hamiltonian
    e_tot, h_evec = diagonalize_eff_H(method)
    
    # Update correlation energies
    e_corr = method.e_corr
    n_states = len(method.ref_wfn_deg)
        
    for state in range(n_states):
        e_corr[state] = e_tot[state] - method.e_ref[state]

    # Update class objects
    method.e_tot = e_tot
    method.e_corr = e_corr
    method.h_evec = h_evec.copy()

    # Determine spin multiplicity for the QDNEVPT states
    method.spin_mult = determine_spin_mult(method)

    return e_tot, e_corr


def diagonalize_eff_H(method):

    e_diag = method.e_tot
    t1 = method.t1
    t1_0 = method.t1_0

    # Einsum definition from kernel
    einsum = method.interface.einsum
    einsum_type = method.interface.einsum_type

    ncore = method.ncore - method.nfrozen
    ncas = method.ncas
    nextern = method.nextern

    h_eff = np.diag(e_diag)
    dim = h_eff.shape[0]

    t1_ccee = t1_0

    ## One-electron integrals
    h_ca = method.h1eff.ca
    h_ce = method.h1eff.ce
    h_ae = method.h1eff.ae

    ## Two-electron integrals
    v_cace = method.v2e.cace
    v_caca = method.v2e.caca
    v_ceaa = method.v2e.ceaa
    v_caae = method.v2e.caae
    v_caaa = method.v2e.caaa
    v_aaae = method.v2e.aaae

    # Compute the effective Hamiltonian matrix elements
    for I in range(dim):
        for J in range(I):
            # Compute transition density matrices
            trdm_ca, trdm_ccaa, trdm_cccaaa = method.interface.compute_rdm123(method.ref_wfn[I], method.ref_wfn[J], method.ref_nelecas[I])

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
                chunks = tools.calculate_chunks(method, nextern, [ncore, ncas, nextern], ntensors = 3)
                for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                    cput1 = (logger.process_clock(), logger.perf_counter())
                    method.log.debug("t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                    t1_caee = t1[J].caee[:,:,s_chunk:f_chunk,:]
                    v_ceae = method.v2e.ceae[:,s_chunk:f_chunk,:,:]

                    H_IJ += einsum('ixab,iayb,yx', t1_caee, v_ceae, trdm_ca, optimize = einsum_type)

                    v_ceae = method.v2e.ceae[:,:,:,s_chunk:f_chunk]

                    H_IJ -= 1/2 * einsum('ixab,ibya,yx', t1_caee, v_ceae, trdm_ca, optimize = einsum_type)

                    method.log.timer_debug("contracting t1.xaee", *cput1)
                    del (t1_caee, v_ceae)

            if nextern > 0 and ncas > 0:
                chunks = tools.calculate_chunks(method, nextern, [ncas, ncas, nextern], ntensors = 3)
                for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                    cput1 = (logger.process_clock(), logger.perf_counter())
                    method.log.debug("t1.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                    t1_aaee = t1[J].aaee[:,:,s_chunk:f_chunk,:]
                    v_aeae = method.v2e.aeae[:,s_chunk:f_chunk,:,:]

                    H_IJ += 1/4 * einsum('xyab,zawb,zwxy', t1_aaee, v_aeae, trdm_ccaa, optimize = einsum_type)

                    method.log.timer_debug("contracting t1.aeae", *cput1)
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
                chunks = tools.calculate_chunks(method, nextern, [ncore, ncas, nextern], ntensors = 3)
                for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                    cput1 = (logger.process_clock(), logger.perf_counter())
                    method.log.debug("t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                    t1_caee = t1[I].caee[:,:,s_chunk:f_chunk,:]
                    v_ceae = method.v2e.ceae[:,s_chunk:f_chunk,:,:]

                    H_IJ += einsum('ixab,iayb,xy', t1_caee, v_ceae, trdm_ca, optimize = einsum_type)

                    v_ceae = method.v2e.ceae[:,:,:,s_chunk:f_chunk]

                    H_IJ -= 1/2 * einsum('ixab,ibya,xy', t1_caee, v_ceae, trdm_ca, optimize = einsum_type)

                    del (t1_caee, v_ceae)

            if nextern > 0 and ncas > 0:
                chunks = tools.calculate_chunks(method, nextern, [ncas, ncas, nextern], ntensors = 3)
                for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                    cput1 = (logger.process_clock(), logger.perf_counter())
                    method.log.debug("t1.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                    t1_aaee = t1[I].aaee[:,:,s_chunk:f_chunk,:]
                    v_aeae = method.v2e.aeae[:,s_chunk:f_chunk,:,:]

                    H_IJ += 1/4 * einsum('xyab,zawb,xyzw', t1_aaee, v_aeae, trdm_ccaa, optimize = einsum_type)

                    method.log.timer_debug("contracting t1.aeae", *cput1)
                    del (t1_aaee, v_aeae)

            h_eff[I, J] = H_IJ
            h_eff[J, I] = H_IJ
   
    # print intruder states for qd-nevpt2
    check_intruder_states(method, dim, h_eff, t1)

    h_eval, h_evec = np.linalg.eigh(h_eff)

    return h_eval, h_evec


def compute_properties(method):

    n_states = len(method.e_tot)

    osc_str = None

    # Get Oscillator Strengths for transitions from ground state
    if n_states > 1:

        # Calculate ground state degeneracy:
        deg_gs = 1
        for i in range(len(method.e_tot)-1):
            if (np.abs(method.e_tot[i+1]-method.e_tot[0])) < 1e-5:
                deg_gs += 1
            else:
                break

        rdm_mo = method.make_rdm1() # for all osc calculation
        
        # Calculate oscillator strengths for transitions from the first state
        osc_str_full=[]
        osc_str = np.zeros(len(method.e_tot)-1)
        
        # Get perturbative energy contributions if needed
        if (method.pe is not None and method.pe_method == 'pert'):
            from prism.solvent import pol_embed
            
            osc_str_full_uncorrected = []
            ptss, ptlr = pol_embed.get_pert_pe_corrections(method, rdms = rdm_mo)
            method.properties["ptss_corrections"] = ptss
            method.properties["ptlr_corrections"] = ptlr
            
        # Uncorrected PE data
        if (method.pe is not None and method.pe_method == 'pert' and method.verbose >= 5):
                osc_str_full_uncorrected = []
            
        for gs_index in range(deg_gs):
            e_diff = method.e_tot - method.e_tot[gs_index]
            e_diff = e_diff[gs_index+1:]
        
            if (method.pe is not None and method.pe_method == 'pert'):
                e_diff_uncorrected = e_diff
                    
                ptss, ptlr = pol_embed.get_pert_pe_corrections(method, state = gs_index, rdms = rdm_mo)
                e_diff = [e_diff[i] + (ptss[i]) + (ptlr[i]) for i in range(len(ptss))]
                
                if method.verbose >= 5:
                    osc_uncorrected = trans_prop.osc_strength(method.interface, e_diff_uncorrected, rdm_mo[gs_index, gs_index+1:])
                    osc_str_full_uncorrected.append(osc_uncorrected)
                
            osc = trans_prop.osc_strength(method.interface, e_diff, rdm_mo[gs_index, gs_index+1:])
            
            osc_str_full.append(osc)
            osc_str[gs_index:] += osc 

        method.properties["osc_strengths"] = osc_str

        # Compute oscillator strengths starting from each state
        if method.verbose >= 5:
                
            for gs_index in range(deg_gs, len(method.e_tot)):  
                e_diff = method.e_tot - method.e_tot[gs_index]
                e_diff = e_diff[gs_index+1:]
                
                if (method.pe is not None and method.pe_method == 'pert'):
                    
                    e_diff_uncorrected = e_diff
                    
                    ptss, ptlr = pol_embed.get_pert_pe_corrections(method, state = gs_index, rdms = rdm_mo)
                    e_diff = [e_diff[i] + (ptss[i]) + (ptlr[i]) for i in range(len(ptss))]
                    
                    osc_uncorrected = trans_prop.osc_strength(method.interface, e_diff_uncorrected, rdm_mo[gs_index, gs_index+1:])
                    osc_str_full_uncorrected.append(osc_uncorrected)
 
                osc_str_full.append(trans_prop.osc_strength(method.interface, e_diff, rdm_mo[  gs_index, gs_index+1:]))

            method.properties["osc_strengths_full"] = osc_str_full
            
            if (method.pe is not None and method.pe_method == 'pert'):
                method.properties["osc_strengths_full_uncorrected"] = osc_str_full_uncorrected
            
    # Compute magnetic properties
    if (method.gtensor or method.mag_av or  method.sus_av or  method.mag_vec or  method.sus_tensor) and method.soc:
        from prism.nevpt import soc
        # Call make_rdm1 function directly to bypass including SOC effects
        rdm_sf = make_rdm1(method)

        # Compute compute_magnetic_properties
        properties_mag = soc.compute_magnetic_properties(method, rdm_sf)
        method.properties.update(properties_mag)


def determine_spin_mult(method):

    evec = method.h_evec

    spin_mult_old = method.ref_wfn_spin_mult
    spin_mult_new = []

    for root in range(evec.shape[1]):
        index = np.argmax(np.abs(evec[:, root]))
        spin_mult_new.append(spin_mult_old[index])

    return spin_mult_new


def make_rdm1(method, L = None, R = None, type = 'all', t1 = None, t1_0 = None, evec = None):

    if evec is None: 
        evec = method.h_evec

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
        
    avail_types = ["all", "ss", "state-specific"]
    if type not in avail_types:
        raise ValueError(f"Invalid type: {type}. "f"Allowed types are {avail_types}.")
    
    # Compute model state 1RDM
    rdm_casci = nevpt.make_rdm1(method)
    
    # Compute qdnevpt2 1RDMS
    rdm_qd = einsum('Im,IJpq,Jn->mnpq', evec, rdm_casci, evec, optimize = einsum_type)
    
    # Initial rdm array
    rdm_final = np.zeros((L_list.shape[0], R_list.shape[0], nmo, nmo))

    # Loop structure
    for ind_I, I in enumerate(L_list):
        for ind_J, J in enumerate(R_list):
            rdm_final[ind_I, ind_J] = rdm_qd[I, J]

    # Single pair of states
    if L is not None and R is not None:
        rdm_final = rdm_final[0, 0]

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


def make_rdm1s(method, wfn=None, wfn_ref_nelecas=None , L = None, R = None, type = 'all'):
    ncore = method.ncore
    ncas = method.ncas
    n_micro_states = sum(method.ref_wfn_deg)
    einsum = method.interface.einsum   
    einsum_type = method.interface.einsum_type
    nmo = method.nmo

    L_states = 0
    R_states = 0
    L_list = None
    R_list = None

    if method.rdm_order == 2:
        raise ValueError(f"Invalid type: corelation not implement in spin-orbital RDM. ")

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
    rdm_final = np.zeros((2, L_list.shape[0], R_list.shape[0], nmo, nmo))
    
    #method's wfn
    if wfn is None:
        wfn = einsum('ij,iab->jab', method.h_evec, method.ref_wfn, optimize = einsum_type)
        wfn = list(wfn)

    #wfn's ref_nelecas
    if wfn_ref_nelecas is None:
        wfn_ref_nelecas= method.ref_nelecas

    # Looping over states I,J
    for ind_I, I in enumerate(L_list):
        for ind_J, J in enumerate(R_list): 

            if type in ("ss", "state-specific") and I != J:
                continue
            
            if (wfn_ref_nelecas[I] == wfn_ref_nelecas[J]):
                tmprdm_aabb = method.interface.trans_rdm1s(wfn[J], wfn[I], ncas, wfn_ref_nelecas[ind_I])
                rdm_final[0, ind_I, ind_J, ncore:ncore+ncas, ncore:ncore+ncas] = tmprdm_aabb[0]
                rdm_final[1, ind_I, ind_J, ncore:ncore+ncas, ncore:ncore+ncas] = tmprdm_aabb[1]

                if I == J:
                    #uncorrelated diagonal terms
                    rdm_final[:,ind_I, ind_J, :ncore, :ncore] =   np.identity(ncore)    

    # Single pair of states
    if L is not None and R is not None:
        rdm_final = rdm_final[:,0,0]

    # One state on the left or right
    if L_list.shape[0] == 1 and R_list.shape[0] > 1:
        rdm_final[0] = rdm_final[0, 0, :, :]
        rdm_final[1] = rdm_final[1, 0, :, :]

    if L_list.shape[0] > 1 and R_list.shape[0] == 1:
        rdm_final = rdm_final[:, 0, :, :]

    # State-specific
    if type in ("ss", "state-specific"):
        rdm_final[0] = np.diagonal(rdm_final[0], axis1=0, axis2=1)
        rdm_final[0] = np.moveaxis(rdm_final[0], -1, 0)

        rdm_final[1] = np.diagonal(rdm_final[1], axis1=0, axis2=1)
        rdm_final[1] = np.moveaxis(rdm_final[1], -1, 0)
        
    return rdm_final

  
def check_intruder_states(method, dim, h_eff, t1):

    #Compute the coupling elements of h_eff > %cutoff_intruder Eh give warnings
    interface = method.interface
    cutoff_intruder = method.cutoff_intruder

    data          = []
    coupling_data = []

    I, J = np.tril_indices(dim, k=-1)
    mask = abs(h_eff[I, J]) > cutoff_intruder
    
    vals = h_eff[I, J]

    if np.any(mask):

        interface.log.info("\nWARNING: Large coupling detected (>%f Eh), possible intruder state!!!!!!"  %(cutoff_intruder))
        
        header_fmt = "{:<8} " + "{:>8} " * 13

        row_fmt = "{:<8} " + "{:>8.4f} " * 13

        coupling_fmt = "{:<12} {:>12.8f}"

        # print data for coupling elements of H_eff
        for i, j, val in zip(I[mask], J[mask], vals[mask]):
            coupling_data.append(
                coupling_fmt.format(
                    f"({i+1},{j+1})",
                    val
                )
            )

        states_with_large_coupling = np.unique(np.concatenate((I[mask], J[mask])))

        for state in states_with_large_coupling:

            # smallest denominators
            tp1  = method.den_t1_p1[state]
            tp2  = method.den_t1_p2[state]
            tp3  = method.den_t1_0p[state]
            tp4  = method.den_t1_p1p[state]
            tp5  = method.den_t1_m1p[state]
            tp6  = method.den_t1_m1[state]
            tp7  = method.den_t1_m2[state]

            # amplitudes norms
            t1_ccae = np.linalg.norm(t1[state].ccae)
            t1_caee = np.linalg.norm(t1[state].caee)
            t1_ccaa = np.linalg.norm(t1[state].ccaa)
            t1_aaee = np.linalg.norm(t1[state].aaee)
            t1_caea = np.linalg.norm(t1[state].caea)
            t1_caaa = np.linalg.norm(t1[state].caaa)
            t1_aaae = np.linalg.norm(t1[state].aaae)

            data.append(
                row_fmt.format(
                    " %d" % (state + 1),
                    tp1,
                    tp6,
                    tp2,
                    tp7,
                    tp3,
                    tp4,
                    tp5,
                    t1_ccae,
                    t1_caee,
                    t1_ccaa,
                    t1_aaee,
                    t1_caea,
                    t1_caaa,
                    t1_aaae
             )
            )

    # print the coupling element data
    if coupling_data and method.verbose >= 4:
        
        interface.log.info("\nCoupling Elements of Effective Hamiltonian")
        interface.log.info("-" * 30)
        interface.log.info("{:<12} {:>12}".format("(I,J)", "H_eff"))
        interface.log.info("-" * 30)

        for row in coupling_data:
            interface.log.info(row)

        interface.log.info("-" * 30)
    
    # print the results for amplitudes and denominators
    if data and method.verbose >= 4:

        separator = "-" * 130
        
        interface.log.info("\nIntruder States and Corresponding Denominators and Amplitude Norms for a Specific Class")

        interface.log.info(separator)

        interface.log.info(
            header_fmt.format(
                "State",
                "deno[+1]",
                "deno[-1]",
                "deno[+2]",
                "deno[-2]",
                "deno[0']",
                "deno[+1']",
                "deno[-1']", 
                "amp[+1]", 
                "amp[-1]",
                "amp[+2]",
                "amp[-2]",
                "amp[0']",   
                "amp[+1']",
                "amp[-1']",
            )
        )

        interface.log.info(separator)

        for row in data:
            interface.log.info(row)

        interface.log.info(separator)

    return

  
def analyze_eigenvectors(method, weight_cutoff=0.01):

    from pyscf.fci import cistring

    ncore = method.ncore
    ncas = method.ncas
    n_states = method.h_evec.shape[1]
    h_evec = method.h_evec
    nelecas = method.ref_nelecas[0]
    n_alpha_elec, n_beta_elec = nelecas[0], nelecas[1]

    # Generate physical orbital occupations for each string address
    alpha_strings = cistring.gen_occslst(range(ncas), n_alpha_elec)
    beta_strings = cistring.gen_occslst(range(ncas), n_beta_elec)

    # Rotate the CASSCF wavefunctions into the QD-NEVPT2 eigenbasis
    ref_wfn = np.array(method.ref_wfn)
    n_alpha_str, n_beta_str = ref_wfn.shape[1], ref_wfn.shape[2]
    ref_wfn_flat = ref_wfn.reshape(ref_wfn.shape[0], -1)

    qd_ci_flat = h_evec.T @ ref_wfn_flat
    qd_ci = qd_ci_flat.reshape(n_states, n_alpha_str, n_beta_str)

    mo = method.mo
    ovlp = method.ovlp

    method.log.info("\n ** QD-NEVPT2 Eigenvector Analysis **\n")
    for n in range(n_states):
        method.log.info("  State %d:" % (n + 1))

        method.log.info("    Dominant Configurations:")
        weights_2d = qd_ci[n] ** 2
        for i_a in range(n_alpha_str):
            for i_b in range(n_beta_str):
                weight = weights_2d[i_a, i_b]
                if weight > weight_cutoff:
                    coeff = qd_ci[n, i_a, i_b]
                    alpha_occs = "[%s]" % " ".join(str(int(x)) for x in alpha_strings[i_a])
                    beta_occs  = "[%s]" % " ".join(str(int(x)) for x in beta_strings[i_b])
                    method.log.info("      [alpha occ] %s  [beta occ] %s  coeff: %12.6f  weight: %10.6f"
                                    % (alpha_occs, beta_occs, coeff, weight))

        # Compute Natural Occupations by diagonalizing the 1RDM
        rdm_mo = make_rdm1(method, L=n, R=n)
        nat_occ_global, nat_orb_global = np.linalg.eigh(rdm_mo)
        active_weights = np.sum(nat_orb_global[ncore:ncore+ncas, :]**2, axis=0)
        active_no_indices = np.argsort(active_weights)[-ncas:]
        active_nat_occ = np.sort(nat_occ_global[active_no_indices])[::-1]
        method.log.info("    Natural Occupations (active space): %s" % np.array2string(active_nat_occ, precision=4, suppress_small=True))

        # For open-shell systems, compute atomic Mulliken spin populations in the AO basis
        if n_alpha_elec != n_beta_elec:
            rdm1s = make_rdm1s(method, L=n, R=n)
            d_ao_a = mo @ rdm1s[0] @ mo.T
            d_ao_b = mo @ rdm1s[1] @ mo.T
            spin_dm_ao = d_ao_a - d_ao_b
            spin_pop_ao = np.einsum('ij,ji->i', spin_dm_ao, ovlp)
            mol = method.interface.mol
            method.log.info("    Mulliken Spin Populations:")
            for ia in range(mol.natm):
                ao_start, ao_stop = mol.aoslice_by_atom()[ia][2], mol.aoslice_by_atom()[ia][3]
                spin_atom = np.sum(spin_pop_ao[ao_start:ao_stop])
                method.log.info("      Atom %d %-4s  spin pop: %10.6f" % (ia, mol.atom_symbol(ia), spin_atom))
