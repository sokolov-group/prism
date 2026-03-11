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

from prism.nevpt import amplitudes
from prism.nevpt import nevpt
from prism.nevpt import soc
from prism.tools import transition

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
    method.h_evec = h_evec

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
    nelecas = method.ref_nelecas
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

    h_eval, h_evec = np.linalg.eigh(h_eff)

    return h_eval, h_evec


def compute_properties(method):
    # Get Oscillator Strengths for transitions from ground state
    e_diff = method.e_tot - method.e_tot[0]
    e_diff = e_diff[1:]

    rdm_mo = method.make_rdm1(L = 0)
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

    # Compute magnetic properties
    if method.gtensor and method.soc:

        # Call make_rdm1 function directly to bypass including SOC effects
        rdm_sf = make_rdm1(method)

        # Compute g-values
        soc.compute_magnetic_properties(method, rdm_sf)

    return osc_str


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
    rdm_qd = einsum('Im,IJpq,Jn->mnpq', evec, rdm_casci, evec)
    
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


def make_rdm1s(method, L = None, R = None, type = 'all'):
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
    wfn = np.einsum('ij,iab->jab',method.h_evec,method.ref_wfn)
    wfn = list(wfn)

    # Looping over states I,J
    for ind_I, I in enumerate(L_list):
        for ind_J, J in enumerate(R_list): 

            if type in ("ss", "state-specific") and I != J:
                continue

            tmprdm_aabb = method.interface.trans_rdm1s(wfn[J], wfn[I], method.ncas, method.ref_nelecas[ind_I])
            rdm_final[0, ind_I, ind_J, method.ncore:method.ncore+method.ncas, method.ncore:method.ncore+method.ncas] = tmprdm_aabb[0]
            rdm_final[1, ind_I, ind_J, method.ncore:method.ncore+method.ncas, method.ncore:method.ncore+method.ncas] = tmprdm_aabb[1]            

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
