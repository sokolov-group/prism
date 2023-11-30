# Copyright 2023 Prism Developers. All Rights Reserved.
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
#          Carlos E. V. de Moura <carlosevmoura@gmail.com>
#

import sys
import time
import numpy as np
from functools import reduce
import prism.mr_adc_intermediates as mr_adc_intermediates
import prism.mr_adc_overlap as mr_adc_overlap
import prism.mr_adc_integrals as mr_adc_integrals

def compute_amplitudes(mr_adc):

    start_time = time.time()

    # First-order amplitudes
    compute_t1_amplitudes(mr_adc)

    # Second-order amplitudes
    compute_t2_amplitudes(mr_adc)

    # Compute CVS amplitudes and remove non-CVS core integrals, amplitudes and unnecessary RDMs
    if mr_adc.method_type == "cvs-ip":
        compute_cvs_amplitudes(mr_adc)

    print("Time for computing amplitudes:                     %f sec\n" % (time.time() - start_time))

def compute_t1_amplitudes(mr_adc):

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_0p, e_p1p, e_m1p, e_0, e_p1, e_m1, e_p2, e_m2 = (0.0,) * 8

    if mr_adc.outcore_expensive_tensors:
        mr_adc.t1.chk = mr_adc.interface.create_HDF5_temp_file()

    # First-order amplitudes
    if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
        if ncore > 0 and nextern > 0 and ncas > 0:
            print("Computing T[0']^(1) amplitudes...")
            sys.stdout.flush()
            e_0p, mr_adc.t1.ce, mr_adc.t1.caea, mr_adc.t1.caae = compute_t1_0p(mr_adc)

            print("Norm of T[0']^(1):                           %20.12f" % (np.linalg.norm(mr_adc.t1.ce) +
                                                                            np.linalg.norm(mr_adc.t1.caea)))
            print("Correlation energy [0']:                     %20.12f\n" % e_0p)
        else:
            mr_adc.t1.ce = np.zeros((ncore, nextern))
            mr_adc.t1.caea = np.zeros((ncore, ncas, nextern, ncas))
            mr_adc.t1.caae = np.zeros((ncore, ncas, ncas, nextern))

        if ncore > 0 and ncas > 0:
            print("Computing T[+1']^(1) amplitudes...")
            sys.stdout.flush()
            e_p1p, mr_adc.t1.ca, mr_adc.t1.caaa = compute_t1_p1p(mr_adc)

            print("Norm of T[+1']^(1):                          %20.12f" % (np.linalg.norm(mr_adc.t1.ca) +
                                                                            np.linalg.norm(mr_adc.t1.caaa)))
            print("Correlation energy [+1']:                    %20.12f\n" % e_p1p)
        else:
            mr_adc.t1.ca = np.zeros((ncore, ncas))
            mr_adc.t1.caaa = np.zeros((ncore, ncas, ncas, ncas))

        if nextern > 0 and ncas > 0:
            print("Computing T[-1']^(1) amplitudes...")
            sys.stdout.flush()
            e_m1p, mr_adc.t1.ae, mr_adc.t1.aaae = compute_t1_m1p(mr_adc)

            print("Norm of T[-1']^(1):                          %20.12f" % (np.linalg.norm(mr_adc.t1.ae) +
                                                                            np.linalg.norm(mr_adc.t1.aaae)))
            print("Correlation energy [-1']:                    %20.12f\n" % e_m1p)
        else:
            mr_adc.t1.ae = np.zeros((ncas, nextern))
            mr_adc.t1.aaae = np.zeros((ncas, ncas, ncas, nextern))

    else:
        mr_adc.t1.ce = np.zeros((ncore, nextern))
        mr_adc.t1.caea = np.zeros((ncore, ncas, nextern, ncas))
        mr_adc.t1.caae = np.zeros((ncore, ncas, ncas, nextern))
        mr_adc.t1.ca = np.zeros((ncore, ncas))
        mr_adc.t1.caaa = np.zeros((ncore, ncas, ncas, ncas))
        mr_adc.t1.ae = np.zeros((ncas, nextern))
        mr_adc.t1.aaae = np.zeros((ncas, ncas, ncas, nextern))

    if ((mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x")) or
        (mr_adc.method == "mr-adc(1)" and mr_adc.method_type in ("ee", "cvs-ee"))):

        if ncore > 0 and nextern > 0:
            print("Computing T[0]^(1) amplitudes...")
            sys.stdout.flush()
            e_0, mr_adc.t1.ccee = compute_t1_0(mr_adc)
            print("Norm of T[0]^(1):                            %20.12f" % np.linalg.norm(mr_adc.t1.ccee))
            print("Correlation energy [0]:                      %20.12f\n" % e_0)
        else:
            mr_adc.t1.ccee = np.zeros((ncore, ncore, nextern, nextern))

        if ncore > 0 and nextern > 0 and ncas > 0:
            print("Computing T[+1]^(1) amplitudes...")
            sys.stdout.flush()
            e_p1, mr_adc.t1.ccae = compute_t1_p1(mr_adc)
            print("Norm of T[+1]^(1):                           %20.12f" % np.linalg.norm(mr_adc.t1.ccae))
            print("Correlation energy [+1]:                     %20.12f\n" % e_p1)

            print("Computing T[-1]^(1) amplitudes...")
            sys.stdout.flush()
            e_m1, mr_adc.t1.caee = compute_t1_m1(mr_adc)
            print("Norm of T[-1]^(1):                           %20.12f" % np.linalg.norm(mr_adc.t1.caee))
            print("Correlation energy [-1]:                     %20.12f\n" % e_m1)
        else:
            mr_adc.t1.ccae = np.zeros((ncore, ncore, ncas, nextern))
            mr_adc.t1.caee = np.zeros((ncore, ncas, nextern, nextern))

        if ncore > 0 and ncas > 0:
            print("Computing T[+2]^(1) amplitudes...")
            sys.stdout.flush()
            e_p2, mr_adc.t1.ccaa = compute_t1_p2(mr_adc)
            print("Norm of T[+2]^(1):                           %20.12f" % np.linalg.norm(mr_adc.t1.ccaa))
            print("Correlation energy [+2]:                     %20.12f\n" % e_p2)
        else:
            mr_adc.t1.ccaa = np.zeros((ncore, ncore, ncas, ncas))

        if nextern > 0 and ncas > 0:
            print("Computing T[-2]^(1) amplitudes...")
            sys.stdout.flush()
            e_m2, mr_adc.t1.aaee = compute_t1_m2(mr_adc)
            print("Norm of T[-2]^(1):                           %20.12f" % np.linalg.norm(mr_adc.t1.aaee))
            print("Correlation energy [-2]:                     %20.12f\n" % e_m2)
        else:
            mr_adc.t1.aaee = np.zeros((ncas, ncas, nextern, nextern))

    else:
        mr_adc.t1.ccee = np.zeros((ncore, ncore, nextern, nextern))
        mr_adc.t1.ccae = np.zeros((ncore, ncore, ncas, nextern))
        mr_adc.t1.caee = np.zeros((ncore, ncas, nextern, nextern))
        mr_adc.t1.ccaa = np.zeros((ncore, ncore, ncas, ncas))
        mr_adc.t1.aaee = np.zeros((ncas, ncas, nextern, nextern))

    e_corr = e_0p + e_p1p + e_m1p + e_0 + e_p1 + e_m1 + e_p2 + e_m2
    e_tot = mr_adc.e_casscf + e_corr

    print("CASSCF reference energy:                     %20.12f" % mr_adc.e_casscf)
    print("PC-NEVPT2 correlation energy:                %20.12f" % e_corr)
    print("Total PC-NEVPT2 energy:                      %20.12f\n" % e_tot)

def compute_t2_amplitudes(mr_adc):

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    approx_trans_moments = mr_adc.approx_trans_moments

    # Approximate second-order amplitudes
    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):

        if (ncore > 0) and (nextern > 0) and not (approx_trans_moments):
            print("Computing T[0']^(2) amplitudes...")
            sys.stdout.flush()
            mr_adc.t2.ce = compute_t2_0p_singles(mr_adc)
            print("Norm of T[0']^(2):                           %20.12f\n" % np.linalg.norm(mr_adc.t2.ce))
            sys.stdout.flush()

        else:
            mr_adc.t2.ce = np.zeros((ncore, nextern))

    else:
        mr_adc.t2.ce = np.zeros((ncore, nextern))

def compute_cvs_amplitudes(mr_adc):
    'Create CVS amplitudes tensors and remove core integrals, core amplitudes and RDMs not used in CVS calculations'

    start_time = time.time()

    print("Computing CVS amplitudes...")
    sys.stdout.flush()

    if mr_adc.method_type == "cvs-ip":

        # Variables from kernel
        ncvs = mr_adc.ncvs
        nval = mr_adc.nval
        ncore = mr_adc.ncore
        ncas = mr_adc.ncas
        nextern = mr_adc.nextern

        del(mr_adc.rdm.ccccaaaa)

        if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            if mr_adc.outcore_expensive_tensors:
                mr_adc.t1.xxee = mr_adc.t1.chk.create_dataset('xxee', (ncvs, ncvs, nextern, nextern), 'f8')
                mr_adc.t1.xvee = mr_adc.t1.chk.create_dataset('xvee', (ncvs, nval, nextern, nextern), 'f8')
                mr_adc.t1.vxee = mr_adc.t1.chk.create_dataset('vxee', (nval, ncvs, nextern, nextern), 'f8')
                mr_adc.t1.vvee = mr_adc.t1.chk.create_dataset('vvee', (nval, nval, nextern, nextern), 'f8')

                mr_adc.t1.xaee = mr_adc.t1.chk.create_dataset('xaee', (ncvs, ncas, nextern, nextern), 'f8')
                mr_adc.t1.vaee = mr_adc.t1.chk.create_dataset('vaee', (nval, ncas, nextern, nextern), 'f8')
            else:
                mr_adc.t1.xxee = np.zeros((ncvs, ncvs, nextern, nextern))
                mr_adc.t1.xvee = np.zeros((ncvs, nval, nextern, nextern))
                mr_adc.t1.vxee = np.zeros((nval, ncvs, nextern, nextern))
                mr_adc.t1.vvee = np.zeros((nval, nval, nextern, nextern))

                mr_adc.t1.xaee = np.zeros((ncvs, ncas, nextern, nextern))
                mr_adc.t1.vaee = np.zeros((nval, ncas, nextern, nextern))

            chunk_size = mr_adc_integrals.calculate_chunk_size(mr_adc, nextern, [ncore, ncore, nextern])
            for s_chunk in range(0, nextern, chunk_size):
                f_chunk = s_chunk + chunk_size

                mr_adc.t1.xxee[:,:,s_chunk:f_chunk] = mr_adc.t1.ccee[:ncvs, :ncvs, s_chunk:f_chunk, :]
                mr_adc.t1.xvee[:,:,s_chunk:f_chunk] = mr_adc.t1.ccee[:ncvs, ncvs:, s_chunk:f_chunk, :]
                mr_adc.t1.vxee[:,:,s_chunk:f_chunk] = mr_adc.t1.ccee[ncvs:, :ncvs, s_chunk:f_chunk, :]
                mr_adc.t1.vvee[:,:,s_chunk:f_chunk] = mr_adc.t1.ccee[ncvs:, ncvs:, s_chunk:f_chunk, :]
            del(mr_adc.t1.ccee)


            chunk_size = mr_adc_integrals.calculate_chunk_size(mr_adc, nextern, [ncore, ncas, nextern])
            for s_chunk in range(0, nextern, chunk_size):
                f_chunk = s_chunk + chunk_size

                mr_adc.t1.xaee[:,:,s_chunk:f_chunk] = mr_adc.t1.caee[:ncvs, :, s_chunk:f_chunk]
                mr_adc.t1.vaee[:,:,s_chunk:f_chunk] = mr_adc.t1.caee[ncvs:, :, s_chunk:f_chunk]
            del(mr_adc.t1.caee)

            mr_adc.t1.xe = np.ascontiguousarray(mr_adc.t1.ce[:ncvs, :])
            mr_adc.t1.ve = np.ascontiguousarray(mr_adc.t1.ce[ncvs:, :])
            del(mr_adc.t1.ce)

            mr_adc.t1.xaea = np.ascontiguousarray(mr_adc.t1.caea[:ncvs, :, :, :])
            mr_adc.t1.vaea = np.ascontiguousarray(mr_adc.t1.caea[ncvs:, :, :, :])
            del(mr_adc.t1.caea)

            mr_adc.t1.xaae = np.ascontiguousarray(mr_adc.t1.caae[:ncvs, :, :, :])
            mr_adc.t1.vaae = np.ascontiguousarray(mr_adc.t1.caae[ncvs:, :, :, :])
            del(mr_adc.t1.caae)

            mr_adc.t1.xa = np.ascontiguousarray(mr_adc.t1.ca[:ncvs, :])
            mr_adc.t1.va = np.ascontiguousarray(mr_adc.t1.ca[ncvs:, :])
            del(mr_adc.t1.ca)

            mr_adc.t1.xaaa = np.ascontiguousarray(mr_adc.t1.caaa[:ncvs, :, :, :])
            mr_adc.t1.vaaa = np.ascontiguousarray(mr_adc.t1.caaa[ncvs:, :, :, :])
            del(mr_adc.t1.caaa)

            mr_adc.t1.xxae = np.ascontiguousarray(mr_adc.t1.ccae[:ncvs, :ncvs, :, :])
            mr_adc.t1.xvae = np.ascontiguousarray(mr_adc.t1.ccae[:ncvs, ncvs:, :, :])
            mr_adc.t1.vxae = np.ascontiguousarray(mr_adc.t1.ccae[ncvs:, :ncvs, :, :])
            mr_adc.t1.vvae = np.ascontiguousarray(mr_adc.t1.ccae[ncvs:, ncvs:, :, :])
            del(mr_adc.t1.ccae)

            mr_adc.t1.xxaa = np.ascontiguousarray(mr_adc.t1.ccaa[:ncvs, :ncvs, :, :])
            mr_adc.t1.xvaa = np.ascontiguousarray(mr_adc.t1.ccaa[:ncvs, ncvs:, :, :])
            mr_adc.t1.vxaa = np.ascontiguousarray(mr_adc.t1.ccaa[ncvs:, :ncvs, :, :])
            mr_adc.t1.vvaa = np.ascontiguousarray(mr_adc.t1.ccaa[ncvs:, ncvs:, :, :])
            del(mr_adc.t1.ccaa)

            mr_adc.t2.xe = np.ascontiguousarray(mr_adc.t2.ce[:ncvs, :])
            mr_adc.t2.ve = np.ascontiguousarray(mr_adc.t2.ce[ncvs:, :])
            del(mr_adc.t2.ce)

    print("Time for computing CVS amplitudes:                 %f sec\n" % (time.time() - start_time))

def compute_t1_0(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncore = mr_adc.ncore
    nextern = mr_adc.nextern

    ## Molecular Orbitals Energies
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    # Compute denominators
    d_ij = e_core[:,None] + e_core

    chunk_size = mr_adc_integrals.calculate_chunk_size(mr_adc, nextern, [ncore, ncore, nextern], 3)
    if mr_adc.outcore_expensive_tensors:
        t1_ccee = mr_adc.t1.chk.create_dataset('ccee', (ncore, ncore, nextern, nextern), 'f8')
    else:
        t1_ccee = np.zeros((ncore, ncore, nextern, nextern))

    e_0 = 0.0
    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Two-electron integrals
        v_cece = mr_adc.v2e.cece[:,s_chunk:f_chunk]

        # Compute denominators
        d_ab = e_extern[s_chunk:f_chunk][:,None] + e_extern
        temp = -d_ij.reshape(-1,1) + d_ab.reshape(-1)
        temp = temp.reshape((ncore, ncore, -1, nextern))
        temp = temp**(-1)

        # Compute T[0] t1_ccee tensor: V1_0 / D2 = - < Psi_0 | a^{\dag}_I a^{\dag}_J a_B a_A V | Psi_0> / D2
        temp *= - einsum('IAJB->IJAB', v_cece, optimize = einsum_type)

        # Compute electronic correlation energy for T[0]
        e_0 += 2 * einsum('ijab,iajb', temp, v_cece, optimize = einsum_type)
        e_0 -= einsum('ijab,jaib', temp, v_cece, optimize = einsum_type)

        t1_ccee[:,:,s_chunk:f_chunk] = temp

    return e_0, t1_ccee

def compute_t1_p1(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncore = mr_adc.ncore
    nextern = mr_adc.nextern

    ## Molecular Orbitals Energies
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    ## Two-electron integrals
    v_cace = mr_adc.v2e.cace

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca

    # Compute K_ac matrix
    K_ac = mr_adc_intermediates.compute_K_ac(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_p1_12_inv_act = mr_adc_overlap.compute_S12_p1(mr_adc)

    if hasattr(mr_adc.S12, "cca"):
        mr_adc.S12.cca = S_p1_12_inv_act.copy()

    # Compute K^{-1} matrix
    SKS = reduce(np.dot, (S_p1_12_inv_act.T, K_ac, S_p1_12_inv_act))
    evals, evecs = np.linalg.eigh(SKS)
    del(SKS)

    # Compute R.H.S. of the equation
    ## V tensor: - < Psi_0 | a^{\dag}_I a^{\dag}_J a_X a_A V | Psi_0>
    V1_p1 =- einsum('JXIA->IJAX', v_cace, optimize = einsum_type).copy()
    V1_p1 += 1/2 * einsum('JxIA,Xx->IJAX', v_cace, rdm_ca, optimize = einsum_type)

    ## Compute denominators
    d_ap = (e_extern[:,None] + evals).reshape(-1)
    d_ij = (e_core[:,None] + e_core).reshape(-1)

    d_apij = (d_ap[:,None] - d_ij).reshape(nextern, evals.shape[0], ncore, ncore)
    d_apij = d_apij**(-1)

    # Compute T[+1] amplitudes
    S_12_V_p1 = einsum("IJAX,Xm->IJAm", V1_p1, S_p1_12_inv_act, optimize = einsum_type)
    S_12_V_p1 = einsum("mp,IJAm->IJAp", evecs, S_12_V_p1, optimize = einsum_type)
    S_12_V_p1 = einsum("ApIJ,IJAp->IJAp", d_apij, S_12_V_p1, optimize = einsum_type)
    S_12_V_p1 = einsum("mp,IJAp->IJAm", evecs, S_12_V_p1, optimize = einsum_type)
    del(V1_p1, d_ap, d_ij, d_apij, evals, evecs)

    ## Compute T[+1] t1_ccae tensor
    t1_ccae = einsum("IJAm,Xm->JIXA", S_12_V_p1, S_p1_12_inv_act, optimize = einsum_type).copy()
    del(S_12_V_p1, S_p1_12_inv_act)

    # Compute electronic correlation energy for T[+1]
    e_p1 =- 2 * einsum('ijxa,jxia', t1_ccae, v_cace, optimize = einsum_type)
    e_p1 += 4 * einsum('ijxa,ixja', t1_ccae, v_cace, optimize = einsum_type)
    e_p1 += einsum('ijxa,jyia,xy', t1_ccae, v_cace, rdm_ca, optimize = einsum_type)
    e_p1 -= 2 * einsum('ijxa,iyja,xy', t1_ccae, v_cace, rdm_ca, optimize = einsum_type)

    return e_p1, t1_ccae

def compute_t1_m1(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    ## Molecular Orbitals Energies
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    ## Two-electron integrals
    v_ceae = mr_adc.v2e.ceae

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca

    # Compute K_ca matrix
    K_ca = mr_adc_intermediates.compute_K_ca(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_m1_12_inv_act = mr_adc_overlap.compute_S12_m1(mr_adc)

    if hasattr(mr_adc.S12, "cae"):
        mr_adc.S12.cae = S_m1_12_inv_act.copy()

    # Compute K^{-1} matrix
    SKS = reduce(np.dot, (S_m1_12_inv_act.T, K_ca, S_m1_12_inv_act))
    evals, evecs = np.linalg.eigh(SKS)
    del(SKS)

    # Compute R.H.S. of the equation
    ## Compute denominators
    d_ix = (e_core[:,None] - evals).reshape(-1)

    chunk_size = mr_adc_integrals.calculate_chunk_size(mr_adc, nextern, [ncore, ncas, nextern], 2)
    if mr_adc.outcore_expensive_tensors:
        t1_caee = mr_adc.t1.chk.create_dataset('caee', (ncore, ncas, nextern, nextern), 'f8')
    else:
        t1_caee = np.zeros((ncore, ncas, nextern, nextern))

    e_m1 = 0.0
    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Compute denominators
        d_ab = (e_extern[s_chunk:f_chunk][:,None] + e_extern).reshape(-1)
        d_abix = (d_ab[:,None] - d_ix).reshape(-1, nextern, ncore, evals.shape[0])
        d_abix = d_abix**(-1)

        ## V matrix: - < Psi_0 | a^{\dag}_I a^{\dag}_X a_B a_A V | Psi_0>
        V1_m1 =- 1/2 * einsum('IAxB,Xx->IXAB', v_ceae[:,s_chunk:f_chunk], rdm_ca, optimize = einsum_type)

        # Compute T[-1] amplitudes
        S_12_V_m1 = einsum("IXAB,Xm->ImAB", V1_m1, S_m1_12_inv_act, optimize = einsum_type)
        S_12_V_m1 = einsum("mp,ImAB->IpAB", evecs, S_12_V_m1, optimize = einsum_type)
        S_12_V_m1 = einsum("ABIp,IpAB->IpAB", d_abix, S_12_V_m1, optimize = einsum_type)
        S_12_V_m1 = einsum("mp,IpAB->ImAB", evecs, S_12_V_m1, optimize = einsum_type)
        del(V1_m1, d_abix)

        ## Compute T[-1] t1_caee tensor
        temp = einsum("ImAB,Xm->IXAB", S_12_V_m1, S_m1_12_inv_act, optimize = einsum_type).copy()
        del(S_12_V_m1)

        # Compute electronic correlation energy for T[-1]
        e_m1 += 2 * einsum('ixab,iayb,xy', temp, v_ceae[:,s_chunk:f_chunk], rdm_ca, optimize = einsum_type)
        e_m1 -= einsum('ixab,ibya,xy', temp, v_ceae[:,:,:,s_chunk:f_chunk], rdm_ca, optimize = einsum_type)

        t1_caee[:,:,s_chunk:f_chunk] = temp

    return e_m1, t1_caee

def compute_t1_p2(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas

    ## Molecular Orbitals Energies
    e_core = mr_adc.mo_energy.c

    ## Two-electron integrals
    v_caca = mr_adc.v2e.caca

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Compute K_aaccc matrix
    K_aacc = mr_adc_intermediates.compute_K_aacc(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_p2_12_inv_act = mr_adc_overlap.compute_S12_p2(mr_adc)

    # Compute K^{-1} matrix
    SKS = reduce(np.dot, (S_p2_12_inv_act.T, K_aacc, S_p2_12_inv_act))
    evals, evecs = np.linalg.eigh(SKS)
    del(SKS)

    # Compute R.H.S. of the equation
    ## V tensor: - < Psi_0 | a^{\dag}_I a^{\dag}_J a_Y a_X V | Psi_0>
    V1_p2 =- einsum('IXJY->IJXY', v_caca, optimize = einsum_type).copy()
    V1_p2 += 1/2 * einsum('IXJx,Yx->IJXY', v_caca, rdm_ca, optimize = einsum_type)
    V1_p2 += 1/2 * einsum('IxJY,Xx->IJXY', v_caca, rdm_ca, optimize = einsum_type)
    V1_p2 -= 1/3 * einsum('IxJy,XYxy->IJXY', v_caca, rdm_ccaa, optimize = einsum_type)
    V1_p2 -= 1/6 * einsum('IxJy,XYyx->IJXY', v_caca, rdm_ccaa, optimize = einsum_type)
    V1_p2 = V1_p2.reshape(ncore, ncore, ncas**2)

    # Compute denominators
    d_ij = (e_core[:,None] + e_core).reshape(-1)
    d_pij = (evals[:,None] - d_ij).reshape(evals.shape[0], ncore, ncore)
    d_pij = d_pij**(-1)

    # Compute T[+2] amplitudes
    S_12_V_p2 = einsum("IJX,Xm->IJm", V1_p2, S_p2_12_inv_act, optimize = einsum_type)
    S_12_V_p2 = einsum("mp,IJm->IJp", evecs, S_12_V_p2, optimize = einsum_type)
    S_12_V_p2 = einsum("pIJ,IJp->IJp", d_pij, S_12_V_p2, optimize = einsum_type)
    S_12_V_p2 = einsum("mp,IJp->IJm", evecs, S_12_V_p2, optimize = einsum_type)
    del(V1_p2, d_ij, d_pij, evals, evecs)

    ## Compute T[+2] t1_ccaa tensor
    t1_ccaa = einsum("IJm,Xm->IJX", S_12_V_p2, S_p2_12_inv_act, optimize = einsum_type)
    t1_ccaa = t1_ccaa.reshape(ncore, ncore, ncas, ncas)
    del(S_12_V_p2, S_p2_12_inv_act)

    # Compute electronic correlation energy for T[+2]
    e_p2  = 2 * einsum('ijxy,ixjy', t1_ccaa, v_caca, optimize = einsum_type)
    e_p2 -= einsum('ijxy,jxiy', t1_ccaa, v_caca, optimize = einsum_type)
    e_p2 += 1/2 * einsum('ijxy,izjw,xyzw', t1_ccaa, v_caca, rdm_ccaa, optimize = einsum_type)
    e_p2 += einsum('ijxy,iyjz,xz', t1_ccaa, v_caca, rdm_ca, optimize = einsum_type)
    e_p2 -= 2 * einsum('ijxy,ixjz,yz', t1_ccaa, v_caca, rdm_ca, optimize = einsum_type)

    return e_p2, t1_ccaa

def compute_t1_m2(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    ## Molecular Orbitals Energies
    e_extern = mr_adc.mo_energy.e

    ## Reduced density matrices
    rdm_ccaa = mr_adc.rdm.ccaa

    # Compute K_ccaa matrix
    K_ccaa = mr_adc_intermediates.compute_K_ccaa(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_m2_12_inv_act = mr_adc_overlap.compute_S12_m2(mr_adc)

    # Compute K^{-1} matrix
    SKS = reduce(np.dot, (S_m2_12_inv_act.T, K_ccaa, S_m2_12_inv_act))
    evals, evecs = np.linalg.eigh(SKS)
    del(SKS)

    # Compute R.H.S. of the equation
    chunk_size = mr_adc_integrals.calculate_chunk_size(mr_adc, nextern, [ncas, ncas, nextern], 3)
    if mr_adc.outcore_expensive_tensors:
        t1_aaee = mr_adc.t1.chk.create_dataset('aaee', (ncas, ncas, nextern, nextern), 'f8')
    else:
        t1_aaee = np.zeros((ncas, ncas, nextern, nextern))

    e_m2 = 0.0
    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Two-electron integrals
        v_aeae = mr_adc.v2e.aeae[:,s_chunk:f_chunk]

        ## Compute denominators
        d_ab = (e_extern[s_chunk:f_chunk][:,None] + e_extern).reshape(-1)
        d_abp = (d_ab[:,None] + evals).reshape(-1, nextern, evals.shape[0])
        d_abp = d_abp**(-1)

        ## V tensor: - < Psi_0 | a^{\dag}_X a^{\dag}_Y a_B a_A V | Psi_0>
        V1_m2 =- 1/3 * einsum('xAyB,XYxy->XYAB', v_aeae, rdm_ccaa, optimize = einsum_type)
        V1_m2 -= 1/6 * einsum('xAyB,XYyx->XYAB', v_aeae, rdm_ccaa, optimize = einsum_type)
        V1_m2 = V1_m2.reshape(ncas**2, -1, nextern)

        # Compute T[-2] amplitudes
        S_12_V_m2 = einsum("XAB,Xm->mAB", V1_m2, S_m2_12_inv_act, optimize = einsum_type)
        S_12_V_m2 = einsum("mp,mAB->pAB", evecs, S_12_V_m2, optimize = einsum_type)
        S_12_V_m2 = einsum("ABp,pAB->pAB", d_abp, S_12_V_m2, optimize = einsum_type)
        S_12_V_m2 = einsum("mp,pAB->mAB", evecs, S_12_V_m2, optimize = einsum_type)
        del(V1_m2)

        ## Compute T[-2] t1_aaee tensor
        temp = einsum("mAB,Xm->XAB", S_12_V_m2, S_m2_12_inv_act, optimize = einsum_type)
        temp = temp.reshape(ncas, ncas, -1, nextern)
        del(S_12_V_m2)

        # Compute electronic correlation energy for T[-2]
        e_m2 += 1/2 * einsum('xyab,zawb,xyzw', temp, v_aeae, rdm_ccaa, optimize = einsum_type)

        t1_aaee[:,:,s_chunk:f_chunk] = temp

    return e_m2, t1_aaee

def compute_t1_0p(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    ## Molecular Orbitals Energies
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    ## One-electron integrals
    h_ce = mr_adc.h1eff.ce

    ## Two-electron integrals
    v_caae = mr_adc.v2e.caae
    v_ceaa = mr_adc.v2e.ceaa

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Compute K_caca matrix
    K_caca = mr_adc_intermediates.compute_K_caca(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_0p_12_inv_act = mr_adc_overlap.compute_S12_0p_gno_projector(mr_adc)

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

    return e_0p, t1_ce, t1_caea, t1_caae

def compute_t1_p1p(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas

    ## Molecular Orbitals Energies
    e_core = mr_adc.mo_energy.c

    ## One-electron integrals
    h_ca = mr_adc.h1eff.ca

    ## Two-electron integrals
    v_caaa = mr_adc.v2e.caaa

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Compute K_p1p matrix
    K_p1p = mr_adc_intermediates.compute_K_p1p(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_p1p_12_inv_act = mr_adc_overlap.compute_S12_p1p_gno_projector(mr_adc)

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

    return e_p1p, t1_ca, t1_caaa

def compute_t1_m1p(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    ## Molecular Orbitals Energies
    e_extern = mr_adc.mo_energy.e

    ## One-electron integrals
    h_ae = mr_adc.h1eff.ae

    ## Two-electron integrals
    v_aaae = mr_adc.v2e.aaae

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Compute K_m1p matrix
    K_m1p = mr_adc_intermediates.compute_K_m1p(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_m1p_12_inv_act = mr_adc_overlap.compute_S12_m1p_gno_projector(mr_adc)

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

    return e_m1p, t1_ae, t1_aaae

def compute_t2_0p_singles(mr_adc):

    # Import Prism interface
    interface = mr_adc.interface

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    ## Molecular Orbitals Energies
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    ## One-electron integrals
    h_ca = mr_adc.h1eff.ca
    h_ce = mr_adc.h1eff.ce
    h_aa = mr_adc.h1eff.aa
    h_ae = mr_adc.h1eff.ae

    ## Two-electron integrals
    v_aaaa = mr_adc.v2e.aaaa

    v_ccca = mr_adc.v2e.ccca
    v_ccce = mr_adc.v2e.ccce

    v_ccaa = mr_adc.v2e.ccaa
    v_ccae = mr_adc.v2e.ccae

    v_caac = mr_adc.v2e.caac
    v_caec = mr_adc.v2e.caec

    v_cace = mr_adc.v2e.cace

    v_caaa = mr_adc.v2e.caaa
    v_ceae = mr_adc.v2e.ceae
    v_caae = mr_adc.v2e.caae
    v_ceaa = mr_adc.v2e.ceaa

    v_caea = mr_adc.v2e.caea

    v_aaae = mr_adc.v2e.aaae

    ## Amplitudes
    t1_ce = mr_adc.t1.ce
    t1_caea = mr_adc.t1.caea
    t1_caae = mr_adc.t1.caae

    t1_ca = mr_adc.t1.ca
    t1_caaa = mr_adc.t1.caaa

    t1_ae   = mr_adc.t1.ae
    t1_aaae = mr_adc.t1.aaae

    t1_ccae = mr_adc.t1.ccae
    t1_ccaa = mr_adc.t1.ccaa

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Compute R.H.S. of the equation
    # V1 block: - 1/2 < Psi_0 | a^{\dag}_I a_A [V + H^{(1)}, T - T^\dag] | Psi_0 >
    V1  = einsum('Ix,xA->IA', h_ca, t1_ae, optimize = einsum_type)
    V1 += einsum('ix,IixA->IA', h_ca, t1_ccae, optimize = einsum_type)
    V1 -= 2 * einsum('ix,iIxA->IA', h_ca, t1_ccae, optimize = einsum_type)
    V1 -= einsum('xA,Ix->IA', h_ae, t1_ca, optimize = einsum_type)
    V1 += einsum('Iixy,ixAy->IA', t1_ccaa, v_caea, optimize = einsum_type)
    V1 -= 2 * einsum('Iixy,iyAx->IA', t1_ccaa, v_caea, optimize = einsum_type)
    V1 -= einsum('ijxA,iIjx->IA', t1_ccae, v_ccca, optimize = einsum_type)
    V1 += 2 * einsum('ijxA,jIix->IA', t1_ccae, v_ccca, optimize = einsum_type)
    V1 += einsum('ix,IixA->IA', t1_ca, v_ccae, optimize = einsum_type)
    V1 += einsum('ix,IxiA->IA', t1_ca, v_cace, optimize = einsum_type)
    V1 -= 2 * einsum('ix,ixAI->IA', t1_ca, v_caec, optimize = einsum_type)
    V1 -= 2 * einsum('ix,ixIA->IA', t1_ca, v_cace, optimize = einsum_type)
    V1 += 1/2 * einsum('A,IixA,ix->IA', e_extern, t1_ccae, t1_ca, optimize = einsum_type)
    V1 -= einsum('A,iIxA,ix->IA', e_extern, t1_ccae, t1_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,xA,Ix->IA', e_extern, t1_ae, t1_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,IixA,ix->IA', e_core, t1_ccae, t1_ca, optimize = einsum_type)
    V1 += einsum('I,iIxA,ix->IA', e_core, t1_ccae, t1_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,xA,Ix->IA', e_core, t1_ae, t1_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,IixA,ix->IA', e_core, t1_ccae, t1_ca, optimize = einsum_type)
    V1 += einsum('i,iIxA,ix->IA', e_core, t1_ccae, t1_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ix,IixA->IA', e_core, t1_ca, t1_ccae, optimize = einsum_type)
    V1 += einsum('i,ix,iIxA->IA', e_core, t1_ca, t1_ccae, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ix,xyzA,zy->IA', h_ca, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += einsum('Ix,yxzA,zy->IA', h_ca, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,IiyA,xy->IA', h_ca, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('ix,iIyA,xy->IA', h_ca, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xA,Iyxz,yz->IA', h_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xA,Iyzx,yz->IA', h_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,IixA,iy->IA', h_aa, t1_ccae, t1_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xy,iIxA,iy->IA', h_aa, t1_ccae, t1_ca, optimize = einsum_type)
    V1 += einsum('xy,xA,Iy->IA', h_aa, t1_ae, t1_ca, optimize = einsum_type)
    V1 += einsum('IixA,ixyz,zy->IA', t1_ccae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('IixA,iyzw,xzyw->IA', t1_ccae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('IixA,iyzx,yz->IA', t1_ccae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixy,ixAz,yz->IA', t1_ccaa, v_caea, rdm_ca, optimize = einsum_type)
    V1 += einsum('Iixy,iyAz,xz->IA', t1_ccaa, v_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixy,izAw,xywz->IA', t1_ccaa, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Iixy,izAx,yz->IA', t1_ccaa, v_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixy,izAy,xz->IA', t1_ccaa, v_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ix,xyzA,yz->IA', t1_ca, v_aaae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Ix,yzxA,zy->IA', t1_ca, v_aaae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Ixyz,wuyA,xwzu->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixyz,wuzA,xwyu->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ixyz,wxuA,yzuw->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixyz,ywuA,xuzw->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixyz,ywzA,xw->IA', t1_caaa, v_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixyz,zwuA,xuwy->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('Ixyz,zwyA,xw->IA', t1_caaa, v_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('iIxA,ixyz,zy->IA', t1_ccae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('iIxA,iyzw,xzyw->IA', t1_ccae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('iIxA,iyzx,yz->IA', t1_ccae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ijxA,iIjy,xy->IA', t1_ccae, v_ccca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ijxA,jIiy,xy->IA', t1_ccae, v_ccca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,IiyA,xy->IA', t1_ca, v_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,IyiA,xy->IA', t1_ca, v_cace, rdm_ca, optimize = einsum_type)
    V1 += einsum('ix,iyAI,xy->IA', t1_ca, v_caec, rdm_ca, optimize = einsum_type)
    V1 += einsum('ix,iyIA,xy->IA', t1_ca, v_cace, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixAy,Iiyz,xz->IA', t1_caea, v_ccaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixAy,Iizw,ywxz->IA', t1_caea, v_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixAy,Izwi,yzxw->IA', t1_caea, v_caac, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixAy,Izyi,xz->IA', t1_caea, v_caac, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyA,Iiyz,xz->IA', t1_caae, v_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyA,Iizw,ywxz->IA', t1_caae, v_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyA,Izwi,yzwx->IA', t1_caae, v_caac, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ixyA,Izyi,xz->IA', t1_caae, v_caac, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,IiwA,xwzy->IA', t1_caaa, v_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ixyz,IiyA,xz->IA', t1_caaa, v_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,IizA,xy->IA', t1_caaa, v_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,IwiA,xwzy->IA', t1_caaa, v_cace, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ixyz,IyiA,zx->IA', t1_caaa, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,IziA,yx->IA', t1_caaa, v_cace, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixyz,iwAI,xwzy->IA', t1_caaa, v_caec, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ixyz,iwIA,xwzy->IA', t1_caaa, v_cace, rdm_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('ixyz,iyAI,xz->IA', t1_caaa, v_caec, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('ixyz,iyIA,zx->IA', t1_caaa, v_cace, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixyz,izAI,xy->IA', t1_caaa, v_caec, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixyz,izIA,yx->IA', t1_caaa, v_cace, rdm_ca, optimize = einsum_type)
    V1 += einsum('xA,Ixyz,yz->IA', t1_ae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xA,Iyzx,zy->IA', t1_ae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzA,Iwux,zwuy->IA', t1_aaae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzA,Iwuy,zwxu->IA', t1_aaae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzA,Iwzu,yxwu->IA', t1_aaae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzA,Ixwu,zuyw->IA', t1_aaae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzA,Ixwy,zw->IA', t1_aaae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzA,Iywu,zuxw->IA', t1_aaae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzA,Iywx,zw->IA', t1_aaae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,IixA,iy,xy->IA', e_extern, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('A,IixA,iyxz,zy->IA', e_extern, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,IixA,iyzw,xyzw->IA', e_extern, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,IixA,iyzx,zy->IA', e_extern, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('A,iIxA,iy,xy->IA', e_extern, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('A,iIxA,iyxz,zy->IA', e_extern, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('A,iIxA,iyzw,xyzw->IA', e_extern, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('A,iIxA,iyzx,zy->IA', e_extern, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,xA,Iyxz,yz->IA', e_extern, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,xA,Iyzx,yz->IA', e_extern, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,xyzA,Iwux,zuwy->IA', e_extern, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,xyzA,Iwuy,zuxw->IA', e_extern, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,xyzA,Iwxu,zuyw->IA', e_extern, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,xyzA,Iwxy,zw->IA', e_extern, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,xyzA,Iwyu,zuxw->IA', e_extern, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,xyzA,Iwyx,zw->IA', e_extern, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,xyzA,Ix,zy->IA', e_extern, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,xyzA,Iy,zx->IA', e_extern, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,xyzA,Izwu,yxwu->IA', e_extern, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,IixA,iy,xy->IA', e_core, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,IixA,iyxz,zy->IA', e_core, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,IixA,iyzw,xyzw->IA', e_core, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,IixA,iyzx,zy->IA', e_core, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,iIxA,iy,xy->IA', e_core, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += einsum('I,iIxA,iyxz,zy->IA', e_core, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,iIxA,iyzw,xyzw->IA', e_core, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,iIxA,iyzx,zy->IA', e_core, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,xA,Iyxz,yz->IA', e_core, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,xA,Iyzx,yz->IA', e_core, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,xyzA,Iwux,zuwy->IA', e_core, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,xyzA,Iwuy,zuxw->IA', e_core, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,xyzA,Iwxu,zuyw->IA', e_core, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,xyzA,Iwxy,zw->IA', e_core, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,xyzA,Iwyu,zuxw->IA', e_core, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,xyzA,Iwyx,zw->IA', e_core, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,xyzA,Ix,zy->IA', e_core, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,xyzA,Iy,zx->IA', e_core, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,xyzA,Izwu,yxwu->IA', e_core, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,IixA,iy,xy->IA', e_core, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,IixA,iyxz,yz->IA', e_core, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,IixA,iyzw,xyzw->IA', e_core, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,IixA,iyzx,yz->IA', e_core, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,iIxA,iy,xy->IA', e_core, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += einsum('i,iIxA,iyxz,yz->IA', e_core, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,iIxA,iyzw,xyzw->IA', e_core, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iIxA,iyzx,yz->IA', e_core, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyz,IiyA,xz->IA', e_core, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixyz,IizA,xy->IA', e_core, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('i,ixyz,iIyA,xz->IA', e_core, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyz,iIzA,xy->IA', e_core, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,IixA,iz,yz->IA', h_aa, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,IixA,izwu,yzwu->IA', h_aa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,IixA,izwy,wz->IA', h_aa, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,IixA,izyw,wz->IA', h_aa, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Ixzw,uwvA,yuzv->IA', h_aa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Ixzw,uzvA,yuwv->IA', h_aa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Ixzw,wA,yz->IA', h_aa, t1_caaa, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Ixzw,wuvA,yuvz->IA', h_aa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Ixzw,wzuA,yu->IA', h_aa, t1_caaa, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Ixzw,zA,yw->IA', h_aa, t1_caaa, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Ixzw,zuvA,yuwv->IA', h_aa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Ixzw,zwuA,yu->IA', h_aa, t1_caaa, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izwx,uvzA,ywuv->IA', h_aa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Izwx,uwvA,yvzu->IA', h_aa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Izwx,wA,yz->IA', h_aa, t1_caaa, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Izwx,wuvA,yvzu->IA', h_aa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izxw,uvzA,ywvu->IA', h_aa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Izxw,uwvA,yvzu->IA', h_aa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Izxw,wA,yz->IA', h_aa, t1_caaa, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Izxw,wuvA,yvuz->IA', h_aa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,iIxA,iz,yz->IA', h_aa, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,iIxA,izwu,yzwu->IA', h_aa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,iIxA,izwy,wz->IA', h_aa, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xy,iIxA,izyw,wz->IA', h_aa, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ix,IizA,yz->IA', h_aa, t1_ca, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ix,iIzA,yz->IA', h_aa, t1_ca, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixzw,IiuA,yuwz->IA', h_aa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixzw,IiwA,yz->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixzw,IizA,yw->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixzw,iIuA,yuwz->IA', h_aa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixzw,iIwA,yz->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,ixzw,iIzA,yw->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izwx,IiuA,ywzu->IA', h_aa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izwx,IiwA,yz->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izwx,iIuA,ywzu->IA', h_aa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,izwx,iIwA,yz->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izxw,IiuA,ywuz->IA', h_aa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izxw,IiwA,yz->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izxw,iIuA,ywuz->IA', h_aa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izxw,iIwA,yz->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,xA,Izwy,zw->IA', h_aa, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,xA,Izyw,zw->IA', h_aa, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,xzwA,Iuvy,wvuz->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzwA,Iuvz,yuwv->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,xzwA,Iuyv,wvzu->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,xzwA,Iuyz,wu->IA', h_aa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xzwA,Iuzv,yuwv->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,xzwA,Iuzy,wu->IA', h_aa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,xzwA,Iwuv,yzvu->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,xzwA,Iy,wz->IA', h_aa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xzwA,Iz,yw->IA', h_aa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxA,Iuvw,yvzu->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxA,Iuvz,yvuw->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zwxA,Iuwv,yvzu->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zwxA,Iuwz,yu->IA', h_aa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxA,Iuzv,yvwu->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxA,Iuzw,yu->IA', h_aa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zwxA,Iw,yz->IA', h_aa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zwxA,Iyuv,wzuv->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxA,Iz,yw->IA', h_aa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zxwA,Iuvy,wvzu->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwA,Iuvz,yuvw->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,zxwA,Iuyv,wvzu->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,zxwA,Iuyz,wu->IA', h_aa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwA,Iuzv,yuwv->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zxwA,Iuzy,wu->IA', h_aa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zxwA,Iwuv,yzuv->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,zxwA,Iy,wz->IA', h_aa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwA,Iz,yw->IA', h_aa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixAy,Iizw,zw,yx->IA', t1_caea, v_ccaa, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixAy,Izwi,wz,yx->IA', t1_caea, v_caac, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyA,Iizw,zw,yx->IA', t1_caae, v_ccaa, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('ixyA,Izwi,wz,yx->IA', t1_caae, v_caac, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,IixA,iu,zuwy->IA', v_aaaa, t1_ccae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IixA,iuvs,zvsuwy->IA', v_aaaa, t1_ccae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IixA,iuvs,zvsuyw->IA', v_aaaa, t1_ccae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IixA,iuvs,zvswuy->IA', v_aaaa, t1_ccae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IixA,iuvs,zvswyu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IixA,iuvs,zvsyuw->IA', v_aaaa, t1_ccae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IixA,iuvs,zvsywu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,IixA,iuvw,zvuy->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,IixA,iuvy,zvwu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,IixA,iuwv,zvyu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,IixA,iuwy,zu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,IixA,iuyv,zvwu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,IixA,iuyw,zu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,IixA,iw,zy->IA', v_aaaa, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,IixA,iy,zw->IA', v_aaaa, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,IixA,izuv,ywuv->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuvx,stuA,ztswvy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuvx,svtA,zustwy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuvx,svtA,zustyw->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuvx,svtA,zuswty->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,Iuvx,svtA,zuswyt->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuvx,svtA,zusytw->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuvx,svtA,zusywt->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuvx,vA,zuwy->IA', v_aaaa, t1_caaa, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuvx,vstA,zustwy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuvx,vstA,zustyw->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuvx,vstA,zuswty->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuvx,vstA,zuswyt->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuvx,vstA,zusytw->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuvx,vstA,zusywt->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuxv,stuA,ztsvwy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuxv,stuA,ztsvyw->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuxv,stuA,ztswvy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iuxv,stuA,ztswyv->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuxv,stuA,ztsyvw->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuxv,stuA,ztsywv->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuxv,svtA,zustwy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuxv,svtA,zustyw->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuxv,svtA,zuswty->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuxv,svtA,zuswyt->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuxv,svtA,zusytw->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuxv,svtA,zusywt->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxv,vA,zuwy->IA', v_aaaa, t1_caaa, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxv,vstA,zuswty->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuxz,vsuA,ywsv->IA', v_aaaa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Ixuv,sutA,zvtswy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Ixuv,sutA,zvtsyw->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Ixuv,sutA,zvtwsy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,Ixuv,sutA,zvtwys->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Ixuv,sutA,zvtysw->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Ixuv,sutA,zvtyws->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,svtA,zutswy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,svtA,zutsyw->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,svtA,zutwsy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Ixuv,svtA,zutwys->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,svtA,zutysw->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,svtA,zutyws->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Ixuv,uA,zvwy->IA', v_aaaa, t1_caaa, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,ustA,zvtswy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,ustA,zvtsyw->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,ustA,zvtwsy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Ixuv,ustA,zvtwys->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,ustA,zvtysw->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,ustA,zvtyws->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Ixuv,uvsA,zswy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Ixuv,vA,zuwy->IA', v_aaaa, t1_caaa, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Ixuv,vstA,zutwsy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Ixuv,vusA,zswy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iIxA,iu,zuwy->IA', v_aaaa, t1_ccae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iIxA,iuvs,zvsuwy->IA', v_aaaa, t1_ccae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iIxA,iuvs,zvsuyw->IA', v_aaaa, t1_ccae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iIxA,iuvs,zvswuy->IA', v_aaaa, t1_ccae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,iIxA,iuvs,zvswyu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iIxA,iuvs,zvsyuw->IA', v_aaaa, t1_ccae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iIxA,iuvs,zvsywu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,iIxA,iuvw,zvuy->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,iIxA,iuvy,zvwu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,iIxA,iuwv,zvyu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,iIxA,iuwy,zu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xyzw,iIxA,iuyv,zvwu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('xyzw,iIxA,iuyw,zu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,iIxA,iw,zy->IA', v_aaaa, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xyzw,iIxA,iy,zw->IA', v_aaaa, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,iIxA,izuv,ywuv->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IisA,zusvwy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IisA,zusvyw->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IisA,zuswvy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,IisA,zuswyv->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IisA,zusyvw->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IisA,zusywv->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuvx,IivA,zuwy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,iIsA,zusvwy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,iIsA,zusvyw->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,iIsA,zuswvy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,iuvx,iIsA,zuswyv->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,iIsA,zusyvw->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,iIsA,zusywv->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('xyzw,iuvx,iIvA,zuwy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,IisA,zuswvy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,IivA,zuwy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxv,iIsA,zuswvy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxv,iIvA,zuwy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxz,IivA,ywvu->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxz,iIvA,ywvu->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ix,IiuA,zuwy->IA', v_aaaa, t1_ca, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ix,iIuA,zuwy->IA', v_aaaa, t1_ca, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,IisA,zvuwys->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixuv,IiuA,zvwy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,IivA,zuwy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixuv,iIsA,zvuwys->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,ixuv,iIuA,zvwy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixuv,iIvA,zuwy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxA,Istu,zvswty->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvxA,Istv,zustwy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvxA,Istv,zustyw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvxA,Istv,zuswty->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uvxA,Istv,zuswyt->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvxA,Istv,zusytw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvxA,Istv,zusywt->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uvxA,Istz,ywtsuv->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uvxA,Istz,ywtsvu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,uvxA,Istz,ywtusv->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uvxA,Istz,ywtuvs->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uvxA,Istz,ywtvsu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uvxA,Istz,ywtvus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvxA,Isut,zvstwy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvxA,Isut,zvstyw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvxA,Isut,zvswty->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uvxA,Isut,zvswyt->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvxA,Isut,zvsytw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvxA,Isut,zvsywt->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxA,Isuv,zswy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxA,Isuz,ywvs->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uvxA,Isvt,zustwy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uvxA,Isvt,zustyw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uvxA,Isvt,zuswty->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,uvxA,Isvt,zuswyt->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uvxA,Isvt,zusytw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uvxA,Isvt,zusywt->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uvxA,Isvu,zswy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uvxA,Isvz,ywus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uvxA,Iszt,ywtuvs->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxA,Iszu,ywsv->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxA,Iszv,ywus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxA,Iu,zvwy->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uvxA,Iv,zuwy->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uvxA,Iwst,zvusty->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uvxA,Iwst,zvusyt->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,uvxA,Iwst,zvutsy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uvxA,Iwst,zvutys->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uvxA,Iwst,zvuyst->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uvxA,Iwst,zvuyts->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uvxA,Iyst,zvuwst->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uvxA,Iz,ywuv->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxvA,Istu,zvtwsy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,uxvA,Istw,zvtsuy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uxvA,Istw,zvtsyu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uxvA,Istw,zvtusy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uxvA,Istw,zvtuys->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uxvA,Istw,zvtysu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uxvA,Istw,zvtyus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Isty,zvtwus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxvA,Isut,zvtswy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxvA,Isut,zvtsyw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxvA,Isut,zvtwsy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxvA,Isut,zvtwys->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxvA,Isut,zvtysw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxvA,Isut,zvtyws->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Isuw,zvsy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Isuy,zvws->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Iswt,zvtyus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Iswu,zvys->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Iswy,zvsu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,uxvA,Isyt,zvtwus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,uxvA,Isyu,zvws->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,uxvA,Isyw,zvsu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxvA,Iu,zvwy->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uxvA,Ivst,zstuwy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uxvA,Ivst,zstuyw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uxvA,Ivst,zstwuy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uxvA,Ivst,zstwyu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uxvA,Ivst,zstyuw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uxvA,Ivst,zstywu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Iw,zvyu->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,uxvA,Iy,zvwu->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,uxvA,Izst,ywustv->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxvA,Izst,ywusvt->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxvA,Izst,ywutsv->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxvA,Izst,ywutvs->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxvA,Izst,ywuvst->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxvA,Izst,ywuvts->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uxvA,Izsu,ywsv->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uxvA,Izus,ywvs->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xA,Iuvw,zvuy->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xA,Iuvy,zvwu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xA,Iuwv,zvyu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xA,Iuwy,zu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,xA,Iuyv,zvwu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,xA,Iuyw,zu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xA,Iw,zy->IA', v_aaaa, t1_ae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,xA,Iy,zw->IA', v_aaaa, t1_ae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xA,Izuv,ywuv->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuvA,Istu,zvtswy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuvA,Istu,zvtsyw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuvA,Istu,zvtwsy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuvA,Istu,zvtwys->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuvA,Istu,zvtysw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuvA,Istu,zvtyws->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xuvA,Istw,zvtsyu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,xuvA,Isty,zvtsuw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,xuvA,Isty,zvtswu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,xuvA,Isty,zvtusw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,xuvA,Isty,zvtuws->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,xuvA,Isty,zvtwsu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,xuvA,Isty,zvtwus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuvA,Isut,zvtswy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuvA,Isut,zvtsyw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuvA,Isut,zvtwsy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,xuvA,Isut,zvtwys->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuvA,Isut,zvtysw->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuvA,Isut,zvtyws->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,xuvA,Isuw,zvsy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,xuvA,Isuy,zvws->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,xuvA,Iswt,zvtsuy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,xuvA,Iswt,zvtsyu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,xuvA,Iswt,zvtusy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,xuvA,Iswt,zvtuys->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,xuvA,Iswt,zvtysu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,xuvA,Iswt,zvtyus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xuvA,Iswu,zvsy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xuvA,Iswy,zvus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xuvA,Isyt,zvtwus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xuvA,Isyu,zvws->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xuvA,Isyw,zvsu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xuvA,Iu,zvwy->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,xuvA,Ivst,zstwuy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xuvA,Iw,zvuy->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xuvA,Iy,zvwu->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xuvA,Izst,ywuvts->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,xuvA,Izsu,ywvs->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xuvA,Izus,ywvs->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,zxuA,Iuvs,ywvs->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,zxuA,Ivsw,yvsu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,zxuA,Ivsy,wvus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,zxuA,Ivws,yvus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,zxuA,Ivwy,uv->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,zxuA,Ivys,wvus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,zxuA,Ivyw,uv->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,zxuA,Iw,yu->IA', v_aaaa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,zxuA,Iy,wu->IA', v_aaaa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)

    chunk_size = mr_adc_integrals.calculate_chunk_size(mr_adc, nextern, [ncore, ncore, nextern], 2)
    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Molecular Orbitals Energies
        e_extern = mr_adc.mo_energy.e[s_chunk:f_chunk]

        ## Amplitudes
        t1_ccee = mr_adc.t1.ccee[:,:,s_chunk:f_chunk]

        temp =- einsum('A,IiAa,ia->IA', e_extern, t1_ccee, t1_ce, optimize = einsum_type)
        temp -= einsum('A,IiAa,ixay,yx->IA', e_extern, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('A,IiAa,ixya,yx->IA', e_extern, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('A,iIAa,ia->IA', e_extern, t1_ccee, t1_ce, optimize = einsum_type)
        temp += 1/2 * einsum('A,iIAa,ixay,yx->IA', e_extern, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('A,iIAa,ixya,yx->IA', e_extern, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)

        ## Molecular Orbitals Energies
        e_extern = mr_adc.mo_energy.e

        temp -= 2 * einsum('a,ia,IiAa->IA', e_extern, t1_ce, t1_ccee, optimize = einsum_type)
        temp -= 2 * einsum('a,ixay,IiAa,yx->IA', e_extern, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += einsum('a,ixya,IiAa,yx->IA', e_extern, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += einsum('a,ia,iIAa->IA', e_extern, t1_ce, t1_ccee, optimize = einsum_type)
        temp += einsum('a,ixay,iIAa,yx->IA', e_extern, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('a,ixya,iIAa,yx->IA', e_extern, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 2 * einsum('ia,IiAa->IA', h_ce, t1_ccee, optimize = einsum_type)
        temp += einsum('ia,iIAa->IA', h_ce, t1_ccee, optimize = einsum_type)
        temp += einsum('I,IiAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
        temp -= 1/2 * einsum('I,iIAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
        temp += einsum('I,IiAa,ixay,yx->IA', e_core, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('I,iIAa,ixay,yx->IA', e_core, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
        temp += 1/4 * einsum('I,iIAa,ixya,yx->IA', e_core, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('I,IiAa,ixya,yx->IA', e_core, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
        temp -= 2 * einsum('IiAa,iaxy,yx->IA', t1_ccee, v_ceaa, rdm_ca, optimize = einsum_type)
        temp += einsum('IiAa,ixya,xy->IA', t1_ccee, v_caae, rdm_ca, optimize = einsum_type)
        temp += einsum('xy,ixaz,IiAa,yz->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('xy,ixza,IiAa,yz->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= einsum('xy,izax,IiAa,yz->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xy,izxa,IiAa,yz->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= einsum('xyzw,iuax,IiAa,zuwy->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,iuxa,IiAa,zuwy->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp += einsum('xyzw,ixau,IiAa,zuwy->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xyzw,ixua,IiAa,zuwy->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp += einsum('iIAa,iaxy,yx->IA', t1_ccee, v_ceaa, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('iIAa,ixya,xy->IA', t1_ccee, v_caae, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('xy,ixaz,iIAa,yz->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/4 * einsum('xy,ixza,iIAa,yz->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xy,izax,iIAa,yz->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,izxa,iIAa,yz->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,iuax,iIAa,zuwy->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,iuxa,iIAa,zuwy->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xyzw,ixau,iIAa,zuwy->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,ixua,iIAa,zuwy->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp += 2 * einsum('ijAa,iIja->IA', t1_ccee, v_ccce, optimize = einsum_type)
        temp -= einsum('ijAa,jIia->IA', t1_ccee, v_ccce, optimize = einsum_type)
        temp += einsum('i,IiAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
        temp += einsum('i,ia,IiAa->IA', e_core, t1_ce, t1_ccee, optimize = einsum_type)
        temp += einsum('i,IiAa,ixay,xy->IA', e_core, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('i,IiAa,ixya,xy->IA', e_core, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
        temp += einsum('i,ixay,IiAa,xy->IA', e_core, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('i,ixya,IiAa,xy->IA', e_core, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('i,iIAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
        temp -= 1/2 * einsum('i,ia,iIAa->IA', e_core, t1_ce, t1_ccee, optimize = einsum_type)
        temp -= 1/2 * einsum('i,iIAa,ixay,xy->IA', e_core, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
        temp += 1/4 * einsum('i,iIAa,ixya,xy->IA', e_core, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('i,ixay,iIAa,xy->IA', e_core, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/4 * einsum('i,ixya,iIAa,xy->IA', e_core, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(t1_ccee)

    chunk_size = mr_adc_integrals.calculate_chunk_size(mr_adc, nextern, [ncore, ncas, nextern], 2)
    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Molecular Orbitals Energies
        e_extern = mr_adc.mo_energy.e[s_chunk:f_chunk]

        ## Amplitudes
        t1_caee = mr_adc.t1.caee[:,:,:,s_chunk:f_chunk]

        temp  = 1/4 * einsum('A,IxaA,ya,xy->IA', e_extern, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
        temp += 1/4 * einsum('A,IxaA,yzwa,xwzy->IA', e_extern, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)

        ## Molecular Orbitals Energies
        e_extern = mr_adc.mo_energy.e

        temp += 1/2 * einsum('a,xa,IyaA,xy->IA', e_extern, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('a,xyza,IwaA,zwxy->IA', e_extern, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('ixaA,iIya,xy->IA', t1_caee, v_ccae, rdm_ca, optimize = einsum_type)
        temp += einsum('ixaA,Iyai,xy->IA', t1_caee, v_caec, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xa,IyaA,xy->IA', h_ae, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('I,IxaA,ya,xy->IA', e_core, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('I,IxaA,yzwa,xwzy->IA', e_core, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,IxaA,za,yz->IA', h_aa, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,IxaA,zwua,yuwz->IA', h_aa, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,xa,IzaA,yz->IA', h_aa, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,xzwa,IuaA,yzwu->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('xy,zwxa,IuaA,yuzw->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,IxaA,ua,zuwy->IA', v_aaaa, t1_caee, t1_ae, rdm_ccaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,IxaA,uvsa,zvuswy->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,IxaA,uvsa,zvusyw->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,IxaA,uvsa,zvuwsy->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,IxaA,uvsa,zvuwys->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,IxaA,uvsa,zvuysw->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,IxaA,uvsa,zvuyws->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,IxaA,uvza,ywvu->IA', v_aaaa, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,uvxa,IsaA,zuvwys->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,uxva,IsaA,zvswuy->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,xa,IuaA,zuwy->IA', v_aaaa, t1_ae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,xuva,IsaA,zvsuwy->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,xuva,IsaA,zvsuyw->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,xuva,IsaA,zvswuy->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,xuva,IsaA,zvswyu->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,xuva,IsaA,zvsyuw->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,xuva,IsaA,zvsywu->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,zxua,IvaA,ywvu->IA', v_aaaa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('IxaA,yzwa,xzwy->IA', t1_caee, v_aaae, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(t1_caee)

    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Molecular Orbitals Energies
        e_extern = mr_adc.mo_energy.e[s_chunk:f_chunk]

        ## Amplitudes
        t1_caee = mr_adc.t1.caee[:,:,s_chunk:f_chunk]

        temp =- 1/2 * einsum('A,IxAa,ya,xy->IA', e_extern, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('A,IxAa,yzwa,xwzy->IA', e_extern, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)

        ## Molecular Orbitals Energies
        e_extern = mr_adc.mo_energy.e

        temp -= einsum('a,xa,IyAa,xy->IA', e_extern, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= einsum('a,xyza,IwAa,zwxy->IA', e_extern, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('ixAa,Iyai,xy->IA', t1_caee, v_caec, rdm_ca, optimize = einsum_type)
        temp += einsum('ixAa,iIya,xy->IA', t1_caee, v_ccae, rdm_ca, optimize = einsum_type)
        temp -= einsum('xa,IyAa,xy->IA', h_ae, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= einsum('IxAa,yzwa,xzwy->IA', t1_caee, v_aaae, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('I,IxAa,ya,xy->IA', e_core, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('I,IxAa,yzwa,xwzy->IA', e_core, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xy,IxAa,za,yz->IA', h_aa, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xy,IxAa,zwua,yuwz->IA', h_aa, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xy,xa,IzAa,yz->IA', h_aa, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xy,xzwa,IuAa,yzwu->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xy,zwxa,IuAa,yuzw->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xy,zxwa,IuAa,yzuw->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,IxAa,ua,zuwy->IA', v_aaaa, t1_caee, t1_ae, rdm_ccaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,IxAa,uvsa,zvuswy->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,IxAa,uvsa,zvusyw->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,IxAa,uvsa,zvuwsy->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp += 1/3 * einsum('xyzw,IxAa,uvsa,zvuwys->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,IxAa,uvsa,zvuysw->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,IxAa,uvsa,zvuyws->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,IxAa,uvza,ywvu->IA', v_aaaa, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xyzw,uvxa,IsAa,zuvwys->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,uxva,IsAa,zvswuy->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,xa,IuAa,zuwy->IA', v_aaaa, t1_ae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,xuva,IsAa,zvsuwy->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,xuva,IsAa,zvsuyw->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,xuva,IsAa,zvswuy->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/3 * einsum('xyzw,xuva,IsAa,zvswyu->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,xuva,IsAa,zvsyuw->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,xuva,IsAa,zvsywu->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,zxua,IvAa,ywvu->IA', v_aaaa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(t1_caee)

    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Amplitudes
        t1_caee = mr_adc.t1.caee[:,:,:,s_chunk:f_chunk]

        temp =- 1/4 * einsum('xy,zxwa,IuaA,yzuw->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(t1_caee)

    chunk_size = mr_adc_integrals.calculate_chunk_size(mr_adc, nextern, [ncas, ncas, nextern], 2)
    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Amplitudes
        t1_aaee = mr_adc.t1.aaee[:,:,s_chunk:f_chunk]

        temp  = 1/2 * einsum('xyAa,Izaw,xyzw->IA', t1_aaee, v_caea, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(t1_aaee)

    chunk_size = mr_adc_integrals.calculate_chunk_size(mr_adc, nextern, [ncore, ncore, nextern], 2)
    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Two-electron integrals
        v_ccee = mr_adc.v2e.ccee[:,:,s_chunk:f_chunk]

        temp =- 1/2 * einsum('ixya,iIAa,xy->IA', t1_caae, v_ccee, rdm_ca, optimize = einsum_type)
        temp += einsum('ia,iIAa->IA', t1_ce, v_ccee, optimize = einsum_type)
        temp += einsum('ixay,iIAa,xy->IA', t1_caea, v_ccee, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(v_ccee)

    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Two-electron integrals
        v_ceec = mr_adc.v2e.ceec[:,s_chunk:f_chunk]

        temp  = einsum('ixya,IAai,xy->IA', t1_caae, v_ceec, rdm_ca, optimize = einsum_type)
        temp -= 2 * einsum('ia,IAai->IA', t1_ce, v_ceec, optimize = einsum_type)
        temp -= 2 * einsum('ixay,IAai,xy->IA', t1_caea, v_ceec, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(v_ceec)

    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Two-electron integrals
        v_cece = mr_adc.v2e.cece[:,s_chunk:f_chunk]

        temp =- 1/2 * einsum('ixya,iAIa,yx->IA', t1_caae, v_cece, rdm_ca, optimize = einsum_type)
        temp += einsum('ia,iAIa->IA', t1_ce, v_cece, optimize = einsum_type)
        temp += einsum('ixay,iAIa,yx->IA', t1_caea, v_cece, rdm_ca, optimize = einsum_type)
        temp += einsum('ixya,IAia,yx->IA', t1_caae, v_cece, rdm_ca, optimize = einsum_type)
        temp -= 2 * einsum('ia,IAia->IA', t1_ce, v_cece, optimize = einsum_type)
        temp -= 2 * einsum('ixay,IAia,yx->IA', t1_caea, v_cece, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(v_cece)

    chunk_size = mr_adc_integrals.calculate_chunk_size(mr_adc, nextern, [ncas, ncas, nextern], 2)
    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Two-electron integrals
        v_aaee = mr_adc.v2e.aaee[:,:,s_chunk:f_chunk]

        temp  = 1/2 * einsum('Ixya,zwAa,xwyz->IA', t1_caae, v_aaee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('Ixya,zyAa,xz->IA', t1_caae, v_aaee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('Ixya,zwAa,zw,xy->IA', t1_caae, v_aaee, rdm_ca, rdm_ca, optimize = einsum_type)
        temp -= einsum('Ixay,zwAa,xwyz->IA', t1_caea, v_aaee, rdm_ccaa, optimize = einsum_type)
        temp -= einsum('Ixay,zyAa,xz->IA', t1_caea, v_aaee, rdm_ca, optimize = einsum_type)
        temp += einsum('Ixay,zwAa,zw,xy->IA', t1_caea, v_aaee, rdm_ca, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(v_aaee)

    chunk_size = mr_adc_integrals.calculate_chunk_size(mr_adc, nextern, [ncas, ncas, nextern], 2)
    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Two-electron integrals
        v_aeea = mr_adc.v2e.aeea[:,s_chunk:f_chunk]

        temp =- einsum('Ixya,yAaz,xz->IA', t1_caae, v_aeea, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('Ixya,zAaw,xzwy->IA', t1_caae, v_aeea, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('Ixya,zAaw,wz,xy->IA', t1_caae, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('Ixay,yAaz,xz->IA', t1_caea, v_aeea, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('Ixay,zAaw,xzyw->IA', t1_caea, v_aeea, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('Ixay,zAaw,wz,xy->IA', t1_caea, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(v_aeea)

    chunk_size = mr_adc_integrals.calculate_chunk_size(mr_adc, nextern, [ncas, ncas, nextern], 2)
    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Two-electron integrals
        v_caee = mr_adc.v2e.caee[:,:,s_chunk:f_chunk]

        temp  = einsum('Iixa,ixAa->IA', t1_ccae, v_caee, optimize = einsum_type)
        temp -= 2 * einsum('iIxa,ixAa->IA', t1_ccae, v_caee, optimize = einsum_type)
        temp -= 1/2 * einsum('Iixa,iyAa,xy->IA', t1_ccae, v_caee, rdm_ca, optimize = einsum_type)
        temp += einsum('iIxa,iyAa,xy->IA', t1_ccae, v_caee, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(v_caee)

    chunk_size = mr_adc_integrals.calculate_chunk_size(mr_adc, nextern, [ncas, ncas, nextern], 2)
    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Two-electron integrals
        v_caee = mr_adc.v2e.caee[:,:,:,s_chunk:f_chunk]

        temp  = 1/2 * einsum('xa,IyaA,xy->IA', t1_ae, v_caee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xyza,IwaA,zwxy->IA', t1_aaae, v_caee, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(v_caee)

    chunk_size = mr_adc_integrals.calculate_chunk_size(mr_adc, nextern, [ncas, ncas, nextern], 2)
    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Two-electron integrals
        v_ceae = mr_adc.v2e.ceae[:,s_chunk:f_chunk]

        temp =- einsum('xa,IAya,xy->IA', t1_ae, v_ceae, rdm_ca, optimize = einsum_type)
        temp -= einsum('xyza,IAwa,zwxy->IA', t1_aaae, v_ceae, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(v_ceae)

    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Two-electron integrals
        v_ceae = mr_adc.v2e.ceae[:,:,:,s_chunk:f_chunk]

        temp  = 1/2 * einsum('xa,IayA,xy->IA', t1_ae, v_ceae, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xyza,IawA,zwxy->IA', t1_aaae, v_ceae, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(v_ceae)

    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Two-electron integrals
        v_ceea = mr_adc.v2e.ceea[:,:,s_chunk:f_chunk]

        temp =- 2 * einsum('Iixa,iaAx->IA', t1_ccae, v_ceea, optimize = einsum_type)
        temp += einsum('iIxa,iaAx->IA', t1_ccae, v_ceea, optimize = einsum_type)
        temp += einsum('Iixa,iaAy,xy->IA', t1_ccae, v_ceea, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('iIxa,iaAy,xy->IA', t1_ccae, v_ceea, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(v_ceea)

    for s_chunk in range(0, nextern, chunk_size):
        f_chunk = s_chunk + chunk_size

        ## Two-electron integrals
        v_ceea = mr_adc.v2e.ceea[:,s_chunk:f_chunk]

        temp =- einsum('xa,IAay,xy->IA', t1_ae, v_ceea, rdm_ca, optimize = einsum_type)
        temp -= einsum('xyza,IAaw,zwxy->IA', t1_aaae, v_ceea, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
    del(v_ceea)

    chunk_size = mr_adc_integrals.calculate_chunk_sizes(mr_adc, ncore, [nextern, nextern, nextern],
                                                                       [ncore, nextern, nextern])
    for s_chunk in range(0, ncore, chunk_size):
        f_chunk = s_chunk + chunk_size
        if interface.with_df:
            v_ceee = mr_adc_integrals.get_oeee_df(mr_adc, mr_adc.v2e.Lce, mr_adc.v2e.Lee, s_chunk, chunk_size)

        else:
            v_ceee = mr_adc_integrals.unpack_v2e_oeee(mr_adc.v2e.ceee[s_chunk:f_chunk], nextern)

        ## Amplitudes
        t1_ccee = mr_adc.t1.ccee[:,s_chunk:f_chunk]

        V1 += einsum('Iiab,iaAb->IA', t1_ccee, v_ceee, optimize = einsum_type)
        V1 -= 2 * einsum('Iiab,ibAa->IA', t1_ccee, v_ceee, optimize = einsum_type)
    del(v_ceee, t1_ccee)

    chunk_size = mr_adc_integrals.calculate_chunk_sizes(mr_adc, ncas, [nextern, nextern, nextern],
                                                                      [ncore, nextern, nextern])
    for s_v_chunk in range(0, ncas, chunk_size):
        f_v_chunk = s_v_chunk + chunk_size
        if interface.with_df:
            v_aeee = mr_adc_integrals.get_oeee_df(mr_adc, mr_adc.v2e.Lae, mr_adc.v2e.Lee, s_v_chunk, chunk_size)
        else:
            v_aeee = mr_adc_integrals.unpack_v2e_oeee(mr_adc.v2e.aeee[s_v_chunk:f_v_chunk], nextern)

        for s_t_chunk in range(0, ncas, chunk_size):
            f_t_chunk = s_t_chunk + chunk_size

            ## Amplitudes
            t1_caee = mr_adc.t1.caee[:,s_t_chunk:f_t_chunk]

            ## Reduced density matrices
            rdm_ca = mr_adc.rdm.ca[s_t_chunk:f_t_chunk,s_v_chunk:f_v_chunk]

            V1 += 1/2 * einsum('Ixab,yaAb,xy->IA', t1_caee, v_aeee, rdm_ca, optimize = einsum_type)
            V1 -= einsum('Ixab,ybAa,xy->IA', t1_caee, v_aeee, rdm_ca, optimize = einsum_type)
    del(v_aeee, t1_caee, rdm_ca)

    ## Molecular Orbitals Energies
    e_core = mr_adc.mo_energy.c

    ## Compute denominators
    d_ai = (e_extern[:,None] - e_core)
    d_ai = d_ai**(-1)

    # Compute T2[0'] t2_ce amplitudes
    t2_ce = einsum("ai,ia->ia", d_ai, V1, optimize = einsum_type)

    return t2_ce
