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
#                  Ilia M. Mazin <ilia.mazin@gmail.com>
#

import numpy as np
from functools import reduce

import prism.mr_adc_intermediates as mr_adc_intermediates
import prism.mr_adc_overlap as mr_adc_overlap
import prism.mr_adc_integrals as mr_adc_integrals

import prism.lib.logger as logger
import prism.lib.tools as tools

### warning import for t2_aa
import warnings
###

def compute_amplitudes(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.info("\nComputing NEVPT2 amplitudes...")

    # First-order amplitudes
    compute_t1_amplitudes(mr_adc)

    # Second-order amplitudes
    compute_t2_amplitudes(mr_adc)

    # Compute CVS amplitudes and remove non-CVS core integrals, amplitudes and unnecessary RDMs
    if mr_adc.method_type in ("cvs-ip", "cvs-ee"):
        compute_cvs_amplitudes(mr_adc)

    mr_adc.log.timer("computing amplitudes", *cput0)

def compute_t1_amplitudes(mr_adc):

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nelecas = mr_adc.nelecas
    nextern = mr_adc.nextern

    e_0p, e_p1p, e_m1p, e_0, e_p1, e_m1, e_p2, e_m2 = (0.0,) * 8

    # Create temporary files
    if mr_adc.outcore_expensive_tensors:
        mr_adc.tmpfile.t1 = tools.create_temp_file(mr_adc) # Non-core indices' amplitudes
        mr_adc.tmpfile.ct1 = tools.create_temp_file(mr_adc) # Core indices' amplitudes
    else:
        mr_adc.tmpfile.t1 = None
        mr_adc.tmpfile.ct1 = None

    # First-order amplitudes
    if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
        if ncore > 0 and nextern > 0 and ncas > 0:
            e_0p, mr_adc.t1.ce, mr_adc.t1.caea, mr_adc.t1.caae = compute_t1_0p(mr_adc)
        else:
            mr_adc.t1.ce = np.zeros((ncore, nextern))
            mr_adc.t1.caea = np.zeros((ncore, ncas, nextern, ncas))
            mr_adc.t1.caae = np.zeros((ncore, ncas, ncas, nextern))

        if ncore > 0 and ncas > 0:
            e_p1p, mr_adc.t1.ca, mr_adc.t1.caaa = compute_t1_p1p(mr_adc)
        else:
            mr_adc.t1.ca = np.zeros((ncore, ncas))
            mr_adc.t1.caaa = np.zeros((ncore, ncas, ncas, ncas))

        if nextern > 0 and ncas > 0:
            e_m1p, mr_adc.t1.ae, mr_adc.t1.aaae = compute_t1_m1p(mr_adc)
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
            e_0, mr_adc.t1.ccee = compute_t1_0(mr_adc)
        else:
            mr_adc.t1.ccee = np.zeros((ncore, ncore, nextern, nextern))

        if ncore > 0 and nextern > 0 and ncas > 0:
            e_p1, mr_adc.t1.ccae = compute_t1_p1(mr_adc)
        else:
            mr_adc.t1.ccae = np.zeros((ncore, ncore, ncas, nextern))

        if ncore > 0 and nextern > 0 and ncas > 0 and sum(nelecas) > 0:
            e_m1, mr_adc.t1.caee = compute_t1_m1(mr_adc)
        else:
            mr_adc.t1.caee = np.zeros((ncore, ncas, nextern, nextern))

        if ncore > 0 and ncas > 0:
            e_p2, mr_adc.t1.ccaa = compute_t1_p2(mr_adc)
        else:
            mr_adc.t1.ccaa = np.zeros((ncore, ncore, ncas, ncas))

        if nextern > 0 and ncas > 0 and sum(nelecas) > 0:
            e_m2, mr_adc.t1.aaee = compute_t1_m2(mr_adc)
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

    mr_adc.log.log("\nCASSCF reference energy:                     %20.12f" % mr_adc.e_casscf)
    mr_adc.log.info("PC-NEVPT2 correlation energy:                %20.12f" % e_corr)
    mr_adc.log.log("Total PC-NEVPT2 energy:                      %20.12f" % e_tot)

def compute_t2_amplitudes(mr_adc):

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    approx_trans_moments = mr_adc.approx_trans_moments

    # Approximate second-order amplitudes
    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
#    if mr_adc.method in ("mr-adc(1)"): ##checking M_00 block

        if (ncore > 0) and (nextern > 0) and not (approx_trans_moments):
            mr_adc.t2.ce = compute_t2_0p_singles(mr_adc)

        else:
            mr_adc.t2.ce = np.zeros((ncore, nextern))

        # EE and CVS-EE amplitudes 
        if mr_adc.method_type == "cvs-ee":
            mr_adc.t2.ae = compute_t2_m1p_singles(mr_adc)
            ###TODO - finish_t2_0pp function!
            mr_adc.t2.aa = compute_t2_0pp_singles(mr_adc)  
            ###TODO - testing for t2_ca (SKS correct, but slight difference in V makes t2_ca not an exact match to SO code)
            mr_adc.t2.ca = compute_t2_p1p_singles(mr_adc) 

        mr_adc.t2.aaae = np.zeros((ncas, ncas, ncas, nextern))
    else:
        mr_adc.t2.ce = np.zeros((ncore, nextern))

def compute_cvs_amplitudes(mr_adc):
    'Create CVS amplitudes tensors and remove core integrals, core amplitudes and RDMs not used in CVS calculations'

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing CVS amplitudes...")

    if mr_adc.outcore_expensive_tensors:
        mr_adc.tmpfile.xt1 = tools.create_temp_file(mr_adc)
    else:
        mr_adc.tmpfile.xt1 = None
    tmpfile = mr_adc.tmpfile.xt1

    if mr_adc.method_type in ("cvs-ip", "cvs-ee"):

        # Variables from kernel
        ncvs = mr_adc.ncvs
        nval = mr_adc.nval
        ncore = mr_adc.ncore
        ncas = mr_adc.ncas
        nextern = mr_adc.nextern

        if mr_adc.method_type == "cvs-ip":
            del(mr_adc.rdm.ccccaaaa)

        if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
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

            mr_adc.t1.xxee = tools.create_dataset('xxee', tmpfile, (ncvs, ncvs, nextern, nextern))
            mr_adc.t1.xvee = tools.create_dataset('xvee', tmpfile, (ncvs, nval, nextern, nextern))
            mr_adc.t1.vxee = tools.create_dataset('vxee', tmpfile, (nval, ncvs, nextern, nextern))
            mr_adc.t1.vvee = tools.create_dataset('vvee', tmpfile, (nval, nval, nextern, nextern))

            mr_adc.t1.xaee = tools.create_dataset('xaee', tmpfile, (ncvs, ncas, nextern, nextern))
            mr_adc.t1.vaee = tools.create_dataset('vaee', tmpfile, (nval, ncas, nextern, nextern))

            chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncore, nextern])
            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("t1.xxee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.t1.xxee[:,:,s_chunk:f_chunk] = mr_adc.t1.ccee[:ncvs, :ncvs, s_chunk:f_chunk, :]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS t1.xxee", *cput1)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("t1.xvee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.t1.xvee[:,:,s_chunk:f_chunk] = mr_adc.t1.ccee[:ncvs, ncvs:, s_chunk:f_chunk, :]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS t1.xvee", *cput1)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("t1.vxee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.t1.vxee[:,:,s_chunk:f_chunk] = mr_adc.t1.ccee[ncvs:, :ncvs, s_chunk:f_chunk, :]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS t1.vxee", *cput1)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("t1.vvee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.t1.vvee[:,:,s_chunk:f_chunk] = mr_adc.t1.ccee[ncvs:, ncvs:, s_chunk:f_chunk, :]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS t1.vvee", *cput1)
            del(mr_adc.t1.ccee)

            chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncas, nextern])
            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("t1.xaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.t1.xaee[:,:,s_chunk:f_chunk] = mr_adc.t1.caee[:ncvs, :, s_chunk:f_chunk]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS t1.xaee", *cput1)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("t1.vaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.t1.vaee[:,:,s_chunk:f_chunk] = mr_adc.t1.caee[ncvs:, :, s_chunk:f_chunk]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS t1.vaee", *cput1)
            del(mr_adc.t1.caee)

    if mr_adc.outcore_expensive_tensors:
        mr_adc.tmpfile.ct1.close()

    mr_adc.log.timer("computing CVS amplitudes", *cput0)

def compute_t1_0(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing T[0]^(1) amplitudes...")

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    ctmpfile = mr_adc.tmpfile.ct1

    # Variables from kernel
    ncore = mr_adc.ncore
    nextern = mr_adc.nextern

    ## Molecular Orbitals Energies
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    # Compute denominators
    d_ij = e_core[:,None] + e_core

    t1_ccee = tools.create_dataset('ccee', ctmpfile, (ncore, ncore, nextern, nextern))
    chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncore, nextern], ntensors = 3)

    e_0 = 0.0
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.ccee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

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
        tools.flush(ctmpfile)
        mr_adc.log.timer_debug("computing t1.ccee", *cput1)

    mr_adc.log.extra("Norm of T[0]^(1):                            %20.12f" % np.linalg.norm(t1_ccee))
    mr_adc.log.info("Correlation energy [0]:                      %20.12f" % e_0)
    mr_adc.log.timer("computing T[0]^(1) amplitudes", *cput0)

    return e_0, t1_ccee

def compute_t1_p1(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing T[+1]^(1) amplitudes...")

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

    mr_adc.log.extra("Norm of T[+1]^(1):                           %20.12f" % np.linalg.norm(t1_ccae))
    mr_adc.log.info("Correlation energy [+1]:                     %20.12f" % e_p1)
    mr_adc.log.timer("computing T[+1]^(1) amplitudes", *cput0)

    return e_p1, t1_ccae

def compute_t1_m1(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing T[-1]^(1) amplitudes...")

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    ctmpfile = mr_adc.tmpfile.ct1

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

    t1_caee = tools.create_dataset('caee', ctmpfile, (ncore, ncas, nextern, nextern))
    chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncas, nextern], ntensors = 2)

    e_m1 = 0.0
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

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
        tools.flush(ctmpfile)
        mr_adc.log.timer_debug("computing t1.caee", *cput1)

    mr_adc.log.extra("Norm of T[-1]^(1):                           %20.12f" % np.linalg.norm(t1_caee))
    mr_adc.log.info("Correlation energy [-1]:                     %20.12f" % e_m1)
    mr_adc.log.timer("computing T[-1]^(1) amplitudes", *cput0)

    return e_m1, t1_caee

def compute_t1_p2(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing T[+2]^(1) amplitudes...")

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

    mr_adc.log.extra("Norm of T[+2]^(1):                           %20.12f" % np.linalg.norm(t1_ccaa))
    mr_adc.log.info("Correlation energy [+2]:                     %20.12f" % e_p2)
    mr_adc.log.timer("computing T[+2]^(1) amplitudes", *cput0)

    return e_p2, t1_ccaa

def compute_t1_m2(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing T[-2]^(1) amplitudes...")

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    tmpfile = mr_adc.tmpfile.t1

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
    t1_aaee = tools.create_dataset('aaee', tmpfile, (ncas, ncas, nextern, nextern))
    chunks = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 3)

    e_m2 = 0.0
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

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
        tools.flush(tmpfile)
        mr_adc.log.timer_debug("computing t1.aaee", *cput1)

    mr_adc.log.extra("Norm of T[-2]^(1):                           %20.12f" % np.linalg.norm(t1_aaee))
    mr_adc.log.info("Correlation energy [-2]:                     %20.12f" % e_m2)
    mr_adc.log.timer("computing T[-2]^(1) amplitudes", *cput0)

    return e_m2, t1_aaee

def compute_t1_0p(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing T[0']^(1) amplitudes...")

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

    mr_adc.log.extra("Norm of T[0']^(1):                           %20.12f" % (np.linalg.norm(t1_ce) +
                                                                              np.linalg.norm(t1_caea)))
    mr_adc.log.info("Correlation energy [0']:                     %20.12f" % e_0p)
    mr_adc.log.timer("computing T[0']^(1) amplitudes", *cput0)

    return e_0p, t1_ce, t1_caea, t1_caae

def compute_t1_p1p(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing T[+1']^(1) amplitudes...")

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
    
##    NEW_V2_aa_aa =- 1/2 * einsum('IY,XZ->IXYZ', h_ca, rdm_ca, optimize = einsum_type)
##    NEW_V2_aa_aa += 1/2 * einsum('IZ,XY->IXYZ', h_ca, rdm_ca, optimize = einsum_type)
##    NEW_V2_aa_aa -= 1/6 * einsum('Ix,XxYZ->IXYZ', h_ca, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_aa_aa += 1/6 * einsum('Ix,XxZY->IXYZ', h_ca, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_aa_aa -= 1/2 * einsum('IYxZ,Xx->IXYZ', v_caaa, rdm_ca, optimize = einsum_type)
##    NEW_V2_aa_aa -= 1/2 * einsum('IYxy,XyZx->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_aa_aa += 1/2 * einsum('IZxY,Xx->IXYZ', v_caaa, rdm_ca, optimize = einsum_type)
##    NEW_V2_aa_aa += 1/2 * einsum('IZxy,XyYx->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_aa_aa += 1/6 * einsum('IxyY,XxZy->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_aa_aa -= 1/6 * einsum('IxyY,XxyZ->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_aa_aa -= 1/6 * einsum('IxyZ,XxYy->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_aa_aa += 1/6 * einsum('IxyZ,XxyY->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_aa_aa -= 1/6 * einsum('Ixyz,XxzYZy->IXYZ', v_caaa, rdm_cccaaa, optimize = einsum_type)
##    NEW_V2_aa_aa += 1/6 * einsum('Ixyz,XxzZYy->IXYZ', v_caaa, rdm_cccaaa, optimize = einsum_type)

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

##    NEW_V2_ab_ba =- 1/2 * einsum('IY,XZ->IXYZ', h_ca, rdm_ca, optimize = einsum_type)
##    NEW_V2_ab_ba += 1/6 * einsum('Ix,XxYZ->IXYZ', h_ca, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_ab_ba += 1/3 * einsum('Ix,XxZY->IXYZ', h_ca, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_ab_ba -= 1/2 * einsum('IYxZ,Xx->IXYZ', v_caaa, rdm_ca, optimize = einsum_type)
##    NEW_V2_ab_ba -= 1/2 * einsum('IYxy,XyZx->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_ab_ba += 1/3 * einsum('IxyY,XxZy->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_ab_ba += 1/6 * einsum('IxyY,XxyZ->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_ab_ba += 1/6 * einsum('IxyZ,XxYy->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_ab_ba += 1/3 * einsum('IxyZ,XxyY->IXYZ', v_caaa, rdm_ccaa, optimize = einsum_type)
##    NEW_V2_ab_ba -= 1/6 * einsum('Ixyz,XxzYyZ->IXYZ', v_caaa, rdm_cccaaa, optimize = einsum_type)
##    NEW_V2_ab_ba += 1/6 * einsum('Ixyz,XxzZYy->IXYZ', v_caaa, rdm_cccaaa, optimize = einsum_type)
##    NEW_V2_ab_ba -= 1/6 * einsum('Ixyz,XxzZyY->IXYZ', v_caaa, rdm_cccaaa, optimize = einsum_type)
##    NEW_V2_ab_ba -= 1/6 * einsum('Ixyz,XxzyYZ->IXYZ', v_caaa, rdm_cccaaa, optimize = einsum_type)
##    NEW_V2_ab_ba -= 1/6 * einsum('Ixyz,XxzyZY->IXYZ', v_caaa, rdm_cccaaa, optimize = einsum_type)
##
##    print('V2_t1_p1p check! does the reindex affect things strongly?')
##    print('V2_aa_aa: ', np.linalg.norm(-NEW_V2_aa_aa - V2_aa_aa)) ##NEW_aa_aa is the negative of Carlos' aa_aa
##    print('V2_ab_ba: ', np.linalg.norm(NEW_V2_ab_ba - V2_ab_ba))

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

    mr_adc.log.extra("Norm of T[+1']^(1):                          %20.12f" % (np.linalg.norm(t1_ca) +
                                                                              np.linalg.norm(t1_caaa)))
    mr_adc.log.info("Correlation energy [+1']:                    %20.12f" % e_p1p)
    mr_adc.log.timer("computing T[+1']^(1) amplitudes", *cput0)

    return e_p1p, t1_ca, t1_caaa

def compute_t1_m1p(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing T[-1']^(1) amplitudes...")

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

    mr_adc.log.extra("Norm of T[-1']^(1):                          %20.12f" % (np.linalg.norm(t1_ae) +
                                                                              np.linalg.norm(t1_aaae)))
    mr_adc.log.info("Correlation energy [-1']:                    %20.12f" % e_m1p)
    mr_adc.log.timer("computing T[-1']^(1) amplitudes", *cput0)

    return e_m1p, t1_ae, t1_aaae

def compute_t2_0p_singles(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("Computing T[0']^(2) amplitudes...")

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

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncore, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.ccee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

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
        mr_adc.log.timer_debug("contracting t1.ccee", *cput1)
    del(t1_ccee)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

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
        mr_adc.log.timer_debug("contracting t1.caee", *cput1)
    del(t1_caee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

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
        mr_adc.log.timer_debug("contracting t1.caee", *cput1)
    del(t1_caee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Amplitudes
        t1_caee = mr_adc.t1.caee[:,:,:,s_chunk:f_chunk]

        temp =- 1/4 * einsum('xy,zxwa,IuaA,yzuw->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting t1.caee", *cput1)
    del(t1_caee)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Amplitudes
        t1_aaee = mr_adc.t1.aaee[:,:,s_chunk:f_chunk]

        temp  = 1/2 * einsum('xyAa,Izaw,xyzw->IA', t1_aaee, v_caea, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting t1.aaee", *cput1)
    del(t1_aaee)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncore, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ccee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_ccee = mr_adc.v2e.ccee[:,:,s_chunk:f_chunk]

        temp =- 1/2 * einsum('ixya,iIAa,xy->IA', t1_caae, v_ccee, rdm_ca, optimize = einsum_type)
        temp += einsum('ia,iIAa->IA', t1_ce, v_ccee, optimize = einsum_type)
        temp += einsum('ixay,iIAa,xy->IA', t1_caea, v_ccee, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.ccee", *cput1)
    del(v_ccee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceec [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_ceec = mr_adc.v2e.ceec[:,s_chunk:f_chunk]

        temp  = einsum('ixya,IAai,xy->IA', t1_caae, v_ceec, rdm_ca, optimize = einsum_type)
        temp -= 2 * einsum('ia,IAai->IA', t1_ce, v_ceec, optimize = einsum_type)
        temp -= 2 * einsum('ixay,IAai,xy->IA', t1_caea, v_ceec, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.ceec", *cput1)
    del(v_ceec)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.cece [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_cece = mr_adc.v2e.cece[:,s_chunk:f_chunk]

        temp =- 1/2 * einsum('ixya,iAIa,yx->IA', t1_caae, v_cece, rdm_ca, optimize = einsum_type)
        temp += einsum('ia,iAIa->IA', t1_ce, v_cece, optimize = einsum_type)
        temp += einsum('ixay,iAIa,yx->IA', t1_caea, v_cece, rdm_ca, optimize = einsum_type)
        temp += einsum('ixya,IAia,yx->IA', t1_caae, v_cece, rdm_ca, optimize = einsum_type)
        temp -= 2 * einsum('ia,IAia->IA', t1_ce, v_cece, optimize = einsum_type)
        temp -= 2 * einsum('ixay,IAia,yx->IA', t1_caea, v_cece, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.cece", *cput1)
    del(v_cece)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_aaee = mr_adc.v2e.aaee[:,:,s_chunk:f_chunk]

        temp  = 1/2 * einsum('Ixya,zwAa,xwyz->IA', t1_caae, v_aaee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('Ixya,zyAa,xz->IA', t1_caae, v_aaee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('Ixya,zwAa,zw,xy->IA', t1_caae, v_aaee, rdm_ca, rdm_ca, optimize = einsum_type)
        temp -= einsum('Ixay,zwAa,xwyz->IA', t1_caea, v_aaee, rdm_ccaa, optimize = einsum_type)
        temp -= einsum('Ixay,zyAa,xz->IA', t1_caea, v_aaee, rdm_ca, optimize = einsum_type)
        temp += einsum('Ixay,zwAa,zw,xy->IA', t1_caea, v_aaee, rdm_ca, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.aaee", *cput1)
    del(v_aaee)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.aeea [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_aeea = mr_adc.v2e.aeea[:,s_chunk:f_chunk]

        temp =- einsum('Ixya,yAaz,xz->IA', t1_caae, v_aeea, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('Ixya,zAaw,xzwy->IA', t1_caae, v_aeea, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('Ixya,zAaw,wz,xy->IA', t1_caae, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('Ixay,yAaz,xz->IA', t1_caea, v_aeea, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('Ixay,zAaw,xzyw->IA', t1_caea, v_aeea, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('Ixay,zAaw,wz,xy->IA', t1_caea, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.aeea", *cput1)
    del(v_aeea)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_caee = mr_adc.v2e.caee[:,:,s_chunk:f_chunk]

        temp  = einsum('Iixa,ixAa->IA', t1_ccae, v_caee, optimize = einsum_type)
        temp -= 2 * einsum('iIxa,ixAa->IA', t1_ccae, v_caee, optimize = einsum_type)
        temp -= 1/2 * einsum('Iixa,iyAa,xy->IA', t1_ccae, v_caee, rdm_ca, optimize = einsum_type)
        temp += einsum('iIxa,iyAa,xy->IA', t1_ccae, v_caee, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.caee", *cput1)
    del(v_caee)

    chunk_size = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_caee = mr_adc.v2e.caee[:,:,:,s_chunk:f_chunk]

        temp  = 1/2 * einsum('xa,IyaA,xy->IA', t1_ae, v_caee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xyza,IwaA,zwxy->IA', t1_aaae, v_caee, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.caee", *cput1)
    del(v_caee)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceae [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_ceae = mr_adc.v2e.ceae[:,s_chunk:f_chunk]

        temp =- einsum('xa,IAya,xy->IA', t1_ae, v_ceae, rdm_ca, optimize = einsum_type)
        temp -= einsum('xyza,IAwa,zwxy->IA', t1_aaae, v_ceae, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.ceae", *cput1)
    del(v_ceae)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceae [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_ceae = mr_adc.v2e.ceae[:,:,:,s_chunk:f_chunk]

        temp  = 1/2 * einsum('xa,IayA,xy->IA', t1_ae, v_ceae, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xyza,IawA,zwxy->IA', t1_aaae, v_ceae, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.ceae", *cput1)
    del(v_ceae)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceea [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_ceea = mr_adc.v2e.ceea[:,:,s_chunk:f_chunk]

        temp =- 2 * einsum('Iixa,iaAx->IA', t1_ccae, v_ceea, optimize = einsum_type)
        temp += einsum('iIxa,iaAx->IA', t1_ccae, v_ceea, optimize = einsum_type)
        temp += einsum('Iixa,iaAy,xy->IA', t1_ccae, v_ceea, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('iIxa,iaAy,xy->IA', t1_ccae, v_ceea, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.ceea", *cput1)
    del(v_ceea)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceea [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_ceea = mr_adc.v2e.ceea[:,s_chunk:f_chunk]

        temp =- einsum('xa,IAay,xy->IA', t1_ae, v_ceea, rdm_ca, optimize = einsum_type)
        temp -= einsum('xyza,IAaw,zwxy->IA', t1_aaae, v_ceea, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.ceea", *cput1)
    del(v_ceea)

    chunks = tools.calculate_double_chunks(mr_adc, ncore, [nextern, nextern, nextern],
                                                                     [ncore, nextern, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        if interface.with_df:
            v_ceee = mr_adc_integrals.get_oeee_df(mr_adc, mr_adc.v2e.Lce, mr_adc.v2e.Lee, s_chunk, f_chunk)

        else:
            v_ceee = mr_adc_integrals.unpack_v2e_oeee(mr_adc, mr_adc.v2e.ceee[s_chunk:f_chunk])

        ## Amplitudes
        t1_ccee = mr_adc.t1.ccee[:,s_chunk:f_chunk]

        V1 += einsum('Iiab,iaAb->IA', t1_ccee, v_ceee, optimize = einsum_type)
        V1 -= 2 * einsum('Iiab,ibAa->IA', t1_ccee, v_ceee, optimize = einsum_type)

        del(v_ceee, t1_ccee)
        mr_adc.log.timer_debug("contracting v2e.ceee", *cput1)

    chunks = tools.calculate_double_chunks(mr_adc, ncas, [nextern, nextern, nextern],
                                                                    [ncore, nextern, nextern], ntensors = 2)
    for i_v_chunk, (s_v_chunk, f_v_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.aeee [%i/%i], chunk [%i:%i]", i_v_chunk + 1, len(chunks), s_v_chunk, f_v_chunk)

        if interface.with_df:
            v_aeee = mr_adc_integrals.get_oeee_df(mr_adc, mr_adc.v2e.Lae, mr_adc.v2e.Lee, s_v_chunk, f_v_chunk)
        else:
            v_aeee = mr_adc_integrals.unpack_v2e_oeee(mr_adc, mr_adc.v2e.aeee[s_v_chunk:f_v_chunk])

        for s_t_chunk, f_t_chunk in chunks:

            ## Amplitudes
            t1_caee = mr_adc.t1.caee[:,s_t_chunk:f_t_chunk]

            ## Reduced density matrices
            rdm_ca = mr_adc.rdm.ca[s_t_chunk:f_t_chunk,s_v_chunk:f_v_chunk]

            V1 += 1/2 * einsum('Ixab,yaAb,xy->IA', t1_caee, v_aeee, rdm_ca, optimize = einsum_type)
            V1 -= einsum('Ixab,ybAa,xy->IA', t1_caee, v_aeee, rdm_ca, optimize = einsum_type)

        del(v_aeee, t1_caee, rdm_ca)
        mr_adc.log.timer_debug("contracting v2e.aeee", *cput1)

    ## Molecular Orbitals Energies
    e_core = mr_adc.mo_energy.c

    ## Compute denominators
    d_ai = (e_extern[:,None] - e_core)
    d_ai = d_ai**(-1)

    # Compute T2[0'] t2_ce amplitudes
    t2_ce = einsum("ai,ia->ia", d_ai, V1, optimize = einsum_type)

    mr_adc.log.extra("Norm of T[0']^(2):                          %20.12f" % np.linalg.norm(t2_ce))
    mr_adc.log.timer("computing T[0']^(2) amplitudes", *cput0)

    return t2_ce

def compute_t2_m1p_singles(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing T[-1']^(2) amplitudes...")

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
    h_ce = mr_adc.h1eff.ce
    h_ca = mr_adc.h1eff.ca
    h_ae = mr_adc.h1eff.ae
    h_aa = mr_adc.h1eff.aa

    ## Two-electron integrals
    v_aaaa = mr_adc.v2e.aaaa

    v_caca = mr_adc.v2e.caca
    
    v_cece = mr_adc.v2e.cece
    v_cace = mr_adc.v2e.cace
    
    v_caaa = mr_adc.v2e.caaa
    v_ceae = mr_adc.v2e.ceae
    v_caae = mr_adc.v2e.caae
    v_ceaa = mr_adc.v2e.ceaa
    
    v_caea = mr_adc.v2e.caea
    v_ceee = mr_adc.v2e.ceee
    v_caee = mr_adc.v2e.caee
    v_ceea = mr_adc.v2e.ceea
    
    v_aaae = mr_adc.v2e.aaae
    
    v_aeee = mr_adc.v2e.aeee
    v_aaee = mr_adc.v2e.aaee
    v_aeea = mr_adc.v2e.aeea
    v_aeae = mr_adc.v2e.aeae

    ## Amplitudes
    t1_ce = mr_adc.t1.ce
    t1_caea = mr_adc.t1.caea
    t1_caae = mr_adc.t1.caae

    t1_ca = mr_adc.t1.ca
    t1_caaa = mr_adc.t1.caaa

    t1_ae   = mr_adc.t1.ae
    t1_aaae = mr_adc.t1.aaae
    t1_aaee = mr_adc.t1.aaee

    t1_ccee = mr_adc.t1.ccee
    t1_ccae = mr_adc.t1.ccae
    t1_ccaa = mr_adc.t1.ccaa
    t1_caee = mr_adc.t1.caee

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa
    rdm_ccccaaaa = mr_adc.rdm.ccccaaaa

    # Compute K_ca matrix
    K_ca = mr_adc_intermediates.compute_K_ca(mr_adc)
    
    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_m1_12_inv_act = mr_adc_overlap.compute_S12_m1(mr_adc)

    # Compute K^{-1} matrix
    SKS = reduce(np.dot, (S_m1_12_inv_act.T, K_ca, S_m1_12_inv_act))
    evals, evecs = np.linalg.eigh(SKS)
    del(SKS)

    # Compute R.H.S. of the equation
    # V1 block: - 1/2 < Psi_0 | a^{\dag}_A a_E [V + H^{(1)}, T - T^\dag] | Psi_0 >
    V1  = 1/2 * einsum('iA,ix,Xx->XA', h_ce, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('iA,ixyz,Xxyz->XA', h_ce, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ia,ixAa,Xx->XA', h_ce, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ia,ixaA,Xx->XA', h_ce, t1_caee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,iA,Xx->XA', h_ca, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,iyAx,Xy->XA', h_ca, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,iyAz,Xzxy->XA', h_ca, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ix,iyxA,Xy->XA', h_ca, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,iyzA,Xzyx->XA', h_ca, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xa,yzAa,Xxyz->XA', h_ae, t1_aaee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('iA,ixyz,Xyxz->XA', t1_ce, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ia,iAxa,Xx->XA', t1_ce, v_ceae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ia,iaAx,Xx->XA', t1_ce, v_ceea, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ia,iaxA,Xx->XA', t1_ce, v_ceae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ia,ixAa,Xx->XA', t1_ce, v_caee, rdm_ca, optimize = einsum_type)
    V1 += einsum('ijAa,ixja,Xx->XA', t1_ccee, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ijAa,jxia,Xx->XA', t1_ccee, v_cace, rdm_ca, optimize = einsum_type)
    V1 += einsum('ijxA,ixjy,Xy->XA', t1_ccae, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ijxA,jxiy,Xy->XA', t1_ccae, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ijxA,jyiz,Xxyz->XA', t1_ccae, v_caca, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ijxa,iAja,Xx->XA', t1_ccae, v_cece, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ijxa,jAia,Xx->XA', t1_ccae, v_cece, rdm_ca, optimize = einsum_type)
    V1 += einsum('ijxy,ixjA,Xy->XA', t1_ccaa, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ijxy,jxiA,Xy->XA', t1_ccaa, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ijxy,jziA,Xzxy->XA', t1_ccaa, v_cace, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,iAyx,Xy->XA', t1_ca, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,iAyz,Xzxy->XA', t1_ca, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ix,ixAy,Xy->XA', t1_ca, v_caea, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ix,ixyA,Xy->XA', t1_ca, v_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,iyAx,Xy->XA', t1_ca, v_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,iyAz,Xxzy->XA', t1_ca, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,iyzA,Xyzx->XA', t1_ca, v_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixAa,iayz,Xyxz->XA', t1_caee, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixAa,iyza,Xzyx->XA', t1_caee, v_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixAy,iyzw,Xzxw->XA', t1_caea, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('ixAy,izwu,Xywuxz->XA', t1_caea, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('ixAy,izwu,Xywuzx->XA', t1_caea, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('ixAy,izwu,Xywxuz->XA', t1_caea, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('ixAy,izwu,Xywxzu->XA', t1_caea, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('ixAy,izwu,Xywzux->XA', t1_caea, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('ixAy,izwu,Xywzxu->XA', t1_caea, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixAy,izwy,Xwzx->XA', t1_caea, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ixaA,iayz,Xyxz->XA', t1_caee, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixaA,iyza,Xzxy->XA', t1_caee, v_caae, rdm_ccaa, optimize = einsum_type)
    #V1 -= einsum('ixab,iaAb,Xx->XA', t1_caee, v_ceee, rdm_ca, optimize = einsum_type)
    #V1 += 1/2 * einsum('ixab,ibAa,Xx->XA', t1_caee, v_ceee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixay,iAza,Xxzy->XA', t1_caea, v_ceae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ixay,iaAy,Xx->XA', t1_caea, v_ceea, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixay,iaAz,Xyzx->XA', t1_caea, v_ceea, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ixay,iazA,Xxzy->XA', t1_caea, v_ceae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixay,iyAa,Xx->XA', t1_caea, v_caee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixay,izAa,Xyzx->XA', t1_caea, v_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ixyA,iyzw,Xzxw->XA', t1_caae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('ixyA,izwu,Xywuxz->XA', t1_caae, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('ixyA,izwu,Xywuzx->XA', t1_caae, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('ixyA,izwu,Xywxuz->XA', t1_caae, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/12 * einsum('ixyA,izwu,Xywxzu->XA', t1_caae, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('ixyA,izwu,Xywzux->XA', t1_caae, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('ixyA,izwu,Xywzxu->XA', t1_caae, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyA,izwy,Xwxz->XA', t1_caae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixya,iAza,Xxyz->XA', t1_caae, v_ceae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixya,iaAy,Xx->XA', t1_caae, v_ceea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixya,iaAz,Xyzx->XA', t1_caae, v_ceea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixya,iazA,Xxzy->XA', t1_caae, v_ceae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ixya,iyAa,Xx->XA', t1_caae, v_caee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixya,izAa,Xyxz->XA', t1_caae, v_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,iAwu,Xxuyzw->XA', t1_caaa, v_ceaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,iAwy,Xxwz->XA', t1_caaa, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,iAwz,Xxyw->XA', t1_caaa, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 += 11/24 * einsum('ixyz,iwAu,Xyzuwx->XA', t1_caaa, v_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('ixyz,iwAu,Xyzuxw->XA', t1_caaa, v_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('ixyz,iwAu,Xyzwux->XA', t1_caaa, v_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('ixyz,iwAu,Xyzwxu->XA', t1_caaa, v_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('ixyz,iwAu,Xyzxuw->XA', t1_caaa, v_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('ixyz,iwAu,Xyzxwu->XA', t1_caaa, v_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,iwAy,Xzwx->XA', t1_caaa, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,iwAz,Xyxw->XA', t1_caaa, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('ixyz,iwuA,Xxwuyz->XA', t1_caaa, v_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('ixyz,iwuA,Xxwuzy->XA', t1_caaa, v_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('ixyz,iwuA,Xxwyuz->XA', t1_caaa, v_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('ixyz,iwuA,Xxwyzu->XA', t1_caaa, v_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('ixyz,iwuA,Xxwzuy->XA', t1_caaa, v_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('ixyz,iwuA,Xxwzyu->XA', t1_caaa, v_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('ixyz,iyAw,Xzwx->XA', t1_caaa, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ixyz,iyAz,Xx->XA', t1_caaa, v_caea, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixyz,iywA,Xxwz->XA', t1_caaa, v_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,izAw,Xywx->XA', t1_caaa, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,izAy,Xx->XA', t1_caaa, v_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,izwA,Xxwy->XA', t1_caaa, v_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xa,yAaz,Xzyx->XA', t1_ae, v_aeea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xa,yAza,Xxyz->XA', t1_ae, v_aeae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xa,yzAa,Xyxz->XA', t1_ae, v_aaee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyAa,zwua,Xuzwxy->XA', t1_aaee, v_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyAa,zwua,Xuzwyx->XA', t1_aaee, v_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyAa,zwua,Xuzxwy->XA', t1_aaee, v_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 3/8 * einsum('xyAa,zwua,Xuzxyw->XA', t1_aaee, v_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyAa,zwua,Xuzywx->XA', t1_aaee, v_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyAa,zwua,Xuzyxw->XA', t1_aaee, v_aaae, rdm_cccaaa, optimize = einsum_type)
    #V1 -= 1/2 * einsum('xyab,zbAa,Xzxy->XA', t1_aaee, v_aeee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyza,wAau,Xzuwxy->XA', t1_aaae, v_aeea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyza,wAau,Xzuwyx->XA', t1_aaae, v_aeea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyza,wAau,Xzuxwy->XA', t1_aaae, v_aeea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyza,wAau,Xzuxyw->XA', t1_aaae, v_aeea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyza,wAau,Xzuywx->XA', t1_aaae, v_aeea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyza,wAau,Xzuyxw->XA', t1_aaae, v_aeea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyza,wAua,Xyxuwz->XA', t1_aaae, v_aeae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyza,wAua,Xyxuzw->XA', t1_aaae, v_aeae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 11/24 * einsum('xyza,wAua,Xyxwuz->XA', t1_aaae, v_aeae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyza,wAua,Xyxwzu->XA', t1_aaae, v_aeae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyza,wAua,Xyxzuw->XA', t1_aaae, v_aeae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyza,wAua,Xyxzwu->XA', t1_aaae, v_aeae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyza,wuAa,Xzwyxu->XA', t1_aaae, v_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyza,wzAa,Xwyx->XA', t1_aaae, v_aaee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyza,zAaw,Xwxy->XA', t1_aaae, v_aeea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,iA,ix,Xx->XA', e_extern, t1_ce, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,iA,ixyz,Xxyz->XA', e_extern, t1_ce, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('A,ijAa,ijxa,Xx->XA', e_extern, t1_ccee, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,ijAa,jixa,Xx->XA', e_extern, t1_ccee, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('A,ijxA,ijxy,Xy->XA', e_extern, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,ijxA,jixy,Xy->XA', e_extern, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,ijxA,jiyz,Xxyz->XA', e_extern, t1_ccae, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixAa,ia,Xx->XA', e_extern, t1_caee, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixAa,iyaz,Xyxz->XA', e_extern, t1_caee, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixAa,iyza,Xyzx->XA', e_extern, t1_caee, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixAy,iy,Xx->XA', e_extern, t1_caea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixAy,iz,Xyzx->XA', e_extern, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('A,ixAy,izwu,Xyzuwx->XA', e_extern, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('A,ixAy,izwu,Xyzuxw->XA', e_extern, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('A,ixAy,izwu,Xyzwux->XA', e_extern, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('A,ixAy,izwu,Xyzwxu->XA', e_extern, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('A,ixAy,izwu,Xyzxuw->XA', e_extern, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('A,ixAy,izwu,Xyzxwu->XA', e_extern, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixAy,izwy,Xzwx->XA', e_extern, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixAy,izyw,Xzxw->XA', e_extern, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,ixaA,ia,Xx->XA', e_extern, t1_caee, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,ixaA,iyaz,Xyxz->XA', e_extern, t1_caee, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixaA,iyza,Xyxz->XA', e_extern, t1_caee, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,ixyA,iy,Xx->XA', e_extern, t1_caae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixyA,iz,Xyxz->XA', e_extern, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixyA,izwu,Xyzxwu->XA', e_extern, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixyA,izwy,Xzxw->XA', e_extern, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,ixyA,izyw,Xzxw->XA', e_extern, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,xyAa,za,Xzxy->XA', e_extern, t1_aaee, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/16 * einsum('A,xyAa,zwua,Xwzuxy->XA', e_extern, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('A,xyAa,zwua,Xwzuyx->XA', e_extern, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('A,xyAa,zwua,Xwzxuy->XA', e_extern, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 3/16 * einsum('A,xyAa,zwua,Xwzxyu->XA', e_extern, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('A,xyAa,zwua,Xwzyux->XA', e_extern, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('A,xyAa,zwua,Xwzyxu->XA', e_extern, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,ia,ixAa,Xx->XA', e_extern, t1_ce, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= einsum('a,ia,ixaA,Xx->XA', e_extern, t1_ce, t1_caee, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,ijxa,ijAa,Xx->XA', e_extern, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,ijxa,jiAa,Xx->XA', e_extern, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,ixay,izAa,Xxzy->XA', e_extern, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('a,ixay,izaA,Xxzy->XA', e_extern, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,ixya,izAa,Xxyz->XA', e_extern, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,ixya,izaA,Xxzy->XA', e_extern, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,xa,yzAa,Xxyz->XA', e_extern, t1_ae, t1_aaee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/8 * einsum('a,xyza,wuAa,Xyxuwz->XA', e_extern, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('a,xyza,wuAa,Xyxuzw->XA', e_extern, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 3/8 * einsum('a,xyza,wuAa,Xyxwuz->XA', e_extern, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('a,xyza,wuAa,Xyxwzu->XA', e_extern, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('a,xyza,wuAa,Xyxzuw->XA', e_extern, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('a,xyza,wuAa,Xyxzwu->XA', e_extern, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iA,ix,Xx->XA', e_core, t1_ce, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iA,ixyz,Xxyz->XA', e_core, t1_ce, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ia,ixAa,Xx->XA', e_core, t1_ce, t1_caee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ia,ixaA,Xx->XA', e_core, t1_ce, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ijAa,ijxa,Xx->XA', e_core, t1_ccee, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ijAa,jixa,Xx->XA', e_core, t1_ccee, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ijxA,ijxy,Xy->XA', e_core, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ijxA,ijyx,Xy->XA', e_core, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ijxA,ijyz,Xxzy->XA', e_core, t1_ccae, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ijxa,ijAa,Xx->XA', e_core, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ijxa,jiAa,Xx->XA', e_core, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ijxy,ijxA,Xy->XA', e_core, t1_ccaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ijxy,ijyA,Xx->XA', e_core, t1_ccaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ijxy,ijzA,Xzyx->XA', e_core, t1_ccaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ijxy,jixA,Xy->XA', e_core, t1_ccaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ijxy,jiyA,Xx->XA', e_core, t1_ccaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ijxy,jizA,Xzxy->XA', e_core, t1_ccaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ix,iyAx,Xy->XA', e_core, t1_ca, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ix,iyxA,Xy->XA', e_core, t1_ca, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixAa,ia,Xx->XA', e_core, t1_caee, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixAa,iyaz,Xyxz->XA', e_core, t1_caee, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixAa,iyza,Xyzx->XA', e_core, t1_caee, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixAy,ixzw,Xyzw->XA', e_core, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixAy,iy,Xx->XA', e_core, t1_caea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixAy,iz,Xyzx->XA', e_core, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('i,ixAy,izwu,Xyzuwx->XA', e_core, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('i,ixAy,izwu,Xyzuxw->XA', e_core, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('i,ixAy,izwu,Xyzwux->XA', e_core, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('i,ixAy,izwu,Xyzwxu->XA', e_core, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('i,ixAy,izwu,Xyzxuw->XA', e_core, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('i,ixAy,izwu,Xyzxwu->XA', e_core, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixAy,izwy,Xzwx->XA', e_core, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixAy,izyw,Xzxw->XA', e_core, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ixaA,ia,Xx->XA', e_core, t1_caee, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ixaA,iyaz,Xyxz->XA', e_core, t1_caee, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixaA,iyza,Xyxz->XA', e_core, t1_caee, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixay,izAa,Xxzy->XA', e_core, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ixay,izaA,Xxzy->XA', e_core, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixyA,ixzw,Xywz->XA', e_core, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ixyA,iy,Xx->XA', e_core, t1_caae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyA,iz,Xyxz->XA', e_core, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('i,ixyA,izwu,Xyzuwx->XA', e_core, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('i,ixyA,izwu,Xyzuxw->XA', e_core, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('i,ixyA,izwu,Xyzwux->XA', e_core, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('i,ixyA,izwu,Xyzwxu->XA', e_core, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('i,ixyA,izwu,Xyzxuw->XA', e_core, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('i,ixyA,izwu,Xyzxwu->XA', e_core, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixyA,izwy,Xzxw->XA', e_core, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ixyA,izyw,Xzxw->XA', e_core, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixya,izAa,Xxyz->XA', e_core, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixya,izaA,Xxzy->XA', e_core, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixyz,iwAy,Xxwz->XA', e_core, t1_caaa, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixyz,iwAz,Xxyw->XA', e_core, t1_caaa, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ixyz,iwyA,Xxwz->XA', e_core, t1_caaa, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixyz,iwzA,Xxwy->XA', e_core, t1_caaa, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixyz,ixAw,Xwyz->XA', e_core, t1_caaa, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixyz,ixwA,Xwzy->XA', e_core, t1_caaa, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,jiAa,ijxa,Xx->XA', e_core, t1_ccee, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,jiAa,jixa,Xx->XA', e_core, t1_ccee, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,jixA,ijxy,Xy->XA', e_core, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,jixA,ijyx,Xy->XA', e_core, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,jixA,ijyz,Xxyz->XA', e_core, t1_ccae, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,jixa,ijAa,Xx->XA', e_core, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,jixa,jiAa,Xx->XA', e_core, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,ijxA,ijyz,Xz->XA', h_aa, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ijxA,jiyz,Xz->XA', h_aa, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ijxA,jizw,Xyzw->XA', h_aa, t1_ccae, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ijxa,ijAa,Xy->XA', h_aa, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ijxa,jiAa,Xy->XA', h_aa, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ijxz,ijwA,Xwzy->XA', h_aa, t1_ccaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ijxz,ijzA,Xy->XA', h_aa, t1_ccaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ijxz,jiwA,Xwyz->XA', h_aa, t1_ccaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ijxz,jizA,Xy->XA', h_aa, t1_ccaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ix,iA,Xy->XA', h_aa, t1_ca, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ix,izAw,Xwyz->XA', h_aa, t1_ca, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ix,izwA,Xwzy->XA', h_aa, t1_ca, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixAa,ia,Xy->XA', h_aa, t1_caee, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixAa,izaw,Xzyw->XA', h_aa, t1_caee, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixAa,izwa,Xzwy->XA', h_aa, t1_caee, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixAz,iw,Xzwy->XA', h_aa, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,ixAz,iwuv,Xzwuvy->XA', h_aa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xy,ixAz,iwuv,Xzwuyv->XA', h_aa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,ixAz,iwuv,Xzwvuy->XA', h_aa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,ixAz,iwuv,Xzwvyu->XA', h_aa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,ixAz,iwuv,Xzwyuv->XA', h_aa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,ixAz,iwuv,Xzwyvu->XA', h_aa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixAz,iwuz,Xwuy->XA', h_aa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixAz,iwzu,Xwyu->XA', h_aa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixAz,iz,Xy->XA', h_aa, t1_caea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ixaA,ia,Xy->XA', h_aa, t1_caee, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ixaA,izaw,Xzyw->XA', h_aa, t1_caee, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixaA,izwa,Xzyw->XA', h_aa, t1_caee, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixaz,iwAa,Xywz->XA', h_aa, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ixaz,iwaA,Xywz->XA', h_aa, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzA,iw,Xzyw->XA', h_aa, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzA,iwuv,Xzwyuv->XA', h_aa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzA,iwuz,Xwyu->XA', h_aa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ixzA,iwzu,Xwyu->XA', h_aa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ixzA,iz,Xy->XA', h_aa, t1_caae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixza,iwAa,Xyzw->XA', h_aa, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixza,iwaA,Xywz->XA', h_aa, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzw,iA,Xyzw->XA', h_aa, t1_caaa, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,ixzw,iuAv,Xyvuwz->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,ixzw,iuAv,Xyvuzw->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,ixzw,iuAv,Xyvwuz->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,ixzw,iuAv,Xyvwzu->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,ixzw,iuAv,Xyvzuw->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xy,ixzw,iuAv,Xyvzwu->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzw,iuAw,Xyzu->XA', h_aa, t1_caaa, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzw,iuAz,Xyuw->XA', h_aa, t1_caaa, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzw,iuvA,Xyvuwz->XA', h_aa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzw,iuwA,Xyuz->XA', h_aa, t1_caaa, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ixzw,iuzA,Xyuw->XA', h_aa, t1_caaa, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izAx,iw,Xywz->XA', h_aa, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izAx,iwuv,Xywuvz->XA', h_aa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xy,izAx,iwuv,Xywuzv->XA', h_aa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izAx,iwuv,Xywvuz->XA', h_aa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izAx,iwuv,Xywvzu->XA', h_aa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izAx,iwuv,Xywzuv->XA', h_aa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izAx,iwuv,Xywzvu->XA', h_aa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izAx,iwuy,Xwuz->XA', h_aa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izAx,iwyu,Xwzu->XA', h_aa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izAx,iy,Xz->XA', h_aa, t1_caea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izax,iwAa,Xzwy->XA', h_aa, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,izax,iwaA,Xzwy->XA', h_aa, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izwx,iA,Xzwy->XA', h_aa, t1_caaa, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izwx,iuAv,Xzvuwy->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izwx,iuAv,Xzvuyw->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izwx,iuAv,Xzvwuy->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xy,izwx,iuAv,Xzvwyu->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izwx,iuAv,Xzvyuw->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izwx,iuAv,Xzvywu->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izwx,iuAw,Xzuy->XA', h_aa, t1_caaa, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izwx,iuvA,Xzvuyw->XA', h_aa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,izwx,iuwA,Xzuy->XA', h_aa, t1_caaa, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxA,iw,Xyzw->XA', h_aa, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxA,iwuv,Xywzuv->XA', h_aa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izxA,iwuy,Xwzu->XA', h_aa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,izxA,iwyu,Xwzu->XA', h_aa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,izxA,iy,Xz->XA', h_aa, t1_caae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxa,iwAa,Xzyw->XA', h_aa, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxa,iwaA,Xzwy->XA', h_aa, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxw,iA,Xzyw->XA', h_aa, t1_caaa, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxw,iuAv,Xzvywu->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxw,iuAw,Xzyu->XA', h_aa, t1_caaa, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xy,izxw,iuvA,Xzvuwy->XA', h_aa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izxw,iuvA,Xzvuyw->XA', h_aa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izxw,iuvA,Xzvwuy->XA', h_aa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izxw,iuvA,Xzvwyu->XA', h_aa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izxw,iuvA,Xzvyuw->XA', h_aa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,izxw,iuvA,Xzvywu->XA', h_aa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxw,iuwA,Xzuy->XA', h_aa, t1_caaa, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,xa,zwAa,Xyzw->XA', h_aa, t1_ae, t1_aaee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,xzAa,wa,Xwyz->XA', h_aa, t1_aaee, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,xzAa,wuva,Xuwyzv->XA', h_aa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,xzwa,uvAa,Xyzuvw->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xy,xzwa,uvAa,Xyzuwv->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,xzwa,uvAa,Xyzvuw->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,xzwa,uvAa,Xyzvwu->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,xzwa,uvAa,Xyzwuv->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,xzwa,uvAa,Xyzwvu->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xy,zwxa,uvAa,Xzwuvy->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 3/16 * einsum('xy,zwxa,uvAa,Xzwuyv->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xy,zwxa,uvAa,Xzwvuy->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xy,zwxa,uvAa,Xzwvyu->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xy,zwxa,uvAa,Xzwyuv->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xy,zwxa,uvAa,Xzwyvu->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zxAa,wa,Xwzy->XA', h_aa, t1_aaee, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,zxAa,wuva,Xuwvyz->XA', h_aa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,zxAa,wuva,Xuwvzy->XA', h_aa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,zxAa,wuva,Xuwyvz->XA', h_aa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,zxAa,wuva,Xuwyzv->XA', h_aa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,zxAa,wuva,Xuwzvy->XA', h_aa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xy,zxAa,wuva,Xuwzyv->XA', h_aa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zxwa,uvAa,Xyzuvw->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xa,yAaz,zy,Xx->XA', t1_ae, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xa,yzAa,yz,Xx->XA', t1_ae, v_aaee, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyza,wAau,uw,Xzyx->XA', t1_aaae, v_aeea, rdm_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,wuAa,wu,Xzyx->XA', t1_aaae, v_aaee, rdm_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ijxA,ijwu,Xyuz->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,ijxA,ijyu,Xwuz->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,ijxA,ijyw,Xz->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 11/48 * einsum('xyzw,ijxA,jiuv,Xywuvz->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,ijxA,jiuv,Xywuzv->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,ijxA,jiuv,Xywvuz->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,ijxA,jiuv,Xywvzu->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,ijxA,jiuv,Xywzuv->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,ijxA,jiuv,Xywzvu->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ijxA,jiwu,Xyzu->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ijxA,jiyu,Xwuz->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ijxA,jiyw,Xz->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ijxa,ijAa,Xzyw->XA', v_aaaa, t1_ccae, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ijxa,jiAa,Xzyw->XA', v_aaaa, t1_ccae, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ijxu,ijuA,Xzyw->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,ijxu,ijvA,Xzvuwy->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ijxu,ijvA,Xzvuyw->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ijxu,ijvA,Xzvwuy->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ijxu,ijvA,Xzvwyu->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ijxu,ijvA,Xzvyuw->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ijxu,ijvA,Xzvywu->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ijxu,jiuA,Xzyw->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ijxu,jivA,Xzvuwy->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ijxu,jivA,Xzvuyw->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ijxu,jivA,Xzvwuy->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ijxu,jivA,Xzvwyu->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ijxu,jivA,Xzvyuw->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ijxu,jivA,Xzvywu->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ijxz,jiuA,Xuyw->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuAx,iv,Xywuvz->XA', v_aaaa, t1_caea, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuAx,iv,Xywuzv->XA', v_aaaa, t1_caea, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,iuAx,iv,Xywvuz->XA', v_aaaa, t1_caea, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuAx,iv,Xywvzu->XA', v_aaaa, t1_caea, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuAx,iv,Xywzuv->XA', v_aaaa, t1_caea, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuAx,iv,Xywzvu->XA', v_aaaa, t1_caea, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 17/480 * einsum('xyzw,iuAx,ivst,Xywvstuz->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,iuAx,ivst,Xywvstzu->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,iuAx,ivst,Xywvsutz->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/48 * einsum('xyzw,iuAx,ivst,Xywvsuzt->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 17/480 * einsum('xyzw,iuAx,ivst,Xywvsztu->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,iuAx,ivst,Xywvszut->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 17/480 * einsum('xyzw,iuAx,ivst,Xywvtsuz->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,iuAx,ivst,Xywvtszu->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 23/480 * einsum('xyzw,iuAx,ivst,Xywvtusz->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,iuAx,ivst,Xywvtuzs->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/160 * einsum('xyzw,iuAx,ivst,Xywvtzsu->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 17/240 * einsum('xyzw,iuAx,ivst,Xywvtzus->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/60 * einsum('xyzw,iuAx,ivst,Xywvustz->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 47/480 * einsum('xyzw,iuAx,ivst,Xywvuszt->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/480 * einsum('xyzw,iuAx,ivst,Xywvutzs->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 23/480 * einsum('xyzw,iuAx,ivst,Xywvuzst->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/96 * einsum('xyzw,iuAx,ivst,Xywvuzts->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/60 * einsum('xyzw,iuAx,ivst,Xywvzstu->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,iuAx,ivst,Xywvzsut->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/96 * einsum('xyzw,iuAx,ivst,Xywvztus->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 11/96 * einsum('xyzw,iuAx,ivst,Xywvzust->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/480 * einsum('xyzw,iuAx,ivst,Xywvzuts->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,iuAx,ivsw,Xyvsuz->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuAx,ivsw,Xyvszu->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuAx,ivsw,Xyvusz->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuAx,ivsw,Xyvuzs->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuAx,ivsw,Xyvzsu->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuAx,ivsw,Xyvzus->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuAx,ivsy,Xwvszu->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuAx,ivws,Xyvzus->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuAx,ivwy,Xvzu->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuAx,ivys,Xwvsuz->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuAx,ivys,Xwvszu->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuAx,ivys,Xwvusz->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,iuAx,ivys,Xwvuzs->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuAx,ivys,Xwvzsu->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuAx,ivys,Xwvzus->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuAx,ivyw,Xvuz->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuAx,iw,Xyzu->XA', v_aaaa, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuAx,iy,Xwuz->XA', v_aaaa, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,iuAx,izvs,Xywsuv->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,iuAx,izvs,Xywsvu->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,iuAx,izvs,Xywusv->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,iuAx,izvs,Xywuvs->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,iuAx,izvs,Xywvsu->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 3/16 * einsum('xyzw,iuAx,izvs,Xywvus->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,iuax,ivAa,Xzuvwy->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuax,ivAa,Xzuvyw->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuax,ivAa,Xzuwvy->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuax,ivAa,Xzuwyv->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuax,ivAa,Xzuyvw->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuax,ivAa,Xzuywv->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('xyzw,iuax,ivaA,Xzuvwy->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuax,ivaA,Xzuvyw->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuax,ivaA,Xzuwvy->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuax,ivaA,Xzuwyv->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuax,ivaA,Xzuyvw->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuax,ivaA,Xzuywv->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,iuvx,iA,Xzuvwy->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,iA,Xzuvyw->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,iA,Xzuwvy->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,iA,Xzuwyv->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,iA,Xzuyvw->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,iA,Xzuywv->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 3/80 * einsum('xyzw,iuvx,isAt,Xzutsvwy->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,iuvx,isAt,Xzutsvyw->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 3/80 * einsum('xyzw,iuvx,isAt,Xzutswvy->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuvx,isAt,Xzutswyv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/80 * einsum('xyzw,iuvx,isAt,Xzutsyvw->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/15 * einsum('xyzw,iuvx,isAt,Xzutsywv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/240 * einsum('xyzw,iuvx,isAt,Xzutvswy->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 13/240 * einsum('xyzw,iuvx,isAt,Xzutvsyw->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/240 * einsum('xyzw,iuvx,isAt,Xzutvwsy->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,isAt,Xzutvwys->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,iuvx,isAt,Xzutvysw->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/60 * einsum('xyzw,iuvx,isAt,Xzutvyws->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,iuvx,isAt,Xzutwsvy->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/240 * einsum('xyzw,iuvx,isAt,Xzutwsyv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,iuvx,isAt,Xzutwvsy->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 23/240 * einsum('xyzw,iuvx,isAt,Xzutwvys->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 17/240 * einsum('xyzw,iuvx,isAt,Xzutwysv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,iuvx,isAt,Xzutwyvs->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/40 * einsum('xyzw,iuvx,isAt,Xzutysvw->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 19/240 * einsum('xyzw,iuvx,isAt,Xzutyswv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/40 * einsum('xyzw,iuvx,isAt,Xzutyvsw->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,iuvx,isAt,Xzutyvws->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/240 * einsum('xyzw,iuvx,isAt,Xzutywsv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/80 * einsum('xyzw,iuvx,isAt,Xzutywvs->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,iuvx,isAv,Xzuswy->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,isAv,Xzusyw->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,isAv,Xzuwsy->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,isAv,Xzuwys->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,isAv,Xzuysw->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,isAv,Xzuyws->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/30 * einsum('xyzw,iuvx,istA,Xzutsvwy->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/60 * einsum('xyzw,iuvx,istA,Xzutsvyw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,iuvx,istA,Xzutswvy->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyzw,iuvx,istA,Xzutswyv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,iuvx,istA,Xzutsyvw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,iuvx,istA,Xzutsywv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/40 * einsum('xyzw,iuvx,istA,Xzutvswy->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,iuvx,istA,Xzutvsyw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/30 * einsum('xyzw,iuvx,istA,Xzutvwsy->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,istA,Xzutvwys->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/30 * einsum('xyzw,iuvx,istA,Xzutvysw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/40 * einsum('xyzw,iuvx,istA,Xzutvyws->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,iuvx,istA,Xzutwsvy->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/8 * einsum('xyzw,iuvx,istA,Xzutwsyv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,iuvx,istA,Xzutwvsy->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/30 * einsum('xyzw,iuvx,istA,Xzutwvys->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/15 * einsum('xyzw,iuvx,istA,Xzutwysv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/40 * einsum('xyzw,iuvx,istA,Xzutwyvs->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,iuvx,istA,Xzutysvw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/120 * einsum('xyzw,iuvx,istA,Xzutyswv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,iuvx,istA,Xzutyvsw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/30 * einsum('xyzw,iuvx,istA,Xzutyvws->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/60 * einsum('xyzw,iuvx,istA,Xzutywsv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,iuvx,istA,Xzutywvs->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('xyzw,iuvx,isvA,Xzuswy->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,isvA,Xzusyw->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,isvA,Xzuwsy->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,isvA,Xzuwys->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,isvA,Xzuysw->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,isvA,Xzuyws->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,iuxA,iv,Xywuvz->XA', v_aaaa, t1_caae, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxA,iv,Xywuzv->XA', v_aaaa, t1_caae, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxA,iv,Xywvuz->XA', v_aaaa, t1_caae, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxA,iv,Xywvzu->XA', v_aaaa, t1_caae, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxA,iv,Xywzuv->XA', v_aaaa, t1_caae, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxA,iv,Xywzvu->XA', v_aaaa, t1_caae, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 3/160 * einsum('xyzw,iuxA,ivst,Xywvstuz->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/160 * einsum('xyzw,iuxA,ivst,Xywvstzu->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/96 * einsum('xyzw,iuxA,ivst,Xywvsutz->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/240 * einsum('xyzw,iuxA,ivst,Xywvsuzt->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,iuxA,ivst,Xywvsztu->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 3/80 * einsum('xyzw,iuxA,ivst,Xywvszut->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 5/96 * einsum('xyzw,iuxA,ivst,Xywvtsuz->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 13/480 * einsum('xyzw,iuxA,ivst,Xywvtszu->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/32 * einsum('xyzw,iuxA,ivst,Xywvtusz->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,iuxA,ivst,Xywvtuzs->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/480 * einsum('xyzw,iuxA,ivst,Xywvtzsu->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,iuxA,ivst,Xywvtzus->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 17/240 * einsum('xyzw,iuxA,ivst,Xywvustz->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 79/480 * einsum('xyzw,iuxA,ivst,Xywvuszt->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,iuxA,ivst,Xywvutsz->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/96 * einsum('xyzw,iuxA,ivst,Xywvutzs->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 17/480 * einsum('xyzw,iuxA,ivst,Xywvuzst->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,iuxA,ivst,Xywvuzts->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,iuxA,ivst,Xywvzstu->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 31/480 * einsum('xyzw,iuxA,ivst,Xywvzsut->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/15 * einsum('xyzw,iuxA,ivst,Xywvztsu->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/32 * einsum('xyzw,iuxA,ivst,Xywvztus->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/160 * einsum('xyzw,iuxA,ivst,Xywvzust->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/480 * einsum('xyzw,iuxA,ivst,Xywvzuts->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxA,ivsw,Xyvusz->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuxA,ivsy,Xwvsuz->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuxA,ivsy,Xwvszu->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuxA,ivsy,Xwvusz->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,iuxA,ivsy,Xwvuzs->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuxA,ivsy,Xwvzsu->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuxA,ivsy,Xwvzus->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuxA,ivws,Xyvsuz->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuxA,ivws,Xyvszu->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuxA,ivws,Xyvusz->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,iuxA,ivws,Xyvuzs->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuxA,ivws,Xyvzsu->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuxA,ivws,Xyvzus->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxA,ivwy,Xvuz->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,iuxA,ivys,Xwvsuz->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,iuxA,ivys,Xwvszu->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,iuxA,ivys,Xwvusz->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 2/3 * einsum('xyzw,iuxA,ivys,Xwvuzs->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,iuxA,ivys,Xwvzsu->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,iuxA,ivys,Xwvzus->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('xyzw,iuxA,ivyw,Xvuz->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxA,iw,Xyuz->XA', v_aaaa, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,iuxA,iy,Xwuz->XA', v_aaaa, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,iuxA,izvs,Xywsuv->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,iuxA,izvs,Xywsvu->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,iuxA,izvs,Xywusv->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,iuxA,izvs,Xywuvs->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,iuxA,izvs,Xywvsu->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,iuxA,izvs,Xywvus->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxa,ivAa,Xzuvwy->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxa,ivAa,Xzuvyw->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxa,ivAa,Xzuwvy->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxa,ivAa,Xzuwyv->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxa,ivAa,Xzuyvw->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,iuxa,ivAa,Xzuywv->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,iuxa,ivaA,Xzuvwy->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxa,ivaA,Xzuvyw->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxa,ivaA,Xzuwvy->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxa,ivaA,Xzuwyv->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxa,ivaA,Xzuyvw->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxa,ivaA,Xzuywv->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxv,iA,Xzuvwy->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxv,iA,Xzuvyw->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxv,iA,Xzuwvy->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxv,iA,Xzuwyv->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxv,iA,Xzuyvw->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,iuxv,iA,Xzuywv->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,iuxv,isAt,Xzutsvwy->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,iuxv,isAt,Xzutsvyw->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,iuxv,isAt,Xzutswyv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/120 * einsum('xyzw,iuxv,isAt,Xzutsyvw->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 3/80 * einsum('xyzw,iuxv,isAt,Xzutsywv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,iuxv,isAt,Xzutvswy->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,iuxv,isAt,Xzutvsyw->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/40 * einsum('xyzw,iuxv,isAt,Xzutvwsy->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,iuxv,isAt,Xzutvwys->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/30 * einsum('xyzw,iuxv,isAt,Xzutvysw->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 3/80 * einsum('xyzw,iuxv,isAt,Xzutvyws->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/240 * einsum('xyzw,iuxv,isAt,Xzutwsvy->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,iuxv,isAt,Xzutwsyv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/240 * einsum('xyzw,iuxv,isAt,Xzutwvsy->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,iuxv,isAt,Xzutwysv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 11/240 * einsum('xyzw,iuxv,isAt,Xzutwyvs->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 3/80 * einsum('xyzw,iuxv,isAt,Xzutysvw->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/60 * einsum('xyzw,iuxv,isAt,Xzutyswv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/240 * einsum('xyzw,iuxv,isAt,Xzutyvsw->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,iuxv,isAt,Xzutyvws->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/240 * einsum('xyzw,iuxv,isAt,Xzutywsv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 61/240 * einsum('xyzw,iuxv,isAt,Xzutywvs->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxv,isAv,Xzuswy->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxv,isAv,Xzusyw->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxv,isAv,Xzuwsy->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxv,isAv,Xzuwys->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxv,isAv,Xzuysw->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,iuxv,isAv,Xzuyws->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,iuxv,istA,Xzutsvwy->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/40 * einsum('xyzw,iuxv,istA,Xzutsvyw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxv,istA,Xzutswvy->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/15 * einsum('xyzw,iuxv,istA,Xzutswyv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/30 * einsum('xyzw,iuxv,istA,Xzutsyvw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/60 * einsum('xyzw,iuxv,istA,Xzutsywv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,iuxv,istA,Xzutvswy->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/120 * einsum('xyzw,iuxv,istA,Xzutvsyw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 3/40 * einsum('xyzw,iuxv,istA,Xzutvwsy->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/15 * einsum('xyzw,iuxv,istA,Xzutvwys->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/120 * einsum('xyzw,iuxv,istA,Xzutvysw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/60 * einsum('xyzw,iuxv,istA,Xzutvyws->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxv,istA,Xzutwsvy->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/120 * einsum('xyzw,iuxv,istA,Xzutwsyv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/60 * einsum('xyzw,iuxv,istA,Xzutwvsy->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/40 * einsum('xyzw,iuxv,istA,Xzutwvys->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/120 * einsum('xyzw,iuxv,istA,Xzutwysv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/30 * einsum('xyzw,iuxv,istA,Xzutwyvs->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,iuxv,istA,Xzutysvw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,iuxv,istA,Xzutyswv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/20 * einsum('xyzw,iuxv,istA,Xzutyvsw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,iuxv,istA,Xzutyvws->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxv,istA,Xzutywsv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 2/15 * einsum('xyzw,iuxv,istA,Xzutywvs->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,iuxv,isvA,Xzuswy->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxv,isvA,Xzusyw->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxv,isvA,Xzuwsy->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxv,isvA,Xzuwys->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxv,isvA,Xzuysw->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxv,isvA,Xzuyws->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuxz,iA,Xuyw->XA', v_aaaa, t1_caaa, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxz,ivAs,Xusvwy->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxz,ivAs,Xusvyw->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxz,ivAs,Xuswvy->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxz,ivAs,Xuswyv->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuxz,ivAs,Xusyvw->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,iuxz,ivAs,Xusywv->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,iuxz,ivsA,Xusvwy->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxz,ivsA,Xusvyw->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxz,ivsA,Xuswvy->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxz,ivsA,Xuswyv->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxz,ivsA,Xusyvw->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxz,ivsA,Xusywv->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ix,iA,Xzyw->XA', v_aaaa, t1_ca, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ix,iuAv,Xzvuwy->XA', v_aaaa, t1_ca, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ix,iuAv,Xzvuyw->XA', v_aaaa, t1_ca, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ix,iuAv,Xzvwuy->XA', v_aaaa, t1_ca, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ix,iuAv,Xzvwyu->XA', v_aaaa, t1_ca, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ix,iuAv,Xzvyuw->XA', v_aaaa, t1_ca, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ix,iuAv,Xzvywu->XA', v_aaaa, t1_ca, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,ix,iuvA,Xzvuwy->XA', v_aaaa, t1_ca, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,ix,iuvA,Xzvuyw->XA', v_aaaa, t1_ca, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,ix,iuvA,Xzvwuy->XA', v_aaaa, t1_ca, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,ix,iuvA,Xzvwyu->XA', v_aaaa, t1_ca, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,ix,iuvA,Xzvyuw->XA', v_aaaa, t1_ca, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,ix,iuvA,Xzvywu->XA', v_aaaa, t1_ca, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAa,ia,Xzyw->XA', v_aaaa, t1_caee, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAa,iuav,Xzuvwy->XA', v_aaaa, t1_caee, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAa,iuav,Xzuvyw->XA', v_aaaa, t1_caee, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAa,iuav,Xzuwvy->XA', v_aaaa, t1_caee, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAa,iuav,Xzuwyv->XA', v_aaaa, t1_caee, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAa,iuav,Xzuyvw->XA', v_aaaa, t1_caee, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixAa,iuav,Xzuywv->XA', v_aaaa, t1_caee, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAa,iuaz,Xuyw->XA', v_aaaa, t1_caee, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,ixAa,iuva,Xzuvwy->XA', v_aaaa, t1_caee, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAa,iuva,Xzuvyw->XA', v_aaaa, t1_caee, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAa,iuva,Xzuwvy->XA', v_aaaa, t1_caee, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAa,iuva,Xzuwyv->XA', v_aaaa, t1_caee, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAa,iuva,Xzuyvw->XA', v_aaaa, t1_caee, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAa,iuva,Xzuywv->XA', v_aaaa, t1_caee, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAa,iuza,Xuwy->XA', v_aaaa, t1_caee, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,iu,Xzyw->XA', v_aaaa, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,ixAu,iv,Xzuvwy->XA', v_aaaa, t1_caea, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,iv,Xzuvyw->XA', v_aaaa, t1_caea, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,iv,Xzuwvy->XA', v_aaaa, t1_caea, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,iv,Xzuwyv->XA', v_aaaa, t1_caea, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,iv,Xzuyvw->XA', v_aaaa, t1_caea, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,iv,Xzuywv->XA', v_aaaa, t1_caea, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 11/480 * einsum('xyzw,ixAu,ivst,Xzuvstwy->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/96 * einsum('xyzw,ixAu,ivst,Xzuvstyw->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 13/480 * einsum('xyzw,ixAu,ivst,Xzuvswty->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 5/48 * einsum('xyzw,ixAu,ivst,Xzuvswyt->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,ixAu,ivst,Xzuvsytw->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/240 * einsum('xyzw,ixAu,ivst,Xzuvsywt->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 9/160 * einsum('xyzw,ixAu,ivst,Xzuvtswy->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/160 * einsum('xyzw,ixAu,ivst,Xzuvtsyw->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 13/480 * einsum('xyzw,ixAu,ivst,Xzuvtwsy->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 13/240 * einsum('xyzw,ixAu,ivst,Xzuvtwys->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,ixAu,ivst,Xzuvtysw->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 11/240 * einsum('xyzw,ixAu,ivst,Xzuvtyws->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/120 * einsum('xyzw,ixAu,ivst,Xzuvwsty->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 29/480 * einsum('xyzw,ixAu,ivst,Xzuvwsyt->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,ixAu,ivst,Xzuvwtsy->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,ixAu,ivst,Xzuvwtys->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/160 * einsum('xyzw,ixAu,ivst,Xzuvwyst->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 23/480 * einsum('xyzw,ixAu,ivst,Xzuvwyts->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/30 * einsum('xyzw,ixAu,ivst,Xzuvystw->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/96 * einsum('xyzw,ixAu,ivst,Xzuvyswt->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 23/480 * einsum('xyzw,ixAu,ivst,Xzuvytws->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/32 * einsum('xyzw,ixAu,ivst,Xzuvywst->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,ixAu,ivst,Xzuvywts->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,ixAu,ivsu,Xzvswy->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,ivsu,Xzvsyw->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,ivsu,Xzvwsy->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,ivsu,Xzvwys->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,ivsu,Xzvysw->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,ivsu,Xzvyws->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,ivsz,Xuvswy->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,ixAu,ivsz,Xuvsyw->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,ivsz,Xuvwsy->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,ivsz,Xuvwys->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,ivsz,Xuvysw->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixAu,ivsz,Xuvyws->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAu,ivus,Xzvswy->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAu,ivus,Xzvsyw->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAu,ivus,Xzvwsy->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAu,ivus,Xzvwys->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAu,ivus,Xzvysw->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixAu,ivus,Xzvyws->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,ivuz,Xvyw->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAu,ivzs,Xuvswy->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAu,ivzs,Xuvsyw->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAu,ivzs,Xuvwsy->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixAu,ivzs,Xuvwys->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAu,ivzs,Xuvysw->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixAu,ivzs,Xuvyws->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,ivzu,Xvwy->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,iz,Xuwy->XA', v_aaaa, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ixaA,ia,Xzyw->XA', v_aaaa, t1_caee, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixaA,iuav,Xzuvwy->XA', v_aaaa, t1_caee, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixaA,iuav,Xzuvyw->XA', v_aaaa, t1_caee, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixaA,iuav,Xzuwvy->XA', v_aaaa, t1_caee, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixaA,iuav,Xzuwyv->XA', v_aaaa, t1_caee, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixaA,iuav,Xzuyvw->XA', v_aaaa, t1_caee, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,ixaA,iuav,Xzuywv->XA', v_aaaa, t1_caee, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ixaA,iuaz,Xuyw->XA', v_aaaa, t1_caee, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixaA,iuva,Xzuvwy->XA', v_aaaa, t1_caee, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixaA,iuva,Xzuvyw->XA', v_aaaa, t1_caee, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixaA,iuva,Xzuwvy->XA', v_aaaa, t1_caee, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixaA,iuva,Xzuwyv->XA', v_aaaa, t1_caee, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixaA,iuva,Xzuyvw->XA', v_aaaa, t1_caee, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixaA,iuva,Xzuywv->XA', v_aaaa, t1_caee, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixaA,iuza,Xuyw->XA', v_aaaa, t1_caee, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixau,ivAa,Xywuvz->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixau,ivAa,Xywuzv->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,ixau,ivAa,Xywvuz->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixau,ivAa,Xywvzu->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixau,ivAa,Xywzuv->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixau,ivAa,Xywzvu->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixau,ivaA,Xywuvz->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixau,ivaA,Xywuzv->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/12 * einsum('xyzw,ixau,ivaA,Xywvuz->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixau,ivaA,Xywvzu->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixau,ivaA,Xywzuv->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixau,ivaA,Xywzvu->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ixuA,iu,Xzyw->XA', v_aaaa, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,iv,Xzuvwy->XA', v_aaaa, t1_caae, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,iv,Xzuvyw->XA', v_aaaa, t1_caae, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,iv,Xzuwvy->XA', v_aaaa, t1_caae, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,iv,Xzuwyv->XA', v_aaaa, t1_caae, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,iv,Xzuyvw->XA', v_aaaa, t1_caae, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixuA,iv,Xzuywv->XA', v_aaaa, t1_caae, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/240 * einsum('xyzw,ixuA,ivst,Xzuvstwy->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/240 * einsum('xyzw,ixuA,ivst,Xzuvstyw->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/160 * einsum('xyzw,ixuA,ivst,Xzuvswty->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/160 * einsum('xyzw,ixuA,ivst,Xzuvswyt->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 13/480 * einsum('xyzw,ixuA,ivst,Xzuvsytw->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 19/480 * einsum('xyzw,ixuA,ivst,Xzuvsywt->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/240 * einsum('xyzw,ixuA,ivst,Xzuvtswy->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/240 * einsum('xyzw,ixuA,ivst,Xzuvtsyw->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/160 * einsum('xyzw,ixuA,ivst,Xzuvtwsy->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 3/160 * einsum('xyzw,ixuA,ivst,Xzuvtwys->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 13/480 * einsum('xyzw,ixuA,ivst,Xzuvtysw->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/480 * einsum('xyzw,ixuA,ivst,Xzuvtyws->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 3/160 * einsum('xyzw,ixuA,ivst,Xzuvwsty->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/160 * einsum('xyzw,ixuA,ivst,Xzuvwsyt->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 13/480 * einsum('xyzw,ixuA,ivst,Xzuvwtsy->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 19/480 * einsum('xyzw,ixuA,ivst,Xzuvwtys->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/60 * einsum('xyzw,ixuA,ivst,Xzuvwyst->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,ixuA,ivst,Xzuvwyts->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/32 * einsum('xyzw,ixuA,ivst,Xzuvystw->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/160 * einsum('xyzw,ixuA,ivst,Xzuvyswt->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/480 * einsum('xyzw,ixuA,ivst,Xzuvytsw->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/96 * einsum('xyzw,ixuA,ivst,Xzuvytws->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 13/60 * einsum('xyzw,ixuA,ivst,Xzuvywst->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,ixuA,ivst,Xzuvywts->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,ivsu,Xzvswy->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,ivsu,Xzvsyw->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,ivsu,Xzvwsy->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,ivsu,Xzvwys->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,ivsu,Xzvysw->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixuA,ivsu,Xzvyws->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuA,ivsz,Xuvysw->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixuA,ivus,Xzvswy->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixuA,ivus,Xzvsyw->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixuA,ivus,Xzvwsy->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixuA,ivus,Xzvwys->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixuA,ivus,Xzvysw->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,ixuA,ivus,Xzvyws->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ixuA,ivuz,Xvyw->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,ivzs,Xuvswy->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,ivzs,Xuvsyw->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,ivzs,Xuvwsy->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,ivzs,Xuvwys->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuA,ivzs,Xuvysw->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,ixuA,ivzs,Xuvyws->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuA,ivzu,Xvyw->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuA,iz,Xuyw->XA', v_aaaa, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,ixua,ivAa,Xywuvz->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixua,ivAa,Xywuzv->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixua,ivAa,Xywvuz->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixua,ivAa,Xywvzu->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixua,ivAa,Xywzuv->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixua,ivAa,Xywzvu->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixua,ivaA,Xywuvz->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixua,ivaA,Xywuzv->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,ixua,ivaA,Xywvuz->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixua,ivaA,Xywvzu->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixua,ivaA,Xywzuv->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixua,ivaA,Xywzvu->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 3/16 * einsum('xyzw,ixuv,iA,Xywuvz->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,ixuv,iA,Xywuzv->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,ixuv,iA,Xywvuz->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,ixuv,iA,Xywvzu->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,ixuv,iA,Xywzuv->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,ixuv,iA,Xywzvu->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/30 * einsum('xyzw,ixuv,isAt,Xywtsuvz->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/15 * einsum('xyzw,ixuv,isAt,Xywtsvuz->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/240 * einsum('xyzw,ixuv,isAt,Xywtsvzu->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/240 * einsum('xyzw,ixuv,isAt,Xywtszuv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/20 * einsum('xyzw,ixuv,isAt,Xywtszvu->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,ixuv,isAt,Xywtusvz->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,ixuv,isAt,Xywtuszv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,ixuv,isAt,Xywtuvsz->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 31/240 * einsum('xyzw,ixuv,isAt,Xywtuvzs->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/40 * einsum('xyzw,ixuv,isAt,Xywtuzvs->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,ixuv,isAt,Xywtvsuz->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/30 * einsum('xyzw,ixuv,isAt,Xywtvszu->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,ixuv,isAt,Xywtvusz->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/30 * einsum('xyzw,ixuv,isAt,Xywtvuzs->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,ixuv,isAt,Xywtvzsu->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/80 * einsum('xyzw,ixuv,isAt,Xywtvzus->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,ixuv,isAt,Xywtzsuv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,ixuv,isAt,Xywtzsvu->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,ixuv,isAt,Xywtzusv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/240 * einsum('xyzw,ixuv,isAt,Xywtzuvs->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/40 * einsum('xyzw,ixuv,isAt,Xywtzvsu->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/48 * einsum('xyzw,ixuv,isAt,Xywtzvus->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,ixuv,isAu,Xywsvz->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isAu,Xywszv->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isAu,Xywvsz->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isAu,Xywvzs->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isAu,Xywzsv->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isAu,Xywzvs->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isAv,Xywsuz->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isAv,Xywszu->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,ixuv,isAv,Xywusz->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isAv,Xywuzs->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isAv,Xywzsu->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isAv,Xywzus->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 7/240 * einsum('xyzw,ixuv,istA,Xywtsuvz->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/240 * einsum('xyzw,ixuv,istA,Xywtsuzv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,ixuv,istA,Xywtsvuz->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 5/48 * einsum('xyzw,ixuv,istA,Xywtsvzu->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/240 * einsum('xyzw,ixuv,istA,Xywtszuv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,ixuv,istA,Xywtszvu->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,ixuv,istA,Xywtusvz->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/30 * einsum('xyzw,ixuv,istA,Xywtuszv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,ixuv,istA,Xywtuvsz->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/120 * einsum('xyzw,ixuv,istA,Xywtuvzs->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,ixuv,istA,Xywtuzsv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/15 * einsum('xyzw,ixuv,istA,Xywtuzvs->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyzw,ixuv,istA,Xywtvszu->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/20 * einsum('xyzw,ixuv,istA,Xywtvuzs->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/20 * einsum('xyzw,ixuv,istA,Xywtvzsu->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,ixuv,istA,Xywtvzus->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/60 * einsum('xyzw,ixuv,istA,Xywtzsuv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/30 * einsum('xyzw,ixuv,istA,Xywtzsvu->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/60 * einsum('xyzw,ixuv,istA,Xywtzusv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,ixuv,istA,Xywtzuvs->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,ixuv,istA,Xywtzvsu->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,ixuv,istA,Xywtzvus->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/12 * einsum('xyzw,ixuv,isuA,Xywsvz->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,isuA,Xywszv->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,isuA,Xywvsz->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,isuA,Xywvzs->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,isuA,Xywzsv->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,isuA,Xywzvs->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,ixuv,isvA,Xywsuz->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isvA,Xywszu->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isvA,Xywusz->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isvA,Xywuzs->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isvA,Xywzsu->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,isvA,Xywzus->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,uvxa,stAa,Xzuvstwy->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/160 * einsum('xyzw,uvxa,stAa,Xzuvstyw->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 13/480 * einsum('xyzw,uvxa,stAa,Xzuvswty->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 5/48 * einsum('xyzw,uvxa,stAa,Xzuvswyt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/480 * einsum('xyzw,uvxa,stAa,Xzuvsytw->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/240 * einsum('xyzw,uvxa,stAa,Xzuvsywt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/160 * einsum('xyzw,uvxa,stAa,Xzuvtswy->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/32 * einsum('xyzw,uvxa,stAa,Xzuvtsyw->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/32 * einsum('xyzw,uvxa,stAa,Xzuvtwsy->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/120 * einsum('xyzw,uvxa,stAa,Xzuvtwys->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/96 * einsum('xyzw,uvxa,stAa,Xzuvtysw->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/120 * einsum('xyzw,uvxa,stAa,Xzuvtyws->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/30 * einsum('xyzw,uvxa,stAa,Xzuvwsty->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 41/480 * einsum('xyzw,uvxa,stAa,Xzuvwsyt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/60 * einsum('xyzw,uvxa,stAa,Xzuvwtsy->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/96 * einsum('xyzw,uvxa,stAa,Xzuvwtys->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/160 * einsum('xyzw,uvxa,stAa,Xzuvwyst->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 23/480 * einsum('xyzw,uvxa,stAa,Xzuvwyts->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,uvxa,stAa,Xzuvystw->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/480 * einsum('xyzw,uvxa,stAa,Xzuvyswt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/240 * einsum('xyzw,uvxa,stAa,Xzuvytsw->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 5/96 * einsum('xyzw,uvxa,stAa,Xzuvytws->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 47/480 * einsum('xyzw,uvxa,stAa,Xzuvywst->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/160 * einsum('xyzw,uvxa,stAa,Xzuvywts->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,uxAa,va,Xzvuwy->XA', v_aaaa, t1_aaee, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAa,va,Xzvuyw->XA', v_aaaa, t1_aaee, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAa,va,Xzvwuy->XA', v_aaaa, t1_aaee, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAa,va,Xzvwyu->XA', v_aaaa, t1_aaee, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAa,va,Xzvyuw->XA', v_aaaa, t1_aaee, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAa,va,Xzvywu->XA', v_aaaa, t1_aaee, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 += 11/240 * einsum('xyzw,uxAa,vsta,Xzsvtuwy->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 13/240 * einsum('xyzw,uxAa,vsta,Xzsvtuyw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,uxAa,vsta,Xzsvtwuy->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/120 * einsum('xyzw,uxAa,vsta,Xzsvtwyu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/240 * einsum('xyzw,uxAa,vsta,Xzsvtyuw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/20 * einsum('xyzw,uxAa,vsta,Xzsvtywu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/30 * einsum('xyzw,uxAa,vsta,Xzsvutwy->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/15 * einsum('xyzw,uxAa,vsta,Xzsvutyw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,uxAa,vsta,Xzsvuwty->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/48 * einsum('xyzw,uxAa,vsta,Xzsvuwyt->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 3/80 * einsum('xyzw,uxAa,vsta,Xzsvuywt->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/30 * einsum('xyzw,uxAa,vsta,Xzsvwtuy->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,uxAa,vsta,Xzsvwtyu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/20 * einsum('xyzw,uxAa,vsta,Xzsvwuty->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/80 * einsum('xyzw,uxAa,vsta,Xzsvwuyt->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 13/240 * einsum('xyzw,uxAa,vsta,Xzsvwytu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 3/80 * einsum('xyzw,uxAa,vsta,Xzsvwyut->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,uxAa,vsta,Xzsvytuw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,uxAa,vsta,Xzsvytwu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/40 * einsum('xyzw,uxAa,vsta,Xzsvyutw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,uxAa,vsta,Xzsvyuwt->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,uxAa,vsta,Xzsvywtu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 17/240 * einsum('xyzw,uxAa,vsta,Xzsvywut->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,uxAa,vsza,Xvsuwy->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAa,vsza,Xvsuyw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAa,vsza,Xvswuy->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAa,vsza,Xvswyu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAa,vsza,Xvsyuw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAa,vsza,Xvsywu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 17/480 * einsum('xyzw,uxva,stAa,Xywustvz->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 21/160 * einsum('xyzw,uxva,stAa,Xywustzv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/160 * einsum('xyzw,uxva,stAa,Xywusvtz->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/240 * einsum('xyzw,uxva,stAa,Xywusvzt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 19/480 * einsum('xyzw,uxva,stAa,Xywusztv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/240 * einsum('xyzw,uxva,stAa,Xywuszvt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/480 * einsum('xyzw,uxva,stAa,Xywutsvz->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 41/480 * einsum('xyzw,uxva,stAa,Xywutszv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/160 * einsum('xyzw,uxva,stAa,Xywutvsz->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/240 * einsum('xyzw,uxva,stAa,Xywutvzs->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 19/480 * einsum('xyzw,uxva,stAa,Xywutzsv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/240 * einsum('xyzw,uxva,stAa,Xywutzvs->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/30 * einsum('xyzw,uxva,stAa,Xywuvstz->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/480 * einsum('xyzw,uxva,stAa,Xywuvszt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 23/480 * einsum('xyzw,uxva,stAa,Xywuvtzs->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/32 * einsum('xyzw,uxva,stAa,Xywuvzst->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/32 * einsum('xyzw,uxva,stAa,Xywuvzts->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,uxva,stAa,Xywuzstv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/480 * einsum('xyzw,uxva,stAa,Xywuzsvt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 11/240 * einsum('xyzw,uxva,stAa,Xywuztsv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/96 * einsum('xyzw,uxva,stAa,Xywuztvs->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/160 * einsum('xyzw,uxva,stAa,Xywuzvst->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/160 * einsum('xyzw,uxva,stAa,Xywuzvts->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/48 * einsum('xyzw,xa,uvAa,Xywuvz->XA', v_aaaa, t1_ae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,xa,uvAa,Xywuzv->XA', v_aaaa, t1_ae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,xa,uvAa,Xywvuz->XA', v_aaaa, t1_ae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,xa,uvAa,Xywvzu->XA', v_aaaa, t1_ae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,xa,uvAa,Xywzuv->XA', v_aaaa, t1_ae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,xa,uvAa,Xywzvu->XA', v_aaaa, t1_ae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,xuAa,va,Xzvuwy->XA', v_aaaa, t1_aaee, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,xuAa,va,Xzvuyw->XA', v_aaaa, t1_aaee, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,xuAa,va,Xzvwuy->XA', v_aaaa, t1_aaee, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,xuAa,va,Xzvwyu->XA', v_aaaa, t1_aaee, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,xuAa,va,Xzvyuw->XA', v_aaaa, t1_aaee, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,xuAa,va,Xzvywu->XA', v_aaaa, t1_aaee, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/240 * einsum('xyzw,xuAa,vsta,Xzsvtwuy->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/60 * einsum('xyzw,xuAa,vsta,Xzsvtwyu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 3/80 * einsum('xyzw,xuAa,vsta,Xzsvtyuw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/60 * einsum('xyzw,xuAa,vsta,Xzsvtywu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,xuAa,vsta,Xzsvuwty->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/240 * einsum('xyzw,xuAa,vsta,Xzsvuwyt->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,xuAa,vsta,Xzsvuytw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/240 * einsum('xyzw,xuAa,vsta,Xzsvuywt->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/80 * einsum('xyzw,xuAa,vsta,Xzsvwtuy->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/30 * einsum('xyzw,xuAa,vsta,Xzsvwtyu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/40 * einsum('xyzw,xuAa,vsta,Xzsvwuty->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,xuAa,vsta,Xzsvwuyt->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,xuAa,vsta,Xzsvwytu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/60 * einsum('xyzw,xuAa,vsta,Xzsvwyut->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 3/80 * einsum('xyzw,xuAa,vsta,Xzsvytuw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/60 * einsum('xyzw,xuAa,vsta,Xzsvytwu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,xuAa,vsta,Xzsvyutw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/240 * einsum('xyzw,xuAa,vsta,Xzsvyuwt->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,xuAa,vsta,Xzsvywtu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 13/60 * einsum('xyzw,xuAa,vsta,Xzsvywut->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,xuAa,vsza,Xvsuwy->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,xuAa,vsza,Xvsuyw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,xuAa,vsza,Xvswuy->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,xuAa,vsza,Xvswyu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,xuAa,vsza,Xvsyuw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 11/48 * einsum('xyzw,xuAa,vsza,Xvsywu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 17/480 * einsum('xyzw,xuva,stAa,Xywustvz->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,xuva,stAa,Xywustzv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,xuva,stAa,Xywusvtz->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/48 * einsum('xyzw,xuva,stAa,Xywusvzt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 17/480 * einsum('xyzw,xuva,stAa,Xywusztv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,xuva,stAa,Xywuszvt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 17/480 * einsum('xyzw,xuva,stAa,Xywutsvz->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,xuva,stAa,Xywutszv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 23/480 * einsum('xyzw,xuva,stAa,Xywutvsz->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,xuva,stAa,Xywutvzs->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/160 * einsum('xyzw,xuva,stAa,Xywutzsv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 17/240 * einsum('xyzw,xuva,stAa,Xywutzvs->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/60 * einsum('xyzw,xuva,stAa,Xywuvstz->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 47/480 * einsum('xyzw,xuva,stAa,Xywuvszt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/480 * einsum('xyzw,xuva,stAa,Xywuvtzs->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 23/480 * einsum('xyzw,xuva,stAa,Xywuvzst->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/96 * einsum('xyzw,xuva,stAa,Xywuvzts->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/60 * einsum('xyzw,xuva,stAa,Xywuzstv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,xuva,stAa,Xywuzsvt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/96 * einsum('xyzw,xuva,stAa,Xywuztvs->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 11/96 * einsum('xyzw,xuva,stAa,Xywuzvst->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/480 * einsum('xyzw,xuva,stAa,Xywuzvts->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,xzAa,ua,Xuyw->XA', v_aaaa, t1_aaee, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,xzAa,uvsa,Xvuswy->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,xzAa,uvsa,Xvusyw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,xzAa,uvsa,Xvuwsy->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,xzAa,uvsa,Xvuwys->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,xzAa,uvsa,Xvuysw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 3/16 * einsum('xyzw,xzAa,uvsa,Xvuyws->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,zxua,vsAa,Xywsuv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,zxua,vsAa,Xywsvu->XA', v_aaaa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,zxua,vsAa,Xywusv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,zxua,vsAa,Xywuvs->XA', v_aaaa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 += 11/48 * einsum('xyzw,zxua,vsAa,Xywvsu->XA', v_aaaa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,zxua,vsAa,Xywvus->XA', v_aaaa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)

    chunks = tools.calculate_double_chunks(mr_adc, ncore, [nextern, nextern, nextern],
                                                                     [ncas, nextern, nextern], ntensors = 2)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        if interface.with_df:
            v_ceee = mr_adc_integrals.get_oeee_df(mr_adc, mr_adc.v2e.Lce, mr_adc.v2e.Lee, s_chunk, f_chunk)
        else:
            v_ceee = mr_adc_integrals.unpack_v2e_oeee(mr_adc, mr_adc.v2e.ceee[s_chunk:f_chunk])

        ## Amplitudes
        t1_caee = mr_adc.t1.caee[s_chunk:f_chunk]

        V1 -= einsum('ixab,iaAb,Xx->XA', t1_caee, v_ceee, rdm_ca, optimize = einsum_type)
        V1 += 1/2 * einsum('ixab,ibAa,Xx->XA', t1_caee, v_ceee, rdm_ca, optimize = einsum_type)

    del(v_ceee)
    mr_adc.log.timer_debug("contracting v2e.ceee", *cput1)

    chunks = tools.calculate_double_chunks(mr_adc, ncas, [nextern, nextern, nextern],
                                                                     [ncas, nextern, nextern], ntensors = 2)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.aeee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        if interface.with_df:
            v_aeee = mr_adc_integrals.get_oeee_df(mr_adc, mr_adc.v2e.Lae, mr_adc.v2e.Lee, s_chunk, f_chunk)
        else:
            v_aeee = mr_adc_integrals.unpack_v2e_oeee(mr_adc, mr_adc.v2e.aeee[s_chunk:f_chunk])

        ##Amplitudes
        t1_aaee = mr_adc.t1.aaee[s_chunk:f_chunk]
        rdm_ccaa = rdm_ccaa[:, s_chunk:f_chunk]
    
        V1 -= 1/2 * einsum('xyab,zbAa,Xzxy->XA', t1_aaee, v_aeee, rdm_ccaa, optimize = einsum_type) 
    
    del(v_aaee)
    mr_adc.log.timer_debug("contracting v2e.aeee", *cput1)

    S_12_V = np.einsum("Pa,Pm->ma", V1, S_m1_12_inv_act)
    S_12_V = np.einsum("mp,ma->pa", evecs, S_12_V)

    # Compute denominators
    d_pa = (evals[:,None] + e_extern)
    d_pa = d_pa**(-1)

    S_12_V *= d_pa
    S_12_V = np.einsum("mp,pa->ma", evecs, S_12_V)

    # Compute T2[-1'] t2_ae amplitudes
    t2_ae = np.einsum("Pm,ma->Pa", S_m1_12_inv_act, S_12_V)

    mr_adc.log.extra("Norm of T[-1']^(2):                          %20.12f" % np.linalg.norm(t2_ae))
    mr_adc.log.timer("computing T[-1']^(2) amplitudes", *cput0)

    return t2_ae

def compute_t2_p1p_singles(mr_adc):
    ##TODO: finish tests for correctness

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing T[+1']^(2) amplitudes...")

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
    h_ce = mr_adc.h1eff.ce
    h_ca = mr_adc.h1eff.ca
    h_ae = mr_adc.h1eff.ae
    h_aa = mr_adc.h1eff.aa

    ## Two-electron integrals
    v_aaaa = mr_adc.v2e.aaaa
    
    v_ccce = mr_adc.v2e.ccce
    v_ccca = mr_adc.v2e.ccca
    
    v_cece = mr_adc.v2e.cece
    v_caca = mr_adc.v2e.caca    
    v_ccaa = mr_adc.v2e.ccaa
    v_caac = mr_adc.v2e.caac    
 
    v_cace = mr_adc.v2e.cace 
    v_ccae = mr_adc.v2e.ccae
    v_caec = mr_adc.v2e.caec
    
    v_caaa = mr_adc.v2e.caaa

    v_ceae = mr_adc.v2e.ceae
    v_aeae = mr_adc.v2e.aeae

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
    t1_aaee = mr_adc.t1.aaee

    t1_ccee = mr_adc.t1.ccee
    t1_ccae = mr_adc.t1.ccae
    t1_ccaa = mr_adc.t1.ccaa
    t1_caee = mr_adc.t1.caee

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa
    rdm_ccccaaaa = mr_adc.rdm.ccccaaaa

    # Compute K_ac matrix
    K_p1p = mr_adc_intermediates.compute_K_ac(mr_adc)
    
    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_p1p_12_inv_act = mr_adc_overlap.compute_S12_p1(mr_adc)

    # Compute K^{-1} matrix
    SKS = reduce(np.dot, (S_p1p_12_inv_act.T, K_p1p, S_p1p_12_inv_act))
    evals, evecs = np.linalg.eigh(SKS)
    del(SKS)

    # Compute R.H.S. of the equation
    # V1 block: - 1/2 < Psi_0 | a^{\dag}_I a_X  [V + H^{(1)}, T - T^\dag] | Psi_0 > 
    V1 =- einsum('Ia,Xa->IX', h_ce, t1_ae, optimize = einsum_type)
    V1 -= einsum('Xa,Ia->IX', h_ae, t1_ce, optimize = einsum_type)
    V1 -= 2 * einsum('ia,IiXa->IX', h_ce, t1_ccae, optimize = einsum_type)
    V1 += einsum('ia,iIXa->IX', h_ce, t1_ccae, optimize = einsum_type)
    V1 -= 2 * einsum('ix,IiXx->IX', h_ca, t1_ccaa, optimize = einsum_type)
    V1 += einsum('ix,IixX->IX', h_ca, t1_ccaa, optimize = einsum_type)
    V1 += einsum('Iiab,iaXb->IX', t1_ccee, v_ceae, optimize = einsum_type)
    V1 -= 2 * einsum('Iiab,ibXa->IX', t1_ccee, v_ceae, optimize = einsum_type)
    V1 -= 2 * einsum('Iixa,iaXx->IX', t1_ccae, v_ceaa, optimize = einsum_type)
    V1 += einsum('Iixa,ixXa->IX', t1_ccae, v_caae, optimize = einsum_type)
    V1 += einsum('Iixy,ixXy->IX', t1_ccaa, v_caaa, optimize = einsum_type)
    V1 -= 2 * einsum('Iixy,iyXx->IX', t1_ccaa, v_caaa, optimize = einsum_type)
    V1 += einsum('iIxa,iaXx->IX', t1_ccae, v_ceaa, optimize = einsum_type)
    V1 -= 2 * einsum('iIxa,ixXa->IX', t1_ccae, v_caae, optimize = einsum_type)
    V1 += einsum('iXab,Iaib->IX', t1_caee, v_cece, optimize = einsum_type)
    V1 -= 2 * einsum('iXab,Ibia->IX', t1_caee, v_cece, optimize = einsum_type)
    V1 -= 2 * einsum('iXax,Ixia->IX', t1_caea, v_cace, optimize = einsum_type)
    V1 += einsum('iXax,ixIa->IX', t1_caea, v_cace, optimize = einsum_type)
    V1 += einsum('iXxa,Ixia->IX', t1_caae, v_cace, optimize = einsum_type)
    V1 -= 2 * einsum('iXxa,ixIa->IX', t1_caae, v_cace, optimize = einsum_type)
    V1 += einsum('iXxy,Ixiy->IX', t1_caaa, v_caca, optimize = einsum_type)
    V1 -= 2 * einsum('iXxy,Iyix->IX', t1_caaa, v_caca, optimize = einsum_type)
    V1 -= 2 * einsum('ia,IXai->IX', t1_ce, v_caec, optimize = einsum_type)
    V1 -= 2 * einsum('ia,IXia->IX', t1_ce, v_cace, optimize = einsum_type)
    V1 += einsum('ia,iIXa->IX', t1_ce, v_ccae, optimize = einsum_type)
    V1 += einsum('ia,iXIa->IX', t1_ce, v_cace, optimize = einsum_type)
    V1 += 2 * einsum('ijXa,iIja->IX', t1_ccae, v_ccce, optimize = einsum_type)
    V1 -= einsum('ijXa,jIia->IX', t1_ccae, v_ccce, optimize = einsum_type)
    V1 += 2 * einsum('ijXx,iIjx->IX', t1_ccaa, v_ccca, optimize = einsum_type)
    V1 -= einsum('ijXx,jIix->IX', t1_ccaa, v_ccca, optimize = einsum_type)
    V1 -= 2 * einsum('ix,IXix->IX', t1_ca, v_caca, optimize = einsum_type)
    V1 -= 2 * einsum('ix,IXxi->IX', t1_ca, v_caac, optimize = einsum_type)
    V1 += einsum('ix,IixX->IX', t1_ca, v_ccaa, optimize = einsum_type)
    V1 += einsum('ix,IxiX->IX', t1_ca, v_caca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,Ia,Xa->IX', e_core, t1_ce, t1_ae, optimize = einsum_type)
    V1 += einsum('I,IiXa,ia->IX', e_core, t1_ccae, t1_ce, optimize = einsum_type)
    V1 += einsum('I,IiXx,ix->IX', e_core, t1_ccaa, t1_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iiab,iXab->IX', e_core, t1_ccee, t1_caee, optimize = einsum_type)
    V1 += einsum('I,Iiab,iXba->IX', e_core, t1_ccee, t1_caee, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,IixX,ix->IX', e_core, t1_ccaa, t1_ca, optimize = einsum_type)
    V1 += einsum('I,Iixa,iXax->IX', e_core, t1_ccae, t1_caea, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iixa,iXxa->IX', e_core, t1_ccae, t1_caae, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iixy,iXxy->IX', e_core, t1_ccaa, t1_caaa, optimize = einsum_type)
    V1 += einsum('I,Iixy,iXyx->IX', e_core, t1_ccaa, t1_caaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,iIXa,ia->IX', e_core, t1_ccae, t1_ce, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,iIxa,iXax->IX', e_core, t1_ccae, t1_caea, optimize = einsum_type)
    V1 += einsum('I,iIxa,iXxa->IX', e_core, t1_ccae, t1_caae, optimize = einsum_type)
    V1 -= einsum('a,Xa,Ia->IX', e_extern, t1_ae, t1_ce, optimize = einsum_type)
    V1 += einsum('a,iXab,Iiab->IX', e_extern, t1_caee, t1_ccee, optimize = einsum_type)
    V1 -= 2 * einsum('a,iXab,Iiba->IX', e_extern, t1_caee, t1_ccee, optimize = einsum_type)
    V1 -= 2 * einsum('a,iXax,Iixa->IX', e_extern, t1_caea, t1_ccae, optimize = einsum_type)
    V1 += einsum('a,iXax,iIxa->IX', e_extern, t1_caea, t1_ccae, optimize = einsum_type)
    V1 -= 2 * einsum('a,iXba,Iiab->IX', e_extern, t1_caee, t1_ccee, optimize = einsum_type)
    V1 += einsum('a,iXba,Iiba->IX', e_extern, t1_caee, t1_ccee, optimize = einsum_type)
    V1 += einsum('a,iXxa,Iixa->IX', e_extern, t1_caae, t1_ccae, optimize = einsum_type)
    V1 -= 2 * einsum('a,iXxa,iIxa->IX', e_extern, t1_caae, t1_ccae, optimize = einsum_type)
    V1 -= 2 * einsum('a,ia,IiXa->IX', e_extern, t1_ce, t1_ccae, optimize = einsum_type)
    V1 += einsum('a,ia,iIXa->IX', e_extern, t1_ce, t1_ccae, optimize = einsum_type)
    V1 += einsum('i,IiXa,ia->IX', e_core, t1_ccae, t1_ce, optimize = einsum_type)
    V1 += einsum('i,IiXx,ix->IX', e_core, t1_ccaa, t1_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,Iiab,iXab->IX', e_core, t1_ccee, t1_caee, optimize = einsum_type)
    V1 += einsum('i,Iiab,iXba->IX', e_core, t1_ccee, t1_caee, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,IixX,ix->IX', e_core, t1_ccaa, t1_ca, optimize = einsum_type)
    V1 += einsum('i,Iixa,iXax->IX', e_core, t1_ccae, t1_caea, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,Iixa,iXxa->IX', e_core, t1_ccae, t1_caae, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,Iixy,iXxy->IX', e_core, t1_ccaa, t1_caaa, optimize = einsum_type)
    V1 += einsum('i,Iixy,iXyx->IX', e_core, t1_ccaa, t1_caaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iIXa,ia->IX', e_core, t1_ccae, t1_ce, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iIxa,iXax->IX', e_core, t1_ccae, t1_caea, optimize = einsum_type)
    V1 += einsum('i,iIxa,iXxa->IX', e_core, t1_ccae, t1_caae, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iXab,Iiab->IX', e_core, t1_caee, t1_ccee, optimize = einsum_type)
    V1 += einsum('i,iXab,Iiba->IX', e_core, t1_caee, t1_ccee, optimize = einsum_type)
    V1 += einsum('i,iXax,Iixa->IX', e_core, t1_caea, t1_ccae, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iXax,iIxa->IX', e_core, t1_caea, t1_ccae, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iXxa,Iixa->IX', e_core, t1_caae, t1_ccae, optimize = einsum_type)
    V1 += einsum('i,iXxa,iIxa->IX', e_core, t1_caae, t1_ccae, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iXxy,Iixy->IX', e_core, t1_caaa, t1_ccaa, optimize = einsum_type)
    V1 += einsum('i,iXxy,Iiyx->IX', e_core, t1_caaa, t1_ccaa, optimize = einsum_type)
    V1 += einsum('i,ia,IiXa->IX', e_core, t1_ce, t1_ccae, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ia,iIXa->IX', e_core, t1_ce, t1_ccae, optimize = einsum_type)
    V1 += einsum('i,ix,IiXx->IX', e_core, t1_ca, t1_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ix,IixX->IX', e_core, t1_ca, t1_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ia,Xxya,xy->IX', h_ce, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Ia,xXya,xy->IX', h_ce, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ia,xa,Xx->IX', h_ce, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ia,xyza,Xzyx->IX', h_ce, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('Xa,Ixay,xy->IX', h_ae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xa,Ixya,xy->IX', h_ae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += einsum('ia,Iixa,Xx->IX', h_ce, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ia,iIxa,Xx->IX', h_ce, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('ix,IiXy,xy->IX', h_ca, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,Iixy,Xy->IX', h_ca, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,IiyX,xy->IX', h_ca, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('ix,Iiyx,Xy->IX', h_ca, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,Iiyz,Xxyz->IX', h_ca, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xa,Ia,Xx->IX', h_ae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xa,IyXa,xy->IX', h_ae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xa,IyaX,xy->IX', h_ae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xa,Iyaz,Xyxz->IX', h_ae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xa,Iyza,Xyzx->IX', h_ae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('Xx,Iixa,ia->IX', h_aa, t1_ccae, t1_ce, optimize = einsum_type)
    V1 -= einsum('Xx,Iixy,iy->IX', h_aa, t1_ccaa, t1_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,Iiyx,iy->IX', h_aa, t1_ccaa, t1_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,iIxa,ia->IX', h_aa, t1_ccae, t1_ce, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xx,ixab,Iiab->IX', h_aa, t1_caee, t1_ccee, optimize = einsum_type)
    V1 += einsum('Xx,ixab,Iiba->IX', h_aa, t1_caee, t1_ccee, optimize = einsum_type)
    V1 += einsum('Xx,ixay,Iiya->IX', h_aa, t1_caea, t1_ccae, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xx,ixay,iIya->IX', h_aa, t1_caea, t1_ccae, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xx,ixya,Iiya->IX', h_aa, t1_caae, t1_ccae, optimize = einsum_type)
    V1 += einsum('Xx,ixya,iIya->IX', h_aa, t1_caae, t1_ccae, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xx,ixyz,Iiyz->IX', h_aa, t1_caaa, t1_ccaa, optimize = einsum_type)
    V1 += einsum('Xx,ixyz,Iizy->IX', h_aa, t1_caaa, t1_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,xa,Ia->IX', h_aa, t1_ae, t1_ce, optimize = einsum_type)
    V1 -= 2 * einsum('xy,IiXx,iy->IX', h_aa, t1_ccaa, t1_ca, optimize = einsum_type)
    V1 += einsum('xy,IixX,iy->IX', h_aa, t1_ccaa, t1_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xy,Iixa,iXay->IX', h_aa, t1_ccae, t1_caea, optimize = einsum_type)
    V1 += einsum('xy,Iixa,iXya->IX', h_aa, t1_ccae, t1_caae, optimize = einsum_type)
    V1 += einsum('xy,Iixz,iXyz->IX', h_aa, t1_ccaa, t1_caaa, optimize = einsum_type)
    V1 -= 2 * einsum('xy,Iixz,iXzy->IX', h_aa, t1_ccaa, t1_caaa, optimize = einsum_type)
    V1 -= 2 * einsum('xy,Iizx,iXyz->IX', h_aa, t1_ccaa, t1_caaa, optimize = einsum_type)
    V1 += einsum('xy,Iizx,iXzy->IX', h_aa, t1_ccaa, t1_caaa, optimize = einsum_type)
    V1 += einsum('xy,iIxa,iXay->IX', h_aa, t1_ccae, t1_caea, optimize = einsum_type)
    V1 -= 2 * einsum('xy,iIxa,iXya->IX', h_aa, t1_ccae, t1_caae, optimize = einsum_type)
    V1 += 1/2 * einsum('Ia,Xxya,yx->IX', t1_ce, v_aaae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Ia,xyXa,xy->IX', t1_ce, v_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ia,xyza,Xyzx->IX', t1_ce, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('IiXa,iaxy,xy->IX', t1_ccae, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('IiXa,ixya,yx->IX', t1_ccae, v_caae, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('IiXx,ixyz,yz->IX', t1_ccaa, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('IiXx,iyzw,xzyw->IX', t1_ccaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('IiXx,iyzx,zy->IX', t1_ccaa, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iiab,iaxb,Xx->IX', t1_ccee, v_ceae, rdm_ca, optimize = einsum_type)
    V1 += einsum('Iiab,ibxa,Xx->IX', t1_ccee, v_ceae, rdm_ca, optimize = einsum_type)
    V1 += einsum('IixX,ixyz,yz->IX', t1_ccaa, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('IixX,iyzw,xzyw->IX', t1_ccaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('IixX,iyzx,zy->IX', t1_ccaa, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Iixa,iaXy,xy->IX', t1_ccae, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Iixa,iayx,Xy->IX', t1_ccae, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Iixa,iayz,Xzxy->IX', t1_ccae, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixa,ixya,Xy->IX', t1_ccae, v_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixa,iyXa,xy->IX', t1_ccae, v_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixa,iyza,Xyxz->IX', t1_ccae, v_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixy,ixXz,yz->IX', t1_ccaa, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixy,ixzw,Xwyz->IX', t1_ccaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixy,ixzy,Xz->IX', t1_ccaa, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Iixy,iyXz,xz->IX', t1_ccaa, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Iixy,iyzw,Xwxz->IX', t1_ccaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Iixy,iyzx,Xz->IX', t1_ccaa, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixy,izXw,xywz->IX', t1_ccaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Iixy,izXx,yz->IX', t1_ccaa, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixy,izXy,xz->IX', t1_ccaa, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixy,izwu,Xzuxyw->IX', t1_ccaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixy,izwx,Xzwy->IX', t1_ccaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixy,izwy,Xzxw->IX', t1_ccaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('IxXa,yzwa,xzwy->IX', t1_caae, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('IxaX,yzwa,xzwy->IX', t1_caea, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('Ixab,Xayb,xy->IX', t1_caee, v_aeae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixab,Xbya,xy->IX', t1_caee, v_aeae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixab,yazb,Xxyz->IX', t1_caee, v_aeae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixay,Xyza,xz->IX', t1_caea, v_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixay,Xzwa,xzyw->IX', t1_caea, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('Ixay,zwXa,xwyz->IX', t1_caea, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/3 * einsum('Ixay,zwua,Xxwuyz->IX', t1_caea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('Ixay,zwua,Xxwuzy->IX', t1_caea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('Ixay,zwua,Xxwyuz->IX', t1_caea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('Ixay,zwua,Xxwyzu->IX', t1_caea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('Ixay,zwua,Xxwzuy->IX', t1_caea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('Ixay,zwua,Xxwzyu->IX', t1_caea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('Ixay,zyXa,xz->IX', t1_caea, v_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixay,zywa,Xxwz->IX', t1_caea, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('Ixya,Xyza,xz->IX', t1_caae, v_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixya,Xzwa,xzwy->IX', t1_caae, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixya,zwXa,xwyz->IX', t1_caae, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixya,zwua,Xxwyuz->IX', t1_caae, v_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixya,zyXa,xz->IX', t1_caae, v_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixya,zywa,Xxzw->IX', t1_caae, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('Xa,Iaxy,xy->IX', t1_ae, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xa,Ixya,yx->IX', t1_ae, v_caae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Xxab,Iayb,xy->IX', t1_aaee, v_ceae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxab,Ibya,xy->IX', t1_aaee, v_ceae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxya,Iazw,xwyz->IX', t1_aaae, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxya,Iazy,xz->IX', t1_aaae, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Xxya,Iyza,xz->IX', t1_aaae, v_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxya,Izwa,xzwy->IX', t1_aaae, v_caae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('iIXa,iaxy,xy->IX', t1_ccae, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('iIXa,ixya,yx->IX', t1_ccae, v_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('iIxa,iaXy,xy->IX', t1_ccae, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('iIxa,iayx,Xy->IX', t1_ccae, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('iIxa,iayz,Xzxy->IX', t1_ccae, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('iIxa,ixya,Xy->IX', t1_ccae, v_caae, rdm_ca, optimize = einsum_type)
    V1 += einsum('iIxa,iyXa,xy->IX', t1_ccae, v_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('iIxa,iyza,Xyzx->IX', t1_ccae, v_caae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('iXax,Iyia,xy->IX', t1_caea, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('iXax,iyIa,xy->IX', t1_caea, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('iXxa,Iyia,xy->IX', t1_caae, v_cace, rdm_ca, optimize = einsum_type)
    V1 += einsum('iXxa,iyIa,xy->IX', t1_caae, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('iXxy,Ixiz,yz->IX', t1_caaa, v_caca, rdm_ca, optimize = einsum_type)
    V1 += einsum('iXxy,Iyiz,xz->IX', t1_caaa, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('iXxy,Iziw,yxzw->IX', t1_caaa, v_caca, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('iXxy,Izix,yz->IX', t1_caaa, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('iXxy,Iziy,xz->IX', t1_caaa, v_caca, rdm_ca, optimize = einsum_type)
    V1 += einsum('ia,Ixai,Xx->IX', t1_ce, v_caec, rdm_ca, optimize = einsum_type)
    V1 += einsum('ia,Ixia,Xx->IX', t1_ce, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ia,iIxa,Xx->IX', t1_ce, v_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ia,ixIa,Xx->IX', t1_ce, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ijXx,iIjy,xy->IX', t1_ccaa, v_ccca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ijXx,jIiy,xy->IX', t1_ccaa, v_ccca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ijxa,iIja,Xx->IX', t1_ccae, v_ccce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ijxa,jIia,Xx->IX', t1_ccae, v_ccce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ijxy,iIjx,Xy->IX', t1_ccaa, v_ccca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ijxy,iIjz,Xzxy->IX', t1_ccaa, v_ccca, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ijxy,jIix,Xy->IX', t1_ccaa, v_ccca, rdm_ca, optimize = einsum_type)
    V1 += einsum('ix,IXiy,xy->IX', t1_ca, v_caca, rdm_ca, optimize = einsum_type)
    V1 += einsum('ix,IXyi,xy->IX', t1_ca, v_caac, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,Iixy,Xy->IX', t1_ca, v_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,IiyX,xy->IX', t1_ca, v_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,Iiyz,Xyxz->IX', t1_ca, v_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,Ixiy,Xy->IX', t1_ca, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,IyiX,xy->IX', t1_ca, v_caca, rdm_ca, optimize = einsum_type)
    V1 += einsum('ix,Iyix,Xy->IX', t1_ca, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,Iyiz,Xxyz->IX', t1_ca, v_caca, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ix,Iyxi,Xy->IX', t1_ca, v_caac, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,Iyzi,Xzyx->IX', t1_ca, v_caac, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixXa,Iyai,xy->IX', t1_caae, v_caec, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixXa,iIya,xy->IX', t1_caae, v_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixXy,Iiyz,xz->IX', t1_caaa, v_ccaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixXy,Iizw,ywxz->IX', t1_caaa, v_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixXy,Izwi,yzxw->IX', t1_caaa, v_caac, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixXy,Izyi,xz->IX', t1_caaa, v_caac, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixaX,Iyai,xy->IX', t1_caea, v_caec, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixaX,iIya,xy->IX', t1_caea, v_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixab,Iaib,Xx->IX', t1_caee, v_cece, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixab,Ibia,Xx->IX', t1_caee, v_cece, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('ixay,IXai,yx->IX', t1_caea, v_caec, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('ixay,IXia,xy->IX', t1_caea, v_cace, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixay,Iyia,Xx->IX', t1_caea, v_cace, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixay,Izai,Xxzy->IX', t1_caea, v_caec, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ixay,Izia,Xyzx->IX', t1_caea, v_cace, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ixay,iIXa,yx->IX', t1_caea, v_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixay,iIza,Xxzy->IX', t1_caea, v_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ixay,iXIa,xy->IX', t1_caea, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixay,iyIa,Xx->IX', t1_caea, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixay,izIa,Xyzx->IX', t1_caea, v_cace, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyX,Iiyz,xz->IX', t1_caaa, v_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyX,Iizw,ywxz->IX', t1_caaa, v_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyX,Izwi,yzwx->IX', t1_caaa, v_caac, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ixyX,Izyi,xz->IX', t1_caaa, v_caac, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixya,IXai,yx->IX', t1_caae, v_caec, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixya,IXia,xy->IX', t1_caae, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixya,Iyia,Xx->IX', t1_caae, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixya,Izai,Xxzy->IX', t1_caae, v_caec, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixya,Izia,Xyzx->IX', t1_caae, v_cace, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixya,iIXa,yx->IX', t1_caae, v_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixya,iIza,Xxyz->IX', t1_caae, v_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixya,iXIa,xy->IX', t1_caae, v_cace, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixya,iyIa,Xx->IX', t1_caae, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixya,izIa,Xyxz->IX', t1_caae, v_cace, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ixyz,IXiw,xwzy->IX', t1_caaa, v_caca, rdm_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('ixyz,IXiy,xz->IX', t1_caaa, v_caca, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixyz,IXiz,xy->IX', t1_caaa, v_caca, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixyz,IXwi,xwzy->IX', t1_caaa, v_caac, rdm_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('ixyz,IXyi,zx->IX', t1_caaa, v_caac, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixyz,IXzi,yx->IX', t1_caaa, v_caac, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,IiwX,xwzy->IX', t1_caaa, v_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,Iiwu,Xxwyzu->IX', t1_caaa, v_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('ixyz,IiyX,zx->IX', t1_caaa, v_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,Iiyw,Xxwz->IX', t1_caaa, v_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,IizX,yx->IX', t1_caaa, v_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,Iizw,Xxyw->IX', t1_caaa, v_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,IwiX,xwzy->IX', t1_caaa, v_caca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,Iwiu,Xyzwux->IX', t1_caaa, v_caca, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('ixyz,Iwiy,Xzwx->IX', t1_caaa, v_caca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,Iwiz,Xywx->IX', t1_caaa, v_caca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/6 * einsum('ixyz,Iwui,Xxuwyz->IX', t1_caaa, v_caac, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('ixyz,Iwui,Xxuwzy->IX', t1_caaa, v_caac, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('ixyz,Iwui,Xxuywz->IX', t1_caaa, v_caac, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('ixyz,Iwui,Xxuyzw->IX', t1_caaa, v_caac, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('ixyz,Iwui,Xxuzwy->IX', t1_caaa, v_caac, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('ixyz,Iwui,Xxuzyw->IX', t1_caaa, v_caac, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('ixyz,Iwyi,Xxwz->IX', t1_caaa, v_caac, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,Iwzi,Xxwy->IX', t1_caaa, v_caac, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ixyz,IyiX,xz->IX', t1_caaa, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,Iyiw,Xzwx->IX', t1_caaa, v_caca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,Iyiz,Xx->IX', t1_caaa, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,IziX,xy->IX', t1_caaa, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,Iziw,Xyxw->IX', t1_caaa, v_caca, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ixyz,Iziy,Xx->IX', t1_caaa, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xXya,Iazw,xwyz->IX', t1_aaae, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xXya,Iazy,xz->IX', t1_aaae, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xXya,Iyza,xz->IX', t1_aaae, v_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xXya,Izwa,xzyw->IX', t1_aaae, v_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xa,IXay,xy->IX', t1_ae, v_caea, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xa,IXya,xy->IX', t1_ae, v_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xa,IayX,xy->IX', t1_ae, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xa,Iayz,Xyxz->IX', t1_ae, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xa,IyaX,xy->IX', t1_ae, v_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xa,Iyaz,Xxyz->IX', t1_ae, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xa,Iyza,Xzyx->IX', t1_ae, v_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyXa,Izaw,xyzw->IX', t1_aaae, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyab,Iazb,Xzxy->IX', t1_aaee, v_ceae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyza,IXaw,zwxy->IX', t1_aaae, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyza,IXwa,zwxy->IX', t1_aaae, v_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,IawX,zwxy->IX', t1_aaae, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,Iawu,Xzwyxu->IX', t1_aaae, v_ceaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,Iawz,Xwyx->IX', t1_aaae, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,IwaX,zwxy->IX', t1_aaae, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,Iwau,Xyxwuz->IX', t1_aaae, v_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyza,Iwua,Xzuwxy->IX', t1_aaae, v_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyza,Iwua,Xzuwyx->IX', t1_aaae, v_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyza,Iwua,Xzuxwy->IX', t1_aaae, v_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyza,Iwua,Xzuxyw->IX', t1_aaae, v_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyza,Iwua,Xzuywx->IX', t1_aaae, v_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyza,Iwua,Xzuyxw->IX', t1_aaae, v_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,Izwa,Xwxy->IX', t1_aaae, v_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('Xxyz,iy,Iixz->IX', v_aaaa, t1_ca, t1_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,iy,Iizx->IX', v_aaaa, t1_ca, t1_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('xyzw,Iixz,iXwy->IX', v_aaaa, t1_ccaa, t1_caaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixz,iXyw->IX', v_aaaa, t1_ccaa, t1_caaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ia,Xxya,xy->IX', e_core, t1_ce, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,Ia,xXya,xy->IX', e_core, t1_ce, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ia,xa,Xx->IX', e_core, t1_ce, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ia,xyza,Xzyx->IX', e_core, t1_ce, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('I,IiXa,ixay,xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,IiXa,ixya,xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,IiXx,iy,xy->IX', e_core, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += einsum('I,IiXx,iyxz,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,IiXx,iyzw,xyzw->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,IiXx,iyzx,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,Iiab,ixab,Xx->IX', e_core, t1_ccee, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iiab,ixba,Xx->IX', e_core, t1_ccee, t1_caee, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,IixX,iy,xy->IX', e_core, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,IixX,iyxz,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,IixX,iyzw,xyzw->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,IixX,iyzx,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iixa,iXay,xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,Iixa,iXya,xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iixa,ia,Xx->IX', e_core, t1_ccae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iixa,iyax,Xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iixa,iyaz,Xzxy->IX', e_core, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,Iixa,iyxa,Xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,Iixa,iyza,Xzxy->IX', e_core, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,Iixy,iXxz,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iixy,iXyz,xz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,Iixy,iXzw,xywz->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iixy,iXzx,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,Iixy,iXzy,xz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,Iixy,ix,Xy->IX', e_core, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iixy,iy,Xx->IX', e_core, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,Iixy,iz,Xzxy->IX', e_core, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,Iixy,izwu,Xwuxyz->IX', e_core, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,Iixy,izwx,Xwzy->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,Iixy,izwy,Xwxz->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,Iixy,izxw,Xwyz->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,Iixy,izxy,Xz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iixy,izyw,Xwxz->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iixy,izyx,Xz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,IxXa,ya,xy->IX', e_core, t1_caae, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,IxXa,yzwa,xwzy->IX', e_core, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,IxaX,ya,xy->IX', e_core, t1_caea, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,IxaX,yzwa,xwzy->IX', e_core, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('I,Ixab,Xyab,xy->IX', e_core, t1_caee, t1_aaee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixab,Xyba,xy->IX', e_core, t1_caee, t1_aaee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixab,yzab,Xxyz->IX', e_core, t1_caee, t1_aaee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('I,Ixay,Xa,xy->IX', e_core, t1_caea, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixay,Xzwa,xwyz->IX', e_core, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixay,Xzya,xz->IX', e_core, t1_caea, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,Ixay,zXwa,xwyz->IX', e_core, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('I,Ixay,zXya,xz->IX', e_core, t1_caea, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixay,za,Xxzy->IX', e_core, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('I,Ixay,zwua,Xxuwyz->IX', e_core, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('I,Ixay,zwua,Xxuwzy->IX', e_core, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('I,Ixay,zwua,Xxuywz->IX', e_core, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('I,Ixay,zwua,Xxuyzw->IX', e_core, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('I,Ixay,zwua,Xxuzwy->IX', e_core, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('I,Ixay,zwua,Xxuzyw->IX', e_core, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixay,zwya,Xxwz->IX', e_core, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixya,Xa,xy->IX', e_core, t1_caae, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixya,Xzwa,xwzy->IX', e_core, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('I,Ixya,Xzya,xz->IX', e_core, t1_caae, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixya,zXwa,xwyz->IX', e_core, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixya,zXya,xz->IX', e_core, t1_caae, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixya,za,Xxyz->IX', e_core, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixya,zwua,Xxuywz->IX', e_core, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixya,zwya,Xxzw->IX', e_core, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,iIXa,ixay,xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,iIXa,ixya,xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,iIxa,iXay,xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,iIxa,iXya,xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,iIxa,ia,Xx->IX', e_core, t1_ccae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,iIxa,iyax,Xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,iIxa,iyaz,Xzxy->IX', e_core, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,iIxa,iyxa,Xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,iIxa,iyza,Xzyx->IX', e_core, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('a,Xa,Ixay,xy->IX', e_extern, t1_ae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Xa,Ixya,xy->IX', e_extern, t1_ae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('a,Xxab,Iyab,xy->IX', e_extern, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Xxab,Iyba,xy->IX', e_extern, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Xxba,Iyab,xy->IX', e_extern, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= einsum('a,Xxba,Iyba,xy->IX', e_extern, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Xxya,Ia,xy->IX', e_extern, t1_aaae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Xxya,Izaw,xwyz->IX', e_extern, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Xxya,Izay,xz->IX', e_extern, t1_aaae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Xxya,Izwa,xwzy->IX', e_extern, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('a,Xxya,Izya,xz->IX', e_extern, t1_aaae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,iXax,Iiya,xy->IX', e_extern, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,iXax,iIya,xy->IX', e_extern, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,iXxa,Iiya,xy->IX', e_extern, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,iXxa,iIya,xy->IX', e_extern, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,ia,Iixa,Xx->IX', e_extern, t1_ce, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,ia,iIxa,Xx->IX', e_extern, t1_ce, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,ixab,Iiab,Xx->IX', e_extern, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,ixab,Iiba,Xx->IX', e_extern, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('a,ixay,IiXa,yx->IX', e_extern, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,ixay,Iiya,Xx->IX', e_extern, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,ixay,Iiza,Xyzx->IX', e_extern, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('a,ixay,iIXa,yx->IX', e_extern, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,ixay,iIya,Xx->IX', e_extern, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,ixay,iIza,Xyzx->IX', e_extern, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('a,ixba,Iiab,Xx->IX', e_extern, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,ixba,Iiba,Xx->IX', e_extern, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,ixya,IiXa,yx->IX', e_extern, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,ixya,Iiya,Xx->IX', e_extern, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,ixya,Iiza,Xyzx->IX', e_extern, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,ixya,iIXa,yx->IX', e_extern, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,ixya,iIya,Xx->IX', e_extern, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,ixya,iIza,Xyxz->IX', e_extern, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('a,xXya,Ia,xy->IX', e_extern, t1_aaae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= einsum('a,xXya,Izaw,xwyz->IX', e_extern, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('a,xXya,Izay,xz->IX', e_extern, t1_aaae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xXya,Izwa,xwyz->IX', e_extern, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xXya,Izya,xz->IX', e_extern, t1_aaae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xa,Ia,Xx->IX', e_extern, t1_ae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= einsum('a,xa,IyXa,xy->IX', e_extern, t1_ae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xa,IyaX,xy->IX', e_extern, t1_ae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xa,Iyaz,Xyxz->IX', e_extern, t1_ae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xa,Iyza,Xyzx->IX', e_extern, t1_ae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xyab,Izab,Xzxy->IX', e_extern, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xyab,Izba,Xzyx->IX', e_extern, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xyza,Ia,Xzyx->IX', e_extern, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('a,xyza,IwXa,zwxy->IX', e_extern, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xyza,IwaX,zwxy->IX', e_extern, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xyza,Iwau,Xzwyxu->IX', e_extern, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xyza,Iwaz,Xwyx->IX', e_extern, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/3 * einsum('a,xyza,Iwua,Xzwuxy->IX', e_extern, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('a,xyza,Iwua,Xzwuyx->IX', e_extern, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('a,xyza,Iwua,Xzwxuy->IX', e_extern, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('a,xyza,Iwua,Xzwxyu->IX', e_extern, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('a,xyza,Iwua,Xzwyux->IX', e_extern, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('a,xyza,Iwua,Xzwyxu->IX', e_extern, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xyza,Iwza,Xwxy->IX', e_extern, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('i,IiXa,ixay,xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,IiXa,ixya,xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,IiXx,iy,xy->IX', e_core, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += einsum('i,IiXx,iyxz,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,IiXx,iyzw,xyzw->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,IiXx,iyzx,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,Iiab,ixab,Xx->IX', e_core, t1_ccee, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,Iiab,ixba,Xx->IX', e_core, t1_ccee, t1_caee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,IixX,iy,xy->IX', e_core, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,IixX,iyxz,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,IixX,iyzw,xyzw->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,IixX,iyzx,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,Iixa,iXay,xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,Iixa,iXya,xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,Iixa,ia,Xx->IX', e_core, t1_ccae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,Iixa,iyax,Xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,Iixa,iyaz,Xzxy->IX', e_core, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,Iixa,iyxa,Xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,Iixa,iyza,Xzxy->IX', e_core, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,Iixy,iXxz,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,Iixy,iXyz,xz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixy,iXzw,xywz->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,Iixy,iXzx,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,Iixy,iXzy,xz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,Iixy,ix,Xy->IX', e_core, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,Iixy,iy,Xx->IX', e_core, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixy,iz,Xzxy->IX', e_core, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixy,izwu,Xwuxyz->IX', e_core, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,Iixy,izwx,Xwzy->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,Iixy,izwy,Xwxz->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,Iixy,izxw,Xwyz->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,Iixy,izxy,Xz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,Iixy,izyw,Xwxz->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,Iixy,izyx,Xz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iIXa,ixay,xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,iIXa,ixya,xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,iIxa,iXay,xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iIxa,iXya,xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,iIxa,ia,Xx->IX', e_core, t1_ccae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,iIxa,iyax,Xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,iIxa,iyaz,Xzxy->IX', e_core, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iIxa,iyxa,Xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,iIxa,iyza,Xzyx->IX', e_core, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iXax,Iiya,xy->IX', e_core, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,iXax,iIya,xy->IX', e_core, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,iXxa,Iiya,xy->IX', e_core, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iXxa,iIya,xy->IX', e_core, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,iXxy,Iixz,yz->IX', e_core, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iXxy,Iiyz,xz->IX', e_core, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iXxy,Iizx,yz->IX', e_core, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,iXxy,Iizy,xz->IX', e_core, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ia,Iixa,Xx->IX', e_core, t1_ce, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ia,iIxa,Xx->IX', e_core, t1_ce, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ix,Iixy,Xy->IX', e_core, t1_ca, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ix,Iiyx,Xy->IX', e_core, t1_ca, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixab,Iiab,Xx->IX', e_core, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixab,Iiba,Xx->IX', e_core, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += einsum('i,ixay,IiXa,xy->IX', e_core, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixay,Iiya,Xx->IX', e_core, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixay,Iiza,Xyzx->IX', e_core, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixay,iIXa,xy->IX', e_core, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixay,iIya,Xx->IX', e_core, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixay,iIza,Xyzx->IX', e_core, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixya,IiXa,xy->IX', e_core, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixya,Iiya,Xx->IX', e_core, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixya,Iiza,Xyzx->IX', e_core, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixya,iIXa,xy->IX', e_core, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixya,iIya,Xx->IX', e_core, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixya,iIza,Xyxz->IX', e_core, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('i,ixyz,IiXy,xz->IX', e_core, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyz,IiXz,xy->IX', e_core, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyz,Iiwy,Xzwx->IX', e_core, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixyz,Iiwz,Xywx->IX', e_core, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyz,IiyX,xz->IX', e_core, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixyz,Iiyw,Xzwx->IX', e_core, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixyz,Iiyz,Xx->IX', e_core, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixyz,IizX,xy->IX', e_core, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixyz,Iizw,Xyxw->IX', e_core, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyz,Iizy,Xx->IX', e_core, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Xx,Iixa,iyaz,yz->IX', h_aa, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,Iixa,iyza,yz->IX', h_aa, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,Iixy,iz,yz->IX', h_aa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,Iixy,izwu,yzwu->IX', h_aa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,Iixy,izwy,zw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Xx,Iixy,izyw,zw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xx,Iiyx,iz,yz->IX', h_aa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xx,Iiyx,izwu,yzwu->IX', h_aa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xx,Iiyx,izwy,zw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,Iiyx,izyw,zw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xx,Iyax,za,yz->IX', h_aa, t1_caea, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xx,Iyax,zwua,yuwz->IX', h_aa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xx,Iyxa,za,yz->IX', h_aa, t1_caae, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xx,Iyxa,zwua,yuwz->IX', h_aa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,iIxa,iyaz,yz->IX', h_aa, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xx,iIxa,iyza,yz->IX', h_aa, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xx,ixay,Iiza,yz->IX', h_aa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xx,ixay,iIza,yz->IX', h_aa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xx,ixya,Iiza,yz->IX', h_aa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xx,ixya,iIza,yz->IX', h_aa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xx,ixyz,Iiwu,zywu->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xx,ixyz,Iiwy,zw->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xx,ixyz,Iiwz,yw->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xx,ixyz,Iiyw,zw->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xx,ixyz,Iizw,yw->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,xa,Iyaz,yz->IX', h_aa, t1_ae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xx,xa,Iyza,yz->IX', h_aa, t1_ae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,xyab,Izab,yz->IX', h_aa, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xx,xyab,Izba,yz->IX', h_aa, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xx,xyza,Ia,yz->IX', h_aa, t1_aaae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xx,xyza,Iwau,yuzw->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xx,xyza,Iwaz,yw->IX', h_aa, t1_aaae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xx,xyza,Iwua,yuwz->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,xyza,Iwza,yw->IX', h_aa, t1_aaae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,yxza,Ia,yz->IX', h_aa, t1_aaae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,yxza,Iwau,yuzw->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xx,yxza,Iwaz,yw->IX', h_aa, t1_aaae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xx,yxza,Iwua,yuzw->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xx,yxza,Iwza,yw->IX', h_aa, t1_aaae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,IiXx,iz,yz->IX', h_aa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,IiXx,izwu,yzwu->IX', h_aa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,IiXx,izwy,wz->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xy,IiXx,izyw,wz->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,IixX,iz,yz->IX', h_aa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,IixX,izwu,yzwu->IX', h_aa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,IixX,izwy,wz->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,IixX,izyw,wz->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Iixa,iXaz,yz->IX', h_aa, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Iixa,iXza,yz->IX', h_aa, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Iixa,ia,Xy->IX', h_aa, t1_ccae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Iixa,izaw,Xwyz->IX', h_aa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,Iixa,izay,Xz->IX', h_aa, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Iixa,izwa,Xwyz->IX', h_aa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Iixa,izya,Xz->IX', h_aa, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Iixz,iXwu,yzuw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,Iixz,iXwy,zw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Iixz,iXwz,yw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Iixz,iXyw,zw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Iixz,iXzw,yw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Iixz,iw,Xwyz->IX', h_aa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Iixz,iwuv,Xuvyzw->IX', h_aa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Iixz,iwuy,Xuwz->IX', h_aa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Iixz,iwuz,Xuyw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Iixz,iwyu,Xuzw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Iixz,iwyz,Xw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Iixz,iwzu,Xuyw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,Iixz,iwzy,Xw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Iixz,iy,Xz->IX', h_aa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Iixz,iz,Xy->IX', h_aa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Iizx,iXwu,yzwu->IX', h_aa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Iizx,iXwy,zw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Iizx,iXwz,yw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,Iizx,iXyw,zw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Iizx,iXzw,yw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Iizx,iw,Xwzy->IX', h_aa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,Iizx,iwuv,Xuvwyz->IX', h_aa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,Iizx,iwuv,Xuvwzy->IX', h_aa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,Iizx,iwuv,Xuvywz->IX', h_aa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,Iizx,iwuv,Xuvyzw->IX', h_aa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,Iizx,iwuv,Xuvzwy->IX', h_aa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xy,Iizx,iwuv,Xuvzyw->IX', h_aa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Iizx,iwuy,Xuzw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Iizx,iwuz,Xuwy->IX', h_aa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,Iizx,iwyu,Xuzw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,Iizx,iwyz,Xw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Iizx,iwzu,Xuyw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Iizx,iwzy,Xw->IX', h_aa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,Iizx,iy,Xz->IX', h_aa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Iizx,iz,Xy->IX', h_aa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,IxXa,za,yz->IX', h_aa, t1_caae, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,IxXa,zwua,yuwz->IX', h_aa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,IxaX,za,yz->IX', h_aa, t1_caea, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,IxaX,zwua,yuwz->IX', h_aa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Ixab,Xzab,yz->IX', h_aa, t1_caee, t1_aaee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixab,Xzba,yz->IX', h_aa, t1_caee, t1_aaee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixab,zwab,Xyzw->IX', h_aa, t1_caee, t1_aaee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Ixaz,Xa,yz->IX', h_aa, t1_caea, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixaz,Xwua,yuzw->IX', h_aa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixaz,Xwza,yw->IX', h_aa, t1_caea, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Ixaz,wXua,yuzw->IX', h_aa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Ixaz,wXza,yw->IX', h_aa, t1_caea, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixaz,wa,Xywz->IX', h_aa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,Ixaz,wuva,Xyvuwz->IX', h_aa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xy,Ixaz,wuva,Xyvuzw->IX', h_aa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,Ixaz,wuva,Xyvwuz->IX', h_aa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,Ixaz,wuva,Xyvwzu->IX', h_aa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,Ixaz,wuva,Xyvzuw->IX', h_aa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,Ixaz,wuva,Xyvzwu->IX', h_aa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixaz,wuza,Xyuw->IX', h_aa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixza,Xa,yz->IX', h_aa, t1_caae, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixza,Xwua,yuwz->IX', h_aa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Ixza,Xwza,yw->IX', h_aa, t1_caae, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixza,wXua,yuzw->IX', h_aa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixza,wXza,yw->IX', h_aa, t1_caae, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixza,wa,Xyzw->IX', h_aa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixza,wuva,Xyvzuw->IX', h_aa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixza,wuza,Xywu->IX', h_aa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Izax,Xa,yz->IX', h_aa, t1_caea, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izax,Xwua,ywzu->IX', h_aa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Izax,Xwya,zw->IX', h_aa, t1_caea, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Izax,wXua,ywzu->IX', h_aa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,Izax,wXya,zw->IX', h_aa, t1_caea, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izax,wa,Xzwy->IX', h_aa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,Izax,wuva,Xzvuwy->IX', h_aa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xy,Izax,wuva,Xzvuyw->IX', h_aa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,Izax,wuva,Xzvwuy->IX', h_aa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,Izax,wuva,Xzvwyu->IX', h_aa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,Izax,wuva,Xzvyuw->IX', h_aa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,Izax,wuva,Xzvywu->IX', h_aa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Izax,wuya,Xzuw->IX', h_aa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izxa,Xa,yz->IX', h_aa, t1_caae, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izxa,Xwua,ywuz->IX', h_aa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,Izxa,Xwya,zw->IX', h_aa, t1_caae, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izxa,wXua,ywzu->IX', h_aa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Izxa,wXya,zw->IX', h_aa, t1_caae, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izxa,wa,Xzyw->IX', h_aa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izxa,wuva,Xzvyuw->IX', h_aa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Izxa,wuya,Xzwu->IX', h_aa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Xxab,Izab,yz->IX', h_aa, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Xxab,Izba,yz->IX', h_aa, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Xxza,Ia,yz->IX', h_aa, t1_aaae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Xxza,Iwau,yuzw->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Xxza,Iwaz,yw->IX', h_aa, t1_aaae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Xxza,Iwua,yuwz->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Xxza,Iwza,yw->IX', h_aa, t1_aaae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Xzxa,Ia,yz->IX', h_aa, t1_aaae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Xzxa,Iwau,ywzu->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Xzxa,Iwua,ywuz->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,iIxa,iXaz,yz->IX', h_aa, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,iIxa,iXza,yz->IX', h_aa, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,iIxa,ia,Xy->IX', h_aa, t1_ccae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,iIxa,izaw,Xwyz->IX', h_aa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,iIxa,izay,Xz->IX', h_aa, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,iIxa,izwa,Xwzy->IX', h_aa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,iIxa,izya,Xz->IX', h_aa, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,iXax,Iiza,yz->IX', h_aa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,iXax,iIza,yz->IX', h_aa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,iXxa,Iiza,yz->IX', h_aa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,iXxa,iIza,yz->IX', h_aa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,iXxz,Iiwu,yzuw->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,iXxz,Iiwz,yw->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,iXxz,Iizw,yw->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,iXzx,Iiwu,yzwu->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,iXzx,Iiwz,yw->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,iXzx,Iizw,yw->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ix,IiXz,yz->IX', h_aa, t1_ca, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ix,IizX,yz->IX', h_aa, t1_ca, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ix,Iizw,Xyzw->IX', h_aa, t1_ca, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixab,Iiab,Xy->IX', h_aa, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixab,Iiba,Xy->IX', h_aa, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,ixaz,IiXa,yz->IX', h_aa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixaz,Iiwa,Xzwy->IX', h_aa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixaz,Iiza,Xy->IX', h_aa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixaz,iIXa,yz->IX', h_aa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixaz,iIwa,Xzwy->IX', h_aa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixaz,iIza,Xy->IX', h_aa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixza,IiXa,yz->IX', h_aa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixza,Iiwa,Xzwy->IX', h_aa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixza,Iiza,Xy->IX', h_aa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixza,iIXa,yz->IX', h_aa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixza,iIwa,Xzyw->IX', h_aa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixza,iIza,Xy->IX', h_aa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixzw,IiXu,yuwz->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixzw,IiXw,yz->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,ixzw,IiXz,yw->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixzw,IiuX,yuwz->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixzw,Iiuv,Xwzuyv->IX', h_aa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixzw,Iiuw,Xzuy->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixzw,Iiuz,Xwuy->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixzw,IiwX,yz->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixzw,Iiwu,Xzyu->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixzw,Iiwz,Xy->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixzw,IizX,yw->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixzw,Iizu,Xwuy->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixzw,Iizw,Xy->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xy,izax,IiXa,yz->IX', h_aa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izax,Iiwa,Xywz->IX', h_aa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izax,iIXa,yz->IX', h_aa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izax,iIwa,Xywz->IX', h_aa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izwx,IiXu,ywzu->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,izwx,IiXw,yz->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izwx,IiuX,ywzu->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,izwx,Iiuv,Xywuvz->IX', h_aa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xy,izwx,Iiuv,Xywuzv->IX', h_aa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,izwx,Iiuv,Xywvuz->IX', h_aa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,izwx,Iiuv,Xywvzu->IX', h_aa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,izwx,Iiuv,Xywzuv->IX', h_aa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,izwx,Iiuv,Xywzvu->IX', h_aa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izwx,Iiuw,Xyuz->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izwx,IiwX,yz->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izwx,Iiwu,Xyuz->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izxa,IiXa,yz->IX', h_aa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izxa,Iiwa,Xywz->IX', h_aa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izxa,iIXa,yz->IX', h_aa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izxa,iIwa,Xyzw->IX', h_aa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izxw,IiXu,ywuz->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izxw,IiXw,yz->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izxw,IiuX,ywuz->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izxw,Iiuv,Xywuvz->IX', h_aa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izxw,Iiuw,Xyuz->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izxw,IiwX,yz->IX', h_aa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izxw,Iiwu,Xyzu->IX', h_aa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xXza,Ia,yz->IX', h_aa, t1_aaae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xXza,Iwau,yuzw->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xXza,Iwaz,yw->IX', h_aa, t1_aaae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xXza,Iwua,yuzw->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xXza,Iwza,yw->IX', h_aa, t1_aaae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xa,Ia,Xy->IX', h_aa, t1_ae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xa,IzXa,yz->IX', h_aa, t1_ae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xa,IzaX,yz->IX', h_aa, t1_ae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xa,Izaw,Xzyw->IX', h_aa, t1_ae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xa,Izwa,Xzwy->IX', h_aa, t1_ae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzab,Iwab,Xwyz->IX', h_aa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzab,Iwba,Xwzy->IX', h_aa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzwa,Ia,Xwzy->IX', h_aa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xzwa,IuXa,yzwu->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzwa,IuaX,yzwu->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,xzwa,Iuav,Xwuvyz->IX', h_aa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,xzwa,Iuav,Xwuvzy->IX', h_aa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,xzwa,Iuav,Xwuyvz->IX', h_aa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,xzwa,Iuav,Xwuyzv->IX', h_aa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,xzwa,Iuav,Xwuzvy->IX', h_aa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xy,xzwa,Iuav,Xwuzyv->IX', h_aa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzwa,Iuaw,Xuzy->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzwa,Iuva,Xwuvyz->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzwa,Iuwa,Xuyz->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zXxa,Ia,yz->IX', h_aa, t1_aaae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zXxa,Iwau,ywzu->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zXxa,Iwua,ywzu->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxa,Ia,Xywz->IX', h_aa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zwxa,IuXa,yuzw->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxa,IuaX,yuzw->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxa,Iuav,Xyuwzv->IX', h_aa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,zwxa,Iuva,Xyuvwz->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xy,zwxa,Iuva,Xyuvzw->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,zwxa,Iuva,Xyuwvz->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,zwxa,Iuva,Xyuwzv->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,zwxa,Iuva,Xyuzvw->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xy,zwxa,Iuva,Xyuzwv->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwa,Ia,Xwyz->IX', h_aa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,zxwa,IuXa,yzuw->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwa,IuaX,yzuw->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwa,Iuav,Xwuyzv->IX', h_aa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwa,Iuaw,Xuyz->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,zxwa,Iuva,Xwuvyz->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xy,zxwa,Iuva,Xwuvzy->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,zxwa,Iuva,Xwuyvz->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,zxwa,Iuva,Xwuyzv->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,zxwa,Iuva,Xwuzvy->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xy,zxwa,Iuva,Xwuzyv->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwa,Iuwa,Xuzy->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,Iiyz,yz,Xx->IX', t1_ca, v_ccaa, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('ix,Iyzi,zy,Xx->IX', t1_ca, v_caac, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixXy,Iizw,zw,yx->IX', t1_caaa, v_ccaa, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixXy,Izwi,wz,yx->IX', t1_caaa, v_caac, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyX,Iizw,zw,yx->IX', t1_caaa, v_ccaa, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('ixyX,Izwi,wz,yx->IX', t1_caaa, v_caac, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,Iiwu,wu,Xxyz->IX', t1_caaa, v_ccaa, rdm_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('ixyz,Iwui,uw,Xxyz->IX', t1_caaa, v_caac, rdm_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iiwx,iu,ywzu->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iiwx,iuvs,ywuzvs->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iiwx,iuvw,yuzv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iiwx,iuwv,yuzv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iiwx,iw,yz->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,Iiwx,izuv,ywvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,Iiwx,izuw,yu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,Iiwx,izwu,yu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iiwz,iu,ywux->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,Iiwz,iuvs,ywusvx->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,Iiwz,iuvs,ywusxv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,Iiwz,iuvs,ywuvsx->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('Xxyz,Iiwz,iuvs,ywuvxs->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,Iiwz,iuvs,ywuxsv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,Iiwz,iuvs,ywuxvs->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iiwz,iuvw,yuvx->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iiwz,iuwv,yuxv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iiwz,iw,yx->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Xxyz,Iixa,ia,yz->IX', v_aaaa, t1_ccae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Xxyz,Iixa,iwau,ywzu->IX', v_aaaa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iixa,iwua,ywzu->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,Iixa,izaw,yw->IX', v_aaaa, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,Iixa,izwa,yw->IX', v_aaaa, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iixw,iu,ywzu->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iixw,iuvs,ywuzvs->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iixw,iuvw,yuzv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('Xxyz,Iixw,iuwv,yuzv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('Xxyz,Iixw,iw,yz->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,Iixw,izuv,ywvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,Iixw,izuw,yu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Xxyz,Iixw,izwu,yu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iixz,iw,yw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iixz,iwuv,ywuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iiza,ia,yx->IX', v_aaaa, t1_ccae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iiza,iwau,ywxu->IX', v_aaaa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iiza,iwua,ywxu->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iizw,iu,ywxu->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iizw,iuvs,ywuxvs->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iizw,iuvw,yuxv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iizw,iuwv,yuxv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iizw,iw,yx->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iizx,iw,yw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iizx,iwuv,ywuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,Iwax,ua,yuzw->IX', v_aaaa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,Iwax,uvsa,yvuzws->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iwax,uzva,yuwv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iwax,za,yw->IX', v_aaaa, t1_caea, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iwax,zuva,yuvw->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,Iwaz,ua,yuwx->IX', v_aaaa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,Iwaz,uvsa,yvuswx->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,Iwaz,uvsa,yvusxw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,Iwaz,uvsa,yvuwsx->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('Xxyz,Iwaz,uvsa,yvuwxs->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,Iwaz,uvsa,yvuxsw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,Iwaz,uvsa,yvuxws->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,Iwxa,ua,yuzw->IX', v_aaaa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,Iwxa,uvsa,yvuzws->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iwxa,uzva,yuwv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iwxa,za,yw->IX', v_aaaa, t1_caae, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iwxa,zuva,yuvw->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,Iwza,ua,yuxw->IX', v_aaaa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,Iwza,uvsa,yvuxws->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iyab,wuab,xzwu->IX', v_aaaa, t1_caee, t1_aaee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iyaw,ua,xzuw->IX', v_aaaa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,Iyaw,uvsa,xzsuvw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,Iyaw,uvsa,xzsuwv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,Iyaw,uvsa,xzsvuw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('Xxyz,Iyaw,uvsa,xzsvwu->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,Iyaw,uvsa,xzswuv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,Iyaw,uvsa,xzswvu->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iyaw,uvwa,xzvu->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iywa,ua,xzwu->IX', v_aaaa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iywa,uvsa,xzswvu->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iywa,uvwa,xzuv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,iIxa,ia,yz->IX', v_aaaa, t1_ccae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,iIxa,iwau,ywzu->IX', v_aaaa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,iIxa,iwua,ywzu->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iIxa,izaw,yw->IX', v_aaaa, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,iIxa,izwa,yw->IX', v_aaaa, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,iIza,ia,yx->IX', v_aaaa, t1_ccae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,iIza,iwau,ywxu->IX', v_aaaa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,iIza,iwua,ywux->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,iway,Iiua,xzuw->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('Xxyz,iway,Iixa,zw->IX', v_aaaa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('Xxyz,iway,Iiza,xw->IX', v_aaaa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,iway,iIua,xzuw->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,iway,iIxa,zw->IX', v_aaaa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iway,iIza,xw->IX', v_aaaa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,iwuy,Iiuv,xzvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,iwuy,Iiux,zw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iwuy,Iiuz,xw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,iwuy,Iivs,xzusvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,iwuy,Iivs,xzuswv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,iwuy,Iivs,xzuvsw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('Xxyz,iwuy,Iivs,xzuvws->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,iwuy,Iivs,xzuwsv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,iwuy,Iivs,xzuwvs->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,iwuy,Iivu,xzvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iwuy,Iivx,zuwv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iwuy,Iivz,xuvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('Xxyz,iwuy,Iixu,zw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Xxyz,iwuy,Iixv,zuwv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,iwuy,Iixz,uw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Xxyz,iwuy,Iizu,xw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iwuy,Iizv,xuwv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iwuy,Iizx,uw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,iwya,Iiua,xzuw->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,iwya,Iixa,zw->IX', v_aaaa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iwya,Iiza,xw->IX', v_aaaa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,iwya,iIua,xzwu->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iwya,iIxa,zw->IX', v_aaaa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('Xxyz,iwya,iIza,xw->IX', v_aaaa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,iwyu,Iiuv,xzwv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iwyu,Iiux,zw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Xxyz,iwyu,Iiuz,xw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,iwyu,Iivs,xzuvsw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,iwyu,Iivu,xzvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iwyu,Iivx,zuvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,iwyu,Iivz,xuvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,iwyu,Iixu,zw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Xxyz,iwyu,Iixv,zuvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('Xxyz,iwyu,Iixz,uw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iwyu,Iizu,xw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iwyu,Iizv,xuvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,iwyu,Iizx,uw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,ixab,Iiab,yz->IX', v_aaaa, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += einsum('Xxyz,ixab,Iiba,yz->IX', v_aaaa, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,ixaw,Iiua,yuzw->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,ixaw,Iiwa,yz->IX', v_aaaa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,ixaw,Iiza,yw->IX', v_aaaa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,ixaw,iIua,yuzw->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,ixaw,iIwa,yz->IX', v_aaaa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,ixaw,iIza,yw->IX', v_aaaa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,ixwa,Iiua,yuzw->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,ixwa,Iiwa,yz->IX', v_aaaa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,ixwa,Iiza,yw->IX', v_aaaa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,ixwa,iIua,yuzw->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,ixwa,iIwa,yz->IX', v_aaaa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,ixwa,iIza,yw->IX', v_aaaa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,ixwu,Iiuv,yvzw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,ixwu,Iiuw,yz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,ixwu,Iiuz,yw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,ixwu,Iivs,yvsuwz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,ixwu,Iivs,yvsuzw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,ixwu,Iivs,yvswuz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,ixwu,Iivs,yvswzu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('Xxyz,ixwu,Iivs,yvszuw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,ixwu,Iivs,yvszwu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,ixwu,Iivu,yvzw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,ixwu,Iivw,yvzu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,ixwu,Iivz,yvwu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,ixwu,Iiwu,yz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,ixwu,Iiwv,yvzu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,ixwu,Iiwz,yu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,ixwu,Iizu,yw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,ixwu,Iizv,yvuw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,ixwu,Iizw,yu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,iy,Iiwu,xzwu->IX', v_aaaa, t1_ca, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iy,Iiwx,zw->IX', v_aaaa, t1_ca, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Xxyz,iy,Iiwz,xw->IX', v_aaaa, t1_ca, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Xxyz,iy,Iixw,zw->IX', v_aaaa, t1_ca, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iy,Iizw,xw->IX', v_aaaa, t1_ca, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,izab,Iiab,yx->IX', v_aaaa, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,izab,Iiba,yx->IX', v_aaaa, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,izaw,Iiua,yuwx->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,izaw,Iiwa,yx->IX', v_aaaa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,izaw,iIua,yuwx->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,izaw,iIwa,yx->IX', v_aaaa, t1_caea, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,izwa,Iiua,yuwx->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,izwa,Iiwa,yx->IX', v_aaaa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,izwa,iIua,yuxw->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,izwa,iIwa,yx->IX', v_aaaa, t1_caae, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,izwu,Iiuv,yvxw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,izwu,Iiuw,yx->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,izwu,Iivs,yvsuxw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,izwu,Iivu,yvwx->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,izwu,Iivw,yvux->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,izwu,Iiwu,yx->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,izwu,Iiwv,yvux->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,wuya,Ia,xzuw->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,wuya,Ivas,xzvuws->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,wuya,Ivax,zvwu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,wuya,Ivaz,xvuw->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,wuya,Ivsa,xzvsuw->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('Xxyz,wuya,Ivsa,xzvswu->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,wuya,Ivsa,xzvusw->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,wuya,Ivsa,xzvuws->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,wuya,Ivsa,xzvwsu->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('Xxyz,wuya,Ivsa,xzvwus->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('Xxyz,wuya,Ivxa,zvwu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,wuya,Ivza,xvwu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,wxua,Ia,ywzu->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,wxua,Ivas,ywszuv->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,wxua,Ivau,ywzv->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,wxua,Ivaz,ywvu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,wxua,Ivsa,ywszuv->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,wxua,Ivua,ywzv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,wxua,Ivza,ywvu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,wzua,Ia,ywxu->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,wzua,Ivas,ywsxuv->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,wzua,Ivau,ywxv->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,wzua,Ivsa,ywsuvx->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,wzua,Ivsa,ywsuxv->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('Xxyz,wzua,Ivsa,ywsvux->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,wzua,Ivsa,ywsvxu->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,wzua,Ivsa,ywsxuv->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,wzua,Ivsa,ywsxvu->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,wzua,Ivua,ywvx->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,xa,Ia,yz->IX', v_aaaa, t1_ae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,xa,Iwau,yuzw->IX', v_aaaa, t1_ae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,xa,Iwaz,yw->IX', v_aaaa, t1_ae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xa,Iwua,yuzw->IX', v_aaaa, t1_ae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xa,Iwza,yw->IX', v_aaaa, t1_ae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,xwab,Iuab,ywzu->IX', v_aaaa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xwab,Iuba,ywzu->IX', v_aaaa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xwua,Ia,ywzu->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xwua,Ivas,ywszuv->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xwua,Ivau,ywzv->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xwua,Ivaz,ywvu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,xwua,Ivsa,ywsuvz->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,xwua,Ivsa,ywsuzv->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,xwua,Ivsa,ywsvuz->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,xwua,Ivsa,ywsvzu->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,xwua,Ivsa,ywszuv->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('Xxyz,xwua,Ivsa,ywszvu->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,xwua,Ivua,ywzv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xwua,Ivza,ywuv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,xzab,Iwab,yw->IX', v_aaaa, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xzab,Iwba,yw->IX', v_aaaa, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xzwa,Ia,yw->IX', v_aaaa, t1_aaae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xzwa,Iuav,yvwu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xzwa,Iuaw,yu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xzwa,Iuva,yvuw->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,xzwa,Iuwa,yu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,za,Ia,yx->IX', v_aaaa, t1_ae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,za,Iwau,yuxw->IX', v_aaaa, t1_ae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,za,Iwua,yuwx->IX', v_aaaa, t1_ae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zwab,Iuab,ywxu->IX', v_aaaa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zwab,Iuba,ywux->IX', v_aaaa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zwua,Ia,ywux->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,zwua,Ivas,ywsuvx->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('Xxyz,zwua,Ivas,ywsuxv->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,zwua,Ivas,ywsvux->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,zwua,Ivas,ywsvxu->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,zwua,Ivas,ywsxuv->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Xxyz,zwua,Ivas,ywsxvu->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zwua,Ivau,ywvx->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zwua,Ivsa,ywsuvx->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zwua,Ivua,ywxv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,zxwa,Ia,yw->IX', v_aaaa, t1_aaae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,zxwa,Iuav,yvwu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,zxwa,Iuaw,yu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zxwa,Iuva,yvwu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zxwa,Iuwa,yu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,IiXx,iu,zuwy->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IiXx,iuvs,zvsuwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IiXx,iuvs,zvsuyw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IiXx,iuvs,zvswuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,IiXx,iuvs,zvswyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IiXx,iuvs,zvsyuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IiXx,iuvs,zvsywu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,IiXx,iuvw,zvuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,IiXx,iuvy,zvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,IiXx,iuwv,zvyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,IiXx,iuwy,zu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xyzw,IiXx,iuyv,zvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('xyzw,IiXx,iuyw,zu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,IiXx,iw,zy->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xyzw,IiXx,iy,zw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,IiXx,izuv,ywuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iiux,iXuv,zvwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,iXuw,zy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,Iiux,iXuy,zw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,iXvs,zsvuwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,iXvs,zsvuyw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iiux,iXvs,zsvwuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,iXvs,zsvwyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,iXvs,zsvyuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,iXvs,zsvywu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iiux,iXvu,zvwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,iXvw,zvuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,iXvy,zvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iiux,iXwu,zy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,iXwv,zvyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,iXwy,zu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xyzw,Iiux,iXyu,zw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,Iiux,iXyv,zvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iiux,iXyw,zu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iiux,iu,Xzyw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iiux,iv,Xzvuwy->IX', v_aaaa, t1_ccaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,iv,Xzvuyw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,iv,Xzvwuy->IX', v_aaaa, t1_ccaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,iv,Xzvwyu->IX', v_aaaa, t1_ccaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,iv,Xzvyuw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,iv,Xzvywu->IX', v_aaaa, t1_ccaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/240 * einsum('xyzw,Iiux,ivst,Xzstuvwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/240 * einsum('xyzw,Iiux,ivst,Xzstuvyw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,Iiux,ivst,Xzstuwvy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/60 * einsum('xyzw,Iiux,ivst,Xzstuwyv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,Iiux,ivst,Xzstuyvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 2/15 * einsum('xyzw,Iiux,ivst,Xzstuywv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,Iiux,ivst,Xzstvwuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,Iiux,ivst,Xzstvwyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,Iiux,ivst,Xzstvyuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/120 * einsum('xyzw,Iiux,ivst,Xzstvywu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/30 * einsum('xyzw,Iiux,ivst,Xzstwuvy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/120 * einsum('xyzw,Iiux,ivst,Xzstwuyv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/240 * einsum('xyzw,Iiux,ivst,Xzstwvuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/240 * einsum('xyzw,Iiux,ivst,Xzstwvyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,ivst,Xzstwyuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,Iiux,ivst,Xzstwyvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 3/40 * einsum('xyzw,Iiux,ivst,Xzstyuvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/20 * einsum('xyzw,Iiux,ivst,Xzstyuwv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 17/240 * einsum('xyzw,Iiux,ivst,Xzstyvuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 17/240 * einsum('xyzw,Iiux,ivst,Xzstyvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/120 * einsum('xyzw,Iiux,ivst,Xzstywuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/15 * einsum('xyzw,Iiux,ivst,Xzstywvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iiux,ivsu,Xzsvwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,ivsu,Xzsvyw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,ivsu,Xzswvy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,ivsu,Xzswyv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,ivsu,Xzsyvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iiux,ivsu,Xzsywv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,ivsw,Xzsuvy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iiux,ivsy,Xzsuvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,Iiux,ivsy,Xzsuwv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iiux,ivsy,Xzsvuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iiux,ivsy,Xzsvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iiux,ivsy,Xzswuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iiux,ivsy,Xzswvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iiux,ivus,Xzsywv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,ivuw,Xzyv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,ivuy,Xzvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iiux,ivws,Xzsuvy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,Iiux,ivws,Xzsuyv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iiux,ivws,Xzsvuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iiux,ivws,Xzsvyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iiux,ivws,Xzsyuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iiux,ivws,Xzsyvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,ivwu,Xzvy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,ivwy,Xzuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,Iiux,ivys,Xzsuvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 2/3 * einsum('xyzw,Iiux,ivys,Xzsuwv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,Iiux,ivys,Xzsvuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,Iiux,ivys,Xzsvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,Iiux,ivys,Xzswuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,Iiux,ivys,Xzswvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iiux,ivyu,Xzvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iiux,ivyw,Xzuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,iw,Xzuy->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iiux,iy,Xzuw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iiux,izuv,Xvyw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iiux,izvs,Xsvuwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iiux,izvu,Xvwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,IixX,iu,zuwy->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IixX,iuvs,zvsuwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IixX,iuvs,zvsuyw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IixX,iuvs,zvswuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IixX,iuvs,zvswyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IixX,iuvs,zvsyuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IixX,iuvs,zvsywu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,IixX,iuvw,zvuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,IixX,iuvy,zvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,IixX,iuwv,zvyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,IixX,iuwy,zu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,IixX,iuyv,zvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,IixX,iuyw,zu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,IixX,iw,zy->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,IixX,iy,zw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,IixX,izuv,ywuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iixa,iXau,zuwy->IX', v_aaaa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixa,iXaw,zy->IX', v_aaaa, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xyzw,Iixa,iXay,zw->IX', v_aaaa, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iixa,iXua,zuwy->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixa,iXwa,zy->IX', v_aaaa, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,Iixa,iXya,zw->IX', v_aaaa, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iixa,ia,Xzyw->IX', v_aaaa, t1_ccae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iixa,iuav,Xzvywu->IX', v_aaaa, t1_ccae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixa,iuaw,Xzyu->IX', v_aaaa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixa,iuay,Xzuw->IX', v_aaaa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iixa,iuva,Xzvywu->IX', v_aaaa, t1_ccae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixa,iuwa,Xzyu->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixa,iuya,Xzuw->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixa,izau,Xuyw->IX', v_aaaa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iixa,izua,Xuyw->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iixu,iXuv,zvwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixu,iXuw,zy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xyzw,Iixu,iXuy,zw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iixu,iXvs,zsvwyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iixu,iXvu,zvwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,iXvw,zvyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixu,iXvy,zvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,iXwu,zy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,iXwv,zvuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixu,iXwy,zu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,Iixu,iXyu,zw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,iXyv,zvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,iXyw,zu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iixu,iu,Xzyw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iixu,iv,Xzvywu->IX', v_aaaa, t1_ccaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/60 * einsum('xyzw,Iixu,ivst,Xzstuvwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/60 * einsum('xyzw,Iixu,ivst,Xzstuvyw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,Iixu,ivst,Xzstuwvy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,Iixu,ivst,Xzstuwyv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,Iixu,ivst,Xzstuyvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,Iixu,ivst,Xzstuywv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 13/480 * einsum('xyzw,Iixu,ivst,Xzstvwuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 11/480 * einsum('xyzw,Iixu,ivst,Xzstvwyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,Iixu,ivst,Xzstvyuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,Iixu,ivst,Xzstvywu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,Iixu,ivst,Xzstwuvy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,Iixu,ivst,Xzstwuyv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/160 * einsum('xyzw,Iixu,ivst,Xzstwvuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/160 * einsum('xyzw,Iixu,ivst,Xzstwvyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,Iixu,ivst,Xzstwyuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,Iixu,ivst,Xzstwyvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,Iixu,ivst,Xzstyuvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,Iixu,ivst,Xzstyuwv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/160 * einsum('xyzw,Iixu,ivst,Xzstyvuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/160 * einsum('xyzw,Iixu,ivst,Xzstyvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 59/240 * einsum('xyzw,Iixu,ivst,Xzstywuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/240 * einsum('xyzw,Iixu,ivst,Xzstywvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iixu,ivsu,Xzsywv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iixu,ivsw,Xzsuvy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iixu,ivsw,Xzsuyv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iixu,ivsw,Xzsvuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iixu,ivsw,Xzsvyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iixu,ivsw,Xzsyuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,Iixu,ivsw,Xzsyvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,ivsy,Xzsvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iixu,ivus,Xzsywv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixu,ivuw,Xzyv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixu,ivuy,Xzvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,ivws,Xzsyuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,ivwu,Xzyv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,ivwy,Xzvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iixu,ivys,Xzsuvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,Iixu,ivys,Xzsuwv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iixu,ivys,Xzsvuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iixu,ivys,Xzsvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iixu,ivys,Xzswuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iixu,ivys,Xzswvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,ivyu,Xzvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,ivyw,Xzuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,iw,Xzyu->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,iy,Xzuw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,izuv,Xvyw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iixu,izvs,Xsvuwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iixu,izvs,Xsvuyw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iixu,izvs,Xsvwuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iixu,izvs,Xsvwyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iixu,izvs,Xsvyuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iixu,izvs,Xsvywu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iixu,izvu,Xvyw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iixz,iXuv,ywvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixz,iXuw,yu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,Iixz,iXuy,wu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,Iixz,iXwu,yu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixz,iXyu,wu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iixz,iu,Xuyw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iixz,iuvs,Xvsuwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iixz,iuvs,Xvsuyw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iixz,iuvs,Xvswuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iixz,iuvs,Xvswyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iixz,iuvs,Xvsyuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iixz,iuvs,Xvsywu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixz,iuvw,Xvyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixz,iuvy,Xvuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixz,iuwv,Xvyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixz,iuwy,Xu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixz,iuyv,Xvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixz,iuyw,Xu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,Iixz,iw,Xy->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixz,iy,Xw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iuax,Xa,zuwy->IX', v_aaaa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuax,Xvsa,zusvwy->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuax,Xvsa,zusvyw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuax,Xvsa,zuswvy->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iuax,Xvsa,zuswyv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuax,Xvsa,zusyvw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuax,Xvsa,zusywv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuax,Xvwa,zuvy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuax,Xvya,zuwv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuax,Xzva,ywuv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iuax,vXsa,zusvwy->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iuax,vXsa,zusvyw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iuax,vXsa,zuswvy->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,Iuax,vXsa,zuswyv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iuax,vXsa,zusyvw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iuax,vXsa,zusywv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('xyzw,Iuax,vXwa,zuvy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,Iuax,vXya,zuwv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iuax,va,Xzuvwy->IX', v_aaaa, t1_caea, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuax,va,Xzuvyw->IX', v_aaaa, t1_caea, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuax,va,Xzuwvy->IX', v_aaaa, t1_caea, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuax,va,Xzuwyv->IX', v_aaaa, t1_caea, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuax,va,Xzuyvw->IX', v_aaaa, t1_caea, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuax,va,Xzuywv->IX', v_aaaa, t1_caea, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/96 * einsum('xyzw,Iuax,vsta,Xzutsvwy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/96 * einsum('xyzw,Iuax,vsta,Xzutsvyw->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,Iuax,vsta,Xzutswvy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuax,vsta,Xzutswyv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/96 * einsum('xyzw,Iuax,vsta,Xzutsyvw->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuax,vsta,Xzutsywv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/96 * einsum('xyzw,Iuax,vsta,Xzutvswy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/96 * einsum('xyzw,Iuax,vsta,Xzutvsyw->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,Iuax,vsta,Xzutvwsy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuax,vsta,Xzutvwys->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/96 * einsum('xyzw,Iuax,vsta,Xzutvysw->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Iuax,vsta,Xzutvyws->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,Iuax,vsta,Xzutwsvy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 5/32 * einsum('xyzw,Iuax,vsta,Xzutwsyv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,Iuax,vsta,Xzutwvsy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,Iuax,vsta,Xzutwvys->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 47/480 * einsum('xyzw,Iuax,vsta,Xzutwysv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/160 * einsum('xyzw,Iuax,vsta,Xzutwyvs->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/48 * einsum('xyzw,Iuax,vsta,Xzutysvw->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,Iuax,vsta,Xzutyswv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/48 * einsum('xyzw,Iuax,vsta,Xzutyvsw->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/32 * einsum('xyzw,Iuax,vsta,Xzutyvws->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/96 * einsum('xyzw,Iuax,vsta,Xzutywsv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 5/96 * einsum('xyzw,Iuax,vsta,Xzutywvs->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuax,vswa,Xzusvy->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuax,vsya,Xzusvw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,Iuax,vsya,Xzuswv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuax,vsya,Xzuvsw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuax,vsya,Xzuvws->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuax,vsya,Xzuwsv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuax,vsya,Xzuwvs->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuax,vzsa,Xusvwy->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuax,vzsa,Xusvyw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuax,vzsa,Xuswvy->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuax,vzsa,Xuswyv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuax,vzsa,Xusyvw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuax,vzsa,Xusywv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuax,zXva,ywuv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuax,za,Xuwy->IX', v_aaaa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuax,zvsa,Xusvyw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuxa,Xa,zuwy->IX', v_aaaa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuxa,Xvsa,zuswvy->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuxa,Xvwa,zuyv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,Iuxa,Xvya,zuwv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxa,Xzva,ywvu->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuxa,vXsa,zusvwy->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuxa,vXsa,zusvyw->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuxa,vXsa,zuswvy->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Iuxa,vXsa,zuswyv->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuxa,vXsa,zusyvw->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Iuxa,vXsa,zusywv->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuxa,vXwa,zuvy->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuxa,vXya,zuwv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuxa,va,Xzuywv->IX', v_aaaa, t1_caae, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,Iuxa,vsta,Xzutsvwy->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,Iuxa,vsta,Xzutsvyw->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/160 * einsum('xyzw,Iuxa,vsta,Xzutswvy->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,Iuxa,vsta,Xzutswyv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/160 * einsum('xyzw,Iuxa,vsta,Xzutsyvw->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,Iuxa,vsta,Xzutsywv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,Iuxa,vsta,Xzutvswy->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,Iuxa,vsta,Xzutvsyw->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/32 * einsum('xyzw,Iuxa,vsta,Xzutvwsy->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,Iuxa,vsta,Xzutvwys->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/160 * einsum('xyzw,Iuxa,vsta,Xzutvysw->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,Iuxa,vsta,Xzutvyws->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/480 * einsum('xyzw,Iuxa,vsta,Xzutwsvy->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/160 * einsum('xyzw,Iuxa,vsta,Xzutwsyv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/480 * einsum('xyzw,Iuxa,vsta,Xzutwvsy->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/160 * einsum('xyzw,Iuxa,vsta,Xzutwvys->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,Iuxa,vsta,Xzutwysv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/80 * einsum('xyzw,Iuxa,vsta,Xzutwyvs->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/480 * einsum('xyzw,Iuxa,vsta,Xzutysvw->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/160 * einsum('xyzw,Iuxa,vsta,Xzutyswv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/480 * einsum('xyzw,Iuxa,vsta,Xzutyvsw->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/160 * einsum('xyzw,Iuxa,vsta,Xzutyvws->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 59/240 * einsum('xyzw,Iuxa,vsta,Xzutywsv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/240 * einsum('xyzw,Iuxa,vsta,Xzutywvs->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuxa,vswa,Xzusvy->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuxa,vswa,Xzusyv->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuxa,vswa,Xzuvsy->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuxa,vswa,Xzuvys->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuxa,vswa,Xzuysv->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,Iuxa,vswa,Xzuyvs->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuxa,vsya,Xzuvws->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxa,vzsa,Xusywv->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxa,zXva,ywuv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxa,za,Xuyw->IX', v_aaaa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuxa,zvsa,Xusvwy->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuxa,zvsa,Xusvyw->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuxa,zvsa,Xuswvy->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuxa,zvsa,Xuswyv->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Iuxa,zvsa,Xusyvw->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuxa,zvsa,Xusywv->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,IxXa,ua,zuwy->IX', v_aaaa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IxXa,uvsa,zvuswy->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IxXa,uvsa,zvusyw->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IxXa,uvsa,zvuwsy->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,IxXa,uvsa,zvuwys->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IxXa,uvsa,zvuysw->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IxXa,uvsa,zvuyws->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,IxXa,uvza,ywvu->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,IxaX,ua,zuwy->IX', v_aaaa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxaX,uvsa,zvuswy->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxaX,uvsa,zvusyw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxaX,uvsa,zvuwsy->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IxaX,uvsa,zvuwys->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxaX,uvsa,zvuysw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxaX,uvsa,zvuyws->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,IxaX,uvza,ywvu->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Ixab,Xuab,zuwy->IX', v_aaaa, t1_caee, t1_aaee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixab,Xuba,zuwy->IX', v_aaaa, t1_caee, t1_aaee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixab,uvab,Xywuvz->IX', v_aaaa, t1_caee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Ixau,Xa,zuwy->IX', v_aaaa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,Xvsa,zuvswy->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,Xvsa,zuvsyw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,Xvsa,zuvwsy->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Ixau,Xvsa,zuvwys->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,Xvsa,zuvysw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,Xvsa,zuvyws->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixau,Xvua,zvwy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixau,Xvza,ywuv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Ixau,vXsa,zuvswy->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Ixau,vXsa,zuvsyw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Ixau,vXsa,zuvwsy->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,Ixau,vXsa,zuvwys->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Ixau,vXsa,zuvysw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Ixau,vXsa,zuvyws->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Ixau,vXua,zvwy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Ixau,vXza,ywuv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,va,Xywuvz->IX', v_aaaa, t1_caea, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,va,Xywuzv->IX', v_aaaa, t1_caea, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Ixau,va,Xywvuz->IX', v_aaaa, t1_caea, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,va,Xywvzu->IX', v_aaaa, t1_caea, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,va,Xywzuv->IX', v_aaaa, t1_caea, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,va,Xywzvu->IX', v_aaaa, t1_caea, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixau,vsta,Xywtsuzv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixau,vsta,Xywtsvzu->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,vsta,Xywtszuv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Ixau,vsta,Xywtszvu->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,Ixau,vsta,Xywtusvz->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/48 * einsum('xyzw,Ixau,vsta,Xywtuszv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,Ixau,vsta,Xywtuvsz->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,Ixau,vsta,Xywtuvzs->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,Ixau,vsta,Xywtuzsv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,Ixau,vsta,Xywtuzvs->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixau,vsta,Xywtvszu->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixau,vsta,Xywtvuzs->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Ixau,vsta,Xywtvzsu->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Ixau,vsta,Xywtvzus->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/240 * einsum('xyzw,Ixau,vsta,Xywtzsuv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,Ixau,vsta,Xywtzsvu->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,Ixau,vsta,Xywtzusv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,Ixau,vsta,Xywtzuvs->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,Ixau,vsta,Xywtzvsu->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 11/240 * einsum('xyzw,Ixau,vsta,Xywtzvus->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixau,vsua,Xywsvz->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Ixau,vsza,Xywsuv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,vsza,Xywsvu->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,vsza,Xywusv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,vsza,Xywuvs->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,vsza,Xywvsu->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixau,vsza,Xywvus->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,Xa,zuwy->IX', v_aaaa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,Xvsa,zuvwsy->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Ixua,Xvua,zvwy->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,Xvza,ywvu->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixua,vXsa,zuvswy->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixua,vXsa,zuvsyw->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixua,vXsa,zuvwsy->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Ixua,vXsa,zuvwys->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixua,vXsa,zuvysw->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixua,vXsa,zuvyws->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,vXua,zvwy->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,vXza,ywuv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,va,Xywuvz->IX', v_aaaa, t1_caae, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 += 7/240 * einsum('xyzw,Ixua,vsta,Xywtsuzv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/240 * einsum('xyzw,Ixua,vsta,Xywtsvzu->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/240 * einsum('xyzw,Ixua,vsta,Xywtszuv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/240 * einsum('xyzw,Ixua,vsta,Xywtszvu->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/240 * einsum('xyzw,Ixua,vsta,Xywtusvz->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 13/60 * einsum('xyzw,Ixua,vsta,Xywtuszv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/240 * einsum('xyzw,Ixua,vsta,Xywtuvsz->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,Ixua,vsta,Xywtuvzs->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/30 * einsum('xyzw,Ixua,vsta,Xywtuzsv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/120 * einsum('xyzw,Ixua,vsta,Xywtuzvs->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/240 * einsum('xyzw,Ixua,vsta,Xywtvszu->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,Ixua,vsta,Xywtvuzs->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/240 * einsum('xyzw,Ixua,vsta,Xywtvzsu->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,Ixua,vsta,Xywtvzus->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/240 * einsum('xyzw,Ixua,vsta,Xywtzsuv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/120 * einsum('xyzw,Ixua,vsta,Xywtzsvu->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,Ixua,vsta,Xywtzusv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/240 * einsum('xyzw,Ixua,vsta,Xywtzuvs->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/120 * einsum('xyzw,Ixua,vsta,Xywtzvsu->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 17/240 * einsum('xyzw,Ixua,vsta,Xywtzvus->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixua,vsua,Xywsvz->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixua,vsua,Xywszv->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Ixua,vsua,Xywvsz->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixua,vsua,Xywvzs->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixua,vsua,Xywzsv->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Ixua,vsua,Xywzvs->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,vsza,Xywusv->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Xuxa,Ia,zuwy->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Xuxa,Ivas,zusvwy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Xuxa,Ivas,zusvyw->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Xuxa,Ivas,zuswvy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,Xuxa,Ivas,zuswyv->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Xuxa,Ivas,zusyvw->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Xuxa,Ivas,zusywv->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Xuxa,Ivsa,zuswvy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Xxab,Iuab,zuwy->IX', v_aaaa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Xxab,Iuba,zuwy->IX', v_aaaa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Xxua,Ia,zuwy->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Xxua,Ivas,zuvswy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Xxua,Ivas,zuvsyw->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Xxua,Ivas,zuvwsy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,Xxua,Ivas,zuvwys->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Xxua,Ivas,zuvysw->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Xxua,Ivas,zuvyws->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Xxua,Ivau,zvwy->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Xxua,Ivsa,zuvwsy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Xxua,Ivua,zvwy->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iIxa,iXau,zuwy->IX', v_aaaa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,iIxa,iXaw,zy->IX', v_aaaa, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,iIxa,iXay,zw->IX', v_aaaa, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iIxa,iXua,zuwy->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,iIxa,iXwa,zy->IX', v_aaaa, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xyzw,iIxa,iXya,zw->IX', v_aaaa, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iIxa,ia,Xzyw->IX', v_aaaa, t1_ccae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iIxa,iuav,Xzvywu->IX', v_aaaa, t1_ccae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,iIxa,iuaw,Xzyu->IX', v_aaaa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,iIxa,iuay,Xzuw->IX', v_aaaa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iIxa,iuva,Xzvuwy->IX', v_aaaa, t1_ccae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iIxa,iuva,Xzvuyw->IX', v_aaaa, t1_ccae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iIxa,iuva,Xzvwuy->IX', v_aaaa, t1_ccae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iIxa,iuva,Xzvwyu->IX', v_aaaa, t1_ccae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iIxa,iuva,Xzvyuw->IX', v_aaaa, t1_ccae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iIxa,iuva,Xzvywu->IX', v_aaaa, t1_ccae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,iIxa,iuwa,Xzuy->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,iIxa,iuya,Xzuw->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iIxa,izau,Xuyw->IX', v_aaaa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iIxa,izua,Xuwy->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iXax,Iiua,zuwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iXax,iIua,zuwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iXux,Iiuv,zvwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iXux,Iivs,zvsuwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iXux,Iivs,zvsuyw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iXux,Iivs,zvswuy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iXux,Iivs,zvswyu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iXux,Iivs,zvsyuw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iXux,Iivs,zvsywu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iXux,Iivu,zvwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iXxa,Iiua,zuwy->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iXxa,iIua,zuwy->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iXxu,Iiuv,zvwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iXxu,Iivs,zvswuy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iXxu,Iivu,zvwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iXzx,Iiuv,ywuv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,iuax,IiXa,zuwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuax,Iiva,Xywuvz->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuax,Iiva,Xywuzv->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,iuax,Iiva,Xywvuz->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuax,Iiva,Xywvzu->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuax,Iiva,Xywzuv->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuax,Iiva,Xywzvu->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuax,iIXa,zuwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuax,iIva,Xywuvz->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuax,iIva,Xywuzv->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuax,iIva,Xywvuz->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuax,iIva,Xywvzu->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuax,iIva,Xywzuv->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuax,iIva,Xywzvu->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,IiXs,zusvwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,IiXs,zusvyw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,IiXs,zuswvy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,iuvx,IiXs,zuswyv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,IiXs,zusyvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,IiXs,zusywv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('xyzw,iuvx,IiXv,zuwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IisX,zusvwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IisX,zusvyw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IisX,zuswvy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,IisX,zuswyv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IisX,zusyvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IisX,zusywv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,iuvx,Iist,Xywvstzu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,iuvx,Iist,Xywvsuzt->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,Iist,Xywvsztu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,Iist,Xywvszut->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,iuvx,Iist,Xywvtszu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,iuvx,Iist,Xywvtuzs->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,Iist,Xywvtzsu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,Iist,Xywvtzus->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,iuvx,Iist,Xywvustz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/48 * einsum('xyzw,iuvx,Iist,Xywvuszt->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/48 * einsum('xyzw,iuvx,Iist,Xywvutsz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,iuvx,Iist,Xywvutzs->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,iuvx,Iist,Xywvuzst->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,iuvx,Iist,Xywvuzts->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,iuvx,Iist,Xywvzstu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/240 * einsum('xyzw,iuvx,Iist,Xywvzsut->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,iuvx,Iist,Xywvztsu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 11/240 * einsum('xyzw,iuvx,Iist,Xywvztus->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,iuvx,Iist,Xywvzust->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/16 * einsum('xyzw,iuvx,Iist,Xywvzuts->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,iuvx,Iisv,Xywsuz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,Iisv,Xywszu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,Iisv,Xywusz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,Iisv,Xywuzs->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,Iisv,Xywzsu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,Iisv,Xywzus->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuvx,IivX,zuwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuvx,Iivs,Xywsuz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,Iivs,Xywszu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,Iivs,Xywusz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,Iivs,Xywuzs->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,Iivs,Xywzsu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,Iivs,Xywzus->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxa,IiXa,zuwy->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxa,Iiva,Xywuvz->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxa,Iiva,Xywuzv->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuxa,Iiva,Xywvuz->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxa,Iiva,Xywvzu->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxa,Iiva,Xywzuv->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxa,Iiva,Xywzvu->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxa,iIXa,zuwy->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxa,iIva,Xywuvz->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxv,IiXs,zuswvy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxv,IiXv,zuwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,IisX,zuswvy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 11/80 * einsum('xyzw,iuxv,Iist,Xywvstzu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,iuxv,Iist,Xywvsuzt->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,iuxv,Iist,Xywvsztu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/80 * einsum('xyzw,iuxv,Iist,Xywvtszu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/40 * einsum('xyzw,iuxv,Iist,Xywvtuzs->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,iuxv,Iist,Xywvtzsu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,iuxv,Iist,Xywvzstu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/60 * einsum('xyzw,iuxv,Iist,Xywvzsut->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/16 * einsum('xyzw,iuxv,Iist,Xywvztsu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/60 * einsum('xyzw,iuxv,Iist,Xywvztus->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuxv,Iisv,Xywsuz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxv,Iisv,Xywszu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxv,Iisv,Xywusz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxv,Iisv,Xywuzs->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxv,Iisv,Xywzsu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxv,Iisv,Xywzus->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,IivX,zuwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,Iivs,Xywusz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxz,IiXv,ywvu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxz,IivX,ywvu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxz,Iivs,Xywsuv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxz,Iivs,Xywsvu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxz,Iivs,Xywusv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxz,Iivs,Xywuvs->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,iuxz,Iivs,Xywvsu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxz,Iivs,Xywvus->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ix,IiXu,zuwy->IX', v_aaaa, t1_ca, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ix,IiuX,zuwy->IX', v_aaaa, t1_ca, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ix,Iiuv,Xywuvz->IX', v_aaaa, t1_ca, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixab,Iiab,Xzyw->IX', v_aaaa, t1_caee, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixab,Iiba,Xzyw->IX', v_aaaa, t1_caee, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,ixau,IiXa,zuwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixau,Iiua,Xzyw->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,ixau,Iiva,Xzuvwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ixau,Iiva,Xzuvyw->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ixau,Iiva,Xzuwvy->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ixau,Iiva,Xzuwyv->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ixau,Iiva,Xzuyvw->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ixau,Iiva,Xzuywv->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixau,iIXa,zuwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixau,iIua,Xzyw->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ixau,iIva,Xzuvwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixau,iIva,Xzuvyw->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixau,iIva,Xzuwvy->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixau,iIva,Xzuwyv->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixau,iIva,Xzuyvw->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixau,iIva,Xzuywv->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixua,IiXa,zuwy->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixua,Iiua,Xzyw->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ixua,Iiva,Xzuvwy->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixua,Iiva,Xzuvyw->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixua,Iiva,Xzuwvy->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixua,Iiva,Xzuwyv->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixua,Iiva,Xzuyvw->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixua,Iiva,Xzuywv->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixua,iIXa,zuwy->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixua,iIua,Xzyw->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixua,iIva,Xzuywv->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixuv,IiXs,zvuwys->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,ixuv,IiXu,zvwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixuv,IiXv,zuwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,IisX,zvuwys->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,ixuv,Iist,Xzvustwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,ixuv,Iist,Xzvustyw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,ixuv,Iist,Xzvuswty->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 43/480 * einsum('xyzw,ixuv,Iist,Xzvuswyt->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/80 * einsum('xyzw,ixuv,Iist,Xzvusytw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 47/480 * einsum('xyzw,ixuv,Iist,Xzvusywt->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/160 * einsum('xyzw,ixuv,Iist,Xzvutswy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/160 * einsum('xyzw,ixuv,Iist,Xzvutsyw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,ixuv,Iist,Xzvutwsy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 17/480 * einsum('xyzw,ixuv,Iist,Xzvutwys->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 3/80 * einsum('xyzw,ixuv,Iist,Xzvutysw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 13/480 * einsum('xyzw,ixuv,Iist,Xzvutyws->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/160 * einsum('xyzw,ixuv,Iist,Xzvuwsty->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 11/120 * einsum('xyzw,ixuv,Iist,Xzvuwsyt->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 19/480 * einsum('xyzw,ixuv,Iist,Xzvuwtsy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/240 * einsum('xyzw,ixuv,Iist,Xzvuwtys->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/160 * einsum('xyzw,ixuv,Iist,Xzvuwyst->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 23/480 * einsum('xyzw,ixuv,Iist,Xzvuwyts->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/160 * einsum('xyzw,ixuv,Iist,Xzvuystw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 11/120 * einsum('xyzw,ixuv,Iist,Xzvuyswt->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 19/480 * einsum('xyzw,ixuv,Iist,Xzvuytsw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/240 * einsum('xyzw,ixuv,Iist,Xzvuytws->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 59/480 * einsum('xyzw,ixuv,Iist,Xzvuywst->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/480 * einsum('xyzw,ixuv,Iist,Xzvuywts->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,ixuv,Iisu,Xzvswy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ixuv,Iisu,Xzvsyw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ixuv,Iisu,Xzvwsy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ixuv,Iisu,Xzvwys->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ixuv,Iisu,Xzvysw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ixuv,Iisu,Xzvyws->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ixuv,Iisv,Xzuswy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,Iisv,Xzusyw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,Iisv,Xzuwsy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,Iisv,Xzuwys->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,Iisv,Xzuysw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,Iisv,Xzuyws->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixuv,IiuX,zvwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,ixuv,Iius,Xzvswy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,Iius,Xzvsyw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,Iius,Xzvwsy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,Iius,Xzvwys->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,Iius,Xzvysw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,Iius,Xzvyws->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,Iiuv,Xzyw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,IivX,zuwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,Iivs,Xzuyws->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixuv,Iivu,Xzyw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uXxa,Ia,zuwy->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uXxa,Ivas,zusvwy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uXxa,Ivas,zusvyw->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uXxa,Ivas,zuswvy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('xyzw,uXxa,Ivas,zuswyv->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uXxa,Ivas,zusyvw->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uXxa,Ivas,zusywv->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uXxa,Ivsa,zusvwy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uXxa,Ivsa,zusvyw->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uXxa,Ivsa,zuswvy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,uXxa,Ivsa,zuswyv->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uXxa,Ivsa,zusyvw->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uXxa,Ivsa,zusywv->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxa,Ia,Xywvuz->IX', v_aaaa, t1_aaae, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uvxa,IsXa,zuvwys->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxa,IsaX,zuvwys->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/48 * einsum('xyzw,uvxa,Isat,Xywsuvzt->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 5/48 * einsum('xyzw,uvxa,Isat,Xywsuzvt->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 7/48 * einsum('xyzw,uvxa,Isat,Xywsvuzt->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 5/48 * einsum('xyzw,uvxa,Isat,Xywsvzut->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uvxa,Isat,Xywsztuv->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uvxa,Isat,Xywsztvu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uvxa,Isat,Xywszutv->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 3/80 * einsum('xyzw,uvxa,Isat,Xywszuvt->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uvxa,Isat,Xywszvtu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/240 * einsum('xyzw,uvxa,Isat,Xywszvut->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 19/480 * einsum('xyzw,uvxa,Ista,Xywstuzv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 41/480 * einsum('xyzw,uvxa,Ista,Xywstvzu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/160 * einsum('xyzw,uvxa,Ista,Xywstzuv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 13/160 * einsum('xyzw,uvxa,Ista,Xywstzvu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/480 * einsum('xyzw,uvxa,Ista,Xywsutvz->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 17/80 * einsum('xyzw,uvxa,Ista,Xywsutzv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/480 * einsum('xyzw,uvxa,Ista,Xywsuvtz->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 13/160 * einsum('xyzw,uvxa,Ista,Xywsuvzt->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 11/240 * einsum('xyzw,uvxa,Ista,Xywsuztv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 41/480 * einsum('xyzw,uvxa,Ista,Xywsuzvt->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/480 * einsum('xyzw,uvxa,Ista,Xywsvtuz->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 7/80 * einsum('xyzw,uvxa,Ista,Xywsvtzu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/480 * einsum('xyzw,uvxa,Ista,Xywsvutz->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 13/160 * einsum('xyzw,uvxa,Ista,Xywsvuzt->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 19/240 * einsum('xyzw,uvxa,Ista,Xywsvztu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 41/480 * einsum('xyzw,uvxa,Ista,Xywsvzut->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 61/480 * einsum('xyzw,uvxa,Ista,Xywsztuv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/480 * einsum('xyzw,uvxa,Ista,Xywsztvu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 61/480 * einsum('xyzw,uvxa,Ista,Xywszutv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/48 * einsum('xyzw,uvxa,Ista,Xywszuvt->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/480 * einsum('xyzw,uvxa,Ista,Xywszvtu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/80 * einsum('xyzw,uvxa,Ista,Xywszvut->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxva,Ia,Xzvywu->IX', v_aaaa, t1_aaae, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uxva,IsXa,zvswuy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxva,IsaX,zvswuy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/20 * einsum('xyzw,uxva,Isat,Xzvstwuy->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/30 * einsum('xyzw,uxva,Isat,Xzvswyut->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxva,Isat,Xzvsywut->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxva,Isav,Xzsywu->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/8 * einsum('xyzw,uxva,Ista,Xzvstwuy->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyzw,uxva,Ista,Xzvstyuw->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/20 * einsum('xyzw,uxva,Ista,Xzvsuwty->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/40 * einsum('xyzw,uxva,Ista,Xzvsuwyt->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/20 * einsum('xyzw,uxva,Ista,Xzvsuytw->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/40 * einsum('xyzw,uxva,Ista,Xzvsuywt->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyzw,uxva,Ista,Xzvswtuy->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 11/120 * einsum('xyzw,uxva,Ista,Xzvswyut->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyzw,uxva,Ista,Xzvsytuw->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyzw,uxva,Ista,Xzvsywut->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxva,Isva,Xzsuwy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxva,Isva,Xzsuyw->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxva,Isva,Xzswuy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxva,Isva,Xzswyu->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxva,Isva,Xzsyuw->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxva,Isva,Xzsywu->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xXua,Ia,zuwy->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xXua,Ivas,zuvswy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xXua,Ivas,zuvsyw->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xXua,Ivas,zuvwsy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,xXua,Ivas,zuvwys->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xXua,Ivas,zuvysw->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xXua,Ivas,zuvyws->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xXua,Ivau,zvwy->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xXua,Ivsa,zuvswy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xXua,Ivsa,zuvsyw->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xXua,Ivsa,zuvwsy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xXua,Ivsa,zuvwys->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xXua,Ivsa,zuvysw->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xXua,Ivsa,zuvyws->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xXua,Ivua,zvwy->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xa,Ia,Xzyw->IX', v_aaaa, t1_ae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xa,IuXa,zuwy->IX', v_aaaa, t1_ae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xa,IuaX,zuwy->IX', v_aaaa, t1_ae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xa,Iuav,Xzuywv->IX', v_aaaa, t1_ae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xa,Iuva,Xzuvwy->IX', v_aaaa, t1_ae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xa,Iuva,Xzuvyw->IX', v_aaaa, t1_ae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xa,Iuva,Xzuwvy->IX', v_aaaa, t1_ae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xa,Iuva,Xzuwyv->IX', v_aaaa, t1_ae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xa,Iuva,Xzuyvw->IX', v_aaaa, t1_ae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xa,Iuva,Xzuywv->IX', v_aaaa, t1_ae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuab,Ivab,Xzvywu->IX', v_aaaa, t1_aaee, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuab,Ivba,Xzvuwy->IX', v_aaaa, t1_aaee, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuab,Ivba,Xzvuyw->IX', v_aaaa, t1_aaee, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuab,Ivba,Xzvwuy->IX', v_aaaa, t1_aaee, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuab,Ivba,Xzvwyu->IX', v_aaaa, t1_aaee, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuab,Ivba,Xzvyuw->IX', v_aaaa, t1_aaee, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuab,Ivba,Xzvywu->IX', v_aaaa, t1_aaee, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuva,Ia,Xzvuwy->IX', v_aaaa, t1_aaae, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,Ia,Xzvuyw->IX', v_aaaa, t1_aaae, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,Ia,Xzvwuy->IX', v_aaaa, t1_aaae, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,Ia,Xzvwyu->IX', v_aaaa, t1_aaae, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,Ia,Xzvyuw->IX', v_aaaa, t1_aaae, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,Ia,Xzvywu->IX', v_aaaa, t1_aaae, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuva,IsXa,zvsuwy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuva,IsXa,zvsuyw->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuva,IsXa,zvswuy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,xuva,IsXa,zvswyu->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuva,IsXa,zvsyuw->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuva,IsXa,zvsywu->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,IsaX,zvsuwy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,IsaX,zvsuyw->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,IsaX,zvswuy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuva,IsaX,zvswyu->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,IsaX,zvsyuw->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,IsaX,zvsywu->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/8 * einsum('xyzw,xuva,Isat,Xzvsuwyt->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyzw,xuva,Isat,Xzvsuywt->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,xuva,Isat,Xzvswtuy->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,xuva,Isat,Xzvswtyu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,xuva,Isat,Xzvswuty->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/6 * einsum('xyzw,xuva,Isat,Xzvswuyt->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,xuva,Isat,Xzvswytu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 2/15 * einsum('xyzw,xuva,Isat,Xzvswyut->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/8 * einsum('xyzw,xuva,Isat,Xzvsytuw->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/8 * einsum('xyzw,xuva,Isat,Xzvsytwu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/8 * einsum('xyzw,xuva,Isat,Xzvsyutw->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/8 * einsum('xyzw,xuva,Isat,Xzvsywtu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuva,Isav,Xzsuwy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,Isav,Xzsuyw->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,Isav,Xzswuy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,Isav,Xzswyu->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,Isav,Xzsyuw->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,Isav,Xzsywu->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyzw,xuva,Ista,Xzvstwuy->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/8 * einsum('xyzw,xuva,Ista,Xzvstyuw->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyzw,xuva,Ista,Xzvsuwty->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyzw,xuva,Ista,Xzvsuwyt->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/8 * einsum('xyzw,xuva,Ista,Xzvsuytw->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/8 * einsum('xyzw,xuva,Ista,Xzvsuywt->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyzw,xuva,Ista,Xzvswtyu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 11/120 * einsum('xyzw,xuva,Ista,Xzvswyut->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyzw,xuva,Ista,Xzvsytwu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,xuva,Ista,Xzvsywtu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/8 * einsum('xyzw,xuva,Ista,Xzvsywut->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuva,Isva,Xzsywu->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xzab,Iuab,Xuyw->IX', v_aaaa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xzua,Ivua,Xvyw->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,zxua,Ia,Xuyw->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,zxua,IvXa,ywvu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,zxua,IvaX,ywvu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,zxua,Ivas,Xuvswy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,zxua,Ivas,Xuvsyw->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,zxua,Ivas,Xuvwsy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,zxua,Ivas,Xuvwys->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,zxua,Ivas,Xuvysw->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,zxua,Ivas,Xuvyws->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,zxua,Ivau,Xvyw->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,zxua,Ivsa,Xuvswy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)

    # Compute denominators
    d_ip = (-e_core[:,None] + evals)
    d_ip = d_ip**(-1)

    # Compute T[+1']^(2) amplitudes
    S_12_V_p1p = einsum("iP,Pm->im", V1, S_p1p_12_inv_act, optimize = einsum_type)
    S_12_V_p1p = einsum("mp,im->ip", evecs, S_12_V_p1p, optimize = einsum_type)
    S_12_V_p1p = einsum("ip,ip->ip", d_ip, S_12_V_p1p, optimize = einsum_type)
    S_12_V_p1p = einsum("mp,ip->im", evecs, S_12_V_p1p, optimize = einsum_type)
    del(V1, d_ip, evals, evecs)

    ## Compute T[+1']^(2) t2_ca tensor
    t2_ca = einsum("Pm,im->iP", S_p1p_12_inv_act, S_12_V_p1p, optimize = einsum_type)
    del(S_p1p_12_inv_act, S_12_V_p1p)
 
    mr_adc.log.extra("Norm of T[+1']^(2):                          %20.12f" % np.linalg.norm(t2_ca))
    mr_adc.log.timer("computing T[+1']^(2) amplitudes", *cput0)

    return t2_ca

def compute_t2_0pp_singles(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing T[0'']^(2) amplitudes...")

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

    ### One-electron integrals
    #h_ce = mr_adc.h1eff.ce
    #h_ca = mr_adc.h1eff.ca
    #h_ae = mr_adc.h1eff.ae
    #h_aa = mr_adc.h1eff.aa

    ### Two-electron integrals
    #v_aaaa = mr_adc.v2e.aaaa
    #
    #v_ccce = mr_adc.v2e.ccce
    #v_ccca = mr_adc.v2e.ccca
    #
    #v_cece = mr_adc.v2e.cece
    #v_caca = mr_adc.v2e.caca    
    #v_ccaa = mr_adc.v2e.ccaa
    #v_caac = mr_adc.v2e.caac    
 
    #v_cace = mr_adc.v2e.cace 
    #v_ccae = mr_adc.v2e.ccae
    #v_caec = mr_adc.v2e.caec
    #
    #v_caaa = mr_adc.v2e.caaa

    #v_ceae = mr_adc.v2e.ceae
    #v_aeae = mr_adc.v2e.aeae

    #v_caae = mr_adc.v2e.caae
    #v_ceaa = mr_adc.v2e.ceaa 
    #v_caea = mr_adc.v2e.caea
    #
    #v_aaae = mr_adc.v2e.aaae
    #
    ### Amplitudes
    #t1_ce = mr_adc.t1.ce
    #t1_caea = mr_adc.t1.caea
    #t1_caae = mr_adc.t1.caae

    #t1_ca = mr_adc.t1.ca
    #t1_caaa = mr_adc.t1.caaa

    #t1_ae   = mr_adc.t1.ae
    #t1_aaae = mr_adc.t1.aaae
    #t1_aaee = mr_adc.t1.aaee

    #t1_ccee = mr_adc.t1.ccee
    #t1_ccae = mr_adc.t1.ccae
    #t1_ccaa = mr_adc.t1.ccaa
    #t1_caee = mr_adc.t1.caee

    ### Reduced density matrices
    #rdm_ca = mr_adc.rdm.ca
    #rdm_ccaa = mr_adc.rdm.ccaa
    #rdm_cccaaa = mr_adc.rdm.cccaaa
    #rdm_ccccaaaa = mr_adc.rdm.ccccaaaa

    # Compute K_caca matrix
    K_caca = mr_adc_intermediates.compute_K_0pp(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_0p_12_inv_act = mr_adc_overlap.compute_S12_0pp(mr_adc)

    # Compute K^{-1} matrix
    SKS = reduce(np.dot, (S_0p_12_inv_act.T, K_caca, S_0p_12_inv_act))
    evals, evecs = np.linalg.eigh(SKS)

    if np.any(evals) and (np.amin(np.absolute(evals)) < 1e-6):
        print ("WARNING: Small denominator of %e is detected in the T[0'']^(2) amplitude equations...\n" % np.amin(np.absolute(evals)))
 
    # Compute R.H.S. of the equation
    # V1 block: - 1/2 < Psi_0 | a^{\dag}_X a_Y  [V + H^{(1)}, T - T^\dag] | Psi_0 >
    ##code block pending
    ##TODO: check if V1 or V1_sym produces correct results

    ##prism beta code below!
    #S_12_V0p = np.einsum("P,Pm->m", V1, S_0p_12_inv_act)

    ## Multiply r.h.s. by U (e_a - e_i + e_mu)^-1 U^dag
    #S_12_V0p = np.einsum("mp,m->p", evecs, S_12_V0p)
 
    ## Compute denominators
    #evals = evals**(-1)
 
    #S_12_V0p = evals * S_12_V0p
    #S_12_V0p = np.einsum("mp,p->m", evecs, S_12_V0p)
 
    #t_0p = np.einsum("m,Pm->P", S_12_V0p, S_0p_12_inv_act)
 
    #t_xy = np.zeros((ncas_so, ncas_so))
    #t_xy[xy_ind[0], xy_ind[1]] = t_0p
    #t_xy[xy_ind[1], xy_ind[0]] = -t_0p

    print('\n###           CAUTION: t2_aa has been set to zero!           ###\n')
    t2_aa = np.zeros((ncas, ncas))
 
    mr_adc.log.extra("Norm of T[0'']^(2):                          %20.12f" % np.linalg.norm(t2_aa))
    mr_adc.log.timer("computing T[0'']^(2) amplitudes", *cput0)

    return t2_aa 
