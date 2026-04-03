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
#                  Ilia M. Mazin <ilia.mazin@gmail.com>
#              Donna H. Odhiambo <donna.odhiambo@proton.me>
#

import numpy as np
from functools import reduce

from prism.mr_adc import intermediates
from prism.mr_adc import overlap
from prism.mr_adc import integrals

import prism.lib.logger as logger
import prism.lib.tools as tools

def compute_reference_energy(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.info("\nComputing NEVPT2 amplitudes...")

    # First-order amplitudes
    e_corr = compute_t1_amplitudes(mr_adc)

    # Second-order amplitudes
    compute_t2_amplitudes(mr_adc)

    # Compute CVS amplitudes and remove non-CVS core integrals, amplitudes and unnecessary RDMs
    if mr_adc.method_type in ("cvs-ip", "cvs-ee"):
        compute_cvs_amplitudes(mr_adc)

    e_tot = mr_adc.e_ref[0] + e_corr

    mr_adc.log.info("\nReference energy:                            %20.12f" % mr_adc.e_ref[0])
    mr_adc.log.info("NEVPT2 correlation energy:                   %20.12f" % e_corr)
    mr_adc.log.info("Total NEVPT2 energy:                         %20.12f" % e_tot)

    mr_adc.log.timer("computing amplitudes", *cput0)

    mr_adc.e_ref_nevpt2 = e_tot

    return e_tot, e_corr

def compute_t1_amplitudes(mr_adc):

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nelecas = mr_adc.ref_nelecas
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
    if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-sx", "mr-adc(2)-x"):
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

    if ((mr_adc.method in ("mr-adc(2)", "mr-adc(2)-sx", "mr-adc(2)-x")) or
        (mr_adc.method == "mr-adc(1)" and mr_adc.method_type in ("ee", "cvs-ee"))):

        nelecas_total = 0
        if isinstance(nelecas, (list)):
            nelecas_total = sum(nelecas[0])
        else:
            nelecas_total = sum(nelecas)

        if ncore > 0 and nextern > 0:
            e_0, mr_adc.t1.ccee = compute_t1_0(mr_adc)
        else:
            mr_adc.t1.ccee = np.zeros((ncore, ncore, nextern, nextern))

        if ncore > 0 and nextern > 0 and ncas > 0:
            e_p1, mr_adc.t1.ccae = compute_t1_p1(mr_adc)
        else:
            mr_adc.t1.ccae = np.zeros((ncore, ncore, ncas, nextern))

        if ncore > 0 and nextern > 0 and ncas > 0 and nelecas_total > 0:
            e_m1, mr_adc.t1.caee = compute_t1_m1(mr_adc)
        else:
            mr_adc.t1.caee = np.zeros((ncore, ncas, nextern, nextern))

        if ncore > 0 and ncas > 0:
            e_p2, mr_adc.t1.ccaa = compute_t1_p2(mr_adc)
        else:
            mr_adc.t1.ccaa = np.zeros((ncore, ncore, ncas, ncas))

        if nextern > 0 and ncas > 0 and nelecas_total > 1:
            e_m2, mr_adc.t1.aaee = compute_t1_m2(mr_adc)
        else:
            mr_adc.t1.aaee = np.zeros((ncas, ncas, nextern, nextern))

    else:
        mr_adc.t1.ccee = np.zeros((ncore, ncore, nextern, nextern))
        mr_adc.t1.ccae = np.zeros((ncore, ncore, ncas, nextern))
        mr_adc.t1.caee = np.zeros((ncore, ncas, nextern, nextern))
        mr_adc.t1.ccaa = np.zeros((ncore, ncore, ncas, ncas))
        mr_adc.t1.aaee = np.zeros((ncas, ncas, nextern, nextern))

    corr_cont = [("[0']", e_0p), ("[+1']", e_p1p), ("[-1']", e_m1p),
                ("[0]",  e_0),  ("[+1]",  e_p1),  ("[-1]",  e_m1),
                ("[+2]", e_p2), ("[-2]",  e_m2)]

    if positive := [k for k, v in corr_cont if v > 0]:
        mr_adc.log.warn(f'Positive correlation energies found for {", ".join(positive)}.')

    e_corr = sum(v for _, v in corr_cont)

    return e_corr

def compute_t2_amplitudes(mr_adc):

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern
    nelecas = mr_adc.ref_nelecas

    nelecas_total = 0
    if isinstance(nelecas, list):
        nelecas_total = sum(nelecas[0])
    else:
        nelecas_total = sum(nelecas)

    approx_trans_moments = mr_adc.approx_trans_moments

    # Approximate second-order amplitudes
    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-sx", "mr-adc(2)-x"):

        if ncore > 0 and ncas > 0 and nextern > 0 and not approx_trans_moments:
            mr_adc.t2.ce = compute_t2_0p_singles(mr_adc)
        else:
            mr_adc.t2.ce = np.zeros((ncore, nextern))

        if mr_adc.method_type == "cvs-ee":

            if ncas > 0 and nextern > 0 and nelecas_total > 0:
                mr_adc.t2.ae = compute_t2_m1p_singles(mr_adc)
            else:
                mr_adc.t2.ae = np.zeros((ncas, nextern))

            if ncas > 0 and nelecas_total > 0:
                mr_adc.t2.aa = compute_t2_0pp_singles(mr_adc)
            else:
                mr_adc.t2.aa = np.zeros((ncas, ncas))

            if ncore > 0 and ncas > 0:
                mr_adc.t2.ca = compute_t2_p1p_singles(mr_adc)
            else:
                mr_adc.t2.ca = np.zeros((ncore, ncas))


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

        if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x", "mr-adc(2)-sx"):
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

            if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x", "mr-adc(2)-sx"): 
                mr_adc.t2.xe = np.ascontiguousarray(mr_adc.t2.ce[:ncvs, :])
                mr_adc.t2.ve = np.ascontiguousarray(mr_adc.t2.ce[ncvs:, :])
                del(mr_adc.t2.ce)

                if mr_adc.method_type == "cvs-ee":
                    mr_adc.t2.xa = np.ascontiguousarray(mr_adc.t2.ca[:ncvs, :])
                    mr_adc.t2.va = np.ascontiguousarray(mr_adc.t2.ca[ncvs:, :])
                    del(mr_adc.t2.ca)

            mr_adc.t1.xxee = tools.create_dataset('xxee', tmpfile, (ncvs, ncvs, nextern, nextern))
            mr_adc.t1.xvee = tools.create_dataset('xvee', tmpfile, (ncvs, nval, nextern, nextern))
            mr_adc.t1.vxee = tools.create_dataset('vxee', tmpfile, (nval, ncvs, nextern, nextern))
            mr_adc.t1.vvee = tools.create_dataset('vvee', tmpfile, (nval, nval, nextern, nextern))

            mr_adc.t1.xaee = tools.create_dataset('xaee', tmpfile, (ncvs, ncas, nextern, nextern))
            mr_adc.t1.vaee = tools.create_dataset('vaee', tmpfile, (nval, ncas, nextern, nextern))

            if ncore > 0 and nextern > 0:
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

            if ncore > 0 and nextern > 0 and ncas > 0:
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
    K_ac = intermediates.compute_K_ac(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_p1_12_inv_act = overlap.compute_S12_p1(mr_adc)

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
    K_ca = intermediates.compute_K_ca(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_m1_12_inv_act = overlap.compute_S12_m1(mr_adc)

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
    K_aacc = intermediates.compute_K_aacc(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_p2_12_inv_act = overlap.compute_S12_p2(mr_adc)

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
    K_ccaa = intermediates.compute_K_ccaa(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_m2_12_inv_act = overlap.compute_S12_m2(mr_adc)

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
    K_caca = intermediates.compute_K_caca(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_0p_12_inv_act = overlap.compute_S12_0p_gno_projector(mr_adc)

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
    K_p1p = intermediates.compute_K_p1p(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    if mr_adc.semi_internal_projector == "gno":
        S_p1p_12_inv_act = overlap.compute_S12_p1p_gno_projector(mr_adc)
    else:
        S_p1p_12_inv_act = overlap.compute_S12_p1p_gs_projector(mr_adc)

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
    e_p1p -= einsum('ixyz,iwuz,yuwx', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= einsum('ixyz,izwu,ywxu', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= einsum('ixyz,iwuv,xwvzyu', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
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
    K_m1p = intermediates.compute_K_m1p(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    if mr_adc.semi_internal_projector == "gno":
        S_m1p_12_inv_act = overlap.compute_S12_m1p_gno_projector(mr_adc)
    else:
        S_m1p_12_inv_act = overlap.compute_S12_m1p_gs_projector(mr_adc)

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
    e_m1p += einsum('xyza,wuva,zvwxyu', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p += einsum('xyza,wzua,xywu', t1_aaae, v_aaae, rdm_ccaa, optimize = einsum_type)

    mr_adc.log.extra("Norm of T[-1']^(1):                          %20.12f" % (np.linalg.norm(t1_ae) +
                                                                              np.linalg.norm(t1_aaae)))
    mr_adc.log.info("Correlation energy [-1']:                    %20.12f" % e_m1p)
    mr_adc.log.timer("computing T[-1']^(1) amplitudes", *cput0)

    return e_m1p, t1_ae, t1_aaae

def compute_t2_0p_singles(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("Computing T[0']^(2) amplitudes...")

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
    v_ccca = mr_adc.v2e.ccca
    v_ccce = mr_adc.v2e.ccce
    v_ccaa = mr_adc.v2e.ccaa
    v_ccae = mr_adc.v2e.ccae
    v_cace = mr_adc.v2e.cace
    v_caac = mr_adc.v2e.caac
    v_caaa = mr_adc.v2e.caaa
    v_caae = mr_adc.v2e.caae
    v_caec = mr_adc.v2e.caec
    v_caea = mr_adc.v2e.caea
    v_ceaa = mr_adc.v2e.ceaa
    v_ceae = mr_adc.v2e.ceae
    v_aaaa = mr_adc.v2e.aaaa
    v_aaae = mr_adc.v2e.aaae

    ## Amplitudes
    t1_ca = mr_adc.t1.ca
    t1_ce = mr_adc.t1.ce
    t1_ae = mr_adc.t1.ae

    t1_ccaa = mr_adc.t1.ccaa
    t1_ccae = mr_adc.t1.ccae
    t1_caaa = mr_adc.t1.caaa
    t1_caae = mr_adc.t1.caae
    t1_caea = mr_adc.t1.caea
    t1_aaae = mr_adc.t1.aaae

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
    V1 -= einsum('i,ix,IixA->IA', e_core, t1_ca, t1_ccae, optimize = einsum_type)
    V1 += 2 * einsum('i,ix,iIxA->IA', e_core, t1_ca, t1_ccae, optimize = einsum_type)
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
    V1 += 1/2 * einsum('i,ix,IiyA,xy->IA', e_core, t1_ca, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,ix,iIyA,xy->IA', e_core, t1_ca, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ixyz,IiwA,xwzy->IA', e_core, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('i,ixyz,IiyA,xz->IA', e_core, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ixyz,IizA,xy->IA', e_core, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,ixyz,iIwA,xwzy->IA', e_core, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 2 * einsum('i,ixyz,iIyA,xz->IA', e_core, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,ixyz,iIzA,xy->IA', e_core, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
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
    V1 -= 1/4 * einsum('xyzw,IixA,iuvs,zvswyu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
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
    V1 += 1/2 * einsum('xyzw,Iuvx,svtA,zuswyt->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuvx,vA,zuwy->IA', v_aaaa, t1_caaa, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuvx,vstA,zuswyt->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuxv,stuA,ztswyv->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxv,svtA,zuswyt->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxv,vA,zuwy->IA', v_aaaa, t1_caaa, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxv,vstA,zuswty->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuxz,vsuA,ywsv->IA', v_aaaa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Ixuv,sutA,zvtwys->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Ixuv,svtA,zutwys->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Ixuv,uA,zvwy->IA', v_aaaa, t1_caaa, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Ixuv,ustA,zvtwys->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Ixuv,uvsA,zswy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Ixuv,vA,zuwy->IA', v_aaaa, t1_caaa, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Ixuv,vstA,zutwsy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Ixuv,vusA,zswy->IA', v_aaaa, t1_caaa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iIxA,iu,zuwy->IA', v_aaaa, t1_ccae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iIxA,iuvs,zvswyu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,iIxA,iuvw,zvuy->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,iIxA,iuvy,zvwu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,iIxA,iuwv,zvyu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,iIxA,iuwy,zu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xyzw,iIxA,iuyv,zvwu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('xyzw,iIxA,iuyw,zu->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,iIxA,iw,zy->IA', v_aaaa, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xyzw,iIxA,iy,zw->IA', v_aaaa, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,iIxA,izuv,ywuv->IA', v_aaaa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuvx,IisA,zuswyv->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuvx,IivA,zuwy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuvx,iIsA,zuswyv->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
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
    V1 += 1/4 * einsum('xyzw,uvxA,Istv,zuswyt->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uvxA,Istz,ywtusv->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxA,Isut,zvswyt->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxA,Isuv,zswy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxA,Isuz,ywvs->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uvxA,Isvt,zuswyt->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uvxA,Isvu,zswy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uvxA,Isvz,ywus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uvxA,Iszt,ywtuvs->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxA,Iszu,ywsv->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxA,Iszv,ywus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxA,Iu,zvwy->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uvxA,Iv,zuwy->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uvxA,Iwst,zvutsy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uvxA,Iyst,zvuwst->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uvxA,Iz,ywuv->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxvA,Istu,zvtwsy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Istw,zvtsuy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Isty,zvtwus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxvA,Isut,zvtwys->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Isuw,zvsy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Isuy,zvws->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Iswt,zvtyus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Iswu,zvys->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Iswy,zvsu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,uxvA,Isyt,zvtwus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,uxvA,Isyu,zvws->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,uxvA,Isyw,zvsu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxvA,Iu,zvwy->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uxvA,Ivst,zstwyu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxvA,Iw,zvyu->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,uxvA,Iy,zvwu->IA', v_aaaa, t1_aaae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uxvA,Izst,ywustv->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
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
    V1 -= 1/4 * einsum('xyzw,xuvA,Istu,zvtwys->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xuvA,Istw,zvtsyu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xuvA,Isty,zvtwsu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xuvA,Isut,zvtwys->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,xuvA,Isuw,zvsy->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,xuvA,Isuy,zvws->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xuvA,Iswt,zvtuys->IA', v_aaaa, t1_aaae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
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
        e_extern_A = mr_adc.mo_energy.e[s_chunk:f_chunk]

        ## Amplitudes
        t1_ccee = mr_adc.t1.ccee[:, :, s_chunk:f_chunk]

        temp =- einsum('A,IiAa,ia->IA', e_extern_A, t1_ccee, t1_ce, optimize = einsum_type)
        temp += 1/2 * einsum('A,iIAa,ia->IA', e_extern_A, t1_ccee, t1_ce, optimize = einsum_type)
        temp -= einsum('A,IiAa,ixay,yx->IA', e_extern_A, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('A,IiAa,ixya,yx->IA', e_extern_A, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('A,iIAa,ixay,yx->IA', e_extern_A, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('A,iIAa,ixya,yx->IA', e_extern_A, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
        temp -= 2 * einsum('ia,IiAa->IA', h_ce, t1_ccee, optimize = einsum_type)
        temp += einsum('ia,iIAa->IA', h_ce, t1_ccee, optimize = einsum_type)
        temp += 2 * einsum('ijAa,iIja->IA', t1_ccee, v_ccce, optimize = einsum_type)
        temp -= einsum('ijAa,jIia->IA', t1_ccee, v_ccce, optimize = einsum_type)
        temp += einsum('I,IiAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
        temp -= 1/2 * einsum('I,iIAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
        temp -= 2 * einsum('a,IiAa,ia->IA', e_extern, t1_ccee, t1_ce, optimize = einsum_type)
        temp += einsum('a,iIAa,ia->IA', e_extern, t1_ccee, t1_ce, optimize = einsum_type)
        temp += 2 * einsum('i,IiAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
        temp -= einsum('i,iIAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
        temp -= 2 * einsum('IiAa,iaxy,yx->IA', t1_ccee, v_ceaa, rdm_ca, optimize = einsum_type)
        temp += einsum('IiAa,ixya,xy->IA', t1_ccee, v_caae, rdm_ca, optimize = einsum_type)
        temp += einsum('iIAa,iaxy,yx->IA', t1_ccee, v_ceaa, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('iIAa,ixya,xy->IA', t1_ccee, v_caae, rdm_ca, optimize = einsum_type)
        temp += einsum('I,IiAa,ixay,yx->IA', e_core, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('I,IiAa,ixya,yx->IA', e_core, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('I,iIAa,ixay,yx->IA', e_core, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
        temp += 1/4 * einsum('I,iIAa,ixya,yx->IA', e_core, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
        temp -= 2 * einsum('a,ixay,IiAa,yx->IA', e_extern, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += einsum('a,ixay,iIAa,yx->IA', e_extern, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += einsum('a,ixya,IiAa,yx->IA', e_extern, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('a,ixya,iIAa,yx->IA', e_extern, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 2 * einsum('i,ixay,IiAa,xy->IA', e_core, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= einsum('i,ixay,iIAa,xy->IA', e_core, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= einsum('i,ixya,IiAa,xy->IA', e_core, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('i,ixya,iIAa,xy->IA', e_core, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += einsum('xy,ixaz,IiAa,yz->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('xy,ixaz,iIAa,yz->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('xy,ixza,IiAa,yz->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/4 * einsum('xy,ixza,iIAa,yz->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= einsum('xy,izax,IiAa,yz->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xy,izax,iIAa,yz->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xy,izxa,IiAa,yz->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,izxa,iIAa,yz->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= einsum('xyzw,iuax,IiAa,zuwy->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,iuax,iIAa,zuwy->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,iuxa,IiAa,zuwy->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,iuxa,iIAa,zuwy->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp += einsum('xyzw,ixau,IiAa,zuwy->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xyzw,ixau,iIAa,zuwy->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xyzw,ixua,IiAa,zuwy->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,ixua,iIAa,zuwy->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting t1.ccee", *cput1)
    del(t1_ccee)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Molecular Orbitals Energies
        e_extern_A = mr_adc.mo_energy.e[s_chunk:f_chunk]

        ## Amplitudes
        t1_caee = mr_adc.t1.caee[:, :, :, s_chunk:f_chunk]

        temp  = 1/4 * einsum('A,IxaA,ya,xy->IA', e_extern_A, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
        temp += 1/4 * einsum('A,IxaA,yzwa,xwzy->IA', e_extern_A, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xa,IyaA,xy->IA', h_ae, t1_caee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('IxaA,yzwa,xzwy->IA', t1_caee, v_aaae, rdm_ccaa, optimize = einsum_type)
        temp += einsum('ixaA,Iyai,xy->IA', t1_caee, v_caec, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('ixaA,iIya,xy->IA', t1_caee, v_ccae, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('I,IxaA,ya,xy->IA', e_core, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('I,IxaA,yzwa,xwzy->IA', e_core, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('a,xa,IyaA,xy->IA', e_extern, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('a,xyza,IwaA,zwxy->IA', e_extern, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,IxaA,za,yz->IA', h_aa, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,IxaA,zwua,yuwz->IA', h_aa, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,xa,IzaA,yz->IA', h_aa, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,xzwa,IuaA,yzwu->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('xy,zwxa,IuaA,yuzw->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,zxwa,IuaA,yzuw->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,IxaA,ua,zuwy->IA', v_aaaa, t1_caee, t1_ae, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,IxaA,uvsa,zvuwys->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,IxaA,uvza,ywvu->IA', v_aaaa, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,uvxa,IsaA,zuvwys->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,uxva,IsaA,zvswuy->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,xa,IuaA,zuwy->IA', v_aaaa, t1_ae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,xuva,IsaA,zvswyu->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,zxua,IvaA,ywvu->IA', v_aaaa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting t1.caee", *cput1)
    del(t1_caee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Molecular Orbitals Energies
        e_extern_A = mr_adc.mo_energy.e[s_chunk:f_chunk]

        ## Amplitudes
        t1_caee = mr_adc.t1.caee[:, :, s_chunk:f_chunk]
 
        temp =- 1/2 * einsum('A,IxAa,ya,xy->IA', e_extern_A, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('A,IxAa,yzwa,xwzy->IA', e_extern_A, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
        temp -= einsum('xa,IyAa,xy->IA', h_ae, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= einsum('IxAa,yzwa,xzwy->IA', t1_caee, v_aaae, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('ixAa,Iyai,xy->IA', t1_caee, v_caec, rdm_ca, optimize = einsum_type)
        temp += einsum('ixAa,iIya,xy->IA', t1_caee, v_ccae, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('I,IxAa,ya,xy->IA', e_core, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('I,IxAa,yzwa,xwzy->IA', e_core, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
        temp -= einsum('a,xa,IyAa,xy->IA', e_extern, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= einsum('a,xyza,IwAa,zwxy->IA', e_extern, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xy,IxAa,za,yz->IA', h_aa, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xy,IxAa,zwua,yuwz->IA', h_aa, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xy,xa,IzAa,yz->IA', h_aa, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xy,xzwa,IuAa,yzwu->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xy,zwxa,IuAa,yuzw->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xy,zxwa,IuAa,yzuw->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,IxAa,ua,zuwy->IA', v_aaaa, t1_caee, t1_ae, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,IxAa,uvsa,zvuwys->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,IxAa,uvza,ywvu->IA', v_aaaa, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xyzw,uvxa,IsAa,zuvwys->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,uxva,IsAa,zvswuy->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,xa,IuAa,zuwy->IA', v_aaaa, t1_ae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,xuva,IsAa,zvswyu->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,zxua,IvAa,ywvu->IA', v_aaaa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting t1.caee", *cput1)
    del(t1_caee)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Amplitudes
        t1_aaee = mr_adc.t1.aaee[:, :, s_chunk:f_chunk]

        temp  = 1/2 * einsum('xyAa,Izaw,xyzw->IA', t1_aaee, v_caea, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting t1.aaee", *cput1)
    del(t1_aaee)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncore, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ccee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_ccee = mr_adc.v2e.ccee[:, :, s_chunk:f_chunk]

        temp  = einsum('ia,iIAa->IA', t1_ce, v_ccee, optimize = einsum_type)
        temp += einsum('ixay,iIAa,xy->IA', t1_caea, v_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('ixya,iIAa,xy->IA', t1_caae, v_ccee, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.ccee", *cput1)
    del(v_ccee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceec [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_ceec = mr_adc.v2e.ceec[:, s_chunk:f_chunk]

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
        v_cece = mr_adc.v2e.cece[:, s_chunk:f_chunk]

        temp =- 2 * einsum('ia,IAia->IA', t1_ce, v_cece, optimize = einsum_type)
        temp += einsum('ia,iAIa->IA', t1_ce, v_cece, optimize = einsum_type)
        temp -= 2 * einsum('ixay,IAia,yx->IA', t1_caea, v_cece, rdm_ca, optimize = einsum_type)
        temp += einsum('ixay,iAIa,yx->IA', t1_caea, v_cece, rdm_ca, optimize = einsum_type)
        temp += einsum('ixya,IAia,yx->IA', t1_caae, v_cece, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('ixya,iAIa,yx->IA', t1_caae, v_cece, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.cece", *cput1)
    del(v_cece)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_aaee = mr_adc.v2e.aaee[:, :, s_chunk:f_chunk]

        temp =- einsum('Ixay,zwAa,xwyz->IA', t1_caea, v_aaee, rdm_ccaa, optimize = einsum_type)
        temp -= einsum('Ixay,zyAa,xz->IA', t1_caea, v_aaee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('Ixya,zwAa,xwyz->IA', t1_caae, v_aaee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('Ixya,zyAa,xz->IA', t1_caae, v_aaee, rdm_ca, optimize = einsum_type)
        temp += einsum('Ixay,zwAa,zw,xy->IA', t1_caea, v_aaee, rdm_ca, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('Ixya,zwAa,zw,xy->IA', t1_caae, v_aaee, rdm_ca, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.aaee", *cput1)
    del(v_aaee)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.aeea [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_aeea = mr_adc.v2e.aeea[:, s_chunk:f_chunk]

        temp  = 1/2 * einsum('Ixay,yAaz,xz->IA', t1_caea, v_aeea, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('Ixay,zAaw,xzyw->IA', t1_caea, v_aeea, rdm_ccaa, optimize = einsum_type)
        temp -= einsum('Ixya,yAaz,xz->IA', t1_caae, v_aeea, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('Ixya,zAaw,xzwy->IA', t1_caae, v_aeea, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('Ixay,zAaw,wz,xy->IA', t1_caea, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)
        temp += 1/4 * einsum('Ixya,zAaw,wz,xy->IA', t1_caae, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.aeea", *cput1)
    del(v_aeea)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_caee = mr_adc.v2e.caee[:, :, s_chunk:f_chunk]

        temp  = einsum('Iixa,ixAa->IA', t1_ccae, v_caee, optimize = einsum_type)
        temp -= 2 * einsum('iIxa,ixAa->IA', t1_ccae, v_caee, optimize = einsum_type)
        temp -= 1/2 * einsum('Iixa,iyAa,xy->IA', t1_ccae, v_caee, rdm_ca, optimize = einsum_type)
        temp += einsum('iIxa,iyAa,xy->IA', t1_ccae, v_caee, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.caee", *cput1)
    del(v_caee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_caee = mr_adc.v2e.caee[:, :, :, s_chunk:f_chunk]

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
        v_ceae = mr_adc.v2e.ceae[:, s_chunk:f_chunk]

        temp =- einsum('xa,IAya,xy->IA', t1_ae, v_ceae, rdm_ca, optimize = einsum_type)
        temp -= einsum('xyza,IAwa,zwxy->IA', t1_aaae, v_ceae, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.ceae", *cput1)
    del(v_ceae)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceae [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_ceae = mr_adc.v2e.ceae[:, :, :, s_chunk:f_chunk]

        temp  = 1/2 * einsum('xa,IayA,xy->IA', t1_ae, v_ceae, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xyza,IawA,zwxy->IA', t1_aaae, v_ceae, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.ceae", *cput1)
    del(v_ceae)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceea [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_ceea = mr_adc.v2e.ceea[:, :, s_chunk:f_chunk]

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
        v_ceea = mr_adc.v2e.ceea[:, s_chunk:f_chunk]

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

        if mr_adc.interface.with_df:
            v_ceee = integrals.get_oeee_df(mr_adc, mr_adc.v2e.Lce, mr_adc.v2e.Lee, s_chunk, f_chunk)

        else:
            v_ceee = integrals.unpack_v2e_oeee(mr_adc, mr_adc.v2e.ceee[s_chunk:f_chunk])

        ## Amplitudes
        t1_ccee = mr_adc.t1.ccee[:, s_chunk:f_chunk]

        V1 += einsum('Iiab,iaAb->IA', t1_ccee, v_ceee, optimize = einsum_type)
        V1 -= 2 * einsum('Iiab,ibAa->IA', t1_ccee, v_ceee, optimize = einsum_type)

        mr_adc.log.timer_debug("contracting v2e.ceee", *cput1)
    del(v_ceee, t1_ccee)

    chunks = tools.calculate_double_chunks(mr_adc, ncas, [nextern, nextern, nextern],
                                                                    [ncore, nextern, nextern], ntensors = 2)
    for i_v_chunk, (s_v_chunk, f_v_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.aeee [%i/%i], chunk [%i:%i]", i_v_chunk + 1, len(chunks), s_v_chunk, f_v_chunk)

        if mr_adc.interface.with_df:
            v_aeee = integrals.get_oeee_df(mr_adc, mr_adc.v2e.Lae, mr_adc.v2e.Lee, s_v_chunk, f_v_chunk)
        else:
            v_aeee = integrals.unpack_v2e_oeee(mr_adc, mr_adc.v2e.aeee[s_v_chunk:f_v_chunk])

        for s_t_chunk, f_t_chunk in chunks:

            ## Amplitudes
            t1_caee = mr_adc.t1.caee[:, s_t_chunk:f_t_chunk]

            ## Reduced density matrices
            rdm_ca = mr_adc.rdm.ca[s_t_chunk:f_t_chunk, s_v_chunk:f_v_chunk]

            V1 += 1/2 * einsum('Ixab,yaAb,xy->IA', t1_caee, v_aeee, rdm_ca, optimize = einsum_type)
            V1 -= einsum('Ixab,ybAa,xy->IA', t1_caee, v_aeee, rdm_ca, optimize = einsum_type)

        mr_adc.log.timer_debug("contracting v2e.aeee", *cput1)

    if mr_adc.method_type == "cvs-ip":
        del(mr_adc.v2e.ceee, mr_adc.v2e.aeee)

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
    h_ae = mr_adc.h1eff.ae
    h_aa = mr_adc.h1eff.aa

    ## Two-electron integrals
    v_caca = mr_adc.v2e.caca
    v_cace = mr_adc.v2e.cace
    v_caaa = mr_adc.v2e.caaa
    v_caae = mr_adc.v2e.caae
    v_caea = mr_adc.v2e.caea
    v_ceaa = mr_adc.v2e.ceaa
    v_aaaa = mr_adc.v2e.aaaa
    v_aaae = mr_adc.v2e.aaae

    ## Amplitudes
    t1_ca   = mr_adc.t1.ca
    t1_ce   = mr_adc.t1.ce
    t1_ae   = mr_adc.t1.ae
    t1_ccaa = mr_adc.t1.ccaa
    t1_ccae = mr_adc.t1.ccae
    t1_caaa = mr_adc.t1.caaa
    t1_caae = mr_adc.t1.caae
    t1_caea = mr_adc.t1.caea
    t1_aaae = mr_adc.t1.aaae

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa
    rdm_ccccaaaa = mr_adc.rdm.ccccaaaa

    # Compute K_ca matrix
    K_ca = intermediates.compute_K_ca(mr_adc)
    
    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_m1_12_inv_act = overlap.compute_S12_m1(mr_adc)

    # Compute K^{-1} matrix
    SKS = reduce(np.dot, (S_m1_12_inv_act.T, K_ca, S_m1_12_inv_act))
    evals, evecs = np.linalg.eigh(SKS)
    del(SKS)

    # Compute R.H.S. of the equation
    # V1 block: - 1/2 < Psi_0 | a^{\dag}_X a_A [V + H^{(1)}, T - T^\dag] | Psi_0 >
    V1  = 1/2 * einsum('iA,ix,Xx->XA', h_ce, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('iA,ixyz,Xxyz->XA', h_ce, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,iA,Xx->XA', h_ca, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,iyAx,Xy->XA', h_ca, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,iyAz,Xzxy->XA', h_ca, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ix,iyxA,Xy->XA', h_ca, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,iyzA,Xzyx->XA', h_ca, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('iA,ixyz,Xyxz->XA', t1_ce, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ijxA,ixjy,Xy->XA', t1_ccae, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ijxA,jxiy,Xy->XA', t1_ccae, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ijxA,jyiz,Xxyz->XA', t1_ccae, v_caca, rdm_ccaa, optimize = einsum_type)
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
    V1 += 1/2 * einsum('ixAy,iyzw,Xzxw->XA', t1_caea, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixAy,izwu,Xywzxu->XA', t1_caea, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixAy,izwy,Xwzx->XA', t1_caea, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ixyA,iyzw,Xzxw->XA', t1_caae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyA,izwu,Xywxzu->XA', t1_caae, v_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyA,izwy,Xwxz->XA', t1_caae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,iAwu,Xxuyzw->XA', t1_caaa, v_ceaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,iAwy,Xxwz->XA', t1_caaa, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,iAwz,Xxyw->XA', t1_caaa, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,iwAu,Xyzuwx->XA', t1_caaa, v_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,iwAy,Xzwx->XA', t1_caaa, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,iwAz,Xyxw->XA', t1_caaa, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,iwuA,Xxwuzy->XA', t1_caaa, v_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('ixyz,iyAw,Xzwx->XA', t1_caaa, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ixyz,iyAz,Xx->XA', t1_caaa, v_caea, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixyz,iywA,Xxwz->XA', t1_caaa, v_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,izAw,Xywx->XA', t1_caaa, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,izAy,Xx->XA', t1_caaa, v_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,izwA,Xxwy->XA', t1_caaa, v_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,iA,ix,Xx->XA', e_extern, t1_ce, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,iA,ixyz,Xxyz->XA', e_extern, t1_ce, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('A,ijxA,ijxy,Xy->XA', e_extern, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,ijxA,jixy,Xy->XA', e_extern, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,ijxA,jiyz,Xxyz->XA', e_extern, t1_ccae, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixAy,iy,Xx->XA', e_extern, t1_caea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixAy,iz,Xyzx->XA', e_extern, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixAy,izwu,Xyzwxu->XA', e_extern, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixAy,izwy,Xzwx->XA', e_extern, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixAy,izyw,Xzxw->XA', e_extern, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,ixyA,iy,Xx->XA', e_extern, t1_caae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixyA,iz,Xyxz->XA', e_extern, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixyA,izwu,Xyzxwu->XA', e_extern, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,ixyA,izwy,Xzxw->XA', e_extern, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,ixyA,izyw,Xzxw->XA', e_extern, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('i,ijxy,ijxA,Xy->XA', e_core, t1_ccaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ijxy,ijyA,Xx->XA', e_core, t1_ccaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ijxy,ijzA,Xzyx->XA', e_core, t1_ccaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ijxy,jixA,Xy->XA', e_core, t1_ccaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,ijxy,jiyA,Xx->XA', e_core, t1_ccaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ijxy,jizA,Xzxy->XA', e_core, t1_ccaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ix,iA,Xx->XA', e_core, t1_ca, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ix,iyAx,Xy->XA', e_core, t1_ca, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ix,iyAz,Xzxy->XA', e_core, t1_ca, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('i,ix,iyxA,Xy->XA', e_core, t1_ca, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ix,iyzA,Xzyx->XA', e_core, t1_ca, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyz,iA,Xxyz->XA', e_core, t1_caaa, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyz,iwAu,Xxuyzw->XA', e_core, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyz,iwAy,Xxwz->XA', e_core, t1_caaa, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyz,iwAz,Xxyw->XA', e_core, t1_caaa, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyz,iwuA,Xxuwzy->XA', e_core, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('i,ixyz,iwyA,Xxwz->XA', e_core, t1_caaa, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyz,iwzA,Xxwy->XA', e_core, t1_caaa, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,ijxA,ijyz,Xz->XA', h_aa, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ijxA,jiyz,Xz->XA', h_aa, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ijxA,jizw,Xyzw->XA', h_aa, t1_ccae, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ijxz,ijwA,Xwzy->XA', h_aa, t1_ccaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ijxz,ijzA,Xy->XA', h_aa, t1_ccaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ijxz,jiwA,Xwyz->XA', h_aa, t1_ccaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ijxz,jizA,Xy->XA', h_aa, t1_ccaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ix,iA,Xy->XA', h_aa, t1_ca, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ix,izAw,Xwyz->XA', h_aa, t1_ca, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ix,izwA,Xwzy->XA', h_aa, t1_ca, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixAz,iw,Xzwy->XA', h_aa, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixAz,iwuv,Xzwuyv->XA', h_aa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixAz,iwuz,Xwuy->XA', h_aa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixAz,iwzu,Xwyu->XA', h_aa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixAz,iz,Xy->XA', h_aa, t1_caea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzA,iw,Xzyw->XA', h_aa, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzA,iwuv,Xzwyuv->XA', h_aa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzA,iwuz,Xwyu->XA', h_aa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ixzA,iwzu,Xwyu->XA', h_aa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ixzA,iz,Xy->XA', h_aa, t1_caae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzw,iA,Xyzw->XA', h_aa, t1_caaa, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzw,iuAv,Xyvzwu->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzw,iuAw,Xyzu->XA', h_aa, t1_caaa, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzw,iuAz,Xyuw->XA', h_aa, t1_caaa, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzw,iuvA,Xyvuwz->XA', h_aa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzw,iuwA,Xyuz->XA', h_aa, t1_caaa, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ixzw,iuzA,Xyuw->XA', h_aa, t1_caaa, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izAx,iw,Xywz->XA', h_aa, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izAx,iwuv,Xywuzv->XA', h_aa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izAx,iwuy,Xwuz->XA', h_aa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izAx,iwyu,Xwzu->XA', h_aa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izAx,iy,Xz->XA', h_aa, t1_caea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izwx,iA,Xzwy->XA', h_aa, t1_caaa, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izwx,iuAv,Xzvwyu->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izwx,iuAw,Xzuy->XA', h_aa, t1_caaa, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izwx,iuvA,Xzvuyw->XA', h_aa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,izwx,iuwA,Xzuy->XA', h_aa, t1_caaa, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxA,iw,Xyzw->XA', h_aa, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxA,iwuv,Xywzuv->XA', h_aa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izxA,iwuy,Xwzu->XA', h_aa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,izxA,iwyu,Xwzu->XA', h_aa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,izxA,iy,Xz->XA', h_aa, t1_caae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxw,iA,Xzyw->XA', h_aa, t1_caaa, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxw,iuAv,Xzvywu->XA', h_aa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxw,iuAw,Xzyu->XA', h_aa, t1_caaa, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxw,iuvA,Xzvuwy->XA', h_aa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxw,iuwA,Xzuy->XA', h_aa, t1_caaa, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ijxA,ijwu,Xyuz->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,ijxA,ijyu,Xwuz->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,ijxA,ijyw,Xz->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ijxA,jiuv,Xywuvz->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ijxA,jiwu,Xyzu->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ijxA,jiyu,Xwuz->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ijxA,jiyw,Xz->XA', v_aaaa, t1_ccae, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ijxu,ijuA,Xzyw->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ijxu,ijvA,Xzvuwy->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ijxu,jiuA,Xzyw->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ijxu,jivA,Xzvywu->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ijxz,jiuA,Xuyw->XA', v_aaaa, t1_ccaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuAx,iv,Xywvuz->XA', v_aaaa, t1_caea, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuAx,ivst,Xywvszut->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuAx,ivst,Xywvuszt->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuAx,ivst,Xywvuzst->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuAx,ivst,Xywvzstu->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuAx,ivst,Xywvztsu->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuAx,ivst,Xywvztus->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuAx,ivst,Xywvzuts->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuAx,ivsw,Xyvsuz->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuAx,ivsy,Xwvszu->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuAx,ivws,Xyvzus->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuAx,ivwy,Xvzu->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuAx,ivys,Xwvuzs->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuAx,ivyw,Xvuz->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuAx,iw,Xyzu->XA', v_aaaa, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuAx,iy,Xwuz->XA', v_aaaa, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuAx,izvs,Xywvus->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuvx,iA,Xzuvwy->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuvx,isAt,Xzutvyws->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuvx,isAt,Xzutwvys->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuvx,isAt,Xzutwyvs->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuvx,isAt,Xzutysvw->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuvx,isAt,Xzutyswv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuvx,isAt,Xzutyvsw->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuvx,isAt,Xzutywsv->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuvx,isAv,Xzuswy->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuvx,istA,Xzutsyvw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuvx,istA,Xzutvysw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuvx,istA,Xzutvyws->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuvx,istA,Xzutwsyv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuvx,istA,Xzutwyvs->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuvx,istA,Xzutyswv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuvx,istA,Xzutywsv->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,iuvx,isvA,Xzuswy->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuxA,iv,Xywuvz->XA', v_aaaa, t1_caae, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuxA,ivst,Xywvuszt->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxA,ivsw,Xyvusz->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxA,ivsy,Xwvuzs->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxA,ivws,Xyvuzs->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxA,ivwy,Xvuz->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,iuxA,ivys,Xwvuzs->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('xyzw,iuxA,ivyw,Xvuz->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxA,iw,Xyuz->XA', v_aaaa, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,iuxA,iy,Xwuz->XA', v_aaaa, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxA,izvs,Xywuvs->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuxv,iA,Xzuywv->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuxv,isAt,Xzutywvs->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuxv,isAv,Xzuyws->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,istA,Xzutsyvw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,istA,Xzutwsvy->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,istA,Xzutwyvs->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,istA,Xzutysvw->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,istA,Xzutywvs->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuxv,isvA,Xzuswy->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuxz,iA,Xuyw->XA', v_aaaa, t1_caaa, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuxz,ivAs,Xusywv->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuxz,ivsA,Xusvwy->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ix,iA,Xzyw->XA', v_aaaa, t1_ca, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ix,iuAv,Xzvywu->XA', v_aaaa, t1_ca, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ix,iuvA,Xzvuwy->XA', v_aaaa, t1_ca, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,iu,Xzyw->XA', v_aaaa, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,iv,Xzuvwy->XA', v_aaaa, t1_caea, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixAu,ivst,Xzuvsywt->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixAu,ivst,Xzuvwsyt->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixAu,ivst,Xzuvwyst->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,ivst,Xzuvystw->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,ivst,Xzuvytsw->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,ivst,Xzuvytws->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,ivst,Xzuvywts->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,ivsu,Xzvswy->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,ivsz,Xuvsyw->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,ivus,Xzvyws->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,ivuz,Xvyw->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,ivzs,Xuvwys->XA', v_aaaa, t1_caea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,ivzu,Xvwy->XA', v_aaaa, t1_caea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixAu,iz,Xuwy->XA', v_aaaa, t1_caea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ixuA,iu,Xzyw->XA', v_aaaa, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuA,iv,Xzuywv->XA', v_aaaa, t1_caae, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuA,ivst,Xzuvywst->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuA,ivsu,Xzvyws->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuA,ivsz,Xuvysw->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ixuA,ivus,Xzvyws->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ixuA,ivuz,Xvyw->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuA,ivzs,Xuvyws->XA', v_aaaa, t1_caae, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuA,ivzu,Xvyw->XA', v_aaaa, t1_caae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuA,iz,Xuyw->XA', v_aaaa, t1_caae, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuv,iA,Xywuvz->XA', v_aaaa, t1_caaa, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuv,isAt,Xywtuvzs->XA', v_aaaa, t1_caaa, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuv,isAu,Xywsvz->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuv,isAv,Xywusz->XA', v_aaaa, t1_caaa, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,istA,Xywtszvu->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,istA,Xywtvszu->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,istA,Xywtvzsu->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,istA,Xywtzsvu->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,istA,Xywtzvsu->XA', v_aaaa, t1_caaa, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ixuv,isuA,Xywsvz->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuv,isvA,Xywsuz->XA', v_aaaa, t1_caaa, t1_caae, rdm_cccaaa, optimize = einsum_type)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncore, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.cece [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_cece = mr_adc.v2e.cece[:, s_chunk:f_chunk]

        temp  = einsum('ijxa,iAja,Xx->XA', t1_ccae, v_cece, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('ijxa,jAia,Xx->XA', t1_ccae, v_cece, rdm_ca, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.cece", *cput1)
    del(v_cece)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.ccee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Molecular Orbitals Energies
        e_extern_A = mr_adc.mo_energy.e[s_chunk:f_chunk]

        ## Amplitudes
        t1_ccee = mr_adc.t1.ccee[:, :, s_chunk:f_chunk]

        temp  = einsum('ijAa,ixja,Xx->XA', t1_ccee, v_cace, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('ijAa,jxia,Xx->XA', t1_ccee, v_cace, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('A,ijAa,ijxa,Xx->XA', e_extern_A, t1_ccee, t1_ccae, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('A,ijAa,jixa,Xx->XA', e_extern_A, t1_ccee, t1_ccae, rdm_ca, optimize = einsum_type)
        temp += einsum('a,ijxa,ijAa,Xx->XA', e_extern, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('a,ijxa,jiAa,Xx->XA', e_extern, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= einsum('i,ijxa,ijAa,Xx->XA', e_core, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('i,ijxa,jiAa,Xx->XA', e_core, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('i,jixa,ijAa,Xx->XA', e_core, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= einsum('i,jixa,jiAa,Xx->XA', e_core, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xy,ijxa,ijAa,Xy->XA', h_aa, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,ijxa,jiAa,Xy->XA', h_aa, t1_ccae, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,ijxa,ijAa,Xzyw->XA', v_aaaa, t1_ccae, t1_ccee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,ijxa,jiAa,Xzyw->XA', v_aaaa, t1_ccae, t1_ccee, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting t1.ccee", *cput1)
    del(t1_ccee, e_extern_A)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_caee = mr_adc.v2e.caee[:, :, s_chunk:f_chunk]

        temp  = 1/2 * einsum('ia,ixAa,Xx->XA', t1_ce, v_caee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('ixay,iyAa,Xx->XA', t1_caea, v_caee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('ixay,izAa,Xyzx->XA', t1_caea, v_caee, rdm_ccaa, optimize = einsum_type)
        temp -= einsum('ixya,iyAa,Xx->XA', t1_caae, v_caee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('ixya,izAa,Xyxz->XA', t1_caae, v_caee, rdm_ccaa, optimize = einsum_type)
  
        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.caee", *cput1)
    del(v_caee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceae [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_ceae = mr_adc.v2e.ceae[:, s_chunk:f_chunk, :, :]

        temp  = 1/2 * einsum('ia,iAxa,Xx->XA', t1_ce, v_ceae, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('ixay,iAza,Xxzy->XA', t1_caea, v_ceae, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('ixya,iAza,Xxyz->XA', t1_caae, v_ceae, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.ceae", *cput1)
    del(v_ceae)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceae [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_ceae = mr_adc.v2e.ceae[:, :, :, s_chunk:f_chunk]

        temp =- einsum('ia,iaxA,Xx->XA', t1_ce, v_ceae, rdm_ca, optimize = einsum_type)
        temp -= einsum('ixay,iazA,Xxzy->XA', t1_caea, v_ceae, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('ixya,iazA,Xxzy->XA', t1_caae, v_ceae, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.ceae", *cput1)
    del(v_ceae)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceea [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_ceea = mr_adc.v2e.ceea[:, :, s_chunk:f_chunk]

        temp =- einsum('ia,iaAx,Xx->XA', t1_ce, v_ceea, rdm_ca, optimize = einsum_type)
        temp -= einsum('ixay,iaAy,Xx->XA', t1_caea, v_ceea, rdm_ca, optimize = einsum_type)
        temp -= einsum('ixay,iaAz,Xyzx->XA', t1_caea, v_ceea, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('ixya,iaAy,Xx->XA', t1_caae, v_ceea, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('ixya,iaAz,Xyzx->XA', t1_caae, v_ceea, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.ceea", *cput1)
    del(v_ceea)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncas, nextern], ntensors = 3)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Molecular Orbitals Energies
        e_extern_A = mr_adc.mo_energy.e[s_chunk:f_chunk]

        ## Amplitudes
        t1_caee = mr_adc.t1.caee[:, :, s_chunk:f_chunk, :]

        temp  = 1/2 * einsum('ia,ixAa,Xx->XA', h_ce, t1_caee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('ixAa,iayz,Xyxz->XA', t1_caee, v_ceaa, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('ixAa,iyza,Xzyx->XA', t1_caee, v_caae, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('A,ixAa,ia,Xx->XA', e_extern_A, t1_caee, t1_ce, rdm_ca, optimize = einsum_type)
        temp += 1/4 * einsum('A,ixAa,iyaz,Xyxz->XA', e_extern_A, t1_caee, t1_caea, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('A,ixAa,iyza,Xyzx->XA', e_extern_A, t1_caee, t1_caae, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('a,ixAa,ia,Xx->XA', e_extern, t1_caee, t1_ce, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('a,ixay,izAa,Xxzy->XA', e_extern, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('a,ixya,izAa,Xxyz->XA', e_extern, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('i,ixAa,ia,Xx->XA', e_core, t1_caee, t1_ce, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('i,ixay,izAa,Xxzy->XA', e_core, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('i,ixya,izAa,Xxyz->XA', e_core, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,ixAa,ia,Xy->XA', h_aa, t1_caee, t1_ce, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,ixAa,izaw,Xzyw->XA', h_aa, t1_caee, t1_caea, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,ixAa,izwa,Xzwy->XA', h_aa, t1_caee, t1_caae, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,ixaz,iwAa,Xywz->XA', h_aa, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,ixza,iwAa,Xyzw->XA', h_aa, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('xy,izax,iwAa,Xzwy->XA', h_aa, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('xy,izxa,iwAa,Xzyw->XA', h_aa, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,iuax,ivAa,Xzuvwy->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,iuxa,ivAa,Xzuywv->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,ixAa,ia,Xzyw->XA', v_aaaa, t1_caee, t1_ce, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,ixAa,iuav,Xzuywv->XA', v_aaaa, t1_caee, t1_caea, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,ixAa,iuaz,Xuyw->XA', v_aaaa, t1_caee, t1_caea, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,ixAa,iuva,Xzuvwy->XA', v_aaaa, t1_caee, t1_caae, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,ixAa,iuza,Xuwy->XA', v_aaaa, t1_caee, t1_caae, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,ixau,ivAa,Xywvuz->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,ixua,ivAa,Xywuvz->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)

        ## Amplitudes
        t1_caee = mr_adc.t1.caee[:, :, :, s_chunk:f_chunk]

        temp -= einsum('ia,ixaA,Xx->XA', h_ce, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= einsum('ixaA,iayz,Xyxz->XA', t1_caee, v_ceaa, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('ixaA,iyza,Xzxy->XA', t1_caee, v_caae, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('A,ixaA,ia,Xx->XA', e_extern_A, t1_caee, t1_ce, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('A,ixaA,iyaz,Xyxz->XA', e_extern_A, t1_caee, t1_caea, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('A,ixaA,iyza,Xyxz->XA', e_extern_A, t1_caee, t1_caae, rdm_ccaa, optimize = einsum_type)
        temp -= einsum('a,ixaA,ia,Xx->XA', e_extern, t1_caee, t1_ce, rdm_ca, optimize = einsum_type)
        temp -= einsum('a,ixay,izaA,Xxzy->XA', e_extern, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('a,ixya,izaA,Xxzy->XA', e_extern, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += einsum('i,ixaA,ia,Xx->XA', e_core, t1_caee, t1_ce, rdm_ca, optimize = einsum_type)
        temp += einsum('i,ixay,izaA,Xxzy->XA', e_core, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('i,ixya,izaA,Xxzy->XA', e_core, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xy,ixaA,ia,Xy->XA', h_aa, t1_caee, t1_ce, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xy,ixaA,izaw,Xzyw->XA', h_aa, t1_caee, t1_caea, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,ixaA,izwa,Xzyw->XA', h_aa, t1_caee, t1_caae, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xy,ixaz,iwaA,Xywz->XA', h_aa, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,ixza,iwaA,Xywz->XA', h_aa, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xy,izax,iwaA,Xzwy->XA', h_aa, t1_caea, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('xy,izxa,iwaA,Xzwy->XA', h_aa, t1_caae, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xyzw,iuax,ivaA,Xzuvwy->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,iuxa,ivaA,Xzuvwy->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,ixaA,ia,Xzyw->XA', v_aaaa, t1_caee, t1_ce, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,ixaA,iuav,Xzuywv->XA', v_aaaa, t1_caee, t1_caea, rdm_cccaaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,ixaA,iuaz,Xuyw->XA', v_aaaa, t1_caee, t1_caea, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,ixaA,iuva,Xzuywv->XA', v_aaaa, t1_caee, t1_caae, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,ixaA,iuza,Xuyw->XA', v_aaaa, t1_caee, t1_caae, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,ixau,ivaA,Xywvuz->XA', v_aaaa, t1_caea, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,ixua,ivaA,Xywvuz->XA', v_aaaa, t1_caae, t1_caee, rdm_cccaaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting t1.caee", *cput1)
    del(t1_caee, e_extern_A)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_aaee = mr_adc.v2e.aaee[:, :, s_chunk:f_chunk]

        temp =- 1/2 * einsum('xa,yzAa,Xyxz->XA', t1_ae, v_aaee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xyza,wuAa,Xzwyxu->XA', t1_aaae, v_aaee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xyza,wzAa,Xwyx->XA', t1_aaae, v_aaee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xa,yzAa,yz,Xx->XA', t1_ae, v_aaee, rdm_ca, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xyza,wuAa,wu,Xzyx->XA', t1_aaae, v_aaee, rdm_ca, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.aaee", *cput1)
    del(v_aaee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.aeae [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_aeae = mr_adc.v2e.aeae[:, s_chunk:f_chunk]

        temp =- 1/2 * einsum('xa,yAza,Xxyz->XA', t1_ae, v_aeae, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xyza,wAua,Xyxwuz->XA', t1_aaae, v_aeae, rdm_cccaaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.aeae", *cput1)
    del(v_aeae)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.aeea [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integrals
        v_aeea = mr_adc.v2e.aeea[:, s_chunk:f_chunk]

        temp =- 1/2 * einsum('xa,yAaz,Xzyx->XA', t1_ae, v_aeea, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xyza,wAau,Xzuwxy->XA', t1_aaae, v_aeea, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xyza,zAaw,Xwxy->XA', t1_aaae, v_aeea, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xa,yAaz,zy,Xx->XA', t1_ae, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xyza,wAau,uw,Xzyx->XA', t1_aaae, v_aeea, rdm_ca, rdm_ccaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting v2e.aeea", *cput1)
    del(v_aeea)

    chunks = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Molecular Orbitals Energies
        e_extern_A = mr_adc.mo_energy.e[s_chunk:f_chunk]

        ## Amplitudes
        t1_aaee = mr_adc.t1.aaee[:, :, s_chunk:f_chunk]

        temp =- 1/2 * einsum('xa,yzAa,Xxyz->XA', h_ae, t1_aaee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('xyAa,zwua,Xuzxyw->XA', t1_aaee, v_aaae, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('A,xyAa,za,Xzxy->XA', e_extern_A, t1_aaee, t1_ae, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('A,xyAa,zwua,Xwzxyu->XA', e_extern_A, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/2 * einsum('a,xyAa,za,Xzxy->XA', e_extern, t1_aaee, t1_ae, rdm_ccaa, optimize = einsum_type)
        temp -= 1/2 * einsum('a,xyza,wuAa,Xyxwuz->XA', e_extern, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xy,xa,zwAa,Xyzw->XA', h_aa, t1_ae, t1_aaee, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('xy,xzAa,wa,Xwyz->XA', h_aa, t1_aaee, t1_ae, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('xy,xzAa,wuva,Xuwyzv->XA', h_aa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xy,xzwa,uvAa,Xyzuwv->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,zwxa,uvAa,Xzwuyv->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xy,zxAa,wa,Xwzy->XA', h_aa, t1_aaee, t1_ae, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('xy,zxAa,wuva,Xuwzyv->XA', h_aa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xy,zxwa,uvAa,Xyzuvw->XA', h_aa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,uvxa,stAa,Xzuvsywt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/12 * einsum('xyzw,uvxa,stAa,Xzuvwsty->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/6 * einsum('xyzw,uvxa,stAa,Xzuvwsyt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/12 * einsum('xyzw,uvxa,stAa,Xzuvwtsy->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/12 * einsum('xyzw,uvxa,stAa,Xzuvwtys->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/6 * einsum('xyzw,uvxa,stAa,Xzuvwyst->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/12 * einsum('xyzw,uvxa,stAa,Xzuvwyts->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,uvxa,stAa,Xzuvyswt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,uvxa,stAa,Xzuvywst->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,uxAa,va,Xzvuwy->XA', v_aaaa, t1_aaee, t1_ae, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,uxAa,vsta,Xzsvuywt->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,uxAa,vsta,Xzsvwuyt->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,uxAa,vsta,Xzsvwyut->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,uxAa,vsta,Xzsvytuw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,uxAa,vsta,Xzsvytwu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,uxAa,vsta,Xzsvyutw->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,uxAa,vsta,Xzsvywtu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,uxAa,vsza,Xvsuwy->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,uxva,stAa,Xywustzv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,uxva,stAa,Xywusztv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,uxva,stAa,Xywutszv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,uxva,stAa,Xywutzsv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,uxva,stAa,Xywuzstv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,uxva,stAa,Xywuztsv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,xa,uvAa,Xywuvz->XA', v_aaaa, t1_ae, t1_aaee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,xuAa,va,Xzvywu->XA', v_aaaa, t1_aaee, t1_ae, rdm_cccaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,xuAa,vsta,Xzsvywut->XA', v_aaaa, t1_aaee, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,xuAa,vsza,Xvsywu->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp += 1/6 * einsum('xyzw,xuva,stAa,Xywusztv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/12 * einsum('xyzw,xuva,stAa,Xywuszvt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/6 * einsum('xyzw,xuva,stAa,Xywutzsv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/6 * einsum('xyzw,xuva,stAa,Xywutzvs->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,xuva,stAa,Xywuvszt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/12 * einsum('xyzw,xuva,stAa,Xywuvzst->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/6 * einsum('xyzw,xuva,stAa,Xywuvzts->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,xuva,stAa,Xywuzstv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,xuva,stAa,Xywuzsvt->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,xuva,stAa,Xywuztsv->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,xuva,stAa,Xywuztvs->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp -= 1/6 * einsum('xyzw,xuva,stAa,Xywuzvst->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/12 * einsum('xyzw,xuva,stAa,Xywuzvts->XA', v_aaaa, t1_aaae, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,xzAa,ua,Xuyw->XA', v_aaaa, t1_aaee, t1_ae, rdm_ccaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,xzAa,uvsa,Xvuyws->XA', v_aaaa, t1_aaee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,zxua,vsAa,Xywvsu->XA', v_aaaa, t1_aaae, t1_aaee, rdm_cccaaa, optimize = einsum_type)

        V1[:, s_chunk:f_chunk] += temp
        mr_adc.log.timer_debug("contracting t1.aaee", *cput1)
    del(t1_aaee, e_extern_A)

    chunks = tools.calculate_double_chunks(mr_adc, ncore, [nextern, nextern, nextern],
                                                                     [ncas, nextern, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        if mr_adc.interface.with_df:
            v_ceee = integrals.get_oeee_df(mr_adc, mr_adc.v2e.Lce, mr_adc.v2e.Lee, s_chunk, f_chunk)

        else:
            v_ceee = integrals.unpack_v2e_oeee(mr_adc, mr_adc.v2e.ceee[s_chunk:f_chunk])

        ## Amplitudes
        t1_caee = mr_adc.t1.caee[s_chunk:f_chunk]

        temp =- einsum('ixab,iaAb,Xx->XA', t1_caee, v_ceee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('ixab,ibAa,Xx->XA', t1_caee, v_ceee, rdm_ca, optimize = einsum_type)

        V1 += temp
        mr_adc.log.timer_debug("contracting v2e.ceee", *cput1)
    del(v_ceee, t1_caee)

    chunks = tools.calculate_double_chunks(mr_adc, ncas, [nextern, nextern, nextern],
                                                                    [ncas, nextern, nextern], ntensors = 2)
    for i_v_chunk, (s_v_chunk, f_v_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.aeee [%i/%i], chunk [%i:%i]", i_v_chunk + 1, len(chunks), s_v_chunk, f_v_chunk)

        if mr_adc.interface.with_df:
            v_aeee = integrals.get_oeee_df(mr_adc, mr_adc.v2e.Lae, mr_adc.v2e.Lee, s_v_chunk, f_v_chunk)
        else:
            v_aeee = integrals.unpack_v2e_oeee(mr_adc, mr_adc.v2e.aeee[s_v_chunk:f_v_chunk])

        for s_t_chunk, f_t_chunk in chunks:

            ## Amplitudes
            t1_aaee = mr_adc.t1.aaee[s_t_chunk:f_t_chunk]

            ## Reduced density matrices
            rdm_ccaa = mr_adc.rdm.ccaa[:, s_v_chunk:f_v_chunk, s_t_chunk:f_t_chunk]

            V1 -= 1/2 * einsum('xyab,zbAa,Xzxy->XA', t1_aaee, v_aeee, rdm_ccaa, optimize = einsum_type)

        mr_adc.log.timer_debug("contracting v2e.aeee", *cput1)
    del(v_aeee, t1_aaee, rdm_ccaa)

    # Compute denominators
    d_pa = (evals[:,None] + e_extern)
    d_pa = d_pa**(-1)

    # Compute T[-1']^(2) amplitudes
    S_12_V_m1 = einsum('mp,Pa,Pm->pa', evecs, V1, S_m1_12_inv_act, optimize = einsum_type)
    S_12_V_m1 *= d_pa
    S_12_V_m1 = einsum('mp,pa->ma', evecs, S_12_V_m1, optimize = einsum_type)

    # Compute T[-1']^(2) t2_ae tensor
    t2_ae = einsum('Pm,ma->Pa', S_m1_12_inv_act, S_12_V_m1, optimize = einsum_type)

    mr_adc.log.extra("Norm of T[-1']^(2):                          %20.12f" % np.linalg.norm(t2_ae))
    mr_adc.log.timer("computing T[-1']^(2) amplitudes", *cput0)

    return t2_ae

def compute_t2_p1p_singles(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing T[+1']^(2) amplitudes...")

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
    v_ccca = mr_adc.v2e.ccca
    v_ccce = mr_adc.v2e.ccce
    v_ccaa = mr_adc.v2e.ccaa
    v_ccae = mr_adc.v2e.ccae
    v_caca = mr_adc.v2e.caca
    v_cace = mr_adc.v2e.cace
    v_caac = mr_adc.v2e.caac
    v_caaa = mr_adc.v2e.caaa
    v_caae = mr_adc.v2e.caae
    v_caec = mr_adc.v2e.caec
    v_caea = mr_adc.v2e.caea
    v_ceaa = mr_adc.v2e.ceaa
    v_aaaa = mr_adc.v2e.aaaa
    v_aaae = mr_adc.v2e.aaae

    ## Amplitudes
    t1_ca   = mr_adc.t1.ca
    t1_ce   = mr_adc.t1.ce
    t1_ae   = mr_adc.t1.ae
    t1_ccaa = mr_adc.t1.ccaa
    t1_ccae = mr_adc.t1.ccae
    t1_caaa = mr_adc.t1.caaa
    t1_caae = mr_adc.t1.caae
    t1_caea = mr_adc.t1.caea
    t1_aaae = mr_adc.t1.aaae

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa
    rdm_ccccaaaa = mr_adc.rdm.ccccaaaa

    # Compute K_ac matrix
    K_p1p = intermediates.compute_K_ac(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_p1p_12_inv_act = overlap.compute_S12_p1(mr_adc)

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
    V1 -= 2 * einsum('Iixa,iaXx->IX', t1_ccae, v_ceaa, optimize = einsum_type)
    V1 += einsum('Iixa,ixXa->IX', t1_ccae, v_caae, optimize = einsum_type)
    V1 += einsum('Iixy,ixXy->IX', t1_ccaa, v_caaa, optimize = einsum_type)
    V1 -= 2 * einsum('Iixy,iyXx->IX', t1_ccaa, v_caaa, optimize = einsum_type)
    V1 += einsum('iIxa,iaXx->IX', t1_ccae, v_ceaa, optimize = einsum_type)
    V1 -= 2 * einsum('iIxa,ixXa->IX', t1_ccae, v_caae, optimize = einsum_type)
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
    V1 -= 1/2 * einsum('I,IixX,ix->IX', e_core, t1_ccaa, t1_ca, optimize = einsum_type)
    V1 += einsum('I,Iixa,iXax->IX', e_core, t1_ccae, t1_caea, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iixa,iXxa->IX', e_core, t1_ccae, t1_caae, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,Iixy,iXxy->IX', e_core, t1_ccaa, t1_caaa, optimize = einsum_type)
    V1 += einsum('I,Iixy,iXyx->IX', e_core, t1_ccaa, t1_caaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,iIXa,ia->IX', e_core, t1_ccae, t1_ce, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,iIxa,iXax->IX', e_core, t1_ccae, t1_caea, optimize = einsum_type)
    V1 += einsum('I,iIxa,iXxa->IX', e_core, t1_ccae, t1_caae, optimize = einsum_type)
    V1 -= einsum('a,Ia,Xa->IX', e_extern, t1_ce, t1_ae, optimize = einsum_type)
    V1 -= 2 * einsum('a,IiXa,ia->IX', e_extern, t1_ccae, t1_ce, optimize = einsum_type)
    V1 -= 2 * einsum('a,Iixa,iXax->IX', e_extern, t1_ccae, t1_caea, optimize = einsum_type)
    V1 += einsum('a,Iixa,iXxa->IX', e_extern, t1_ccae, t1_caae, optimize = einsum_type)
    V1 += einsum('a,iIXa,ia->IX', e_extern, t1_ccae, t1_ce, optimize = einsum_type)
    V1 += einsum('a,iIxa,iXax->IX', e_extern, t1_ccae, t1_caea, optimize = einsum_type)
    V1 -= 2 * einsum('a,iIxa,iXxa->IX', e_extern, t1_ccae, t1_caae, optimize = einsum_type)
    V1 += 2 * einsum('i,IiXa,ia->IX', e_core, t1_ccae, t1_ce, optimize = einsum_type)
    V1 += 2 * einsum('i,IiXx,ix->IX', e_core, t1_ccaa, t1_ca, optimize = einsum_type)
    V1 -= einsum('i,IixX,ix->IX', e_core, t1_ccaa, t1_ca, optimize = einsum_type)
    V1 += 2 * einsum('i,Iixa,iXax->IX', e_core, t1_ccae, t1_caea, optimize = einsum_type)
    V1 -= einsum('i,Iixa,iXxa->IX', e_core, t1_ccae, t1_caae, optimize = einsum_type)
    V1 -= einsum('i,Iixy,iXxy->IX', e_core, t1_ccaa, t1_caaa, optimize = einsum_type)
    V1 += 2 * einsum('i,Iixy,iXyx->IX', e_core, t1_ccaa, t1_caaa, optimize = einsum_type)
    V1 -= einsum('i,iIXa,ia->IX', e_core, t1_ccae, t1_ce, optimize = einsum_type)
    V1 -= einsum('i,iIxa,iXax->IX', e_core, t1_ccae, t1_caea, optimize = einsum_type)
    V1 += 2 * einsum('i,iIxa,iXxa->IX', e_core, t1_ccae, t1_caae, optimize = einsum_type)
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
    V1 += 1/2 * einsum('Ixay,Xyza,xz->IX', t1_caea, v_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixay,Xzwa,xzyw->IX', t1_caea, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('Ixay,zwXa,xwyz->IX', t1_caea, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixay,zwua,Xxwuyz->IX', t1_caea, v_aaae, rdm_cccaaa, optimize = einsum_type)
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
    V1 -= 1/2 * einsum('ixyz,Iiwu,Xwxyuz->IX', t1_caaa, v_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('ixyz,IiyX,zx->IX', t1_caaa, v_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,Iiyw,Xxwz->IX', t1_caaa, v_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,IizX,yx->IX', t1_caaa, v_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,Iizw,Xxyw->IX', t1_caaa, v_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,IwiX,xwzy->IX', t1_caaa, v_caca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,Iwiu,Xyzwux->IX', t1_caaa, v_caca, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('ixyz,Iwiy,Xzwx->IX', t1_caaa, v_caca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,Iwiz,Xywx->IX', t1_caaa, v_caca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyz,Iwui,Xuxwyz->IX', t1_caaa, v_caac, rdm_cccaaa, optimize = einsum_type)
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
    V1 -= einsum('xyza,IXaw,zwxy->IX', t1_aaae, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyza,IXwa,zwxy->IX', t1_aaae, v_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,IawX,zwxy->IX', t1_aaae, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,Iawu,Xwzyux->IX', t1_aaae, v_ceaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,Iawz,Xwyx->IX', t1_aaae, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,IwaX,zwxy->IX', t1_aaae, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,Iwau,Xyxwuz->IX', t1_aaae, v_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,Iwua,Xuzwyx->IX', t1_aaae, v_caae, rdm_cccaaa, optimize = einsum_type)
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
    V1 += 1/2 * einsum('I,Ixay,Xa,xy->IX', e_core, t1_caea, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixay,Xzwa,xwyz->IX', e_core, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixay,Xzya,xz->IX', e_core, t1_caea, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,Ixay,zXwa,xwyz->IX', e_core, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('I,Ixay,zXya,xz->IX', e_core, t1_caea, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixay,za,Xxzy->IX', e_core, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,Ixay,zwua,Xxuwyz->IX', e_core, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
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
    V1 += 1/2 * einsum('a,Ia,Xxya,xy->IX', e_extern, t1_ce, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('a,Ia,xXya,xy->IX', e_extern, t1_ce, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Ia,xa,Xx->IX', e_extern, t1_ce, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Ia,xyza,Xzyx->IX', e_extern, t1_ce, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('a,IiXa,ixay,yx->IX', e_extern, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,IiXa,ixya,yx->IX', e_extern, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,Iixa,iXay,xy->IX', e_extern, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,Iixa,iXya,xy->IX', e_extern, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,Iixa,ia,Xx->IX', e_extern, t1_ccae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,Iixa,iyax,Xy->IX', e_extern, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,Iixa,iyaz,Xzxy->IX', e_extern, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,Iixa,iyxa,Xy->IX', e_extern, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,Iixa,iyza,Xzxy->IX', e_extern, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('a,IxXa,ya,xy->IX', e_extern, t1_caae, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('a,IxXa,yzwa,xwzy->IX', e_extern, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,IxaX,ya,xy->IX', e_extern, t1_caea, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,IxaX,yzwa,xwzy->IX', e_extern, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('a,Ixay,Xa,xy->IX', e_extern, t1_caea, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Ixay,Xzwa,xwyz->IX', e_extern, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Ixay,Xzya,xz->IX', e_extern, t1_caea, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('a,Ixay,zXwa,xwyz->IX', e_extern, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('a,Ixay,zXya,xz->IX', e_extern, t1_caea, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Ixay,za,Xxzy->IX', e_extern, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Ixay,zwya,Xxwz->IX', e_extern, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Ixya,Xa,xy->IX', e_extern, t1_caae, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Ixya,Xzwa,xwzy->IX', e_extern, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('a,Ixya,Xzya,xz->IX', e_extern, t1_caae, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Ixya,zXwa,xwyz->IX', e_extern, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Ixya,zXya,xz->IX', e_extern, t1_caae, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Ixya,za,Xxyz->IX', e_extern, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,Ixya,zwya,Xxzw->IX', e_extern, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('a,iIXa,ixay,yx->IX', e_extern, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,iIXa,ixya,yx->IX', e_extern, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,iIxa,iXay,xy->IX', e_extern, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,iIxa,iXya,xy->IX', e_extern, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,iIxa,ia,Xx->IX', e_extern, t1_ccae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,iIxa,iyax,Xy->IX', e_extern, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,iIxa,iyaz,Xzxy->IX', e_extern, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('a,iIxa,iyxa,Xy->IX', e_extern, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,iIxa,iyza,Xzyx->IX', e_extern, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xyza,Iwau,Xwzyux->IX', e_extern, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xyza,Iwua,Xwzuyx->IX', e_extern, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 2 * einsum('i,IiXa,ixay,xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,IiXa,ixya,xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,IiXx,iy,xy->IX', e_core, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 2 * einsum('i,IiXx,iyxz,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,IiXx,iyzw,xyzw->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('i,IiXx,iyzx,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,IixX,iy,xy->IX', e_core, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,IixX,iyxz,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,IixX,iyzw,xyzw->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,IixX,iyzx,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,Iixa,iXay,xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixa,iXya,xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,Iixa,ia,Xx->IX', e_core, t1_ccae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,Iixa,iyax,Xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,Iixa,iyaz,Xzxy->IX', e_core, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixa,iyxa,Xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixa,iyza,Xzxy->IX', e_core, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixy,iXxz,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,Iixy,iXyz,xz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixy,iXzw,xywz->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('i,Iixy,iXzx,yz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixy,iXzy,xz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixy,ix,Xy->IX', e_core, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,Iixy,iy,Xx->IX', e_core, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixy,iz,Xzxy->IX', e_core, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixy,izwx,Xwzy->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixy,izwy,Xwxz->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixy,izxw,Xwyz->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,Iixy,izxy,Xz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,Iixy,izyw,Xwxz->IX', e_core, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('i,Iixy,izyx,Xz->IX', e_core, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,iIXa,ixay,xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,iIXa,ixya,xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,iIxa,iXay,xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,iIxa,iXya,xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,iIxa,ia,Xx->IX', e_core, t1_ccae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,iIxa,iyax,Xy->IX', e_core, t1_ccae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,iIxa,iyaz,Xzxy->IX', e_core, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('i,iIxa,iyxa,Xy->IX', e_core, t1_ccae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,iIxa,iyza,Xzyx->IX', e_core, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ixyz,Iiwu,Xyzwux->IX', e_core, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
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
    V1 -= 1/4 * einsum('xy,Iizx,iwuv,Xuvzyw->IX', h_aa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
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
    V1 += 1/2 * einsum('xy,Ixaz,Xa,yz->IX', h_aa, t1_caea, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixaz,Xwua,yuzw->IX', h_aa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixaz,Xwza,yw->IX', h_aa, t1_caea, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Ixaz,wXua,yuzw->IX', h_aa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Ixaz,wXza,yw->IX', h_aa, t1_caea, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixaz,wa,Xywz->IX', h_aa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixaz,wuva,Xvyuwz->IX', h_aa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixaz,wuza,Xyuw->IX', h_aa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixza,Xa,yz->IX', h_aa, t1_caae, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixza,Xwua,yuwz->IX', h_aa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Ixza,Xwza,yw->IX', h_aa, t1_caae, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixza,wXua,yuzw->IX', h_aa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixza,wXza,yw->IX', h_aa, t1_caae, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixza,wa,Xyzw->IX', h_aa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixza,wuva,Xvyzwu->IX', h_aa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixza,wuza,Xywu->IX', h_aa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Izax,Xa,yz->IX', h_aa, t1_caea, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izax,Xwua,ywzu->IX', h_aa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Izax,Xwya,zw->IX', h_aa, t1_caea, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Izax,wXua,ywzu->IX', h_aa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,Izax,wXya,zw->IX', h_aa, t1_caea, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izax,wa,Xzwy->IX', h_aa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izax,wuva,Xzvuyw->IX', h_aa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Izax,wuya,Xzuw->IX', h_aa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izxa,Xa,yz->IX', h_aa, t1_caae, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izxa,Xwua,ywuz->IX', h_aa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,Izxa,Xwya,zw->IX', h_aa, t1_caae, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izxa,wXua,ywzu->IX', h_aa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Izxa,wXya,zw->IX', h_aa, t1_caae, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izxa,wa,Xzyw->IX', h_aa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izxa,wuva,Xzvyuw->IX', h_aa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Izxa,wuya,Xzwu->IX', h_aa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
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
    V1 -= 1/4 * einsum('xy,izwx,Iiuv,Xywuzv->IX', h_aa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
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
    V1 -= 1/4 * einsum('xy,xzwa,Ia,Xwzy->IX', h_aa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xzwa,IuXa,yzwu->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzwa,IuaX,yzwu->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzwa,Iuav,Xwuzyv->IX', h_aa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzwa,Iuaw,Xuzy->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzwa,Iuva,Xwuvyz->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzwa,Iuwa,Xuyz->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zXxa,Ia,yz->IX', h_aa, t1_aaae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zXxa,Iwau,ywzu->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zXxa,Iwua,ywzu->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxa,Ia,Xywz->IX', h_aa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zwxa,IuXa,yuzw->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxa,IuaX,yuzw->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxa,Iuav,Xuywvz->IX', h_aa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxa,Iuva,Xuyvwz->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwa,Ia,Xwyz->IX', h_aa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,zxwa,IuXa,yzuw->IX', h_aa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwa,IuaX,yzuw->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwa,Iuav,Xwuyzv->IX', h_aa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwa,Iuaw,Xuyz->IX', h_aa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwa,Iuva,Xwuvzy->IX', h_aa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
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
    V1 -= 1/4 * einsum('Xxyz,Iiwz,iuvs,ywuvxs->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
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
    V1 += 1/4 * einsum('Xxyz,Iwaz,uvsa,yvuwxs->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,Iwxa,ua,yuzw->IX', v_aaaa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,Iwxa,uvsa,yvuzws->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iwxa,uzva,yuwv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iwxa,za,yw->IX', v_aaaa, t1_caae, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,Iwxa,zuva,yuvw->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,Iwza,ua,yuxw->IX', v_aaaa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,Iwza,uvsa,yvuxws->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iyaw,ua,xzuw->IX', v_aaaa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iyaw,uvsa,wvuzxs->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iyaw,uvwa,xzvu->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iywa,ua,xzwu->IX', v_aaaa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,Iywa,uvsa,wvuxzs->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
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
    V1 -= 1/4 * einsum('Xxyz,iwuy,Iivs,uxzsvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
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
    V1 -= 1/4 * einsum('Xxyz,iwyu,Iivs,uxzwvs->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,iwyu,Iivu,xzvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iwyu,Iivx,zuvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,iwyu,Iivz,xuvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,iwyu,Iixu,zw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Xxyz,iwyu,Iixv,zuvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('Xxyz,iwyu,Iixz,uw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iwyu,Iizu,xw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Xxyz,iwyu,Iizv,xuvw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Xxyz,iwyu,Iizx,uw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ca, optimize = einsum_type)
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
    V1 += 1/4 * einsum('Xxyz,ixwu,Iivs,yvszuw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
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
    V1 += 1/4 * einsum('Xxyz,wuya,Ivas,wuszxv->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,wuya,Ivax,zvwu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,wuya,Ivaz,xvuw->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('Xxyz,wuya,Ivsa,wuszvx->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
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
    V1 -= 1/4 * einsum('Xxyz,wzua,Ivsa,ywsvux->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,wzua,Ivua,ywvx->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,xa,Ia,yz->IX', v_aaaa, t1_ae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,xa,Iwau,yuzw->IX', v_aaaa, t1_ae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,xa,Iwaz,yw->IX', v_aaaa, t1_ae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xa,Iwua,yuzw->IX', v_aaaa, t1_ae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xa,Iwza,yw->IX', v_aaaa, t1_ae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xwua,Ia,ywzu->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xwua,Ivas,ywszuv->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xwua,Ivau,ywzv->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xwua,Ivaz,ywvu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xwua,Ivsa,ywszvu->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,xwua,Ivua,ywzv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xwua,Ivza,ywuv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xzwa,Ia,yw->IX', v_aaaa, t1_aaae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xzwa,Iuav,yvwu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xzwa,Iuaw,yu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,xzwa,Iuva,yvuw->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,xzwa,Iuwa,yu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,za,Ia,yx->IX', v_aaaa, t1_ae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,za,Iwau,yuxw->IX', v_aaaa, t1_ae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,za,Iwua,yuwx->IX', v_aaaa, t1_ae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zwua,Ia,ywux->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zwua,Ivas,ywsuxv->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zwua,Ivau,ywvx->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zwua,Ivsa,ywsuvx->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zwua,Ivua,ywxv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,zxwa,Ia,yw->IX', v_aaaa, t1_aaae, t1_ce, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,zxwa,Iuav,yvwu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Xxyz,zxwa,Iuaw,yu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zxwa,Iuva,yvwu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('Xxyz,zxwa,Iuwa,yu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,IiXx,iu,zuwy->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,IiXx,iuvs,zvswyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
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
    V1 -= 1/4 * einsum('xyzw,Iiux,iXvs,zsvwuy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
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
    V1 -= 1/4 * einsum('xyzw,Iiux,iv,Xvzuyw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iiux,ivst,Xstzuwvy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iiux,ivst,Xstzwuvy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iiux,ivst,Xstzwyvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iiux,ivst,Xstzyuwv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iiux,ivst,Xstzyvuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iiux,ivst,Xstzyvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iiux,ivst,Xstzywuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iiux,ivsu,Xzsvwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,ivsw,Xzsuvy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,ivsy,Xzsuwv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iiux,ivus,Xzsywv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,ivuw,Xzyv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,ivuy,Xzvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,ivws,Xzsuyv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,ivwu,Xzvy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,ivwy,Xzuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iiux,ivys,Xzsuwv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iiux,ivyu,Xzvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iiux,ivyw,Xzuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iiux,iw,Xzuy->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iiux,iy,Xzuw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iiux,izuv,Xvyw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iiux,izvs,Xsvuwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iiux,izvu,Xvwy->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,IixX,iu,zuwy->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,IixX,iuvs,zvswyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
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
    V1 -= 1/4 * einsum('xyzw,Iixu,iv,Xvzyuw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iixu,ivst,Xstzyuvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iixu,ivsu,Xzsywv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,ivsw,Xzsyvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,ivsy,Xzsvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iixu,ivus,Xzsywv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixu,ivuw,Xzyv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixu,ivuy,Xzvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,ivws,Xzsyuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,ivwu,Xzyv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,ivwy,Xzvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,ivys,Xzsuwv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,ivyu,Xzvw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,ivyw,Xzuv->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,iw,Xzyu->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,iy,Xzuw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixu,izuv,Xvyw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iixu,izvs,Xsvywu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iixu,izvu,Xvyw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iixz,iXuv,ywvu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixz,iXuw,yu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,Iixz,iXuy,wu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,Iixz,iXwu,yu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixz,iXyu,wu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iixz,iu,Xuyw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iixz,iuvs,Xvsywu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixz,iuvw,Xvyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixz,iuvy,Xvuw->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixz,iuwv,Xvyu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,Iixz,iuwy,Xu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixz,iuyv,Xvwu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixz,iuyw,Xu->IX', v_aaaa, t1_ccaa, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,Iixz,iw,Xy->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iixz,iy,Xw->IX', v_aaaa, t1_ccaa, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iuax,Xa,zuwy->IX', v_aaaa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuax,Xvsa,zuswyv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuax,Xvwa,zuvy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuax,Xvya,zuwv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuax,Xzva,ywuv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iuax,vXsa,zuswyv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('xyzw,Iuax,vXwa,zuvy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,Iuax,vXya,zuwv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuax,va,Xuzvyw->IX', v_aaaa, t1_caea, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuax,vsta,Xutzswvy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuax,vsta,Xutzwsvy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuax,vsta,Xutzwyvs->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuax,vsta,Xutzyswv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuax,vsta,Xutzyvsw->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuax,vsta,Xutzyvws->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuax,vsta,Xutzywsv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuax,vswa,Xzusvy->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuax,vsya,Xzuswv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuax,vzsa,Xuswyv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuax,zXva,ywuv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuax,za,Xuwy->IX', v_aaaa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuax,zvsa,Xusvyw->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuxa,Xa,zuwy->IX', v_aaaa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuxa,Xvsa,zuswvy->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuxa,Xvwa,zuyv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,Iuxa,Xvya,zuwv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxa,Xzva,ywvu->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuxa,vXsa,zuswyv->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuxa,vXwa,zuvy->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuxa,vXya,zuwv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuxa,va,Xuzyvw->IX', v_aaaa, t1_caae, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuxa,vsta,Xutzysvw->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuxa,vswa,Xzuyvs->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Iuxa,vsya,Xzuvws->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxa,vzsa,Xusywv->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxa,zXva,ywuv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxa,za,Xuyw->IX', v_aaaa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxa,zvsa,Xusyvw->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,IxXa,ua,zuwy->IX', v_aaaa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,IxXa,uvsa,zvuwys->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,IxXa,uvza,ywvu->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,IxaX,ua,zuwy->IX', v_aaaa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,IxaX,uvsa,zvuwys->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,IxaX,uvza,ywvu->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Ixau,Xa,zuwy->IX', v_aaaa, t1_caea, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixau,Xvsa,zuvwys->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixau,Xvua,zvwy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixau,Xvza,ywuv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Ixau,vXsa,zuvwys->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Ixau,vXua,zvwy->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Ixau,vXza,ywuv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixau,va,Xwyvzu->IX', v_aaaa, t1_caea, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Ixau,vsta,Xwytsuzv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Ixau,vsta,Xwytuszv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Ixau,vsta,Xwytuzsv->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixau,vsta,Xwytzsvu->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixau,vsta,Xwytzuvs->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixau,vsta,Xwytzvsu->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixau,vsta,Xwytzvus->IX', v_aaaa, t1_caea, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixau,vsua,Xywsvz->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixau,vsza,Xywsuv->IX', v_aaaa, t1_caea, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,Xa,zuwy->IX', v_aaaa, t1_caae, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,Xvsa,zuvwsy->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Ixua,Xvua,zvwy->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,Xvza,ywvu->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,vXsa,zuvwys->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,vXua,zvwy->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,vXza,ywuv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,va,Xwyuzv->IX', v_aaaa, t1_caae, t1_ae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,vsta,Xwytuzsv->IX', v_aaaa, t1_caae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,vsua,Xywvsz->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixua,vsza,Xywusv->IX', v_aaaa, t1_caae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Xuxa,Ia,zuwy->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Xuxa,Ivas,zuswyv->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Xuxa,Ivsa,zuswvy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Xxua,Ia,zuwy->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Xxua,Ivas,zuvwys->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
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
    V1 -= 1/4 * einsum('xyzw,iIxa,iuva,Xzvuwy->IX', v_aaaa, t1_ccae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,iIxa,iuwa,Xzuy->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,iIxa,iuya,Xzuw->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iIxa,izau,Xuyw->IX', v_aaaa, t1_ccae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iIxa,izua,Xuwy->IX', v_aaaa, t1_ccae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iXax,Iiua,zuwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iXax,iIua,zuwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iXux,Iiuv,zvwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iXux,Iivs,zvswyu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iXux,Iivu,zvwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iXxa,Iiua,zuwy->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iXxa,iIua,zuwy->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iXxu,Iiuv,zvwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iXxu,Iivs,zvswuy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iXxu,Iivu,zvwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iXzx,Iiuv,ywuv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,iuax,IiXa,zuwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuax,Iiva,Xywvuz->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuax,iIXa,zuwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuax,iIva,Xywvuz->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuvx,IiXs,zuswyv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('xyzw,iuvx,IiXv,zuwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuvx,IisX,zuswyv->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuvx,Iist,Xwyvsuzt->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuvx,Iist,Xwyvuszt->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuvx,Iist,Xwyvuzst->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuvx,Iist,Xwyvzstu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuvx,Iist,Xwyvztsu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuvx,Iist,Xwyvztus->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuvx,Iist,Xwyvzuts->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuvx,Iisv,Xywsuz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuvx,IivX,zuwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuvx,Iivs,Xywsuz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxa,IiXa,zuwy->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxa,Iiva,Xywvuz->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxa,iIXa,zuwy->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxa,iIva,Xywuvz->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxv,IiXs,zuswvy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxv,IiXv,zuwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,IisX,zuswvy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,Iist,Xwyvsztu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,Iisv,Xywsuz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,IivX,zuwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,Iivs,Xywusz->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxz,IiXv,ywvu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxz,IivX,ywvu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxz,Iivs,Xywvsu->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ix,IiXu,zuwy->IX', v_aaaa, t1_ca, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ix,IiuX,zuwy->IX', v_aaaa, t1_ca, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ix,Iiuv,Xwyuzv->IX', v_aaaa, t1_ca, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,ixau,IiXa,zuwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixau,Iiua,Xzyw->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixau,Iiva,Xzuvwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixau,iIXa,zuwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixau,iIua,Xzyw->IX', v_aaaa, t1_caea, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixau,iIva,Xzuvwy->IX', v_aaaa, t1_caea, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixua,IiXa,zuwy->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixua,Iiua,Xzyw->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixua,Iiva,Xzuvwy->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixua,iIXa,zuwy->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixua,iIua,Xzyw->IX', v_aaaa, t1_caae, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixua,iIva,Xzuywv->IX', v_aaaa, t1_caae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixuv,IiXs,zvuwys->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += einsum('xyzw,ixuv,IiXu,zvwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixuv,IiXv,zuwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,IisX,zvuwys->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuv,Iist,Xvuzswty->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,Iist,Xvuzwsyt->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,Iist,Xvuzwtsy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,Iist,Xvuzwtys->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,Iist,Xvuzwyst->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuv,Iist,Xvuzystw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuv,Iist,Xvuzywts->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixuv,Iisu,Xzvswy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,Iisv,Xzuswy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixuv,IiuX,zvwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,Iius,Xzvswy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,Iiuv,Xzyw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,IivX,zuwy->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixuv,Iivs,Xzuyws->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixuv,Iivu,Xzyw->IX', v_aaaa, t1_caaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uXxa,Ia,zuwy->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uXxa,Ivas,zuswyv->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uXxa,Ivsa,zuswyv->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxa,Ia,Xwyvzu->IX', v_aaaa, t1_aaae, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uvxa,IsXa,zuvwys->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxa,IsaX,zuvwys->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxa,Isat,Xwysvzut->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uvxa,Ista,Xwystuzv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uvxa,Ista,Xwysutzv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uvxa,Ista,Xwysuztv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uvxa,Ista,Xwysztuv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uvxa,Ista,Xwyszutv->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxva,Ia,Xvzyuw->IX', v_aaaa, t1_aaae, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uxva,IsXa,zvswuy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxva,IsaX,zvswuy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxva,Isat,Xvszyutw->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxva,Isav,Xzsywu->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uxva,Ista,Xvsztuwy->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uxva,Ista,Xvszwuty->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uxva,Ista,Xvszwuyt->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uxva,Ista,Xvszyutw->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uxva,Ista,Xvszyuwt->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxva,Isva,Xzsuwy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xXua,Ia,zuwy->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xXua,Ivas,zuvwys->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xXua,Ivau,zvwy->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xXua,Ivsa,zuvwys->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xXua,Ivua,zvwy->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xa,Ia,Xzyw->IX', v_aaaa, t1_ae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xa,IuXa,zuwy->IX', v_aaaa, t1_ae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xa,IuaX,zuwy->IX', v_aaaa, t1_ae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xa,Iuav,Xuzyvw->IX', v_aaaa, t1_ae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xa,Iuva,Xuzvyw->IX', v_aaaa, t1_ae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuva,Ia,Xvzuyw->IX', v_aaaa, t1_aaae, t1_ce, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xuva,IsXa,zvswyu->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuva,IsaX,zvswyu->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,xuva,Isat,Xvszuwty->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,xuva,Isat,Xvszwuty->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,xuva,Isat,Xvszwytu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuva,Isat,Xvszytuw->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuva,Isat,Xvszytwu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuva,Isat,Xvszyuwt->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuva,Isat,Xvszywut->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuva,Isav,Xzsuwy->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuva,Ista,Xvsztuwy->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuva,Ista,Xvszutwy->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuva,Ista,Xvszuwty->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuva,Ista,Xvszwuty->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,xuva,Ista,Xvszwyut->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,xuva,Ista,Xvszytuw->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,xuva,Ista,Xvszywut->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuva,Isva,Xzsywu->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xzua,Ivua,Xvyw->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,zxua,Ia,Xuyw->IX', v_aaaa, t1_aaae, t1_ce, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,zxua,IvXa,ywvu->IX', v_aaaa, t1_aaae, t1_caae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,zxua,IvaX,ywvu->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,zxua,Ivas,Xuvyws->IX', v_aaaa, t1_aaae, t1_caea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,zxua,Ivau,Xvyw->IX', v_aaaa, t1_aaae, t1_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,zxua,Ivsa,Xuvswy->IX', v_aaaa, t1_aaae, t1_caae, rdm_cccaaa, optimize = einsum_type)

    chunks = tools.calculate_double_chunks(mr_adc, nextern, [ncore, ncore, nextern],
                                                                     [ncore, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.cece t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Amplitude
        t1_caee = mr_adc.t1.caee[:, :, s_chunk:f_chunk, :] 

        ## Two-electron integrals
        v_cece = mr_adc.v2e.cece[:, s_chunk:f_chunk, :, :]

        temp  = einsum('iXab,Iaib->IX', t1_caee, v_cece, optimize = einsum_type)
        temp -= 1/2 * einsum('ixab,Iaib,Xx->IX', t1_caee, v_cece, rdm_ca, optimize = einsum_type)

        V1 += temp
        mr_adc.log.timer_debug("contracting v2e.cece t1.caee", *cput1)
    del(v_cece, t1_caee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):

        ## Amplitude
        t1_caee = mr_adc.t1.caee[:, :, s_chunk:f_chunk, :] 

        ## Two-electron integrals
        v_cece = mr_adc.v2e.cece[:, :, :, s_chunk:f_chunk]

        temp =- 2 * einsum('iXab,Ibia->IX', t1_caee, v_cece, optimize = einsum_type)
        temp += einsum('ixab,Ibia,Xx->IX', t1_caee, v_cece, rdm_ca, optimize = einsum_type)

        V1 += temp
        mr_adc.log.timer_debug("contracting v2e.cece t1.caee", *cput1)
    del(v_cece, t1_caee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceae t1.ccee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Amplitude
        t1_ccee = mr_adc.t1.ccee[:, :, s_chunk:f_chunk, :] 

        ## Two-electron integrals
        v_ceae = mr_adc.v2e.ceae[:, s_chunk:f_chunk, :, :]

        temp  = einsum('Iiab,iaXb->IX', t1_ccee, v_ceae, optimize = einsum_type)
        temp -= 1/2 * einsum('Iiab,iaxb,Xx->IX', t1_ccee, v_ceae, rdm_ca, optimize = einsum_type)

        V1 += temp
        mr_adc.log.timer_debug("contracting v2e.ceae t1.ccee", *cput1)
    del(v_ceae, t1_ccee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):

        ## Amplitude
        t1_ccee = mr_adc.t1.ccee[:, :, s_chunk:f_chunk, :] 

        ## Two-electron integrals
        v_ceae = mr_adc.v2e.ceae[:, :, :, s_chunk:f_chunk]

        temp =- 2 * einsum('Iiab,ibXa->IX', t1_ccee, v_ceae, optimize = einsum_type)
        temp += einsum('Iiab,ibxa,Xx->IX', t1_ccee, v_ceae, rdm_ca, optimize = einsum_type)

        V1 += temp
        mr_adc.log.timer_debug("contracting v2e.ceae t1.ccee", *cput1)
    del(v_ceae, t1_ccee)

    chunks = tools.calculate_double_chunks(mr_adc, nextern, [ncore, ncore, nextern],
                                                                     [ncas, ncas, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.ceae t1.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Amplitude
        t1_aaee = mr_adc.t1.aaee[:, :, s_chunk:f_chunk, :] 

        ## Two-electron integrals
        v_ceae = mr_adc.v2e.ceae[:, s_chunk:f_chunk, :, :]

        temp =- einsum('Xxab,Iayb,xy->IX', t1_aaee, v_ceae, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xyab,Iazb,Xzxy->IX', t1_aaee, v_ceae, rdm_ccaa, optimize = einsum_type)

        V1 += temp
        mr_adc.log.timer_debug("contracting v2e.ceae t1.aaee", *cput1)
    del(v_ceae, t1_aaee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):

        ## Amplitude
        t1_aaee = mr_adc.t1.aaee[:, :, s_chunk:f_chunk, :] 

        ## Two-electron integrals
        v_ceae = mr_adc.v2e.ceae[:, :, :, s_chunk:f_chunk]

        temp = 1/2 * einsum('Xxab,Ibya,xy->IX', t1_aaee, v_ceae, rdm_ca, optimize = einsum_type)

        V1 += temp
        mr_adc.log.timer_debug("contracting v2e.ceae t1.aaee", *cput1)
    del(v_ceae, t1_aaee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.aeae t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Amplitude
        t1_caee = mr_adc.t1.caee[:, :, s_chunk:f_chunk, :] 

        ## Two-electron integrals
        v_aeae = mr_adc.v2e.aeae[:, s_chunk:f_chunk, :, :]

        temp =- einsum('Ixab,Xayb,xy->IX', t1_caee, v_aeae, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('Ixab,yazb,Xxyz->IX', t1_caee, v_aeae, rdm_ccaa, optimize = einsum_type)

        V1 += temp
        mr_adc.log.timer_debug("contracting v2e.aeae t1.caee", *cput1)
    del(v_aeae, t1_caee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):

        ## Amplitude
        t1_caee = mr_adc.t1.caee[:, :, s_chunk:f_chunk, :] 

        ## Two-electron integrals
        v_aeae = mr_adc.v2e.aeae[:, :, :, s_chunk:f_chunk]

        temp = 1/2 * einsum('Ixab,Xbya,xy->IX', t1_caee, v_aeae, rdm_ca, optimize = einsum_type)

        V1 += temp
        mr_adc.log.timer_debug("contracting v2e.aeae t1.caee", *cput1)
    del(v_aeae, t1_caee)

    chunks = tools.calculate_double_chunks(mr_adc, nextern, [ncore, ncore, nextern],
                                                                     [ncore, ncas, nextern], ntensors = 3)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.ccee t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Molecular Orbitals Energies
        e_extern = mr_adc.mo_energy.e[s_chunk:f_chunk]

        ## Amplitude
        t1_ccee = mr_adc.t1.ccee[:, :, s_chunk:f_chunk, :] 
        t1_caee = mr_adc.t1.caee[:, :, s_chunk:f_chunk, :] 

        temp =- 1/2 * einsum('I,Iiab,iXab->IX', e_core, t1_ccee, t1_caee, optimize = einsum_type)
        temp += einsum('a,Iiab,iXab->IX', e_extern, t1_ccee, t1_caee, optimize = einsum_type)
        temp -= einsum('i,Iiab,iXab->IX', e_core, t1_ccee, t1_caee, optimize = einsum_type)
        temp -= 1/2 * einsum('Xx,ixab,Iiab->IX', h_aa, t1_caee, t1_ccee, optimize = einsum_type)
        temp += 1/4 * einsum('I,Iiab,ixab,Xx->IX', e_core, t1_ccee, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('a,Iiab,ixab,Xx->IX', e_extern, t1_ccee, t1_caee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('i,Iiab,ixab,Xx->IX', e_core, t1_ccee, t1_caee, rdm_ca, optimize = einsum_type)
        temp += 1/4 * einsum('xy,ixab,Iiab,Xy->IX', h_aa, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('Xxyz,ixab,Iiab,yz->IX', v_aaaa, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/4 * einsum('Xxyz,izab,Iiab,yx->IX', v_aaaa, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += 1/4 * einsum('xyzw,ixab,Iiab,Xzyw->IX', v_aaaa, t1_caee, t1_ccee, rdm_ccaa, optimize = einsum_type)
        
        ## Amplitude
        t1_caee = mr_adc.t1.caee[:, :, :, s_chunk:f_chunk] 

        temp += einsum('I,Iiab,iXba->IX', e_core, t1_ccee, t1_caee, optimize = einsum_type)
        temp -= 2 * einsum('a,Iiab,iXba->IX', e_extern, t1_ccee, t1_caee, optimize = einsum_type)
        temp += 2 * einsum('i,Iiab,iXba->IX', e_core, t1_ccee, t1_caee, optimize = einsum_type)
        temp -= 1/2 * einsum('I,Iiab,ixba,Xx->IX', e_core, t1_ccee, t1_caee, rdm_ca, optimize = einsum_type)
        temp += einsum('a,Iiab,ixba,Xx->IX', e_extern, t1_ccee, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= einsum('i,Iiab,ixba,Xx->IX', e_core, t1_ccee, t1_caee, rdm_ca, optimize = einsum_type)

        V1 += temp
        mr_adc.log.timer_debug("contracting t1.ccee t1.caee", *cput1)
    del(e_extern, t1_ccee, t1_caee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):

        ## Molecular Orbitals Energies
        e_extern = mr_adc.mo_energy.e[s_chunk:f_chunk]

        ## Amplitude
        t1_ccee = mr_adc.t1.ccee[:, :, :, s_chunk:f_chunk] 
        t1_caee = mr_adc.t1.caee[:, :, s_chunk:f_chunk, :] 

        temp =- 2 * einsum('a,Iiba,iXab->IX', e_extern, t1_ccee, t1_caee, optimize = einsum_type)
        temp += einsum('Xx,ixab,Iiba->IX', h_aa, t1_caee, t1_ccee, optimize = einsum_type)
        temp += einsum('a,Iiba,ixab,Xx->IX', e_extern, t1_ccee, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('xy,ixab,Iiba,Xy->IX', h_aa, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
        temp += einsum('Xxyz,ixab,Iiba,yz->IX', v_aaaa, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('Xxyz,izab,Iiba,yx->IX', v_aaaa, t1_caee, t1_ccee, rdm_ca, optimize = einsum_type)
        temp -= 1/2 * einsum('xyzw,ixab,Iiba,Xzyw->IX', v_aaaa, t1_caee, t1_ccee, rdm_ccaa, optimize = einsum_type)

        ## Amplitude
        t1_caee = mr_adc.t1.caee[:, :, :, s_chunk:f_chunk] 

        temp += einsum('a,Iiba,iXba->IX', e_extern, t1_ccee, t1_caee, optimize = einsum_type)
        temp -= 1/2 * einsum('a,Iiba,ixba,Xx->IX', e_extern, t1_ccee, t1_caee, rdm_ca, optimize = einsum_type)

        V1 += temp
        mr_adc.log.timer_debug("contracting t1.ccee t1.caee", *cput1)
    del(e_extern, t1_ccee, t1_caee)
        
    chunks = tools.calculate_double_chunks(mr_adc, nextern, [ncas, ncas, nextern],
                                                                     [ncore, ncas, nextern], ntensors = 3)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("t1.aaee t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Molecular Orbitals Energies
        e_extern = mr_adc.mo_energy.e[s_chunk:f_chunk]

        ## Amplitudes
        t1_aaee = mr_adc.t1.aaee[:, :, s_chunk:f_chunk, :]
        t1_caee = mr_adc.t1.caee[:, :, s_chunk:f_chunk, :]

        temp  = 1/2 * einsum('I,Ixab,Xyab,xy->IX', e_core, t1_caee, t1_aaee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('I,Ixab,yzab,Xxyz->IX', e_core, t1_caee, t1_aaee, rdm_ccaa, optimize = einsum_type)
        temp -= einsum('a,Ixab,Xyab,xy->IX', e_extern, t1_caee, t1_aaee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('a,Ixab,yzab,Xxyz->IX', e_extern, t1_caee, t1_aaee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('Xx,xyab,Izab,yz->IX', h_aa, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('xy,Ixab,Xzab,yz->IX', h_aa, t1_caee, t1_aaee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,Ixab,zwab,Xyzw->IX', h_aa, t1_caee, t1_aaee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xy,Xxab,Izab,yz->IX', h_aa, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,xzab,Iwab,Xwyz->IX', h_aa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('Xxyz,Iyab,wuab,xzwu->IX', v_aaaa, t1_caee, t1_aaee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('Xxyz,xwab,Iuab,ywzu->IX', v_aaaa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('Xxyz,xzab,Iwab,yw->IX', v_aaaa, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('Xxyz,zwab,Iuab,ywxu->IX', v_aaaa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,Ixab,Xuab,zuwy->IX', v_aaaa, t1_caee, t1_aaee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,Ixab,uvab,Xywuvz->IX', v_aaaa, t1_caee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
        temp += 1/2 * einsum('xyzw,Xxab,Iuab,zuwy->IX', v_aaaa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,xuab,Ivab,Xzvywu->IX', v_aaaa, t1_aaee, t1_caee, rdm_cccaaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,xzab,Iuab,Xuyw->IX', v_aaaa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)

        ## Amplitudes
        t1_caee = mr_adc.t1.caee[:, :, :, s_chunk:f_chunk]

        temp += 1/2 * einsum('a,Ixba,Xyab,xy->IX', e_extern, t1_caee, t1_aaee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('a,Ixba,yzab,Xxzy->IX', e_extern, t1_caee, t1_aaee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('Xx,xyab,Izba,yz->IX', h_aa, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,Xxab,Izba,yz->IX', h_aa, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,xzab,Iwba,Xwzy->IX', h_aa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('Xxyz,xwab,Iuba,ywzu->IX', v_aaaa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('Xxyz,xzab,Iwba,yw->IX', v_aaaa, t1_aaee, t1_caee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('Xxyz,zwab,Iuba,ywux->IX', v_aaaa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,Xxab,Iuba,zuwy->IX', v_aaaa, t1_aaee, t1_caee, rdm_ccaa, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,xuab,Ivba,Xzvuwy->IX', v_aaaa, t1_aaee, t1_caee, rdm_cccaaa, optimize = einsum_type)

        V1 += temp
        mr_adc.log.timer_debug("contracting t1.aaee t1.caee", *cput1)
    del(e_extern, t1_aaee, t1_caee)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):

        ## Molecular Orbitals Energies
        e_extern = mr_adc.mo_energy.e[s_chunk:f_chunk]

        ## Amplitudes
        t1_aaee = mr_adc.t1.aaee[:, :, :, s_chunk:f_chunk]
        t1_caee = mr_adc.t1.caee[:, :, s_chunk:f_chunk, :]

        temp =- 1/4 * einsum('I,Ixab,Xyba,xy->IX', e_core, t1_caee, t1_aaee, rdm_ca, optimize = einsum_type)
        temp += 1/2 * einsum('a,Ixab,Xyba,xy->IX', e_extern, t1_caee, t1_aaee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xy,Ixab,Xzba,yz->IX', h_aa, t1_caee, t1_aaee, rdm_ca, optimize = einsum_type)
        temp -= 1/4 * einsum('xyzw,Ixab,Xuba,zuwy->IX', v_aaaa, t1_caee, t1_aaee, rdm_ccaa, optimize = einsum_type)

        ## Amplitudes
        t1_caee = mr_adc.t1.caee[:, :, :, s_chunk:f_chunk]

        temp -= einsum('a,Ixba,Xyba,xy->IX', e_extern, t1_caee, t1_aaee, rdm_ca, optimize = einsum_type)

        V1 += temp
        mr_adc.log.timer_debug("contracting t1.aaee t1.caee", *cput1)
    del(e_extern, t1_aaee, t1_caee)

    # Compute denominators
    d_ip = (-e_core[:,None] + evals)
    d_ip = d_ip**(-1)

    # Compute T[+1']^(2) amplitudes
    S_12_V_p1p = einsum('mp,iP,Pm->ip', evecs, V1, S_p1p_12_inv_act, optimize = einsum_type)
    S_12_V_p1p *= d_ip
    S_12_V_p1p = einsum('mp,ip->im', evecs, S_12_V_p1p, optimize = einsum_type)

    # Compute T[+1']^(2) t2_ca tensor
    t2_ca = einsum('Pm,im->iP', S_p1p_12_inv_act, S_12_V_p1p, optimize = einsum_type)
 
    mr_adc.log.extra("Norm of T[+1']^(2):                          %20.12f" % np.linalg.norm(t2_ca))
    mr_adc.log.timer("computing T[+1']^(2) amplitudes", *cput0)

    return t2_ca

def compute_t2_0pp_singles(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing T[0'']^(2) amplitudes...")

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    # Compute K_caca matrix
    K_caca = intermediates.compute_K_0pp(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_0p_12_inv_act = overlap.compute_S12_0pp(mr_adc)

    # Compute K^{-1} matrix
    SKS = reduce(np.dot, (S_0p_12_inv_act.T, K_caca, S_0p_12_inv_act))
    evals, evecs = np.linalg.eigh(SKS)
    del(SKS)

    # Define functions to compute amplitude contributions to V1
    def compute_V1__t1_m2(mr_adc, V1):
        ## One-electron integrals
        h_aa = mr_adc.h1eff.aa

        ## Two-electron integrals
        v_aaaa = mr_adc.v2e.aaaa

        ## Reduced density matrices
        rdm_ccaa = mr_adc.rdm.ccaa
        rdm_cccaaa = mr_adc.rdm.cccaaa
        rdm_ccccaaaa = mr_adc.rdm.ccccaaaa

        V1_m2 = np.zeros_like(V1)

        chunks = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 2)
        for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
            cput1 = (logger.process_clock(), logger.perf_counter())
            mr_adc.log.debug("v2e.aeae [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

            ## Two-electron integrals
            v_aeae = mr_adc.v2e.aeae[:, s_chunk:f_chunk, :, :]

            ## Amplitudes
            t1_aaee = mr_adc.t1.aaee[:, :, s_chunk:f_chunk, :]

            INT01 = einsum('Yxab,yazb->Yxyz', t1_aaee, v_aeae, optimize = einsum_type)

            temp =- 1/2 * einsum('Xxyz,Yxyz->XY', rdm_ccaa, INT01, optimize = einsum_type)
            temp -= 1/2 * einsum('Xzxy,xyYz->XY', rdm_ccaa, INT01, optimize = einsum_type)
            temp -= 1/4 * einsum('XxyYzw,xyzw->XY', rdm_cccaaa, INT01, optimize = einsum_type)
            temp -= 1/4 * einsum('XzwYxy,xyzw->XY', rdm_cccaaa, INT01, optimize = einsum_type)

            V1_m2 += temp
            mr_adc.log.timer_debug("contracting v2e.aeae", *cput1)
        del(t1_aaee, v_aeae)

        for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
            cput1 = (logger.process_clock(), logger.perf_counter())
            mr_adc.log.debug("t1.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

            ## Molecular Orbitals Energies
            e_extern = mr_adc.mo_energy.e[s_chunk:f_chunk]

            ## Amplitudes
            t1_aaee = mr_adc.t1.aaee[:, :, s_chunk:f_chunk, :]

            t1_aaee_INT = einsum('xuab,vsab->xuvs', t1_aaee, t1_aaee, optimize = einsum_type)

            #temp =- 1/2 * einsum('a,Yxab,yzab,Xxyz->XY', e_extern, t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
            ###temp -= 1/2 * einsum('a,Yxba,yzab,Xxzy->XY', e_extern, t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
            #temp -= 1/2 * einsum('a,xYab,yzab,Xxzy->XY', e_extern, t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
            #temp -= 1/4 * einsum('a,xyab,zwab,XxyYzw->XY', e_extern, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
            #temp -= 1/4 * einsum('a,xyab,zwab,XzwYxy->XY', e_extern, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
            #temp += 1/4 * einsum('Yx,xyab,zwab,Xyzw->XY', h_aa, t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
            #temp += 1/4 * einsum('xy,Yxab,zwab,Xyzw->XY', h_aa, t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
            #temp += 1/4 * einsum('xy,xzab,Ywab,Xwyz->XY', h_aa, t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
            ###temp += 1/4 * einsum('xy,xzab,Ywba,Xwzy->XY', h_aa, t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
            #temp += 1/4 * einsum('xy,xzab,wYab,Xwzy->XY', h_aa, t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
            #temp += 1/4 * einsum('xy,xzab,wuab,XwuYyz->XY', h_aa, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
            #temp += 1/4 * einsum('xy,xzab,wuab,XyzYwu->XY', h_aa, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
            #temp += 1/4 * einsum('Yxyz,wxab,uvab,Xywvzu->XY', v_aaaa, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
            #temp += 1/4 * einsum('Yxyz,wzab,uvab,Xywxvu->XY', v_aaaa, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
            #temp += 1/4 * einsum('Yxyz,xzab,wuab,Xywu->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
            #temp += 1/4 * einsum('Yxyz,ywab,uvab,Xuvxzw->XY', v_aaaa, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
            #temp += 1/4 * einsum('xyzw,Yxab,uvab,Xywuvz->XY', v_aaaa, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
            #temp += 1/4 * einsum('xyzw,xuab,Yvab,Xzvywu->XY', v_aaaa, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
            ###temp += 1/4 * einsum('xyzw,xuab,Yvba,Xzvuwy->XY', v_aaaa, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
            #temp += 1/4 * einsum('xyzw,xuab,vYab,Xzvuwy->XY', v_aaaa, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
            #temp -= 1/6 * einsum('xyzw,xuab,vsab,XvszYuwy->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/6 * einsum('xyzw,xuab,vsab,XvszYuyw->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/6 * einsum('xyzw,xuab,vsab,XvszYwuy->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 5/24 * einsum('xyzw,xuab,vsab,XvszYwyu->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp += 1/12 * einsum('xyzw,xuab,vsab,XvszYyuw->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 5/24 * einsum('xyzw,xuab,vsab,XvszYywu->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/24 * einsum('xyzw,xuab,vsab,XvszuwyY->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/24 * einsum('xyzw,xuab,vsab,XvszuywY->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/24 * einsum('xyzw,xuab,vsab,XvszwYyu->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/24 * einsum('xyzw,xuab,vsab,XvszwuyY->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/24 * einsum('xyzw,xuab,vsab,XvszwyYu->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/24 * einsum('xyzw,xuab,vsab,XvszwyuY->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp += 1/24 * einsum('xyzw,xuab,vsab,XvszyYuw->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp += 1/24 * einsum('xyzw,xuab,vsab,XvszyuYw->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 5/24 * einsum('xyzw,xuab,vsab,XwyuYsvz->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/6 * einsum('xyzw,xuab,vsab,XwyuYszv->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 5/24 * einsum('xyzw,xuab,vsab,XwyuYvsz->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/6 * einsum('xyzw,xuab,vsab,XwyuYvzs->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/6 * einsum('xyzw,xuab,vsab,XwyuYzsv->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp += 1/12 * einsum('xyzw,xuab,vsab,XwyuYzvs->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/24 * einsum('xyzw,xuab,vsab,XwyusYvz->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp += 1/24 * einsum('xyzw,xuab,vsab,XwyuszYv->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/24 * einsum('xyzw,xuab,vsab,XwyuvYsz->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp += 1/24 * einsum('xyzw,xuab,vsab,XwyuvzYs->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/24 * einsum('xyzw,xuab,vsab,XwyuzYsv->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/24 * einsum('xyzw,xuab,vsab,XwyuzYvs->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/24 * einsum('xyzw,xuab,vsab,XwyuzsvY->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp -= 1/24 * einsum('xyzw,xuab,vsab,XwyuzvsY->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccccaaaa, optimize = einsum_type)
            #temp += 1/4 * einsum('xyzw,xzab,Yuab,Xuyw->XY', v_aaaa, t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
            #temp += 1/8 * einsum('xyzw,xzab,uvab,XuvYyw->XY', v_aaaa, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
            #temp += 1/8 * einsum('xyzw,xzab,uvab,XywYuv->XY', v_aaaa, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)

            temp =- 1/2 * einsum('a,Yxab,yzab,Xxyz->XY', e_extern, t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
            temp -= 1/2 * einsum('a,xYab,yzab,Xxzy->XY', e_extern, t1_aaee, t1_aaee, rdm_ccaa, optimize = einsum_type)
            temp -= 1/4 * einsum('a,xyab,zwab,XxyYzw->XY', e_extern, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)
            temp -= 1/4 * einsum('a,xyab,zwab,XzwYxy->XY', e_extern, t1_aaee, t1_aaee, rdm_cccaaa, optimize = einsum_type)

            temp += 1/4 * einsum('Yx,xyzw,Xyzw->XY', h_aa, t1_aaee_INT, rdm_ccaa, optimize = einsum_type)
            temp += 1/4 * einsum('xy,Yxzw,Xyzw->XY', h_aa, t1_aaee_INT, rdm_ccaa, optimize = einsum_type)
            temp += 1/4 * einsum('xy,xzYw,Xwyz->XY', h_aa, t1_aaee_INT, rdm_ccaa, optimize = einsum_type)
            temp += 1/4 * einsum('xy,xzwY,Xwzy->XY', h_aa, t1_aaee_INT, rdm_ccaa, optimize = einsum_type)
            temp += 1/4 * einsum('xy,xzwu,XwuYyz->XY', h_aa, t1_aaee_INT, rdm_cccaaa, optimize = einsum_type)
            temp += 1/4 * einsum('xy,xzwu,XyzYwu->XY', h_aa, t1_aaee_INT, rdm_cccaaa, optimize = einsum_type)
            temp += 1/4 * einsum('Yxyz,wxuv,Xywvzu->XY', v_aaaa, t1_aaee_INT, rdm_cccaaa, optimize = einsum_type)
            temp += 1/4 * einsum('Yxyz,wzuv,Xywxvu->XY', v_aaaa, t1_aaee_INT, rdm_cccaaa, optimize = einsum_type)
            temp += 1/4 * einsum('Yxyz,xzwu,Xywu->XY', v_aaaa, t1_aaee_INT, rdm_ccaa, optimize = einsum_type)
            temp += 1/4 * einsum('Yxyz,ywuv,Xuvxzw->XY', v_aaaa, t1_aaee_INT, rdm_cccaaa, optimize = einsum_type)
            temp += 1/4 * einsum('xyzw,Yxuv,Xywuvz->XY', v_aaaa, t1_aaee_INT, rdm_cccaaa, optimize = einsum_type)
            temp += 1/4 * einsum('xyzw,xuYv,Xzvywu->XY', v_aaaa, t1_aaee_INT, rdm_cccaaa, optimize = einsum_type)
            temp += 1/4 * einsum('xyzw,xuvY,Xzvuwy->XY', v_aaaa, t1_aaee_INT, rdm_cccaaa, optimize = einsum_type)
            temp -= 1/6 * einsum('xyzw,xuvs,XvszYuwy->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/6 * einsum('xyzw,xuvs,XvszYuyw->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/6 * einsum('xyzw,xuvs,XvszYwuy->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 5/24 * einsum('xyzw,xuvs,XvszYwyu->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp += 1/12 * einsum('xyzw,xuvs,XvszYyuw->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 5/24 * einsum('xyzw,xuvs,XvszYywu->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/24 * einsum('xyzw,xuvs,XvszuwyY->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/24 * einsum('xyzw,xuvs,XvszuywY->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/24 * einsum('xyzw,xuvs,XvszwYyu->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/24 * einsum('xyzw,xuvs,XvszwuyY->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/24 * einsum('xyzw,xuvs,XvszwyYu->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/24 * einsum('xyzw,xuvs,XvszwyuY->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp += 1/24 * einsum('xyzw,xuvs,XvszyYuw->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp += 1/24 * einsum('xyzw,xuvs,XvszyuYw->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 5/24 * einsum('xyzw,xuvs,XwyuYsvz->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/6 * einsum('xyzw,xuvs,XwyuYszv->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 5/24 * einsum('xyzw,xuvs,XwyuYvsz->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/6 * einsum('xyzw,xuvs,XwyuYvzs->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/6 * einsum('xyzw,xuvs,XwyuYzsv->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp += 1/12 * einsum('xyzw,xuvs,XwyuYzvs->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/24 * einsum('xyzw,xuvs,XwyusYvz->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp += 1/24 * einsum('xyzw,xuvs,XwyuszYv->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/24 * einsum('xyzw,xuvs,XwyuvYsz->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp += 1/24 * einsum('xyzw,xuvs,XwyuvzYs->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/24 * einsum('xyzw,xuvs,XwyuzYsv->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/24 * einsum('xyzw,xuvs,XwyuzYvs->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/24 * einsum('xyzw,xuvs,XwyuzsvY->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp -= 1/24 * einsum('xyzw,xuvs,XwyuzvsY->XY', v_aaaa, t1_aaee_INT, rdm_ccccaaaa, optimize = einsum_type)
            temp += 1/4 * einsum('xyzw,xzYu,Xuyw->XY', v_aaaa, t1_aaee_INT, rdm_ccaa, optimize = einsum_type)
            temp += 1/8 * einsum('xyzw,xzuv,XuvYyw->XY', v_aaaa, t1_aaee_INT, rdm_cccaaa, optimize = einsum_type)
            temp += 1/8 * einsum('xyzw,xzuv,XywYuv->XY', v_aaaa, t1_aaee_INT, rdm_cccaaa, optimize = einsum_type)

            V1_m2 += temp
            mr_adc.log.timer_debug("contracting t1.aaee", *cput1)
        del(e_extern, t1_aaee)

        V1 += V1_m2

    def compute_V1__t1_m1(mr_adc, V1):
        ## Molecular Orbitals Energies
        e_core = mr_adc.mo_energy.c

        ## One-electron integrals
        h_aa = mr_adc.h1eff.aa

        ## Two-electron integrals
        v_aaaa = mr_adc.v2e.aaaa

        ## Reduced density matrices
        rdm_ca = mr_adc.rdm.ca
        rdm_ccaa = mr_adc.rdm.ccaa
        rdm_cccaaa = mr_adc.rdm.cccaaa

        V1_m1 = np.zeros_like(V1)

        chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncas, nextern], ntensors = 2)
        for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
            cput1 = (logger.process_clock(), logger.perf_counter())
            mr_adc.log.debug("v2e.ceae [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

            ## Two-electron integrals
            v_ceae = mr_adc.v2e.ceae[:, s_chunk:f_chunk, :, :]

            ## Amplitudes
            t1_caee = mr_adc.t1.caee[:, :, s_chunk:f_chunk, :]

            temp =- einsum('iYab,iaxb,Xx->XY', t1_caee, v_ceae, rdm_ca, optimize = einsum_type)
            temp -= einsum('ixab,iaYb,Xx->XY', t1_caee, v_ceae, rdm_ca, optimize = einsum_type)
            temp -= einsum('ixab,iayb,XxYy->XY', t1_caee, v_ceae, rdm_ccaa, optimize = einsum_type)
            temp -= einsum('ixab,iayb,XyYx->XY', t1_caee, v_ceae, rdm_ccaa, optimize = einsum_type)

            V1_m1 += temp
            mr_adc.log.timer_debug("contracting v2e.ceae", *cput1)
        del(t1_caee, v_ceae)

        for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):

            ## Two-electron integrals
            v_ceae = mr_adc.v2e.ceae[:, :, :, s_chunk:f_chunk]

            ## Amplitudes
            t1_caee = mr_adc.t1.caee[:, :, s_chunk:f_chunk, :]

            temp  = 1/2 * einsum('iYab,ibxa,Xx->XY', t1_caee, v_ceae, rdm_ca, optimize = einsum_type)
            temp += 1/2 * einsum('ixab,ibYa,Xx->XY', t1_caee, v_ceae, rdm_ca, optimize = einsum_type)
            temp += 1/2 * einsum('ixab,ibya,XxYy->XY', t1_caee, v_ceae, rdm_ccaa, optimize = einsum_type)
            temp += 1/2 * einsum('ixab,ibya,XyYx->XY', t1_caee, v_ceae, rdm_ccaa, optimize = einsum_type)

            V1_m1 += temp
            mr_adc.log.timer_debug("contracting v2e.ceae", *cput1)
        del(t1_caee, v_ceae)

        for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
            cput1 = (logger.process_clock(), logger.perf_counter())
            mr_adc.log.debug("t1.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

            ## Molecular Orbitals Energies
            e_extern = mr_adc.mo_energy.e[s_chunk:f_chunk]

            ## Amplitudes
            t1_caee_ab = mr_adc.t1.caee[:, :, s_chunk:f_chunk, :]
            t1_caee_ba = mr_adc.t1.caee[:, :, :, s_chunk:f_chunk]

            temp =- einsum('a,iYab,ixab,Xx->XY', e_extern, t1_caee_ab, t1_caee_ab, rdm_ca, optimize = einsum_type)
            temp += 1/2 * einsum('a,iYab,ixba,Xx->XY', e_extern, t1_caee_ab, t1_caee_ba, rdm_ca, optimize = einsum_type)
            temp += 1/2 * einsum('a,iYba,ixab,Xx->XY', e_extern, t1_caee_ba, t1_caee_ab, rdm_ca, optimize = einsum_type)
            temp -= einsum('a,iYba,ixba,Xx->XY', e_extern, t1_caee_ba, t1_caee_ba, rdm_ca, optimize = einsum_type)
            temp -= einsum('a,ixab,iyab,XxYy->XY', e_extern, t1_caee_ab, t1_caee_ab, rdm_ccaa, optimize = einsum_type)
            temp += 1/2 * einsum('a,ixab,iyba,XxYy->XY', e_extern, t1_caee_ab, t1_caee_ba, rdm_ccaa, optimize = einsum_type)
            temp += 1/2 * einsum('a,ixba,iyab,XxYy->XY', e_extern, t1_caee_ba, t1_caee_ab, rdm_ccaa, optimize = einsum_type)
            temp -= einsum('a,ixba,iyba,XxYy->XY', e_extern, t1_caee_ba, t1_caee_ba, rdm_ccaa, optimize = einsum_type)
            temp += einsum('i,iYab,ixab,Xx->XY', e_core, t1_caee_ab, t1_caee_ab, rdm_ca, optimize = einsum_type)
            temp -= 1/2 * einsum('i,iYab,ixba,Xx->XY', e_core, t1_caee_ab, t1_caee_ba, rdm_ca, optimize = einsum_type)
            temp += 1/2 * einsum('i,ixab,iyab,XxYy->XY', e_core, t1_caee_ab, t1_caee_ab, rdm_ccaa, optimize = einsum_type)
            temp += 1/2 * einsum('i,ixab,iyab,XyYx->XY', e_core, t1_caee_ab, t1_caee_ab, rdm_ccaa, optimize = einsum_type)
            temp -= 1/4 * einsum('i,ixab,iyba,XxYy->XY', e_core, t1_caee_ab, t1_caee_ba, rdm_ccaa, optimize = einsum_type)
            temp -= 1/4 * einsum('i,ixab,iyba,XyYx->XY', e_core, t1_caee_ab, t1_caee_ba, rdm_ccaa, optimize = einsum_type)
            temp += 1/2 * einsum('Yx,ixab,iyab,Xy->XY', h_aa, t1_caee_ab, t1_caee_ab, rdm_ca, optimize = einsum_type)
            temp -= 1/4 * einsum('Yx,ixab,iyba,Xy->XY', h_aa, t1_caee_ab, t1_caee_ba, rdm_ca, optimize = einsum_type)
            temp += 1/2 * einsum('xy,ixab,iYab,Xy->XY', h_aa, t1_caee_ab, t1_caee_ab, rdm_ca, optimize = einsum_type)
            temp -= 1/4 * einsum('xy,ixab,iYba,Xy->XY', h_aa, t1_caee_ab, t1_caee_ba, rdm_ca, optimize = einsum_type)
            temp += 1/2 * einsum('xy,ixab,izab,XyYz->XY', h_aa, t1_caee_ab, t1_caee_ab, rdm_ccaa, optimize = einsum_type)
            temp += 1/2 * einsum('xy,ixab,izab,XzYy->XY', h_aa, t1_caee_ab, t1_caee_ab, rdm_ccaa, optimize = einsum_type)
            temp -= 1/4 * einsum('xy,ixab,izba,XyYz->XY', h_aa, t1_caee_ab, t1_caee_ba, rdm_ccaa, optimize = einsum_type)
            temp -= 1/4 * einsum('xy,ixab,izba,XzYy->XY', h_aa, t1_caee_ab, t1_caee_ba, rdm_ccaa, optimize = einsum_type)
            temp += 1/2 * einsum('Yxyz,ixab,iwab,Xywz->XY', v_aaaa, t1_caee_ab, t1_caee_ab, rdm_ccaa, optimize = einsum_type)
            temp -= 1/4 * einsum('Yxyz,ixab,iwba,Xywz->XY', v_aaaa, t1_caee_ab, t1_caee_ba, rdm_ccaa, optimize = einsum_type)
            temp += 1/2 * einsum('Yxyz,iyab,iwab,Xwxz->XY', v_aaaa, t1_caee_ab, t1_caee_ab, rdm_ccaa, optimize = einsum_type)
            temp -= 1/4 * einsum('Yxyz,iyab,iwba,Xwxz->XY', v_aaaa, t1_caee_ab, t1_caee_ba, rdm_ccaa, optimize = einsum_type)
            temp += 1/2 * einsum('Yxyz,izab,iwab,Xyxw->XY', v_aaaa, t1_caee_ab, t1_caee_ab, rdm_ccaa, optimize = einsum_type)
            temp -= 1/4 * einsum('Yxyz,izab,iwba,Xyxw->XY', v_aaaa, t1_caee_ab, t1_caee_ba, rdm_ccaa, optimize = einsum_type)
            temp += 1/2 * einsum('xyzw,ixab,iYab,Xzyw->XY', v_aaaa, t1_caee_ab, t1_caee_ab, rdm_ccaa, optimize = einsum_type)
            temp -= 1/4 * einsum('xyzw,ixab,iYba,Xzyw->XY', v_aaaa, t1_caee_ab, t1_caee_ba, rdm_ccaa, optimize = einsum_type)
            temp += 1/2 * einsum('xyzw,ixab,iuab,XywYuz->XY', v_aaaa, t1_caee_ab, t1_caee_ab, rdm_cccaaa, optimize = einsum_type)
            temp += 1/2 * einsum('xyzw,ixab,iuab,XzuYwy->XY', v_aaaa, t1_caee_ab, t1_caee_ab, rdm_cccaaa, optimize = einsum_type)
            temp -= 1/4 * einsum('xyzw,ixab,iuba,XywYuz->XY', v_aaaa, t1_caee_ab, t1_caee_ba, rdm_cccaaa, optimize = einsum_type)
            temp -= 1/4 * einsum('xyzw,ixab,iuba,XzuYwy->XY', v_aaaa, t1_caee_ab, t1_caee_ba, rdm_cccaaa, optimize = einsum_type)

            V1_m1 += temp
            mr_adc.log.timer_debug("contracting t1.caee", *cput1)
        del(t1_caee_ab, t1_caee_ba)

        V1 += V1_m1

    def compute_V1__t1_0(mr_adc, V1):
        ## Molecular Orbitals Energies
        e_core = mr_adc.mo_energy.c

        ## Reduced density matrices
        rdm_ca = mr_adc.rdm.ca 

        V1_0 = np.zeros_like(V1)

        chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncore, nextern], ntensors = 2)
        for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
            cput1 = (logger.process_clock(), logger.perf_counter())
            mr_adc.log.debug("v2e.cece [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

            ## Two-electron integrals
            v_cece = mr_adc.v2e.cece[:, s_chunk:f_chunk, :, :]

            ## Amplitudes
            t1_ccee = mr_adc.t1.ccee[:, :, s_chunk:f_chunk, :]

            temp =- 2 * einsum('ijab,iajb,XY->XY', t1_ccee, v_cece, rdm_ca, optimize = einsum_type)
            temp += einsum('ijab,jaib,XY->XY', t1_ccee, v_cece, rdm_ca, optimize = einsum_type)

            V1_0 += temp
            mr_adc.log.timer_debug("contracting v2e.cece", *cput1)
        del(t1_ccee, v_cece)

        for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
            cput1 = (logger.process_clock(), logger.perf_counter())
            mr_adc.log.debug("t1.ccee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

            ## Molecular Orbitals Energies
            e_extern = mr_adc.mo_energy.e[s_chunk:f_chunk]
   
            ## Amplitudes
            t1_ccee = mr_adc.t1.ccee[:, :, s_chunk:f_chunk, :]

            temp =- 2 * einsum('a,ijab,ijab,XY->XY', e_extern, t1_ccee, t1_ccee, rdm_ca, optimize = einsum_type)
            temp += einsum('a,ijab,jiab,XY->XY', e_extern, t1_ccee, t1_ccee, rdm_ca, optimize = einsum_type)
            temp += 2 * einsum('i,ijab,ijab,XY->XY', e_core, t1_ccee, t1_ccee, rdm_ca, optimize = einsum_type)
            temp -= einsum('i,ijab,jiab,XY->XY', e_core, t1_ccee, t1_ccee, rdm_ca, optimize = einsum_type)

            V1_0 += temp
            mr_adc.log.timer_debug("contracting t1.ccee", *cput1)
        del(e_extern, t1_ccee)

        V1 += V1_0

    def compute_V1__t1_p1(mr_adc, V1):
        ## Molecular Orbitals Energies
        e_core = mr_adc.mo_energy.c
        e_extern = mr_adc.mo_energy.e

        ## One-electron integrals
        h_aa = mr_adc.h1eff.aa

        ## Two-electron integrals
        v_cace = mr_adc.v2e.cace
        v_aaaa = mr_adc.v2e.aaaa

        ## Amplitudes
        t1_ccae = mr_adc.t1.ccae

        ## Reduced density matrices
        rdm_ca = mr_adc.rdm.ca 
        rdm_ccaa = mr_adc.rdm.ccaa 
        rdm_cccaaa = mr_adc.rdm.cccaaa 

        V1_p1  = einsum('ijYa,ixja,Xx->XY', t1_ccae, v_cace, rdm_ca, optimize = einsum_type)
        V1_p1 -= 1/2 * einsum('ijYa,jxia,Xx->XY', t1_ccae, v_cace, rdm_ca, optimize = einsum_type)
        V1_p1 += einsum('ijxa,iYja,Xx->XY', t1_ccae, v_cace, rdm_ca, optimize = einsum_type)
        V1_p1 -= 4 * einsum('ijxa,ixja,XY->XY', t1_ccae, v_cace, rdm_ca, optimize = einsum_type)
        V1_p1 += einsum('ijxa,iyja,XxYy->XY', t1_ccae, v_cace, rdm_ccaa, optimize = einsum_type)
        V1_p1 += einsum('ijxa,iyja,XyYx->XY', t1_ccae, v_cace, rdm_ccaa, optimize = einsum_type)
        V1_p1 -= 1/2 * einsum('ijxa,jYia,Xx->XY', t1_ccae, v_cace, rdm_ca, optimize = einsum_type)
        V1_p1 += 2 * einsum('ijxa,jxia,XY->XY', t1_ccae, v_cace, rdm_ca, optimize = einsum_type)
        V1_p1 -= 1/2 * einsum('ijxa,jyia,XxYy->XY', t1_ccae, v_cace, rdm_ccaa, optimize = einsum_type)
        V1_p1 -= 1/2 * einsum('ijxa,jyia,XyYx->XY', t1_ccae, v_cace, rdm_ccaa, optimize = einsum_type)
        V1_p1 += einsum('a,ijYa,ijxa,Xx->XY', e_extern, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 -= 1/4 * einsum('a,ijYa,jixa,Xx->XY', e_extern, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 -= 2 * einsum('a,ijxa,ijxa,XY->XY', e_extern, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 += einsum('a,ijxa,ijya,XxYy->XY', e_extern, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 -= 1/4 * einsum('a,ijxa,jiYa,Xx->XY', e_extern, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 += einsum('a,ijxa,jixa,XY->XY', e_extern, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 -= 1/4 * einsum('a,ijxa,jiya,XxYy->XY', e_extern, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 -= 1/4 * einsum('a,ijxa,jiya,XyYx->XY', e_extern, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 -= einsum('i,ijYa,ijxa,Xx->XY', e_core, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 += 2 * einsum('i,ijxa,ijxa,XY->XY', e_core, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 -= einsum('i,ijxa,ijya,XxYy->XY', e_core, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 += 1/2 * einsum('i,jiYa,ijxa,Xx->XY', e_core, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 -= einsum('i,jiYa,jixa,Xx->XY', e_core, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 += 1/2 * einsum('i,jixa,ijYa,Xx->XY', e_core, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 -= 2 * einsum('i,jixa,ijxa,XY->XY', e_core, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 += 1/2 * einsum('i,jixa,ijya,XxYy->XY', e_core, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 += 1/2 * einsum('i,jixa,ijya,XyYx->XY', e_core, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 += 2 * einsum('i,jixa,jixa,XY->XY', e_core, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 -= einsum('i,jixa,jiya,XxYy->XY', e_core, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 += 1/2 * einsum('Yx,ijxa,ijya,Xy->XY', h_aa, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 -= 1/4 * einsum('Yx,ijxa,jiya,Xy->XY', h_aa, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 += 1/2 * einsum('xy,ijxa,ijYa,Xy->XY', h_aa, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 -= 2 * einsum('xy,ijxa,ijya,XY->XY', h_aa, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 += 1/2 * einsum('xy,ijxa,ijza,XyYz->XY', h_aa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 += 1/2 * einsum('xy,ijxa,ijza,XzYy->XY', h_aa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 -= 1/4 * einsum('xy,ijxa,jiYa,Xy->XY', h_aa, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 += einsum('xy,ijxa,jiya,XY->XY', h_aa, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 -= 1/4 * einsum('xy,ijxa,jiza,XyYz->XY', h_aa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 -= 1/4 * einsum('xy,ijxa,jiza,XzYy->XY', h_aa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 += 1/2 * einsum('Yxyz,ijxa,ijwa,Xywz->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 -= 1/4 * einsum('Yxyz,ijxa,jiwa,Xywz->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 += 1/2 * einsum('Yxyz,ijya,ijwa,Xwxz->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 += einsum('Yxyz,ijya,ijxa,Xz->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 -= 2 * einsum('Yxyz,ijya,ijza,Xx->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 -= 1/4 * einsum('Yxyz,ijya,jiwa,Xwxz->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 -= 1/2 * einsum('Yxyz,ijya,jixa,Xz->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 += einsum('Yxyz,ijya,jiza,Xx->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ca, optimize = einsum_type)
        V1_p1 += 1/2 * einsum('Yxyz,ijza,ijwa,Xyxw->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 -= 1/4 * einsum('Yxyz,ijza,jiwa,Xyxw->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 += 1/2 * einsum('xyzw,ijxa,ijYa,Xzyw->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 += 1/2 * einsum('xyzw,ijxa,ijua,XywYuz->XY', v_aaaa, t1_ccae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
        V1_p1 += 1/2 * einsum('xyzw,ijxa,ijua,XzuYwy->XY', v_aaaa, t1_ccae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
        V1_p1 += einsum('xyzw,ijxa,ijwa,XzYy->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 -= 2 * einsum('xyzw,ijxa,ijya,XzYw->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 -= 1/4 * einsum('xyzw,ijxa,jiYa,Xzyw->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 -= 1/4 * einsum('xyzw,ijxa,jiua,XywYuz->XY', v_aaaa, t1_ccae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
        V1_p1 -= 1/4 * einsum('xyzw,ijxa,jiua,XzuYwy->XY', v_aaaa, t1_ccae, t1_ccae, rdm_cccaaa, optimize = einsum_type)
        V1_p1 -= 1/2 * einsum('xyzw,ijxa,jiwa,XzYy->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)
        V1_p1 += einsum('xyzw,ijxa,jiya,XzYw->XY', v_aaaa, t1_ccae, t1_ccae, rdm_ccaa, optimize = einsum_type)

        V1 += V1_p1

    def compute_V1__t1_p2(mr_adc, V1):
        ## Molecular Orbitals Energies
        e_core = mr_adc.mo_energy.c

        ## One-electron integrals
        h_aa = mr_adc.h1eff.aa

        ## Two-electron integrals
        v_caca = mr_adc.v2e.caca
        v_aaaa = mr_adc.v2e.aaaa

        ## Amplitudes
        t1_ccaa = mr_adc.t1.ccaa

        ## Reduced density matrices
        rdm_ca = mr_adc.rdm.ca 
        rdm_ccaa = mr_adc.rdm.ccaa 
        rdm_cccaaa = mr_adc.rdm.cccaaa 
        rdm_ccccaaaa = mr_adc.rdm.ccccaaaa 

        V1_p2 =- 1/2 * einsum('ijYx,ixjy,Xy->XY', t1_ccaa, v_caca, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('ijYx,iyjz,Xxyz->XY', t1_ccaa, v_caca, rdm_ccaa, optimize = einsum_type)
        V1_p2 += einsum('ijYx,jxiy,Xy->XY', t1_ccaa, v_caca, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('ijxy,iYjx,Xy->XY', t1_ccaa, v_caca, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('ijxy,iYjz,Xzxy->XY', t1_ccaa, v_caca, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 2 * einsum('ijxy,ixjy,XY->XY', t1_ccaa, v_caca, rdm_ca, optimize = einsum_type)
        V1_p2 += einsum('ijxy,ixjz,XyYz->XY', t1_ccaa, v_caca, rdm_ccaa, optimize = einsum_type)
        V1_p2 += einsum('ijxy,ixjz,XzYy->XY', t1_ccaa, v_caca, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('ijxy,iyjz,XxYz->XY', t1_ccaa, v_caca, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('ijxy,iyjz,XzYx->XY', t1_ccaa, v_caca, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('ijxy,izjw,XxyYzw->XY', t1_ccaa, v_caca, rdm_cccaaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('ijxy,izjw,XzwYxy->XY', t1_ccaa, v_caca, rdm_cccaaa, optimize = einsum_type)
        V1_p2 += einsum('ijxy,jYix,Xy->XY', t1_ccaa, v_caca, rdm_ca, optimize = einsum_type)
        V1_p2 += einsum('ijxy,jxiy,XY->XY', t1_ccaa, v_caca, rdm_ca, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('i,ijYx,ijxy,Xy->XY', e_core, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= einsum('i,ijYx,ijyx,Xy->XY', e_core, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('i,ijYx,ijyz,Xxyz->XY', e_core, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += 2 * einsum('i,ijxy,ijxy,XY->XY', e_core, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= einsum('i,ijxy,ijxz,XyYz->XY', e_core, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= einsum('i,ijxy,ijyx,XY->XY', e_core, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 += 1/4 * einsum('i,ijxy,ijzw,XxyYzw->XY', e_core, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 += 1/4 * einsum('i,ijxy,ijzw,XzwYxy->XY', e_core, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('i,ijxy,ijzx,XyYz->XY', e_core, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('i,ijxy,ijzx,XzYy->XY', e_core, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= einsum('i,ijxy,ijzy,XxYz->XY', e_core, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += 1/12 * einsum('i,ijxy,jiYz,Xzxy->XY', e_core, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += 1/6 * einsum('i,ijxy,jiYz,Xzyx->XY', e_core, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= einsum('i,jiYx,ijxy,Xy->XY', e_core, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('i,jiYx,ijyx,Xy->XY', e_core, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/12 * einsum('i,jiYx,ijyz,Xxyz->XY', e_core, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += 1/3 * einsum('i,jiYx,ijyz,Xxzy->XY', e_core, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('Yx,ijxy,ijyz,Xz->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('Yx,ijxy,ijzw,Xyzw->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('Yx,ijxy,jiyz,Xz->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('xy,ijYx,ijyz,Xz->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xy,ijYx,ijzw,Xyzw->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += einsum('xy,ijYx,jiyz,Xz->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xy,ijxz,ijYw,Xwyz->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('xy,ijxz,ijYz,Xy->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xy,ijxz,ijwu,XwuYyz->XY', h_aa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xy,ijxz,ijwu,XyzYwu->XY', h_aa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 += einsum('xy,ijxz,ijyw,XzYw->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 2 * einsum('xy,ijxz,ijyz,XY->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xy,ijxz,ijzw,XwYy->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xy,ijxz,ijzw,XyYw->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xy,ijxz,jiYw,Xwzy->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xy,ijxz,jiYz,Xy->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('xy,ijxz,jiyw,XzYw->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += einsum('xy,ijxz,jiyz,XY->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('xy,ijxz,jizw,XwYy->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('xy,ijxz,jizw,XyYw->XY', h_aa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('Yxyz,ijxw,ijuv,Xywuzv->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('Yxyz,ijxw,ijwu,Xyuz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('Yxyz,ijxw,jiwu,Xyuz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('Yxyz,ijxz,ijwu,Xywu->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('Yxyz,ijyw,ijuv,Xuvxzw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('Yxyz,ijyw,ijwu,Xuxz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('Yxyz,ijyw,ijxu,Xuzw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += einsum('Yxyz,ijyw,ijxw,Xz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('Yxyz,ijyw,ijxz,Xw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 += einsum('Yxyz,ijyw,ijzu,Xuxw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 2 * einsum('Yxyz,ijyw,ijzw,Xx->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('Yxyz,ijyw,jiwu,Xuxz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('Yxyz,ijyw,jixu,Xuwz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('Yxyz,ijyw,jixw,Xz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 += einsum('Yxyz,ijyw,jixz,Xw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('Yxyz,ijyw,jizu,Xuxw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += einsum('Yxyz,ijyw,jizw,Xx->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('Yxyz,ijzw,ijuv,Xywxuv->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('Yxyz,ijzw,ijwu,Xyxu->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('Yxyz,ijzw,jiwu,Xyxu->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xyzw,ijYx,ijuv,Xywuvz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('xyzw,ijYx,ijwu,Xyzu->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('xyzw,ijYx,ijyu,Xwuz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('xyzw,ijYx,ijyw,Xz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('xyzw,ijYx,jiwu,Xyuz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += einsum('xyzw,ijYx,jiyu,Xwuz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += einsum('xyzw,ijYx,jiyw,Xz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('xyzw,ijxu,ijYu,Xzyw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xyzw,ijxu,ijYv,Xzvywu->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xyzw,ijxu,ijuv,XywYvz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xyzw,ijxu,ijuv,XzvYwy->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)

        V1_p2 += 5/24 * einsum('xyzw,ijxu,ijvs,XywuYsvz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/6 * einsum('xyzw,ijxu,ijvs,XywuYszv->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 5/24 * einsum('xyzw,ijxu,ijvs,XywuYvsz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 -= 1/12 * einsum('xyzw,ijxu,ijvs,XywuYvzs->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/6 * einsum('xyzw,ijxu,ijvs,XywuYzsv->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/6 * einsum('xyzw,ijxu,ijvs,XywuYzvs->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 -= 1/24 * einsum('xyzw,ijxu,ijvs,XywusYzv->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/24 * einsum('xyzw,ijxu,ijvs,XywusvYz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 -= 1/24 * einsum('xyzw,ijxu,ijvs,XywuvYzs->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/24 * einsum('xyzw,ijxu,ijvs,XywuvsYz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/24 * einsum('xyzw,ijxu,ijvs,XywuzsYv->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/24 * einsum('xyzw,ijxu,ijvs,XywuzsvY->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/24 * einsum('xyzw,ijxu,ijvs,XywuzvYs->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/24 * einsum('xyzw,ijxu,ijvs,XywuzvsY->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)

        V1_p2 += 5/24 * einsum('xyzw,ijxu,ijvs,XzvsYuwy->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 5/24 * einsum('xyzw,ijxu,ijvs,XzvsYuyw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/6 * einsum('xyzw,ijxu,ijvs,XzvsYwuy->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 -= 1/12 * einsum('xyzw,ijxu,ijvs,XzvsYwyu->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/6 * einsum('xyzw,ijxu,ijvs,XzvsYyuw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/6 * einsum('xyzw,ijxu,ijvs,XzvsYywu->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/24 * einsum('xyzw,ijxu,ijvs,XzvsuYwy->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/24 * einsum('xyzw,ijxu,ijvs,XzvsuYyw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/24 * einsum('xyzw,ijxu,ijvs,XzvswYuy->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/24 * einsum('xyzw,ijxu,ijvs,XzvswYyu->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/24 * einsum('xyzw,ijxu,ijvs,XzvswuYy->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 += 1/24 * einsum('xyzw,ijxu,ijvs,XzvswuyY->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 -= 1/24 * einsum('xyzw,ijxu,ijvs,XzvsywYu->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p2 -= 1/24 * einsum('xyzw,ijxu,ijvs,XzvsywuY->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccccaaaa, optimize = einsum_type)

        V1_p2 += einsum('xyzw,ijxu,ijwu,XzYy->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('xyzw,ijxu,ijwv,XzvYyu->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 -= 2 * einsum('xyzw,ijxu,ijyu,XzYw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += einsum('xyzw,ijxu,ijyv,XzvYwu->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xyzw,ijxu,jiYu,Xzyw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xyzw,ijxu,jiYv,Xzvuwy->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('xyzw,ijxu,jiuv,XywYvz->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('xyzw,ijxu,jiuv,XzvYwy->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('xyzw,ijxu,jiwu,XzYy->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('xyzw,ijxu,jiwv,XzvYuy->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 += einsum('xyzw,ijxu,jiyu,XzYw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('xyzw,ijxu,jiyv,XzvYwu->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 -= 1/4 * einsum('xyzw,ijxz,ijYu,Xuyw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/8 * einsum('xyzw,ijxz,ijuv,XuvYyw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 -= 1/8 * einsum('xyzw,ijxz,ijuv,XywYuv->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_cccaaa, optimize = einsum_type)
        V1_p2 += einsum('xyzw,ijxz,ijyu,XuYw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += einsum('xyzw,ijxz,ijyu,XwYu->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= einsum('xyzw,ijxz,ijyw,XY->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('xyzw,ijxz,jiyu,XuYw->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 -= 1/2 * einsum('xyzw,ijxz,jiyu,XwYu->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ccaa, optimize = einsum_type)
        V1_p2 += 1/2 * einsum('xyzw,ijxz,jiyw,XY->XY', v_aaaa, t1_ccaa, t1_ccaa, rdm_ca, optimize = einsum_type)

        V1 += V1_p2

    def compute_V1__t1_0p(mr_adc, V1):
        ## Molecular Orbitals Energies
        e_core = mr_adc.mo_energy.c
        e_extern = mr_adc.mo_energy.e

        ## One-electron integrals
        h_ce = mr_adc.h1eff.ce
        h_aa = mr_adc.h1eff.aa

        ## Two-electron integrals
        v_aaaa = mr_adc.v2e.aaaa
        v_ceaa = mr_adc.v2e.ceaa
        v_caae = mr_adc.v2e.caae

        ## Amplitudes
        t1_ce = mr_adc.t1.ce
        t1_caae = mr_adc.t1.caae
        t1_caea = mr_adc.t1.caea

        ## Reduced density matrices
        rdm_ca = mr_adc.rdm.ca 
        rdm_ccaa = mr_adc.rdm.ccaa
        rdm_cccaaa = mr_adc.rdm.cccaaa
        rdm_ccccaaaa = mr_adc.rdm.ccccaaaa

        V1_0p =- einsum('ia,iYax,Xx->XY', h_ce, t1_caea, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ia,iYxa,Xx->XY', h_ce, t1_caae, rdm_ca, optimize = einsum_type)
        V1_0p -= 2 * einsum('ia,ia,XY->XY', h_ce, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ia,ixYa,Xx->XY', h_ce, t1_caae, rdm_ca, optimize = einsum_type)
        V1_0p -= einsum('ia,ixaY,Xx->XY', h_ce, t1_caea, rdm_ca, optimize = einsum_type)
        V1_0p -= einsum('ia,ixay,XxYy->XY', h_ce, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('ia,ixay,XyYx->XY', h_ce, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ia,ixya,XxYy->XY', h_ce, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ia,ixya,XyYx->XY', h_ce, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('iYax,iayx,Xy->XY', t1_caea, v_ceaa, rdm_ca, optimize = einsum_type)
        V1_0p -= einsum('iYax,iayz,Xzxy->XY', t1_caea, v_ceaa, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('iYax,ixya,Xy->XY', t1_caea, v_caae, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('iYax,iyza,Xyxz->XY', t1_caea, v_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('iYxa,iayx,Xy->XY', t1_caae, v_ceaa, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('iYxa,iayz,Xzxy->XY', t1_caae, v_ceaa, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('iYxa,ixya,Xy->XY', t1_caae, v_caae, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('iYxa,iyza,Xyzx->XY', t1_caae, v_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ia,iYxa,Xx->XY', t1_ce, v_caae, rdm_ca, optimize = einsum_type)
        V1_0p -= einsum('ia,iaYx,Xx->XY', t1_ce, v_ceaa, rdm_ca, optimize = einsum_type)
        V1_0p -= einsum('ia,iaxY,Xx->XY', t1_ce, v_ceaa, rdm_ca, optimize = einsum_type)
        V1_0p -= einsum('ia,iaxy,XxYy->XY', t1_ce, v_ceaa, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('ia,iaxy,XyYx->XY', t1_ce, v_ceaa, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ia,ixYa,Xx->XY', t1_ce, v_caae, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ia,ixya,XxYy->XY', t1_ce, v_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ia,ixya,XyYx->XY', t1_ce, v_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixYa,iayz,Xyxz->XY', t1_caae, v_ceaa, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixYa,iyza,Xzyx->XY', t1_caae, v_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('ixaY,iayz,Xyxz->XY', t1_caea, v_ceaa, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixaY,iyza,Xzxy->XY', t1_caea, v_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixay,iYza,Xxzy->XY', t1_caea, v_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('ixay,iaYy,Xx->XY', t1_caea, v_ceaa, rdm_ca, optimize = einsum_type)
        V1_0p -= einsum('ixay,iaYz,Xyzx->XY', t1_caea, v_ceaa, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('ixay,iazY,Xxzy->XY', t1_caea, v_ceaa, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('ixay,iazw,XxwYyz->XY', t1_caea, v_ceaa, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= einsum('ixay,iazw,XyzYxw->XY', t1_caea, v_ceaa, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= einsum('ixay,iazy,XxYz->XY', t1_caea, v_ceaa, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('ixay,iazy,XzYx->XY', t1_caea, v_ceaa, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixay,iyYa,Xx->XY', t1_caea, v_caae, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixay,iyza,XxYz->XY', t1_caea, v_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixay,iyza,XzYx->XY', t1_caea, v_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixay,izYa,Xyzx->XY', t1_caea, v_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixay,izwa,XxzYyw->XY', t1_caea, v_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixay,izwa,XywYxz->XY', t1_caea, v_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixya,iYza,Xxyz->XY', t1_caae, v_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixya,iaYy,Xx->XY', t1_caae, v_ceaa, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixya,iaYz,Xyzx->XY', t1_caae, v_ceaa, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixya,iazY,Xxzy->XY', t1_caae, v_ceaa, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixya,iazw,XxwYyz->XY', t1_caae, v_ceaa, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixya,iazw,XyzYxw->XY', t1_caae, v_ceaa, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixya,iazy,XxYz->XY', t1_caae, v_ceaa, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixya,iazy,XzYx->XY', t1_caae, v_ceaa, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('ixya,iyYa,Xx->XY', t1_caae, v_caae, rdm_ca, optimize = einsum_type)
        V1_0p -= einsum('ixya,iyza,XxYz->XY', t1_caae, v_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('ixya,iyza,XzYx->XY', t1_caae, v_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixya,izYa,Xyxz->XY', t1_caae, v_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixya,izwa,XxzYwy->XY', t1_caae, v_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('ixya,izwa,XywYzx->XY', t1_caae, v_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= einsum('a,iYax,ia,Xx->XY', e_extern, t1_caea, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p -= einsum('a,iYax,iyax,Xy->XY', e_extern, t1_caea, t1_caea, rdm_ca, optimize = einsum_type)
        V1_0p -= einsum('a,iYax,iyaz,Xzxy->XY', e_extern, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,iYxa,ia,Xx->XY', e_extern, t1_caae, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,iYxa,iyax,Xy->XY', e_extern, t1_caae, t1_caea, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,iYxa,iyaz,Xzxy->XY', e_extern, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('a,iYxa,iyxa,Xy->XY', e_extern, t1_caae, t1_caae, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,iYxa,iyza,Xzyx->XY', e_extern, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('a,ia,ia,XY->XY', e_extern, t1_ce, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,ixYa,ia,Xx->XY', e_extern, t1_caae, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,ixYa,iyaz,Xyxz->XY', e_extern, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('a,ixYa,iyza,Xyzx->XY', e_extern, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('a,ixaY,ia,Xx->XY', e_extern, t1_caea, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/6 * einsum('a,ixaY,iyaz,Xyxz->XY', e_extern, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/6 * einsum('a,ixaY,iyaz,Xyzx->XY', e_extern, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('a,ixay,ia,XxYy->XY', e_extern, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('a,ixay,ia,XyYx->XY', e_extern, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/6 * einsum('a,ixay,izaY,Xxyz->XY', e_extern, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 5/6 * einsum('a,ixay,izaY,Xxzy->XY', e_extern, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('a,ixay,izay,XxYz->XY', e_extern, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('a,ixay,izay,XzYx->XY', e_extern, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,ixya,iYay,Xx->XY', e_extern, t1_caae, t1_caea, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,ixya,iYaz,Xyzx->XY', e_extern, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,ixya,ia,XxYy->XY', e_extern, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,ixya,ia,XyYx->XY', e_extern, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('a,ixya,izYa,Xxyz->XY', e_extern, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,ixya,izaY,Xxzy->XY', e_extern, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,ixya,izaw,XxwYyz->XY', e_extern, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,ixya,izaw,XyzYxw->XY', e_extern, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,ixya,izay,XxYz->XY', e_extern, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('a,ixya,izay,XzYx->XY', e_extern, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('a,ixya,izwa,XxwYzy->XY', e_extern, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('a,ixya,izwa,XyzYwx->XY', e_extern, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('a,ixya,izya,XxYz->XY', e_extern, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('a,ixya,izya,XzYx->XY', e_extern, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += einsum('i,iYax,ia,Xx->XY', e_core, t1_caea, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p += einsum('i,iYax,iyax,Xy->XY', e_core, t1_caea, t1_caea, rdm_ca, optimize = einsum_type)
        V1_0p += einsum('i,iYax,iyaz,Xzxy->XY', e_core, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,iYxa,ia,Xx->XY', e_core, t1_caae, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,iYxa,iyax,Xy->XY', e_core, t1_caae, t1_caea, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,iYxa,iyaz,Xzxy->XY', e_core, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += einsum('i,iYxa,iyxa,Xy->XY', e_core, t1_caae, t1_caae, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,iYxa,iyza,Xzyx->XY', e_core, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += einsum('i,ia,ia,XY->XY', e_core, t1_ce, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,ixYa,ia,Xx->XY', e_core, t1_caae, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,ixYa,iyaz,Xyxz->XY', e_core, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('i,ixYa,iyza,Xyzx->XY', e_core, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += einsum('i,ixaY,ia,Xx->XY', e_core, t1_caea, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p += 1/6 * einsum('i,ixaY,iyaz,Xyxz->XY', e_core, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/6 * einsum('i,ixaY,iyaz,Xyzx->XY', e_core, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += einsum('i,ixay,ia,XxYy->XY', e_core, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += einsum('i,ixay,ia,XyYx->XY', e_core, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/6 * einsum('i,ixay,izaY,Xxyz->XY', e_core, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 5/6 * einsum('i,ixay,izaY,Xxzy->XY', e_core, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += einsum('i,ixay,izaw,XxwYyz->XY', e_core, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('i,ixay,izay,XxYz->XY', e_core, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('i,ixay,izay,XzYx->XY', e_core, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,ixya,iYay,Xx->XY', e_core, t1_caae, t1_caea, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,ixya,iYaz,Xyzx->XY', e_core, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,ixya,ia,XxYy->XY', e_core, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,ixya,ia,XyYx->XY', e_core, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('i,ixya,izYa,Xxyz->XY', e_core, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,ixya,izaY,Xxzy->XY', e_core, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,ixya,izaw,XxwYyz->XY', e_core, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,ixya,izaw,XyzYxw->XY', e_core, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,ixya,izay,XxYz->XY', e_core, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('i,ixya,izay,XzYx->XY', e_core, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('i,ixya,izwa,XxwYzy->XY', e_core, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('i,ixya,izwa,XyzYwx->XY', e_core, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('i,ixya,izya,XxYz->XY', e_core, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('i,ixya,izya,XzYx->XY', e_core, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yx,ixay,ia,Xy->XY', h_aa, t1_caea, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yx,ixay,izaw,Xwyz->XY', h_aa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yx,ixay,izay,Xz->XY', h_aa, t1_caea, t1_caea, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yx,ixay,izwa,Xwyz->XY', h_aa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yx,ixay,izya,Xz->XY', h_aa, t1_caea, t1_caae, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yx,ixya,ia,Xy->XY', h_aa, t1_caae, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yx,ixya,izaw,Xwyz->XY', h_aa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yx,ixya,izay,Xz->XY', h_aa, t1_caae, t1_caea, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yx,ixya,izwa,Xwzy->XY', h_aa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yx,ixya,izya,Xz->XY', h_aa, t1_caae, t1_caae, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('Yx,iyax,ia,Xy->XY', h_aa, t1_caea, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('Yx,iyax,izaw,Xzyw->XY', h_aa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yx,iyax,izwa,Xzyw->XY', h_aa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yx,iyxa,ia,Xy->XY', h_aa, t1_caae, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yx,iyxa,izaw,Xzyw->XY', h_aa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yx,iyxa,izwa,Xzwy->XY', h_aa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xy,iYax,ia,Xy->XY', h_aa, t1_caea, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xy,iYax,izaw,Xwyz->XY', h_aa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('xy,iYax,izay,Xz->XY', h_aa, t1_caea, t1_caea, rdm_ca, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,iYax,izwa,Xwyz->XY', h_aa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,iYax,izya,Xz->XY', h_aa, t1_caea, t1_caae, rdm_ca, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,iYxa,ia,Xy->XY', h_aa, t1_caae, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,iYxa,izaw,Xwyz->XY', h_aa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,iYxa,izay,Xz->XY', h_aa, t1_caae, t1_caea, rdm_ca, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,iYxa,izwa,Xwzy->XY', h_aa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('xy,iYxa,izya,Xz->XY', h_aa, t1_caae, t1_caae, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixYa,ia,Xy->XY', h_aa, t1_caae, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixYa,izaw,Xzyw->XY', h_aa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixYa,izwa,Xzwy->XY', h_aa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,ixaY,ia,Xy->XY', h_aa, t1_caea, t1_ce, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,ixaY,izaw,Xzyw->XY', h_aa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixaY,izwa,Xzyw->XY', h_aa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,ixaz,iYaw,Xzwy->XY', h_aa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,ixaz,iYaz,Xy->XY', h_aa, t1_caea, t1_caea, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixaz,iYwa,Xzwy->XY', h_aa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixaz,iYza,Xy->XY', h_aa, t1_caea, t1_caae, rdm_ca, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,ixaz,ia,XyYz->XY', h_aa, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,ixaz,ia,XzYy->XY', h_aa, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixaz,iwYa,Xywz->XY', h_aa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,ixaz,iwaY,Xywz->XY', h_aa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,ixaz,iwau,XyuYzw->XY', h_aa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,ixaz,iwau,XzwYyu->XY', h_aa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,ixaz,iwaz,XwYy->XY', h_aa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,ixaz,iwaz,XyYw->XY', h_aa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixaz,iwua,XyuYzw->XY', h_aa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixaz,iwua,XzwYyu->XY', h_aa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixaz,iwza,XwYy->XY', h_aa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixaz,iwza,XyYw->XY', h_aa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixza,iYaw,Xzwy->XY', h_aa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixza,iYaz,Xy->XY', h_aa, t1_caae, t1_caea, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixza,iYwa,Xzyw->XY', h_aa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,ixza,iYza,Xy->XY', h_aa, t1_caae, t1_caae, rdm_ca, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixza,ia,XyYz->XY', h_aa, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixza,ia,XzYy->XY', h_aa, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixza,iwYa,Xyzw->XY', h_aa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixza,iwaY,Xywz->XY', h_aa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixza,iwau,XyuYzw->XY', h_aa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixza,iwau,XzwYyu->XY', h_aa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixza,iwaz,XwYy->XY', h_aa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixza,iwaz,XyYw->XY', h_aa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixza,iwua,XyuYwz->XY', h_aa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xy,ixza,iwua,XzwYuy->XY', h_aa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,ixza,iwza,XwYy->XY', h_aa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,ixza,iwza,XyYw->XY', h_aa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xy,izax,iYaw,Xywz->XY', h_aa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,izax,iYwa,Xywz->XY', h_aa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xy,izax,ia,XyYz->XY', h_aa, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xy,izax,ia,XzYy->XY', h_aa, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,izax,iwYa,Xzwy->XY', h_aa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xy,izax,iwaY,Xzwy->XY', h_aa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xy,izax,iwau,XywYzu->XY', h_aa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xy,izax,iwau,XzuYyw->XY', h_aa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= einsum('xy,izax,iway,XzYw->XY', h_aa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,izax,iwua,XywYzu->XY', h_aa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,izax,iwua,XzuYyw->XY', h_aa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,izax,iwya,XwYz->XY', h_aa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xy,izax,iwya,XzYw->XY', h_aa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,izxa,iYaw,Xywz->XY', h_aa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,izxa,iYwa,Xyzw->XY', h_aa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,izxa,ia,XyYz->XY', h_aa, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,izxa,ia,XzYy->XY', h_aa, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,izxa,iwYa,Xzyw->XY', h_aa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,izxa,iwaY,Xzwy->XY', h_aa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,izxa,iwau,XywYzu->XY', h_aa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,izxa,iwau,XzuYyw->XY', h_aa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,izxa,iwua,XywYuz->XY', h_aa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xy,izxa,iwua,XzuYwy->XY', h_aa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= einsum('xy,izxa,iwya,XzYw->XY', h_aa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('Yxyz,iwax,ia,Xywz->XY', v_aaaa, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('Yxyz,iwax,iuav,Xyuwzv->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yxyz,iwax,iuva,Xyuwzv->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,iwax,izau,Xywu->XY', v_aaaa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,iwax,izua,Xywu->XY', v_aaaa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('Yxyz,iway,ia,Xwxz->XY', v_aaaa, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('Yxyz,iway,iuav,Xwvxzu->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= einsum('Yxyz,iway,iuax,Xwuz->XY', v_aaaa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('Yxyz,iway,iuaz,Xwxu->XY', v_aaaa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yxyz,iway,iuva,Xwvxzu->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,iway,iuxa,Xwuz->XY', v_aaaa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,iway,iuza,Xwxu->XY', v_aaaa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('Yxyz,iwaz,ia,Xyxw->XY', v_aaaa, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('Yxyz,iwaz,iuav,Xyuxwv->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yxyz,iwaz,iuva,Xyuxwv->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yxyz,iwxa,ia,Xywz->XY', v_aaaa, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yxyz,iwxa,iuav,Xyuwzv->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yxyz,iwxa,iuva,Xyuvzw->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,iwxa,izau,Xywu->XY', v_aaaa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,iwxa,izua,Xyuw->XY', v_aaaa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yxyz,iwya,ia,Xwxz->XY', v_aaaa, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yxyz,iwya,iuav,Xwvxzu->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,iwya,iuax,Xwuz->XY', v_aaaa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,iwya,iuaz,Xwxu->XY', v_aaaa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yxyz,iwya,iuva,Xwvxuz->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,iwya,iuxa,Xwzu->XY', v_aaaa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('Yxyz,iwya,iuza,Xwxu->XY', v_aaaa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yxyz,iwza,ia,Xyxw->XY', v_aaaa, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yxyz,iwza,iuav,Xyuxwv->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('Yxyz,iwza,iuva,Xyuxvw->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,ixaw,ia,Xywz->XY', v_aaaa, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,ixaw,iuav,Xyvwzu->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,ixaw,iuaw,Xyuz->XY', v_aaaa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,ixaw,iuaz,Xywu->XY', v_aaaa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,ixaw,iuva,Xyvwzu->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,ixaw,iuwa,Xyuz->XY', v_aaaa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,ixaw,iuza,Xywu->XY', v_aaaa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,ixwa,ia,Xywz->XY', v_aaaa, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,ixwa,iuav,Xyvwzu->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,ixwa,iuaw,Xyuz->XY', v_aaaa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,ixwa,iuaz,Xywu->XY', v_aaaa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,ixwa,iuva,Xyvuzw->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,ixwa,iuwa,Xyuz->XY', v_aaaa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,ixwa,iuza,Xyuw->XY', v_aaaa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,iyaw,ia,Xwxz->XY', v_aaaa, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,iyaw,iuav,Xwuxzv->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,iyaw,iuaw,Xuxz->XY', v_aaaa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,iyaw,iuva,Xwuxzv->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,iyaw,iuwa,Xuxz->XY', v_aaaa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,iywa,ia,Xwxz->XY', v_aaaa, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,iywa,iuav,Xwuxzv->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,iywa,iuaw,Xuxz->XY', v_aaaa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,iywa,iuva,Xwuxvz->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,iywa,iuwa,Xuxz->XY', v_aaaa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,izaw,ia,Xyxw->XY', v_aaaa, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,izaw,iuav,Xyvxwu->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,izaw,iuaw,Xyxu->XY', v_aaaa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,izaw,iuva,Xyvxwu->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,izaw,iuwa,Xyxu->XY', v_aaaa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,izwa,ia,Xyxw->XY', v_aaaa, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,izwa,iuav,Xyvxwu->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,izwa,iuaw,Xyxu->XY', v_aaaa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('Yxyz,izwa,iuva,Xyvxuw->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('Yxyz,izwa,iuwa,Xyxu->XY', v_aaaa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,iYax,ia,Xzyw->XY', v_aaaa, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,iYax,iuav,Xzvywu->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= einsum('xyzw,iYax,iuaw,Xzyu->XY', v_aaaa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('xyzw,iYax,iuay,Xzuw->XY', v_aaaa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,iYax,iuva,Xzvywu->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iYax,iuwa,Xzyu->XY', v_aaaa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iYax,iuya,Xzuw->XY', v_aaaa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iYax,izau,Xuyw->XY', v_aaaa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iYax,izua,Xuyw->XY', v_aaaa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,iYxa,ia,Xzyw->XY', v_aaaa, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,iYxa,iuav,Xzvywu->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iYxa,iuaw,Xzyu->XY', v_aaaa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iYxa,iuay,Xzuw->XY', v_aaaa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,iYxa,iuva,Xzvuwy->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iYxa,iuwa,Xzuy->XY', v_aaaa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= einsum('xyzw,iYxa,iuya,Xzuw->XY', v_aaaa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iYxa,izau,Xuyw->XY', v_aaaa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iYxa,izua,Xuwy->XY', v_aaaa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,iuax,iYav,Xywvuz->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,iuax,iYva,Xywvuz->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,iuax,ia,XywYuz->XY', v_aaaa, t1_caea, t1_ce, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,iuax,ia,XzuYwy->XY', v_aaaa, t1_caea, t1_ce, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,iuax,ivYa,Xzuvwy->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,iuax,ivaY,Xzuvwy->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,ivas,XywvYsuz->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,ivas,XywvYszu->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,ivas,XywvYusz->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,ivas,XywvYzsu->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,ivas,XywvYzus->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,ivas,XzusYvwy->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,ivas,XzusYvyw->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,ivas,XzusYwvy->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,ivas,XzusYyvw->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,ivas,XzusYywv->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= einsum('xyzw,iuax,ivaw,XzuYvy->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= einsum('xyzw,iuax,ivay,XzuYwv->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuax,ivsa,XywvYsuz->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuax,ivsa,XywvYszu->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuax,ivsa,XywvYusz->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuax,ivsa,XywvYzsu->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuax,ivsa,XywvYzus->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuax,ivsa,XzusYvwy->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuax,ivsa,XzusYvyw->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuax,ivsa,XzusYwvy->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuax,ivsa,XzusYyvw->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuax,ivsa,XzusYywv->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,ivwa,XyvYuz->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,ivwa,XzuYvy->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,ivya,XwvYzu->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,ivya,XzuYwv->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,izav,XuvYyw->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuax,izav,XywYuv->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,iuxa,iYav,Xywvuz->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,iuxa,iYva,Xywuvz->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,iuxa,ia,XywYuz->XY', v_aaaa, t1_caae, t1_ce, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,iuxa,ia,XzuYwy->XY', v_aaaa, t1_caae, t1_ce, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,iuxa,ivYa,Xzuywv->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,iuxa,ivaY,Xzuvwy->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuxa,ivas,XywvYsuz->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuxa,ivas,XywvYszu->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuxa,ivas,XywvYusz->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuxa,ivas,XywvYzsu->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuxa,ivas,XywvYzus->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuxa,ivas,XzusYvwy->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuxa,ivas,XzusYvyw->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuxa,ivas,XzusYwvy->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuxa,ivas,XzusYyvw->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuxa,ivas,XzusYywv->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,iuxa,ivsa,XywvYszu->XY', v_aaaa, t1_caae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,iuxa,ivsa,XzusYwvy->XY', v_aaaa, t1_caae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,iuxa,ivwa,XzuYyv->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= einsum('xyzw,iuxa,ivya,XzuYwv->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuxa,izav,XuvYyw->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,iuxa,izav,XywYuv->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/6 * einsum('xyzw,iuxa,izva,XuvYwy->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/12 * einsum('xyzw,iuxa,izva,XuvYyw->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/12 * einsum('xyzw,iuxa,izva,XuvwyY->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/12 * einsum('xyzw,iuxa,izva,XuvywY->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/12 * einsum('xyzw,iuxa,izva,XywYuv->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/6 * einsum('xyzw,iuxa,izva,XywYvu->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/12 * einsum('xyzw,iuxa,izva,XywuvY->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/12 * einsum('xyzw,iuxa,izva,XywvuY->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixYa,ia,Xzyw->XY', v_aaaa, t1_caae, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixYa,iuav,Xzuywv->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixYa,iuaz,Xuyw->XY', v_aaaa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixYa,iuva,Xzuvwy->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixYa,iuza,Xuwy->XY', v_aaaa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,ixaY,ia,Xzyw->XY', v_aaaa, t1_caea, t1_ce, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,ixaY,iuav,Xzuywv->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,ixaY,iuaz,Xuyw->XY', v_aaaa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixaY,iuva,Xzuywv->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixaY,iuza,Xuyw->XY', v_aaaa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,ixau,iYau,Xzyw->XY', v_aaaa, t1_caea, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,ixau,iYav,Xzuvwy->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixau,iYua,Xzyw->XY', v_aaaa, t1_caea, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixau,iYva,Xzuvwy->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,ixau,ia,XywYuz->XY', v_aaaa, t1_caea, t1_ce, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,ixau,ia,XzuYwy->XY', v_aaaa, t1_caea, t1_ce, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixau,ivYa,Xywvuz->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,ixau,ivaY,Xywvuz->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,ixau,ivas,XywsYuvz->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,ixau,ivas,XywsYvuz->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,ixau,ivas,XywsYvzu->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,ixau,ivas,XywsYzuv->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,ixau,ivas,XywsYzvu->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,ixau,ivas,XzuvYswy->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,ixau,ivas,XzuvYsyw->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,ixau,ivas,XzuvYwsy->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,ixau,ivas,XzuvYysw->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/2 * einsum('xyzw,ixau,ivas,XzuvYyws->XY', v_aaaa, t1_caea, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,ixau,ivau,XywYvz->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,ixau,ivau,XzvYwy->XY', v_aaaa, t1_caea, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixau,ivsa,XywsYuvz->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixau,ivsa,XywsYvuz->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixau,ivsa,XywsYvzu->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixau,ivsa,XywsYzuv->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixau,ivsa,XywsYzvu->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixau,ivsa,XzuvYswy->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixau,ivsa,XzuvYsyw->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixau,ivsa,XzuvYwsy->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixau,ivsa,XzuvYysw->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixau,ivsa,XzuvYyws->XY', v_aaaa, t1_caea, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixau,ivua,XywYvz->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixau,ivua,XzvYwy->XY', v_aaaa, t1_caea, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixua,iYau,Xzyw->XY', v_aaaa, t1_caae, t1_caea, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixua,iYav,Xzuvwy->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,ixua,iYua,Xzyw->XY', v_aaaa, t1_caae, t1_caae, rdm_ccaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixua,iYva,Xzuywv->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixua,ia,XywYuz->XY', v_aaaa, t1_caae, t1_ce, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixua,ia,XzuYwy->XY', v_aaaa, t1_caae, t1_ce, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixua,ivYa,Xywuvz->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixua,ivaY,Xywvuz->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixua,ivas,XywsYuvz->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixua,ivas,XywsYvuz->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixua,ivas,XywsYvzu->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixua,ivas,XywsYzuv->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixua,ivas,XywsYzvu->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixua,ivas,XzuvYswy->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixua,ivas,XzuvYsyw->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixua,ivas,XzuvYwsy->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixua,ivas,XzuvYysw->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/4 * einsum('xyzw,ixua,ivas,XzuvYyws->XY', v_aaaa, t1_caae, t1_caea, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixua,ivau,XywYvz->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixua,ivau,XzvYwy->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixua,ivaz,XuvYyw->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixua,ivaz,XywYuv->XY', v_aaaa, t1_caae, t1_caea, rdm_cccaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixua,ivsa,XywsYvzu->XY', v_aaaa, t1_caae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p -= 1/4 * einsum('xyzw,ixua,ivsa,XzuvYwsy->XY', v_aaaa, t1_caae, t1_caae, rdm_ccccaaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,ixua,ivua,XywYvz->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/2 * einsum('xyzw,ixua,ivua,XzvYwy->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/12 * einsum('xyzw,ixua,ivza,XuvwyY->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/12 * einsum('xyzw,ixua,ivza,XuvywY->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/12 * einsum('xyzw,ixua,ivza,XywuvY->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)
        V1_0p += 1/12 * einsum('xyzw,ixua,ivza,XywvuY->XY', v_aaaa, t1_caae, t1_caae, rdm_cccaaa, optimize = einsum_type)

        V1 += V1_0p

    def compute_V1__t1_m1p(mr_adc, V1):
        ## Molecular Orbitals Energies
        e_extern = mr_adc.mo_energy.e

        ## One-electron integrals
        h_ae = mr_adc.h1eff.ae
        h_aa = mr_adc.h1eff.aa

        ## Two-electron integrals
        v_aaaa = mr_adc.v2e.aaaa
        v_aaae = mr_adc.v2e.aaae

        ## Amplitudes
        t1_ae = mr_adc.t1.ae
        t1_aaae = mr_adc.t1.aaae

        ## Reduced density matrices
        rdm_ca = mr_adc.rdm.ca 
        rdm_ccaa = mr_adc.rdm.ccaa
        rdm_cccaaa = mr_adc.rdm.cccaaa
        rdm_ccccaaaa = mr_adc.rdm.ccccaaaa

        V1_m1p =- 1/2 * einsum('Ya,xa,Xx->XY', h_ae, t1_ae, rdm_ca, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('Ya,xyza,Xzyx->XY', h_ae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xa,Ya,Xx->XY', h_ae, t1_ae, rdm_ca, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xa,Yyza,Xyzx->XY', h_ae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xa,yYza,Xyxz->XY', h_ae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xa,ya,XxYy->XY', h_ae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xa,ya,XyYx->XY', h_ae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xa,yzYa,Xxyz->XY', h_ae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xa,yzwa,XxwYzy->XY', h_ae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xa,yzwa,XzyYxw->XY', h_ae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('Ya,xyza,Xyzx->XY', t1_ae, v_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('Yxya,zwua,Xxwyuz->XY', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('Yxya,zywa,Xxzw->XY', t1_aaae, v_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xYya,zwua,Xxwuyz->XY', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xYya,zywa,Xxwz->XY', t1_aaae, v_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xa,Yyza,Xzyx->XY', t1_ae, v_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xa,yYza,Xxyz->XY', t1_ae, v_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xa,yzYa,Xyxz->XY', t1_ae, v_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xa,yzwa,XwyYxz->XY', t1_ae, v_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xa,yzwa,XxzYwy->XY', t1_ae, v_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyYa,zwua,Xuzxyw->XY', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyza,Ywua,Xzuwxy->XY', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyza,Yzwa,Xwxy->XY', t1_aaae, v_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyza,wYua,Xyxwuz->XY', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyza,wuYa,Xzwyxu->XY', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyza,wuva,XyxuYvzw->XY', t1_aaae, v_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyza,wuva,XzvwYxyu->XY', t1_aaae, v_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyza,wzYa,Xwyx->XY', t1_aaae, v_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyza,wzua,XwuYxy->XY', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyza,wzua,XxyYwu->XY', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('a,Ya,xa,Xx->XY', e_extern, t1_ae, t1_ae, rdm_ca, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('a,Yxya,za,Xxyz->XY', e_extern, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/32 * einsum('a,Yxya,zwua,Xxuwyz->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/6 * einsum('a,Yxya,zwua,Xxuywz->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/32 * einsum('a,Yxya,zwua,Xxuyzw->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 13/96 * einsum('a,Yxya,zwua,Xxuzwy->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/16 * einsum('a,Yxya,zwya,Xxwz->XY', e_extern, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 7/16 * einsum('a,Yxya,zwya,Xxzw->XY', e_extern, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('a,xYya,za,Xxzy->XY', e_extern, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('a,xYya,zwua,Xxuwyz->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('a,xYya,zwya,Xxwz->XY', e_extern, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('a,xa,ya,XxYy->XY', e_extern, t1_ae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('a,xa,ya,XyYx->XY', e_extern, t1_ae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('a,xyYa,za,Xzxy->XY', e_extern, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/48 * einsum('a,xyYa,zwua,Xwzuxy->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/6 * einsum('a,xyYa,zwua,Xwzuyx->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 7/48 * einsum('a,xyYa,zwua,Xwzxyu->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/48 * einsum('a,xyYa,zwua,Xwzyux->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('a,xyza,Ya,Xzyx->XY', e_extern, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 19/96 * einsum('a,xyza,Ywua,Xzwuxy->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/6 * einsum('a,xyza,Ywua,Xzwuyx->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 13/96 * einsum('a,xyza,Ywua,Xzwxyu->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 13/96 * einsum('a,xyza,Ywua,Xzwyux->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/6 * einsum('a,xyza,Ywua,Xzwyxu->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/16 * einsum('a,xyza,Ywza,Xwxy->XY', e_extern, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/16 * einsum('a,xyza,Ywza,Xwyx->XY', e_extern, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('a,xyza,wa,XyxYwz->XY', e_extern, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('a,xyza,wa,XzwYxy->XY', e_extern, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/6 * einsum('a,xyza,wuYa,Xyxuwz->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 7/48 * einsum('a,xyza,wuYa,Xyxuzw->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 3/16 * einsum('a,xyza,wuYa,Xyxwuz->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/6 * einsum('a,xyza,wuYa,Xyxwzu->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 7/48 * einsum('a,xyza,wuYa,Xyxzwu->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('a,xyza,wuva,XyxvYuzw->XY', e_extern, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('a,xyza,wuza,XuwYyx->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('a,xyza,wuza,XyxYuw->XY', e_extern, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yx,xa,ya,Xy->XY', h_aa, t1_ae, t1_ae, rdm_ca, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yx,xa,yzwa,Xwzy->XY', h_aa, t1_ae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yx,xyza,wa,Xyzw->XY', h_aa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yx,xyza,wuva,Xyvzuw->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yx,xyza,wuza,Xywu->XY', h_aa, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yx,yxza,wa,Xywz->XY', h_aa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yx,yxza,wuva,Xyvuzw->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yx,yxza,wuza,Xyuw->XY', h_aa, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yx,yzxa,wa,Xwyz->XY', h_aa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yx,yzxa,wuva,Xuwyzv->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,Yxza,wa,Xyzw->XY', h_aa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,Yxza,wuva,Xyvzuw->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,Yxza,wuza,Xywu->XY', h_aa, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,Yzxa,wa,Xzyw->XY', h_aa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,Yzxa,wuva,Xzvyuw->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xy,Yzxa,wuya,Xzwu->XY', h_aa, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xYza,wa,Xywz->XY', h_aa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xYza,wuva,Xyvuzw->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xYza,wuza,Xyuw->XY', h_aa, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xa,Ya,Xy->XY', h_aa, t1_ae, t1_ae, rdm_ca, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xa,Yzwa,Xzwy->XY', h_aa, t1_ae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xa,zYwa,Xzyw->XY', h_aa, t1_ae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xa,za,XyYz->XY', h_aa, t1_ae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xa,za,XzYy->XY', h_aa, t1_ae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xa,zwYa,Xyzw->XY', h_aa, t1_ae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xa,zwua,XwzYyu->XY', h_aa, t1_ae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xa,zwua,XyuYwz->XY', h_aa, t1_ae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xzYa,wa,Xwyz->XY', h_aa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xzYa,wuva,Xuwyzv->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xzwa,Ya,Xwzy->XY', h_aa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xzwa,Yuva,Xwuvyz->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xzwa,Yuwa,Xuyz->XY', h_aa, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xzwa,uYva,Xwuzyv->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xzwa,uYwa,Xuzy->XY', h_aa, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xzwa,ua,XwuYyz->XY', h_aa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xzwa,ua,XyzYwu->XY', h_aa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xzwa,uvYa,Xyzuwv->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xzwa,uvsa,XwvuYyzs->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xzwa,uvsa,XyzsYwvu->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xzwa,uvwa,XuvYyz->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,xzwa,uvwa,XyzYuv->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zYxa,wa,Xzwy->XY', h_aa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zYxa,wuva,Xzvuyw->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xy,zYxa,wuya,Xzuw->XY', h_aa, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zwxa,Ya,Xywz->XY', h_aa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zwxa,Yuva,Xyuvzw->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zwxa,uYva,Xyuwzv->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zwxa,ua,XyuYzw->XY', h_aa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zwxa,ua,XzwYyu->XY', h_aa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zwxa,uvYa,Xzwuyv->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zwxa,uvsa,XyvuYswz->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zwxa,uvsa,XyvuYszw->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zwxa,uvsa,XyvuYwsz->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zwxa,uvsa,XyvuYwzs->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zwxa,uvsa,XyvuYzsw->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zwxa,uvsa,XzwsYuvy->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zwxa,uvsa,XzwsYuyv->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zwxa,uvsa,XzwsYvuy->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zwxa,uvsa,XzwsYvyu->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zwxa,uvsa,XzwsYyuv->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xy,zwxa,uvya,XzwYuv->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zxYa,wa,Xwzy->XY', h_aa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zxYa,wuva,Xuwzyv->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zxwa,Ya,Xwyz->XY', h_aa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zxwa,Yuva,Xwuvzy->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zxwa,Yuwa,Xuzy->XY', h_aa, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zxwa,uYva,Xwuyzv->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zxwa,uYwa,Xuyz->XY', h_aa, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zxwa,ua,XwuYzy->XY', h_aa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zxwa,ua,XyzYuw->XY', h_aa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zxwa,uvYa,Xyzuvw->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zxwa,uvsa,XwvuYsyz->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zxwa,uvsa,XwvuYszy->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zxwa,uvsa,XwvuYysz->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zxwa,uvsa,XwvuYyzs->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zxwa,uvsa,XwvuYzsy->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zxwa,uvsa,XyzsYuvw->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zxwa,uvsa,XyzsYuwv->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zxwa,uvsa,XyzsYvuw->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zxwa,uvsa,XyzsYwuv->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xy,zxwa,uvsa,XyzsYwvu->XY', h_aa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zxwa,uvwa,XvuYyz->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xy,zxwa,uvwa,XyzYvu->XY', h_aa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yxyz,wuxa,va,Xyvwzu->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yxyz,wuxa,vsta,Xysvwzut->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wuxa,vzsa,Xyvwus->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wuxa,za,Xywu->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wuxa,zvsa,Xyvwsu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yxyz,wuya,va,Xwuxzv->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wuya,vsta,Xwutxsvz->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wuya,vsta,Xwutxszv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wuya,vsta,Xwutxvsz->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wuya,vsta,Xwutxvzs->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wuya,vsta,Xwutxzvs->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('Yxyz,wuya,vsxa,Xwuvzs->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('Yxyz,wuya,vsza,Xwuxvs->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yxyz,wuza,va,Xyvxwu->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wuza,vsta,Xysvxtuw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wuza,vsta,Xysvxtwu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wuza,vsta,Xysvxutw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wuza,vsta,Xysvxuwt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wuza,vsta,Xysvxwtu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wxua,va,Xywvzu->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wxua,vsta,Xywtszuv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wxua,vsua,Xywszv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wxua,vsza,Xywsvu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wyua,va,Xuvxwz->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yxyz,wyua,vsta,Xusvxtwz->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yxyz,wyua,vsta,Xusvxtzw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yxyz,wyua,vsta,Xusvxwtz->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yxyz,wyua,vsta,Xusvxztw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yxyz,wyua,vsta,Xusvxzwt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wyua,vsua,Xsvxzw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wzua,va,Xywxvu->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yxyz,wzua,vsta,Xywtxsvu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yxyz,wzua,vsta,Xywtxusv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yxyz,wzua,vsta,Xywtxuvs->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yxyz,wzua,vsta,Xywtxvsu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('Yxyz,wzua,vsta,Xywtxvus->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,wzua,vsua,Xywxsv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,xa,wa,Xywz->XY', v_aaaa, t1_ae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,xa,wuva,Xyvuzw->XY', v_aaaa, t1_ae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,xa,wuza,Xyuw->XY', v_aaaa, t1_ae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,xwua,va,Xywuzv->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,xwua,vsta,Xywtuzsv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,xwua,vsua,Xywvzs->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,xwua,vsza,Xywuvs->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,xzwa,ua,Xywu->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,xzwa,uvsa,Xyswvu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,xzwa,uvwa,Xyuv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,ya,wa,Xwxz->XY', v_aaaa, t1_ae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,ya,wuva,Xuwxzv->XY', v_aaaa, t1_ae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,ywua,va,Xuvxzw->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,ywua,vsta,Xusvxzwt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,ywua,vsua,Xvsxzw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,za,wa,Xyxw->XY', v_aaaa, t1_ae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,za,wuva,Xyvxuw->XY', v_aaaa, t1_ae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,zwua,va,Xywxuv->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,zwua,vsta,Xywtxusv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,zwua,vsua,Xywxvs->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,zxwa,ua,Xyuw->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,zxwa,uvsa,Xysvwu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('Yxyz,zxwa,uvwa,Xyvu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,Yuxa,va,Xzuywv->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,Yuxa,vsta,Xzutywsv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyzw,Yuxa,vswa,Xzuyvs->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyzw,Yuxa,vsya,Xzuvws->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,Yuxa,vzsa,Xusywv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,Yuxa,za,Xuyw->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,Yuxa,zvsa,Xusyvw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,Yxua,va,Xywuvz->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,Yxua,vsta,Xywtuszv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,Yxua,vsua,Xywvsz->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,Yxua,vsza,Xywusv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uYxa,va,Xzuvwy->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uYxa,vsta,Xzutsywv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uYxa,vsta,Xzutwsyv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uYxa,vsta,Xzutwysv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uYxa,vsta,Xzutysvw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uYxa,vsta,Xzutyvsw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uYxa,vsta,Xzutyvws->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uYxa,vsta,Xzutywvs->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyzw,uYxa,vswa,Xzusvy->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyzw,uYxa,vsya,Xzuswv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uYxa,vzsa,Xuswyv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uYxa,za,Xuwy->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uYxa,zvsa,Xusvyw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uvxa,Ya,Xywvuz->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uvxa,Ysta,Xywstzuv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uvxa,Ysta,Xywsutzv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uvxa,Ysta,Xywsuztv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uvxa,Ysta,Xywsztuv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uvxa,Ysta,Xywszutv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uvxa,sYta,Xywsvuzt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uvxa,sa,XywsYuzv->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uvxa,sa,XzuvYwys->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uvxa,stYa,Xzuvsywt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uvxa,stYa,Xzuvwsyt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uvxa,stYa,Xzuvwyst->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uvxa,stYa,Xzuvyswt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uvxa,stYa,Xzuvywst->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyzw,uvxa,stwa,XzuvYsyt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/2 * einsum('xyzw,uvxa,stya,XzuvYwst->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uxYa,va,Xzvuwy->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxYa,vsta,Xzsvuywt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxYa,vsta,Xzsvwuyt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxYa,vsta,Xzsvwyut->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uxYa,vsta,Xzsvytuw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uxYa,vsta,Xzsvytwu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uxYa,vsta,Xzsvyutw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uxYa,vsta,Xzsvywtu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uxYa,vsza,Xvsuwy->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uxva,Ya,Xzvywu->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,Ysta,Xzvstyuw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,Ysta,Xzvswtuy->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,Ysta,Xzvswyut->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,Ysta,Xzvsytuw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,Ysta,Xzvsywut->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uxva,Ysva,Xzsuwy->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uxva,sYta,Xzvsywut->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uxva,sYva,Xzsywu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uxva,sa,XywuYszv->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uxva,sa,XzvsYwuy->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stYa,Xywusztv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stYa,Xywutszv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stYa,Xywutzsv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stYa,Xywuzstv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stYa,Xywuztsv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stva,XywuYstz->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stva,XywuYszt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stva,XywuYtsz->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stva,XywuYzst->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stva,XywuYzts->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stva,XztsYuwy->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stva,XztsYuyw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stva,XztsYwuy->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stva,XztsYyuw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,uxva,stva,XztsYywu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uxva,stza,XvstYuwy->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,uxva,stza,XywuYtsv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xYua,va,Xywvuz->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xYua,vsta,Xywtszuv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xYua,vsta,Xywtuszv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xYua,vsta,Xywtuzsv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xYua,vsta,Xywtzsvu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xYua,vsta,Xywtzuvs->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xYua,vsta,Xywtzvsu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xYua,vsta,Xywtzvus->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xYua,vsua,Xywsvz->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xYua,vsza,Xywsuv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xa,Ya,Xzyw->XY', v_aaaa, t1_ae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xa,Yuva,Xzuvwy->XY', v_aaaa, t1_ae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xa,uYva,Xzuywv->XY', v_aaaa, t1_ae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xa,ua,XywYuz->XY', v_aaaa, t1_ae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xa,ua,XzuYwy->XY', v_aaaa, t1_ae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xa,uvYa,Xywuvz->XY', v_aaaa, t1_ae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xa,uvsa,XywsYuvz->XY', v_aaaa, t1_ae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xa,uvsa,XywsYuzv->XY', v_aaaa, t1_ae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xa,uvsa,XywsYvuz->XY', v_aaaa, t1_ae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xa,uvsa,XywsYzuv->XY', v_aaaa, t1_ae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xa,uvsa,XywsYzvu->XY', v_aaaa, t1_ae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xa,uvsa,XzvuYswy->XY', v_aaaa, t1_ae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xa,uvsa,XzvuYsyw->XY', v_aaaa, t1_ae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xa,uvsa,XzvuYwsy->XY', v_aaaa, t1_ae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xa,uvsa,XzvuYysw->XY', v_aaaa, t1_ae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xa,uvsa,XzvuYyws->XY', v_aaaa, t1_ae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xa,uvza,XuvYwy->XY', v_aaaa, t1_ae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xa,uvza,XywYvu->XY', v_aaaa, t1_ae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuYa,va,Xzvywu->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuYa,vsta,Xzsvywut->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuYa,vsza,Xvsywu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,Ya,Xzvuwy->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,Ysta,Xzvstyuw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,Ysta,Xzvsuytw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,Ysta,Xzvsuywt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,Ysta,Xzvswtyu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,Ysta,Xzvswyut->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,Ysta,Xzvsytwu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,Ysta,Xzvsywtu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,Ysva,Xzsywu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,sYta,Xzvsuywt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,sYta,Xzvswuyt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,sYta,Xzvswyut->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,sYta,Xzvsytuw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,sYta,Xzvsytwu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,sYta,Xzvsyutw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,sYta,Xzvsywtu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,sYva,Xzsuwy->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,sa,XywuYsvz->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,sa,XywuYszv->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,sa,XywuYvsz->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,sa,XywuYzsv->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,sa,XywuYzvs->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,sa,XzvsYuwy->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,sa,XzvsYuyw->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,sa,XzvsYwuy->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,sa,XzvsYyuw->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,sa,XzvsYywu->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,stYa,Xywusztv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,stYa,Xywutzsv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,stYa,Xywutzvs->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,stYa,Xywuvszt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,stYa,Xywuvzts->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,stYa,Xywuzsvt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/4 * einsum('xyzw,xuva,stYa,Xywuzvst->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,stva,XywuYszt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,stva,XzstYwyu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,stza,XvstYywu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xuva,stza,XywuYvst->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xzYa,ua,Xuyw->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xzYa,uvsa,Xvuyws->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,xzua,Yvua,Xvyw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p -= 1/24 * einsum('xyzw,xzua,vsua,XvsYwy->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/24 * einsum('xyzw,xzua,vsua,XvsYyw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/12 * einsum('xyzw,xzua,vsua,XvswyY->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/12 * einsum('xyzw,xzua,vsua,XvsywY->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/24 * einsum('xyzw,xzua,vsua,XywYsv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/24 * einsum('xyzw,xzua,vsua,XywYvs->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/12 * einsum('xyzw,xzua,vsua,XywsvY->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/12 * einsum('xyzw,xzua,vsua,XywvsY->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,zxua,Ya,Xuyw->XY', v_aaaa, t1_aaae, t1_ae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,zxua,Yvsa,Xuvswy->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,zxua,vYsa,Xuvyws->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,zxua,vYua,Xvyw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,zxua,va,XuvYwy->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,zxua,va,XywYvu->XY', v_aaaa, t1_aaae, t1_ae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,zxua,vsYa,Xywvsu->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,zxua,vsta,XusvYwyt->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p += 1/4 * einsum('xyzw,zxua,vsta,XywtYsuv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_ccccaaaa, optimize = einsum_type)
        V1_m1p -= 1/24 * einsum('xyzw,zxua,vsua,XsvYwy->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/8 * einsum('xyzw,zxua,vsua,XsvYyw->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/12 * einsum('xyzw,zxua,vsua,XsvwyY->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/12 * einsum('xyzw,zxua,vsua,XsvywY->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p += 1/8 * einsum('xyzw,zxua,vsua,XywYsv->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/24 * einsum('xyzw,zxua,vsua,XywYvs->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/12 * einsum('xyzw,zxua,vsua,XywsvY->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)
        V1_m1p -= 1/12 * einsum('xyzw,zxua,vsua,XywvsY->XY', v_aaaa, t1_aaae, t1_aaae, rdm_cccaaa, optimize = einsum_type)

        V1 += V1_m1p

    def compute_V1__t1_p1p(mr_adc, V1):
        ## Molecular Orbitals Energies
        e_core = mr_adc.mo_energy.c

        ## One-electron integrals
        h_ca = mr_adc.h1eff.ca
        h_aa = mr_adc.h1eff.aa

        ## Two-electron integrals
        v_aaaa = mr_adc.v2e.aaaa
        v_caaa = mr_adc.v2e.caaa

        ## Amplitudes
        t1_ca = mr_adc.t1.ca
        t1_caaa = mr_adc.t1.caaa

        ## Reduced density matrices
        rdm_ca = mr_adc.rdm.ca 
        rdm_ccaa = mr_adc.rdm.ccaa
        rdm_cccaaa = mr_adc.rdm.cccaaa
        rdm_ccccaaaa = mr_adc.rdm.ccccaaaa

        V1_p1p  = 1/2 * einsum('iY,ix,Xx->XY', h_ca, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('iY,ixyz,Xxyz->XY', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iY,Xx->XY', h_ca, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p -= einsum('ix,iYxy,Xy->XY', h_ca, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iYyx,Xy->XY', h_ca, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iYyz,Xxzy->XY', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 2 * einsum('ix,ix,XY->XY', h_ca, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iy,XxYy->XY', h_ca, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iy,XyYx->XY', h_ca, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iyYx,Xy->XY', h_ca, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iyYz,Xzxy->XY', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('ix,iyxY,Xy->XY', h_ca, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p -= einsum('ix,iyxz,XyYz->XY', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('ix,iyxz,XzYy->XY', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iyzY,Xzyx->XY', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iyzw,XxyYzw->XY', h_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iyzw,XzwYxy->XY', h_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iyzx,XyYz->XY', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iyzx,XzYy->XY', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('iY,ixyz,Xyxz->XY', t1_ca, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('iYxy,ixzw,Xwyz->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('iYxy,ixzy,Xz->XY', t1_caaa, v_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('iYxy,iyzw,Xwxz->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('iYxy,iyzx,Xz->XY', t1_caaa, v_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('iYxy,izwu,Xzuyxw->XY', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('iYxy,izwx,Xzyw->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('iYxy,izwy,Xzwx->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iYyx,Xy->XY', t1_ca, v_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iYyz,Xzxy->XY', t1_ca, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('ix,ixYy,Xy->XY', t1_ca, v_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p -= einsum('ix,ixyY,Xy->XY', t1_ca, v_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p -= einsum('ix,ixyz,XyYz->XY', t1_ca, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('ix,ixyz,XzYy->XY', t1_ca, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iyYx,Xy->XY', t1_ca, v_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iyYz,Xxzy->XY', t1_ca, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iyzY,Xyzx->XY', t1_ca, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iyzw,XxzYyw->XY', t1_ca, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iyzw,XywYxz->XY', t1_ca, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iyzx,XyYz->XY', t1_ca, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ix,iyzx,XzYy->XY', t1_ca, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixYy,iyzw,Xzxw->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixYy,izwu,Xywzxu->XY', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixYy,izwy,Xwzx->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('ixyY,iyzw,Xzxw->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyY,izwu,Xywxzu->XY', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyY,izwy,Xwxz->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,iYwu,Xxuyzw->XY', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,iYwy,Xxwz->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,iYwz,Xxyw->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,iwYu,Xyzuwx->XY', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,iwYy,Xzwx->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,iwYz,Xyxw->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,iwuY,Xxwuzy->XY', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('ixyz,iwuv,XxwvYuyz->XY', t1_caaa, v_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('ixyz,iwuv,XxwvYuzy->XY', t1_caaa, v_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('ixyz,iwuv,XxwvYyuz->XY', t1_caaa, v_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('ixyz,iwuv,XxwvYyzu->XY', t1_caaa, v_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('ixyz,iwuv,XxwvYzuy->XY', t1_caaa, v_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('ixyz,iwuv,XyzuYvwx->XY', t1_caaa, v_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('ixyz,iwuv,XyzuYvxw->XY', t1_caaa, v_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('ixyz,iwuv,XyzuYwvx->XY', t1_caaa, v_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('ixyz,iwuv,XyzuYxvw->XY', t1_caaa, v_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('ixyz,iwuv,XyzuYxwv->XY', t1_caaa, v_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,iwuy,XxwYzu->XY', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,iwuy,XzuYxw->XY', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,iwuz,XxwYuy->XY', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,iwuz,XyuYwx->XY', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('ixyz,iyYw,Xzwx->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('ixyz,iyYz,Xx->XY', t1_caaa, v_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p -= einsum('ixyz,iywY,Xxwz->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('ixyz,iywu,XxuYzw->XY', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('ixyz,iywu,XzwYxu->XY', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('ixyz,iywz,XwYx->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('ixyz,iywz,XxYw->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,izYw,Xywx->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,izYy,Xx->XY', t1_caaa, v_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,izwY,Xxwy->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,izwu,XxuYyw->XY', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,izwu,XywYxu->XY', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,izwy,XwYx->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('ixyz,izwy,XxYw->XY', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,iY,ix,Xx->XY', e_core, t1_ca, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,iY,ixyz,Xxyz->XY', e_core, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,iYxy,izwu,Xwuyxz->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,iYxy,izwx,Xwyz->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,iYxy,izwy,Xwzx->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += einsum('i,iYxy,izxw,Xwyz->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += einsum('i,iYxy,izxy,Xz->XY', e_core, t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,iYxy,izyw,Xwxz->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,iYxy,izyx,Xz->XY', e_core, t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += einsum('i,ix,iYxy,Xy->XY', e_core, t1_ca, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,ix,iYyx,Xy->XY', e_core, t1_ca, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,ix,iYyz,Xxzy->XY', e_core, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += einsum('i,ix,ix,XY->XY', e_core, t1_ca, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('i,ix,iy,XxYy->XY', e_core, t1_ca, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('i,ix,iy,XyYx->XY', e_core, t1_ca, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,ix,iyYx,Xy->XY', e_core, t1_ca, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,ix,iyYz,Xzxy->XY', e_core, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += einsum('i,ix,iyxY,Xy->XY', e_core, t1_ca, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += einsum('i,ix,iyxz,XyYz->XY', e_core, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += einsum('i,ix,iyxz,XzYy->XY', e_core, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,ix,iyzY,Xzyx->XY', e_core, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,ix,iyzw,XxyYzw->XY', e_core, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,ix,iyzw,XzwYxy->XY', e_core, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,ix,iyzx,XyYz->XY', e_core, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,ix,iyzx,XzYy->XY', e_core, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('i,ixYy,izwu,Xyzuwx->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/6 * einsum('i,ixYy,izwu,Xyzuxw->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/6 * einsum('i,ixYy,izwu,Xyzwux->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('i,ixYy,izwu,Xyzwxu->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('i,ixYy,izwu,Xyzxuw->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,ixYy,izwy,Xzwx->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,ixYy,izyw,Xzxw->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/48 * einsum('i,ixyY,ixzw,Xywz->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/48 * einsum('i,ixyY,ixzw,Xyzw->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/6 * einsum('i,ixyY,izwu,Xyzuwx->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/6 * einsum('i,ixyY,izwu,Xyzxwu->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,ixyY,izwy,Xzxw->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += einsum('i,ixyY,izyw,Xzxw->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('i,ixyz,iwYu,Xxuwzy->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/12 * einsum('i,ixyz,iwYu,Xxuywz->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/6 * einsum('i,ixyz,iwYu,Xxuyzw->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/12 * einsum('i,ixyz,iwYu,Xxuzyw->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/6 * einsum('i,ixyz,iwuY,Xxuwyz->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/6 * einsum('i,ixyz,iwuY,Xxuwzy->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/6 * einsum('i,ixyz,iwuY,Xxuywz->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/6 * einsum('i,ixyz,iwuY,Xxuyzw->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/6 * einsum('i,ixyz,iwuY,Xxuzyw->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('i,ixyz,iwuv,XyzwYuxv->XY', e_core, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('i,ixyz,iwuy,XxuYzw->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('i,ixyz,iwuy,XxuwzY->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/12 * einsum('i,ixyz,iwuy,XxuzYw->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('i,ixyz,iwuy,XzwYxu->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('i,ixyz,iwuy,XzwuxY->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/12 * einsum('i,ixyz,iwuy,XzwxYu->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('i,ixyz,iwuz,XxuYwy->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('i,ixyz,iwuz,XywYux->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('i,ixyz,iwyu,XxuYzw->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/6 * einsum('i,ixyz,iwyu,XxuwzY->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/6 * einsum('i,ixyz,iwyu,XxuzYw->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('i,ixyz,iwyu,XzwYxu->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/6 * einsum('i,ixyz,iwyu,XzwuxY->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/6 * einsum('i,ixyz,iwyu,XzwxYu->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('i,ixyz,iwyz,XwYx->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('i,ixyz,iwyz,XxYw->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('i,ixyz,iwzu,XxuYyw->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('i,ixyz,iwzu,XxuwyY->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/12 * einsum('i,ixyz,iwzu,XxuyYw->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('i,ixyz,iwzu,XywYxu->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('i,ixyz,iwzu,XywuxY->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/12 * einsum('i,ixyz,iwzu,XywxYu->XY', e_core, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('i,ixyz,iwzy,XwYx->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('i,ixyz,iwzy,XxYw->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/48 * einsum('i,ixyz,ixwY,Xwyz->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/48 * einsum('i,ixyz,ixwY,Xwzy->XY', e_core, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yx,ix,iy,Xy->XY', h_aa, t1_ca, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yx,ix,iyzw,Xyzw->XY', h_aa, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yx,ixyz,iw,Xwzy->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yx,ixyz,iwuv,Xuvzyw->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yx,ixyz,iwuy,Xuzw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yx,ixyz,iwuz,Xuwy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yx,ixyz,iwyu,Xuzw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yx,ixyz,iwyz,Xw->XY', h_aa, t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yx,ixyz,iwzu,Xuyw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yx,ixyz,iwzy,Xw->XY', h_aa, t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yx,ixyz,iy,Xz->XY', h_aa, t1_caaa, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yx,ixyz,iz,Xy->XY', h_aa, t1_caaa, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yx,iyxz,iw,Xzwy->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yx,iyxz,iwuv,Xzwuyv->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yx,iyxz,iwuz,Xwuy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yx,iyxz,iwzu,Xwyu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yx,iyxz,iz,Xy->XY', h_aa, t1_caaa, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yx,iyzx,iw,Xzyw->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yx,iyzx,iwuv,Xzwyuv->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yx,iyzx,iwuz,Xwyu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('Yx,iyzx,iwzu,Xwyu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('Yx,iyzx,iz,Xy->XY', h_aa, t1_caaa, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,iYxz,iw,Xwzy->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,iYxz,iwuv,Xuvzyw->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,iYxz,iwuy,Xuzw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,iYxz,iwuz,Xuwy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xy,iYxz,iwyu,Xuzw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xy,iYxz,iwyz,Xw->XY', h_aa, t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,iYxz,iwzu,Xuyw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,iYxz,iwzy,Xw->XY', h_aa, t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p -= einsum('xy,iYxz,iy,Xz->XY', h_aa, t1_caaa, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,iYxz,iz,Xy->XY', h_aa, t1_caaa, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,iYzx,iw,Xwyz->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,iYzx,iwuv,Xuvyzw->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,iYzx,iwuy,Xuwz->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,iYzx,iwuz,Xuyw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,iYzx,iwyu,Xuzw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,iYzx,iwyz,Xw->XY', h_aa, t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xy,iYzx,iwzu,Xuyw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xy,iYzx,iwzy,Xw->XY', h_aa, t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,iYzx,iy,Xz->XY', h_aa, t1_caaa, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xy,iYzx,iz,Xy->XY', h_aa, t1_caaa, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ix,iY,Xy->XY', h_aa, t1_ca, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ix,iYzw,Xywz->XY', h_aa, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xy,ix,iy,XY->XY', h_aa, t1_ca, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ix,iz,XyYz->XY', h_aa, t1_ca, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ix,iz,XzYy->XY', h_aa, t1_ca, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ix,izYw,Xwyz->XY', h_aa, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ix,izwY,Xwzy->XY', h_aa, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ix,izwu,XwuYyz->XY', h_aa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ix,izwu,XyzYwu->XY', h_aa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,ix,izwy,XwYz->XY', h_aa, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,ix,izwy,XzYw->XY', h_aa, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xy,ix,izyw,XwYz->XY', h_aa, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xy,ix,izyw,XzYw->XY', h_aa, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixYz,iw,Xzwy->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixYz,iwuv,Xzwuyv->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixYz,iwuz,Xwuy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixYz,iwzu,Xwyu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixYz,iz,Xy->XY', h_aa, t1_caaa, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzY,iw,Xzyw->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzY,iwuv,Xzwyuv->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzY,iwuz,Xwyu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,ixzY,iwzu,Xwyu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,ixzY,iz,Xy->XY', h_aa, t1_caaa, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iY,Xyzw->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iYuv,Xwzvyu->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iYuw,Xzyu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iYuz,Xwuy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iYwu,Xzuy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iYwz,Xy->XY', h_aa, t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,ixzw,iYzu,Xwuy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,ixzw,iYzw,Xy->XY', h_aa, t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iu,XwzYyu->XY', h_aa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iu,XyuYwz->XY', h_aa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iuYv,Xyvzwu->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iuYw,Xyzu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iuYz,Xyuw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iuvY,Xyvuwz->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ixzw,iuvs,XwzuYsvy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ixzw,iuvs,XwzuYsyv->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ixzw,iuvs,XwzuYvsy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ixzw,iuvs,XwzuYvys->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ixzw,iuvs,XwzuYysv->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ixzw,iuvs,XyvsYuwz->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ixzw,iuvs,XyvsYuzw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ixzw,iuvs,XyvsYwuz->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ixzw,iuvs,XyvsYzuw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,ixzw,iuvs,XyvsYzwu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iuvw,XyvYuz->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iuvw,XzuYvy->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iuvz,XwuYyv->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iuvz,XyvYwu->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iuwY,Xyuz->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iuwv,XyvYzu->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iuwv,XzuYyv->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iuwz,XuYy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iuwz,XyYu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,ixzw,iuzY,Xyuw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,ixzw,iuzv,XwuYyv->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,ixzw,iuzv,XyvYwu->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,ixzw,iuzw,XuYy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,ixzw,iuzw,XyYu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iw,XyYz->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,ixzw,iw,XzYy->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,ixzw,iz,XwYy->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,ixzw,iz,XyYw->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izYx,iw,Xywz->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izYx,iwuv,Xywuzv->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,izYx,iwuy,Xwuz->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,izYx,iwyu,Xwzu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,izYx,iy,Xz->XY', h_aa, t1_caaa, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izwx,iY,Xzwy->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izwx,iYuv,Xywvzu->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izwx,iYuw,Xyuz->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xy,izwx,iYwu,Xyuz->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izwx,iu,XywYzu->XY', h_aa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izwx,iu,XzuYyw->XY', h_aa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izwx,iuYv,Xzvwyu->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izwx,iuYw,Xzuy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izwx,iuvY,Xzvuyw->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izwx,iuvs,XywuYzvs->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izwx,iuvs,XzvsYywu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izwx,iuvw,XyuYzv->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izwx,iuvw,XzvYyu->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,izwx,iuvy,XwuYvz->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xy,izwx,iuwY,Xzuy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xy,izwx,iuwv,XyuYzv->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xy,izwx,iuwv,XzvYyu->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('xy,izwx,iuwy,XzYu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,izwx,iuyv,XwuYzv->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,izwx,iuyv,XzvYwu->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,izwx,iuyw,XuYz->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,izwx,iuyw,XzYu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xy,izwx,iw,XyYz->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xy,izwx,iw,XzYy->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxY,iw,Xyzw->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxY,iwuv,Xywzuv->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xy,izxY,iwuy,Xwzu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xy,izxY,iwyu,Xwzu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xy,izxY,iy,Xz->XY', h_aa, t1_caaa, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iY,Xzyw->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iYuv,Xywvuz->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iYuw,Xyzu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iYwu,Xyuz->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iu,XywYuz->XY', h_aa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iu,XzuYwy->XY', h_aa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iuYv,Xzvywu->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iuYw,Xzyu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iuvY,Xzvuwy->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,izxw,iuvs,XywuYsvz->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,izxw,iuvs,XywuYszv->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,izxw,iuvs,XywuYvsz->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,izxw,iuvs,XywuYzsv->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,izxw,iuvs,XywuYzvs->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,izxw,iuvs,XzvsYuwy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,izxw,iuvs,XzvsYuyw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,izxw,iuvs,XzvsYwuy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,izxw,iuvs,XzvsYyuw->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xy,izxw,iuvs,XzvsYywu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iuvw,XyuYvz->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iuvw,XzvYuy->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iuwY,Xzuy->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iuwv,XyuYzv->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iuwv,XzvYyu->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('xy,izxw,iuyv,XwuYzv->XY', h_aa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('xy,izxw,iuyw,XzYu->XY', h_aa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iw,XyYz->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xy,izxw,iw,XzYy->XY', h_aa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('Yxyz,iwux,iu,Xywz->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwux,iv,Xyuwzv->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwux,ivst,Xyuvwzst->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwux,ivsu,Xyvwzs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('Yxyz,iwux,ivus,Xyvwzs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwux,izuv,Xywv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwux,izvs,Xyuwsv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwux,izvu,Xywv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('Yxyz,iwuy,iu,Xwxz->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwuy,iv,Xwvxzu->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwuy,ivst,Xwstxzuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwuy,ivsu,Xwsxzv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwuy,ivsx,Xwsvzu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwuy,ivsz,Xwsxvu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('Yxyz,iwuy,ivus,Xwsxzv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('Yxyz,iwuy,ivux,Xwvz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('Yxyz,iwuy,ivuz,Xwxv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwuy,ivxs,Xwsuzv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwuy,ivxu,Xwvz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwuy,ivxz,Xwuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwuy,ivzs,Xwsxuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwuy,ivzu,Xwxv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwuy,ivzx,Xwvu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwuy,ix,Xwuz->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwuy,iz,Xwxu->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('Yxyz,iwuz,iu,Xyxw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwuz,iv,Xyuxwv->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwuz,ivst,Xyuvxwst->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwuz,ivsu,Xyvxws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('Yxyz,iwuz,ivus,Xyvxws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwxu,iu,Xywz->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwxu,iv,Xyuvzw->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwxu,ivst,Xyuvszwt->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwxu,ivsu,Xyvszw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwxu,ivus,Xyvwzs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwxu,izuv,Xywv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwxu,izvs,Xyuvsw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwxu,izvu,Xyvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwxz,iu,Xyuw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwxz,iuvs,Xyuvws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwyu,iu,Xwxz->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwyu,iv,Xwvxuz->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwyu,ivst,Xwstxuvz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwyu,ivst,Xwstxvuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwyu,ivst,Xwstxvzu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwyu,ivst,Xwstxzuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwyu,ivst,Xwstxzvu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwyu,ivsu,Xwsxvz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwyu,ivsx,Xwsvuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwyu,ivsz,Xwsxuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwyu,ivus,Xwsxzv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwyu,ivux,Xwvz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwyu,ivuz,Xwxv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwyu,ivxs,Xwszuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwyu,ivxu,Xwzv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwyu,ivxz,Xwvu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('Yxyz,iwyu,ivzs,Xwsxuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('Yxyz,iwyu,ivzu,Xwxv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('Yxyz,iwyu,ivzx,Xwvu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iwyu,ix,Xwzu->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('Yxyz,iwyu,iz,Xwxu->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwzu,iu,Xyxw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwzu,iv,Xyuxvw->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwzu,ivst,Xyuvxstw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwzu,ivst,Xyuvxtsw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwzu,ivst,Xyuvxtws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwzu,ivst,Xyuvxwst->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iwzu,ivst,Xyuvxwts->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwzu,ivsu,Xyvxsw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwzu,ivus,Xyvxws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwzx,iu,Xywu->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iwzx,iuvs,Xyuwvs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,ix,iw,Xywz->XY', v_aaaa, t1_ca, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,ix,iwuv,Xywuzv->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,ix,izwu,Xywu->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,ixwu,iu,Xywz->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,ixwu,iv,Xyvuzw->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,ixwu,ivst,Xystuzwv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,ixwu,ivsu,Xysvzw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,ixwu,ivsw,Xysuzv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,ixwu,ivsz,Xysuvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,ixwu,ivus,Xyswzv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,ixwu,ivuw,Xyvz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,ixwu,ivuz,Xywv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,ixwu,ivws,Xysuzv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,ixwu,ivwu,Xyvz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,ixwu,ivwz,Xyuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,ixwu,ivzs,Xysuwv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,ixwu,ivzu,Xyvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,ixwu,ivzw,Xyuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,ixwu,iw,Xyuz->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,ixwu,iz,Xyuw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iy,iw,Xwxz->XY', v_aaaa, t1_ca, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iy,iwuv,Xuvxzw->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iy,iwux,Xuwz->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iy,iwuz,Xuxw->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iy,iwxu,Xuzw->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iy,iwxz,Xw->XY', v_aaaa, t1_ca, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p -= einsum('Yxyz,iy,iwzu,Xuxw->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('Yxyz,iy,iwzx,Xw->XY', v_aaaa, t1_ca, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iy,ix,Xz->XY', v_aaaa, t1_ca, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p -= einsum('Yxyz,iy,iz,Xx->XY', v_aaaa, t1_ca, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iywu,iu,Xwxz->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iywu,iv,Xuwxzv->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iywu,ivst,Xuwvxstz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iywu,ivst,Xuwvxszt->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iywu,ivst,Xuwvxtsz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iywu,ivst,Xuwvxtzs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iywu,ivst,Xuwvxzts->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iywu,ivsu,Xwvxsz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iywu,ivsw,Xuvxzs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iywu,ivus,Xwvxzs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,iywu,ivuw,Xvxz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iywu,ivws,Xuvxzs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iywu,ivwu,Xvxz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,iywu,iw,Xuxz->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iz,iw,Xyxw->XY', v_aaaa, t1_ca, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,iz,iwuv,Xywxuv->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,izwu,iu,Xyxw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,izwu,iv,Xyvxuw->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,izwu,ivst,Xystxuvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,izwu,ivst,Xystxvuw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,izwu,ivst,Xystxvwu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,izwu,ivst,Xystxwuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('Yxyz,izwu,ivst,Xystxwvu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,izwu,ivsu,Xysxvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,izwu,ivsw,Xysxuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,izwu,ivus,Xysxwv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('Yxyz,izwu,ivuw,Xyxv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,izwu,ivws,Xysxuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,izwu,ivwu,Xyxv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('Yxyz,izwu,iw,Xyxu->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,iYux,iu,Xzyw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iYux,iv,Xzvywu->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iYux,ivst,Xzstywuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iYux,ivsu,Xzsywv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYux,ivsw,Xzsyvu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYux,ivsy,Xzsvwu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,iYux,ivus,Xzsywv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iYux,ivuw,Xzyv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iYux,ivuy,Xzvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYux,ivws,Xzsyuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYux,ivwu,Xzyv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYux,ivwy,Xzvu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYux,ivys,Xzsuwv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYux,ivyu,Xzvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYux,ivyw,Xzuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYux,iw,Xzyu->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYux,iy,Xzuw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYux,izuv,Xvyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iYux,izvs,Xsvywu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iYux,izvu,Xvyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iYxu,iu,Xzyw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iYxu,iv,Xzvuwy->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iYxu,ivst,Xzstuywv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iYxu,ivst,Xzstwuyv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iYxu,ivst,Xzstwyuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iYxu,ivst,Xzstyuvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iYxu,ivst,Xzstyvuw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iYxu,ivst,Xzstyvwu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iYxu,ivst,Xzstywvu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iYxu,ivsu,Xzsvwy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYxu,ivsw,Xzsuvy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYxu,ivsy,Xzsuwv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iYxu,ivus,Xzsywv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYxu,ivuw,Xzyv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYxu,ivuy,Xzvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYxu,ivws,Xzsuyv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYxu,ivwu,Xzvy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYxu,ivwy,Xzuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iYxu,ivys,Xzsuwv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iYxu,ivyu,Xzvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iYxu,ivyw,Xzuv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYxu,iw,Xzuy->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iYxu,iy,Xzuw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iYxu,izuv,Xvyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iYxu,izvs,Xsvuwy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iYxu,izvu,Xvwy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iYzx,iu,Xuyw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iYzx,iuvs,Xvsywu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYzx,iuvw,Xvyu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYzx,iuvy,Xvuw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iYzx,iuwv,Xvyu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iYzx,iuwy,Xu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYzx,iuyv,Xvwu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYzx,iuyw,Xu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ca, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iYzx,iw,Xy->XY', v_aaaa, t1_caaa, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iYzx,iy,Xw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ca, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuYx,iv,Xywvuz->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuYx,ivst,Xywvszut->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuYx,ivst,Xywvuszt->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuYx,ivst,Xywvuzst->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuYx,ivst,Xywvzstu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuYx,ivst,Xywvztsu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuYx,ivst,Xywvztus->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuYx,ivst,Xywvzuts->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuYx,ivsw,Xyvsuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuYx,ivsy,Xwvszu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuYx,ivws,Xyvzus->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuYx,ivwy,Xvzu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuYx,ivys,Xwvuzs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuYx,ivyw,Xvuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuYx,iw,Xyzu->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuYx,iy,Xwuz->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuYx,izvs,Xywvus->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,iY,Xzuvwy->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,iYst,Xywvsztu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,iYst,Xywvszut->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,iYst,Xywvtzsu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,iYst,Xywvutzs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,iYst,Xywvuzst->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,iYst,Xywvztus->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,iYst,Xywvzuts->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,iYsv,Xywsuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,iuvx,iYvs,Xywsuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,is,XywvYsuz->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,is,XywvYszu->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,is,XywvYusz->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,is,XywvYzsu->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,is,XywvYzus->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,is,XzusYvwy->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,is,XzusYvyw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,is,XzusYwvy->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,is,XzusYyvw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,is,XzusYywv->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,isYt,Xzutvyws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,isYt,Xzutwvys->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,isYt,Xzutwyvs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,isYt,Xzutysvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,isYt,Xzutyswv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,isYt,Xzutyvsw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,isYt,Xzutywsv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,isYv,Xzuswy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,istY,Xzutsyvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,istY,Xzutvysw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,istY,Xzutvyws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,istY,Xzutwsyv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuvx,istY,Xzutwyvs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,istY,Xzutyswv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,istY,Xzutywsv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,istv,XywsYtuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,istv,XywsYtzu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,istv,XywsYutz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,istv,XywsYztu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,istv,XywsYzut->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,istv,XzutYswy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,istv,XzutYsyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,istv,XzutYwsy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,istv,XzutYysw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,istv,XzutYyws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,istw,XzutYsyv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isty,XzutYwsv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,iuvx,isvY,Xzuswy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isvt,XywsYtuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isvt,XywsYtzu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isvt,XywsYutz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isvt,XywsYztu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isvt,XywsYzut->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isvt,XzutYswy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isvt,XzutYsyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isvt,XzutYwsy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isvt,XzutYysw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isvt,XzutYyws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iuvx,isvw,XzuYsy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iuvx,isvy,XzuYws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,iswt,XyvsYuzt->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,iuvx,iswt,XzutYsvy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,iuvx,iswt,XzutYsyv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,iuvx,iswt,XzutYvsy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,iuvx,iswt,XzutYysv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,iuvx,iswt,XzutYyvs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,iswv,XysYuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,iswv,XzuYsy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isyt,XwvsYzut->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isyt,XzutYwvs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isyv,XwsYzu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,isyv,XzuYws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,iuvx,iv,XywYuz->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,iuvx,iv,XzuYwy->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,izst,XutsYywv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuvx,izst,XywvYuts->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,izvs,XusYyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuvx,izvs,XywYus->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxY,iv,Xywuvz->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxY,ivst,Xywvuszt->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuxY,ivsw,Xyvusz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuxY,ivsy,Xwvuzs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuxY,ivws,Xyvuzs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuxY,ivwy,Xvuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iuxY,ivys,Xwvuzs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iuxY,ivyw,Xvuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuxY,iw,Xyuz->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iuxY,iy,Xwuz->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxY,izvs,Xywuvs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,iY,Xzuywv->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,iYst,Xywvstzu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,iYst,Xywvsztu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,iYst,Xywvtzsu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,iYst,Xywvzstu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,iYst,Xywvztsu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,iYsv,Xywusz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,iYvs,Xywsuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,is,XywvYszu->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,is,XzusYwvy->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,isYt,Xzutywvs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,isYv,Xzuyws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,istY,Xzutsyvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,istY,Xzutwsvy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,istY,Xzutwyvs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,istY,Xzutysvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,istY,Xzutywvs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,istv,XywsYtzu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,istv,XzutYwsy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,isvY,Xzuswy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,isvt,XywsYtuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,isvt,XywsYtzu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,isvt,XywsYutz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,isvt,XywsYztu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,isvt,XywsYzut->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,isvt,XzutYswy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,isvt,XzutYsyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,isvt,XzutYwsy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,isvt,XzutYysw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,isvt,XzutYyws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuxv,iswt,XzutYyvs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuxv,iswv,XzuYys->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iuxv,isyt,XzutYwvs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iuxv,isyv,XzuYws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,iv,XywYuz->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,iv,XzuYwy->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,izst,XutsYvyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,izst,XutsYwvy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,izst,XutsYwyv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,izst,XutsYyvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,izst,XutsYywv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,izst,XywvYsut->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,izst,XywvYtsu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,izst,XywvYtus->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,izst,XywvYust->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxv,izst,XywvYuts->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/6 * einsum('xyzw,iuxv,izsv,XusYwy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('xyzw,iuxv,izsv,XusYyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('xyzw,iuxv,izsv,XuswyY->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('xyzw,iuxv,izsv,XusywY->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/6 * einsum('xyzw,iuxv,izsv,XywYsu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('xyzw,iuxv,izsv,XywYus->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('xyzw,iuxv,izsv,XywsuY->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('xyzw,iuxv,izsv,XywusY->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,izvs,XusYyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,iuxv,izvs,XywYus->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxz,iY,Xuyw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxz,iYvs,Xywsvu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxz,iv,XuvYwy->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxz,iv,XywYvu->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxz,ivYs,Xusywv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxz,ivsY,Xusvwy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxz,ivst,XustYwyv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,iuxz,ivst,XywvYsut->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuxz,ivsy,XusYwv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuxz,ivsy,XwvYus->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iuxz,ivys,XusYwv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iuxz,ivys,XwvYus->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iuxz,ivyw,XuYv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iuxz,iy,XuYw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,iuxz,iy,XwYu->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuzx,ivsy,XusYvw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuzx,ivsy,XwvYsu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuzx,ivys,XusYwv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuzx,ivys,XwvYus->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuzx,ivyw,XuYv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuzx,iy,XuYw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,iuzx,iy,XwYu->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ix,iY,Xzyw->XY', v_aaaa, t1_ca, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ix,iYuv,Xywvuz->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ix,iu,XywYuz->XY', v_aaaa, t1_ca, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ix,iu,XzuYwy->XY', v_aaaa, t1_ca, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ix,iuYv,Xzvywu->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ix,iuvY,Xzvuwy->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ix,iuvs,XywuYsvz->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ix,iuvs,XywuYszv->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ix,iuvs,XywuYvsz->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ix,iuvs,XywuYzsv->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ix,iuvs,XywuYzvs->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ix,iuvs,XzvsYuwy->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ix,iuvs,XzvsYuyw->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ix,iuvs,XzvsYwuy->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ix,iuvs,XzvsYyuw->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ix,iuvs,XzvsYywu->XY', v_aaaa, t1_ca, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ix,iuvw,XyuYvz->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ix,iuvw,XzvYuy->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ix,iuvy,XwuYzv->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ix,iuvy,XzvYwu->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ix,iuwv,XyuYzv->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ix,iuwv,XzvYyu->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,ix,iuyv,XwuYzv->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,ix,iuyv,XzvYwu->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ix,iw,XzYy->XY', v_aaaa, t1_ca, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= einsum('xyzw,ix,iy,XzYw->XY', v_aaaa, t1_ca, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ix,izuv,XvuYwy->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ix,izuv,XywYuv->XY', v_aaaa, t1_ca, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixYu,iu,Xzyw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixYu,iv,Xzuvwy->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixYu,ivst,Xzuvsywt->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixYu,ivst,Xzuvwsyt->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixYu,ivst,Xzuvwyst->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixYu,ivst,Xzuvystw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixYu,ivst,Xzuvytsw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixYu,ivst,Xzuvytws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixYu,ivst,Xzuvywts->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixYu,ivsu,Xzvswy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixYu,ivsz,Xuvsyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixYu,ivus,Xzvyws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixYu,ivuz,Xvyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixYu,ivzs,Xuvwys->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixYu,ivzu,Xvwy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixYu,iz,Xuwy->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ixuY,iu,Xzyw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuY,iv,Xzuywv->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuY,ivst,Xzuvywst->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuY,ivsu,Xzvyws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuY,ivsz,Xuvysw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ixuY,ivus,Xzvyws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ixuY,ivuz,Xvyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuY,ivzs,Xuvyws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuY,ivzu,Xvyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuY,iz,Xuyw->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,iY,Xywuvz->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,iYst,Xzvutyws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,iYst,Xzvuwtys->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,iYst,Xzvuwyts->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,iYst,Xzvuytws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,iYst,Xzvuywts->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,iYsu,Xzvswy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,iYsv,Xzuyws->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ixuv,iYus,Xzvswy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ixuv,iYuv,Xzyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,iYvs,Xzuswy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,iYvu,Xzyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,is,XywsYvzu->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,is,XzvuYwys->XY', v_aaaa, t1_caaa, t1_ca, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,isYt,Xywtuvzs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,isYu,Xywsvz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,isYv,Xywusz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istY,Xywtszvu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istY,Xywtvszu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istY,Xywtvzsu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istY,Xywtzsvu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istY,Xywtzvsu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istu,XywtYsvz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istu,XywtYszv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istu,XywtYvsz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istu,XywtYzsv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istu,XywtYzvs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istu,XzvsYtwy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istu,XzvsYtyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istu,XzvsYwty->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istu,XzvsYytw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,istu,XzvsYywt->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,istv,XywtYszu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,istv,XzusYwty->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ixuv,isuY,Xywsvz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,ixuv,isut,XywtYsvz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,ixuv,isut,XywtYszv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,ixuv,isut,XywtYvsz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,ixuv,isut,XywtYzsv->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,ixuv,isut,XywtYzvs->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,ixuv,isut,XzvsYtwy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,ixuv,isut,XzvsYtyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,ixuv,isut,XzvsYwty->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,ixuv,isut,XzvsYytw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/2 * einsum('xyzw,ixuv,isut,XzvsYywt->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ixuv,isuv,XywYsz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ixuv,isuv,XzsYwy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,isvY,Xywsuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,isvt,XywtYsuz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,isvt,XywtYszu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,isvt,XywtYusz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,isvt,XywtYzsu->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,isvt,XywtYzus->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,isvt,XzusYtwy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,isvt,XzusYtyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,isvt,XzusYwty->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,isvt,XzusYytw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p += 1/4 * einsum('xyzw,ixuv,isvt,XzusYywt->XY', v_aaaa, t1_caaa, t1_caaa, rdm_ccccaaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,isvu,XywYsz->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,isvu,XzsYwy->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,isvz,XusYyw->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,isvz,XywYus->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('xyzw,ixuv,iszv,XuswyY->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('xyzw,ixuv,iszv,XusywY->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('xyzw,ixuv,iszv,XywsuY->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/12 * einsum('xyzw,ixuv,iszv,XywusY->XY', v_aaaa, t1_caaa, t1_caaa, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ixuv,iu,XywYvz->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p += 1/2 * einsum('xyzw,ixuv,iu,XzvYwy->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,iv,XywYuz->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)
        V1_p1p -= 1/4 * einsum('xyzw,ixuv,iv,XzuYwy->XY', v_aaaa, t1_caaa, t1_ca, rdm_cccaaa, optimize = einsum_type)

        V1 += V1_p1p

    # V1 block: - 1/2 < Psi_0 | a^{\dag}_X a_Y [V + H^{(1)}, T - T^\dag] | Psi_0 >
    V1 = np.zeros((ncas, ncas))
    ## T - T^{\dag}: AAEE
    compute_V1__t1_m2(mr_adc, V1)
    ## T - T^{\dag}: CAEE
    compute_V1__t1_m1(mr_adc, V1)
    ## T - T^{\dag}: CCEE
    compute_V1__t1_0(mr_adc, V1)
    ## T - T^{\dag}: CCEA
    compute_V1__t1_p1(mr_adc, V1)
    ## T - T^{\dag}: CCAA
    compute_V1__t1_p2(mr_adc, V1)
    ## T - T^{\dag}: CE-CAEA
    compute_V1__t1_0p(mr_adc, V1)
    ## T - T^{\dag}: AE-AAEA
    compute_V1__t1_m1p(mr_adc, V1)
    ## T - T^{\dag}: CA-CAAA
    compute_V1__t1_p1p(mr_adc, V1)

    # V1 block: - 1/2 < Psi_0 | (a^{\dag}_X a_Y - a^{\dag}_Y a_X) [V + H^{(1)}, T - T^\dag] | Psi_0 >
    V1 -= V1.T

    tril_ind = np.tril_indices(ncas, k=-1)

    V1_sym = V1[tril_ind[0], tril_ind[1]]
    V1 = np.tile(V1_sym, 2)
    del(V1_sym)

    # Compute denominators
    evals = evals**(-1)

    # Compute T[0'']^(2) amplitudes
    S_12_V_0pp = einsum('mp,P,Pm->p', evecs, V1, S_0p_12_inv_act, optimize = einsum_type)
    S_12_V_0pp *= evals
    S_12_V_0pp = einsum('mp,p->m', evecs, S_12_V_0pp, optimize = einsum_type)
    del(V1, evals, evecs)

    ## Compute T[0'']^(2) t2_aa tensor
    t_0p = S_0p_12_inv_act @ S_12_V_0pp
    t_0p = np.split(t_0p, 2)[0]

    t2_aa = np.zeros((ncas, ncas)) 
    t2_aa[tril_ind[0], tril_ind[1]] =  t_0p
    t2_aa[tril_ind[1], tril_ind[0]] = -t_0p

    mr_adc.log.extra("Norm of T[0'']^(2):                          %20.12f" % np.linalg.norm(t2_aa))
    mr_adc.log.timer("computing T[0'']^(2) amplitudes", *cput0)

    return t2_aa

