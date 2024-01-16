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
        remove_non_cvs_variables(mr_adc)

    print("Time for computing amplitudes:                     %f sec\n" % (time.time() - start_time))

def compute_t1_amplitudes(mr_adc):

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_0p, e_p1p, e_m1p, e_0, e_p1, e_m1, e_p2, e_m2 = (0.0,) * 8

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

        if mr_adc.method_type not in ("ee", "cvs-ee"):
            mr_adc.t2.aa = np.zeros((ncas, ncas))
            mr_adc.t2.ca = np.zeros((ncore, ncas))
            mr_adc.t2.ae = np.zeros((ncas, nextern))
            mr_adc.t2.ae = np.zeros((ncas, nextern))

        # EE and CVS-EE amplitudes
        else:
            print("Computing T[-1']^(2) amplitudes...")
            sys.stdout.flush()
            mr_adc.t2.ae = compute_t2_m1p_singles(mr_adc)
            print("Norm of T[-1']^(2):                          %20.12f\n" % np.linalg.norm(mr_adc.t2.ae))
            sys.stdout.flush()
            
            mr_adc.t2.aa = np.zeros((ncas, ncas))

        mr_adc.t2.caea = np.zeros((ncore, ncas, nextern, ncas))
        mr_adc.t2.caae = np.zeros((ncore, ncas, ncas, nextern))
        mr_adc.t2.caaa = np.zeros((ncore, ncas, ncas, ncas))
        mr_adc.t2.aaae = np.zeros((ncas, ncas, ncas, nextern))
        mr_adc.t2.ccee = np.zeros((ncore, ncore, nextern, nextern))
        mr_adc.t2.ccae = np.zeros((ncore, ncore, ncas, nextern))
        mr_adc.t2.caee = np.zeros((ncore, ncas, nextern, nextern))
        mr_adc.t2.ccaa = np.zeros((ncore, ncore, ncas, ncas))
        mr_adc.t2.aaee = np.zeros((ncas, ncas, nextern, nextern))

    else:
        mr_adc.t2.ce = np.zeros((ncore, nextern))
        mr_adc.t2.ca = np.zeros((ncore, ncas))
        mr_adc.t2.aa = np.zeros((ncas, ncas))
        mr_adc.t2.ae = np.zeros((ncas, nextern))

        mr_adc.t2.caea = np.zeros((ncore, ncas, nextern, ncas))
        mr_adc.t2.caae = np.zeros((ncore, ncas, ncas, nextern))
        mr_adc.t2.caaa = np.zeros((ncore, ncas, ncas, ncas))
        mr_adc.t2.aaae = np.zeros((ncas, ncas, ncas, nextern))
        mr_adc.t2.ccee = np.zeros((ncore, ncore, nextern, nextern))
        mr_adc.t2.ccae = np.zeros((ncore, ncore, ncas, nextern))
        mr_adc.t2.caee = np.zeros((ncore, ncas, nextern, nextern))
        mr_adc.t2.ccaa = np.zeros((ncore, ncore, ncas, ncas))
        mr_adc.t2.aaee = np.zeros((ncas, ncas, nextern, nextern))

def compute_cvs_amplitudes(mr_adc):

    start_time = time.time()

    print("Computing CVS amplitudes...")
    sys.stdout.flush()

    if mr_adc.method_type == "cvs-ip":

        # Variables from kernel
        ncvs = mr_adc.ncvs

        if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):

            mr_adc.t1.xe = np.ascontiguousarray(mr_adc.t1.ce[:ncvs, :])
            mr_adc.t1.ve = np.ascontiguousarray(mr_adc.t1.ce[ncvs:, :])

            mr_adc.t1.xaea = np.ascontiguousarray(mr_adc.t1.caea[:ncvs, :, :, :])
            mr_adc.t1.vaea = np.ascontiguousarray(mr_adc.t1.caea[ncvs:, :, :, :])

            mr_adc.t1.xaae = np.ascontiguousarray(mr_adc.t1.caae[:ncvs, :, :, :])
            mr_adc.t1.vaae = np.ascontiguousarray(mr_adc.t1.caae[ncvs:, :, :, :])

            mr_adc.t1.xa = np.ascontiguousarray(mr_adc.t1.ca[:ncvs, :])
            mr_adc.t1.va = np.ascontiguousarray(mr_adc.t1.ca[ncvs:, :])

            mr_adc.t1.xaaa = np.ascontiguousarray(mr_adc.t1.caaa[:ncvs, :, :, :])
            mr_adc.t1.vaaa = np.ascontiguousarray(mr_adc.t1.caaa[ncvs:, :, :, :])

        if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.t1.xxee = np.ascontiguousarray(mr_adc.t1.ccee[:ncvs, :ncvs, :, :])
            mr_adc.t1.xvee = np.ascontiguousarray(mr_adc.t1.ccee[:ncvs, ncvs:, :, :])
            mr_adc.t1.vxee = np.ascontiguousarray(mr_adc.t1.ccee[ncvs:, :ncvs, :, :])
            mr_adc.t1.vvee = np.ascontiguousarray(mr_adc.t1.ccee[ncvs:, ncvs:, :, :])

            mr_adc.t1.xxae = np.ascontiguousarray(mr_adc.t1.ccae[:ncvs, :ncvs, :, :])
            mr_adc.t1.xvae = np.ascontiguousarray(mr_adc.t1.ccae[:ncvs, ncvs:, :, :])
            mr_adc.t1.vxae = np.ascontiguousarray(mr_adc.t1.ccae[ncvs:, :ncvs, :, :])
            mr_adc.t1.vvae = np.ascontiguousarray(mr_adc.t1.ccae[ncvs:, ncvs:, :, :])

            mr_adc.t1.xaee = np.ascontiguousarray(mr_adc.t1.caee[:ncvs, :, :, :])
            mr_adc.t1.vaee = np.ascontiguousarray(mr_adc.t1.caee[ncvs:, :, :, :])

            mr_adc.t1.xxaa = np.ascontiguousarray(mr_adc.t1.ccaa[:ncvs, :ncvs, :, :])
            mr_adc.t1.xvaa = np.ascontiguousarray(mr_adc.t1.ccaa[:ncvs, ncvs:, :, :])
            mr_adc.t1.vxaa = np.ascontiguousarray(mr_adc.t1.ccaa[ncvs:, :ncvs, :, :])
            mr_adc.t1.vvaa = np.ascontiguousarray(mr_adc.t1.ccaa[ncvs:, ncvs:, :, :])

            mr_adc.t2.xe = np.ascontiguousarray(mr_adc.t2.ce[:ncvs, :])
            mr_adc.t2.ve = np.ascontiguousarray(mr_adc.t2.ce[ncvs:, :])

            mr_adc.t2.xaea = np.ascontiguousarray(mr_adc.t2.caea[:ncvs, :, :, :])
            mr_adc.t2.vaea = np.ascontiguousarray(mr_adc.t2.caea[ncvs:, :, :, :])

            mr_adc.t2.xaae = np.ascontiguousarray(mr_adc.t2.caae[:ncvs, :, :, :])
            mr_adc.t2.vaae = np.ascontiguousarray(mr_adc.t2.caae[ncvs:, :, :, :])

            mr_adc.t2.xa = np.ascontiguousarray(mr_adc.t2.ca[:ncvs, :])
            mr_adc.t2.va = np.ascontiguousarray(mr_adc.t2.ca[ncvs:, :])

            mr_adc.t2.xaaa = np.ascontiguousarray(mr_adc.t2.caaa[:ncvs, :, :, :])
            mr_adc.t2.vaaa = np.ascontiguousarray(mr_adc.t2.caaa[ncvs:, :, :, :])

        if mr_adc.method == "mr-adc(2)-x":

            mr_adc.t2.xxee = np.ascontiguousarray(mr_adc.t2.ccee[:ncvs, :ncvs, :, :])
            mr_adc.t2.xvee = np.ascontiguousarray(mr_adc.t2.ccee[:ncvs, ncvs:, :, :])
            mr_adc.t2.vxee = np.ascontiguousarray(mr_adc.t2.ccee[ncvs:, :ncvs, :, :])
            mr_adc.t2.vvee = np.ascontiguousarray(mr_adc.t2.ccee[ncvs:, ncvs:, :, :])

            mr_adc.t2.xxae = np.ascontiguousarray(mr_adc.t2.ccae[:ncvs, :ncvs, :, :])
            mr_adc.t2.xvae = np.ascontiguousarray(mr_adc.t2.ccae[:ncvs, ncvs:, :, :])
            mr_adc.t2.vxae = np.ascontiguousarray(mr_adc.t2.ccae[ncvs:, :ncvs, :, :])

            mr_adc.t2.xaee = np.ascontiguousarray(mr_adc.t2.caee[:ncvs, :, :, :])

            mr_adc.t2.xxaa = np.ascontiguousarray(mr_adc.t2.ccaa[:ncvs, :ncvs, :, :])
            mr_adc.t2.xvaa = np.ascontiguousarray(mr_adc.t2.ccaa[:ncvs, ncvs:, :, :])

    print("Time for computing CVS amplitudes:                 %f sec\n" % (time.time() - start_time))

def remove_non_cvs_variables(mr_adc):
    'Remove core integrals, core amplitudes and RDMs not used in CVS calculations'

    # Import Prism interface
    interface = mr_adc.interface

    if mr_adc.method_type == "cvs-ip":
        del(mr_adc.h1eff.ca, mr_adc.h1eff.ce)

        if interface.with_df:
            del(mr_adc.v2e.Lce, mr_adc.v2e.Lae, mr_adc.v2e.Lee)
        else:
            del(mr_adc.v2e.ccca, mr_adc.v2e.ccce, mr_adc.v2e.ccaa, mr_adc.v2e.ccae, mr_adc.v2e.caac, mr_adc.v2e.caec,
                mr_adc.v2e.caca, mr_adc.v2e.cece, mr_adc.v2e.cace, mr_adc.v2e.caaa, mr_adc.v2e.ceae, mr_adc.v2e.caae,
                mr_adc.v2e.ceaa)

            if mr_adc.method in ("mr-adc(2)-x"):
                del(mr_adc.v2e.cccc, mr_adc.v2e.ccee, mr_adc.v2e.ceec, mr_adc.v2e.caea, mr_adc.v2e.ceee, 
                    mr_adc.v2e.caee, mr_adc.v2e.ceea)

        del(mr_adc.t1.ce, mr_adc.t1.caea, mr_adc.t1.caae,
            mr_adc.t1.ca, mr_adc.t1.caaa, mr_adc.t1.ccee,
            mr_adc.t1.ccae, mr_adc.t1.caee, mr_adc.t1.ccaa,
            mr_adc.t2.ce, mr_adc.t2.caea, mr_adc.t2.caae,
            mr_adc.t2.ca, mr_adc.t2.caaa, mr_adc.t2.ccee,
            mr_adc.t2.ccae, mr_adc.t2.caee, mr_adc.t2.ccaa)

        del(mr_adc.rdm.ccccaaaa)

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

    ## Two-electron integrals
    v_cece =  mr_adc.v2e.cece

    # Compute denominators
    d_ij = e_core[:,None] + e_core
    d_ab = e_extern[:,None] + e_extern
    D2 = -d_ij.reshape(-1,1) + d_ab.reshape(-1)
    D2 = D2.reshape((ncore, ncore, nextern, nextern))

    # Compute V tensor: - < Psi_0 | a^{\dag}_I a^{\dag}_J a_B a_A V | Psi_0>
    V1_0 =- einsum('IAJB->IJAB', v_cece, optimize = einsum_type).copy()

    # Compute T[0] t1_ccee tensor
    t1_ccee = (V1_0 / D2)

    # Compute electronic correlation energy for T[0]
    e_0  = 2 * einsum('ijab,iajb', t1_ccee, v_cece, optimize = einsum_type)
    e_0 -= einsum('ijab,jaib', t1_ccee, v_cece, optimize = einsum_type)

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
    v_cace =  mr_adc.v2e.cace

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

    ## Compute T[+1] t1_ccae tensor
    t1_ccae = einsum("IJAm,Xm->JIXA", S_12_V_p1, S_p1_12_inv_act, optimize = einsum_type).copy()

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
    nextern = mr_adc.nextern

    ## Molecular Orbitals Energies
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    ## Two-electron integrals
    v_ceae =  mr_adc.v2e.ceae

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

    # Compute R.H.S. of the equation
    ## V matrix: - < Psi_0 | a^{\dag}_I a^{\dag}_X a_B a_A V | Psi_0>
    V1_m1 =- 1/2 * einsum('IAxB,Xx->IXAB', v_ceae, rdm_ca, optimize = einsum_type)

    ## Compute denominators
    d_ab = (e_extern[:,None] + e_extern).reshape(-1)
    d_ix = (e_core[:,None] - evals).reshape(-1)

    d_abix = (d_ab[:,None] - d_ix).reshape(nextern, nextern, ncore, evals.shape[0])
    d_abix = d_abix**(-1)

    # Compute T[-1] amplitudes
    S_12_V_m1 = einsum("IXAB,Xm->ImAB", V1_m1, S_m1_12_inv_act, optimize = einsum_type)
    S_12_V_m1 = einsum("mp,ImAB->IpAB", evecs, S_12_V_m1, optimize = einsum_type)
    S_12_V_m1 = einsum("ABIp,IpAB->IpAB", d_abix, S_12_V_m1, optimize = einsum_type)
    S_12_V_m1 = einsum("mp,IpAB->ImAB", evecs, S_12_V_m1, optimize = einsum_type)

    ## Compute T[-1] t1_caee tensor
    t1_caee = einsum("ImAB,Xm->IXAB", S_12_V_m1, S_m1_12_inv_act, optimize = einsum_type).copy()

    # Compute electronic correlation energy for T[-1]
    e_m1  = 2 * einsum('ixab,iayb,xy', t1_caee, v_ceae, rdm_ca, optimize = einsum_type)
    e_m1 -= einsum('ixab,ibya,xy', t1_caee, v_ceae, rdm_ca, optimize = einsum_type)

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
    v_caca =  mr_adc.v2e.caca

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

    ## Compute T[+2] t1_ccaa tensor
    t1_ccaa = einsum("IJm,Xm->IJX", S_12_V_p2, S_p2_12_inv_act, optimize = einsum_type)
    t1_ccaa = t1_ccaa.reshape(ncore, ncore, ncas, ncas)

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

    ## Two-electron integrals
    v_aeae =  mr_adc.v2e.aeae

    ## Reduced density matrices
    rdm_ccaa = mr_adc.rdm.ccaa

    # Compute K_ccaa matrix
    K_ccaa = mr_adc_intermediates.compute_K_ccaa(mr_adc)

    # Compute S^{-1/2} matrix: Orthogonalization and overlap truncation only in the active space
    S_m2_12_inv_act = mr_adc_overlap.compute_S12_m2(mr_adc)

    # Compute K^{-1} matrix
    SKS = reduce(np.dot, (S_m2_12_inv_act.T, K_ccaa, S_m2_12_inv_act))

    evals, evecs = np.linalg.eigh(SKS)

    # Compute R.H.S. of the equation
    ## V tensor: - < Psi_0 | a^{\dag}_X a^{\dag}_Y a_B a_A V | Psi_0>
    V1_m2 =- 1/3 * einsum('xAyB,XYxy->XYAB', v_aeae, rdm_ccaa, optimize = einsum_type)
    V1_m2 -= 1/6 * einsum('xAyB,XYyx->XYAB', v_aeae, rdm_ccaa, optimize = einsum_type)
    V1_m2 = V1_m2.reshape(ncas**2, nextern, nextern)

    ## Compute denominators
    d_ab = (e_extern[:,None] + e_extern).reshape(-1)
    d_abp = (d_ab[:,None] + evals).reshape(nextern, nextern, evals.shape[0])
    d_abp = d_abp**(-1)

    # Compute T[-2] amplitudes
    S_12_V_m2 = einsum("XAB,Xm->mAB", V1_m2, S_m2_12_inv_act, optimize = einsum_type)
    S_12_V_m2 = einsum("mp,mAB->pAB", evecs, S_12_V_m2, optimize = einsum_type)
    S_12_V_m2 = einsum("ABp,pAB->pAB", d_abp, S_12_V_m2, optimize = einsum_type)
    S_12_V_m2 = einsum("mp,pAB->mAB", evecs, S_12_V_m2, optimize = einsum_type)

    ## Compute T[-2] t1_aaee tensor
    t1_aaee = einsum("mAB,Xm->XAB", S_12_V_m2, S_m2_12_inv_act, optimize = einsum_type)
    t1_aaee = t1_aaee.reshape(ncas, ncas, nextern, nextern)

    # Compute electronic correlation energy for T[-2]
    e_m2  = 1/2 * einsum('xyab,zawb,xyzw', t1_aaee, v_aeae, rdm_ccaa, optimize = einsum_type)

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

    V_0p[:,:,V_aa_aa_i:V_aa_aa_f] = V2_aa_aa.copy()
    V_0p[:,:,V_aa_bb_i:V_aa_bb_f] = V2_aa_bb.copy()

    ## Compute denominators
    d_ai = (e_extern[:,None] - e_core).reshape(-1)
    d_aip = (d_ai[:,None] + evals).reshape(nextern, ncore, -1)
    d_aip = d_aip**(-1)

    # Compute T[0'] amplitudes
    S_12_V_0p = einsum("iaP,Pm->iam", V_0p, S_0p_12_inv_act, optimize = einsum_type)
    S_12_V_0p = einsum("mp,iam->iap", evecs, S_12_V_0p, optimize = einsum_type)
    S_12_V_0p = einsum("aip,iap->iap", d_aip, S_12_V_0p, optimize = einsum_type)
    S_12_V_0p = einsum("mp,iap->iam", evecs, S_12_V_0p, optimize = einsum_type)

    ## Compute T[0'] t1_ce, t1_caea and t1_caae tensors
    t_0p = einsum("iam,Pm->iaP", S_12_V_0p, S_0p_12_inv_act, optimize = einsum_type)

    ## Build T[0'] tensors
    t1_ce = t_0p[:,:,0].copy()
    t1_caea_aaaa = t_0p[:,:,V_aa_aa_i:V_aa_aa_f].reshape(ncore, nextern, ncas, ncas).transpose(0,2,1,3)
    t1_caea_abab = t_0p[:,:,V_aa_bb_i:V_aa_bb_f].reshape(ncore, nextern, ncas, ncas).transpose(0,2,1,3)

    t1_caea = t1_caea_abab
    t1_caae = (t1_caea_abab - t1_caea_aaaa).transpose(0,1,3,2).copy()

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
    V_p1p[:,V_aaa_i:V_aaa_f] = V2_aa_aa.copy()
    V_p1p[:,V_bba_i:V_bba_f] = V2_ab_ba.copy()

    ## Compute denominators
    d_ip = (-e_core[:,None] + evals)
    d_ip = d_ip**(-1)

    # Compute T[+1'] amplitudes
    S_12_V_p1p = einsum("iP,Pm->im", V_p1p, S_p1p_12_inv_act, optimize = einsum_type)
    S_12_V_p1p = einsum("mp,im->ip", evecs, S_12_V_p1p, optimize = einsum_type)
    S_12_V_p1p *= d_ip
    S_12_V_p1p = einsum("mp,ip->im", evecs, S_12_V_p1p, optimize = einsum_type)

    ## Compute T[+1'] t1_ca and t1_caaa tensors
    t_p1p = einsum("Pm,im->iP", S_p1p_12_inv_act, S_12_V_p1p, optimize = einsum_type)

    ## Build T[+1'] tensors
    t1_ca = t_p1p[:, V_a_i:V_a_f].copy()
    t1_caaa = t_p1p[:,V_bba_i: V_bba_f].reshape(ncore, ncas, ncas, ncas).copy()

    ## Transpose indices to the conventional order
    t1_caaa = t1_caaa.transpose(0,1,3,2).copy()

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
    V_m1p[V_aaa_i:V_aaa_f, :] = V2_aa_aa.copy()
    V_m1p[V_abb_i:V_abb_f, :] = V2_ab_ba.copy()

    ## Compute denominators
    d_pa = (evals[:,None] + e_extern)
    d_pa = d_pa**(-1)

    # Compute T[-1'] amplitudes
    S_12_V_m1p = einsum("Pa,Pm->ma", V_m1p, S_m1p_12_inv_act, optimize = einsum_type)
    S_12_V_m1p = einsum("mp,ma->pa", evecs, S_12_V_m1p, optimize = einsum_type)
    S_12_V_m1p *= d_pa
    S_12_V_m1p = einsum("mp,pa->ma", evecs, S_12_V_m1p, optimize = einsum_type)

    ## Compute T[-1'] t1_ae and t1_aaea tensors
    t_m1p = einsum("Pm,ma->Pa", S_m1p_12_inv_act, S_12_V_m1p, optimize = einsum_type)

    ## Build T[-1'] tensors
    t1_ae = t_m1p[V_a_i:V_a_f, :].copy()
    t1_aaae = t_m1p[V_abb_i:V_abb_f, :].reshape(ncas, ncas, ncas, nextern).copy()
    t1_aaae = t1_aaae.transpose(1,0,2,3)

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
    v_ccee = mr_adc.v2e.ccee
    
    v_caac = mr_adc.v2e.caac
    v_caec = mr_adc.v2e.caec
    v_ceec = mr_adc.v2e.ceec
    
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

    ## Amplitudes
    t1_ce = mr_adc.t1.ce
    t1_caea = mr_adc.t1.caea
    t1_caae = mr_adc.t1.caae

    t1_ca = mr_adc.t1.ca
    t1_caaa = mr_adc.t1.caaa

    t1_ae   = mr_adc.t1.ae
    t1_aaae = mr_adc.t1.aaae

    t1_ccee = mr_adc.t1.ccee
    t1_ccae = mr_adc.t1.ccae
    t1_ccaa = mr_adc.t1.ccaa
    t1_caee = mr_adc.t1.caee
    t1_aaee = mr_adc.t1.aaee

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Compute R.H.S. of the equation
    # V1 block: - 1/2 < Psi_0 | a^{\dag}_I a_A [V + H^{(1)}, T - T^\dag] | Psi_0 >
    V1  = einsum('Ix,xA->IA', h_ca, t1_ae, optimize = einsum_type)
    V1 -= 2 * einsum('ia,IiAa->IA', h_ce, t1_ccee, optimize = einsum_type)
    V1 += einsum('ia,iIAa->IA', h_ce, t1_ccee, optimize = einsum_type)
    V1 += einsum('ix,IixA->IA', h_ca, t1_ccae, optimize = einsum_type)
    V1 -= 2 * einsum('ix,iIxA->IA', h_ca, t1_ccae, optimize = einsum_type)
    V1 -= einsum('xA,Ix->IA', h_ae, t1_ca, optimize = einsum_type)
    # V1 += einsum('Iiab,iaAb->IA', t1_ccee, v_ceee, optimize = einsum_type)
    # V1 -= 2 * einsum('Iiab,ibAa->IA', t1_ccee, v_ceee, optimize = einsum_type)
    if isinstance(v_ceee, type(None)):
        chnk_size = mr_adc_integrals.calculate_chunk_size(mr_adc)
    else:
        chnk_size = ncore

    a = 0
    for p in range(0, ncore, chnk_size):
        if interface.with_df:
            v_ceee = mr_adc_integrals.get_oeee_df(mr_adc, mr_adc.v2e.Lce, mr_adc.v2e.Lee, p, chnk_size).reshape(-1, nextern, nextern, nextern)
        else:
            v_ceee = mr_adc_integrals.unpack_v2e_oeee(mr_adc.v2e.ceee, nextern)

        k = v_ceee.shape[0]

        V1 += einsum('Iiab,iaAb->IA', t1_ccee[:,a:a+k], v_ceee, optimize = einsum_type)
        V1 -= 2 * einsum('Iiab,ibAa->IA', t1_ccee[:,a:a+k], v_ceee, optimize = einsum_type)

        del v_ceee
        a += k
    V1 -= 2 * einsum('Iixa,iaAx->IA', t1_ccae, v_ceea, optimize = einsum_type)
    V1 += einsum('Iixa,ixAa->IA', t1_ccae, v_caee, optimize = einsum_type)
    V1 += einsum('Iixy,ixAy->IA', t1_ccaa, v_caea, optimize = einsum_type)
    V1 -= 2 * einsum('Iixy,iyAx->IA', t1_ccaa, v_caea, optimize = einsum_type)
    V1 += einsum('iIxa,iaAx->IA', t1_ccae, v_ceea, optimize = einsum_type)
    V1 -= 2 * einsum('iIxa,ixAa->IA', t1_ccae, v_caee, optimize = einsum_type)
    V1 -= 2 * einsum('ia,IAai->IA', t1_ce, v_ceec, optimize = einsum_type)
    V1 -= 2 * einsum('ia,IAia->IA', t1_ce, v_cece, optimize = einsum_type)
    V1 += einsum('ia,iAIa->IA', t1_ce, v_cece, optimize = einsum_type)
    V1 += einsum('ia,iIAa->IA', t1_ce, v_ccee, optimize = einsum_type)
    V1 += 2 * einsum('ijAa,iIja->IA', t1_ccee, v_ccce, optimize = einsum_type)
    V1 -= einsum('ijAa,jIia->IA', t1_ccee, v_ccce, optimize = einsum_type)
    V1 -= einsum('ijxA,iIjx->IA', t1_ccae, v_ccca, optimize = einsum_type)
    V1 += 2 * einsum('ijxA,jIix->IA', t1_ccae, v_ccca, optimize = einsum_type)
    V1 += einsum('ix,IixA->IA', t1_ca, v_ccae, optimize = einsum_type)
    V1 += einsum('ix,IxiA->IA', t1_ca, v_cace, optimize = einsum_type)
    V1 -= 2 * einsum('ix,ixAI->IA', t1_ca, v_caec, optimize = einsum_type)
    V1 -= 2 * einsum('ix,ixIA->IA', t1_ca, v_cace, optimize = einsum_type)
    V1 -= einsum('A,IiAa,ia->IA', e_extern, t1_ccee, t1_ce, optimize = einsum_type)
    V1 += 1/2 * einsum('A,IixA,ix->IA', e_extern, t1_ccae, t1_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('A,iIAa,ia->IA', e_extern, t1_ccee, t1_ce, optimize = einsum_type)
    V1 -= einsum('A,iIxA,ix->IA', e_extern, t1_ccae, t1_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,xA,Ix->IA', e_extern, t1_ae, t1_ca, optimize = einsum_type)
    V1 += einsum('I,IiAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,IixA,ix->IA', e_core, t1_ccae, t1_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,iIAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
    V1 += einsum('I,iIxA,ix->IA', e_core, t1_ccae, t1_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,xA,Ix->IA', e_core, t1_ae, t1_ca, optimize = einsum_type)
    V1 -= 2 * einsum('a,ia,IiAa->IA', e_extern, t1_ce, t1_ccee, optimize = einsum_type)
    V1 += einsum('a,ia,iIAa->IA', e_extern, t1_ce, t1_ccee, optimize = einsum_type)
    V1 += einsum('i,IiAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,IixA,ix->IA', e_core, t1_ccae, t1_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iIAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
    V1 += einsum('i,iIxA,ix->IA', e_core, t1_ccae, t1_ca, optimize = einsum_type)
    V1 += einsum('i,ia,IiAa->IA', e_core, t1_ce, t1_ccee, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ia,iIAa->IA', e_core, t1_ce, t1_ccee, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ix,IixA->IA', e_core, t1_ca, t1_ccae, optimize = einsum_type)
    V1 += einsum('i,ix,iIxA->IA', e_core, t1_ca, t1_ccae, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ix,xyzA,zy->IA', h_ca, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 += einsum('Ix,yxzA,zy->IA', h_ca, t1_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,IiyA,xy->IA', h_ca, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('ix,iIyA,xy->IA', h_ca, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xA,Iyxz,yz->IA', h_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xA,Iyzx,yz->IA', h_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xa,IyAa,xy->IA', h_ae, t1_caee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xa,IyaA,xy->IA', h_ae, t1_caee, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,IixA,iy->IA', h_aa, t1_ccae, t1_ca, optimize = einsum_type)
    V1 -= 2 * einsum('xy,iIxA,iy->IA', h_aa, t1_ccae, t1_ca, optimize = einsum_type)
    V1 += einsum('xy,xA,Iy->IA', h_aa, t1_ae, t1_ca, optimize = einsum_type)
    V1 -= 2 * einsum('IiAa,iaxy,yx->IA', t1_ccee, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('IiAa,ixya,xy->IA', t1_ccee, v_caae, rdm_ca, optimize = einsum_type)
    V1 += einsum('IixA,ixyz,zy->IA', t1_ccae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('IixA,iyzw,xzyw->IA', t1_ccae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('IixA,iyzx,yz->IA', t1_ccae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Iixa,iaAy,xy->IA', t1_ccae, v_ceea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixa,iyAa,xy->IA', t1_ccae, v_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixy,ixAz,yz->IA', t1_ccaa, v_caea, rdm_ca, optimize = einsum_type)
    V1 += einsum('Iixy,iyAz,xz->IA', t1_ccaa, v_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixy,izAw,xywz->IA', t1_ccaa, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Iixy,izAx,yz->IA', t1_ccaa, v_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Iixy,izAy,xz->IA', t1_ccaa, v_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ix,xyzA,yz->IA', t1_ca, v_aaae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Ix,yzxA,zy->IA', t1_ca, v_aaae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('IxAa,yzwa,xzwy->IA', t1_caee, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('IxaA,yzwa,xzwy->IA', t1_caee, v_aaae, rdm_ccaa, optimize = einsum_type)
    # V1 += 1/2 * einsum('Ixab,yaAb,xy->IA', t1_caee, v_aeee, rdm_ca, optimize = einsum_type)
    # V1 -= einsum('Ixab,ybAa,xy->IA', t1_caee, v_aeee, rdm_ca, optimize = einsum_type)
    if not isinstance(v_aeee, type(None)):
        chnk_size = ncas

    a = 0
    for p in range(0, ncas, chnk_size):
        if interface.with_df:
            v_aeee = mr_adc_integrals.get_oeee_df(mr_adc, mr_adc.v2e.Lae, mr_adc.v2e.Lee, p, chnk_size).reshape(-1, nextern, nextern, nextern)
        else:
            v_aeee = mr_adc_integrals.unpack_v2e_oeee(mr_adc.v2e.aeee, nextern)

        k = v_aeee.shape[0]
        V1 += 1/2 * einsum('Ixab,yaAb,xy->IA', t1_caee, v_aeee, rdm_ca[:,a:a+k], optimize = einsum_type)
        V1 -= einsum('Ixab,ybAa,xy->IA', t1_caee, v_aeee, rdm_ca[:,a:a+k], optimize = einsum_type)

        del v_aeee
        a += k
    V1 += 1/2 * einsum('Ixay,yAaz,xz->IA', t1_caea, v_aeea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixay,zAaw,xzyw->IA', t1_caea, v_aeea, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('Ixay,zwAa,xwyz->IA', t1_caea, v_aaee, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('Ixay,zyAa,xz->IA', t1_caea, v_aaee, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Ixya,yAaz,xz->IA', t1_caae, v_aeea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixya,zAaw,xzwy->IA', t1_caae, v_aeea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixya,zwAa,xwyz->IA', t1_caae, v_aaee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixya,zyAa,xz->IA', t1_caae, v_aaee, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Ixyz,wuyA,xwzu->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixyz,wuzA,xwyu->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ixyz,wxuA,yzuw->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixyz,ywuA,xuzw->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixyz,ywzA,xw->IA', t1_caaa, v_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixyz,zwuA,xuwy->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('Ixyz,zwyA,xw->IA', t1_caaa, v_aaae, rdm_ca, optimize = einsum_type)
    V1 += einsum('iIAa,iaxy,yx->IA', t1_ccee, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('iIAa,ixya,xy->IA', t1_ccee, v_caae, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('iIxA,ixyz,zy->IA', t1_ccae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('iIxA,iyzw,xzyw->IA', t1_ccae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('iIxA,iyzx,yz->IA', t1_ccae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('iIxa,iaAy,xy->IA', t1_ccae, v_ceea, rdm_ca, optimize = einsum_type)
    V1 += einsum('iIxa,iyAa,xy->IA', t1_ccae, v_caee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ijxA,iIjy,xy->IA', t1_ccae, v_ccca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ijxA,jIiy,xy->IA', t1_ccae, v_ccca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,IiyA,xy->IA', t1_ca, v_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ix,IyiA,xy->IA', t1_ca, v_cace, rdm_ca, optimize = einsum_type)
    V1 += einsum('ix,iyAI,xy->IA', t1_ca, v_caec, rdm_ca, optimize = einsum_type)
    V1 += einsum('ix,iyIA,xy->IA', t1_ca, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixAa,Iyai,xy->IA', t1_caee, v_caec, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixAa,iIya,xy->IA', t1_caee, v_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixAy,Iiyz,xz->IA', t1_caea, v_ccaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixAy,Iizw,ywxz->IA', t1_caea, v_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixAy,Izwi,yzxw->IA', t1_caea, v_caac, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixAy,Izyi,xz->IA', t1_caea, v_caac, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixaA,Iyai,xy->IA', t1_caee, v_caec, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixaA,iIya,xy->IA', t1_caee, v_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('ixay,IAai,xy->IA', t1_caea, v_ceec, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('ixay,IAia,yx->IA', t1_caea, v_cece, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixay,iAIa,yx->IA', t1_caea, v_cece, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixay,iIAa,xy->IA', t1_caea, v_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyA,Iiyz,xz->IA', t1_caae, v_ccaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyA,Iizw,ywxz->IA', t1_caae, v_ccaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyA,Izwi,yzwx->IA', t1_caae, v_caac, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ixyA,Izyi,xz->IA', t1_caae, v_caac, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixya,IAai,xy->IA', t1_caae, v_ceec, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixya,IAia,yx->IA', t1_caae, v_cece, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixya,iAIa,yx->IA', t1_caae, v_cece, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixya,iIAa,xy->IA', t1_caae, v_ccee, rdm_ca, optimize = einsum_type)
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
    V1 -= einsum('xa,IAay,xy->IA', t1_ae, v_ceea, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xa,IAya,xy->IA', t1_ae, v_ceae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xa,IayA,xy->IA', t1_ae, v_ceae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xa,IyaA,xy->IA', t1_ae, v_caee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyAa,Izaw,xyzw->IA', t1_aaee, v_caea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzA,Iwux,zwuy->IA', t1_aaae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzA,Iwuy,zwxu->IA', t1_aaae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzA,Iwzu,yxwu->IA', t1_aaae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzA,Ixwu,zuyw->IA', t1_aaae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzA,Ixwy,zw->IA', t1_aaae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzA,Iywu,zuxw->IA', t1_aaae, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzA,Iywx,zw->IA', t1_aaae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xyza,IAaw,zwxy->IA', t1_aaae, v_ceea, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyza,IAwa,zwxy->IA', t1_aaae, v_ceae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,IawA,zwxy->IA', t1_aaae, v_ceae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyza,IwaA,zwxy->IA', t1_aaae, v_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('A,IiAa,ixay,yx->IA', e_extern, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('A,IiAa,ixya,yx->IA', e_extern, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,IixA,iy,xy->IA', e_extern, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('A,IixA,iyxz,zy->IA', e_extern, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,IixA,iyzw,xyzw->IA', e_extern, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,IixA,iyzx,zy->IA', e_extern, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,IxAa,ya,xy->IA', e_extern, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,IxAa,yzwa,xwzy->IA', e_extern, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,IxaA,ya,xy->IA', e_extern, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,IxaA,yzwa,xwzy->IA', e_extern, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('A,iIAa,ixay,yx->IA', e_extern, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,iIAa,ixya,yx->IA', e_extern, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
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
    V1 += einsum('I,IiAa,ixay,yx->IA', e_core, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,IiAa,ixya,yx->IA', e_core, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,IixA,iy,xy->IA', e_core, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,IixA,iyxz,zy->IA', e_core, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,IixA,iyzw,xyzw->IA', e_core, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,IixA,iyzx,zy->IA', e_core, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,IxAa,ya,xy->IA', e_core, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,IxAa,yzwa,xwzy->IA', e_core, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,IxaA,ya,xy->IA', e_core, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,IxaA,yzwa,xwzy->IA', e_core, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,iIAa,ixay,yx->IA', e_core, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,iIAa,ixya,yx->IA', e_core, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
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
    V1 -= 2 * einsum('a,ixay,IiAa,yx->IA', e_extern, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,ixay,iIAa,yx->IA', e_extern, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,ixya,IiAa,yx->IA', e_extern, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,ixya,iIAa,yx->IA', e_extern, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= einsum('a,xa,IyAa,xy->IA', e_extern, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xa,IyaA,xy->IA', e_extern, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= einsum('a,xyza,IwAa,zwxy->IA', e_extern, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('a,xyza,IwaA,zwxy->IA', e_extern, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('i,IiAa,ixay,xy->IA', e_core, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,IiAa,ixya,xy->IA', e_core, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,IixA,iy,xy->IA', e_core, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,IixA,iyxz,yz->IA', e_core, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,IixA,iyzw,xyzw->IA', e_core, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('i,IixA,iyzx,yz->IA', e_core, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iIAa,ixay,xy->IA', e_core, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,iIAa,ixya,xy->IA', e_core, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,iIxA,iy,xy->IA', e_core, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += einsum('i,iIxA,iyxz,yz->IA', e_core, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,iIxA,iyzw,xyzw->IA', e_core, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iIxA,iyzx,yz->IA', e_core, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('i,ixay,IiAa,xy->IA', e_core, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixay,iIAa,xy->IA', e_core, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixya,IiAa,xy->IA', e_core, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixya,iIAa,xy->IA', e_core, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyz,IiyA,xz->IA', e_core, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('i,ixyz,IizA,xy->IA', e_core, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('i,ixyz,iIyA,xz->IA', e_core, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,ixyz,iIzA,xy->IA', e_core, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,IixA,iz,yz->IA', h_aa, t1_ccae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,IixA,izwu,yzwu->IA', h_aa, t1_ccae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,IixA,izwy,wz->IA', h_aa, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,IixA,izyw,wz->IA', h_aa, t1_ccae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,IxAa,za,yz->IA', h_aa, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,IxAa,zwua,yuwz->IA', h_aa, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,IxaA,za,yz->IA', h_aa, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,IxaA,zwua,yuwz->IA', h_aa, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
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
    V1 += einsum('xy,ixaz,IiAa,yz->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixaz,iIAa,yz->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixza,IiAa,yz->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixza,iIAa,yz->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixzw,IiuA,yuwz->IA', h_aa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ixzw,IiwA,yz->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixzw,IizA,yw->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixzw,iIuA,yuwz->IA', h_aa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ixzw,iIwA,yz->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,ixzw,iIzA,yw->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xy,izax,IiAa,yz->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izax,iIAa,yz->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izwx,IiuA,ywzu->IA', h_aa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izwx,IiwA,yz->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izwx,iIuA,ywzu->IA', h_aa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,izwx,iIwA,yz->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izxa,IiAa,yz->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izxa,iIAa,yz->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izxw,IiuA,ywuz->IA', h_aa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,izxw,IiwA,yz->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izxw,iIuA,ywuz->IA', h_aa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,izxw,iIwA,yz->IA', h_aa, t1_caaa, t1_ccae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,xA,Izwy,zw->IA', h_aa, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,xA,Izyw,zw->IA', h_aa, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xa,IzAa,yz->IA', h_aa, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xa,IzaA,yz->IA', h_aa, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,xzwA,Iuvy,wvuz->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzwA,Iuvz,yuwv->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,xzwA,Iuyv,wvzu->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,xzwA,Iuyz,wu->IA', h_aa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xzwA,Iuzv,yuwv->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,xzwA,Iuzy,wu->IA', h_aa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,xzwA,Iwuv,yzvu->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,xzwA,Iy,wz->IA', h_aa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xzwA,Iz,yw->IA', h_aa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xzwa,IuAa,yzwu->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzwa,IuaA,yzwu->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxA,Iuvw,yvzu->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxA,Iuvz,yvuw->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zwxA,Iuwv,yvzu->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zwxA,Iuwz,yu->IA', h_aa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxA,Iuzv,yvwu->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxA,Iuzw,yu->IA', h_aa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zwxA,Iw,yz->IA', h_aa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zwxA,Iyuv,wzuv->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxA,Iz,yw->IA', h_aa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zwxa,IuAa,yuzw->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zwxa,IuaA,yuzw->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zxwA,Iuvy,wvzu->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwA,Iuvz,yuvw->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,zxwA,Iuyv,wvzu->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,zxwA,Iuyz,wu->IA', h_aa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwA,Iuzv,yuwv->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zxwA,Iuzy,wu->IA', h_aa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zxwA,Iwuv,yzuv->IA', h_aa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xy,zxwA,Iy,wz->IA', h_aa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwA,Iz,yw->IA', h_aa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,zxwa,IuAa,yzuw->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxwa,IuaA,yzuw->IA', h_aa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ixay,zAaw,wz,xy->IA', t1_caea, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 += einsum('Ixay,zwAa,zw,xy->IA', t1_caea, v_aaee, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('Ixya,zAaw,wz,xy->IA', t1_caae, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ixya,zwAa,zw,xy->IA', t1_caae, v_aaee, rdm_ca, rdm_ca, optimize = einsum_type)
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
    V1 += 1/2 * einsum('xyzw,IxAa,ua,zuwy->IA', v_aaaa, t1_caee, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IxAa,uvsa,zvuswy->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IxAa,uvsa,zvusyw->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IxAa,uvsa,zvuwsy->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,IxAa,uvsa,zvuwys->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IxAa,uvsa,zvuysw->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IxAa,uvsa,zvuyws->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,IxAa,uvza,ywvu->IA', v_aaaa, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,IxaA,ua,zuwy->IA', v_aaaa, t1_caee, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxaA,uvsa,zvuswy->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxaA,uvsa,zvusyw->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxaA,uvsa,zvuwsy->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,IxaA,uvsa,zvuwys->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxaA,uvsa,zvuysw->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxaA,uvsa,zvuyws->IA', v_aaaa, t1_caee, t1_aaae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,IxaA,uvza,ywvu->IA', v_aaaa, t1_caee, t1_aaae, rdm_ccaa, optimize = einsum_type)
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
    V1 -= einsum('xyzw,iuax,IiAa,zuwy->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuax,iIAa,zuwy->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
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
    V1 += 1/2 * einsum('xyzw,iuxa,IiAa,zuwy->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxa,iIAa,zuwy->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,IisA,zuswvy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxv,IivA,zuwy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxv,iIsA,zuswvy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxv,iIvA,zuwy->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iuxz,IivA,ywvu->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iuxz,iIvA,ywvu->IA', v_aaaa, t1_caaa, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ix,IiuA,zuwy->IA', v_aaaa, t1_ca, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ix,iIuA,zuwy->IA', v_aaaa, t1_ca, t1_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,ixau,IiAa,zuwy->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixau,iIAa,zuwy->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ixua,IiAa,zuwy->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ixua,iIAa,zuwy->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
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
    V1 -= 1/2 * einsum('xyzw,uvxa,IsAa,zuvwys->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,uvxa,IsaA,zuvwys->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
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
    V1 += 1/2 * einsum('xyzw,uxva,IsAa,zvswuy->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxva,IsaA,zvswuy->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xA,Iuvw,zvuy->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xA,Iuvy,zvwu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xA,Iuwv,zvyu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xA,Iuwy,zu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,xA,Iuyv,zvwu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,xA,Iuyw,zu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xA,Iw,zy->IA', v_aaaa, t1_ae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,xA,Iy,zw->IA', v_aaaa, t1_ae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xA,Izuv,ywuv->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xa,IuAa,zuwy->IA', v_aaaa, t1_ae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xa,IuaA,zuwy->IA', v_aaaa, t1_ae, t1_caee, rdm_ccaa, optimize = einsum_type)
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
    V1 -= 1/6 * einsum('xyzw,xuva,IsAa,zvsuwy->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuva,IsAa,zvsuyw->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuva,IsAa,zvswuy->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,xuva,IsAa,zvswyu->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuva,IsAa,zvsyuw->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuva,IsAa,zvsywu->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,IsaA,zvsuwy->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,IsaA,zvsuyw->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,IsaA,zvswuy->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuva,IsaA,zvswyu->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,IsaA,zvsyuw->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuva,IsaA,zvsywu->IA', v_aaaa, t1_aaae, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,zxuA,Iuvs,ywvs->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,zxuA,Ivsw,yvsu->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,zxuA,Ivsy,wvus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,zxuA,Ivws,yvus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,zxuA,Ivwy,uv->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,zxuA,Ivys,wvus->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,zxuA,Ivyw,uv->IA', v_aaaa, t1_aaae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,zxuA,Iw,yu->IA', v_aaaa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyzw,zxuA,Iy,wu->IA', v_aaaa, t1_aaae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,zxua,IvAa,ywvu->IA', v_aaaa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,zxua,IvaA,ywvu->IA', v_aaaa, t1_aaae, t1_caee, rdm_ccaa, optimize = einsum_type)

    ## Compute denominators
    d_ai = (e_extern[:,None] - e_core)
    d_ai = d_ai**(-1)

    # Compute T2[0'] t2_ce amplitudes
    t2_ce = einsum("ai,ia->ia", d_ai, V1, optimize = einsum_type)

    return t2_ce

def compute_t2_m1p_singles(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
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
    V1 -= einsum('ixab,iaAb,Xx->XA', t1_caee, v_ceee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixab,ibAa,Xx->XA', t1_caee, v_ceee, rdm_ca, optimize = einsum_type)
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
    V1 -= 1/2 * einsum('xyab,zbAa,Xzxy->XA', t1_aaee, v_aeee, rdm_ccaa, optimize = einsum_type)
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

    S_12_V = np.einsum("Pa,Pm->ma", V1, S_m1_12_inv_act)
    S_12_V = np.einsum("mp,ma->pa", evecs, S_12_V)

    # Compute denominators
    d_pa = (evals[:,None] + e_extern)
    d_pa = d_pa**(-1)

    S_12_V *= d_pa
    S_12_V = np.einsum("mp,pa->ma", evecs, S_12_V)

    # Compute T2[-1'] t2_ae amplitudes
    t2_ae = np.einsum("Pm,ma->Pa", S_m1_12_inv_act, S_12_V)

    return t2_ae
