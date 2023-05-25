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

def transform_integrals_1e(mr_adc):

    start_time = time.time()

    print("Transforming 1e integrals to MO basis...")
    sys.stdout.flush()

    mo = mr_adc.mo

    mr_adc.h1e = reduce(np.dot, (mo.T, mr_adc.interface.h1e_ao, mo))

    if mr_adc.method_type in ('ee','cvs-ee'):

        sys.stdout.flush()
        mr_adc.dip_mom = np.zeros((3, mr_adc.nmo, mr_adc.nmo))

        # Dipole moments
        for i in range(3):
            mr_adc.dip_mom[i] = reduce(np.dot, (mo.T, mr_adc.interface.dip_mom_ao[i], mo))

    print("Time for transforming 1e integrals:                %f sec\n" % (time.time() - start_time))

def transform_2e_chem_incore(interface, mo_1, mo_2, mo_3, mo_4):
    'Two-electron integral transformation in Chemists notation'

    nmo_1 = mo_1.shape[1]
    nmo_2 = mo_2.shape[1]
    nmo_3 = mo_3.shape[1]
    nmo_4 = mo_4.shape[1]

    v2e = interface.transform_2e_chem_incore(interface.v2e_ao, (mo_1, mo_2, mo_3, mo_4), compact=False)
    v2e = v2e.reshape(nmo_1, nmo_2, nmo_3, nmo_4)

    return np.ascontiguousarray(v2e)

def compute_effective_1e(mr_adc, h1e_pq, v2e_ccpq, v2e_cpqc):
    'Effective one-electron integrals'

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    h1eff  = np.ascontiguousarray(h1e_pq)
    h1eff += 2.0 * einsum('rrpq->pq', v2e_ccpq, optimize = einsum_type)
    h1eff -= einsum('rpqr->pq', v2e_cpqc, optimize = einsum_type)

    return h1eff

def transform_integrals_2e_incore(mr_adc):

    start_time = time.time()

    print("Transforming 2e integrals to MO basis (in-core)...")
    sys.stdout.flush()

    # Einsum definition from kernel
    interface = mr_adc.interface

    # Variables from kernel
    ncore = mr_adc.ncore
    nocc = mr_adc.nocc

    mo = mr_adc.mo
    mo_c = mo[:, :ncore].copy()
    mo_a = mo[:, ncore:nocc].copy()
    mo_e = mo[:, nocc:].copy()

    mr_adc.v2e.aaaa = transform_2e_chem_incore(interface, mo_a, mo_a, mo_a, mo_a)

    if mr_adc.method_type == "ip" or mr_adc.method_type == "ea" or mr_adc.method_type == "cvs-ip":
        if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.ccca = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_a)
            mr_adc.v2e.ccce = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_e)

            mr_adc.v2e.ccaa = transform_2e_chem_incore(interface, mo_c, mo_c, mo_a, mo_a)
            mr_adc.v2e.ccae = transform_2e_chem_incore(interface, mo_c, mo_c, mo_a, mo_e)

            mr_adc.v2e.caac = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_c)
            mr_adc.v2e.caec = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_c)

            mr_adc.v2e.caca = transform_2e_chem_incore(interface, mo_c, mo_a, mo_c, mo_a)
            mr_adc.v2e.cece = transform_2e_chem_incore(interface, mo_c, mo_e, mo_c, mo_e)
            mr_adc.v2e.cace = transform_2e_chem_incore(interface, mo_c, mo_a, mo_c, mo_e)

            mr_adc.v2e.caaa = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_a)
            mr_adc.v2e.ceae = transform_2e_chem_incore(interface, mo_c, mo_e, mo_a, mo_e)
            mr_adc.v2e.caae = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_e)
            mr_adc.v2e.ceaa = transform_2e_chem_incore(interface, mo_c, mo_e, mo_a, mo_a)

            mr_adc.v2e.aaae = transform_2e_chem_incore(interface, mo_a, mo_a, mo_a, mo_e)

        if mr_adc.method in ("mr-adc(2)-x"):
            mr_adc.v2e.cccc = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_c)
            mr_adc.v2e.eeee = transform_2e_chem_incore(interface, mo_e, mo_e, mo_e, mo_e)

            mr_adc.v2e.ccee = transform_2e_chem_incore(interface, mo_c, mo_c, mo_e, mo_e)
            mr_adc.v2e.ceec = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_c)

            mr_adc.v2e.caea = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_a)
            mr_adc.v2e.ceee = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_e)
            mr_adc.v2e.caee = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_e)

            mr_adc.v2e.ceea = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_a)

            mr_adc.v2e.aeae = transform_2e_chem_incore(interface, mo_a, mo_e, mo_a, mo_e)

            mr_adc.v2e.aeee = transform_2e_chem_incore(interface, mo_a, mo_e, mo_e, mo_e)

            mr_adc.v2e.aaee = transform_2e_chem_incore(interface, mo_a, mo_a, mo_e, mo_e)

            mr_adc.v2e.aeea = transform_2e_chem_incore(interface, mo_a, mo_e, mo_e, mo_a)

    # Effective one-electron integrals
    v2e_ccac = mr_adc.v2e.ccca.transpose(1,0,3,2)
    v2e_ccec = mr_adc.v2e.ccce.transpose(1,0,3,2)

    mr_adc.h1eff.ca = compute_effective_1e(mr_adc, mr_adc.h1e[:ncore, ncore:nocc], mr_adc.v2e.ccca, v2e_ccac)
    mr_adc.h1eff.ce = compute_effective_1e(mr_adc, mr_adc.h1e[:ncore, nocc:], mr_adc.v2e.ccce, v2e_ccec)
    mr_adc.h1eff.aa = compute_effective_1e(mr_adc, mr_adc.h1e[ncore:nocc, ncore:nocc], mr_adc.v2e.ccaa, mr_adc.v2e.caac)
    mr_adc.h1eff.ae = compute_effective_1e(mr_adc, mr_adc.h1e[ncore:nocc, nocc:], mr_adc.v2e.ccae, mr_adc.v2e.caec)

    # Store diagonal elements of the generalized Fock operator
    mr_adc.mo_energy.c = mr_adc.interface.mo_energy[:ncore]
    mr_adc.mo_energy.e = mr_adc.interface.mo_energy[nocc:]

    print("Time for transforming integrals:                   %f sec\n" % (time.time() - start_time))

def compute_cvs_integrals_2e_incore(mr_adc):

    start_time = time.time()

    print("Computing CVS integrals to MO basis (in-core)...")
    sys.stdout.flush()

    # Variables from kernel
    ncvs = mr_adc.ncvs

    if mr_adc.method_type == "cvs-ip":
        if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.xxxa = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs, :ncvs, :ncvs, :])
            mr_adc.v2e.xxva = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs, :ncvs, ncvs:, :])
            mr_adc.v2e.vxxa = np.ascontiguousarray(mr_adc.v2e.ccca[ncvs:, :ncvs, :ncvs, :])

            mr_adc.v2e.xxxe = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs, :ncvs, :ncvs, :])
            mr_adc.v2e.xxve = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs, :ncvs, ncvs:, :])
            mr_adc.v2e.vxxe = np.ascontiguousarray(mr_adc.v2e.ccce[ncvs:, :ncvs, :ncvs, :])

            mr_adc.v2e.xxaa = np.ascontiguousarray(mr_adc.v2e.ccaa[:ncvs, :ncvs, :, :])
            mr_adc.v2e.xvaa = np.ascontiguousarray(mr_adc.v2e.ccaa[:ncvs, ncvs:, :, :])
            mr_adc.v2e.vxaa = np.ascontiguousarray(mr_adc.v2e.ccaa[ncvs:, :ncvs, :, :])
            mr_adc.v2e.vvaa = np.ascontiguousarray(mr_adc.v2e.ccaa[ncvs:, ncvs:, :, :])

            mr_adc.v2e.xxae = np.ascontiguousarray(mr_adc.v2e.ccae[:ncvs, :ncvs, :, :])
            mr_adc.v2e.xvae = np.ascontiguousarray(mr_adc.v2e.ccae[:ncvs, ncvs:, :, :])
            mr_adc.v2e.vxae = np.ascontiguousarray(mr_adc.v2e.ccae[ncvs:, :ncvs, :, :])
            mr_adc.v2e.vvae = np.ascontiguousarray(mr_adc.v2e.ccae[ncvs:, ncvs:, :, :])

            mr_adc.v2e.xaax = np.ascontiguousarray(mr_adc.v2e.caac[:ncvs, :, :, :ncvs])
            mr_adc.v2e.xaav = np.ascontiguousarray(mr_adc.v2e.caac[:ncvs, :, :, ncvs:])
            mr_adc.v2e.vaax = np.ascontiguousarray(mr_adc.v2e.caac[ncvs:, :, :, :ncvs])
            mr_adc.v2e.vaav = np.ascontiguousarray(mr_adc.v2e.caac[ncvs:, :, :, ncvs:])

            mr_adc.v2e.xaex = np.ascontiguousarray(mr_adc.v2e.caec[:ncvs, :, :, :ncvs])
            mr_adc.v2e.xaev = np.ascontiguousarray(mr_adc.v2e.caec[:ncvs, :, :, ncvs:])
            mr_adc.v2e.vaex = np.ascontiguousarray(mr_adc.v2e.caec[ncvs:, :, :, :ncvs])
            mr_adc.v2e.vaev = np.ascontiguousarray(mr_adc.v2e.caec[ncvs:, :, :, ncvs:])

            mr_adc.v2e.xaxa = np.ascontiguousarray(mr_adc.v2e.caca[:ncvs, :, :ncvs, :])
            mr_adc.v2e.xava = np.ascontiguousarray(mr_adc.v2e.caca[:ncvs, :, ncvs:, :])
            mr_adc.v2e.vaxa = np.ascontiguousarray(mr_adc.v2e.caca[ncvs:, :, :ncvs, :])
            mr_adc.v2e.vava = np.ascontiguousarray(mr_adc.v2e.caca[ncvs:, :, ncvs:, :])

            mr_adc.v2e.xexe = np.ascontiguousarray(mr_adc.v2e.cece[:ncvs, :, :ncvs, :])
            mr_adc.v2e.xeve = np.ascontiguousarray(mr_adc.v2e.cece[:ncvs, :, ncvs:, :])
            mr_adc.v2e.vexe = np.ascontiguousarray(mr_adc.v2e.cece[ncvs:, :, :ncvs, :])
            mr_adc.v2e.veve = np.ascontiguousarray(mr_adc.v2e.cece[ncvs:, :, ncvs:, :])

            mr_adc.v2e.xaxe = np.ascontiguousarray(mr_adc.v2e.cace[:ncvs, :, :ncvs, :])
            mr_adc.v2e.xave = np.ascontiguousarray(mr_adc.v2e.cace[:ncvs, :, ncvs:, :])
            mr_adc.v2e.vaxe = np.ascontiguousarray(mr_adc.v2e.cace[ncvs:, :, :ncvs, :])
            mr_adc.v2e.vave = np.ascontiguousarray(mr_adc.v2e.cace[ncvs:, :, ncvs:, :])

            mr_adc.v2e.xaaa = np.ascontiguousarray(mr_adc.v2e.caaa[:ncvs, :, :, :])
            mr_adc.v2e.vaaa = np.ascontiguousarray(mr_adc.v2e.caaa[ncvs:, :, :, :])

            mr_adc.v2e.xeae = np.ascontiguousarray(mr_adc.v2e.ceae[:ncvs, :, :, :])
            mr_adc.v2e.veae = np.ascontiguousarray(mr_adc.v2e.ceae[ncvs:, :, :, :])

            mr_adc.v2e.xaae = np.ascontiguousarray(mr_adc.v2e.caae[:ncvs, :, :, :])
            mr_adc.v2e.vaae = np.ascontiguousarray(mr_adc.v2e.caae[ncvs:, :, :, :])

            mr_adc.v2e.xeaa = np.ascontiguousarray(mr_adc.v2e.ceaa[:ncvs, :, :, :])
            mr_adc.v2e.veaa = np.ascontiguousarray(mr_adc.v2e.ceaa[ncvs:, :, :, :])

            # Effective one-electron integrals
            mr_adc.h1eff.xa = np.ascontiguousarray(mr_adc.h1eff.ca[:ncvs,:])
            mr_adc.h1eff.va = np.ascontiguousarray(mr_adc.h1eff.ca[ncvs:,:])

            mr_adc.h1eff.xe = np.ascontiguousarray(mr_adc.h1eff.ce[:ncvs,:])
            mr_adc.h1eff.ve = np.ascontiguousarray(mr_adc.h1eff.ce[ncvs:,:])

            # Store diagonal elements of the generalized Fock operator
            mr_adc.mo_energy.x = mr_adc.mo_energy.c[:ncvs]
            mr_adc.mo_energy.v = mr_adc.mo_energy.c[ncvs:]

        if mr_adc.method in ("mr-adc(2)-x"):
            mr_adc.v2e.xxxx = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, :ncvs, :ncvs, :ncvs])
            mr_adc.v2e.xxvv = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, :ncvs, ncvs:, ncvs:])
            mr_adc.v2e.xvvx = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, ncvs:, ncvs:, :ncvs])
            mr_adc.v2e.xxvx = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, :ncvs, ncvs:, :ncvs])
            mr_adc.v2e.xxxv = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, :ncvs, :ncvs, ncvs:])
            mr_adc.v2e.xvxx = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, ncvs:, :ncvs, :ncvs])

            mr_adc.v2e.xxee = np.ascontiguousarray(mr_adc.v2e.ccee[:ncvs, :ncvs, :, :])
            mr_adc.v2e.xvee = np.ascontiguousarray(mr_adc.v2e.ccee[:ncvs, ncvs:, :, :])
            mr_adc.v2e.vxee = np.ascontiguousarray(mr_adc.v2e.ccee[ncvs:, :ncvs, :, :])
            mr_adc.v2e.vvee = np.ascontiguousarray(mr_adc.v2e.ccee[ncvs:, ncvs:, :, :])

            mr_adc.v2e.xeex = np.ascontiguousarray(mr_adc.v2e.ceec[:ncvs, :, :, :ncvs])
            mr_adc.v2e.xeev = np.ascontiguousarray(mr_adc.v2e.ceec[:ncvs, :, :, ncvs:])
            mr_adc.v2e.veex = np.ascontiguousarray(mr_adc.v2e.ceec[ncvs:, :, :, :ncvs])
            mr_adc.v2e.veev = np.ascontiguousarray(mr_adc.v2e.ceec[ncvs:, :, :, ncvs:])

            mr_adc.v2e.xaea = np.ascontiguousarray(mr_adc.v2e.caea[:ncvs, :, :, :])
            mr_adc.v2e.vaea = np.ascontiguousarray(mr_adc.v2e.caea[ncvs:, :, :, :])

            mr_adc.v2e.xaee = np.ascontiguousarray(mr_adc.v2e.caee[:ncvs, :, :, :])
            mr_adc.v2e.vaee = np.ascontiguousarray(mr_adc.v2e.caee[ncvs:, :, :, :])

            mr_adc.v2e.xeea = np.ascontiguousarray(mr_adc.v2e.ceea[:ncvs, :, :, :])
            mr_adc.v2e.veea = np.ascontiguousarray(mr_adc.v2e.ceea[ncvs:, :, :, :])

    print("Time for computing integrals:                      %f sec\n" % (time.time() - start_time))
