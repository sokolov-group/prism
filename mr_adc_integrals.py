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

import prism.lib.logger as logger
import prism.lib.tools as tools

def transform_integrals_1e(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nTransforming 1e integrals to MO basis...")

    mo = mr_adc.mo

    mr_adc.h1e = reduce(np.dot, (mo.T, mr_adc.interface.h1e_ao, mo))

    if mr_adc.method_type in ('ee','cvs-ee'):
        mr_adc.dip_mom = np.zeros((3, mr_adc.nmo, mr_adc.nmo))

        # Dipole moments
        for i in range(3):
            mr_adc.dip_mom[i] = reduce(np.dot, (mo.T, mr_adc.interface.dip_mom_ao[i], mo))

    mr_adc.log.timer("transforming 1e integrals", *cput0)

def transform_2e_chem_incore(interface, mo_1, mo_2, mo_3, mo_4, compacted=False):
    'Two-electron integral transformation in Chemists notation'

    nmo_1 = mo_1.shape[1]
    nmo_2 = mo_2.shape[1]
    nmo_3 = mo_3.shape[1]
    nmo_4 = mo_4.shape[1]

    v2e = interface.transform_2e_chem_incore(interface.v2e_ao, (mo_1, mo_2, mo_3, mo_4), compact=compacted)
    if compacted:
        v2e = v2e.reshape(nmo_1, nmo_2, -1)
    else:
        v2e = v2e.reshape(nmo_1, nmo_2, nmo_3, nmo_4)

    return np.ascontiguousarray(v2e)

def compute_effective_1e(mr_adc, h1e_pq, v2e_ccpq, v2e_cpqc):
    'Effective one-electron integrals'

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    h1eff  = h1e_pq
    h1eff += 2.0 * einsum('rrpq->pq', v2e_ccpq, optimize = einsum_type)
    h1eff -= einsum('rpqr->pq', v2e_cpqc, optimize = einsum_type)

    return h1eff

def transform_integrals_2e_incore(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nTransforming 2e integrals to MO basis (in-core)...")

    # Import Prism interface
    interface = mr_adc.interface

    # Variables from kernel
    ncore = mr_adc.ncore
    nocc = mr_adc.nocc
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    mo = mr_adc.mo
    mo_c = mo[:, :ncore].copy()
    mo_a = mo[:, ncore:nocc].copy()
    mo_e = mo[:, nocc:].copy()

    if mr_adc.outcore_expensive_tensors:
        mr_adc.tmpfile.feri1 = tools.create_temp_file(mr_adc)
    else:
        mr_adc.tmpfile.feri1 = None
    tmpfile = mr_adc.tmpfile.feri1

    mr_adc.v2e.aaaa = transform_2e_chem_incore(interface, mo_a, mo_a, mo_a, mo_a)

    if mr_adc.method_type == "ip" or mr_adc.method_type == "ea" or mr_adc.method_type == "cvs-ip":
        if mr_adc.method in ("mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.ccaa = transform_2e_chem_incore(interface, mo_c, mo_c, mo_a, mo_a)
            mr_adc.v2e.ccae = transform_2e_chem_incore(interface, mo_c, mo_c, mo_a, mo_e)

            mr_adc.v2e.caac = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_c)
            mr_adc.v2e.caec = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_c)

            mr_adc.v2e.caca = transform_2e_chem_incore(interface, mo_c, mo_a, mo_c, mo_a)
            mr_adc.v2e.cace = transform_2e_chem_incore(interface, mo_c, mo_a, mo_c, mo_e)

            mr_adc.v2e.caaa = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_a)
            mr_adc.v2e.caae = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_e)
            mr_adc.v2e.ceaa = transform_2e_chem_incore(interface, mo_c, mo_e, mo_a, mo_a)

            mr_adc.v2e.aaae = transform_2e_chem_incore(interface, mo_a, mo_a, mo_a, mo_e)

            mr_adc.v2e.cece = tools.create_dataset('cece', tmpfile, (ncore, nextern, ncore, nextern))
            mr_adc.v2e.ceae = tools.create_dataset('ceae', tmpfile, (ncore, nextern, ncas, nextern))

            mr_adc.v2e.cece[:] = transform_2e_chem_incore(interface, mo_c, mo_e, mo_c, mo_e)
            mr_adc.v2e.ceae[:] = transform_2e_chem_incore(interface, mo_c, mo_e, mo_a, mo_e)

        if mr_adc.method in ("mr-adc(2)-x"):
            mr_adc.v2e.cccc = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_c)

            mr_adc.v2e.caea = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_a)

            mr_adc.v2e.ccee = tools.create_dataset('ccee', tmpfile, (ncore, ncore, nextern, nextern))
            mr_adc.v2e.ceec = tools.create_dataset('ceec', tmpfile, (ncore, nextern, nextern, ncore))

            mr_adc.v2e.caee = tools.create_dataset('caee', tmpfile, (ncore, ncas, nextern, nextern))
            mr_adc.v2e.ceea = tools.create_dataset('ceea', tmpfile, (ncore, nextern, nextern, ncas))

            mr_adc.v2e.aeae = tools.create_dataset('aeae', tmpfile, (ncas, nextern, ncas, nextern))
            mr_adc.v2e.aaee = tools.create_dataset('aaee', tmpfile, (ncas, ncas, nextern, nextern))
            mr_adc.v2e.aeea = tools.create_dataset('aeea', tmpfile, (ncas, nextern, nextern, ncas))

            mr_adc.v2e.ccee[:] = transform_2e_chem_incore(interface, mo_c, mo_c, mo_e, mo_e)
            mr_adc.v2e.ceec[:] = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_c)

            mr_adc.v2e.caee[:] = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_e)
            mr_adc.v2e.ceea[:] = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_a)

            mr_adc.v2e.aeae[:] = transform_2e_chem_incore(interface, mo_a, mo_e, mo_a, mo_e)
            mr_adc.v2e.aaee[:] = transform_2e_chem_incore(interface, mo_a, mo_a, mo_e, mo_e)
            mr_adc.v2e.aeea[:] = transform_2e_chem_incore(interface, mo_a, mo_e, mo_e, mo_a)

        if mr_adc.method in ("mr-adc(2)-x") or (mr_adc.method in ("mr-adc(2)") and not mr_adc.approx_trans_moments):
            mr_adc.v2e.ceee = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_e, compacted = True)
            mr_adc.v2e.aeee = transform_2e_chem_incore(interface, mo_a, mo_e, mo_e, mo_e, compacted = True)

    # EE and CVS-EE
    elif mr_adc.method_type == "ee" or mr_adc.method_type == "cvs-ee":
        if mr_adc.method in ("mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.ccaa = transform_2e_chem_incore(interface, mo_c, mo_c, mo_a, mo_a)
            mr_adc.v2e.ccae = transform_2e_chem_incore(interface, mo_c, mo_c, mo_a, mo_e)

            mr_adc.v2e.ccca = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_a)
            mr_adc.v2e.ccce = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_e)

            mr_adc.v2e.caac = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_c)
            mr_adc.v2e.caec = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_c)

            mr_adc.v2e.caca = transform_2e_chem_incore(interface, mo_c, mo_a, mo_c, mo_a)
            mr_adc.v2e.cace = transform_2e_chem_incore(interface, mo_c, mo_a, mo_c, mo_e)

            mr_adc.v2e.ccee = tools.create_dataset('ccee', tmpfile, (ncore, ncore, nextern, nextern))
            mr_adc.v2e.ceec = tools.create_dataset('ceec', tmpfile, (ncore, nextern, nextern, ncore))
            mr_adc.v2e.ccee[:] = transform_2e_chem_incore(interface, mo_c, mo_c, mo_e, mo_e)
            mr_adc.v2e.ceec[:] = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_c)

            mr_adc.v2e.cece = tools.create_dataset('cece', tmpfile, (ncore, nextern, ncore, nextern))
            mr_adc.v2e.ceae = tools.create_dataset('ceae', tmpfile, (ncore, nextern, ncas, nextern))
            mr_adc.v2e.cece[:] = transform_2e_chem_incore(interface, mo_c, mo_e, mo_c, mo_e)
            mr_adc.v2e.ceae[:] = transform_2e_chem_incore(interface, mo_c, mo_e, mo_a, mo_e)

            mr_adc.v2e.caaa = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_a)
            mr_adc.v2e.caae = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_e)
            mr_adc.v2e.ceaa = transform_2e_chem_incore(interface, mo_c, mo_e, mo_a, mo_a)

            mr_adc.v2e.aaae = transform_2e_chem_incore(interface, mo_a, mo_a, mo_a, mo_e)
            mr_adc.v2e.aeae = tools.create_dataset('aeae', tmpfile, (ncas, nextern, ncas, nextern))
            mr_adc.v2e.aeae[:] = transform_2e_chem_incore(interface, mo_a, mo_e, mo_a, mo_e)
 
        if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"): ##comment out if checking M_00 block
            mr_adc.v2e.ccca = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_a)
            mr_adc.v2e.ccce = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_e)

            mr_adc.v2e.caea = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_a)

            mr_adc.v2e.caee = tools.create_dataset('caee', tmpfile, (ncore, ncas, nextern, nextern))
            mr_adc.v2e.ceea = tools.create_dataset('ceea', tmpfile, (ncore, nextern, nextern, ncas))
            mr_adc.v2e.caee[:] = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_e)
            mr_adc.v2e.ceea[:] = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_a)

            mr_adc.v2e.aaee = tools.create_dataset('aaee', tmpfile, (ncas, ncas, nextern, nextern))
            mr_adc.v2e.aeea = tools.create_dataset('aeea', tmpfile, (ncas, nextern, nextern, ncas))
            mr_adc.v2e.aaee[:] = transform_2e_chem_incore(interface, mo_a, mo_a, mo_e, mo_e)
            mr_adc.v2e.aeea[:] = transform_2e_chem_incore(interface, mo_a, mo_e, mo_e, mo_a)

            mr_adc.v2e.ceee = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_e, compacted = True)
            mr_adc.v2e.aeee = transform_2e_chem_incore(interface, mo_a, mo_e, mo_e, mo_e, compacted = True)

    # EE and CVS-EE
    elif mr_adc.method_type == "ee" or mr_adc.method_type == "cvs-ee":
        if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.ccaa = transform_2e_chem_incore(interface, mo_c, mo_c, mo_a, mo_a)
            mr_adc.v2e.caca = transform_2e_chem_incore(interface, mo_c, mo_a, mo_c, mo_a)
            mr_adc.v2e.caac = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_c)
            mr_adc.v2e.cace = transform_2e_chem_incore(interface, mo_c, mo_a, mo_c, mo_e)
            mr_adc.v2e.ccae = transform_2e_chem_incore(interface, mo_c, mo_c, mo_a, mo_e)
            mr_adc.v2e.caec = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_c)
            mr_adc.v2e.ccee = transform_2e_chem_incore(interface, mo_c, mo_c, mo_e, mo_e)
            mr_adc.v2e.cece = transform_2e_chem_incore(interface, mo_c, mo_e, mo_c, mo_e)
            mr_adc.v2e.ceec = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_c)

            mr_adc.v2e.caaa = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_a)
            mr_adc.v2e.caae = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_e)
            mr_adc.v2e.ceaa = transform_2e_chem_incore(interface, mo_c, mo_e, mo_a, mo_a)
            mr_adc.v2e.ceae = transform_2e_chem_incore(interface, mo_c, mo_e, mo_a, mo_e)

            mr_adc.v2e.aaae = transform_2e_chem_incore(interface, mo_a, mo_a, mo_a, mo_e)
            mr_adc.v2e.aeae = transform_2e_chem_incore(interface, mo_a, mo_e, mo_a, mo_e)

        if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.ccca = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_a)
            mr_adc.v2e.ccce = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_e)

            mr_adc.v2e.caea = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_a)

            mr_adc.v2e.ceea = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_a)
            mr_adc.v2e.caee = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_e)

            mr_adc.v2e.ceee = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_e)

            mr_adc.v2e.aaee = transform_2e_chem_incore(interface, mo_a, mo_a, mo_e, mo_e)
            mr_adc.v2e.aeea = transform_2e_chem_incore(interface, mo_a, mo_e, mo_e, mo_a)

            mr_adc.v2e.aeee = transform_2e_chem_incore(interface, mo_a, mo_e, mo_e, mo_e)

#        if mr_adc.method in ("mr-adc(2)-x"):
#            mr_adc.v2e.cccc = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_c)
#            mr_adc.v2e.eeee = transform_2e_chem_incore(interface, mo_e, mo_e, mo_e, mo_e)

#    # Effective one-electron integrals
#    ggcc = transform_2e_chem_incore(interface, mo, mo, mo_c, mo_c)
#    gccg = transform_2e_chem_incore(interface, mo, mo_c, mo_c, mo)
#    h1eff = mr_adc.h1e + 2.0 * einsum('pqrr->pq', ggcc, optimize = einsum_type) - einsum('prrq->pq', gccg, optimize = einsum_type)

    # Effective one-electron integrals
    mr_adc.v2e.ccca = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_a)
    mr_adc.v2e.ccce = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_e)

    v2e_ccac = mr_adc.v2e.ccca.transpose(1,0,3,2)
    v2e_ccec = mr_adc.v2e.ccce.transpose(1,0,3,2)

    mr_adc.h1eff.ca = compute_effective_1e(mr_adc, mr_adc.h1e[:ncore, ncore:nocc], mr_adc.v2e.ccca, v2e_ccac)
    mr_adc.h1eff.ce = compute_effective_1e(mr_adc, mr_adc.h1e[:ncore, nocc:], mr_adc.v2e.ccce, v2e_ccec)
    mr_adc.h1eff.aa = compute_effective_1e(mr_adc, mr_adc.h1e[ncore:nocc, ncore:nocc], mr_adc.v2e.ccaa, mr_adc.v2e.caac)
    mr_adc.h1eff.ae = compute_effective_1e(mr_adc, mr_adc.h1e[ncore:nocc, nocc:], mr_adc.v2e.ccae, mr_adc.v2e.caec)

    # Store diagonal elements of the generalized Fock operator
    mr_adc.mo_energy.c = mr_adc.interface.mo_energy[:ncore]
    mr_adc.mo_energy.e = mr_adc.interface.mo_energy[nocc:]

    mr_adc.log.timer("transforming 1e integrals", *cput0)

def transform_Heff_integrals_2e_df(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())

    # Import Prism interface
    interface = mr_adc.interface

    # Variables from kernel
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc

    nmo = mr_adc.nmo
    mo = mr_adc.mo
    mo_c = mo[:, :ncore].copy()
    mo_a = mo[:, ncore:nocc].copy()

    # Create temp file and datasets
    mr_adc.tmpfile.feri0 = tools.create_temp_file(mr_adc) # Non-core indices' integrals
    mr_adc.tmpfile.cferi0 = tools.create_temp_file(mr_adc) # Core indices' integrals

    tmpfile = mr_adc.tmpfile.feri0
    ctmpfile = mr_adc.tmpfile.cferi0

    mr_adc.v2e.aaaa = tools.create_dataset('aaaa', tmpfile, (ncas, ncas, ncas, ncas))
    mr_adc.v2e.ccca = tools.create_dataset('ccca', ctmpfile, (ncore, ncore, ncore, ncas))

    if mr_adc.method_type == "ip" or mr_adc.method_type == "ea" or mr_adc.method_type == "cvs-ip":
        if mr_adc.method in ("mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):

            mr_adc.v2e.ccaa = tools.create_dataset('ccaa', ctmpfile, (ncore, ncore, ncas, ncas))
            mr_adc.v2e.caac = tools.create_dataset('caac', ctmpfile, (ncore, ncas, ncas, ncore))

        if mr_adc.method in ("mr-adc(2)-x"):
            mr_adc.v2e.cccc = tools.create_dataset('cccc', ctmpfile, (ncore, ncore, ncore, ncore))

    # Atomic orbitals auxiliary basis-set
    if interface.reference_df:
        mr_adc.log.extra("\nTransforming Heff 2e integrals to MO basis (density-fitting)...\n")

        with_df = interface.reference_df
        with_df.max_memory = mr_adc.max_memory
        naux = with_df.get_naoaux()

        Lcc = np.empty((naux, ncore, ncore))
        Lca = np.empty((naux, ncore, ncas))
        Lac = np.empty((naux, ncas, ncore))
        Laa = np.empty((naux, ncas, ncas))

        ijslice = (0, nmo, 0, nmo)
        Lpq = None
        p1 = 0

        for eri1 in with_df.loop():
            Lpq = interface.transform_2e_pair_chem_incore(eri1, mo, ijslice, aosym='s2', out=Lpq).reshape(-1, nmo, nmo)

            p0, p1 = p1, p1 + Lpq.shape[0]
            Lcc[p0:p1] = Lpq[:, :ncore, :ncore]
            Lca[p0:p1] = Lpq[:, :ncore, ncore:nocc]

            Lac[p0:p1] = Lpq[:, ncore:nocc, :ncore]
            Laa[p0:p1] = Lpq[:, ncore:nocc, ncore:nocc]
        del(Lpq)

        # Effective Hamiltonian 2e- integrals
        mr_adc.v2e.aaaa[:] = get_v2e_df(mr_adc, Laa, Laa, 'aaaa')
        tools.flush(tmpfile)

        mr_adc.v2e.ccca[:] = get_v2e_df(mr_adc, Lcc, Lca, 'ccca')
        tools.flush(ctmpfile)

        if mr_adc.method_type == "ip" or mr_adc.method_type == "ea" or mr_adc.method_type == "cvs-ip":
            if mr_adc.method in ("mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
                mr_adc.v2e.ccaa[:] = get_v2e_df(mr_adc, Lcc, Laa, 'ccaa')
                tools.flush(ctmpfile)

                mr_adc.v2e.caac[:] = get_v2e_df(mr_adc, Lca, Lac, 'caac')
                tools.flush(ctmpfile)

            if mr_adc.method in ("mr-adc(2)-x"):
                mr_adc.v2e.cccc[:] = get_v2e_df(mr_adc, Lcc, Lcc, 'cccc')
                tools.flush(ctmpfile)
    else:
        mr_adc.log.extra("\nTransforming Heff 2e integrals to MO basis (in-core)...")

        # Effective Hamiltonian 2e- integrals
        mr_adc.v2e.aaaa[:] = transform_2e_chem_incore(interface, mo_a, mo_a, mo_a, mo_a)
        tools.flush(tmpfile)

        mr_adc.v2e.ccca[:] = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_a)
        tools.flush(ctmpfile)

        if mr_adc.method_type == "ip" or mr_adc.method_type == "ea" or mr_adc.method_type == "cvs-ip":
            if mr_adc.method in ("mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
                mr_adc.v2e.ccaa[:] = transform_2e_chem_incore(interface, mo_c, mo_c, mo_a, mo_a)
                tools.flush(ctmpfile)

                mr_adc.v2e.caac[:] = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_c)
                tools.flush(ctmpfile)

            if mr_adc.method in ("mr-adc(2)-x"):
                mr_adc.v2e.cccc[:] = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_c)
                tools.flush(ctmpfile)

    mr_adc.log.timer("transforming 2e integrals", *cput0)

def transform_integrals_2e_df(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nTransforming 2e integrals to MO basis (density-fitting)...\n")

    # Import Prism interface
    interface = mr_adc.interface

    # Atomic orbitals auxiliary basis-set
    with_df = interface.with_df
    with_df.max_memory = mr_adc.max_memory
    naux = interface.get_naux()

    # Variables from kernel
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc
    nextern = mr_adc.nextern

    nmo = mr_adc.nmo
    mo = mr_adc.mo

    mr_adc.v2e.ceee = None
    mr_adc.v2e.aeee = None

    # Create temp file and datasets
    mr_adc.tmpfile.feri1 = tools.create_temp_file(mr_adc) # Non-core indices' integrals
    mr_adc.tmpfile.cferi1 = tools.create_temp_file(mr_adc) # Core indices' integras

    tmpfile = mr_adc.tmpfile.feri1
    ctmpfile = mr_adc.tmpfile.cferi1

    Lcc = np.empty((naux, ncore, ncore))
    Lca = np.empty((naux, ncore, ncas))
    Lac = np.empty((naux, ncas, ncore))
    Laa = np.empty((naux, ncas, ncas))

    Lec = np.empty((naux, nextern,  ncore))
    Lea = np.empty((naux, nextern, ncas))

    mr_adc.naux = naux
    mr_adc.v2e.Lee = tools.create_dataset('Lee', tmpfile, (naux, nextern, nextern))
    mr_adc.v2e.Lce = np.empty((naux, ncore, nextern))
    mr_adc.v2e.Lae = np.empty((naux, ncas, nextern))

    ijslice = (0, nmo, 0, nmo)
    Lpq = None
    p1 = 0

    for eri1 in with_df.loop():
        Lpq = interface.transform_2e_pair_chem_incore(eri1, mo, ijslice, aosym='s2', out=Lpq).reshape(-1, nmo, nmo)

        p0, p1 = p1, p1 + Lpq.shape[0]
        Lcc[p0:p1] = Lpq[:, :ncore, :ncore]
        Lca[p0:p1] = Lpq[:, :ncore, ncore:nocc]
        mr_adc.v2e.Lce[p0:p1] = Lpq[:, :ncore, nocc:]

        Lac[p0:p1] = Lpq[:, ncore:nocc, :ncore]
        Laa[p0:p1] = Lpq[:, ncore:nocc, ncore:nocc]
        mr_adc.v2e.Lae[p0:p1] = Lpq[:, ncore:nocc, nocc:]

        Lec[p0:p1] = Lpq[:, nocc:, :ncore]
        Lea[p0:p1] = Lpq[:, nocc:, ncore:nocc]
        mr_adc.v2e.Lee[p0:p1] = Lpq[:, nocc:, nocc:]
        tools.flush(tmpfile)
    del(Lpq)

    # 2e- integrals
    if mr_adc.method_type == "ip" or mr_adc.method_type == "ea" or mr_adc.method_type == "cvs-ip":
        if mr_adc.method in ("mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.ccae = tools.create_dataset('ccae', ctmpfile, (ncore, ncore, ncas, nextern))
            mr_adc.v2e.caec = tools.create_dataset('caec', ctmpfile, (ncore, ncas, nextern, ncore))
            mr_adc.v2e.cace = tools.create_dataset('cace', ctmpfile, (ncore, ncas, ncore, nextern))

            mr_adc.v2e.caca = tools.create_dataset('caca', ctmpfile, (ncore, ncas, ncore, ncas))

            mr_adc.v2e.caae = tools.create_dataset('caae', ctmpfile, (ncore, ncas, ncas, nextern))
            mr_adc.v2e.ceaa = tools.create_dataset('ceaa', ctmpfile, (ncore, nextern, ncas, ncas))

            mr_adc.v2e.caaa = tools.create_dataset('caaa', ctmpfile, (ncore, ncas, ncas, ncas))
            mr_adc.v2e.aaae = tools.create_dataset('aaae', tmpfile, (ncas, ncas, ncas, nextern))

            mr_adc.v2e.cece = tools.create_dataset('cece', ctmpfile, (ncore, nextern, ncore, nextern))
            mr_adc.v2e.ceae = tools.create_dataset('ceae', ctmpfile, (ncore, nextern, ncas, nextern))


            mr_adc.v2e.ccae[:] = get_v2e_df(mr_adc, Lcc, mr_adc.v2e.Lae, 'ccae')
            tools.flush(ctmpfile)

            mr_adc.v2e.caec[:] = get_v2e_df(mr_adc, Lca, Lec, 'caec')
            tools.flush(ctmpfile)

            mr_adc.v2e.cace[:] = get_v2e_df(mr_adc, Lca, mr_adc.v2e.Lce, 'cace')
            tools.flush(ctmpfile)

            mr_adc.v2e.caca[:] = get_v2e_df(mr_adc, Lca, Lca, 'caca')
            tools.flush(ctmpfile)

            mr_adc.v2e.caae[:] = get_v2e_df(mr_adc, Lca, mr_adc.v2e.Lae, 'caae')
            tools.flush(ctmpfile)

            mr_adc.v2e.ceaa[:] = get_v2e_df(mr_adc, mr_adc.v2e.Lce, Laa, 'ceaa')
            tools.flush(ctmpfile)

            mr_adc.v2e.caaa[:] = get_v2e_df(mr_adc, Lca, Laa, 'caaa')
            tools.flush(ctmpfile)

            mr_adc.v2e.aaae[:] = get_v2e_df(mr_adc, Laa, mr_adc.v2e.Lae, 'aaae')
            tools.flush(tmpfile)

            chunks = tools.calculate_chunks(mr_adc, ncore, [ncore, nextern, nextern], ntensors = 2)
            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                mr_adc.log.debug("v2e.cece [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
                mr_adc.v2e.cece[s_chunk:f_chunk] = get_ooee_df(mr_adc, mr_adc.v2e.Lce, mr_adc.v2e.Lce, s_chunk, f_chunk)
                tools.flush(ctmpfile)

            chunks = tools.calculate_chunks(mr_adc, ncore, [ncas, nextern, nextern], ntensors = 2)
            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                mr_adc.log.debug("v2e.ceae [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
                mr_adc.v2e.ceae[s_chunk:f_chunk] = get_ooee_df(mr_adc, mr_adc.v2e.Lce, mr_adc.v2e.Lae, s_chunk, f_chunk)
                tools.flush(ctmpfile)

        if mr_adc.method in ("mr-adc(2)-x"):
            mr_adc.v2e.caea = tools.create_dataset('caea', ctmpfile, (ncore, ncas, nextern, ncas))

            mr_adc.v2e.ccee = tools.create_dataset('ccee', ctmpfile, (ncore, ncore, nextern, nextern))
            mr_adc.v2e.ceec = tools.create_dataset('ceec', ctmpfile, (ncore, nextern, nextern, ncore))

            mr_adc.v2e.caee = tools.create_dataset('caee', ctmpfile, (ncore, ncas, nextern, nextern))
            mr_adc.v2e.ceea = tools.create_dataset('ceea', ctmpfile, (ncore, nextern, nextern, ncas))

            mr_adc.v2e.aeae = tools.create_dataset('aeae', tmpfile, (ncas, nextern, ncas, nextern))
            mr_adc.v2e.aaee = tools.create_dataset('aaee', tmpfile, (ncas, ncas, nextern, nextern))
            mr_adc.v2e.aeea = tools.create_dataset('aeea', tmpfile, (ncas, nextern, nextern, ncas))

            mr_adc.v2e.caea[:] = get_v2e_df(mr_adc, Lca, Lea, 'caea')
            tools.flush(ctmpfile)

            chunks = tools.calculate_chunks(mr_adc, ncore, [ncore, nextern, nextern], ntensors = 2)
            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                mr_adc.log.debug("v2e.ccee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
                mr_adc.v2e.ccee[s_chunk:f_chunk] = get_ooee_df(mr_adc, Lcc, mr_adc.v2e.Lee, s_chunk, f_chunk)
                tools.flush(ctmpfile)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                mr_adc.log.debug("v2e.ceec [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
                mr_adc.v2e.ceec[s_chunk:f_chunk] = get_ooee_df(mr_adc, mr_adc.v2e.Lce, Lec, s_chunk, f_chunk)
                tools.flush(ctmpfile)

            chunks = tools.calculate_chunks(mr_adc, ncore, [ncas, nextern, nextern], ntensors = 2)
            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                mr_adc.log.debug("v2e.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
                mr_adc.v2e.caee[s_chunk:f_chunk] = get_ooee_df(mr_adc, Lca, mr_adc.v2e.Lee, s_chunk, f_chunk)
                tools.flush(ctmpfile)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                mr_adc.log.debug("v2e.ceea [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
                mr_adc.v2e.ceea[s_chunk:f_chunk] = get_ooee_df(mr_adc, mr_adc.v2e.Lce, Lea, s_chunk, f_chunk)
                tools.flush(ctmpfile)

            chunks = tools.calculate_chunks(mr_adc, ncas, [ncas, nextern, nextern], ntensors = 2)
            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                mr_adc.log.debug("v2e.aeae [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
                mr_adc.v2e.aeae[s_chunk:f_chunk] = get_ooee_df(mr_adc, mr_adc.v2e.Lae, mr_adc.v2e.Lae, s_chunk, f_chunk)
                tools.flush(tmpfile)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                mr_adc.log.debug("v2e.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
                mr_adc.v2e.aaee[s_chunk:f_chunk] = get_ooee_df(mr_adc, Laa, mr_adc.v2e.Lee, s_chunk, f_chunk)
                tools.flush(tmpfile)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                mr_adc.log.debug("v2e.aeea [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
                mr_adc.v2e.aeea[s_chunk:f_chunk] = get_ooee_df(mr_adc, mr_adc.v2e.Lae, Lea, s_chunk, f_chunk)
                tools.flush(tmpfile)

    # Effective one-electron integrals
    mr_adc.v2e.ccce = tools.create_dataset('ccce', ctmpfile, (ncore, ncore, ncore, nextern))
    mr_adc.v2e.ccce[:] = get_v2e_df(mr_adc, Lcc, mr_adc.v2e.Lce, 'ccce')
    tools.flush(ctmpfile)

    mr_adc.v2e.ccac = tools.create_dataset('ccac', ctmpfile, (ncore, ncore, ncas, ncore))
    mr_adc.v2e.ccac[:] = get_v2e_df(mr_adc, Lcc, Lac, 'ccac')
    tools.flush(ctmpfile)

    mr_adc.v2e.ccec = tools.create_dataset('ccec', ctmpfile, (ncore, ncore, nextern, ncore))
    mr_adc.v2e.ccec[:] = get_v2e_df(mr_adc, Lcc, Lec, 'ccec')
    tools.flush(ctmpfile)

    mr_adc.h1eff.ca = compute_effective_1e(mr_adc, mr_adc.h1e[:ncore, ncore:nocc], mr_adc.v2e.ccca, mr_adc.v2e.ccac)
    mr_adc.h1eff.ce = compute_effective_1e(mr_adc, mr_adc.h1e[:ncore, nocc:], mr_adc.v2e.ccce, mr_adc.v2e.ccec)
    mr_adc.h1eff.aa = compute_effective_1e(mr_adc, mr_adc.h1e[ncore:nocc, ncore:nocc], mr_adc.v2e.ccaa, mr_adc.v2e.caac)
    mr_adc.h1eff.ae = compute_effective_1e(mr_adc, mr_adc.h1e[ncore:nocc, nocc:], mr_adc.v2e.ccae, mr_adc.v2e.caec)

    # Store diagonal elements of the generalized Fock operator
    mr_adc.mo_energy.c = mr_adc.interface.mo_energy[:ncore]
    mr_adc.mo_energy.e = mr_adc.interface.mo_energy[nocc:]

    mr_adc.log.timer("transforming 2e integrals", *cput0)

def get_oeee_df(mr_adc, Loe, Lee, s_chunk_occ, f_chunk_occ):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    nextern = mr_adc.nextern
    naux = mr_adc.naux

    chunk_size_occ = f_chunk_occ - s_chunk_occ

    chunks_aux = tools.calculate_double_chunks(mr_adc, naux, [chunk_size_occ, nextern], [nextern, nextern],
                                                        extra_tensors=[[chunk_size_occ, nextern, nextern, nextern],
                                                                       [chunk_size_occ, nextern, nextern, nextern]])

    v_oeee = np.zeros((chunk_size_occ, nextern, nextern, nextern))

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks_aux):
        mr_adc.log.debug("aux [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks_aux), s_chunk, f_chunk)
        cput1 = (logger.process_clock(), logger.perf_counter())

        Loe_chunk = np.ascontiguousarray(Loe[s_chunk:f_chunk,s_chunk_occ:f_chunk_occ])
        Lee_chunk = Lee[s_chunk:f_chunk]

        v_oeee += einsum('iab,icd->abcd', Loe_chunk, Lee_chunk, optimize = einsum_type)

        mr_adc.log.timer_debug("contracting v_oeee DF", *cput1)
    del(Loe_chunk, Lee_chunk)

    return v_oeee

def get_ooee_df(mr_adc, Lpq, Lrs, s_chunk_p, f_chunk_p):

    cput0 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    naux = mr_adc.naux

    chunk_p = f_chunk_p - s_chunk_p
    q = Lpq.shape[2]
    r = Lrs.shape[1]
    s = Lrs.shape[2]

    chunks_aux = tools.calculate_double_chunks(mr_adc, naux, [chunk_p, q], [r, s],
                                                       extra_tensors=[[chunk_p, q, r, s],
                                                                      [chunk_p, q, r, s]])

    v_ooee = np.zeros((chunk_p, q, r, s))

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks_aux):
        mr_adc.log.debug("aux [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks_aux), s_chunk, f_chunk)
        cput1 = (logger.process_clock(), logger.perf_counter())

        Lpq_chunk = np.ascontiguousarray(Lpq[s_chunk:f_chunk,s_chunk_p:f_chunk_p])
        Lrs_chunk = Lrs[s_chunk:f_chunk]

        v_ooee += einsum('iab,icd->abcd', Lpq_chunk, Lrs_chunk, optimize = einsum_type)

        mr_adc.log.timer_debug("contracting v_ooee DF", *cput1)
    del(Lpq_chunk, Lrs_chunk)

    mr_adc.log.timer_debug("computing v_ooee DF", *cput0)
    return v_ooee

def get_v2e_df(mr_adc, Lpq, Lrs, pqrs_string = None):

    cput0 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    v_pqrs = einsum('iab,icd->abcd', Lpq, Lrs, optimize = einsum_type)

    mr_adc.log.timer_debug("computing v2e.{:} DF".format(pqrs_string), *cput0)
    return v_pqrs

def unpack_v2e_oeee(mr_adc, v2e_oeee):

    # Variables from kernel
    nextern = mr_adc.nextern

    n_ee = nextern * (nextern + 1) // 2
    ind_ee = np.tril_indices(nextern)

    v2e_oeee_ = None

    if len(v2e_oeee.shape) == 3:
        if (v2e_oeee.shape[0] == n_ee):
            v2e_oeee_ = np.zeros((nextern, nextern, v2e_oeee.shape[1], v2e_oeee.shape[2]))
            v2e_oeee_[ind_ee[0], ind_ee[1]] = v2e_oeee
            v2e_oeee_[ind_ee[1], ind_ee[0]] = v2e_oeee

        elif (v2e_oeee.shape[2] == n_ee):
            v2e_oeee_ = np.zeros((v2e_oeee.shape[0], v2e_oeee.shape[1], nextern, nextern))
            v2e_oeee_[:, :, ind_ee[0], ind_ee[1]] = v2e_oeee
            v2e_oeee_[:, :, ind_ee[1], ind_ee[0]] = v2e_oeee
        else:
            raise TypeError("ERI dimensions don't match")

    else:
        raise RuntimeError("ERI does not have a correct dimension")

    return v2e_oeee_

def compute_cvs_integrals_2e_incore(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing CVS integrals to MO basis (in-core)...")

    if mr_adc.outcore_expensive_tensors:
        mr_adc.tmpfile.xferi1 = tools.create_temp_file(mr_adc)
    else:
        mr_adc.tmpfile.xferi1 = None
    tmpfile = mr_adc.tmpfile.xferi1

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    if mr_adc.method_type == "cvs-ip":
        if mr_adc.method in ("mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.xxaa = np.ascontiguousarray(mr_adc.v2e.ccaa[:ncvs, :ncvs, :, :])
            mr_adc.v2e.xvaa = np.ascontiguousarray(mr_adc.v2e.ccaa[:ncvs, ncvs:, :, :])
            mr_adc.v2e.vxaa = np.ascontiguousarray(mr_adc.v2e.ccaa[ncvs:, :ncvs, :, :])
            mr_adc.v2e.vvaa = np.ascontiguousarray(mr_adc.v2e.ccaa[ncvs:, ncvs:, :, :])
            del(mr_adc.v2e.ccaa)

            mr_adc.v2e.xxae = np.ascontiguousarray(mr_adc.v2e.ccae[:ncvs, :ncvs, :, :])
            mr_adc.v2e.xvae = np.ascontiguousarray(mr_adc.v2e.ccae[:ncvs, ncvs:, :, :])
            mr_adc.v2e.vxae = np.ascontiguousarray(mr_adc.v2e.ccae[ncvs:, :ncvs, :, :])
            mr_adc.v2e.vvae = np.ascontiguousarray(mr_adc.v2e.ccae[ncvs:, ncvs:, :, :])
            del(mr_adc.v2e.ccae)

            mr_adc.v2e.xaax = np.ascontiguousarray(mr_adc.v2e.caac[:ncvs, :, :, :ncvs])
            mr_adc.v2e.xaav = np.ascontiguousarray(mr_adc.v2e.caac[:ncvs, :, :, ncvs:])
            mr_adc.v2e.vaax = np.ascontiguousarray(mr_adc.v2e.caac[ncvs:, :, :, :ncvs])
            mr_adc.v2e.vaav = np.ascontiguousarray(mr_adc.v2e.caac[ncvs:, :, :, ncvs:])
            del(mr_adc.v2e.caac)

            mr_adc.v2e.xaex = np.ascontiguousarray(mr_adc.v2e.caec[:ncvs, :, :, :ncvs])
            mr_adc.v2e.xaev = np.ascontiguousarray(mr_adc.v2e.caec[:ncvs, :, :, ncvs:])
            mr_adc.v2e.vaex = np.ascontiguousarray(mr_adc.v2e.caec[ncvs:, :, :, :ncvs])
            mr_adc.v2e.vaev = np.ascontiguousarray(mr_adc.v2e.caec[ncvs:, :, :, ncvs:])
            del(mr_adc.v2e.caec)

            mr_adc.v2e.xaxa = np.ascontiguousarray(mr_adc.v2e.caca[:ncvs, :, :ncvs, :])
            mr_adc.v2e.xava = np.ascontiguousarray(mr_adc.v2e.caca[:ncvs, :, ncvs:, :])
            del(mr_adc.v2e.caca)

            mr_adc.v2e.xaxe = np.ascontiguousarray(mr_adc.v2e.cace[:ncvs, :, :ncvs, :])
            mr_adc.v2e.xave = np.ascontiguousarray(mr_adc.v2e.cace[:ncvs, :, ncvs:, :])
            mr_adc.v2e.vaxe = np.ascontiguousarray(mr_adc.v2e.cace[ncvs:, :, :ncvs, :])
            del(mr_adc.v2e.cace)

            mr_adc.v2e.xaaa = np.ascontiguousarray(mr_adc.v2e.caaa[:ncvs, :, :, :])
            mr_adc.v2e.vaaa = np.ascontiguousarray(mr_adc.v2e.caaa[ncvs:, :, :, :])
            del(mr_adc.v2e.caaa)

            mr_adc.v2e.xaae = np.ascontiguousarray(mr_adc.v2e.caae[:ncvs, :, :, :])
            mr_adc.v2e.vaae = np.ascontiguousarray(mr_adc.v2e.caae[ncvs:, :, :, :])
            del(mr_adc.v2e.caae)

            mr_adc.v2e.xeaa = np.ascontiguousarray(mr_adc.v2e.ceaa[:ncvs, :, :, :])
            mr_adc.v2e.veaa = np.ascontiguousarray(mr_adc.v2e.ceaa[ncvs:, :, :, :])
            del(mr_adc.v2e.ceaa)

            mr_adc.v2e.xexe = tools.create_dataset('xexe', tmpfile, (ncvs, nextern, ncvs, nextern))
            mr_adc.v2e.xeve = tools.create_dataset('xeve', tmpfile, (ncvs, nextern, nval, nextern))
            mr_adc.v2e.veve = tools.create_dataset('veve', tmpfile, (nval, nextern, nval, nextern))

            mr_adc.v2e.xexe[:] = np.ascontiguousarray(mr_adc.v2e.cece[:ncvs, :, :ncvs, :])
            tools.flush(tmpfile)

            mr_adc.v2e.xeve[:] = np.ascontiguousarray(mr_adc.v2e.cece[:ncvs, :, ncvs:, :])
            tools.flush(tmpfile)

            mr_adc.v2e.veve[:] = np.ascontiguousarray(mr_adc.v2e.cece[ncvs:, :, ncvs:, :])
            tools.flush(tmpfile)
            del(mr_adc.v2e.cece)

            mr_adc.v2e.xeae = tools.create_dataset('xeae', tmpfile, (ncvs, nextern, ncas, nextern))

            mr_adc.v2e.xeae[:] = np.ascontiguousarray(mr_adc.v2e.ceae[:ncvs, :, :, :])
            tools.flush(tmpfile)
            del(mr_adc.v2e.ceae)

        if mr_adc.method in ("mr-adc(2)-x"):
            mr_adc.v2e.xxxx = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, :ncvs, :ncvs, :ncvs])
            mr_adc.v2e.xxvv = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, :ncvs, ncvs:, ncvs:])
            mr_adc.v2e.xvvx = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, ncvs:, ncvs:, :ncvs])
            mr_adc.v2e.xxvx = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, :ncvs, ncvs:, :ncvs])
            mr_adc.v2e.xxxv = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, :ncvs, :ncvs, ncvs:])
            mr_adc.v2e.xvxx = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, ncvs:, :ncvs, :ncvs])
            del(mr_adc.v2e.cccc)

            mr_adc.v2e.xaea = np.ascontiguousarray(mr_adc.v2e.caea[:ncvs, :, :, :])
            mr_adc.v2e.vaea = np.ascontiguousarray(mr_adc.v2e.caea[ncvs:, :, :, :])
            del(mr_adc.v2e.caea)

            mr_adc.v2e.xxee = tools.create_dataset('xxee', tmpfile, (ncvs, ncvs, nextern, nextern))
            mr_adc.v2e.xvee = tools.create_dataset('xvee', tmpfile, (ncvs, nval, nextern, nextern))
            mr_adc.v2e.vxee = tools.create_dataset('vxee', tmpfile, (nval, ncvs, nextern, nextern))
            mr_adc.v2e.vvee = tools.create_dataset('vvee', tmpfile, (nval, nval, nextern, nextern))

            mr_adc.v2e.xxee[:] = np.ascontiguousarray(mr_adc.v2e.ccee[:ncvs, :ncvs, :, :])
            tools.flush(tmpfile)

            mr_adc.v2e.xvee[:] = np.ascontiguousarray(mr_adc.v2e.ccee[:ncvs, ncvs:, :, :])
            tools.flush(tmpfile)

            mr_adc.v2e.vxee[:] = np.ascontiguousarray(mr_adc.v2e.ccee[ncvs:, :ncvs, :, :])
            tools.flush(tmpfile)

            mr_adc.v2e.vvee[:] = np.ascontiguousarray(mr_adc.v2e.ccee[ncvs:, ncvs:, :, :])
            tools.flush(tmpfile)
            del(mr_adc.v2e.ccee)

            mr_adc.v2e.xeex = tools.create_dataset('xeex', tmpfile, (ncvs, nextern, nextern, ncvs))
            mr_adc.v2e.xeev = tools.create_dataset('xeev', tmpfile, (ncvs, nextern, nextern, nval))
            mr_adc.v2e.veex = tools.create_dataset('veex', tmpfile, (nval, nextern, nextern, ncvs))
            mr_adc.v2e.veev = tools.create_dataset('veev', tmpfile, (nval, nextern, nextern, nval))

            mr_adc.v2e.xeex[:] = np.ascontiguousarray(mr_adc.v2e.ceec[:ncvs, :, :, :ncvs])
            tools.flush(tmpfile)

            mr_adc.v2e.xeev[:] = np.ascontiguousarray(mr_adc.v2e.ceec[:ncvs, :, :, ncvs:])
            tools.flush(tmpfile)

            mr_adc.v2e.veex[:] = np.ascontiguousarray(mr_adc.v2e.ceec[ncvs:, :, :, :ncvs])
            tools.flush(tmpfile)

            mr_adc.v2e.veev[:] = np.ascontiguousarray(mr_adc.v2e.ceec[ncvs:, :, :, ncvs:])
            tools.flush(tmpfile)
            del(mr_adc.v2e.ceec)


            mr_adc.v2e.xaee = tools.create_dataset('xaee', tmpfile, (ncvs, ncas, nextern, nextern))
            mr_adc.v2e.vaee = tools.create_dataset('vaee', tmpfile, (nval, ncas, nextern, nextern))

            mr_adc.v2e.xaee[:] = np.ascontiguousarray(mr_adc.v2e.caee[:ncvs, :, :, :])
            tools.flush(tmpfile)

            mr_adc.v2e.vaee[:] = np.ascontiguousarray(mr_adc.v2e.caee[ncvs:, :, :, :])
            tools.flush(tmpfile)
            del(mr_adc.v2e.caee)


            mr_adc.v2e.xeea = tools.create_dataset('xeea', tmpfile, (ncvs, nextern, nextern, ncas))
            mr_adc.v2e.veea = tools.create_dataset('veea', tmpfile, (nval, nextern, nextern, ncas))

            mr_adc.v2e.xeea[:] = np.ascontiguousarray(mr_adc.v2e.ceea[:ncvs, :, :, :])
            tools.flush(tmpfile)

            mr_adc.v2e.veea[:] = np.ascontiguousarray(mr_adc.v2e.ceea[ncvs:, :, :, :])
            tools.flush(tmpfile)
            del(mr_adc.v2e.ceea)

    # CVS-EE
    elif mr_adc.method_type == "cvs-ee":
        if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.xxaa = np.ascontiguousarray(mr_adc.v2e.ccaa[:ncvs, :ncvs, :, :])
            mr_adc.v2e.xvaa = np.ascontiguousarray(mr_adc.v2e.ccaa[:ncvs, ncvs:, :, :])
            mr_adc.v2e.vxaa = np.ascontiguousarray(mr_adc.v2e.ccaa[ncvs:, :ncvs, :, :])
            mr_adc.v2e.vvaa = np.ascontiguousarray(mr_adc.v2e.ccaa[ncvs:, ncvs:, :, :])
            del(mr_adc.v2e.ccaa)

            mr_adc.v2e.xaxa = np.ascontiguousarray(mr_adc.v2e.caca[:ncvs, :, :ncvs, :])
            mr_adc.v2e.xava = np.ascontiguousarray(mr_adc.v2e.caca[:ncvs, :, ncvs:, :])
            mr_adc.v2e.vaxa = np.ascontiguousarray(mr_adc.v2e.caca[ncvs:, :, :ncvs, :])
            mr_adc.v2e.vava = np.ascontiguousarray(mr_adc.v2e.caca[ncvs:, :, ncvs:, :])
            del(mr_adc.v2e.caca)

            mr_adc.v2e.xaax = np.ascontiguousarray(mr_adc.v2e.caac[:ncvs, :, :, :ncvs])
            mr_adc.v2e.xaav = np.ascontiguousarray(mr_adc.v2e.caac[:ncvs, :, :, ncvs:])
            mr_adc.v2e.vaax = np.ascontiguousarray(mr_adc.v2e.caac[ncvs:, :, :, :ncvs])
            mr_adc.v2e.vaav = np.ascontiguousarray(mr_adc.v2e.caac[ncvs:, :, :, ncvs:])
            del(mr_adc.v2e.caac)

            mr_adc.v2e.xaex = np.ascontiguousarray(mr_adc.v2e.caec[:ncvs, :, :, :ncvs])
            mr_adc.v2e.xaev = np.ascontiguousarray(mr_adc.v2e.caec[:ncvs, :, :, ncvs:])
            mr_adc.v2e.vaex = np.ascontiguousarray(mr_adc.v2e.caec[ncvs:, :, :, :ncvs])
            mr_adc.v2e.vaev = np.ascontiguousarray(mr_adc.v2e.caec[ncvs:, :, :, ncvs:])
            del(mr_adc.v2e.caec)

            mr_adc.v2e.xaxe = np.ascontiguousarray(mr_adc.v2e.cace[:ncvs, :, :ncvs, :])
            mr_adc.v2e.xave = np.ascontiguousarray(mr_adc.v2e.cace[:ncvs, :, ncvs:, :])
            mr_adc.v2e.vaxe = np.ascontiguousarray(mr_adc.v2e.cace[ncvs:, :, :ncvs, :])
            mr_adc.v2e.vave = np.ascontiguousarray(mr_adc.v2e.cace[ncvs:, :, ncvs:, :])
            del(mr_adc.v2e.cace)

            mr_adc.v2e.xxae = np.ascontiguousarray(mr_adc.v2e.ccae[:ncvs, :ncvs, :, :])
            mr_adc.v2e.xvae = np.ascontiguousarray(mr_adc.v2e.ccae[:ncvs, ncvs:, :, :])
            mr_adc.v2e.vxae = np.ascontiguousarray(mr_adc.v2e.ccae[ncvs:, :ncvs, :, :])
            mr_adc.v2e.vvae = np.ascontiguousarray(mr_adc.v2e.ccae[ncvs:, ncvs:, :, :])
            del(mr_adc.v2e.ccae)

            mr_adc.v2e.xxee = np.ascontiguousarray(mr_adc.v2e.ccee[:ncvs, :ncvs, :, :])
            mr_adc.v2e.xvee = np.ascontiguousarray(mr_adc.v2e.ccee[:ncvs, ncvs:, :, :])
            mr_adc.v2e.vxee = np.ascontiguousarray(mr_adc.v2e.ccee[ncvs:, :ncvs, :, :])
            mr_adc.v2e.vvee = np.ascontiguousarray(mr_adc.v2e.ccee[ncvs:, ncvs:, :, :])
            del(mr_adc.v2e.ccee)

            mr_adc.v2e.xexe = np.ascontiguousarray(mr_adc.v2e.cece[:ncvs, :, :ncvs, :])
            mr_adc.v2e.xeve = np.ascontiguousarray(mr_adc.v2e.cece[:ncvs, :, ncvs:, :])
            mr_adc.v2e.vexe = np.ascontiguousarray(mr_adc.v2e.cece[ncvs:, :, :ncvs, :])
            mr_adc.v2e.veve = np.ascontiguousarray(mr_adc.v2e.cece[ncvs:, :, ncvs:, :])
            del(mr_adc.v2e.cece)

            mr_adc.v2e.xeex = np.ascontiguousarray(mr_adc.v2e.ceec[:ncvs, :, :, :ncvs])
            del(mr_adc.v2e.ceec)

            mr_adc.v2e.xaaa = np.ascontiguousarray(mr_adc.v2e.caaa[:ncvs, :, :, :])
            mr_adc.v2e.vaaa = np.ascontiguousarray(mr_adc.v2e.caaa[ncvs:, :, :, :])
            del(mr_adc.v2e.caaa)

            mr_adc.v2e.xaae = np.ascontiguousarray(mr_adc.v2e.caae[:ncvs, :, :, :])
            mr_adc.v2e.vaae = np.ascontiguousarray(mr_adc.v2e.caae[ncvs:, :, :, :])
            del(mr_adc.v2e.caae)

            mr_adc.v2e.xeaa = np.ascontiguousarray(mr_adc.v2e.ceaa[:ncvs, :, :, :])
            mr_adc.v2e.veaa = np.ascontiguousarray(mr_adc.v2e.ceaa[ncvs:, :, :, :])
            del(mr_adc.v2e.ceaa)

            mr_adc.v2e.xeae = np.ascontiguousarray(mr_adc.v2e.ceae[:ncvs, :, :, :])
            mr_adc.v2e.veae = np.ascontiguousarray(mr_adc.v2e.ceae[ncvs:, :, :, :])
            del(mr_adc.v2e.ceae)
            
        if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"): ##comment out if checking M_00 block

            ###WiP: use the inefficient oeee integrals until cvs-ee blocks are fully implemented
            mr_adc.v2e.ceee = unpack_v2e_oeee(mr_adc, mr_adc.v2e.ceee)
            mr_adc.v2e.aeee = unpack_v2e_oeee(mr_adc, mr_adc.v2e.aeee)
            ###
            mr_adc.v2e.xeee = np.ascontiguousarray(mr_adc.v2e.ceee[:ncvs, :, :, :])
            mr_adc.v2e.veee = np.ascontiguousarray(mr_adc.v2e.ceee[ncvs:, :, :, :])
            del(mr_adc.v2e.ceee)

            mr_adc.v2e.xeea = tools.create_dataset('xeea', tmpfile, (ncvs, nextern, nextern, ncas))
            mr_adc.v2e.veea = tools.create_dataset('veea', tmpfile, (nval, nextern, nextern, ncas))

            mr_adc.v2e.xeea[:] = np.ascontiguousarray(mr_adc.v2e.ceea[:ncvs, :, :, :])
            tools.flush(tmpfile)

            mr_adc.v2e.veea[:] = np.ascontiguousarray(mr_adc.v2e.ceea[ncvs:, :, :, :])
            tools.flush(tmpfile)
            del(mr_adc.v2e.ceea)

            mr_adc.v2e.xaea = np.ascontiguousarray(mr_adc.v2e.caea[:ncvs, :, :, :])
            mr_adc.v2e.vaea = np.ascontiguousarray(mr_adc.v2e.caea[ncvs:, :, :, :])
            del(mr_adc.v2e.caea)

            mr_adc.v2e.xaee = np.ascontiguousarray(mr_adc.v2e.caee[:ncvs, :, :, :])
            mr_adc.v2e.vaee = np.ascontiguousarray(mr_adc.v2e.caee[ncvs:, :, :, :])
            del(mr_adc.v2e.caee)

    # Effective one-electron integrals
    mr_adc.v2e.xxxa = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs, :ncvs, :ncvs, :])
    mr_adc.v2e.xxva = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs, :ncvs, ncvs:, :])
    mr_adc.v2e.vxxa = np.ascontiguousarray(mr_adc.v2e.ccca[ncvs:, :ncvs, :ncvs, :])
    del(mr_adc.v2e.ccca)

    mr_adc.v2e.xxxe = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs, :ncvs, :ncvs, :])
    mr_adc.v2e.xxve = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs, :ncvs, ncvs:, :])
    mr_adc.v2e.vxxe = np.ascontiguousarray(mr_adc.v2e.ccce[ncvs:, :ncvs, :ncvs, :])
    del(mr_adc.v2e.ccce)

    mr_adc.h1eff.xa = np.ascontiguousarray(mr_adc.h1eff.ca[:ncvs,:])
    mr_adc.h1eff.va = np.ascontiguousarray(mr_adc.h1eff.ca[ncvs:,:])
    del(mr_adc.h1eff.ca)

    mr_adc.h1eff.xe = np.ascontiguousarray(mr_adc.h1eff.ce[:ncvs,:])
    mr_adc.h1eff.ve = np.ascontiguousarray(mr_adc.h1eff.ce[ncvs:,:])
    del(mr_adc.h1eff.ce)

    # Store diagonal elements of the generalized Fock operator
    mr_adc.mo_energy.x = mr_adc.mo_energy.c[:ncvs]
    mr_adc.mo_energy.v = mr_adc.mo_energy.c[ncvs:]

    mr_adc.log.timer("computing CVS integrals", *cput0)

def compute_cvs_integrals_2e_df(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.extra("\nComputing CVS integrals to MO basis (density-fitting)...")

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    # Remove in-core v2e integrals
    del(mr_adc.v2e.Lce, mr_adc.v2e.Lae, mr_adc.v2e.Lee)

    mr_adc.tmpfile.xferi1 = tools.create_temp_file(mr_adc)
    tmpfile = mr_adc.tmpfile.xferi1

    # Effective one-electron integrals
    mr_adc.v2e.xxxa = tools.create_dataset('xxxa', tmpfile, (ncvs, ncvs, ncvs, ncas))
    mr_adc.v2e.xxva = tools.create_dataset('xxva', tmpfile, (ncvs, ncvs, nval, ncas))
    mr_adc.v2e.vxxa = tools.create_dataset('vxxa', tmpfile, (nval, ncvs, ncvs, ncas))

    mr_adc.v2e.xxxe = tools.create_dataset('xxxe', tmpfile, (ncvs, ncvs, ncvs, nextern))
    mr_adc.v2e.xxve = tools.create_dataset('xxve', tmpfile, (ncvs, ncvs, nval, nextern))
    mr_adc.v2e.vxxe = tools.create_dataset('vxxe', tmpfile, (nval, ncvs, ncvs, nextern))


    mr_adc.v2e.xxxa[:] = mr_adc.v2e.ccca[:ncvs, :ncvs, :ncvs, :]
    tools.flush(tmpfile)

    mr_adc.v2e.xxva[:] = mr_adc.v2e.ccca[:ncvs, :ncvs, ncvs:, :]
    tools.flush(tmpfile)

    mr_adc.v2e.vxxa[:] = mr_adc.v2e.ccca[ncvs:, :ncvs, :ncvs, :]
    tools.flush(tmpfile)
    del(mr_adc.v2e.ccca)


    mr_adc.v2e.xxxe[:] = mr_adc.v2e.ccce[:ncvs, :ncvs, :ncvs, :]
    tools.flush(tmpfile)

    mr_adc.v2e.xxve[:] = mr_adc.v2e.ccce[:ncvs, :ncvs, ncvs:, :]
    tools.flush(tmpfile)

    mr_adc.v2e.vxxe[:] = mr_adc.v2e.ccce[ncvs:, :ncvs, :ncvs, :]
    tools.flush(tmpfile)
    del(mr_adc.v2e.ccce)

    mr_adc.h1eff.xa = np.ascontiguousarray(mr_adc.h1eff.ca[:ncvs,:])
    mr_adc.h1eff.va = np.ascontiguousarray(mr_adc.h1eff.ca[ncvs:,:])
    del(mr_adc.h1eff.ca)

    mr_adc.h1eff.xe = np.ascontiguousarray(mr_adc.h1eff.ce[:ncvs,:])
    mr_adc.h1eff.ve = np.ascontiguousarray(mr_adc.h1eff.ce[ncvs:,:])
    del(mr_adc.h1eff.ce)

    # Compute CVS integrals
    if mr_adc.method_type == "cvs-ip":
        if mr_adc.method in ("mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.xxaa = tools.create_dataset('xxaa', tmpfile, (ncvs, ncvs, ncas, ncas))
            mr_adc.v2e.xvaa = tools.create_dataset('xvaa', tmpfile, (ncvs, nval, ncas, ncas))
            mr_adc.v2e.vxaa = tools.create_dataset('vxaa', tmpfile, (nval, ncvs, ncas, ncas))
            mr_adc.v2e.vvaa = tools.create_dataset('vvaa', tmpfile, (nval, nval, ncas, ncas))

            mr_adc.v2e.xxae = tools.create_dataset('xxae', tmpfile, (ncvs, ncvs, ncas, nextern))
            mr_adc.v2e.xvae = tools.create_dataset('xvae', tmpfile, (ncvs, nval, ncas, nextern))
            mr_adc.v2e.vxae = tools.create_dataset('vxae', tmpfile, (nval, ncvs, ncas, nextern))
            mr_adc.v2e.vvae = tools.create_dataset('vvae', tmpfile, (nval, nval, ncas, nextern))

            mr_adc.v2e.xaax = tools.create_dataset('xaax', tmpfile, (ncvs, ncas, ncas, ncvs))
            mr_adc.v2e.xaav = tools.create_dataset('xaav', tmpfile, (ncvs, ncas, ncas, nval))
            mr_adc.v2e.vaax = tools.create_dataset('vaax', tmpfile, (nval, ncas, ncas, ncvs))
            mr_adc.v2e.vaav = tools.create_dataset('vaav', tmpfile, (nval, ncas, ncas, nval))

            mr_adc.v2e.xaex = tools.create_dataset('xaex', tmpfile, (ncvs, ncas, nextern, ncvs))
            mr_adc.v2e.xaev = tools.create_dataset('xaev', tmpfile, (ncvs, ncas, nextern, nval))
            mr_adc.v2e.vaex = tools.create_dataset('vaex', tmpfile, (nval, ncas, nextern, ncvs))
            mr_adc.v2e.vaev = tools.create_dataset('vaev', tmpfile, (nval, ncas, nextern, nval))

            mr_adc.v2e.xaxa = tools.create_dataset('xaxa', tmpfile, (ncvs, ncas, ncvs, ncas))
            mr_adc.v2e.xava = tools.create_dataset('xava', tmpfile, (ncvs, ncas, nval, ncas))

            mr_adc.v2e.xexe = tools.create_dataset('xexe', tmpfile, (ncvs, nextern, ncvs, nextern))
            mr_adc.v2e.xeve = tools.create_dataset('xeve', tmpfile, (ncvs, nextern, nval, nextern))

            mr_adc.v2e.xaxe = tools.create_dataset('xaxe', tmpfile, (ncvs, ncas, ncvs, nextern))
            mr_adc.v2e.xave = tools.create_dataset('xave', tmpfile, (ncvs, ncas, nval, nextern))
            mr_adc.v2e.vaxe = tools.create_dataset('vaxe', tmpfile, (nval, ncas, ncvs, nextern))

            mr_adc.v2e.xaaa = tools.create_dataset('xaaa', tmpfile, (ncvs, ncas, ncas, ncas))
            mr_adc.v2e.vaaa = tools.create_dataset('vaaa', tmpfile, (nval, ncas, ncas, ncas))

            mr_adc.v2e.xeae = tools.create_dataset('xeae', tmpfile, (ncvs, nextern, ncas, nextern))

            mr_adc.v2e.xaae = tools.create_dataset('xaae', tmpfile, (ncvs, ncas, ncas, nextern))
            mr_adc.v2e.vaae = tools.create_dataset('vaae', tmpfile, (nval, ncas, ncas, nextern))

            mr_adc.v2e.xeaa = tools.create_dataset('xeaa', tmpfile, (ncvs, nextern, ncas, ncas))
            mr_adc.v2e.veaa = tools.create_dataset('veaa', tmpfile, (nval, nextern, ncas, ncas))


            mr_adc.v2e.xxaa[:] = mr_adc.v2e.ccaa[:ncvs, :ncvs, :, :]
            tools.flush(tmpfile)

            mr_adc.v2e.xvaa[:] = mr_adc.v2e.ccaa[:ncvs, ncvs:, :, :]
            tools.flush(tmpfile)

            mr_adc.v2e.vxaa[:] = mr_adc.v2e.ccaa[ncvs:, :ncvs, :, :]
            tools.flush(tmpfile)

            mr_adc.v2e.vvaa[:] = mr_adc.v2e.ccaa[ncvs:, ncvs:, :, :]
            tools.flush(tmpfile)
            del(mr_adc.v2e.ccaa)


            mr_adc.v2e.xxae[:] = mr_adc.v2e.ccae[:ncvs, :ncvs, :, :]
            tools.flush(tmpfile)

            mr_adc.v2e.xvae[:] = mr_adc.v2e.ccae[:ncvs, ncvs:, :, :]
            tools.flush(tmpfile)

            mr_adc.v2e.vxae[:] = mr_adc.v2e.ccae[ncvs:, :ncvs, :, :]
            tools.flush(tmpfile)

            mr_adc.v2e.vvae[:] = mr_adc.v2e.ccae[ncvs:, ncvs:, :, :]
            tools.flush(tmpfile)
            del(mr_adc.v2e.ccae)


            mr_adc.v2e.xaax[:] = mr_adc.v2e.caac[:ncvs, :, :, :ncvs]
            tools.flush(tmpfile)

            mr_adc.v2e.xaav[:] = mr_adc.v2e.caac[:ncvs, :, :, ncvs:]
            tools.flush(tmpfile)

            mr_adc.v2e.vaax[:] = mr_adc.v2e.caac[ncvs:, :, :, :ncvs]
            tools.flush(tmpfile)

            mr_adc.v2e.vaav[:] = mr_adc.v2e.caac[ncvs:, :, :, ncvs:]
            tools.flush(tmpfile)
            del(mr_adc.v2e.caac)


            mr_adc.v2e.xaex[:] = mr_adc.v2e.caec[:ncvs, :, :, :ncvs]
            tools.flush(tmpfile)

            mr_adc.v2e.xaev[:] = mr_adc.v2e.caec[:ncvs, :, :, ncvs:]
            tools.flush(tmpfile)

            mr_adc.v2e.vaex[:] = mr_adc.v2e.caec[ncvs:, :, :, :ncvs]
            tools.flush(tmpfile)

            mr_adc.v2e.vaev[:] = mr_adc.v2e.caec[ncvs:, :, :, ncvs:]
            tools.flush(tmpfile)
            del(mr_adc.v2e.caec)


            mr_adc.v2e.xaxa[:] = mr_adc.v2e.caca[:ncvs, :, :ncvs, :]
            tools.flush(tmpfile)

            mr_adc.v2e.xava[:] = mr_adc.v2e.caca[:ncvs, :, ncvs:, :]
            tools.flush(tmpfile)
            del(mr_adc.v2e.caca)

            mr_adc.v2e.xaxe[:] = mr_adc.v2e.cace[:ncvs, :, :ncvs, :]
            tools.flush(tmpfile)

            mr_adc.v2e.xave[:] = mr_adc.v2e.cace[:ncvs, :, ncvs:, :]
            tools.flush(tmpfile)

            mr_adc.v2e.vaxe[:] = mr_adc.v2e.cace[ncvs:, :, :ncvs, :]
            tools.flush(tmpfile)
            del(mr_adc.v2e.cace)

            mr_adc.v2e.xaaa[:] = mr_adc.v2e.caaa[:ncvs, :, :, :]
            tools.flush(tmpfile)

            mr_adc.v2e.vaaa[:] = mr_adc.v2e.caaa[ncvs:, :, :, :]
            tools.flush(tmpfile)
            del(mr_adc.v2e.caaa)


            mr_adc.v2e.xaae[:] = mr_adc.v2e.caae[:ncvs, :, :, :]
            tools.flush(tmpfile)

            mr_adc.v2e.vaae[:] = mr_adc.v2e.caae[ncvs:, :, :, :]
            tools.flush(tmpfile)
            del(mr_adc.v2e.caae)


            mr_adc.v2e.xeaa[:] = mr_adc.v2e.ceaa[:ncvs, :, :, :]
            tools.flush(tmpfile)

            mr_adc.v2e.veaa[:] = mr_adc.v2e.ceaa[ncvs:, :, :, :]
            tools.flush(tmpfile)
            del(mr_adc.v2e.ceaa)


            chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncore, nextern])
            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.xexe [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.xexe[:,s_chunk:f_chunk] = mr_adc.v2e.cece[:ncvs, s_chunk:f_chunk, :ncvs, :]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.xexe", *cput1)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.xeve [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.xeve[:,s_chunk:f_chunk] = mr_adc.v2e.cece[:ncvs, s_chunk:f_chunk, ncvs:, :]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.xeve", *cput1)
            del(mr_adc.v2e.cece)


            chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncas, nextern])
            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.xeae [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.xeae[:,s_chunk:f_chunk] = mr_adc.v2e.ceae[:ncvs, s_chunk:f_chunk]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.xeae", *cput1)
            del(mr_adc.v2e.ceae)

        if mr_adc.method in ("mr-adc(2)-x"):
            mr_adc.v2e.xxxx = tools.create_dataset('xxxx', tmpfile, (ncvs, ncvs, ncvs, ncvs))
            mr_adc.v2e.xxvv = tools.create_dataset('xxvv', tmpfile, (ncvs, ncvs, nval, nval))
            mr_adc.v2e.xvvx = tools.create_dataset('xvvx', tmpfile, (ncvs, nval, nval, ncvs))
            mr_adc.v2e.xxvx = tools.create_dataset('xxvx', tmpfile, (ncvs, ncvs, nval, ncvs))
            mr_adc.v2e.xxxv = tools.create_dataset('xxxv', tmpfile, (ncvs, ncvs, ncvs, nval))
            mr_adc.v2e.xvxx = tools.create_dataset('xvxx', tmpfile, (ncvs, nval, ncvs, ncvs))

            mr_adc.v2e.xxee = tools.create_dataset('xxee', tmpfile, (ncvs, ncvs, nextern, nextern))
            mr_adc.v2e.xvee = tools.create_dataset('xvee', tmpfile, (ncvs, nval, nextern, nextern))
            mr_adc.v2e.vxee = tools.create_dataset('vxee', tmpfile, (nval, ncvs, nextern, nextern))
            mr_adc.v2e.vvee = tools.create_dataset('vvee', tmpfile, (nval, nval, nextern, nextern))

            mr_adc.v2e.xeex = tools.create_dataset('xeex', tmpfile, (ncvs, nextern, nextern, ncvs))
            mr_adc.v2e.xeev = tools.create_dataset('xeev', tmpfile, (ncvs, nextern, nextern, nval))
            mr_adc.v2e.veex = tools.create_dataset('veex', tmpfile, (nval, nextern, nextern, ncvs))
            mr_adc.v2e.veev = tools.create_dataset('veev', tmpfile, (nval, nextern, nextern, nval))

            mr_adc.v2e.xaea = tools.create_dataset('xaea', tmpfile, (ncvs, ncas, nextern, ncas))
            mr_adc.v2e.vaea = tools.create_dataset('vaea', tmpfile, (nval, ncas, nextern, ncas))

            mr_adc.v2e.xaee = tools.create_dataset('xaee', tmpfile, (ncvs, ncas, nextern, nextern))
            mr_adc.v2e.vaee = tools.create_dataset('vaee', tmpfile, (nval, ncas, nextern, nextern))

            mr_adc.v2e.xeea = tools.create_dataset('xeea', tmpfile, (ncvs, nextern, nextern, ncas))
            mr_adc.v2e.veea = tools.create_dataset('veea', tmpfile, (nval, nextern, nextern, ncas))


            mr_adc.v2e.xxxx[:] = mr_adc.v2e.cccc[:ncvs, :ncvs, :ncvs, :ncvs]
            tools.flush(tmpfile)

            mr_adc.v2e.xxvv[:] = mr_adc.v2e.cccc[:ncvs, :ncvs, ncvs:, ncvs:]
            tools.flush(tmpfile)

            mr_adc.v2e.xvvx[:] = mr_adc.v2e.cccc[:ncvs, ncvs:, ncvs:, :ncvs]
            tools.flush(tmpfile)

            mr_adc.v2e.xxvx[:] = mr_adc.v2e.cccc[:ncvs, :ncvs, ncvs:, :ncvs]
            tools.flush(tmpfile)

            mr_adc.v2e.xxxv[:] = mr_adc.v2e.cccc[:ncvs, :ncvs, :ncvs, ncvs:]
            tools.flush(tmpfile)

            mr_adc.v2e.xvxx[:] = mr_adc.v2e.cccc[:ncvs, ncvs:, :ncvs, :ncvs]
            tools.flush(tmpfile)
            del(mr_adc.v2e.cccc)

            mr_adc.v2e.xaea[:] = mr_adc.v2e.caea[:ncvs, :, :, :]
            tools.flush(tmpfile)

            mr_adc.v2e.vaea[:] = mr_adc.v2e.caea[ncvs:, :, :, :]
            tools.flush(tmpfile)
            del(mr_adc.v2e.caea)


            chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncore, nextern])
            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.xxee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.xxee[:,:,:,s_chunk:f_chunk] = mr_adc.v2e.ccee[:ncvs, :ncvs, :, s_chunk:f_chunk]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.xxee", *cput1)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.xvee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.xvee[:,:,:,s_chunk:f_chunk] = mr_adc.v2e.ccee[:ncvs, ncvs:, :, s_chunk:f_chunk]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.xvee", *cput1)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.vxee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.vxee[:,:,:,s_chunk:f_chunk] = mr_adc.v2e.ccee[ncvs:, :ncvs, :, s_chunk:f_chunk]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.vxee", *cput1)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.vvee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.vvee[:,:,:,s_chunk:f_chunk] = mr_adc.v2e.ccee[ncvs:, ncvs:, :, s_chunk:f_chunk]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.vvee", *cput1)
            del(mr_adc.v2e.ccee)


            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.xeex [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.xeex[:,:,s_chunk:f_chunk] = mr_adc.v2e.ceec[:ncvs, :, s_chunk:f_chunk, :ncvs]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.xeex", *cput1)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.xeev [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.xeev[:,:,s_chunk:f_chunk] = mr_adc.v2e.ceec[:ncvs, :, s_chunk:f_chunk, ncvs:]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.xeev", *cput1)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.veex [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.veex[:,:,s_chunk:f_chunk] = mr_adc.v2e.ceec[ncvs:, :, s_chunk:f_chunk, :ncvs]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.veex", *cput1)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.veev [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.veev[:,:,s_chunk:f_chunk] = mr_adc.v2e.ceec[ncvs:, :, s_chunk:f_chunk, ncvs:]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.veev", *cput1)
            del(mr_adc.v2e.ceec)


            chunks = tools.calculate_chunks(mr_adc, nextern, [ncore, ncas, nextern])
            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.xaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.xaee[:,:,s_chunk:f_chunk] = mr_adc.v2e.caee[:ncvs, :, s_chunk:f_chunk]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.xaee", *cput1)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.vaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.vaee[:,:,s_chunk:f_chunk] = mr_adc.v2e.caee[ncvs:, :, s_chunk:f_chunk]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.vaee", *cput1)
            del(mr_adc.v2e.caee)


            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.xeea [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.xeea[:,s_chunk:f_chunk] = mr_adc.v2e.ceea[:ncvs, s_chunk:f_chunk]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.xeea", *cput1)

            for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
                cput1 = (logger.process_clock(), logger.perf_counter())
                mr_adc.log.debug("v2e.veea [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

                mr_adc.v2e.veea[:,s_chunk:f_chunk] = mr_adc.v2e.ceea[ncvs:, s_chunk:f_chunk]
                tools.flush(tmpfile)
                mr_adc.log.timer_debug("storing CVS v2e.veea", *cput1)
            del(mr_adc.v2e.ceea)

    # Close non-CVS integrals' files
    mr_adc.tmpfile.cferi1.close()

    # Store diagonal elements of the generalized Fock operator
    mr_adc.mo_energy.x = mr_adc.mo_energy.c[:ncvs]
    mr_adc.mo_energy.v = mr_adc.mo_energy.c[ncvs:]

    mr_adc.log.timer("computing CVS integrals", *cput0)
