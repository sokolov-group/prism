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
# Authors: Carlos E. V. de Moura <carlosevmoura@gmail.com>
#          Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#

import numpy as np
from functools import reduce

import prism.lib.logger as logger
import prism.lib.tools as tools

def transform_integrals_1e(nevpt):

    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.extra("\nTransforming 1e integrals to MO basis...")

    mo = nevpt.mo

    nevpt.h1e = reduce(np.dot, (mo.T, nevpt.interface.h1e_ao, mo))

    nevpt.log.timer("transforming 1e integrals", *cput0)

def transform_2e_chem_incore(interface, mo_1, mo_2, mo_3, mo_4, compacted=False):
    'Two-electron integral transformation in Chemists notation'

    nmo_1 = mo_1.shape[1]
    nmo_2 = mo_2.shape[1]
    nmo_3 = mo_3.shape[1]
    nmo_4 = mo_4.shape[1]

    v2e = interface.transform_2e_chem_incore(interface.v2e_ao, (mo_1, mo_2, mo_3, mo_4), compact=compacted)
    if compacted:
        if nmo_1 == 0 or nmo_2 == 0:
            v2e = np.zeros((nmo_1, nmo_2, nmo_3 * nmo_4))
        else:
            v2e = v2e.reshape(nmo_1, nmo_2, -1)
    else:
        if nmo_1 == 0 or nmo_2 == 0 or nmo_3 == 0 or nmo_4 == 0:
            v2e = np.zeros((nmo_1, nmo_2, nmo_3, nmo_4))
        else:
            v2e = v2e.reshape(nmo_1, nmo_2, nmo_3, nmo_4)

    return np.ascontiguousarray(v2e)

def compute_effective_1e(nevpt, h1e_pq, v2e_ccpq, v2e_cpqc):
    'Effective one-electron integrals'

    # Einsum definition from kernel
    einsum = nevpt.interface.einsum
    einsum_type = nevpt.interface.einsum_type

    h1eff  = h1e_pq
    h1eff += 2.0 * einsum('rrpq->pq', v2e_ccpq, optimize = einsum_type)
    h1eff -= einsum('rpqr->pq', v2e_cpqc, optimize = einsum_type)

    return h1eff

def transform_integrals_2e_incore(nevpt):

    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.extra("\nTransforming 2e integrals to MO basis (in-core)...")

    # Import Prism interface
    interface = nevpt.interface

    # Variables from kernel
    nfrozen = nevpt.nfrozen
    ncore = nevpt.ncore
    ncore_wof = ncore - nfrozen
    nocc = nevpt.nocc
    ncas = nevpt.ncas
    nextern = nevpt.nextern

    mo = nevpt.mo
    mo_c = mo[:, :ncore].copy()
    mo_c_wof = mo[:, nfrozen:ncore].copy()
    mo_a = mo[:, ncore:nocc].copy()
    mo_e = mo[:, nocc:].copy()

    if nevpt.outcore_expensive_tensors:
        nevpt.tmpfile.feri1 = tools.create_temp_file(nevpt)
    else:
        nevpt.tmpfile.feri1 = None
    tmpfile = nevpt.tmpfile.feri1

    # Effective one-electron integrals
    nevpt.v2e.ccaa = transform_2e_chem_incore(interface, mo_c, mo_c, mo_a, mo_a)
    nevpt.v2e.ccae = transform_2e_chem_incore(interface, mo_c, mo_c, mo_a, mo_e)
    nevpt.v2e.caac = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_c)
    nevpt.v2e.caec = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_c)
    nevpt.v2e.ccca = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_a)
    nevpt.v2e.ccce = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_e)

    v2e_ccac = nevpt.v2e.ccca.transpose(1,0,3,2)
    v2e_ccec = nevpt.v2e.ccce.transpose(1,0,3,2)

    nevpt.h1eff.ca = compute_effective_1e(nevpt, nevpt.h1e[:ncore, ncore:nocc], nevpt.v2e.ccca, v2e_ccac)
    nevpt.h1eff.ce = compute_effective_1e(nevpt, nevpt.h1e[:ncore, nocc:], nevpt.v2e.ccce, v2e_ccec)
    nevpt.h1eff.aa = compute_effective_1e(nevpt, nevpt.h1e[ncore:nocc, ncore:nocc], nevpt.v2e.ccaa, nevpt.v2e.caac)
    nevpt.h1eff.ae = compute_effective_1e(nevpt, nevpt.h1e[ncore:nocc, nocc:], nevpt.v2e.ccae, nevpt.v2e.caec)

    if nfrozen > 0:
        nevpt.h1eff.ca = nevpt.h1eff.ca[nfrozen:,:].copy()
        nevpt.h1eff.ce = nevpt.h1eff.ce[nfrozen:,:].copy()
        nevpt.mo_energy.c = nevpt.mo_energy.c[nfrozen:]

    # Other integrals
    nevpt.v2e.aaaa = transform_2e_chem_incore(interface, mo_a, mo_a, mo_a, mo_a)

    nevpt.v2e.caca = transform_2e_chem_incore(interface, mo_c_wof, mo_a, mo_c_wof, mo_a)
    nevpt.v2e.cace = transform_2e_chem_incore(interface, mo_c_wof, mo_a, mo_c_wof, mo_e)

    nevpt.v2e.caaa = transform_2e_chem_incore(interface, mo_c_wof, mo_a, mo_a, mo_a)
    nevpt.v2e.caae = transform_2e_chem_incore(interface, mo_c_wof, mo_a, mo_a, mo_e)
    nevpt.v2e.ceaa = transform_2e_chem_incore(interface, mo_c_wof, mo_e, mo_a, mo_a)

    nevpt.v2e.aaae = transform_2e_chem_incore(interface, mo_a, mo_a, mo_a, mo_e)

    nevpt.v2e.cece = tools.create_dataset('cece', tmpfile, (ncore_wof, nextern, ncore_wof, nextern))
    nevpt.v2e.ceae = tools.create_dataset('ceae', tmpfile, (ncore_wof, nextern, ncas, nextern))

    nevpt.v2e.cece[:] = transform_2e_chem_incore(interface, mo_c_wof, mo_e, mo_c_wof, mo_e)
    nevpt.v2e.ceae[:] = transform_2e_chem_incore(interface, mo_c_wof, mo_e, mo_a, mo_e)

    nevpt.v2e.cccc = transform_2e_chem_incore(interface, mo_c_wof, mo_c_wof, mo_c_wof, mo_c_wof)

    nevpt.v2e.caea = transform_2e_chem_incore(interface, mo_c_wof, mo_a, mo_e, mo_a)

    nevpt.v2e.ccee = tools.create_dataset('ccee', tmpfile, (ncore_wof, ncore_wof, nextern, nextern))
    nevpt.v2e.ceec = tools.create_dataset('ceec', tmpfile, (ncore_wof, nextern, nextern, ncore_wof))

    nevpt.v2e.caee = tools.create_dataset('caee', tmpfile, (ncore_wof, ncas, nextern, nextern))
    nevpt.v2e.ceea = tools.create_dataset('ceea', tmpfile, (ncore_wof, nextern, nextern, ncas))

    nevpt.v2e.aeae = tools.create_dataset('aeae', tmpfile, (ncas, nextern, ncas, nextern))
    nevpt.v2e.aaee = tools.create_dataset('aaee', tmpfile, (ncas, ncas, nextern, nextern))
    nevpt.v2e.aeea = tools.create_dataset('aeea', tmpfile, (ncas, nextern, nextern, ncas))

    nevpt.v2e.ccee[:] = transform_2e_chem_incore(interface, mo_c_wof, mo_c_wof, mo_e, mo_e)
    nevpt.v2e.ceec[:] = transform_2e_chem_incore(interface, mo_c_wof, mo_e, mo_e, mo_c_wof)

    nevpt.v2e.caee[:] = transform_2e_chem_incore(interface, mo_c_wof, mo_a, mo_e, mo_e)
    nevpt.v2e.ceea[:] = transform_2e_chem_incore(interface, mo_c_wof, mo_e, mo_e, mo_a)

    nevpt.v2e.aeae[:] = transform_2e_chem_incore(interface, mo_a, mo_e, mo_a, mo_e)
    nevpt.v2e.aaee[:] = transform_2e_chem_incore(interface, mo_a, mo_a, mo_e, mo_e)
    nevpt.v2e.aeea[:] = transform_2e_chem_incore(interface, mo_a, mo_e, mo_e, mo_a)

    nevpt.log.timer("transforming 1e integrals", *cput0)

def transform_Heff_integrals_2e_df(nevpt):

    cput0 = (logger.process_clock(), logger.perf_counter())

    # Import Prism interface
    interface = nevpt.interface

    # Variables from kernel
    ncore = nevpt.ncore
    ncas = nevpt.ncas
    nocc = nevpt.nocc

    nmo = nevpt.nmo
    mo = nevpt.mo
    mo_c = mo[:, :ncore].copy()
    mo_a = mo[:, ncore:nocc].copy()

    # Create temp file and datasets
    nevpt.tmpfile.feri0 = tools.create_temp_file(nevpt) # Non-core indices' integrals
    nevpt.tmpfile.cferi0 = tools.create_temp_file(nevpt) # Core indices' integrals

    tmpfile = nevpt.tmpfile.feri0
    ctmpfile = nevpt.tmpfile.cferi0

    nevpt.v2e.aaaa = tools.create_dataset('aaaa', tmpfile, (ncas, ncas, ncas, ncas))
    nevpt.v2e.ccca = tools.create_dataset('ccca', ctmpfile, (ncore, ncore, ncore, ncas))

    nevpt.v2e.ccaa = tools.create_dataset('ccaa', ctmpfile, (ncore, ncore, ncas, ncas))
    nevpt.v2e.caac = tools.create_dataset('caac', ctmpfile, (ncore, ncas, ncas, ncore))

    # Atomic orbitals auxiliary basis-set
    if interface.reference_df:
        nevpt.log.extra("\nTransforming Heff 2e integrals to MO basis (density-fitting)...\n")

        with_df = interface.reference_df
        with_df.max_memory = nevpt.max_memory
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
        nevpt.v2e.aaaa[:] = get_v2e_df(nevpt, Laa, Laa, 'aaaa')
        tools.flush(tmpfile)

        nevpt.v2e.ccca[:] = get_v2e_df(nevpt, Lcc, Lca, 'ccca')
        tools.flush(ctmpfile)

        nevpt.v2e.ccaa[:] = get_v2e_df(nevpt, Lcc, Laa, 'ccaa')
        tools.flush(ctmpfile)

        nevpt.v2e.caac[:] = get_v2e_df(nevpt, Lca, Lac, 'caac')
        tools.flush(ctmpfile)

    else:
        nevpt.log.extra("\nTransforming Heff 2e integrals to MO basis (in-core)...")

        # Effective Hamiltonian 2e- integrals
        nevpt.v2e.aaaa[:] = transform_2e_chem_incore(interface, mo_a, mo_a, mo_a, mo_a)
        tools.flush(tmpfile)

        nevpt.v2e.ccca[:] = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_a)
        tools.flush(ctmpfile)

        nevpt.v2e.ccaa[:] = transform_2e_chem_incore(interface, mo_c, mo_c, mo_a, mo_a)
        tools.flush(ctmpfile)

        nevpt.v2e.caac[:] = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_c)
        tools.flush(ctmpfile)

    nevpt.log.timer("transforming 2e integrals", *cput0)

def transform_integrals_2e_df(nevpt):

    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.extra("\nTransforming 2e integrals to MO basis (density-fitting)...\n")

    # Import Prism interface
    interface = nevpt.interface

    # Atomic orbitals auxiliary basis-set
    with_df = interface.with_df
    with_df.max_memory = nevpt.max_memory
    naux = interface.get_naux()

    # Variables from kernel
    nfrozen = nevpt.nfrozen
    ncore = nevpt.ncore
    ncore_wof = ncore - nfrozen
    ncas = nevpt.ncas
    nocc = nevpt.nocc
    nextern = nevpt.nextern

    nmo = nevpt.nmo
    mo = nevpt.mo

    # Create temp file and datasets
    nevpt.tmpfile.feri1 = tools.create_temp_file(nevpt) # Non-core indices' integrals
    nevpt.tmpfile.cferi1 = tools.create_temp_file(nevpt) # Core indices' integras

    tmpfile = nevpt.tmpfile.feri1
    ctmpfile = nevpt.tmpfile.cferi1

    Lcc = np.empty((naux, ncore, ncore))
    Lca = np.empty((naux, ncore, ncas))
    Lac = np.empty((naux, ncas, ncore))
    Laa = np.empty((naux, ncas, ncas))

    Lec = np.empty((naux, nextern,  ncore))
    Lea = np.empty((naux, nextern, ncas))

    nevpt.naux = naux
    nevpt.v2e.Lee = tools.create_dataset('Lee', tmpfile, (naux, nextern, nextern))
    nevpt.v2e.Lce = np.empty((naux, ncore, nextern))
    nevpt.v2e.Lae = np.empty((naux, ncas, nextern))

    ijslice = (0, nmo, 0, nmo)
    Lpq = None
    p1 = 0

    for eri1 in with_df.loop():
        Lpq = interface.transform_2e_pair_chem_incore(eri1, mo, ijslice, aosym='s2', out=Lpq).reshape(-1, nmo, nmo)

        p0, p1 = p1, p1 + Lpq.shape[0]
        Lcc[p0:p1] = Lpq[:, :ncore, :ncore]
        Lca[p0:p1] = Lpq[:, :ncore, ncore:nocc]
        nevpt.v2e.Lce[p0:p1] = Lpq[:, :ncore, nocc:]

        Lac[p0:p1] = Lpq[:, ncore:nocc, :ncore]
        Laa[p0:p1] = Lpq[:, ncore:nocc, ncore:nocc]
        nevpt.v2e.Lae[p0:p1] = Lpq[:, ncore:nocc, nocc:]

        Lec[p0:p1] = Lpq[:, nocc:, :ncore]
        Lea[p0:p1] = Lpq[:, nocc:, ncore:nocc]
        nevpt.v2e.Lee[p0:p1] = Lpq[:, nocc:, nocc:]
        tools.flush(tmpfile)
    del(Lpq)

    # Effective one-electron integrals
    nevpt.v2e.ccae = tools.create_dataset('ccae', ctmpfile, (ncore, ncore, ncas, nextern))
    nevpt.v2e.ccae[:] = get_v2e_df(nevpt, Lcc, nevpt.v2e.Lae, 'ccae')
    tools.flush(ctmpfile)

    nevpt.v2e.caec = tools.create_dataset('caec', ctmpfile, (ncore, ncas, nextern, ncore))
    nevpt.v2e.caec[:] = get_v2e_df(nevpt, Lca, Lec, 'caec')
    tools.flush(ctmpfile)

    nevpt.v2e.ccce = tools.create_dataset('ccce', ctmpfile, (ncore, ncore, ncore, nextern))
    nevpt.v2e.ccce[:] = get_v2e_df(nevpt, Lcc, nevpt.v2e.Lce, 'ccce')
    tools.flush(ctmpfile)

    nevpt.v2e.ccac = tools.create_dataset('ccac', ctmpfile, (ncore, ncore, ncas, ncore))
    nevpt.v2e.ccac[:] = get_v2e_df(nevpt, Lcc, Lac, 'ccac')
    tools.flush(ctmpfile)

    nevpt.v2e.ccec = tools.create_dataset('ccec', ctmpfile, (ncore, ncore, nextern, ncore))
    nevpt.v2e.ccec[:] = get_v2e_df(nevpt, Lcc, Lec, 'ccec')
    tools.flush(ctmpfile)

    nevpt.h1eff.ca = compute_effective_1e(nevpt, nevpt.h1e[:ncore, ncore:nocc], nevpt.v2e.ccca, nevpt.v2e.ccac)
    nevpt.h1eff.ce = compute_effective_1e(nevpt, nevpt.h1e[:ncore, nocc:], nevpt.v2e.ccce, nevpt.v2e.ccec)
    nevpt.h1eff.aa = compute_effective_1e(nevpt, nevpt.h1e[ncore:nocc, ncore:nocc], nevpt.v2e.ccaa, nevpt.v2e.caac)
    nevpt.h1eff.ae = compute_effective_1e(nevpt, nevpt.h1e[ncore:nocc, nocc:], nevpt.v2e.ccae, nevpt.v2e.caec)

    if nfrozen > 0:
        nevpt.h1eff.ca = nevpt.h1eff.ca[nfrozen:,:].copy()
        nevpt.h1eff.ce = nevpt.h1eff.ce[nfrozen:,:].copy()
        nevpt.mo_energy.c = nevpt.mo_energy.c[nfrozen:]
        Lcc = Lcc[:, nfrozen:, nfrozen:].copy()
        Lca = Lca[:, nfrozen:, :].copy()
        nevpt.v2e.Lce = nevpt.v2e.Lce[:, nfrozen:, :].copy()
        Lec = Lec[:, :, nfrozen:].copy()

    # Other integrals
    nevpt.v2e.caca = tools.create_dataset('caca', ctmpfile, (ncore_wof, ncas, ncore_wof, ncas))
    nevpt.v2e.cace = tools.create_dataset('cace', ctmpfile, (ncore_wof, ncas, ncore_wof, nextern))

    nevpt.v2e.caae = tools.create_dataset('caae', ctmpfile, (ncore_wof, ncas, ncas, nextern))
    nevpt.v2e.ceaa = tools.create_dataset('ceaa', ctmpfile, (ncore_wof, nextern, ncas, ncas))

    nevpt.v2e.caaa = tools.create_dataset('caaa', ctmpfile, (ncore_wof, ncas, ncas, ncas))
    nevpt.v2e.aaae = tools.create_dataset('aaae', tmpfile, (ncas, ncas, ncas, nextern))

    nevpt.v2e.cece = tools.create_dataset('cece', ctmpfile, (ncore_wof, nextern, ncore_wof, nextern))
    nevpt.v2e.ceae = tools.create_dataset('ceae', ctmpfile, (ncore_wof, nextern, ncas, nextern))

    nevpt.v2e.cace[:] = get_v2e_df(nevpt, Lca, nevpt.v2e.Lce, 'cace')
    tools.flush(ctmpfile)

    nevpt.v2e.caca[:] = get_v2e_df(nevpt, Lca, Lca, 'caca')
    tools.flush(ctmpfile)

    nevpt.v2e.caae[:] = get_v2e_df(nevpt, Lca, nevpt.v2e.Lae, 'caae')
    tools.flush(ctmpfile)

    nevpt.v2e.ceaa[:] = get_v2e_df(nevpt, nevpt.v2e.Lce, Laa, 'ceaa')
    tools.flush(ctmpfile)

    nevpt.v2e.caaa[:] = get_v2e_df(nevpt, Lca, Laa, 'caaa')
    tools.flush(ctmpfile)

    nevpt.v2e.aaae[:] = get_v2e_df(nevpt, Laa, nevpt.v2e.Lae, 'aaae')
    tools.flush(tmpfile)

    chunks = tools.calculate_chunks(nevpt, ncore_wof, [ncore_wof, nextern, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        nevpt.log.debug("v2e.cece [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
        nevpt.v2e.cece[s_chunk:f_chunk] = get_ooee_df(nevpt, nevpt.v2e.Lce, nevpt.v2e.Lce, s_chunk, f_chunk)
        tools.flush(ctmpfile)

    chunks = tools.calculate_chunks(nevpt, ncore_wof, [ncas, nextern, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        nevpt.log.debug("v2e.ceae [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
        nevpt.v2e.ceae[s_chunk:f_chunk] = get_ooee_df(nevpt, nevpt.v2e.Lce, nevpt.v2e.Lae, s_chunk, f_chunk)
        tools.flush(ctmpfile)

    nevpt.v2e.caea = tools.create_dataset('caea', ctmpfile, (ncore_wof, ncas, nextern, ncas))

    nevpt.v2e.ccee = tools.create_dataset('ccee', ctmpfile, (ncore_wof, ncore_wof, nextern, nextern))
    nevpt.v2e.ceec = tools.create_dataset('ceec', ctmpfile, (ncore_wof, nextern, nextern, ncore_wof))

    nevpt.v2e.caee = tools.create_dataset('caee', ctmpfile, (ncore_wof, ncas, nextern, nextern))
    nevpt.v2e.ceea = tools.create_dataset('ceea', ctmpfile, (ncore_wof, nextern, nextern, ncas))

    nevpt.v2e.aeae = tools.create_dataset('aeae', tmpfile, (ncas, nextern, ncas, nextern))
    nevpt.v2e.aaee = tools.create_dataset('aaee', tmpfile, (ncas, ncas, nextern, nextern))
    nevpt.v2e.aeea = tools.create_dataset('aeea', tmpfile, (ncas, nextern, nextern, ncas))

    nevpt.v2e.caea[:] = get_v2e_df(nevpt, Lca, Lea, 'caea')
    tools.flush(ctmpfile)

    chunks = tools.calculate_chunks(nevpt, ncore_wof, [ncore_wof, nextern, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        nevpt.log.debug("v2e.ccee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
        nevpt.v2e.ccee[s_chunk:f_chunk] = get_ooee_df(nevpt, Lcc, nevpt.v2e.Lee, s_chunk, f_chunk)
        tools.flush(ctmpfile)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        nevpt.log.debug("v2e.ceec [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
        nevpt.v2e.ceec[s_chunk:f_chunk] = get_ooee_df(nevpt, nevpt.v2e.Lce, Lec, s_chunk, f_chunk)
        tools.flush(ctmpfile)

    chunks = tools.calculate_chunks(nevpt, ncore_wof, [ncas, nextern, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        nevpt.log.debug("v2e.caee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
        nevpt.v2e.caee[s_chunk:f_chunk] = get_ooee_df(nevpt, Lca, nevpt.v2e.Lee, s_chunk, f_chunk)
        tools.flush(ctmpfile)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        nevpt.log.debug("v2e.ceea [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
        nevpt.v2e.ceea[s_chunk:f_chunk] = get_ooee_df(nevpt, nevpt.v2e.Lce, Lea, s_chunk, f_chunk)
        tools.flush(ctmpfile)

    chunks = tools.calculate_chunks(nevpt, ncas, [ncas, nextern, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        nevpt.log.debug("v2e.aeae [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
        nevpt.v2e.aeae[s_chunk:f_chunk] = get_ooee_df(nevpt, nevpt.v2e.Lae, nevpt.v2e.Lae, s_chunk, f_chunk)
        tools.flush(tmpfile)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        nevpt.log.debug("v2e.aaee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
        nevpt.v2e.aaee[s_chunk:f_chunk] = get_ooee_df(nevpt, Laa, nevpt.v2e.Lee, s_chunk, f_chunk)
        tools.flush(tmpfile)

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        nevpt.log.debug("v2e.aeea [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
        nevpt.v2e.aeea[s_chunk:f_chunk] = get_ooee_df(nevpt, nevpt.v2e.Lae, Lea, s_chunk, f_chunk)
        tools.flush(tmpfile)

    nevpt.log.timer("transforming 2e integrals", *cput0)

def get_ooee_df(nevpt, Lpq, Lrs, s_chunk_p, f_chunk_p):

    cput0 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = nevpt.interface.einsum
    einsum_type = nevpt.interface.einsum_type

    # Variables from kernel
    naux = nevpt.naux

    chunk_p = f_chunk_p - s_chunk_p
    q = Lpq.shape[2]
    r = Lrs.shape[1]
    s = Lrs.shape[2]

    chunks_aux = tools.calculate_double_chunks(nevpt, naux, [chunk_p, q], [r, s],
                                                       extra_tensors=[[chunk_p, q, r, s],
                                                                      [chunk_p, q, r, s]])

    v_ooee = np.zeros((chunk_p, q, r, s))

    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks_aux):
        nevpt.log.debug("aux [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks_aux), s_chunk, f_chunk)
        cput1 = (logger.process_clock(), logger.perf_counter())

        Lpq_chunk = np.ascontiguousarray(Lpq[s_chunk:f_chunk,s_chunk_p:f_chunk_p])
        Lrs_chunk = Lrs[s_chunk:f_chunk]

        v_ooee += einsum('iab,icd->abcd', Lpq_chunk, Lrs_chunk, optimize = einsum_type)

        nevpt.log.timer_debug("contracting v_ooee DF", *cput1)
    del(Lpq_chunk, Lrs_chunk)

    nevpt.log.timer_debug("computing v_ooee DF", *cput0)
    return v_ooee

def get_v2e_df(nevpt, Lpq, Lrs, pqrs_string = None):

    cput0 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = nevpt.interface.einsum
    einsum_type = nevpt.interface.einsum_type

    v_pqrs = einsum('iab,icd->abcd', Lpq, Lrs, optimize = einsum_type)

    nevpt.log.timer_debug("computing v2e.{:} DF".format(pqrs_string), *cput0)
    return v_pqrs

