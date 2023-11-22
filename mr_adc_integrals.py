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

    h1eff  = np.ascontiguousarray(h1e_pq)
    h1eff += 2.0 * einsum('rrpq->pq', v2e_ccpq, optimize = einsum_type)
    h1eff -= einsum('rpqr->pq', v2e_cpqc, optimize = einsum_type)

    return h1eff

def transform_integrals_2e_incore(mr_adc):

    start_time = time.time()

    print("Transforming 2e integrals to MO basis (in-core)...")
    sys.stdout.flush()

    # Import Prism interface
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
        if mr_adc.method in ("mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
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

            mr_adc.v2e.ccee = transform_2e_chem_incore(interface, mo_c, mo_c, mo_e, mo_e)
            mr_adc.v2e.ceec = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_c)

            mr_adc.v2e.caea = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_a)
            mr_adc.v2e.caee = transform_2e_chem_incore(interface, mo_c, mo_a, mo_e, mo_e)

            mr_adc.v2e.ceea = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_a)

            mr_adc.v2e.aeae = transform_2e_chem_incore(interface, mo_a, mo_e, mo_a, mo_e)

            mr_adc.v2e.aaee = transform_2e_chem_incore(interface, mo_a, mo_a, mo_e, mo_e)

            mr_adc.v2e.aeea = transform_2e_chem_incore(interface, mo_a, mo_e, mo_e, mo_a)

        if mr_adc.method in ("mr-adc(2)-x") or (mr_adc.method in ("mr-adc(2)") and not mr_adc.approx_trans_moments):
            mr_adc.v2e.ceee = transform_2e_chem_incore(interface, mo_c, mo_e, mo_e, mo_e, compacted = True)
            mr_adc.v2e.aeee = transform_2e_chem_incore(interface, mo_a, mo_e, mo_e, mo_e, compacted = True)

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

    print("Time for transforming integrals:                   %f sec\n" % (time.time() - start_time))

def transform_integrals_2e_df(mr_adc):

    start_time = time.time()

    print("Transforming 2e integrals to MO basis (density-fitting)...")
    sys.stdout.flush()

    # Import Prism interface
    interface = mr_adc.interface

    with_df = interface.with_df
    naux = interface.naux

    # Variables from kernel
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc
    nextern = mr_adc.nextern

    nmo = mr_adc.nmo
    mo = mr_adc.mo
    mo_c = mo[:, :ncore].copy()
    mo_a = mo[:, ncore:nocc].copy()
    mo_e = mo[:, nocc:].copy()

    mr_adc.v2e.ceee = None
    mr_adc.v2e.aeee = None

    Lcc = np.empty((naux, ncore, ncore))
    Lca = np.empty((naux, ncore, ncas))
    Lac = np.empty((naux, ncas, ncore))
    Laa = np.empty((naux, ncas, ncas))

    Lec = np.empty((naux, nextern,  ncore))
    Lea = np.empty((naux, nextern, ncas))

    mr_adc.v2e.Lee = np.empty((naux, nextern, nextern))
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

    Lcc = Lcc.reshape(naux, ncore*ncore)
    Lca = Lca.reshape(naux, ncore*ncas)
    mr_adc.v2e.Lce = mr_adc.v2e.Lce.reshape(naux, ncore*nextern)

    Lac = Lac.reshape(naux, ncas*ncore)
    Laa = Laa.reshape(naux, ncas*ncas)
    mr_adc.v2e.Lae = mr_adc.v2e.Lae.reshape(naux, ncas*nextern)

    Lec = Lec.reshape(naux, nextern*ncore)
    Lea = Lea.reshape(naux, nextern*ncas)
    mr_adc.v2e.Lee = mr_adc.v2e.Lee.reshape(naux, nextern*nextern)

    mr_adc.v2e.feri1 = interface.create_HDF5_temp_file()
    mr_adc.v2e.aaaa = mr_adc.v2e.feri1.create_dataset('aaaa', (ncas, ncas, ncas, ncas), 'f8')
    mr_adc.v2e.aaaa[:] = transform_2e_chem_incore(interface, mo_a, mo_a, mo_a, mo_a)

    if mr_adc.method_type == "ip" or mr_adc.method_type == "ea" or mr_adc.method_type == "cvs-ip":
        if mr_adc.method in ("mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.ccaa = mr_adc.v2e.feri1.create_dataset('ccaa', (ncore, ncore, ncas, ncas), 'f8')
            mr_adc.v2e.ccaa[:] = transform_2e_chem_incore(interface, mo_c, mo_c, mo_a, mo_a)

            mr_adc.v2e.ccae = mr_adc.v2e.feri1.create_dataset('ccae', (ncore, ncore, ncas, nextern), 'f8')
            mr_adc.v2e.ccae[:] = np.dot(Lcc.T, mr_adc.v2e.Lae).reshape(ncore, ncore, ncas, nextern)

            mr_adc.v2e.caac = mr_adc.v2e.feri1.create_dataset('caac', (ncore, ncas, ncas, ncore), 'f8')
            mr_adc.v2e.caac[:] = transform_2e_chem_incore(interface, mo_c, mo_a, mo_a, mo_c)

            mr_adc.v2e.caec = mr_adc.v2e.feri1.create_dataset('caec', (ncore, ncas, nextern, ncore), 'f8')
            mr_adc.v2e.caec[:] = np.dot(Lca.T, Lec).reshape(ncore, ncas, nextern, ncore)

            mr_adc.v2e.caca = mr_adc.v2e.feri1.create_dataset('caca', (ncore, ncas, ncore, ncas), 'f8')
            mr_adc.v2e.caca[:] = np.dot(Lca.T, Lca).reshape(ncore, ncas, ncore, ncas)

            mr_adc.v2e.cece = mr_adc.v2e.feri1.create_dataset('cece', (ncore, nextern, ncore, nextern), 'f8')
            chnk_size = calculate_chunk_size_oee(mr_adc, ncore)
            for p in range(0, ncore, chnk_size):
                mr_adc.v2e.cece[p:p+chnk_size] = get_ooee_df(mr_adc, mr_adc.v2e.Lce, p, nextern,
                                                                     mr_adc.v2e.Lce, ncore, nextern, chnk_size)

            mr_adc.v2e.cace = mr_adc.v2e.feri1.create_dataset('cace', (ncore, ncas, ncore, nextern), 'f8')
            mr_adc.v2e.cace[:] = np.dot(Lca.T, mr_adc.v2e.Lce).reshape(ncore, ncas, ncore, nextern)

            mr_adc.v2e.caaa = mr_adc.v2e.feri1.create_dataset('caaa', (ncore, ncas, ncas, ncas), 'f8')
            mr_adc.v2e.caaa[:] = np.dot(Lca.T, Laa).reshape(ncore, ncas, ncas, ncas)

            mr_adc.v2e.ceae = mr_adc.v2e.feri1.create_dataset('ceae', (ncore, nextern, ncas, nextern), 'f8')
            chnk_size = calculate_chunk_size_oee(mr_adc, ncas)
            for p in range(0, ncore, chnk_size):
                mr_adc.v2e.ceae[p:p+chnk_size] = get_ooee_df(mr_adc, mr_adc.v2e.Lce, p, nextern,
                                                                     mr_adc.v2e.Lae, ncas, nextern, chnk_size)

            mr_adc.v2e.caae = mr_adc.v2e.feri1.create_dataset('caae', (ncore, ncas, ncas, nextern), 'f8')
            mr_adc.v2e.caae[:] = np.dot(Lca.T, mr_adc.v2e.Lae).reshape(ncore, ncas, ncas, nextern)

            mr_adc.v2e.ceaa = mr_adc.v2e.feri1.create_dataset('ceaa', (ncore, nextern, ncas, ncas), 'f8')
            mr_adc.v2e.ceaa[:] = np.dot(mr_adc.v2e.Lce.T, Laa).reshape(ncore, nextern, ncas, ncas)

            mr_adc.v2e.aaae = mr_adc.v2e.feri1.create_dataset('aaae', (ncas, ncas, ncas, nextern), 'f8')
            mr_adc.v2e.aaae[:] = np.dot(Laa.T, mr_adc.v2e.Lae).reshape(ncas, ncas, ncas, nextern)

        if mr_adc.method in ("mr-adc(2)-x"):
            mr_adc.v2e.cccc = mr_adc.v2e.feri1.create_dataset('cccc', (ncore, ncore, ncore, ncore), 'f8')
            mr_adc.v2e.cccc[:] = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_c)

            mr_adc.v2e.ccee = mr_adc.v2e.feri1.create_dataset('ccee', (ncore, ncore, nextern, nextern), 'f8')
            chnk_size = calculate_chunk_size_oee(mr_adc, ncore)
            for p in range(0, ncore, chnk_size):
                mr_adc.v2e.ccee[p:p+chnk_size] = get_ooee_df(mr_adc, Lcc, p, ncore,
                                                                     mr_adc.v2e.Lee, nextern, nextern, chnk_size)

            mr_adc.v2e.ceec = mr_adc.v2e.feri1.create_dataset('ceec', (ncore, nextern, nextern, ncore), 'f8')
            chnk_size = calculate_chunk_size_oee(mr_adc, ncore)
            for p in range(0, ncore, chnk_size):
                mr_adc.v2e.ceec[p:p+chnk_size] = get_ooee_df(mr_adc, mr_adc.v2e.Lce, p, nextern,
                                                                     Lec, nextern, ncore, chnk_size)

            mr_adc.v2e.caea = mr_adc.v2e.feri1.create_dataset('caea', (ncore, ncas, nextern, ncas), 'f8')
            mr_adc.v2e.caea[:] = np.dot(Lca.T, Lea).reshape(ncore, ncas, nextern, ncas)

            mr_adc.v2e.caee = mr_adc.v2e.feri1.create_dataset('caee', (ncore, ncas, nextern, nextern), 'f8')
            chnk_size = calculate_chunk_size_oee(mr_adc, ncas)
            for p in range(0, ncore, chnk_size):
                mr_adc.v2e.caee[p:p+chnk_size] = get_ooee_df(mr_adc, Lca, p, ncas,
                                                                     mr_adc.v2e.Lee, nextern, nextern, chnk_size)

            mr_adc.v2e.ceea = mr_adc.v2e.feri1.create_dataset('ceea', (ncore, nextern, nextern, ncas), 'f8')
            chnk_size = calculate_chunk_size_oee(mr_adc, ncas)
            for p in range(0, ncore, chnk_size):
                mr_adc.v2e.ceea[p:p+chnk_size] = get_ooee_df(mr_adc, mr_adc.v2e.Lce, p, nextern,
                                                                     Lea, nextern, ncas, chnk_size)

            mr_adc.v2e.aeae = mr_adc.v2e.feri1.create_dataset('aeae', (ncas, nextern, ncas, nextern), 'f8')
            chnk_size = calculate_chunk_size_oee(mr_adc, ncas)
            for p in range(0, ncas, chnk_size):
                mr_adc.v2e.aeae[p:p+chnk_size] = get_ooee_df(mr_adc, mr_adc.v2e.Lae, p, nextern,
                                                                     mr_adc.v2e.Lae, ncas, nextern, chnk_size)

            mr_adc.v2e.aaee = mr_adc.v2e.feri1.create_dataset('aaee', (ncas, ncas, nextern, nextern), 'f8')
            chnk_size = calculate_chunk_size_oee(mr_adc, ncas)
            for p in range(0, ncas, chnk_size):
                mr_adc.v2e.aaee[p:p+chnk_size] = get_ooee_df(mr_adc, Laa, p, ncas,
                                                                     mr_adc.v2e.Lee, nextern, nextern, chnk_size)

            mr_adc.v2e.aeea = mr_adc.v2e.feri1.create_dataset('aeea', (ncas, nextern, nextern, ncas), 'f8')
            chnk_size = calculate_chunk_size_oee(mr_adc, ncas)
            for p in range(0, ncas, chnk_size):
                mr_adc.v2e.aeea[p:p+chnk_size] = get_ooee_df(mr_adc, mr_adc.v2e.Lae, p, nextern,
                                                                     Lea, nextern, ncas, chnk_size)

    # Effective one-electron integrals
    mr_adc.v2e.ccca = mr_adc.v2e.feri1.create_dataset('ccca', (ncore, ncore, ncore, ncas), 'f8')
    mr_adc.v2e.ccca[:] = transform_2e_chem_incore(interface, mo_c, mo_c, mo_c, mo_a)

    mr_adc.v2e.ccce = mr_adc.v2e.feri1.create_dataset('ccce', (ncore, ncore, ncore, nextern), 'f8')
    mr_adc.v2e.ccce[:] = np.dot(Lcc.T, mr_adc.v2e.Lce).reshape(ncore, ncore, ncore, nextern)

    mr_adc.v2e.ccac = mr_adc.v2e.feri1.create_dataset('ccac', (ncore, ncore, ncas, ncore), 'f8')
    mr_adc.v2e.ccac[:] = np.dot(Lcc.T, Lac).reshape(ncore, ncore, ncas, ncore)

    mr_adc.v2e.ccec = mr_adc.v2e.feri1.create_dataset('ccec', (ncore, ncore, nextern, ncore), 'f8')
    mr_adc.v2e.ccec[:] = np.dot(Lcc.T, Lec).reshape(ncore, ncore, nextern, ncore)

    mr_adc.h1eff.ca = compute_effective_1e(mr_adc, mr_adc.h1e[:ncore, ncore:nocc], mr_adc.v2e.ccca, mr_adc.v2e.ccac)
    mr_adc.h1eff.ce = compute_effective_1e(mr_adc, mr_adc.h1e[:ncore, nocc:], mr_adc.v2e.ccce, mr_adc.v2e.ccec)
    mr_adc.h1eff.aa = compute_effective_1e(mr_adc, mr_adc.h1e[ncore:nocc, ncore:nocc], mr_adc.v2e.ccaa, mr_adc.v2e.caac)
    mr_adc.h1eff.ae = compute_effective_1e(mr_adc, mr_adc.h1e[ncore:nocc, nocc:], mr_adc.v2e.ccae, mr_adc.v2e.caec)

    # Store diagonal elements of the generalized Fock operator
    mr_adc.mo_energy.c = mr_adc.interface.mo_energy[:ncore]
    mr_adc.mo_energy.e = mr_adc.interface.mo_energy[nocc:]

    print("Time for transforming integrals:                   %f sec\n" % (time.time() - start_time))

def calculate_chunk_size_oeee(mr_adc):

    avail_mem = (mr_adc.max_memory - mr_adc.current_memory()[0]) * 0.5
    eee_mem = (mr_adc.nextern**3) * 8/1e6

    chunk_size = int(avail_mem / eee_mem)

    if chunk_size <= 0 :
        chunk_size = 1

    return chunk_size

def calculate_chunk_size_oee(mr_adc, nocc):

    avail_mem = (mr_adc.max_memory - mr_adc.current_memory()[0]) * 0.5
    ee_mem = (nocc * mr_adc.nextern**2) * 8/1e6

    chunk_size = int(avail_mem / ee_mem)

    if chunk_size <= 0 :
        chunk_size = 1

    return chunk_size

def get_oeee_df(mr_adc, Loe, Lee, p, chnk_size):

    # Import Prism interface
    interface = mr_adc.interface

    naux = interface.naux

    # Variables from kernel
    nextern = mr_adc.nextern

    Lee = Lee.reshape(naux, nextern*nextern)
    Loe = Loe.reshape(naux, -1, nextern)
    nocc = Loe.shape[1]

    if chnk_size < nocc:
        Loe_temp = np.ascontiguousarray(Loe.transpose(1,2,0)[p:p+chnk_size].reshape(-1, naux))
    else:
        Loe_temp = np.ascontiguousarray(Loe.transpose(1,2,0).reshape(-1, naux))

    oeee = np.dot(Loe_temp, Lee)
    oeee = oeee.reshape(-1, nextern, nextern, nextern)
    return oeee

def get_ooee_df(mr_adc, Lpq, p, q, Lrs, r, s, chnk_size):

    # Import Prism interface
    interface = mr_adc.interface
    naux = interface.naux

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    Lpq = Lpq.reshape(naux, -1, q)
    Lpq = np.ascontiguousarray(Lpq[:,p:p+chnk_size])
    Lpq = Lpq.reshape(naux, -1)

    ooee = einsum('ji,jk', Lpq, Lrs, optimize = einsum_type)
    ooee = ooee.reshape(-1, q, r, s)

    return ooee

def unpack_v2e_oeee(v2e_oeee, norb):

    n_ee = norb * (norb + 1) // 2
    ind_ee = np.tril_indices(norb)

    v2e_oeee_ = None

    if len(v2e_oeee.shape) == 3:
        if (v2e_oeee.shape[0] == n_ee):
            v2e_oeee_ = np.zeros((norb, norb, v2e_oeee.shape[1], v2e_oeee.shape[2]))
            v2e_oeee_[ind_ee[0], ind_ee[1]] = v2e_oeee
            v2e_oeee_[ind_ee[1], ind_ee[0]] = v2e_oeee

        elif (v2e_oeee.shape[2] == n_ee):
            v2e_oeee_ = np.zeros((v2e_oeee.shape[0], v2e_oeee.shape[1], norb, norb))
            v2e_oeee_[:, :, ind_ee[0], ind_ee[1]] = v2e_oeee
            v2e_oeee_[:, :, ind_ee[1], ind_ee[0]] = v2e_oeee
        else:
            raise TypeError("ERI dimensions don't match")

    else:
        raise RuntimeError("ERI does not have a correct dimension")

    return v2e_oeee_

def compute_cvs_integrals_2e_incore(mr_adc):

    start_time = time.time()

    print("Computing CVS integrals to MO basis (in-core)...")
    sys.stdout.flush()

    # Variables from kernel
    ncvs = mr_adc.ncvs

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
            mr_adc.v2e.vaxa = np.ascontiguousarray(mr_adc.v2e.caca[ncvs:, :, :ncvs, :])
            mr_adc.v2e.vava = np.ascontiguousarray(mr_adc.v2e.caca[ncvs:, :, ncvs:, :])
            del(mr_adc.v2e.caca)

            mr_adc.v2e.xexe = np.ascontiguousarray(mr_adc.v2e.cece[:ncvs, :, :ncvs, :])
            mr_adc.v2e.xeve = np.ascontiguousarray(mr_adc.v2e.cece[:ncvs, :, ncvs:, :])
            mr_adc.v2e.vexe = np.ascontiguousarray(mr_adc.v2e.cece[ncvs:, :, :ncvs, :])
            mr_adc.v2e.veve = np.ascontiguousarray(mr_adc.v2e.cece[ncvs:, :, ncvs:, :])
            del(mr_adc.v2e.cece)

            mr_adc.v2e.xaxe = np.ascontiguousarray(mr_adc.v2e.cace[:ncvs, :, :ncvs, :])
            mr_adc.v2e.xave = np.ascontiguousarray(mr_adc.v2e.cace[:ncvs, :, ncvs:, :])
            mr_adc.v2e.vaxe = np.ascontiguousarray(mr_adc.v2e.cace[ncvs:, :, :ncvs, :])
            mr_adc.v2e.vave = np.ascontiguousarray(mr_adc.v2e.cace[ncvs:, :, ncvs:, :])
            del(mr_adc.v2e.cace)

            mr_adc.v2e.xaaa = np.ascontiguousarray(mr_adc.v2e.caaa[:ncvs, :, :, :])
            mr_adc.v2e.vaaa = np.ascontiguousarray(mr_adc.v2e.caaa[ncvs:, :, :, :])
            del(mr_adc.v2e.caaa)

            mr_adc.v2e.xeae = np.ascontiguousarray(mr_adc.v2e.ceae[:ncvs, :, :, :])
            mr_adc.v2e.veae = np.ascontiguousarray(mr_adc.v2e.ceae[ncvs:, :, :, :])
            del(mr_adc.v2e.ceae)

            mr_adc.v2e.xaae = np.ascontiguousarray(mr_adc.v2e.caae[:ncvs, :, :, :])
            mr_adc.v2e.vaae = np.ascontiguousarray(mr_adc.v2e.caae[ncvs:, :, :, :])
            del(mr_adc.v2e.caae)

            mr_adc.v2e.xeaa = np.ascontiguousarray(mr_adc.v2e.ceaa[:ncvs, :, :, :])
            mr_adc.v2e.veaa = np.ascontiguousarray(mr_adc.v2e.ceaa[ncvs:, :, :, :])
            del(mr_adc.v2e.ceaa)

        if mr_adc.method in ("mr-adc(2)-x"):
            mr_adc.v2e.xxxx = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, :ncvs, :ncvs, :ncvs])
            mr_adc.v2e.xxvv = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, :ncvs, ncvs:, ncvs:])
            mr_adc.v2e.xvvx = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, ncvs:, ncvs:, :ncvs])
            mr_adc.v2e.xxvx = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, :ncvs, ncvs:, :ncvs])
            mr_adc.v2e.xxxv = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, :ncvs, :ncvs, ncvs:])
            mr_adc.v2e.xvxx = np.ascontiguousarray(mr_adc.v2e.cccc[:ncvs, ncvs:, :ncvs, :ncvs])
            del(mr_adc.v2e.cccc)

            mr_adc.v2e.xxee = np.ascontiguousarray(mr_adc.v2e.ccee[:ncvs, :ncvs, :, :])
            mr_adc.v2e.xvee = np.ascontiguousarray(mr_adc.v2e.ccee[:ncvs, ncvs:, :, :])
            mr_adc.v2e.vxee = np.ascontiguousarray(mr_adc.v2e.ccee[ncvs:, :ncvs, :, :])
            mr_adc.v2e.vvee = np.ascontiguousarray(mr_adc.v2e.ccee[ncvs:, ncvs:, :, :])
            del(mr_adc.v2e.ccee)

            mr_adc.v2e.xeex = np.ascontiguousarray(mr_adc.v2e.ceec[:ncvs, :, :, :ncvs])
            mr_adc.v2e.xeev = np.ascontiguousarray(mr_adc.v2e.ceec[:ncvs, :, :, ncvs:])
            mr_adc.v2e.veex = np.ascontiguousarray(mr_adc.v2e.ceec[ncvs:, :, :, :ncvs])
            mr_adc.v2e.veev = np.ascontiguousarray(mr_adc.v2e.ceec[ncvs:, :, :, ncvs:])
            del(mr_adc.v2e.ceec)

            mr_adc.v2e.xaea = np.ascontiguousarray(mr_adc.v2e.caea[:ncvs, :, :, :])
            mr_adc.v2e.vaea = np.ascontiguousarray(mr_adc.v2e.caea[ncvs:, :, :, :])
            del(mr_adc.v2e.caea)

            mr_adc.v2e.xaee = np.ascontiguousarray(mr_adc.v2e.caee[:ncvs, :, :, :])
            mr_adc.v2e.vaee = np.ascontiguousarray(mr_adc.v2e.caee[ncvs:, :, :, :])
            del(mr_adc.v2e.caee)

            mr_adc.v2e.xeea = np.ascontiguousarray(mr_adc.v2e.ceea[:ncvs, :, :, :])
            mr_adc.v2e.veea = np.ascontiguousarray(mr_adc.v2e.ceea[ncvs:, :, :, :])
            del(mr_adc.v2e.ceea)

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

    print("Time for computing CVS integrals:                  %f sec\n" % (time.time() - start_time))

def compute_cvs_integrals_2e_df(mr_adc):

    start_time = time.time()

    print("Computing CVS integrals to MO basis (density-fitting)...")
    sys.stdout.flush()

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    # Compute CVS integrals
    if mr_adc.method_type == "cvs-ip":
        if mr_adc.method in ("mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.xxaa = mr_adc.v2e.feri1.create_dataset('xxaa', (ncvs, ncvs, ncas, ncas), 'f8')
            mr_adc.v2e.xvaa = mr_adc.v2e.feri1.create_dataset('xvaa', (ncvs, nval, ncas, ncas), 'f8')
            mr_adc.v2e.vxaa = mr_adc.v2e.feri1.create_dataset('vxaa', (nval, ncvs, ncas, ncas), 'f8')
            mr_adc.v2e.vvaa = mr_adc.v2e.feri1.create_dataset('vvaa', (nval, nval, ncas, ncas), 'f8')

            mr_adc.v2e.xxae = mr_adc.v2e.feri1.create_dataset('xxae', (ncvs, ncvs, ncas, nextern), 'f8')
            mr_adc.v2e.xvae = mr_adc.v2e.feri1.create_dataset('xvae', (ncvs, nval, ncas, nextern), 'f8')
            mr_adc.v2e.vxae = mr_adc.v2e.feri1.create_dataset('vxae', (nval, ncvs, ncas, nextern), 'f8')
            mr_adc.v2e.vvae = mr_adc.v2e.feri1.create_dataset('vvae', (nval, nval, ncas, nextern), 'f8')

            mr_adc.v2e.xaax = mr_adc.v2e.feri1.create_dataset('xaax', (ncvs, ncas, ncas, ncvs), 'f8')
            mr_adc.v2e.xaav = mr_adc.v2e.feri1.create_dataset('xaav', (ncvs, ncas, ncas, nval), 'f8')
            mr_adc.v2e.vaax = mr_adc.v2e.feri1.create_dataset('vaax', (nval, ncas, ncas, ncvs), 'f8')
            mr_adc.v2e.vaav = mr_adc.v2e.feri1.create_dataset('vaav', (nval, ncas, ncas, nval), 'f8')

            mr_adc.v2e.xaex = mr_adc.v2e.feri1.create_dataset('xaex', (ncvs, ncas, nextern, ncvs), 'f8')
            mr_adc.v2e.xaev = mr_adc.v2e.feri1.create_dataset('xaev', (ncvs, ncas, nextern, nval), 'f8')
            mr_adc.v2e.vaex = mr_adc.v2e.feri1.create_dataset('vaex', (nval, ncas, nextern, ncvs), 'f8')
            mr_adc.v2e.vaev = mr_adc.v2e.feri1.create_dataset('vaev', (nval, ncas, nextern, nval), 'f8')

            mr_adc.v2e.xaxa = mr_adc.v2e.feri1.create_dataset('xaxa', (ncvs, ncas, ncvs, ncas), 'f8')
            mr_adc.v2e.xava = mr_adc.v2e.feri1.create_dataset('xava', (ncvs, ncas, nval, ncas), 'f8')
            mr_adc.v2e.vaxa = mr_adc.v2e.feri1.create_dataset('vaxa', (nval, ncas, ncvs, ncas), 'f8')
            mr_adc.v2e.vava = mr_adc.v2e.feri1.create_dataset('vava', (nval, ncas, nval, ncas), 'f8')

            mr_adc.v2e.xexe = mr_adc.v2e.feri1.create_dataset('xexe', (ncvs, nextern, ncvs, nextern), 'f8')
            mr_adc.v2e.xeve = mr_adc.v2e.feri1.create_dataset('xeve', (ncvs, nextern, nval, nextern), 'f8')
            mr_adc.v2e.vexe = mr_adc.v2e.feri1.create_dataset('vexe', (nval, nextern, ncvs, nextern), 'f8')
            mr_adc.v2e.veve = mr_adc.v2e.feri1.create_dataset('veve', (nval, nextern, nval, nextern), 'f8')

            mr_adc.v2e.xaxe = mr_adc.v2e.feri1.create_dataset('xaxe', (ncvs, ncas, ncvs, nextern), 'f8')
            mr_adc.v2e.xave = mr_adc.v2e.feri1.create_dataset('xave', (ncvs, ncas, nval, nextern), 'f8')
            mr_adc.v2e.vaxe = mr_adc.v2e.feri1.create_dataset('vaxe', (nval, ncas, ncvs, nextern), 'f8')
            mr_adc.v2e.vave = mr_adc.v2e.feri1.create_dataset('vave', (nval, ncas, nval, nextern), 'f8')

            mr_adc.v2e.xaaa = mr_adc.v2e.feri1.create_dataset('xaaa', (ncvs, ncas, ncas, ncas), 'f8')
            mr_adc.v2e.vaaa = mr_adc.v2e.feri1.create_dataset('vaaa', (nval, ncas, ncas, ncas), 'f8')

            mr_adc.v2e.xeae = mr_adc.v2e.feri1.create_dataset('xeae', (ncvs, nextern, ncas, nextern), 'f8')
            mr_adc.v2e.veae = mr_adc.v2e.feri1.create_dataset('veae', (nval, nextern, ncas, nextern), 'f8')

            mr_adc.v2e.xaae = mr_adc.v2e.feri1.create_dataset('xaae', (ncvs, ncas, ncas, nextern), 'f8')
            mr_adc.v2e.vaae = mr_adc.v2e.feri1.create_dataset('vaae', (nval, ncas, ncas, nextern), 'f8')

            mr_adc.v2e.xeaa = mr_adc.v2e.feri1.create_dataset('xeaa', (ncvs, nextern, ncas, ncas), 'f8')
            mr_adc.v2e.veaa = mr_adc.v2e.feri1.create_dataset('veaa', (nval, nextern, ncas, ncas), 'f8')

            mr_adc.v2e.xxaa[:] = mr_adc.v2e.ccaa[:ncvs, :ncvs, :, :]
            mr_adc.v2e.xvaa[:] = mr_adc.v2e.ccaa[:ncvs, ncvs:, :, :]
            mr_adc.v2e.vxaa[:] = mr_adc.v2e.ccaa[ncvs:, :ncvs, :, :]
            mr_adc.v2e.vvaa[:] = mr_adc.v2e.ccaa[ncvs:, ncvs:, :, :]

            mr_adc.v2e.xxae[:] = mr_adc.v2e.ccae[:ncvs, :ncvs, :, :]
            mr_adc.v2e.xvae[:] = mr_adc.v2e.ccae[:ncvs, ncvs:, :, :]
            mr_adc.v2e.vxae[:] = mr_adc.v2e.ccae[ncvs:, :ncvs, :, :]
            mr_adc.v2e.vvae[:] = mr_adc.v2e.ccae[ncvs:, ncvs:, :, :]

            mr_adc.v2e.xaax[:] = mr_adc.v2e.caac[:ncvs, :, :, :ncvs]
            mr_adc.v2e.xaav[:] = mr_adc.v2e.caac[:ncvs, :, :, ncvs:]
            mr_adc.v2e.vaax[:] = mr_adc.v2e.caac[ncvs:, :, :, :ncvs]
            mr_adc.v2e.vaav[:] = mr_adc.v2e.caac[ncvs:, :, :, ncvs:]

            mr_adc.v2e.xaex[:] = mr_adc.v2e.caec[:ncvs, :, :, :ncvs]
            mr_adc.v2e.xaev[:] = mr_adc.v2e.caec[:ncvs, :, :, ncvs:]
            mr_adc.v2e.vaex[:] = mr_adc.v2e.caec[ncvs:, :, :, :ncvs]
            mr_adc.v2e.vaev[:] = mr_adc.v2e.caec[ncvs:, :, :, ncvs:]

            mr_adc.v2e.xaxa[:] = mr_adc.v2e.caca[:ncvs, :, :ncvs, :]
            mr_adc.v2e.xava[:] = mr_adc.v2e.caca[:ncvs, :, ncvs:, :]
            mr_adc.v2e.vaxa[:] = mr_adc.v2e.caca[ncvs:, :, :ncvs, :]
            mr_adc.v2e.vava[:] = mr_adc.v2e.caca[ncvs:, :, ncvs:, :]

            mr_adc.v2e.xexe[:] = mr_adc.v2e.cece[:ncvs, :, :ncvs, :]
            mr_adc.v2e.xeve[:] = mr_adc.v2e.cece[:ncvs, :, ncvs:, :]
            mr_adc.v2e.vexe[:] = mr_adc.v2e.cece[ncvs:, :, :ncvs, :]
            mr_adc.v2e.veve[:] = mr_adc.v2e.cece[ncvs:, :, ncvs:, :]

            mr_adc.v2e.xaxe[:] = mr_adc.v2e.cace[:ncvs, :, :ncvs, :]
            mr_adc.v2e.xave[:] = mr_adc.v2e.cace[:ncvs, :, ncvs:, :]
            mr_adc.v2e.vaxe[:] = mr_adc.v2e.cace[ncvs:, :, :ncvs, :]
            mr_adc.v2e.vave[:] = mr_adc.v2e.cace[ncvs:, :, ncvs:, :]

            mr_adc.v2e.xaaa[:] = mr_adc.v2e.caaa[:ncvs, :, :, :]
            mr_adc.v2e.vaaa[:] = mr_adc.v2e.caaa[ncvs:, :, :, :]

            mr_adc.v2e.xeae[:] = mr_adc.v2e.ceae[:ncvs, :, :, :]
            mr_adc.v2e.veae[:] = mr_adc.v2e.ceae[ncvs:, :, :, :]

            mr_adc.v2e.xaae[:] = mr_adc.v2e.caae[:ncvs, :, :, :]
            mr_adc.v2e.vaae[:] = mr_adc.v2e.caae[ncvs:, :, :, :]

            mr_adc.v2e.xeaa[:] = mr_adc.v2e.ceaa[:ncvs, :, :, :]
            mr_adc.v2e.veaa[:] = mr_adc.v2e.ceaa[ncvs:, :, :, :]

        if mr_adc.method in ("mr-adc(2)-x"):
            mr_adc.v2e.xxxx = mr_adc.v2e.feri1.create_dataset('xxxx', (ncvs, ncvs, ncvs, ncvs), 'f8')
            mr_adc.v2e.xxvv = mr_adc.v2e.feri1.create_dataset('xxvv', (ncvs, ncvs, nval, nval), 'f8')
            mr_adc.v2e.xvvx = mr_adc.v2e.feri1.create_dataset('xvvx', (ncvs, nval, nval, ncvs), 'f8')
            mr_adc.v2e.xxvx = mr_adc.v2e.feri1.create_dataset('xxvx', (ncvs, ncvs, nval, ncvs), 'f8')
            mr_adc.v2e.xxxv = mr_adc.v2e.feri1.create_dataset('xxxv', (ncvs, ncvs, ncvs, nval), 'f8')
            mr_adc.v2e.xvxx = mr_adc.v2e.feri1.create_dataset('xvxx', (ncvs, nval, ncvs, ncvs), 'f8')

            mr_adc.v2e.xxee = mr_adc.v2e.feri1.create_dataset('xxee', (ncvs, ncvs, nextern, nextern), 'f8')
            mr_adc.v2e.xvee = mr_adc.v2e.feri1.create_dataset('xvee', (ncvs, nval, nextern, nextern), 'f8')
            mr_adc.v2e.vxee = mr_adc.v2e.feri1.create_dataset('vxee', (nval, ncvs, nextern, nextern), 'f8')
            mr_adc.v2e.vvee = mr_adc.v2e.feri1.create_dataset('vvee', (nval, nval, nextern, nextern), 'f8')

            mr_adc.v2e.xeex = mr_adc.v2e.feri1.create_dataset('xeex', (ncvs, nextern, nextern, ncvs), 'f8')
            mr_adc.v2e.xeev = mr_adc.v2e.feri1.create_dataset('xeev', (ncvs, nextern, nextern, nval), 'f8')
            mr_adc.v2e.veex = mr_adc.v2e.feri1.create_dataset('veex', (nval, nextern, nextern, ncvs), 'f8')
            mr_adc.v2e.veev = mr_adc.v2e.feri1.create_dataset('veev', (nval, nextern, nextern, nval), 'f8')

            mr_adc.v2e.xaea = mr_adc.v2e.feri1.create_dataset('xaea', (ncvs, ncas, nextern, ncas), 'f8')
            mr_adc.v2e.vaea = mr_adc.v2e.feri1.create_dataset('vaea', (nval, ncas, nextern, ncas), 'f8')

            mr_adc.v2e.xaee = mr_adc.v2e.feri1.create_dataset('xaee', (ncvs, ncas, nextern, nextern), 'f8')
            mr_adc.v2e.vaee = mr_adc.v2e.feri1.create_dataset('vaee', (nval, ncas, nextern, nextern), 'f8')

            mr_adc.v2e.xeea = mr_adc.v2e.feri1.create_dataset('xeea', (ncvs, nextern, nextern, ncas), 'f8')
            mr_adc.v2e.veea = mr_adc.v2e.feri1.create_dataset('veea', (nval, nextern, nextern, ncas), 'f8')

            mr_adc.v2e.xxxx[:] = mr_adc.v2e.cccc[:ncvs, :ncvs, :ncvs, :ncvs]
            mr_adc.v2e.xxvv[:] = mr_adc.v2e.cccc[:ncvs, :ncvs, ncvs:, ncvs:]
            mr_adc.v2e.xvvx[:] = mr_adc.v2e.cccc[:ncvs, ncvs:, ncvs:, :ncvs]
            mr_adc.v2e.xxvx[:] = mr_adc.v2e.cccc[:ncvs, :ncvs, ncvs:, :ncvs]
            mr_adc.v2e.xxxv[:] = mr_adc.v2e.cccc[:ncvs, :ncvs, :ncvs, ncvs:]
            mr_adc.v2e.xvxx[:] = mr_adc.v2e.cccc[:ncvs, ncvs:, :ncvs, :ncvs]

            mr_adc.v2e.xxee[:] = mr_adc.v2e.ccee[:ncvs, :ncvs, :, :]
            mr_adc.v2e.xvee[:] = mr_adc.v2e.ccee[:ncvs, ncvs:, :, :]
            mr_adc.v2e.vxee[:] = mr_adc.v2e.ccee[ncvs:, :ncvs, :, :]
            mr_adc.v2e.vvee[:] = mr_adc.v2e.ccee[ncvs:, ncvs:, :, :]

            mr_adc.v2e.xeex[:] = mr_adc.v2e.ceec[:ncvs, :, :, :ncvs]
            mr_adc.v2e.xeev[:] = mr_adc.v2e.ceec[:ncvs, :, :, ncvs:]
            mr_adc.v2e.veex[:] = mr_adc.v2e.ceec[ncvs:, :, :, :ncvs]
            mr_adc.v2e.veev[:] = mr_adc.v2e.ceec[ncvs:, :, :, ncvs:]

            mr_adc.v2e.xaea[:] = mr_adc.v2e.caea[:ncvs, :, :, :]
            mr_adc.v2e.vaea[:] = mr_adc.v2e.caea[ncvs:, :, :, :]

            mr_adc.v2e.xaee[:] = mr_adc.v2e.caee[:ncvs, :, :, :]
            mr_adc.v2e.vaee[:] = mr_adc.v2e.caee[ncvs:, :, :, :]

            mr_adc.v2e.xeea[:] = mr_adc.v2e.ceea[:ncvs, :, :, :]
            mr_adc.v2e.veea[:] = mr_adc.v2e.ceea[ncvs:, :, :, :]

    # Effective one-electron integrals
    mr_adc.v2e.xxxa = mr_adc.v2e.feri1.create_dataset('xxxa', (ncvs, ncvs, ncvs, ncas), 'f8')
    mr_adc.v2e.xxva = mr_adc.v2e.feri1.create_dataset('xxva', (ncvs, ncvs, nval, ncas), 'f8')
    mr_adc.v2e.vxxa = mr_adc.v2e.feri1.create_dataset('vxxa', (nval, ncvs, ncvs, ncas), 'f8')

    mr_adc.v2e.xxxe = mr_adc.v2e.feri1.create_dataset('xxxe', (ncvs, ncvs, ncvs, nextern), 'f8')
    mr_adc.v2e.xxve = mr_adc.v2e.feri1.create_dataset('xxve', (ncvs, ncvs, nval, nextern), 'f8')
    mr_adc.v2e.vxxe = mr_adc.v2e.feri1.create_dataset('vxxe', (nval, ncvs, ncvs, nextern), 'f8')

    mr_adc.v2e.xxxa[:] = mr_adc.v2e.ccca[:ncvs, :ncvs, :ncvs, :]
    mr_adc.v2e.xxva[:] = mr_adc.v2e.ccca[:ncvs, :ncvs, ncvs:, :]
    mr_adc.v2e.vxxa[:] = mr_adc.v2e.ccca[ncvs:, :ncvs, :ncvs, :]

    mr_adc.v2e.xxxe[:] = mr_adc.v2e.ccce[:ncvs, :ncvs, :ncvs, :]
    mr_adc.v2e.xxve[:] = mr_adc.v2e.ccce[:ncvs, :ncvs, ncvs:, :]
    mr_adc.v2e.vxxe[:] = mr_adc.v2e.ccce[ncvs:, :ncvs, :ncvs, :]

    mr_adc.h1eff.xa = np.ascontiguousarray(mr_adc.h1eff.ca[:ncvs,:])
    mr_adc.h1eff.va = np.ascontiguousarray(mr_adc.h1eff.ca[ncvs:,:])
    del(mr_adc.h1eff.ca)

    mr_adc.h1eff.xe = np.ascontiguousarray(mr_adc.h1eff.ce[:ncvs,:])
    mr_adc.h1eff.ve = np.ascontiguousarray(mr_adc.h1eff.ce[ncvs:,:])
    del(mr_adc.h1eff.ce)

    # Remove in-core v2e integrals
    del(mr_adc.v2e.Lce, mr_adc.v2e.Lae, mr_adc.v2e.Lee)

    # Store diagonal elements of the generalized Fock operator
    mr_adc.mo_energy.x = mr_adc.mo_energy.c[:ncvs]
    mr_adc.mo_energy.v = mr_adc.mo_energy.c[ncvs:]

    print("Time for computing CVS integrals:                  %f sec\n" % (time.time() - start_time))
