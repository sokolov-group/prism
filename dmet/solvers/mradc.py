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
# Authors: Bryce Pickett <pickettosu@gmail.com>
#

import sys
from contextlib import nullcontext

import numpy as np
from pyscf import ao2mo, gto, scf, mcscf

import prism.lib.logger as logger
from prism.dmet.utils import silent_stdout
from prism.dmet.cas_selectors import (natorb_active_space, fix_cas_spin,
                                      stabilize_scf, set_reference_no_scf)


def solve(fock, tei, norb, nel, nimp, dm_guess_rhf,
          ncas, nelecas,
          chempot_imp=0.0, verbose=logger.INFO,
          prism_backend='opt_einsum',
          select_reference=None,
          casscf_kwargs=None,
          mradc_kwargs=None,
          spin=None,
          cas_select='energy',
          embed_level_shift=0.0, scf_stability=False,
          cas_spin=None, cas_spin_shift=0.2,
          natorb_occ_thresh=0.02, natorb_max_superset=None,
          deg_tol=1e-3, casci_conv_tol=1e-10, embedding_data=None,
          embedded_ref=None, no_kernel=False):
    import prism.interface
    import prism.mr_adc

    if cas_select not in ('energy', 'natorb'):
        raise ValueError(
            f"Invalid cas_select='{cas_select}'. Valid: ['energy', 'natorb']")

    casscf_kwargs = casscf_kwargs or {}
    mradc_kwargs = mradc_kwargs or {}

    log = logger.Logger(sys.stdout, verbose)
    printoutput = verbose >= logger.INFO

    _spin = spin if spin is not None else (nel % 2)
    _use_rohf = (_spin != 0)

    fock_copy = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy[orb, orb] -= chempot_imp

    ctx = silent_stdout() if not printoutput else nullcontext()

    with ctx:
        # Container for the embedded Hamiltonian, supplied through get_hcore/get_ovlp/_eri.
        mol = gto.Mole()
        mol.build(verbose=0)
        mol.nelectron = nel
        mol.spin = _spin
        mol.incore_anyway = True

        mf = scf.ROHF(mol) if _use_rohf else scf.RHF(mol)
        mf.get_hcore = lambda *args: fock_copy
        mf.get_ovlp = lambda *args: np.eye(norb)
        mf._eri = ao2mo.restore(8, tei, norb)
        mf.verbose = 4 if printoutput else 0
        # Level shift opens the near-degenerate gap so the embedded SCF is deterministic.
        mf.level_shift = embed_level_shift
        if no_kernel:
            set_reference_no_scf(
                mf, embedded_ref if embedded_ref is not None else dm_guess_rhf,
                nel, _spin, log=log)
        else:
            mf.scf(dm_guess_rhf)
            if not mf.converged:
                mf.max_cycle = 300
                mf.diis_space = 12
                mf.scf(mf.make_rdm1())
            if embed_level_shift != 0.0:
                # The shifted fixed point should be stationary for the real H;
                # motion indicates a masked instability.
                e_shifted = mf.e_tot
                mf.level_shift = 0.0
                mf.scf(mf.make_rdm1())
                log.info("level-shift verification: E(shift=%s)=%.10f  "
                         "E(shift removed, reconverged)=%.10f  dE=%.2e Ha"
                         % (embed_level_shift, e_shifted, mf.e_tot, abs(mf.e_tot - e_shifted)))
            if scf_stability:
                stabilize_scf(mf, log=log)
        if _use_rohf:
            log.info("embedded ROHF (spin=%d, nel=%d, norb=%d)"
                     % (_spin, nel, norb))

        mo_natorb = None
        if cas_select == 'natorb':
            mo_natorb, ncas, nelecas = natorb_active_space(
                mf, ncas, occ_thresh=natorb_occ_thresh, max_superset=natorb_max_superset,
                sa_nstates=1, cas_spin=cas_spin, cas_spin_shift=cas_spin_shift,
                deg_tol=deg_tol, conv_tol=casci_conv_tol, log=log)

        # MR-ADC takes a casscf or casci reference, so the embedded CASSCF is single-state.
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.verbose = 5 if printoutput else 0
        for key, val in casscf_kwargs.items():
            setattr(mc, key, val)
        if cas_spin is not None:
            fix_cas_spin(mc.fcisolver, cas_spin, cas_spin_shift)

        if mo_natorb is not None:
            mc.mo_coeff = mo_natorb
            log.info("CAS selection by %s, CAS(%d,%d)"
                     % (cas_select, nelecas, ncas))

        mc.kernel()
        log.info("\nembedded CASSCF energy: %.10f Ha" % mc.e_tot)

        interface = prism.interface.PYSCF(
            mf, mc,
            backend=prism_backend,
            select_reference=select_reference,
        )
        # Molecule and orbitals for Dyson orbital output.
        if embedding_data is not None:
            interface.emb_mol = embedding_data['mol']
            interface.emb_ao2emb = embedding_data['ao2emb']
            interface.emb_core_dm_ao = embedding_data['core_dm_ao']

        mradc_obj = prism.mr_adc.MRADC(interface)
        mradc_obj.verbose = 4 if printoutput else 0
        for key, val in mradc_kwargs.items():
            setattr(mradc_obj, key, val)

        e_exc, spec_factors, x = mradc_obj.kernel()
        e_exc = np.atleast_1d(e_exc)

        log.info("\nMR-ADC %s energies:" % mradc_obj.method_type.upper())
        for i, e in enumerate(e_exc):
            log.info("  Root %d: %.6f eV" % (i, e))

    return mc, mradc_obj, e_exc, spec_factors


def execute(task):
    mc, mradc_obj, e_exc, spec_factors = solve(
        task['dmet_fock'],
        task['dmet_tei'],
        task['norb'],
        task['nel'],
        task['nimp'],
        task.get('dm_guess_rhf'),
        ncas=task.get('ncas'),
        nelecas=task.get('nelecas'),
        chempot_imp=task.get('chempot_imp', 0.0),
        verbose=task.get('verbose', logger.INFO),
        casscf_kwargs=task.get('casscf_kwargs', {}),
        mradc_kwargs=task.get('mradc_kwargs', {}),
        spin=task.get('spin'),
        cas_select=task.get('cas_select', 'energy'),
        embed_level_shift=task.get('embed_level_shift', 0.0),
        scf_stability=task.get('scf_stability', False),
        cas_spin=task.get('cas_spin'),
        cas_spin_shift=task.get('cas_spin_shift', 0.2),
        natorb_occ_thresh=task.get('natorb_occ_thresh', 0.02),
        natorb_max_superset=task.get('natorb_max_superset'),
        deg_tol=task.get('deg_tol', 1e-3),
        casci_conv_tol=task.get('casci_conv_tol', 1e-10),
        embedding_data=task.get('embedding_data'),
        embedded_ref=task.get('embedded_ref'),
        no_kernel=task.get('no_kernel', False),
    )

    rdm1 = mc.make_rdm1()
    mradc_res = {
        'e_exc': e_exc,   # eV, as returned by the Prism MR-ADC kernel
        'spec_factors': spec_factors,
        'e_cas': mc.e_tot,
        'mc': mc,
        'mradc': mradc_obj,
    }
    return mc.e_tot, rdm1, mradc_res
