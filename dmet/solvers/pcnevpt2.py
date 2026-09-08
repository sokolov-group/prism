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
from prism.dmet.utils import silent_stdout, auto_nfrozen
from prism.dmet.cas_selectors import (natorb_active_space, fix_cas_spin,
                                      stabilize_scf, set_reference_no_scf)

_eV = 27.21138602


def solve(fock, tei, norb, nel, nimp, dm_guess_rhf,
          ncas, nelecas,
          sa_nstates=1, sa_weights=None,
          chempot_imp=0.0, verbose=logger.INFO,
          prism_backend='opt_einsum',
          nfrozen=None, nfrozen_cutoff=-2.0,
          compute_singles=False,
          s_thresh_singles=1e-8,
          s_thresh_doubles=1e-8,
          select_reference=None,
          casscf_kwargs=None,
          nevpt_kwargs=None,
          spin=None,
          cas_select='energy',
          embed_level_shift=0.0, scf_stability=False,
          cas_spin=None, cas_spin_shift=0.2,
          natorb_occ_thresh=0.02, natorb_max_superset=None,
          deg_tol=1e-3, casci_conv_tol=1e-10, dip_mom_ao=None, embedding_data=None,
          embedded_ref=None, no_kernel=False):
    import prism.interface
    import prism.nevpt

    if cas_select not in ('energy', 'natorb'):
        raise ValueError(
            f"Invalid cas_select='{cas_select}'. Valid: ['energy', 'natorb']")

    if sa_nstates > 1:
        if sa_weights is None:
            sa_weights = [1.0 / sa_nstates] * sa_nstates
        sa_weights = np.array(sa_weights, dtype=float)
        sa_weights /= sa_weights.sum()
    else:
        sa_weights = np.array([1.0])

    casscf_kwargs = casscf_kwargs or {}
    nevpt_kwargs = nevpt_kwargs  or {}

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
                sa_nstates=sa_nstates, cas_spin=cas_spin, cas_spin_shift=cas_spin_shift,
                deg_tol=deg_tol, conv_tol=casci_conv_tol, log=log)

        mc = mcscf.CASSCF(mf, ncas, nelecas)
        if sa_nstates > 1:
            mc = mcscf.state_average_(mc, weights=sa_weights.tolist())
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

        if sa_nstates > 1:
            log.info("\nembedded SA-CASSCF (%d states, ncas=%d, nelecas=%d)"
                     % (sa_nstates, ncas, nelecas))
            for i, e in enumerate(mc.e_states):
                log.info("  State %d: %.10f Ha  (weight=%.4f)" % (i, e, sa_weights[i]))
            log.info("  SA-weighted e_tot: %.10f Ha" % mc.e_tot)
        else:
            log.info("\nembedded CASSCF energy: %.10f Ha" % mc.e_tot)

        interface = prism.interface.PYSCF(
            mf, mc,
            backend=prism_backend,
            select_reference=select_reference,
        )
        # Dipole integrals over the molecule, supplied by the driver.
        if dip_mom_ao is not None:
            interface.dip_mom_ao = dip_mom_ao
        # Molecule and orbitals for spin-orbit integrals and NTO output.
        if embedding_data is not None:
            interface.emb_mol = embedding_data['mol']
            interface.emb_ao2emb = embedding_data['ao2emb']
            interface.emb_core_dm_ao = embedding_data['core_dm_ao']

        # prism.nevpt.NEVPT ('ss') is FIC-NEVPT2, equivalent to PC-NEVPT2.
        nevpt_obj = prism.nevpt.NEVPT(interface)
        nevpt_obj.compute_singles_amplitudes = compute_singles
        nevpt_obj.s_thresh_singles = s_thresh_singles
        nevpt_obj.s_thresh_doubles = s_thresh_doubles
        if dip_mom_ao is None:
            _n = sa_nstates
            def _skip_osc():
                nevpt_obj.properties["osc_strengths"] = np.zeros(_n - 1) if _n > 1 else None
            nevpt_obj.compute_properties = _skip_osc
        if nfrozen == 'auto':
            nfrozen = auto_nfrozen(mf, nfrozen_cutoff)
            log.info("nfrozen='auto': %d embedded orbitals below %s Ha"
                     % (nfrozen, nfrozen_cutoff))
        if nfrozen is not None:
            if nfrozen >= nel // 2:
                raise ValueError(
                    f"nfrozen={nfrozen} must be smaller than the embedded doubly occupied "
                    f"count {nel // 2} (nelec={nel}). It is sized to the embedded problem.")
            nevpt_obj.nfrozen = nfrozen
        for key, val in nevpt_kwargs.items():
            setattr(nevpt_obj, key, val)

        e_tot, e_corr, _ = nevpt_obj.kernel()
        e_tot = np.atleast_1d(e_tot)
        e_corr = np.atleast_1d(e_corr)

        log.info("\nPC-NEVPT2 state energies (excitation = state - state 0):")
        for i, (et, ec) in enumerate(zip(e_tot, e_corr)):
            de_ev = (et - e_tot[0]) * _eV
            log.info("  State %d: E_tot %.10f Ha, dE %+.4f eV" % (i, et, de_ev))

    return e_tot, e_corr, mc, nevpt_obj


def execute(task):
    # Named params are popped; the rest is forwarded to the Prism NEVPT via setattr.
    _kw = dict(task.get('pcnevpt2_kwargs', {}))
    e_tot, e_corr, mc, nevpt_obj = solve(
        task['dmet_fock'],
        task['dmet_tei'],
        task['norb'],
        task['nel'],
        task['nimp'],
        task.get('dm_guess_rhf'),
        ncas=task.get('ncas'),
        nelecas=task.get('nelecas'),
        sa_nstates=task.get('sa_nstates', 1),
        sa_weights=task.get('sa_weights'),
        chempot_imp=task.get('chempot_imp', 0.0),
        verbose=task.get('verbose', logger.INFO),
        prism_backend=_kw.pop('prism_backend', 'opt_einsum'),
        nfrozen=_kw.pop('nfrozen', None),
        nfrozen_cutoff=_kw.pop('nfrozen_cutoff', -2.0),
        compute_singles=_kw.pop('compute_singles', False),
        s_thresh_singles=_kw.pop('s_thresh_singles', 1e-8),
        s_thresh_doubles=_kw.pop('s_thresh_doubles', 1e-8),
        select_reference=_kw.pop('select_reference', None),
        casscf_kwargs=task.get('casscf_kwargs', {}),
        nevpt_kwargs=_kw,
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
        dip_mom_ao=task.get('dip_mom_ao'),
        embedding_data=task.get('embedding_data'),
        embedded_ref=task.get('embedded_ref'),
        no_kernel=task.get('no_kernel', False),
    )

    rdm1 = mc.make_rdm1()
    pcnevpt2_res = {
        'e_tot': e_tot,
        'e_corr': e_corr,
        'mc': mc,
        'nevpt': nevpt_obj,
    }
    # Embedded CASSCF, before the PT2. Only state-averaged runs carry e_states.
    if task.get('sa_nstates', 1) > 1:
        pcnevpt2_res['e_cas_states'] = np.asarray(mc.e_states)
    return e_tot[0], rdm1, pcnevpt2_res
