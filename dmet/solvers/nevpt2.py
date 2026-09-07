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
import numpy as np
from pyscf import ao2mo, gto, scf, mcscf, mrpt

import prism.lib.logger as logger
from prism.dmet.utils import silent_stdout, nullcontext
from prism.dmet.cas_selectors import natorb_active_space, fix_cas_spin, multiseed_casscf
from prism.dmet.solvers.casscf import _stabilize_rohf

_eV = 27.21138602


def solve(const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf,
          ncas, nelecas,
          nstates=1, sa_weights=None,
          chempot_imp=0.0, verbose=logger.INFO,
          casscf_kwargs=None, nevpt2_kwargs=None,
          spin=None,
          cas_select='energy',
          embed_level_shift=0.0, rohf_stability=False, cas_multiseed=False,
          cas_spin=None, cas_spin_shift=0.2,
          natorb_occ_thresh=0.02, natorb_max_superset=None,
          deg_tol=1e-3, casci_conv_tol=1e-10):
    casscf_kwargs = casscf_kwargs or {}
    nevpt2_kwargs = nevpt2_kwargs or {}
    if cas_select not in ('energy', 'natorb'):
        raise ValueError(
            f"nevpt2::solve: unknown cas_select='{cas_select}'. Valid: ['energy', 'natorb']")

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
        mol = gto.Mole()
        mol.build(verbose=0)
        mol.atom.append(('C', (0, 0, 0)))
        mol.nelectron = nel
        mol.spin = _spin
        mol.incore_anyway = True

        mf = scf.ROHF(mol) if _use_rohf else scf.RHF(mol)
        mf.get_hcore = lambda *args: fock_copy
        mf.get_ovlp  = lambda *args: np.eye(norb)
        mf._eri      = ao2mo.restore(8, tei, norb)
        # Level shift opens the near-degenerate trap gap for a deterministic embedded SCF.
        mf.level_shift = embed_level_shift
        mf.scf(dm_guess_rhf)
        if not mf.converged:
            mf.max_cycle = 300
            mf.diis_space = 12
            mf.scf(mf.make_rdm1())
        if embed_level_shift != 0.0:
            # Confirm the shifted fixed point is also a stationary point of the real
            # (unshifted) Hamiltonian; if it moves, the shift masked the instability.
            e_shifted = mf.e_tot
            mf.level_shift = 0.0
            mf.scf(mf.make_rdm1())
            log.info("nevpt2::solve : level-shift verification: E(shift=%s)=%.10f  "
                     "E(shift removed, reconverged)=%.10f  dE=%.2e Ha"
                     % (embed_level_shift, e_shifted, mf.e_tot, abs(mf.e_tot - e_shifted)))
        if rohf_stability and _use_rohf:
            _stabilize_rohf(mf, tag='nevpt2::solve', log=log)
        if _use_rohf:
            log.info("nevpt2::solve : embedded ROHF (spin=%d, nel=%d, norb=%d)"
                     % (_spin, nel, norb))

        if nstates == 1:
            mo_natorb = None
            if cas_select == 'natorb':
                mo_natorb, ncas, nelecas = natorb_active_space(
                    mf, ncas, occ_thresh=natorb_occ_thresh, max_superset=natorb_max_superset,
                    sa_nstates=1, cas_spin=cas_spin, cas_spin_shift=cas_spin_shift,
                    deg_tol=deg_tol, conv_tol=casci_conv_tol, log=log)

            mc = mcscf.CASSCF(mf, ncas, nelecas)
            mc.verbose = 5 if printoutput else 0
            for key, val in casscf_kwargs.items():
                setattr(mc, key, val)

            if cas_spin is not None:
                fix_cas_spin(mc.fcisolver, cas_spin, cas_spin_shift)

            if mo_natorb is not None:
                mc.mo_coeff = mo_natorb
                log.info("nevpt2::solve : CAS selection by %s, CAS(%d,%d)"
                         % (cas_select, nelecas, ncas))

            if cas_multiseed:
                multiseed_casscf(mc, mc.mo_coeff, log=log)
            else:
                mc.kernel()

            log.info("\nnevpt2::solve : embedded CASSCF energy = %.10f Ha" % mc.e_tot)

            nevpt_obj = mrpt.NEVPT(mc, root=0)
            nevpt_obj.verbose = 5 if printoutput else 0
            for key, val in nevpt2_kwargs.items():
                setattr(nevpt_obj, key, val)
            # mrpt.NEVPT.kernel() calls self.canonicalize(..., cas_natorb=True), which
            # invokes mc.cas_natorb() -> orth.orth_ao(mc.mol, 'meta_lowdin') -> fails on
            # the dummy mol (no real AO basis). Override to skip the natorb step; Fock
            # diagonalization still canonicalizes inactive/external orbitals correctly.
            _mc_ref = mc
            nevpt_obj.canonicalize = lambda mo, ci, eris=None, sort=False, cas_natorb=True, casdm1=None, verbose=None: \
                _mc_ref.canonicalize(mo, ci, eris, sort, False, casdm1, verbose)
            e_c = nevpt_obj.kernel()

            e_tot  = np.array([nevpt_obj.e_tot])
            e_corr = np.array([e_c])
            nevpt_objs = [nevpt_obj]

        else:
            if sa_weights is None:
                sa_weights = [1.0 / nstates] * nstates
            sa_weights = np.array(sa_weights, dtype=float)
            sa_weights /= sa_weights.sum()

            mo_natorb = None
            if cas_select == 'natorb':
                mo_natorb, ncas, nelecas = natorb_active_space(
                    mf, ncas, occ_thresh=natorb_occ_thresh, max_superset=natorb_max_superset,
                    sa_nstates=nstates, cas_spin=cas_spin, cas_spin_shift=cas_spin_shift,
                    deg_tol=deg_tol, conv_tol=casci_conv_tol, log=log)

            mc_sa = mcscf.CASSCF(mf, ncas, nelecas)
            mc_sa = mcscf.state_average_(mc_sa, weights=sa_weights.tolist())
            mc_sa.verbose = 5 if printoutput else 0
            for key, val in casscf_kwargs.items():
                setattr(mc_sa, key, val)
            if cas_spin is not None:
                fix_cas_spin(mc_sa.fcisolver, cas_spin, cas_spin_shift)

            if mo_natorb is not None:
                mc_sa.mo_coeff = mo_natorb
                log.info("nevpt2::solve : CAS selection by %s (SA), CAS(%d,%d)"
                         % (cas_select, nelecas, ncas))

            if cas_multiseed:
                multiseed_casscf(mc_sa, mc_sa.mo_coeff, log=log)
            else:
                mc_sa.kernel()
            sa_mo = mc_sa.mo_coeff

            log.info("\nnevpt2::solve : embedded SA-CASSCF (%d states) done." % nstates)
            for i, e in enumerate(mc_sa.e_states):
                log.info("  State %d: %.10f Ha  (weight=%.4f)" % (i, e, sa_weights[i]))

            # Multi-root CASCI on the frozen SA-CASSCF MOs, then per-root SC-NEVPT2.
            mc = mcscf.CASCI(mf, ncas, nelecas)
            mc.verbose = 4 if printoutput else 0
            mc.fcisolver.nroots = nstates
            if cas_spin is not None:
                fix_cas_spin(mc.fcisolver, cas_spin, cas_spin_shift)
                mc.fcisolver.nroots = nstates

            mc.kernel(sa_mo)

            log.info("\nnevpt2::solve : Multi-root CASCI energies:")
            for i, e in enumerate(mc.e_tot):
                log.info("  State %d: %.10f Ha" % (i, e))

            e_tot  = np.zeros(nstates)
            e_corr = np.zeros(nstates)
            nevpt_objs = []
            _mc_ref = mc
            for i in range(nstates):
                nevpt_i = mrpt.NEVPT(mc, root=i)
                nevpt_i.verbose = 5 if printoutput else 0
                for key, val in nevpt2_kwargs.items():
                    setattr(nevpt_i, key, val)
                nevpt_i.canonicalize = lambda mo, ci, eris=None, sort=False, cas_natorb=True, casdm1=None, verbose=None: \
                    _mc_ref.canonicalize(mo, ci, eris, sort, False, casdm1, verbose)
                e_c_i = nevpt_i.kernel()
                # mc.e_tot is the full multi-root CASCI array; nevpt_i.e_tot would
                # broadcast (e_corr + array). Take the scalar total for root i.
                e_tot[i]  = mc.e_tot[i] + e_c_i
                e_corr[i] = e_c_i
                nevpt_objs.append(nevpt_i)

        log.info("\nnevpt2::solve : NEVPT2 results:")
        log.info("  %5s  %16s  %14s  %16s" % ('State', 'E_tot (Ha)', 'E_corr (Ha)', 'dE from GS (eV)'))
        log.info("  " + "-" * 56)
        for i, (et, ec) in enumerate(zip(e_tot, e_corr)):
            de_ev = (et - e_tot[0]) * _eV
            log.info("  %5d  %16.10f  %14.10f  %+16.4f" % (i, et, ec, de_ev))

    return e_tot, e_corr, mc, nevpt_objs


def execute(task):
    e_tot, e_corr, mc, nevpt_objs = solve(
        task['const'],
        task['dmet_oei'],
        task['dmet_fock'],
        task['dmet_tei'],
        task['norb'],
        task['nel'],
        task['nimp'],
        task.get('dm_guess_rhf'),
        ncas=task.get('ncas'),
        nelecas=task.get('nelecas'),
        nstates=task.get('sa_nstates', 1),
        sa_weights=task.get('sa_weights'),
        chempot_imp=task.get('chempot_imp', 0.0),
        verbose=task.get('verbose', logger.INFO),
        casscf_kwargs=task.get('casscf_kwargs', {}),
        nevpt2_kwargs=task.get('nevpt2_kwargs', {}),
        spin=task.get('spin'),
        cas_select=task.get('cas_select', 'energy'),
        embed_level_shift=task.get('embed_level_shift', 0.0),
        rohf_stability=task.get('rohf_stability', False),
        cas_multiseed=task.get('cas_multiseed', False),
        cas_spin=task.get('cas_spin'),
        cas_spin_shift=task.get('cas_spin_shift', 0.2),
        natorb_occ_thresh=task.get('natorb_occ_thresh', 0.02),
        natorb_max_superset=task.get('natorb_max_superset'),
        deg_tol=task.get('deg_tol', 1e-3),
        casci_conv_tol=task.get('casci_conv_tol', 1e-10),
    )

    # 1-RDM (cluster orbital basis) for the dmet driver: prefer the NEVPT2
    # relaxed density if available, else the CASSCF/CASCI density.
    nevpt_gs = nevpt_objs[0]
    if hasattr(nevpt_gs, 'onerdm') and nevpt_gs.onerdm is not None:
        rdm1 = nevpt_gs.onerdm
    else:
        rdm1 = mc.make_rdm1()

    nevpt2_res = {
        'e_tot'      : e_tot,
        'e_corr'     : e_corr,
        'mc'         : mc,
        'nevpt_objs' : nevpt_objs,
    }
    return e_tot[0], rdm1, nevpt2_res
