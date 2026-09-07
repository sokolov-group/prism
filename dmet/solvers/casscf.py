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
from pyscf import fci as pyscf_fci

import prism.lib.logger as logger
from prism.dmet.utils import silent_stdout
from prism.dmet.cas_selectors import (natorb_active_space, fix_cas_spin,
                                      stabilize_rohf)


def solve(const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf,
          ncas=None, nelecas=None,
          chempot_imp=0.0, verbose=logger.INFO,
          sa_nstates=1, sa_weights=None,
          frozen=None,
          mo_guess=None, ci_guess=None,
          spin=None,
          cas_select='energy',
          embed_level_shift=0.0, rohf_stability=False,
          cas_spin=None, cas_spin_shift=0.2,
          natorb_occ_thresh=0.02, natorb_max_superset=None,
          deg_tol=1e-3, casci_conv_tol=1e-10,
          **casscf_kwargs):
    if ncas is None:
        ncas = norb
    if nelecas is None:
        nelecas = nel
    if cas_select not in ('energy', 'natorb'):
        raise ValueError(
            f"Invalid cas_select='{cas_select}'. Valid: ['energy', 'natorb']")

    log = logger.Logger(sys.stdout, verbose)
    printoutput = verbose >= logger.INFO

    _spin = spin if spin is not None else (nel % 2)
    _use_rohf = (_spin != 0)

    if ncas > norb:
        raise ValueError(f"Invalid ncas={ncas}: cannot exceed norb={norb}.")
    if nelecas > nel:
        raise ValueError(f"Invalid nelecas={nelecas}: cannot exceed nel={nel}.")
    if not _use_rohf:
        if nel % 2 != 0:
            raise ValueError(f"Invalid nel={nel}: must be even (RHF reference required).")
        if (nel - nelecas) % 2 != 0:
            raise ValueError(
                "Invalid nel/nelecas parity: (nel - nelecas) must be even "
                "(frozen core must be closed-shell).")

    if sa_nstates > 1:
        if sa_weights is None:
            sa_weights = [1.0 / sa_nstates] * sa_nstates
        if len(sa_weights) != sa_nstates:
            raise ValueError("Invalid sa_weights: length must equal sa_nstates.")
        sa_weights = np.array(sa_weights, dtype=float)
        sa_weights /= sa_weights.sum()
    else:
        sa_weights = np.array([1.0])

    ctx = silent_stdout() if not printoutput else nullcontext()

    fock_copy = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy[orb, orb] -= chempot_imp

    with ctx:
        mol = gto.Mole()
        mol.build(verbose=0)
        mol.atom.append(('C', (0, 0, 0)))
        mol.nelectron = nel
        mol.spin = _spin
        mol.incore_anyway = True

        if _use_rohf:
            mf = scf.ROHF(mol)
            log.info("Using ROHF reference (spin=%d, nel=%d)" % (_spin, nel))
        else:
            mf = scf.RHF(mol)

        mf.get_hcore = lambda *args: fock_copy
        mf.get_ovlp = lambda *args: np.eye(norb)
        mf._eri = ao2mo.restore(8, tei, norb)
        # Level shift opens the near-degenerate trap gap for a deterministic embedded SCF.
        mf.level_shift = embed_level_shift
        mf.scf(dm_guess_rhf)
        dm_loc = mf.mo_coeff @ np.diag(mf.mo_occ) @ mf.mo_coeff.T
        if not mf.converged:
            mf.max_cycle = 300
            mf.diis_space = 12
            mf.scf(dm_loc)
            dm_loc = mf.mo_coeff @ np.diag(mf.mo_occ) @ mf.mo_coeff.T
        if embed_level_shift != 0.0:
            # Check the shifted fixed point is stationary for the real H; motion = masked instability.
            e_shifted = mf.e_tot
            mf.level_shift = 0.0
            mf.scf(dm_loc)
            log.info("level-shift verification: E(shift=%s)=%.10f  "
                     "E(shift removed, reconverged)=%.10f  dE=%.2e Ha"
                     % (embed_level_shift, e_shifted, mf.e_tot, abs(mf.e_tot - e_shifted)))

        if rohf_stability and _use_rohf:
            stabilize_rohf(mf, log=log)

        # Skip selection with a warm-restart guess: mc.kernel would overwrite mo_coeff.
        mo_natorb = None
        if cas_select == 'natorb' and mo_guess is None:
            mo_natorb, ncas, nelecas = natorb_active_space(
                mf, ncas, occ_thresh=natorb_occ_thresh, max_superset=natorb_max_superset,
                sa_nstates=sa_nstates, cas_spin=cas_spin, cas_spin_shift=cas_spin_shift,
                deg_tol=deg_tol, conv_tol=casci_conv_tol, log=log)

        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.verbose = 5 if printoutput else 0
        if frozen is not None:
            mc.frozen = frozen
        for key, val in casscf_kwargs.items():
            setattr(mc, key, val)

        if cas_spin is not None:
            fix_cas_spin(mc.fcisolver, cas_spin, cas_spin_shift)

        if sa_nstates > 1:
            mc = mcscf.state_average_(mc, weights=sa_weights.tolist())

        if mo_natorb is not None:
            mc.mo_coeff = mo_natorb
            log.info("CAS selection by %s, CAS(%d,%d)" % (cas_select, nelecas, ncas))

        _mo0 = mo_guess if mo_guess is not None else None
        _ci0 = ci_guess if ci_guess is not None else None
        mc.kernel(_mo0, _ci0)

        ncore = mc.ncore
        if _use_rohf:
            _nelecas_fci = ((nelecas + _spin) // 2, (nelecas - _spin) // 2)
            ci_solver_base = pyscf_fci.direct_spin1.FCI()
        else:
            _nelecas_fci = nelecas
            ci_solver_base = pyscf_fci.direct_spin0.FCI()

        if sa_nstates > 1:
            # SA-CASSCF: per-state CAS RDMs, then take weighted average.
            ci_vecs = mc.ci
            rdm1_cas = np.zeros((ncas, ncas))
            rdm2_cas = np.zeros((ncas, ncas, ncas, ncas))
            for w, ci_vec in zip(sa_weights, ci_vecs):
                r1, r2 = ci_solver_base.make_rdm12(ci_vec, ncas, _nelecas_fci)
                rdm1_cas += w * r1
                rdm2_cas += w * r2
            e_states = np.array(mc.e_states)
            e_tot = mc.e_tot   # weighted average
            log.info("\nSA-CASSCF state energies:")
            for i, e in enumerate(e_states):
                log.info("  State %d: %.10f Ha  (weight=%.4f)" % (i, e, sa_weights[i]))
            log.info("  Weighted average: %.10f Ha" % e_tot)
        else:
            rdm1_cas, rdm2_cas = ci_solver_base.make_rdm12(mc.ci, ncas, _nelecas_fci)
            e_states = None
            e_tot = mc.e_tot
            log.info("\nCASSCF energy: %.10f Ha" % e_tot)
            log.info("  ncore=%d, ncas=%d, nelecas=%d" % (ncore, ncas, nelecas))

        nmo = mf.mo_coeff.shape[1]
        dm1_mo = np.zeros((nmo, nmo))
        dm2_mo = np.zeros((nmo, nmo, nmo, nmo))

        for i in range(ncore):
            dm1_mo[i, i] = 2.0
            for j in range(ncore):
                dm2_mo[i, i, j, j] += 4.0
                dm2_mo[i, j, j, i] -= 2.0
            for p in range(ncas):
                for q in range(ncas):
                    dm2_mo[i, i, ncore+p, ncore+q] += 2.0 * rdm1_cas[p, q]
                    dm2_mo[ncore+p, ncore+q, i, i] += 2.0 * rdm1_cas[p, q]
                    dm2_mo[i, ncore+q, ncore+p, i] -= rdm1_cas[p, q]
                    dm2_mo[ncore+p, i, i, ncore+q] -= rdm1_cas[p, q]

        dm1_mo[ncore:ncore+ncas, ncore:ncore+ncas] = rdm1_cas
        dm2_mo[ncore:ncore+ncas, ncore:ncore+ncas,
               ncore:ncore+ncas, ncore:ncore+ncas] = rdm2_cas

        log.info("Full-space 1-RDM trace: %.6f" % np.trace(dm1_mo))

        # Rotate from MO basis to local dmet orbital basis.
        C = mc.mo_coeff
        pyscf_rdm1 = C @ dm1_mo @ C.T
        pyscf_rdm2 = np.einsum('ai,ijkl->ajkl', C, dm2_mo)
        pyscf_rdm2 = np.einsum('bj,ajkl->abkl', C, pyscf_rdm2)
        pyscf_rdm2 = np.einsum('ck,abkl->abcl', C, pyscf_rdm2)
        pyscf_rdm2 = np.einsum('dl,abcl->abcd', C, pyscf_rdm2)

        impurity_energy = (
            const
            + 0.25  * np.einsum('ij,ij->', pyscf_rdm1[:nimp, :],     fock[:nimp, :] + oei[:nimp, :])
            + 0.25  * np.einsum('ij,ij->', pyscf_rdm1[:, :nimp],     fock[:, :nimp] + oei[:, :nimp])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:nimp, :, :, :], tei[:nimp, :, :, :])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:, :nimp, :, :], tei[:, :nimp, :, :])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:, :, :nimp, :], tei[:, :, :nimp, :])
            + 0.125 * np.einsum('ijkl,ijkl->', pyscf_rdm2[:, :, :, :nimp], tei[:, :, :, :nimp])
        )

    cas_results = {
        'e_tot': e_tot,
        'e_states': e_states,
        'e_imp': impurity_energy,
        'ncas': ncas,
        'nelecas': nelecas,
        'ncore': mc.ncore,
        'nstates': sa_nstates,
        'weights': sa_weights,
        'ci': mc.ci,
        'mo_coeff': mc.mo_coeff,
        'cas_select': cas_select,
    }

    return impurity_energy, pyscf_rdm1, cas_results


def execute(task):
    return solve(
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
        chempot_imp=task.get('chempot_imp', 0.0),
        verbose=task.get('verbose', logger.INFO),
        sa_nstates=task.get('sa_nstates', 1),
        sa_weights=task.get('sa_weights'),
        mo_guess=task.get('mo_guess'),
        ci_guess=task.get('ci_guess'),
        spin=task.get('spin'),
        cas_select=task.get('cas_select', 'energy'),
        embed_level_shift=task.get('embed_level_shift', 0.0),
        rohf_stability=task.get('rohf_stability', False),
        cas_spin=task.get('cas_spin'),
        cas_spin_shift=task.get('cas_spin_shift', 0.2),
        natorb_occ_thresh=task.get('natorb_occ_thresh', 0.02),
        natorb_max_superset=task.get('natorb_max_superset'),
        deg_tol=task.get('deg_tol', 1e-3),
        casci_conv_tol=task.get('casci_conv_tol', 1e-10),
        **task.get('casscf_kwargs', {}),
    )
