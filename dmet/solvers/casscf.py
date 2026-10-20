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
from pyscf import ao2mo, fci, gto, scf, mcscf

import prism.lib.logger as logger
from prism.dmet.utils import attach_eri, silent_stdout
from prism.dmet.cas_selectors import (natorb_active_space, fix_cas_spin,
                                      stabilize_scf, set_reference_no_scf)


def solve(const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf,
          ncas=None, nelecas=None,
          chempot_imp=0.0, verbose=logger.INFO,
          sa_nstates=1, sa_weights=None, sa_mix=None,
          frozen=None,
          mo_guess=None, ci_guess=None,
          spin=None,
          cas_select='energy',
          embed_level_shift=0.0, scf_stability=False,
          cas_spin=None, cas_spin_shift=0.2,
          natorb_occ_thresh=0.02, natorb_max_superset=None,
          deg_tol=1e-3, casci_conv_tol=1e-10,
          embedded_ref=None, no_kernel=False,
          cderi=None, max_memory=None,
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

    if sa_mix is not None:
        if cas_spin is not None:
            raise ValueError(
                "sa_mix and cas_spin conflict: cas_spin penalizes every state outside "
                "one spin, including the ones sa_mix averages over. Drop cas_spin.")
        sa_mix = [(int(spin2), int(nroots)) for spin2, nroots in sa_mix]
        if any(n < 1 for _, n in sa_mix):
            raise ValueError("Invalid sa_mix: every entry needs at least one root.")
        sa_nstates = sum(n for _, n in sa_mix)

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
        # The embedded problem carries no Mole of its own, so PySCF's default budget
        # would otherwise size every downstream block.
        if max_memory is not None:
            mol.max_memory = max_memory

        if _use_rohf:
            mf = scf.ROHF(mol)
            log.info("Using ROHF reference (spin=%d, nel=%d)" % (_spin, nel))
        else:
            mf = scf.RHF(mol)

        mf = attach_eri(mf, tei, cderi, norb)
        mf.get_hcore = lambda *args: fock_copy
        mf.get_ovlp = lambda *args: np.eye(norb)
        # Level shift opens the near-degenerate trap gap for a deterministic embedded SCF.
        mf.level_shift = embed_level_shift
        if no_kernel:
            set_reference_no_scf(
                mf, embedded_ref if embedded_ref is not None else dm_guess_rhf,
                nel, _spin, log=log)
        else:
            mf.scf(dm_guess_rhf)
            dm_loc = mf.mo_coeff @ np.diag(mf.mo_occ) @ mf.mo_coeff.T
            if not mf.converged:
                mf.max_cycle = 300
                mf.diis_space = 12
                mf.scf(dm_loc)
                dm_loc = mf.mo_coeff @ np.diag(mf.mo_occ) @ mf.mo_coeff.T
            if embed_level_shift != 0.0:
                # The shifted fixed point should be stationary for the real H;
                # motion indicates a masked instability.
                e_shifted = mf.e_tot
                mf.level_shift = 0.0
                mf.scf(dm_loc)
                log.info("level-shift verification: E(shift=%s)=%.10f  "
                         "E(shift removed, reconverged)=%.10f  dE=%.2e Ha"
                         % (embed_level_shift, e_shifted, mf.e_tot, abs(mf.e_tot - e_shifted)))
            if scf_stability:
                stabilize_scf(mf, log=log)

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
            fix_cas_spin(mc, cas_spin, cas_spin_shift)

        if sa_mix is not None:
            # One orbital set averaged over several spins, each with its own root
            # count. Every solver is penalized toward the spin it was asked for.
            solvers = []
            for spin2, nroots in sa_mix:
                solver = (fci.direct_spin0.FCI(mol) if spin2 == 0
                          else fci.direct_spin1.FCI(mol))
                solver.spin = spin2
                solver.nroots = nroots
                ss = (spin2 / 2.0) * (spin2 / 2.0 + 1.0)
                fci.addons.fix_spin_(solver, shift=cas_spin_shift, ss=ss)
                solvers.append(solver)
            mc = mcscf.state_average_mix_(mc, solvers, sa_weights.tolist())
            spins = ", ".join("%d roots at 2S=%d" % (n, spin2) for spin2, n in sa_mix)
            log.info("Mixed-spin state average: %s" % spins)
        elif sa_nstates > 1:
            mc = mcscf.state_average_(mc, weights=sa_weights.tolist())

        if mo_natorb is not None:
            mc.mo_coeff = mo_natorb
            log.info("CAS selection by %s, CAS(%d,%d)" % (cas_select, nelecas, ncas))

        _mo0 = mo_guess if mo_guess is not None else None
        _ci0 = ci_guess if ci_guess is not None else None
        mc.kernel(_mo0, _ci0)

        ncore = mc.ncore
        # The solver holds the CAS spin and root weights.
        rdm1_cas, rdm2_cas = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)

        if sa_nstates > 1:
            e_states = np.array(mc.e_states)
            e_tot = mc.e_tot   # weighted average
            log.info("\nSA-CASSCF state energies:")
            for i, e in enumerate(e_states):
                log.info("  State %d: %.10f Ha  (weight=%.4f)" % (i, e, sa_weights[i]))
            log.info("  Weighted average: %.10f Ha" % e_tot)
        else:
            e_states = None
            e_tot = mc.e_tot
            log.info("\nCASSCF energy: %.10f Ha" % e_tot)
            log.info("  ncore=%d, ncas=%d, nelecas=%d" % (ncore, ncas, nelecas))

        nmo = mf.mo_coeff.shape[1]
        dm1_mo = np.zeros((nmo, nmo))
        dm1_mo[np.diag_indices(ncore)] = 2.0
        dm1_mo[ncore:ncore+ncas, ncore:ncore+ncas] = rdm1_cas

        log.info("Full-space 1-RDM trace: %.6f" % np.trace(dm1_mo))

        # Rotate from MO basis to local dmet orbital basis.
        C = mc.mo_coeff
        pyscf_rdm1 = C @ dm1_mo @ C.T

        # The full-space 2-RDM is the mean-field product of pyscf_rdm1 plus a cumulant
        # confined to the active block. The four impurity projections are equal by
        # symmetry, giving the 0.5 weights.
        vj, vk = mf.get_jk(dm=pyscf_rdm1, hermi=1)
        veff = vj - 0.5 * vk

        mo_cas = C[:, ncore:ncore+ncas]
        mo_imp = np.zeros_like(mo_cas)
        mo_imp[:nimp] = mo_cas[:nimp]
        mos = [mo_imp, mo_cas, mo_cas, mo_cas]
        if cderi is not None:
            v_cas = mf.with_df.ao2mo(mos, compact=False)
        else:
            v_cas = ao2mo.incore.general(mf._eri, mos, compact=False)
        v_cas = v_cas.reshape([ncas] * 4)
        j_cas = np.einsum('pqrs,rs->pq', v_cas, rdm1_cas)
        k_cas = np.einsum('pqrs,rq->ps', v_cas, rdm1_cas)
        e_cumulant = (np.einsum('pqrs,pqrs->', rdm2_cas, v_cas)
                      - np.einsum('pq,pq->', rdm1_cas, j_cas)
                      + 0.5 * np.einsum('ps,ps->', rdm1_cas, k_cas))

        impurity_energy = (
            const
            + 0.25 * np.einsum('ij,ij->', pyscf_rdm1[:nimp, :], fock[:nimp, :] + oei[:nimp, :])
            + 0.25 * np.einsum('ij,ij->', pyscf_rdm1[:, :nimp], fock[:, :nimp] + oei[:, :nimp])
            + 0.5  * np.einsum('ij,ij->', pyscf_rdm1[:nimp, :], veff[:nimp, :])
            + 0.5  * e_cumulant
        )

    # The roots carry different spins, so report each one's <S^2> and confirm they
    # came back with the spins that were requested.
    spin_square = None
    if sa_mix is not None:
        spin_square = np.asarray(
            mc.fcisolver.states_spin_square(mc.ci, mc.ncas, mc.nelecas)[0])
        want = [(spin2 / 2.0) * (spin2 / 2.0 + 1.0)
                for spin2, nroots in sa_mix for _ in range(nroots)]
        off = [i for i, (w, g) in enumerate(zip(want, spin_square)) if abs(w - g) > 0.1]
        if off:
            raise RuntimeError(
                f"sa_mix asked for <S^2> {want} but got "
                f"{np.round(spin_square, 3).tolist()}; roots {off} are off. Raise "
                f"cas_spin_shift, or the requested spins are not the lowest available.")

    cas_results = {
        'e_tot': e_tot,
        'e_states': e_states,
        'spin_square': spin_square,
        'sa_mix': sa_mix,
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
        sa_mix=task.get('sa_mix'),
        mo_guess=task.get('mo_guess'),
        ci_guess=task.get('ci_guess'),
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
        embedded_ref=task.get('embedded_ref'),
        no_kernel=task.get('no_kernel', False),
        cderi=task.get('dmet_cderi'),
        max_memory=task.get('max_memory'),
        **task.get('casscf_kwargs', {}),
    )
