# Copyright 2026 Prism Developers. All Rights Reserved.
# Portions adapted from mrh (mrh.my_dmet.pyscf_casscf)
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

import prism.lib.logger as logger


def _get_log(log):
    return log if log is not None else logger.Logger(sys.stdout, logger.INFO)


def stabilize_rohf(mf, max_iter=5, log=None):
    # Follow ROHF instabilities until the solution is stable.
    from pyscf import scf
    log = _get_log(log)
    if not isinstance(mf, scf.rohf.ROHF):
        return
    for i in range(max_iter):
        mo_i, _, stable_i, _ = mf.stability(return_status=True)
        if stable_i:
            if i > 0:
                log.info("ROHF stable after %d stability follow(s), E=%.10f" % (i, mf.e_tot))
            else:
                log.info("ROHF internally stable, E=%.10f" % mf.e_tot)
            return
        log.info("ROHF internal instability (iteration %d), reconverging along the unstable mode"
                 % (i + 1))
        mf.scf(mf.make_rdm1(mo_i, mf.mo_occ))
    log.warn("ROHF stability not reached after %d reconverges, E=%.10f"
             % (max_iter, mf.e_tot))


def canonicalize_degenerate_active_nos(cas_no, act_idx, no_occ, f_emb, deg_tol=1e-3):
    # Diagonalize the projected embedded Fock within each degenerate block.
    act_cols = cas_no[:, act_idx].copy()
    occ_vals = no_occ[act_idx]
    i = 0
    while i < len(act_idx):
        j = i + 1
        while j < len(act_idx) and abs(occ_vals[j] - occ_vals[i]) < deg_tol:
            j += 1
        if j - i > 1:
            blk = act_cols[:, i:j]
            _, U = np.linalg.eigh(blk.T @ f_emb @ blk)
            act_cols[:, i:j] = np.dot(blk, U)
        i = j
    for c in range(act_cols.shape[1]):
        if act_cols[np.argmax(np.abs(act_cols[:, c])), c] < 0:
            act_cols[:, c] *= -1
    cas_no[:, act_idx] = act_cols


def fix_cas_spin(fcisolver, cas_spin, shift=0.2):
    # Penalize CAS states outside spin 2S=cas_spin via pyscf fix_spin_ (ss=S(S+1)).
    from pyscf.fci.addons import fix_spin_
    s = cas_spin / 2.0
    fcisolver.spin = cas_spin
    fix_spin_(fcisolver, shift=shift, ss=s * (s + 1))
    return fcisolver


def natorb_active_space(mf, n_superset, occ_thresh=0.02, deg_tol=1e-3, max_superset=None,
                        sa_nstates=1, cas_spin=None, cas_spin_shift=0.2, conv_tol=1e-10,
                        log=None):
    from pyscf import mcscf
    from math import comb
    log = _get_log(log)
    C = np.asarray(mf.mo_coeff)
    if C.ndim == 3:
        raise NotImplementedError(
            "Natorb selection expects a single set of orbitals (RHF/ROHF reference).")
    mo_e = np.asarray(mf.mo_energy)
    occ = np.asarray(mf.mo_occ)
    norb = C.shape[1]
    nocc = int(np.sum(occ > 0))

    half = max(1, int(n_superset) // 2)
    lo, hi = max(0, nocc - half), min(norb, nocc + half)
    while lo > 0 and (mo_e[lo] - mo_e[lo - 1]) < deg_tol:
        lo -= 1
    while hi < norb and (mo_e[hi] - mo_e[hi - 1]) < deg_tol:
        hi += 1
    ncas_s = hi - lo
    if max_superset is not None and ncas_s > max_superset:
        raise ValueError(
            f"Manifold snapping inflated the superset to {ncas_s} "
            f"orbitals (window [{lo}, {hi})), exceeding max_superset={max_superset}. A dense "
            f"near-degenerate manifold at the Fermi level pushed the exact CASCI past the "
            f"tractable/QD-NEVPT2 limit. Lower n_superset, or raise max_superset "
            f"(exact-CASCI cost grows ~factorially)."
        )
    win = occ[lo:hi]
    na = int(np.sum(np.rint(win) >= 1))   # alpha occupied in window
    nb = int(np.sum(np.rint(win) >= 2))   # beta (doubly) occupied in window

    # Restrict the superset CASCI to spin 2S = na - nb via fix_spin_.
    if cas_spin is not None:
        ne_win = na + nb
        if cas_spin < 0 or cas_spin > ne_win or (ne_win - cas_spin) % 2 != 0:
            raise ValueError(
                f"Invalid cas_spin={cas_spin} (2S): incompatible with "
                f"{ne_win} superset electrons; need 0 <= cas_spin <= {ne_win} with matching parity.")
        na, nb = (ne_win + cas_spin) // 2, (ne_win - cas_spin) // 2

    # Report superset size and FCI cost before the (possibly long) CASCI.
    fci_dim = comb(ncas_s, na) * comb(ncas_s, nb)
    log.info("natorb superset window [%d, %d): %d orbitals, CASCI(%d,%d) x %d root(s), "
             "FCI dim ~%.2e dets - starting CASCI..."
             % (lo, hi, ncas_s, na + nb, ncas_s, sa_nstates, fci_dim))

    mc = mcscf.CASCI(mf, ncas_s, (na, nb))
    mc.fcisolver.conv_tol = conv_tol
    mc.verbose = 0
    if cas_spin is not None:
        fix_cas_spin(mc.fcisolver, cas_spin, cas_spin_shift)
    if sa_nstates > 1:
        mc.fcisolver.nroots = sa_nstates
    mc.kernel()

    # State-averaged selection density when targeting >1 state (CISNO-style).
    if sa_nstates > 1:
        dm1 = sum(mc.fcisolver.make_rdm1(civec, ncas_s, mc.nelecas)
                  for civec in mc.ci) / sa_nstates
    else:
        dm1 = mc.fcisolver.make_rdm1(mc.ci, ncas_s, mc.nelecas)
    no_occ, u = np.linalg.eigh(dm1)
    order = np.argsort(no_occ)[::-1]
    no_occ, u = no_occ[order], u[:, order]
    cas_no = np.dot(mc.mo_coeff[:, mc.ncore:mc.ncore + ncas_s], u)

    is_core = no_occ >= 2 - occ_thresh
    is_act = (no_occ > occ_thresh) & (no_occ < 2 - occ_thresh)
    if not np.any(is_act):
        raise ValueError(
            f"No fractionally occupied NOs in the CAS({na+nb},{ncas_s}) "
            f"superset (occupations {np.round(no_occ, 3).tolist()}). Increase n_superset.")

    # Pin the orientation of degenerate active natural orbitals.
    f_emb = np.dot(mf.mo_coeff * mf.mo_energy, mf.mo_coeff.T)
    act_idx = np.where(is_act)[0]
    if len(act_idx) > 1:
        canonicalize_degenerate_active_nos(cas_no, act_idx, no_occ, f_emb, deg_tol)

    mo = C.copy()
    mo[:, lo:hi] = np.hstack([cas_no[:, is_core], cas_no[:, is_act], cas_no[:, ~is_core & ~is_act]])
    ncas = int(np.sum(is_act))
    nelecas = int(round(float(np.sum(no_occ[is_act]))))

    if sa_nstates > 1:
        spin = int(mf.mol.spin)
        na_sel, nb_sel = (nelecas + spin) // 2, (nelecas - spin) // 2
        if comb(ncas, na_sel) * comb(ncas, nb_sel) < sa_nstates:
            raise ValueError(
                f"Selected CAS({nelecas},{ncas}) supports fewer than "
                f"sa_nstates={sa_nstates} states, so a state-averaged solver cannot build that "
                f"many roots. The natorb density found little multireference character in this "
                f"window; widen n_superset or loosen occ_thresh."
            )

    log.info("natorb CAS from CASCI(%d,%d) superset -> CAS(%d,%d); active NO occupations %s."
             % (na + nb, ncas_s, nelecas, ncas, np.round(no_occ[is_act], 4).tolist()))
    return mo, ncas, nelecas


def project_amo_manually(old_mo_coeff, ncas, ncore, new_fock, norb, log=None):
    # Project old active MOs onto the new basis; fidelity ~1 means the CAS survived.
    log = _get_log(log)
    if old_mo_coeff.shape != (norb, norb):
        raise ValueError(
            f"Expected old_mo_coeff shape ({norb},{norb}), "
            f"got {old_mo_coeff.shape}")
    nocc = ncore + ncas

    old_amo = old_mo_coeff[:, ncore:nocc]

    proj = np.dot(old_amo, old_amo.T)
    evals, evecs = np.linalg.eigh(proj)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    new_amo = evecs[:, :ncas].copy()
    new_imo = evecs[:, ncas:].copy()
    fidelity = evals[:ncas].copy()

    # Align sign of each new AMO with the old to avoid CI vector phase flips.
    overlap = np.dot(new_amo.T, old_amo)
    for i in range(ncas):
        if overlap[i, i] < 0:
            new_amo[:, i] *= -1

    fock_imo = new_imo.T @ new_fock @ new_imo
    imo_evals, imo_evecs = np.linalg.eigh(fock_imo)
    new_imo = np.dot(new_imo, imo_evecs)
    new_cmo = new_imo[:, :ncore]
    new_vmo = new_imo[:, ncore:]

    new_mo = np.concatenate([new_cmo, new_amo, new_vmo], axis=1)

    log.info("Active-space projection fidelity: %s" % fidelity)
    return new_mo, fidelity
