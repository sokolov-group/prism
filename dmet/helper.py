# Copyright 2026 Prism Developers. All Rights Reserved.
# Adapted from QC-DMET (Copyright 2015 Sebastian Wouters)
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


def orbital_entropy(occupations):
    # Single-orbital entropy in nats for a determinant at occupation n, p = n/2.
    p = np.clip(np.asarray(occupations, dtype=float) / 2.0, 0.0, 1.0)
    interior = (p > 0.0) & (p < 1.0)
    terms = np.zeros_like(p)
    terms[interior] = -(p[interior] * np.log(p[interior])
                        + (1.0 - p[interior]) * np.log(1.0 - p[interior]))
    return 2.0 * terms


def rhf_response(norb, num_pairs, h1_start, h1_row, h1_col, oei):
    # Idempotent-RDM response dD/du_k by first-order PT (NumPy port of the C rhf_response).
    evals, evecs = np.linalg.eigh(oei)
    occ, virt = evecs[:, :num_pairs], evecs[:, num_pairs:]
    denom = -1.0 / (evals[num_pairs:, None] - evals[None, :num_pairs])  # (nvir, nocc)
    nterms = len(h1_start) - 1
    rdm_deriv = np.empty((nterms, norb, norb))
    for t in range(nterms):
        sl = slice(h1_start[t], h1_start[t + 1])
        w1 = (np.dot(virt[h1_row[sl], :].T, occ[h1_col[sl], :])) * denom
        half = 2.0 * (virt @ w1 @ occ.T)
        rdm_deriv[t] = half + half.T
    return rdm_deriv


class DMETHelper:

    def __init__(self, ints, h1_terms, use_constrained_opt, min_func, log=None):
        self.ints = ints
        self.log = log if log is not None else logger.Logger(sys.stdout, logger.INFO)
        self._is_open_shell = (self.ints.nelec % 2 != 0)
        self.num_pairs = self.ints.nelec // 2

        self.num_alpha = (self.ints.nelec + self.ints.mol.spin) // 2
        self.num_beta = (self.ints.nelec - self.ints.mol.spin) // 2
        self.altcf = use_constrained_opt
        self.min_func = None

        if self.altcf:
            min_func_upper = min_func.upper() if min_func else min_func
            if min_func_upper not in ('OEI', 'FOCK_INIT'):
                raise ValueError(
                    f"Invalid min_func='{min_func}'. Must be 'OEI' or 'FOCK_INIT'.")
            self.min_func = min_func_upper

        self.h1_terms = h1_terms
        self.h1_start, self.h1_row, self.h1_col = self._convert_h1_sparse()
        self.bath_spectrum = None

    def _convert_h1_sparse(self):
        h1_start, h1_row, h1_col = [0], [], []
        total = 0
        for mat in self.h1_terms:
            rows, cols = np.where(mat == 1)
            total += len(rows)
            h1_start.append(total)
            h1_row.extend(rows)
            h1_col.extend(cols)
        return (np.array(h1_start, dtype=int),
                np.array(h1_row, dtype=int),
                np.array(h1_col, dtype=int))

    def construct_1rdm_loc(self, umat_loc):
        if self.altcf and self.min_func == 'OEI':
            oei = self.ints.loc_oei() + umat_loc
        else:
            oei = self.ints.loc_fock() + umat_loc
        return self._build_1rdm(oei, self.num_pairs)

    def construct_1rdm_response(self, umat_loc, no_rotation):
        if self._is_open_shell:
            raise NotImplementedError(
                "DMET chemical-potential optimization requires an even electron count.")
        oei = self.ints.loc_fock() + umat_loc
        if no_rotation is not None:
            oei = no_rotation.T @ oei @ no_rotation
        return rhf_response(self.ints.norb, self.num_pairs,
                            self.h1_start, self.h1_row, self.h1_col, oei)

    def _build_1rdm(self, oei, num_pairs):
        eigenvals, eigenvecs = np.linalg.eigh(oei)
        idx = eigenvals.argsort()

        if self._is_open_shell:
            dm_a = np.dot(eigenvecs[:, idx[:self.num_alpha]], eigenvecs[:, idx[:self.num_alpha]].T)
            dm_b = np.dot(eigenvecs[:, idx[:self.num_beta]], eigenvecs[:, idx[:self.num_beta]].T)
            return dm_a + dm_b
        else:
            return 2 * np.dot(eigenvecs[:, idx[:num_pairs]], eigenvecs[:, idx[:num_pairs]].T)

    def construct_bath(self, one_rdm, impurity_orbs, num_bath_orbs, threshold=1e-13,
                       keep_degenerate=False, deg_rtol=1e-6):
        embedding_orbs = np.array(1 - impurity_orbs, dtype=float)
        if embedding_orbs.ndim == 1:
            embedding_orbs = embedding_orbs[:, np.newaxis]  # (norb, 1)
        is_embedding = np.dot(embedding_orbs, embedding_orbs.T) == 1
        num_embed_orbs = int(np.sum(embedding_orbs))
        embedding_rdm1 = np.reshape(one_rdm[is_embedding], (num_embed_orbs, num_embed_orbs))

        num_imp_orbs = int(np.sum(impurity_orbs))
        num_total_orbs = len(impurity_orbs)

        eigenvals, eigenvecs = np.linalg.eigh(embedding_rdm1)
        # Rank by distance from 0 or 2, breaking ties on occupation to keep symmetry
        # partners adjacent.
        idx = np.lexsort((eigenvals, np.maximum(-eigenvals, eigenvals - 2.0)))
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        # Distance from 0 or 2, largest first; orbitals at 0 or 2 decouple from the impurity.
        occ_deviation = np.minimum(eigenvals, 2.0 - eigenvals)
        to_keep = int(np.sum(occ_deviation > threshold))

        self.log.info("Bath: %d entangled environment orbitals, %d requested."
                      % (to_keep, num_bath_orbs))
        if to_keep < num_bath_orbs:
            self.log.info("Throwing out %d orbitals within %s of 0 or 2."
                          % (num_bath_orbs - to_keep, threshold))
        requested = num_bath_orbs
        num_bath_orbs = min(to_keep, num_bath_orbs)

        # Symmetry partners share an occupation and enter the bath as a set.
        # The window is the same near 0 and near 2.
        if keep_degenerate and 0 < num_bath_orbs < to_keep:
            while (num_bath_orbs < to_keep and
                   abs(eigenvals[num_bath_orbs] - eigenvals[num_bath_orbs - 1])
                   <= deg_rtol * max(abs(eigenvals[num_bath_orbs - 1]),
                                     abs(2.0 - eigenvals[num_bath_orbs - 1]))):
                num_bath_orbs += 1
            if num_bath_orbs > requested:
                self.log.info("Bath extended to %d to complete a degenerate set."
                              % num_bath_orbs)

        if to_keep > num_bath_orbs:
            msg = ("Bath capped at %d; %d entangled orbitals discarded."
                   % (num_bath_orbs, to_keep - num_bath_orbs))
            if not keep_degenerate:
                msg += (" Degenerate partners may be split; set keep_degenerate=True "
                        "or raise n_bath_orbs.")
            self.log.warn(msg)

        entropy = orbital_entropy(eigenvals)
        if 0 < num_bath_orbs < len(occ_deviation):
            self.log.info("At the cut: last kept dev %.3e (S %.3e), "
                          "first discarded dev %.3e (S %.3e)."
                          % (occ_deviation[num_bath_orbs - 1], entropy[num_bath_orbs - 1],
                             occ_deviation[num_bath_orbs], entropy[num_bath_orbs]))

        self.bath_spectrum = {
            'occupation': eigenvals.copy(),
            'occ_deviation': occ_deviation.copy(),
            'entropy': entropy,
            'num_bath_orbs': num_bath_orbs,
            'num_entangled': to_keep,
        }

        pure_env_vals = -eigenvals[num_bath_orbs:]
        pure_env_vecs = eigenvecs[:, num_bath_orbs:]
        env_idx = pure_env_vals.argsort()
        eigenvecs[:, num_bath_orbs:] = pure_env_vecs[:, env_idx]
        pure_env_vals = -pure_env_vals[env_idx]
        core_occupations = np.hstack((np.zeros((num_imp_orbs + num_bath_orbs,)), pure_env_vals))

        for col_idx in range(num_imp_orbs):
            eigenvecs = np.insert(eigenvecs, col_idx, 0.0, axis=1)
        imp_slot = 0
        for orb_idx in range(num_total_orbs):
            if impurity_orbs[orb_idx]:
                eigenvecs = np.insert(eigenvecs, orb_idx, 0.0, axis=0)
                eigenvecs[orb_idx, imp_slot] = 1.0
                imp_slot += 1
        if imp_slot != num_imp_orbs:
            raise RuntimeError("Impurity orbital count mismatch.")

        if np.linalg.norm(np.dot(eigenvecs.T, eigenvecs) - np.identity(num_total_orbs)) >= 1e-12:
            raise RuntimeError("Embedding orbitals are not orthonormal.")
        return num_bath_orbs, eigenvecs, core_occupations
