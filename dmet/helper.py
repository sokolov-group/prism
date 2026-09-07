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


def rhf_response(norb, numPairs, H1start, H1row, H1col, oei):
    # Idempotent-RDM response dD/du_k by first-order perturbation theory over
    # occupied-virtual pairs (NumPy port of the QC-DMET rhf_response C kernel).
    evals, evecs = np.linalg.eigh(oei)
    occ, virt = evecs[:, :numPairs], evecs[:, numPairs:]
    denom = -1.0 / (evals[numPairs:, None] - evals[None, :numPairs])  # (nvir, nocc)
    nterms = len(H1start) - 1
    rdm_deriv = np.empty((nterms, norb, norb))
    for t in range(nterms):
        sl = slice(H1start[t], H1start[t + 1])
        w1 = (virt[H1row[sl], :].T @ occ[H1col[sl], :]) * denom
        half = 2.0 * virt @ w1 @ occ.T
        rdm_deriv[t] = half + half.T
    return rdm_deriv


class DMETHelper:

    def __init__(self, locints, list_H1, use_constrained_opt, minFunc, log=None):
        self.locints  = locints
        self.log      = log if log is not None else logger.Logger(sys.stdout, logger.INFO)
        self._is_open_shell = (self.locints.Nelec % 2 != 0)
        self.numPairs = self.locints.Nelec // 2

        self.num_alpha = (self.locints.Nelec + self.locints.mol.spin) // 2
        self.num_beta  = (self.locints.Nelec - self.locints.mol.spin) // 2
        self.altcf     = use_constrained_opt
        self.minFunc   = None

        if self.altcf:
            _mf = minFunc.upper() if minFunc else minFunc
            assert _mf in ('OEI', 'FOCK_INIT'), \
                f"DMETHelper: minFunc must be 'OEI' or 'FOCK_INIT', got '{minFunc}'"
            self.minFunc = _mf

        self.list_H1 = list_H1
        self.H1start, self.H1row, self.H1col = self._convert_H1_sparse()
        self.Nterms  = len(self.H1start) - 1

    def _convert_H1_sparse(self):
        H1start, H1row, H1col = [0], [], []
        total = 0
        for mat in self.list_H1:
            rows, cols = np.where(mat == 1)
            total += len(rows)
            H1start.append(total)
            H1row.extend(rows)
            H1col.extend(cols)
        return (np.array(H1start, dtype=int),
                np.array(H1row,   dtype=int),
                np.array(H1col,   dtype=int))

    def construct1RDM_loc(self, umat_loc):
        if self.altcf and self.minFunc == 'OEI':
            oei = self.locints.loc_oei() + umat_loc
        else:
            oei = self.locints.loc_fock() + umat_loc
        return self._build_1rdm(oei, self.numPairs)

    def construct1RDM_response(self, umat_loc, NOrotation):
        if self._is_open_shell:
            raise NotImplementedError(
                "DMET chemical-potential optimization requires an even electron count.")
        oei = self.locints.loc_fock() + umat_loc
        if NOrotation is not None:
            oei = np.dot(np.dot(NOrotation.T, oei), NOrotation)
        return rhf_response(self.locints.Norbs, self.numPairs,
                            self.H1start, self.H1row, self.H1col, oei)

    def _build_1rdm(self, oei, numPairs):
        eigenvals, eigenvecs = np.linalg.eigh(oei)
        idx = eigenvals.argsort()

        if self._is_open_shell:
            dm_a = np.dot(eigenvecs[:, idx[:self.num_alpha]], eigenvecs[:, idx[:self.num_alpha]].T)
            dm_b = np.dot(eigenvecs[:, idx[:self.num_beta]],  eigenvecs[:, idx[:self.num_beta]].T)
            return dm_a + dm_b
        else:
            return 2 * np.dot(eigenvecs[:, idx[:numPairs]], eigenvecs[:, idx[:numPairs]].T)

    def constructbath(self, OneDM, impurity_orbs, numBathOrbs, threshold=1e-13):
        embeddingOrbs = np.array(1 - impurity_orbs, dtype=float)
        if embeddingOrbs.ndim == 1:
            embeddingOrbs = embeddingOrbs[:, np.newaxis]  # (Norbs, 1)
        isEmbedding   = np.dot(embeddingOrbs, embeddingOrbs.T) == 1
        numEmbedOrbs  = int(np.sum(embeddingOrbs))
        embedding1RDM = np.reshape(OneDM[isEmbedding], (numEmbedOrbs, numEmbedOrbs))

        num_imp_orbs = int(np.sum(impurity_orbs))
        numTotalOrbs = len(impurity_orbs)

        eigenvals, eigenvecs = np.linalg.eigh(embedding1RDM)
        idx    = np.maximum(-eigenvals, eigenvals - 2.0).argsort()
        tokeep = np.sum(-np.maximum(-eigenvals, eigenvals - 2.0)[idx] > threshold)
        if tokeep < numBathOrbs:
            self.log.info("DMET::constructbath : Throwing out %d orbitals within %s of 0 or 2."
                          % (numBathOrbs - tokeep, threshold))
        numBathOrbs = min(int(tokeep), numBathOrbs)

        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        pureEnvVals = -eigenvals[numBathOrbs:]
        pureEnvVecs = eigenvecs[:, numBathOrbs:]
        env_idx = pureEnvVals.argsort()
        eigenvecs[:, numBathOrbs:] = pureEnvVecs[:, env_idx]
        pureEnvVals = -pureEnvVals[env_idx]
        coreOccupations = np.hstack((np.zeros([num_imp_orbs + numBathOrbs]), pureEnvVals))

        for counter in range(num_imp_orbs):
            eigenvecs = np.insert(eigenvecs, counter, 0.0, axis=1)
        counter = 0
        for counter2 in range(numTotalOrbs):
            if impurity_orbs[counter2]:
                eigenvecs = np.insert(eigenvecs, counter2, 0.0, axis=0)
                eigenvecs[counter2, counter] = 1.0
                counter += 1
        assert counter == num_imp_orbs

        assert np.linalg.norm(np.dot(eigenvecs.T, eigenvecs) - np.identity(numTotalOrbs)) < 1e-12
        return numBathOrbs, eigenvecs, coreOccupations
