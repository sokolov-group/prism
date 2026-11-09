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
# Authors: Bryce Pickett <brycepickett22@outlook.com>
#

import warnings
import numpy as np
from pyscf.fci import cistring

def _entropy(rho):
    '''Von Neumann entropy of a density matrix or probability vector.'''
    w = np.linalg.eigvalsh(rho) if rho.ndim == 2 else rho
    w = w[w > 1e-14]
    return -np.sum(w * np.log(w))

def _occupations(strings, norb):
    return (np.asarray(strings, dtype=np.int64)[:, None] >> np.arange(norb)) & 1

def orbital_entropies(ci, norb, nelec, weights = None):
    '''
    Single- and two-orbital entropies (s1, s2) of a CI vector.

    A list of CI vectors is an ensemble with the given weights (default equal).
    The diagonal of s2 is s1.
    '''
    if isinstance(ci, np.ndarray) and ci.ndim == 2:
        ci = [ci]
    ci = [np.asarray(c) for c in ci]
    if weights is None:
        weights = np.ones(len(ci)) / len(ci)

    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    stra = cistring.make_strings(range(norb), neleca)
    strb = cistring.make_strings(range(norb), nelecb)
    occa = _occupations(stra, norb)
    occb = _occupations(strb, norb)

    # Electrons below each orbital
    below_a = np.cumsum(occa, axis = 1) - occa
    below_b = np.cumsum(occb, axis = 1) - occb

    # One-orbital RDM (diagonal)
    s1 = np.zeros(norb)
    for i in range(norb):
        code = (occa[:, i][:, None] + 2 * occb[:, i][None, :]).ravel()
        p = np.zeros(4)
        for c, w in zip(ci, weights):
            p += w * np.bincount(code, weights = (c ** 2).ravel(), minlength = 4)
        s1[i] = _entropy(p)

    # Two-orbital RDM, grouping determinants by occupation outside i and j
    s2 = np.zeros((norb, norb))
    for i in range(norb):
        for j in range(i + 1, norb):
            mask = ~((1 << i) | (1 << j))

            # Fermionic sign
            exp_a = occa[:, i] * below_a[:, i] + occa[:, j] * (below_a[:, j] - occa[:, i])
            exp_b = occb[:, i] * below_b[:, i] + occb[:, j] * (below_b[:, j] - occb[:, i])
            sign = np.outer(1 - 2 * (exp_a % 2), 1 - 2 * (exp_b % 2))

            _, rest_a = np.unique(stra & mask, return_inverse = True)
            rest_b_unique, rest_b = np.unique(strb & mask, return_inverse = True)
            group = (rest_a[:, None] * len(rest_b_unique) + rest_b[None, :]).ravel()
            local = ((occa[:, i] + 2 * occa[:, j])[:, None]
                     + 4 * (occb[:, i] + 2 * occb[:, j])[None, :]).ravel()

            rho = np.zeros((16, 16))
            for c, w in zip(ci, weights):
                amp = np.zeros((group.max() + 1, 16))
                amp[group, local] = (c * sign).ravel()
                rho += w * (amp.T @ amp)
            s2[i, j] = s2[j, i] = _entropy(rho)

    np.fill_diagonal(s2, s1)
    return s1, s2

def orbital_entropies_from_rdm(rdm1s, rdm2ab):
    '''
    Single-orbital entropies from spin-resolved RDMs.

    rdm2ab follows PySCF: rdm2ab[p,q,r,s] = <p_a^+ r_b^+ s_b q_a>.
    '''
    na = np.diag(rdm1s[0]).real
    nb = np.diag(rdm1s[1]).real
    d = np.einsum('iiii->i', rdm2ab).real

    p = np.stack([1.0 - na - nb + d, na - d, nb - d, d], axis = 1)
    if p.min() < -1e-8:
        warnings.warn("Negative occupation probabilities (min %.2e) set to zero; "
                      "RDMs are not N-representable." % p.min())
    return np.array([_entropy(p_i) for p_i in p])

def mutual_information(s1, s2):
    '''Mutual information I(i,j) = (s1_i + s1_j - s2_ij) / 2.'''
    imat = 0.5 * (s1[:, None] + s1[None, :] - s2)
    np.fill_diagonal(imat, 0.0)
    return imat
