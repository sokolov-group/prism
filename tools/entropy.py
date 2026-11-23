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

import itertools
import warnings
import numpy as np

def _entropy(rho):
    '''Von Neumann entropy of a density matrix or probability vector.'''
    w = np.linalg.eigvalsh(rho) if rho.ndim == 2 else rho
    w = w[w > 1e-14]
    return -np.sum(w * np.log(w))

def _strings(norb, nelec):
    '''Occupation strings of nelec electrons in norb orbitals, in PySCF CI order.'''
    strings = [sum(1 << k for k in occ) for occ in itertools.combinations(range(norb), nelec)]
    return np.array(sorted(strings), dtype = np.int64)

def _occupations(strings, norb):
    return (strings[:, None] >> np.arange(norb)) & 1

def _natural_orbitals(rdm1, orbsym = None):
    '''Natural occupations (descending) and the rotation to natural orbitals.

    With orbsym, the 1-RDM is diagonalized within each irrep.
    '''
    rdm1 = 0.5 * (rdm1 + rdm1.T)
    orbsym = np.zeros(len(rdm1), dtype = int) if orbsym is None else np.asarray(orbsym)
    noons = np.zeros(len(rdm1))
    u = np.zeros_like(rdm1)
    for irrep in np.unique(orbsym):
        idx = np.where(orbsym == irrep)[0]
        noons[idx], u[np.ix_(idx, idx)] = np.linalg.eigh(rdm1[np.ix_(idx, idx)])
    order = np.argsort(-noons, kind = 'stable')
    return noons[order], u[:, order]

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
    stra = _strings(norb, neleca)
    strb = _strings(norb, nelecb)
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

def analyze(interface, basis = 'casscf', basis_state = 1, ensemble = False, npairs = 10):
    '''Orbital entropies of the reference wavefunction.

    Prints the occupation n and single-orbital entropy s1 of every active orbital
    for each reference state, and the largest mutual information pairs.

    Args:
        interface : PRISM interface
            Interface with a CASCI, CASSCF or SA-CASSCF reference.

    Kwargs:
        basis : str
            Orbitals: 'casscf' (the CASSCF active orbitals) or 'natural'.
            Default is 'casscf'.
        basis_state : int or str
            Natural orbitals used when basis = 'natural': those of state k
            (numbered from 1) for every state, 'average' for those of the
            averaged density, or 'own' for each state's own. Default is 1.
        ensemble : bool
            Gives the entropies of the state-averaged ensemble (SA-CASSCF
            weights, otherwise equal weights) instead of each state.
            Default is False.
        npairs : int
            Number of mutual information pairs printed. Default is 10.

    Returns:
        A tuple (s1, s2, mi). The first axis runs over states, so s1 has shape
        (nstates, ncas). With ensemble = True there is no state axis.

    Examples:

    >>> interface = prism.interface.PYSCF(mf, mc)
    >>> s1, s2, mi = entropy.analyze(interface, basis = 'natural', basis_state = 1)
    '''
    log = interface.log

    if interface.ref_wfn is None:
        raise ValueError("Orbital entropies require a CASCI/CASSCF reference.")
    if hasattr(interface.mc.fcisolver, 'fcisolvers'):
        raise NotImplementedError("Entropies for state_average_mix references are not supported.")

    ci = list(interface.ref_wfn)
    nstates = len(ci)
    if not all(isinstance(c, np.ndarray) and c.ndim == 2 for c in ci):
        raise ValueError("The reference wavefunction must be an FCI vector.")
    if basis not in ('casscf', 'natural'):
        raise ValueError(f"Invalid basis: {basis}. Allowed bases are 'casscf' and 'natural'.")
    if basis_state not in ('average', 'own') and basis_state not in range(1, nstates + 1):
        raise ValueError(f"Invalid basis_state: {basis_state}. Valid values are 1 to {nstates}, 'average' or 'own'.")

    ncore, ncas = interface.ncore, interface.ncas
    nelecas = interface.ref_nelecas[0]
    weights = getattr(interface, 'weights', None)
    if weights is None or len(weights) != nstates:
        weights = np.ones(nstates) / nstates

    rdm1 = [interface.compute_rdm1(c, c, nelecas) for c in ci]
    rdm1_avg = sum(w * d for w, d in zip(weights, rdm1))

    # Rotation of the active orbitals for each state
    if basis == 'casscf':
        name = 'CASSCF active orbitals'
        u = [np.eye(ncas)] * nstates
    else:
        orbsym = getattr(interface.mc.mo_coeff, 'orbsym', None)
        if orbsym is not None:
            orbsym = orbsym[ncore:ncore + ncas]
        if basis_state == 'own' and not ensemble:
            name = 'natural orbitals of each state'
            dms = rdm1
        elif basis_state in ('own', 'average'):
            name = 'natural orbitals of the averaged density'
            dms = [rdm1_avg] * nstates
        else:
            name = 'natural orbitals of state %d' % basis_state
            dms = [rdm1[basis_state - 1]] * nstates
        noons, u = zip(*[_natural_orbitals(dm, orbsym) for dm in dms])
        if orbsym is None and any((np.abs(np.diff(n)) < 1e-6).any() for n in noons):
            log.warn("Degenerate natural orbitals make entropies depend on their rotation. "
                     "Use mol.symmetry = True for reproducible results.")
        ci = [interface.transform_ci_for_orbital_rotation(c, ncas, nelecas, v) for c, v in zip(ci, u)]

    if ensemble:
        labels = ['Ensemble']
        s1, s2 = orbital_entropies(ci, ncas, nelecas, weights)
        s1, s2 = s1[None], s2[None]
        occ = [np.diag(u[0].T @ rdm1_avg @ u[0])]
    else:
        labels = ['State %d' % (k + 1) for k in range(nstates)]
        s1 = np.zeros((nstates, ncas))
        s2 = np.zeros((nstates, ncas, ncas))
        for k in range(nstates):
            s1[k], s2[k] = orbital_entropies(ci[k], ncas, nelecas)
        occ = [np.diag(v.T @ d @ v) for v, d in zip(u, rdm1)]
    mi = np.array([mutual_information(a, b) for a, b in zip(s1, s2)])

    log.note("\nOrbital occupations n and entropies s1 (%s)" % name)
    log.note("  Orbital" + "".join("  %16s" % label for label in labels))
    log.note("         " + "".join("  %7s %8s" % ('n', 's1') for label in labels))
    for i in range(ncas):
        log.note("  %7d" % (ncore + i + 1) + "".join("  %7.4f %8.6f" % (n[i], s[i]) for n, s in zip(occ, s1)))

    p, q = np.triu_indices(ncas, 1)
    log.note("\nLargest mutual information, I(i,j) = (s1_i + s1_j - s2_ij) / 2")
    for label, m in zip(labels, mi):
        log.note("  %s\n  Orbital  Orbital          I" % label)
        for k in np.argsort(-m[p, q])[:npairs]:
            log.note("  %7d  %7d  %9.6f" % (ncore + p[k] + 1, ncore + q[k] + 1, m[p[k], q[k]]))

    if ensemble:
        return s1[0], s2[0], mi[0]
    return s1, s2, mi
