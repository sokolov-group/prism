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

'''
Ranked-orbital active space selection
D. S. King and L. Gagliardi, J. Chem. Theory Comput. 17, 2817 (2021)
'''

import numpy as np
from math import comb
from pyscf.fci import direct_spin1
from pyscf.fci.addons import transform_ci_for_orbital_rotation
from pyscf.mcscf import apc
from prism.tools import entropy

CRITERIA = ('noon', 'entropy', 'mi', 'apc')

def get_ncsf(nelec, norb):
    '''Number of CSFs of an active space at the lowest Sz (Eq. 2 of the reference above).

    Args:
        nelec : int
            Number of active electrons.
        norb : int
            Number of active orbitals.

    Returns:
        Number of CSFs (int).

    Examples:

    >>> get_ncsf(8, 8)
    1764
    '''
    na = (nelec + 1) // 2
    nb = nelec // 2
    return comb(norb, na) * comb(norb, nb) - comb(norb, na + 1) * (comb(norb, nb - 1) if nb > 0 else 0)

def _occupations(noons, nelec, two_s):
    '''Integer occupations assigned by descending natural occupation.

    The (nelec - two_s) // 2 orbitals with the largest occupations get 2, the
    next two_s get 1 and the rest get 0.

    Args:
        noons : 1D array
            Natural orbital occupation numbers.
        nelec : int
            Total number of electrons.
        two_s : int
            Number of unpaired electrons (2S).

    Returns:
        1D array of occupations (2, 1 or 0).
    '''
    nb = (nelec - two_s) // 2
    order = np.argsort(-noons, kind = 'stable')
    occ = np.zeros(len(noons))
    occ[order[:nb]] = 2
    occ[order[nb:nb + two_s]] = 1
    return occ

def _natural_orbitals(rdm1, mo):
    '''Natural orbitals of a 1-RDM.

    Args:
        rdm1 : 2D array
            1-RDM in the basis of mo.
        mo : 2D array
            Orbital coefficients (AO x MO).

    Returns:
        A tuple (noons, u, no): the natural occupations in descending order,
        the rotation from mo to the natural orbitals, and the natural
        orbitals mo @ u.
    '''
    noons, u = np.linalg.eigh(0.5 * (rdm1 + rdm1.T))
    idx = np.argsort(-noons)
    return noons[idx], u[:, idx], mo @ u[:, idx]

def _rank_reference(interface, criterion):
    '''Ranks the natural orbitals of the reference active space.

    Args:
        interface : PRISM interface
            Interface with a CASCI/CASSCF reference.
        criterion : str
            'noon', 'entropy' or 'mi'.

    Returns:
        A tuple (mo, noons, importance, candidates): the orbitals with the
        active block rotated to natural orbitals, the occupation and importance
        of every orbital, and the indices of the active orbitals.
    '''
    ci, weights = entropy.reference_wfn(interface)
    cis = ci if isinstance(ci, list) else [ci]
    weights = weights if weights is not None else [1.0]
    ncore, ncas = interface.ncore, interface.ncas
    nelecas = interface.ref_nelecas[0]
    active = slice(ncore, ncore + ncas)

    casdm1 = sum(w * direct_spin1.make_rdm1(c, ncas, nelecas) for w, c in zip(weights, cis))
    noons_cas, u, mo_cas = _natural_orbitals(casdm1, interface.mo[:, active])

    mo = interface.mo.copy()
    mo[:, active] = mo_cas
    noons = np.zeros(mo.shape[1])
    noons[:ncore] = 2.0
    noons[active] = noons_cas

    if criterion == 'noon':
        score = np.minimum(noons_cas, 2.0 - noons_cas)
    else:
        if not interface.mol.symmetry and (np.abs(np.diff(noons_cas)) < 1e-6).any():
            interface.log.warn("Degenerate natural orbitals make entropy rankings depend on their rotation. "
                               "Use mol.symmetry = True for reproducible rankings.")
        ci_no = [transform_ci_for_orbital_rotation(c, ncas, nelecas, u) for c in cis]
        s1, s2 = entropy.orbital_entropies(ci_no, ncas, nelecas, weights)
        score = s1 if criterion == 'entropy' else entropy.mutual_information(s1, s2).sum(axis = 1)

    importance = np.zeros(mo.shape[1])
    importance[active] = score
    return mo, noons, importance, np.arange(ncore, ncore + ncas)

def _rank_apc(interface):
    '''Ranks the SCF orbitals by approximate pair coefficient (APC) entropies.

    Args:
        interface : PRISM interface

    Returns:
        A tuple (mo, noons, importance, candidates) as in _rank_reference.
        All SCF orbitals are candidates.
    '''
    mf = interface.mf
    importance = apc.APC(mf, verbose = interface.verbose)._calc_apc_entropies(mf)
    return mf.mo_coeff, np.asarray(mf.mo_occ, dtype = float), importance, np.arange(mf.mo_coeff.shape[1])

def _rank_correlated(method, criterion, state):
    '''Ranks the natural orbitals of a correlated 1-RDM.

    Args:
        method : NEVPT or QDNEVPT object
            Run with rdm_order = 2.
        criterion : str
            Must be 'noon'.
        state : int
            State of the 1-RDM, numbered from 1.

    Returns:
        A tuple (mo, noons, importance, candidates) as in _rank_reference.
        All orbitals are candidates.
    '''
    if criterion != 'noon':
        raise ValueError("Correlated methods provide a 1-RDM only; use criterion = 'noon'.")
    if getattr(method, 'rdm_order', 2) != 2:
        raise ValueError("The 1-RDM is uncorrelated. Set method.rdm_order = 2 before selection.")

    rdm1 = method.make_rdm1(L = state, R = state)
    noons, _, mo = _natural_orbitals(rdm1, method.mo)
    importance = np.minimum(noons, 2.0 - noons)
    return mo, noons, importance, np.arange(mo.shape[1])

def select(obj, max_size, criterion = 'noon', state = 1):
    '''Selects an active space with the ranked-orbital approach.

    All candidate orbitals start in the active space. The least important
    orbital is dropped until the active space is no larger than max_size.

    Args:
        obj : PRISM interface, or NEVPT/QDNEVPT object after kernel()
            An interface ranks the natural orbitals of its reference active
            space ('apc' ranks the SCF orbitals). A NEVPT/QDNEVPT object ranks
            the natural orbitals of its correlated 1-RDM.
        max_size : tuple or int
            (nelecas, ncas) of the largest allowed active space, compared by
            number of CSFs, or the largest number of active orbitals.

    Kwargs:
        criterion : str
            Orbital importance: 'noon' (min(n, 2 - n)), 'entropy' (s1),
            'mi' (sum_j I(i,j)) or 'apc' (APC entropies). Default is 'noon'.
        state : int
            State of the correlated 1-RDM, numbered from 1. Default is 1.

    Returns:
        A tuple (ncas, nelecas, mo): the number of active orbitals, the active
        electrons (nalpha, nbeta), and the orbitals ordered core | active | external.

    Examples:

    >>> interface = prism.interface.PYSCF(mf, mc)
    >>> ncas, nelecas, mo = active_space.select(interface, (8, 8), 'entropy')
    >>> mc = pyscf.mcscf.CASSCF(mf, ncas, nelecas).run(mo)
    '''
    if criterion not in CRITERIA:
        raise ValueError(f"Invalid criterion: {criterion}. Allowed criteria are {CRITERIA}.")

    if hasattr(obj, 'interface'):
        interface = obj.interface
        mo, noons, importance, candidates = _rank_correlated(obj, criterion, state)
    elif criterion == 'apc':
        interface = obj
        mo, noons, importance, candidates = _rank_apc(obj)
    else:
        interface = obj
        mo, noons, importance, candidates = _rank_reference(obj, criterion)
    log = interface.log

    nmo = mo.shape[1]
    na, nb = interface.ref_nelecas[0]
    two_s = na - nb
    occ = _occupations(noons, interface.nelec, two_s)
    if isinstance(max_size, (list, tuple, np.ndarray)):
        max_size = tuple(int(x) for x in max_size)

    # Candidate set within the size limit
    nelec_cand = int(occ[candidates].sum())
    if isinstance(max_size, tuple):
        fits = get_ncsf(nelec_cand, len(candidates)) <= get_ncsf(*max_size)
    else:
        fits = len(candidates) <= max_size

    if fits:
        active = candidates
    else:
        # Non-candidates rank below every candidate
        importance_all = importance.copy()
        other = np.setdiff1d(np.arange(nmo), candidates)
        importance_all[other] = importance[candidates].min() - 1.0
        chooser = apc.Chooser(mo, occ, importance_all, max_size, verbose = interface.verbose)
        active = np.sort(np.asarray(chooser.kernel()[3]))
        if not np.isin(active, candidates).all():
            raise RuntimeError("The selected active space contains non-candidate orbitals.")

    rest = np.setdiff1d(np.arange(nmo), active)
    if (occ[rest] == 1).any():
        raise ValueError("A singly occupied orbital lies outside the selected active space.")
    core = rest[occ[rest] == 2]
    external = rest[occ[rest] == 0]

    ncas = len(active)
    nelec = int(occ[active].sum())
    nelecas = ((nelec + two_s) // 2, (nelec - two_s) // 2)

    log.note("\nRanked-orbital active space selection (criterion = %s)" % criterion)
    log.note("  Orbital  Occupation  Importance")
    for k, p in enumerate(active):
        log.note("  %7d  %10.6f  %10.6f" % (len(core) + k + 1, noons[p], importance[p]))
    if isinstance(max_size, tuple):
        log.note("Selected active space: (%d, %d), %d CSFs (maximum %d)"
                 % (nelec, ncas, get_ncsf(nelec, ncas), get_ncsf(*max_size)))
    else:
        log.note("Selected active space: (%d, %d) (maximum %d orbitals)" % (nelec, ncas, max_size))

    mo_out = np.hstack((mo[:, core], mo[:, active], mo[:, external]))
    return ncas, nelecas, mo_out
