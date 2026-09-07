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

import numpy as np
import scipy
import scipy.linalg

import prism.lib.logger as logger


def construct_p_list(mol, pmol):
    # 1 = AO in the pmol reference, 0 = complement space; pmol drops zero-width atoms.
    kept_atoms = [ia for ia, (_, _, ao_start, ao_stop) in enumerate(mol.aoslice_by_atom())
                  if ao_stop > ao_start]
    ref = set()
    for atom_id, _, nl, m in pmol.ao_labels(fmt=None):
        ref.add((kept_atoms[atom_id], nl, m))
    p_list = np.zeros((mol.nao_nr(),), dtype=int)
    for i, (atom_id, _, nl, m) in enumerate(mol.ao_labels(fmt=None)):
        if (atom_id, nl, m) in ref:
            p_list[i] = 1
    if np.sum(p_list) != pmol.nao_nr():
        raise RuntimeError(
            f"Reference-basis orbital count mismatch: matched {int(np.sum(p_list))} "
            f"of {pmol.nao_nr()} reference functions.")
    return p_list


def orthogonalize_iao(coeff, ovlp):
    # Knizia, JCTC 9, 4834-4843, 2013 -- appendix C, third equation
    eigs, vecs = scipy.linalg.eigh(coeff.T @ ovlp @ coeff)
    coeff = coeff @ vecs @ np.diag(np.power(eigs, -0.5)) @ vecs.T
    return coeff


def _build_pmol_with_ghosts(mol, minao=None):
    # Like reference_mol() but keeps ghost vacancy atoms; GTH basis for pseudo mols.
    import pyscf.gto
    if minao is None:
        minao = 'gth-szv-molopt-sr' if getattr(mol, 'pseudo', None) else 'minao'
    aoslice = mol.aoslice_by_atom()
    pmol = pyscf.gto.Mole()
    pmol.unit = 'Bohr'
    pmol.atom = [mol._atom[ia] for ia, (_, _, s, e) in enumerate(aoslice) if e > s]
    pmol.pseudo = getattr(mol, 'pseudo', None)
    pmol.ecp = {}
    pmol.spin = 0
    pmol.charge = 0
    pmol.verbose = 0
    pmol.build(dump_input=False, parse_arg=False, basis=minao)
    return pmol


def _iao_with_pmol(mol, ao2occ, pmol):
    # IAO construction (Knizia JCTC 2013) with caller-supplied reference mol.
    from pyscf import gto, scf
    from pyscf.lo.orth import vec_lowdin

    s1 = mol.intor_symmetric('int1e_ovlp')
    s2 = pmol.intor_symmetric('int1e_ovlp')
    s12 = gto.mole.intor_cross('int1e_ovlp', mol, pmol)

    s2cd = scipy.linalg.cho_factor(s2)
    ctild = scipy.linalg.cho_solve(s2cd, np.dot(s12.T, ao2occ))
    try:
        s1cd = scipy.linalg.cho_factor(s1)
        p12 = scipy.linalg.cho_solve(s1cd, s12)
        ctild = scipy.linalg.cho_solve(s1cd, np.dot(s12, ctild))
    except np.linalg.LinAlgError:
        x = scf.addons.canonical_orth_(s1, 1e-8)
        p12 = x @ x.T @ s12
        ctild = np.dot(p12, ctild)

    ctild = vec_lowdin(ctild, s1)
    ccs1 = ao2occ @ ao2occ.T @ s1
    ccs2 = ctild @ ctild.T @ s1
    return p12 + 2 * (ccs1 @ ccs2 @ p12) - np.dot(ccs1, p12) - np.dot(ccs2, p12)


def resort_orbitals(mol, ao2loc):
    # Sort the orbitals according to the atom list.
    norb = mol.nao_nr()
    coords = np.zeros((norb, 3), dtype=float)
    rvec = mol.intor('int1e_r', comp=3)
    for cart in range(3):
        coords[:, cart] = np.diag(ao2loc.T @ rvec[cart] @ ao2loc)
    atomid = np.zeros((norb,), dtype=int)
    for orb in range(norb):
        min_id = 0
        min_distance = np.linalg.norm(coords[orb, :] - mol.atom_coord(0))
        for atom in range(1, mol.natm):
            current_distance = np.linalg.norm(coords[orb, :] - mol.atom_coord(atom))
            if current_distance < min_distance:
                min_distance = current_distance
                min_id = atom
        atomid[orb] = min_id
    resort = []
    for atom in range(0, mol.natm):
        for orb in range(norb):
            if atomid[orb] == atom:
                resort.append(orb)
    resort = np.array(resort)
    ao2loc = ao2loc[:, resort]
    return ao2loc


def construct_iao(mol, mf):
    # Knizia, JCTC 9, 4834-4843, 2013 -- appendix C
    # UKS/UHF: spin-average the density, take its occupied natural orbitals.
    if np.ndim(mf.mo_coeff) == 3:
        mo_a, mo_b = mf.mo_coeff[0], mf.mo_coeff[1]
        occ_a, occ_b = mf.mo_occ[0], mf.mo_occ[1]
        dm_a = np.dot(mo_a[:, occ_a > 0.5], mo_a[:, occ_a > 0.5].T)
        dm_b = np.dot(mo_b[:, occ_b > 0.5], mo_b[:, occ_b > 0.5].T)
        DM1 = 0.5 * (dm_a + dm_b)
        eigs, vecs = np.linalg.eigh(DM1)
        ao2occ = vecs[:, eigs > 0.5]
    else:
        ao2occ = mf.mo_coeff[:, mf.mo_occ > 0.5]

    pmol = _build_pmol_with_ghosts(mol)
    S1 = mol.intor_symmetric('int1e_ovlp')
    ao2iao = _iao_with_pmol(mol, ao2occ, pmol)
    ao2iao = orthogonalize_iao(ao2iao, S1)
    return ao2iao, S1, pmol


def _log_iao_check(mol, pmol, ao2loc, S1):
    log = logger.Logger(mol.stdout, mol.verbose)
    should_be_1 = ao2loc.T @ S1 @ ao2loc
    dev = np.linalg.norm(should_be_1 - np.eye(should_be_1.shape[0]))
    log.info("IAO reference basis: %d orbitals (pmol), %d orbitals (mol)" % (pmol.nao_nr(), mol.nao_nr()))
    log.info("IAO orthonormality deviation |I - C.T S C|: %e" % dev)


def localize_iao(mol, mf):
    norb = mol.nao_nr()
    ao2iao, S1, pmol = construct_iao(mol, mf)
    num_iao = ao2iao.shape[1]

    p_list = construct_p_list(mol, pmol)

    # No complement needed when every AO is in the reference basis.
    if np.all(p_list == 1):
        ao2loc = orthogonalize_iao(ao2iao, S1)
        _log_iao_check(mol, pmol, ao2loc, S1)
        return ao2loc

    # Determine the complement of the IAO space.
    DM_iao = np.dot(ao2iao, ao2iao.T)
    mx = S1 @ DM_iao @ S1
    eigs, vecs = scipy.linalg.eigh(a=mx, b=S1)  # Small to large in scipy
    ao2com = vecs[:, :norb - num_iao]

    # Redo the IAO construction for the complement space.
    S31 = S1[p_list == 0, :]
    S3 = S31[:, p_list == 0]
    X = np.linalg.solve(S3, np.dot(S31, ao2com))
    P13 = np.linalg.solve(S1, S31.T)
    Cp = np.dot(P13, X)
    Cp = orthogonalize_iao(Cp, S1)
    DM1 = np.dot(ao2com, ao2com.T)
    DM3 = np.dot(Cp, Cp.T)
    A = 2 * (DM1 @ S1 @ DM3 @ S31.T) + P13 - np.dot(DM1 + DM3, S31.T)
    ao2com = orthogonalize_iao(A, S1)
    ao2loc = np.hstack((ao2iao, ao2com))
    ao2loc = resort_orbitals(mol, ao2loc)
    ao2loc = orthogonalize_iao(ao2loc, S1)

    _log_iao_check(mol, pmol, ao2loc, S1)
    return ao2loc
