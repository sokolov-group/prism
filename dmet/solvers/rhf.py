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
from pyscf import ao2mo, gto, scf


def solve(const, oei, fock, tei, norb, nel, nimp, dm_guess_rhf, chempot_imp=0.0):

    fock_copy = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy[orb, orb] -= chempot_imp

    mol = gto.Mole()
    mol.build(verbose=0)
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = nel
    mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: fock_copy
    mf.get_ovlp = lambda *args: np.eye(norb)
    mf._eri = ao2mo.restore(8, tei, norb)
    mf.scf(dm_guess_rhf)
    dm_loc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
    if not mf.converged:
        mf.max_cycle = 300
        mf.diis_space = 12
        mf.scf(dm_loc)

    RDM1 = mf.make_rdm1()
    JK   = mf.get_veff(None, dm=RDM1)

    # Half-projector: 0.5*(oei + fock) avoids double-counting JK.
    impurity_energy = const \
        + 0.25 * np.einsum('ji,ij->', RDM1[:, :nimp], fock[:nimp, :] + oei[:nimp, :]) \
        + 0.25 * np.einsum('ji,ij->', RDM1[:nimp, :], fock[:, :nimp] + oei[:, :nimp]) \
        + 0.25 * np.einsum('ji,ij->', RDM1[:, :nimp], JK[:nimp, :]) \
        + 0.25 * np.einsum('ji,ij->', RDM1[:nimp, :], JK[:, :nimp])

    return (impurity_energy, RDM1)


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
        task.get('chempot_imp', 0.0),
    )
