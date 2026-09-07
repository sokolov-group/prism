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

from contextlib import nullcontext

import numpy as np
from pyscf import fci

from prism.dmet.utils import silent_stdout


def solve(const, oei, fock, tei, norb, nel, nimp, chempot_imp=0.0, printoutput=False):

    fock_copy = fock.copy()
    if chempot_imp != 0.0:
        for orb in range(nimp):
            fock_copy[orb, orb] -= chempot_imp

    if nel % 2 == 0:
        fci_nel = nel
        cisolver = fci.direct_spin0.FCI()
    else:
        fci_nel = ((nel + 1) // 2, nel // 2)
        cisolver = fci.direct_spin1.FCI()

    with silent_stdout() if not printoutput else nullcontext():
        cisolver.verbose = 0
        cisolver.max_cycle = 200
        cisolver.conv_tol = 1e-12
        _, fci_vector = cisolver.kernel(fock_copy, tei, norb, fci_nel, ecore=const)
        two_rdm = cisolver.make_rdm2(fci_vector, norb, fci_nel)

    one_rdm = np.einsum('ijkk->ij', two_rdm) / (nel - 1)

    # FCI uses 0.5/0.5 vs CASSCF/NEVPT2 0.25/0.125; identical only for exact wfns.
    impurity_energy = const
    impurity_energy += 0.5 * np.einsum('ij,ij->', one_rdm[:nimp, :], oei[:nimp, :] + fock[:nimp, :])
    impurity_energy += 0.5 * np.einsum('ijkl,ijkl->', two_rdm[:nimp, :, :, :], tei[:nimp, :, :, :])
    return impurity_energy, one_rdm


def execute(task):
    return solve(
        task['const'],
        task['dmet_oei'],
        task['dmet_fock'],
        task['dmet_tei'],
        task['norb'],
        task['nel'],
        task['nimp'],
        task.get('chempot_imp', 0.0),
    )
