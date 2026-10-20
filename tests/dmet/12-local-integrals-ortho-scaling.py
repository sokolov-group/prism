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

import unittest
import numpy as np
import pyscf.gto
import pyscf.scf
from prism.dmet import LocalIntegrals

np.set_printoptions(linewidth=150, suppress=True)


def chain(n, spacing=1.0):
    mol = pyscf.gto.Mole()
    mol.atom = [['H', (spacing * i, 0.0, 0.0)] for i in range(n)]
    mol.basis = 'sto-3g'
    mol.build(verbose=0)
    return mol


def ortho(mol):
    mf = pyscf.scf.RHF(mol)
    mf.verbose = 0
    mf.scf()
    ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'lowdin')
    return ints, ints.loc_ortho()


class KnownValues(unittest.TestCase):

    def test_ortho_is_the_largest_pair_deviation(self):
        mol = chain(8)
        ints, val = ortho(mol)
        ref = np.abs(ints.ao2loc.T @ ints.ovlp @ ints.ao2loc
                     - np.eye(ints.norb)).max()
        self.assertAlmostEqual(val, ref, 14)

    def test_ortho_does_not_grow_with_system_size(self):
        # The largest pair deviation is a per-pair quantity, so it stays flat as the
        # orbital count rises and the tolerance means the same thing at any size.
        _, small = ortho(chain(6))
        _, large = ortho(chain(24))
        self.assertLess(small, 1e-8)
        self.assertLess(large, 1e-8)
        self.assertLess(large, 20.0 * max(small, 1e-15))


if __name__ == "__main__":
    print("Orthonormality measure of the localized basis\n")
    unittest.main()
