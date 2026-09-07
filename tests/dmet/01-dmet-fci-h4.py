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
import pyscf.fci
from prism.dmet import DMET, LocalIntegrals, make_fragments

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

mol = pyscf.gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22'
mol.basis = 'sto-3g'
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# Full-molecule FCI, exact in this basis
e_fci = pyscf.fci.FCI(mf).kernel()[0]
print("Full-molecule FCI energy: %f\n" % e_fci)

# DMET with one H2 fragment per pair of atoms
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0, 1], [2, 3]])
dmet = DMET(ints, frags, False, method='FCI', sc_method='NONE')

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mf.e_tot,  -2.097899612083, 5)
        self.assertAlmostEqual(e_fci,     -2.138889912873, 5)

    def test_prism(self):

        e_dmet = dmet.oneshot()

        self.assertAlmostEqual(e_dmet,    -2.138889912873, 5)

        # This partition is exact, so the embedding must reproduce full-molecule FCI.
        self.assertAlmostEqual(e_dmet, e_fci, 8)

if __name__ == "__main__":
    print("DMET-FCI test")
    unittest.main()
