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
import pyscf.mcscf
from prism.dmet import DMET, LocalIntegrals, make_fragments

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [['N', (0, 0, 0)], ['N', (0, 0, 1.5)]]
mol.basis = 'sto-3g'
mol.verbose = 4
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# Full-molecule CASSCF(4e,4o)
mc = pyscf.mcscf.CASSCF(mf, 4, 4)
mc.conv_tol = 1e-11
emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

# DMET with one N atom per fragment, same active space on each cluster
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0], [1]])
dmet = DMET(ints, frags, False, method='CASSCF', ncas=4, nelecas=4,
            sc_method='NONE', casscf_kwargs={'conv_tol': 1e-11})

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mf.e_tot, -107.272448501206, 5)
        self.assertAlmostEqual(mc.e_tot, -107.500321551571, 5)

    def test_prism(self):

        e_dmet = dmet.oneshot()

        self.assertAlmostEqual(e_dmet,   -107.504843893484, 5)

        # Each cluster keeps 8 of the 10 orbitals, so 2 are frozen into the core.
        self.assertEqual(dmet.dmet_orbs[0].shape[1], 8)

if __name__ == "__main__":
    print("DMET-CASSCF test")
    unittest.main()
