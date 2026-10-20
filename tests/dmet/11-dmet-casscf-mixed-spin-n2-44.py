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
from prism.dmet import DMET, LocalIntegrals, make_fragments

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [['N', (0, 0, 0)], ['N', (0, 0, 1.1)]]
mol.basis = 'sto-3g'
mol.build()

mf = pyscf.scf.RHF(mol)
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0], [1]])

# One orbital set averaged over 5 singlet and 4 triplet roots.
dmet = DMET(ints, frags, False, method='CASSCF', ncas=4, nelecas=4,
            sa_mix=[(0, 5), (2, 4)], sc_method='NONE')

class KnownValues(unittest.TestCase):

    def test_prism(self):

        e_dmet = dmet.oneshot()
        res = dmet.cas_results[0]
        ss = np.asarray(res['spin_square'])
        e_states = np.asarray(res['e_states'])

        self.assertAlmostEqual(e_dmet, -107.244051615, 5)
        self.assertEqual(len(e_states), 9)

        # The roots carry the spins that were requested.
        self.assertEqual(int(np.sum(ss < 0.5)), 5)
        self.assertEqual(int(np.sum(np.abs(ss - 2.0) < 0.1)), 4)
        self.assertLess(np.max(ss), 2.1)

        gap = (e_states[ss > 1.5].min() - e_states[ss < 0.5].min()) * 27.211386245988
        self.assertAlmostEqual(gap, 8.176, 2)

    def test_rejects_a_spin_penalty(self):

        # cas_spin penalizes the spins the mixed average is built from.
        with self.assertRaises(ValueError):
            DMET(ints, frags, False, method='CASSCF', ncas=4, nelecas=4,
                 sa_mix=[(0, 5), (2, 4)], cas_spin=0, sc_method='NONE').oneshot()

if __name__ == "__main__":
    print("DMET-CASSCF mixed-spin state average test")
    unittest.main()
