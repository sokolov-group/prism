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

import unittest
import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
from prism.tools import entropy

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [
            ['N', (0.0,  0.0, 0.0)],
            ['H', (0.0,  1.0, 0.3)],
            ['H', (0.0, -1.0, 0.3)]]
mol.basis = 'sto-3g'
mol.spin = 1
mol.build()

# ROHF calculation
mf = pyscf.scf.ROHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# CASCI calculation
mc = pyscf.mcscf.CASCI(mf, 6, 7)

emc = mc.kernel()[0]
print("CASCI energy: %f\n" % emc)

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot, -54.823403805500, 5)

    def test_prism(self):

        s1, s2 = entropy.orbital_entropies(mc.ci, mc.ncas, mc.nelecas)
        mi = entropy.mutual_information(s1, s2)
        print("s1:", s1)
        print("Mutual information:\n", mi)

        s1_ref = [0.103090838600, 0.111997047400, 0.062445697600,
                  0.033246424000, 0.152202133400, 0.093920878100]
        for i in range(mc.ncas):
            self.assertAlmostEqual(s1[i], s1_ref[i], 5)
        self.assertAlmostEqual(mi[1, 5], 0.050554885400, 5)
        self.assertAlmostEqual(mi[1, 5], mi.max(), 10)

if __name__ == "__main__":
    print("Orbital entropy test")
    unittest.main()
