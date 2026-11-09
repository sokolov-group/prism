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
            ['N', (0.0, 0.0, 0.0)],
            ['N', (0.0, 0.0, 1.8)]]
mol.basis = 'sto-3g'
mol.symmetry = True
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# CASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 6, 6)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6

emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot, -107.477475114900, 5)

    def test_prism(self):

        s1, s2 = entropy.orbital_entropies(mc.ci, mc.ncas, mc.nelecas)
        mi = entropy.mutual_information(s1, s2)
        print("s1:", s1)
        print("Mutual information:\n", mi)

        s1_ref = [0.604600880800, 1.088016877500, 1.088016877500,
                  1.088008068300, 1.088008068300, 0.604644967900]
        for i in range(mc.ncas):
            self.assertAlmostEqual(s1[i], s1_ref[i], 5)
        self.assertAlmostEqual(mi[1, 4], 0.714070831400, 5)
        self.assertAlmostEqual(mi[1, 4], mi.max(), 10)

        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = mc.fcisolver.make_rdm12s(mc.ci, mc.ncas, mc.nelecas)
        s1_rdm = entropy.orbital_entropies_from_rdm((dm1a, dm1b), dm2ab)
        for i in range(mc.ncas):
            self.assertAlmostEqual(s1_rdm[i], s1[i], 10)

if __name__ == "__main__":
    print("Orbital entropy test")
    unittest.main()
