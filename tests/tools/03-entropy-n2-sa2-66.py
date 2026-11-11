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
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.fci
import prism.interface
from prism.tools import entropy

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

# SA-CASSCF calculation
weights = [0.7, 0.3]
mc = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights)
mc.fix_spin_(ss = 0)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6

emc = mc.mc1step()[0]
print("SA-CASSCF energy: %f\n" % emc)

# PRISM interface
interface = prism.interface.PYSCF(mf, mc)

class KnownValues(unittest.TestCase):

    def test_prism(self):

        s1, s2, mi = entropy.analyze(interface)

        # s1 from weight-averaged RDMs
        ncas, nelecas = interface.ncas, interface.ref_nelecas[0]
        rdms = [pyscf.fci.direct_spin1.make_rdm12s(c, ncas, nelecas) for c in interface.ref_wfn]
        dm1a = sum(w * r[0][0] for w, r in zip(weights, rdms))
        dm1b = sum(w * r[0][1] for w, r in zip(weights, rdms))
        dm2ab = sum(w * r[1][1] for w, r in zip(weights, rdms))
        s1_rdm = entropy.orbital_entropies_from_rdm((dm1a, dm1b), dm2ab)
        for i in range(ncas):
            self.assertAlmostEqual(s1[i], s1_rdm[i], 10)

        # Equal weights give different entropies
        s1_equal, s2_equal = entropy.orbital_entropies(interface.ref_wfn, ncas, nelecas)
        self.assertGreater(abs(s1 - s1_equal).max(), 1e-3)

if __name__ == "__main__":
    print("Orbital entropy test")
    unittest.main()
