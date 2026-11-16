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

# CASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 6, 6)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6

emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

# SA-CASSCF calculation
weights = [0.7, 0.3]
mc_sa = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights)
mc_sa.fix_spin_(ss = 0)
mc_sa.conv_tol = 1e-11
mc_sa.conv_tol_grad = 1e-6

emc_sa = mc_sa.mc1step()[0]
print("SA-CASSCF energy: %f\n" % emc_sa)

def s1_from_rdms(interface, weights):
    ncas, nelecas = interface.ncas, interface.ref_nelecas[0]
    rdms = [pyscf.fci.direct_spin1.make_rdm12s(c, ncas, nelecas) for c in interface.ref_wfn]
    dm1a = sum(w * r[0][0] for w, r in zip(weights, rdms))
    dm1b = sum(w * r[0][1] for w, r in zip(weights, rdms))
    dm2ab = sum(w * r[1][1] for w, r in zip(weights, rdms))
    return entropy.orbital_entropies_from_rdm((dm1a, dm1b), dm2ab)

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot, -107.477475114900, 5)

    def test_casscf(self):

        interface = prism.interface.PYSCF(mf, mc)
        s1, s2, mi = entropy.analyze(interface)

        s1_ref = [0.604600880800, 1.088016877500, 1.088016877500,
                  1.088008068300, 1.088008068300, 0.604644967900]
        for i in range(interface.ncas):
            self.assertAlmostEqual(s1[i], s1_ref[i], 5)
        self.assertAlmostEqual(mi[1, 4], 0.714070831400, 5)
        self.assertAlmostEqual(mi[1, 4], mi.max(), 10)

        s1_rdm = s1_from_rdms(interface, [1.0])
        for i in range(interface.ncas):
            self.assertAlmostEqual(s1[i], s1_rdm[i], 10)

    def test_sa_casscf(self):

        interface = prism.interface.PYSCF(mf, mc_sa)
        s1, s2, mi = entropy.analyze(interface)

        s1_rdm = s1_from_rdms(interface, weights)
        for i in range(interface.ncas):
            self.assertAlmostEqual(s1[i], s1_rdm[i], 10)

if __name__ == "__main__":
    print("Orbital entropy test")
    unittest.main()
