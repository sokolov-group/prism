# Copyright 2025 Prism Developers. All Rights Reserved.
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
# Authors: Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#          Carlos E. V. de Moura <carlosevmoura@gmail.com>
#          James D. Serna <jserna456@gmail.com>
 
import unittest
import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.nevpt

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

r = 0.96
x = r * math.sin(104.5 * math.pi/(2 * 180.0))
y = r * math.cos(104.5 * math.pi/(2 * 180.0))

mol = pyscf.gto.Mole()
mol.atom = [
            ['O', (0.0, 0.0, 0.0)],
            ['H', (0.0,  -x,   y)],
            ['H', (0.0,   x,   y)]]
mol.basis = 'aug-cc-pvdz'
mol.symmetry = False
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# CASSCF calculation
n_states = 6
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6
emc = mc.mc1step()[0]

# QD-NEVPT2 calculation
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
nevpt = prism.nevpt.NEVPT(interface)
nevpt.compute_singles_amplitudes = False
nevpt.semi_internal_projector = "gno"
nevpt.s_thresh_singles = 1e-8
nevpt.s_thresh_doubles = 1e-8
nevpt.method = "qd-nevpt2"

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,  -75.8195380454856, 6)
        self.assertAlmostEqual(mc.e_cas,  -12.6639537183245, 6)

    def test_prism(self):

        e_tot, e_corr, osc = nevpt.kernel()

        self.assertAlmostEqual(e_tot[0], -76.267176933501, 5)
        self.assertAlmostEqual(e_tot[1], -75.989861278395, 5)
        self.assertAlmostEqual(e_tot[2], -75.977156822450, 5)
        self.assertAlmostEqual(e_tot[3], -75.917821026148, 5)
        self.assertAlmostEqual(e_tot[4], -75.911455809535, 5)
        self.assertAlmostEqual(e_tot[5], -75.902504277921, 5)
        
        self.assertAlmostEqual(e_corr[0], -0.227497485644, 5)
        self.assertAlmostEqual(e_corr[1], -0.159162632919, 5)
        self.assertAlmostEqual(e_corr[2], -0.160421631008, 5)
        self.assertAlmostEqual(e_corr[3], -0.164316953799, 5)
        self.assertAlmostEqual(e_corr[4], -0.166517119857, 5)
        self.assertAlmostEqual(e_corr[5], -0.170832051808, 5)
        
        self.assertAlmostEqual(osc[0], 0.0, 5)
        self.assertAlmostEqual(osc[1], 0.0191822507, 5)
        self.assertAlmostEqual(osc[2], 0.0, 5)
        self.assertAlmostEqual(osc[3], 0.0, 5)
        self.assertAlmostEqual(osc[4], 0.0, 5)

if __name__ == "__main__":
    print("QD-NEVPT2 test")
    unittest.main()
