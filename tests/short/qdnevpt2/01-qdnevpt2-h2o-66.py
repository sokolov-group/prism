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
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# SA-CASSCF calculation
n_states = 4
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6

emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

# NEVPT2 calculation
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
nevpt = prism.nevpt.NEVPT(interface)
nevpt.compute_singles_amplitudes = False
nevpt.semi_internal_projector = "gno"
nevpt.s_thresh_singles = 1e-8
nevpt.s_thresh_doubles = 1e-8
nevpt.method = "qd-nevpt2"

nevpt.print_level = 4
e_tot, e_corr = nevpt.kernel()

class KnownValues(unittest.TestCase):

    def test_pyscf(self):

        self.assertAlmostEqual(mc.e_tot, -75.8655558776989, 6)
        self.assertAlmostEqual(mc.e_cas, -13.2153988780588, 6)

    def test_prism(self):
        nevpt.print_level = 4
        e_tot, e_corr = nevpt.kernel()

        # SO QD-NEVPT2 Values
        self.assertAlmostEqual(e_tot[0], -76.260867964041, 5)
        self.assertAlmostEqual(e_tot[1], -75.990378036191, 5)
        self.assertAlmostEqual(e_tot[2], -75.977012303684, 5)
        self.assertAlmostEqual(e_tot[3], -75.914251153082, 5)

        self.assertAlmostEqual(e_corr[0], -0.205858747216, 5)
        self.assertAlmostEqual(e_corr[1], -0.152107670030, 5)
        self.assertAlmostEqual(e_corr[2], -0.152036495574, 5)
        self.assertAlmostEqual(e_corr[3], -0.170283033380, 5)

if __name__ == "__main__":
    print("QD-NEVPT2 test")
    unittest.main()
