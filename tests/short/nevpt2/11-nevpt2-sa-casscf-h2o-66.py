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
mol.symmetry = True
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

# NEVPT2 calculation
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
nevpt = prism.nevpt.NEVPT(interface)
nevpt.compute_singles_amplitudes = False
nevpt.semi_internal_projector = "gno"
nevpt.s_thresh_singles = 1e-10
nevpt.s_thresh_doubles = 1e-10
nevpt.method = "nevpt2"

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,  -75.6915177093492, 6)
        self.assertAlmostEqual(mc.e_cas,  -12.5683871410396, 6)

    def test_prism(self):

        e_tot, e_corr = nevpt.kernel()

        self.assertAlmostEqual(e_tot[0], -76.270147938631, 5)
        self.assertAlmostEqual(e_tot[1], -75.901798736790, 5)
        self.assertAlmostEqual(e_tot[2], -75.886638208021, 5)
        self.assertAlmostEqual(e_tot[3], -75.771289628079, 5)
        self.assertAlmostEqual(e_tot[4], -75.750692912900, 5)
        self.assertAlmostEqual(e_tot[5], -75.697791199399, 5)

if __name__ == "__main__":
    print("NEVPT2 test")
    unittest.main()
