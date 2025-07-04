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
mol.symmetry = True
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# MS-CASCI calculation
n_states = 9
mc = pyscf.mcscf.CASCI(mf, 6, 6)
mc.fcisolver.nroots = n_states
emc = mc.casci()[0]

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
        self.assertAlmostEqual(mc.e_tot[0], -76.0416783372576, 6)
        self.assertAlmostEqual(mc.e_tot[1], -75.6244274698262, 6)
        self.assertAlmostEqual(mc.e_tot[2], -75.6147308447999, 6)
        self.assertAlmostEqual(mc.e_tot[3], -75.5083800718727, 6)
        self.assertAlmostEqual(mc.e_tot[4], -75.4917263897341, 6)
        self.assertAlmostEqual(mc.e_tot[5], -75.4284853679674, 6)
        self.assertAlmostEqual(mc.e_tot[6], -75.4231449561817, 6)
        self.assertAlmostEqual(mc.e_tot[7], -74.8920998737286, 6)
        self.assertAlmostEqual(mc.e_tot[8], -74.8081434103929, 6)
        self.assertAlmostEqual(mc.e_cas[0], -12.9969982062835, 6)
        self.assertAlmostEqual(mc.e_cas[1], -12.5797473388521, 6)
        self.assertAlmostEqual(mc.e_cas[2], -12.5700507138258, 6)
        self.assertAlmostEqual(mc.e_cas[3], -12.4636999408986, 6)
        self.assertAlmostEqual(mc.e_cas[4], -12.4470462587600, 6)
        self.assertAlmostEqual(mc.e_cas[5], -12.3838052369933, 6)
        self.assertAlmostEqual(mc.e_cas[6], -12.3784648252076, 6)
        self.assertAlmostEqual(mc.e_cas[7], -11.8474197427545, 6)
        self.assertAlmostEqual(mc.e_cas[8], -11.7634632794188, 6)

    def test_prism(self):

        e_tot, e_corr, osc = nevpt.kernel()

        self.assertAlmostEqual(e_tot[0], -76.274529690518, 6)
        self.assertAlmostEqual(e_tot[1], -75.924219959731, 6)
        self.assertAlmostEqual(e_tot[2], -75.909199702092, 6)
        self.assertAlmostEqual(e_tot[3], -75.792804957278, 6)
        self.assertAlmostEqual(e_tot[4], -75.785708232791, 6)
        self.assertAlmostEqual(e_tot[5], -75.682245942366, 6)
        self.assertAlmostEqual(e_tot[6], -75.664193875799, 6)
        self.assertAlmostEqual(e_tot[7], -75.458925609842, 6)
        self.assertAlmostEqual(e_tot[8], -75.317396820200, 6)
        
        self.assertAlmostEqual(osc[0], 0.0, 6)
        self.assertAlmostEqual(osc[1], 0.05948429, 6)
        self.assertAlmostEqual(osc[2], 0.0, 6)
        self.assertAlmostEqual(osc[3], 0.00193822, 6)
        self.assertAlmostEqual(osc[4], 0.0, 6)
        self.assertAlmostEqual(osc[5], 0.00640231, 6)
        self.assertAlmostEqual(osc[6], 0.00009423, 6)
        self.assertAlmostEqual(osc[7], 0.0, 6)

if __name__ == "__main__":
    print("NEVPT2 test")
    unittest.main()
