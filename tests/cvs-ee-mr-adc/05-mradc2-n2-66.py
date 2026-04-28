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
# Authors: Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#          Carlos E. V. de Moura <carlosevmoura@gmail.com>

import unittest
import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

r = 1.098
mol = pyscf.gto.Mole()
mol.atom = [
            ['N', (0.0, 0.0, -r/2)],
            ['N', (0.0, 0.0,  r/2)]]
mol.basis = 'aug-cc-pvdz'
mol.symmetry = True
mol.verbose = 1
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12
mf.scf()
print("SCF energy: %#.15g\n" % mf.e_tot)

## CASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 6, 6)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6
mc.kernel()
print('CASSCF E = %#.15g  E(CI) = %#.15g\n' % (mc.e_tot, mc.e_cas))

# Run MR-ADC computation
interface = prism.interface.PYSCF(mf, mc, backend = None)
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.nroots = 12
mr_adc.ncvs = 2
mr_adc.s_thresh_singles = 1e-6
mr_adc.s_thresh_doubles = 1e-10
mr_adc.method_type = "cvs-ee"

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot, -109.096203344053, 6)
        self.assertAlmostEqual(mc.e_cas, -13.6480313746735, 6)

    def test_prism(self):

        e, p, x = mr_adc.kernel()

        self.assertAlmostEqual(e[0],  404.2803, 4)
        self.assertAlmostEqual(e[1],  404.2803, 4)
        self.assertAlmostEqual(e[2],  404.3095, 4)
        self.assertAlmostEqual(e[3],  404.3095, 4)
        self.assertAlmostEqual(e[4],  411.6867, 4)
        self.assertAlmostEqual(e[5],  411.7510, 4)
        self.assertAlmostEqual(e[6],  412.3128, 4)
        self.assertAlmostEqual(e[7],  412.3128, 4)
        self.assertAlmostEqual(e[8],  412.3215, 4)
        self.assertAlmostEqual(e[9],  412.3803, 4)
        self.assertAlmostEqual(e[10], 412.4013, 4)
        self.assertAlmostEqual(e[11], 412.4013, 4)

        self.assertAlmostEqual(p[0],  0.010537, 4)
        self.assertAlmostEqual(p[1],  0.010537, 4)
        self.assertAlmostEqual(p[2],  0.000000, 4)
        self.assertAlmostEqual(p[3],  0.000000, 4)
        self.assertAlmostEqual(p[4],  0.000781, 4)
        self.assertAlmostEqual(p[5],  0.000000, 4)
        self.assertAlmostEqual(p[6],  0.000000, 4)
        self.assertAlmostEqual(p[7],  0.000000, 4)
        self.assertAlmostEqual(p[8],  0.000000, 4)
        self.assertAlmostEqual(p[9],  0.000001, 4)
        self.assertAlmostEqual(p[10], 0.001190, 4)
        self.assertAlmostEqual(p[11], 0.001190, 4)

if __name__ == "__main__":
    print("CVS-EE calculations for different CVS-EE-MR-ADC methods")
    unittest.main()
