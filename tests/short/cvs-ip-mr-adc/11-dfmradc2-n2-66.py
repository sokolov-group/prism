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
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# CASSCF calculation
mc = pyscf.mcscf.DFCASSCF(mf, 6, 6)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6

emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

# Run MR-ADC computation
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True).density_fit('aug-cc-pvdz-ri')
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.nroots = 12
mr_adc.ncvs = 2
mr_adc.s_thresh_singles = 1e-6
mr_adc.s_thresh_doubles = 1e-10
mr_adc.method_type = "cvs-ip"
mr_adc.method = "mr-adc(2)"

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot, -109.095971211736, 6)
        self.assertAlmostEqual(mc.e_cas, -13.6479741036814, 6)

    def test_prism(self):

        e, p, x = mr_adc.kernel()

        self.assertAlmostEqual(e[0],   413.2917, 4)
        self.assertAlmostEqual(e[1],   413.3778, 4)
#        self.assertAlmostEqual(e[2],  435.6945, 4)
#        self.assertAlmostEqual(e[3],  435.7686, 4)
#        self.assertAlmostEqual(e[4],  437.2937, 4)
#        self.assertAlmostEqual(e[5],  437.2937, 4)
#        self.assertAlmostEqual(e[6],  437.2937, 4)
#        self.assertAlmostEqual(e[7],  437.2937, 4)
#        self.assertAlmostEqual(e[8],  437.3701, 4)
#        self.assertAlmostEqual(e[9],  437.3701, 4)
#        self.assertAlmostEqual(e[10], 437.3701, 4)
#        self.assertAlmostEqual(e[11], 437.3701, 4)

        self.assertAlmostEqual(p[0],   1.61990120, 4)
        self.assertAlmostEqual(p[1],   1.61943131, 4)
#        self.assertAlmostEqual(p[2],  0.00000000, 4)
#        self.assertAlmostEqual(p[3],  0.00223971, 4)
#        self.assertAlmostEqual(p[4],  0.00208403, 4)
#        self.assertAlmostEqual(p[5],  0.00000045, 4)
#        self.assertAlmostEqual(p[6],  0.00000426, 4)
#        self.assertAlmostEqual(p[7],  0.00000287, 4)
#        self.assertAlmostEqual(p[8],  0.00000184, 4)
#        self.assertAlmostEqual(p[9],  0.00000198, 4)
#        self.assertAlmostEqual(p[10], 0.00000084, 4)
#        self.assertAlmostEqual(p[11], 0.00000205, 4)

if __name__ == "__main__":
    print("CVS-IP calculations for different CVS-IP-MR-ADC methods")
    unittest.main()
