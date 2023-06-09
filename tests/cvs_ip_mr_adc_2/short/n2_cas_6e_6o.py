# Copyright 2023 Prism Developers. All Rights Reserved.
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
#

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
mc = pyscf.mcscf.CASSCF(mf, 6, 6)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6

emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

# Run MR-ADC computation
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.nroots = 8
mr_adc.ncvs = 2
mr_adc.s_thresh_singles = 1e-6
mr_adc.s_thresh_doubles = 1e-10
mr_adc.method_type = "cvs-ip"
mr_adc.method = "mr-adc(2)"

class KnownValues(unittest.TestCase):

    def test_ip_mr_adc2(self):

        e, p, x = mr_adc.kernel()

        self.assertAlmostEqual(e[0], 413.28898349, 4)
        self.assertAlmostEqual(e[1], 413.37510246, 4)
        self.assertAlmostEqual(e[2], 435.69778380, 4)
        self.assertAlmostEqual(e[3], 435.77181575, 4)
        self.assertAlmostEqual(e[4], 437.29326671, 4)
        self.assertAlmostEqual(e[5], 437.29326671, 4)
        self.assertAlmostEqual(e[6], 437.29326672, 4)
        self.assertAlmostEqual(e[7], 437.29326672, 4)

        self.assertAlmostEqual(p[0], 1.61992576, 4)
        self.assertAlmostEqual(p[1], 1.61945675, 4)
        self.assertAlmostEqual(p[2], 0.00223781, 4)
        self.assertAlmostEqual(p[3], 0.00208223, 4)
        self.assertAlmostEqual(p[4], 0.00000231, 4)
        self.assertAlmostEqual(p[5], 0.00000240, 4)
        self.assertAlmostEqual(p[6], 0.00000291, 4)
        self.assertAlmostEqual(p[7], 0.00000180, 4)

if __name__ == "__main__":
    print("IP calculations for different IP-MR-ADC methods")
    unittest.main()
