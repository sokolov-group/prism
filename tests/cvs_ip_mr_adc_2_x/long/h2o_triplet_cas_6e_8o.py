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
# Tests prepared for Prism 0.4, PySCF 2.5.0 and NumPy 1.26.4
# Results can deviate according to their versions
#

import unittest
import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

r = 0.96
x = r * math.sin(104.5 * math.pi/(2 * 180.0))
y = r * math.cos(104.5 * math.pi/(2 * 180.0))

mol = pyscf.gto.Mole()
mol.atom = [
            ['O', (0.0, 0.0, 0.0)],
            ['H', (0.0,  -x,   y)],
            ['H', (0.0,   x,   y)]]
mol.basis = 'cc-pvdz'
mol.symmetry = True
#mol.verbose = 4
mol.spin = 2
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# CASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 8, (3,3))
mc.max_cycle = 100
mc.conv_tol = 1e-10
mc.conv_tol_grad = 1e-6
mc.fix_spin_(ss = 2)

emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

#import prism_beta.interface
#import prism_beta.mr_adc
#interface = prism_beta.interface.PYSCF(mf, mc, opt_einsum = True)
#mr_adc = prism_beta.mr_adc.MRADC(interface)
#mr_adc.ncvs = 2
#mr_adc.nroots = 36
#mr_adc.s_thresh_singles = 1e-6
#mr_adc.s_thresh_doubles = 1e-10
#mr_adc.method_type = "cvs-ip"
#mr_adc.method = "mr-adc(2)-x"
#mr_adc.kernel()

# MR-ADC calculation
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.ncvs = 2
mr_adc.nroots = 12
mr_adc.s_thresh_singles = 1e-6
mr_adc.s_thresh_doubles = 1e-10
mr_adc.method_type = "cvs-ip"
mr_adc.method = "mr-adc(2)-x"

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot, -75.842737037233, 5)
        self.assertAlmostEqual(mc.e_cas, -12.999366522545, 4)

    def test_prism(self):

        e, p, x = mr_adc.kernel()

        self.assertAlmostEqual(e[0], 22.82433465, 3)
        self.assertAlmostEqual(e[1], 23.15309242, 3)
        self.assertAlmostEqual(e[2], 33.04909016, 3)
        self.assertAlmostEqual(e[3], 34.39028793, 3)
        self.assertAlmostEqual(e[4], 35.30391694, 3)
        self.assertAlmostEqual(e[5], 37.8130614 , 3)

        self.assertAlmostEqual(p[0], 0.0003889, 4)
        self.assertAlmostEqual(p[1], 0.       , 4)
        self.assertAlmostEqual(p[2], 1.3426945, 4)
        self.assertAlmostEqual(p[3], 0.0000057, 4)
        self.assertAlmostEqual(p[4], 0.1099799, 4)
        self.assertAlmostEqual(p[5], 0.0000024, 4)

if __name__ == "__main__":
    print("CVS-IP calculations for different CVS-IP-MR-ADC methods")
    unittest.main()
