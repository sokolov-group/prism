#!/usr/bin/env python3
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
mol.spin = 2
mol.build()

# RHF calculation
mf = pyscf.scf.ROHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# MR-ADC calculation
interface = prism.interface.PYSCF(mf, backend = 'opt_einsum').density_fit('cc-pvdz-ri')
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.ncvs = 2
mr_adc.nroots = 12
mr_adc.s_thresh_singles = 1e-5
mr_adc.s_thresh_doubles = 1e-10
mr_adc.method_type = "cvs-ip"
mr_adc.method = "mr-adc(2)-x"

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mf.e_tot, -75.775513684361, 5)

    def test_prism(self):

        e, p, x = mr_adc.kernel()

        self.assertAlmostEqual(e[0], 27.4030, 3)
        self.assertAlmostEqual(e[1], 27.5074, 3)
        self.assertAlmostEqual(e[2], 36.0549, 3)
        self.assertAlmostEqual(e[3], 37.8929, 3)
        self.assertAlmostEqual(e[4], 39.0754, 3)
        self.assertAlmostEqual(e[5], 39.4953, 3)

        self.assertAlmostEqual(p[0], 0.000406, 2)
        self.assertAlmostEqual(p[1], 0.000000, 2)
        self.assertAlmostEqual(p[2], 1.347866, 4)
        self.assertAlmostEqual(p[3], 0.000002, 4)
        self.assertAlmostEqual(p[4], 0.000071, 4)
        self.assertAlmostEqual(p[5], 0.097861, 4)

if __name__ == "__main__":
    print("CVS-IP calculations for different CVS-IP-MR-ADC methods")
    unittest.main()
