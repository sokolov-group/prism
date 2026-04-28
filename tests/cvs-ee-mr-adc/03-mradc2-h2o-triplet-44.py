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
mol.verbose = 1
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12
mf.scf()
print("SCF energy: %#.15g\n" % mf.e_tot)

# CASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 4, (2,2))
mc.max_cycle = 100
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6
mc.fix_spin_(ss = 2)
mc.kernel()
print('CASSCF E = %#.15g  E(CI) = %#.15g' % (mc.e_tot, mc.e_cas))

# MR-ADC calculation
interface = prism.interface.PYSCF(mf, mc, backend = None)
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.ncvs = 2
mr_adc.nroots = 12
mr_adc.s_thresh_singles = 1e-6
mr_adc.s_thresh_doubles = 1e-10
mr_adc.method_type = "cvs-ee"

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,  -75.7804754098704, 5)
        self.assertAlmostEqual(mc.e_cas,  -5.92898662523508, 4)

    def test_prism(self):

        e, p, x = mr_adc.kernel()

        self.assertAlmostEqual(e[0],  19.8773, 4)
        self.assertAlmostEqual(e[1],  30.0463, 4)
        self.assertAlmostEqual(e[2],  33.5786, 4)
        self.assertAlmostEqual(e[3],  40.0088, 4)
        self.assertAlmostEqual(e[4],  43.6361, 4)
        self.assertAlmostEqual(e[5],  44.2359, 4)
        self.assertAlmostEqual(e[6],  44.2448, 4)
        self.assertAlmostEqual(e[7],  44.6987, 4)
        self.assertAlmostEqual(e[8],  46.9852, 4)
        self.assertAlmostEqual(e[9],  48.8466, 4)
        self.assertAlmostEqual(e[10], 49.1752, 4)
        self.assertAlmostEqual(e[11], 49.2921, 4)

        self.assertAlmostEqual(p[0],  0.243323, 4)
        self.assertAlmostEqual(p[1],  0.038404, 4)
        self.assertAlmostEqual(p[2],  0.123329, 4)
        self.assertAlmostEqual(p[3],  0.000000, 4)
        self.assertAlmostEqual(p[4],  0.000000, 4)
        self.assertAlmostEqual(p[5],  0.000000, 4)
        self.assertAlmostEqual(p[6],  0.000000, 4)
        self.assertAlmostEqual(p[7],  0.048239, 4)
        self.assertAlmostEqual(p[8],  0.011185, 4)
        self.assertAlmostEqual(p[9],  0.000000, 4)
        self.assertAlmostEqual(p[10], 0.002222, 4)
        self.assertAlmostEqual(p[11], 0.000000, 4)

if __name__ == "__main__":
    print("CVS-EE calculations for different CVS-EE-MR-ADC methods")
    unittest.main()
