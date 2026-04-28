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

# Version:
#Python 3.14.3 | [GCC 14.3.0]
#numpy 2.4.3  scipy 1.17.1  h5py 3.16.0 PySCF version 2.12.1

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
mol.basis = 'aug-cc-pvdz'
mol.symmetry = True
mol.verbose = 1
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12
mf.scf()
print("SCF energy: %#.15g\n" % mf.e_tot)

# CASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 4, 4)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6
mc.kernel()
print('CASSCF E = %#.15g  E(CI) = %#.15g' % (mc.e_tot, mc.e_cas))

# MR-ADC calculation
interface = prism.interface.PYSCF(mf, mc, backend = None)
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.ncvs = 1
mr_adc.nroots = 4
mr_adc.s_thresh_singles = 1e-6
mr_adc.s_thresh_doubles = 1e-10
mr_adc.method_type = "cvs-ee"

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot, -76.0585829208871, 6)
        self.assertAlmostEqual(mc.e_cas, -6.55621478196034, 6)

    def test_prism(self):

        e, p, x = mr_adc.kernel()

        self.assertAlmostEqual(e[0], 537.0551, 4)
        self.assertAlmostEqual(e[1], 537.9879, 4)
        self.assertAlmostEqual(e[2], 540.4205, 4)
        self.assertAlmostEqual(e[3], 540.6919, 4)

        self.assertAlmostEqual(p[0], 0.000412, 4)
        self.assertAlmostEqual(p[1], 0.001140, 4)
        self.assertAlmostEqual(p[2], 0.000934, 4)
        self.assertAlmostEqual(p[3], 0.000940, 4)

if __name__ == "__main__":
    print("CVS-EE calculations for different CVS-EE-MR-ADC methods")
    unittest.main()

