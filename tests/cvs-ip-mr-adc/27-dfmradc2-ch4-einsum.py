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
import importlib
import numpy as np
import pyscf.gto
import pyscf.dft
import pyscf.mcscf
import prism.interface
import prism.mr_adc

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

mol = pyscf.gto.Mole()
mol.atom = '''
C    0.0000   0.0000   0.0000
H    0.6276   0.6276   0.6276
H    0.6276  -0.6276  -0.6276
H   -0.6276   0.6276  -0.6276
H   -0.6276  -0.6276   0.6276 
'''
mol.basis = 'cc-pvdz'
mol.symmetry = True
mol.build()

# RKS calculation
mf = pyscf.dft.RKS(mol)
mf.xc = 'B3LYP'
mf.conv_tol = 1e-12
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# CASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 7, 6)
mc.max_cycle = 100
mc.conv_tol = 1e-10
mc.conv_tol_grad = 1e-6
emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

# MR-ADC calculation
interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum').density_fit('cc-pvdz-ri')
mr_adc_1 = prism.mr_adc.CVSIPMRADC(interface)
mr_adc_1.ncvs = 1
mr_adc_1.nroots = 9
mr_adc_1.s_thresh_singles = 1e-5
mr_adc_1.s_thresh_doubles = 1e-10

# MR-ADC calculation
interface = prism.interface.PYSCF(mf, mc, backend = 'pytblis').density_fit('cc-pvdz-ri')
mr_adc_2 = prism.mr_adc.CVSIPMRADC(interface)
mr_adc_2.ncvs = 1
mr_adc_2.nroots = 9
mr_adc_2.s_thresh_singles = 1e-5
mr_adc_2.s_thresh_doubles = 1e-10

# MR-ADC calculation
interface = prism.interface.PYSCF(mf, mc, backend = 'numpy').density_fit('cc-pvdz-ri')
mr_adc_3 = prism.mr_adc.CVSIPMRADC(interface)
mr_adc_3.ncvs = 1
mr_adc_3.nroots = 9
mr_adc_3.s_thresh_singles = 1e-5
mr_adc_3.s_thresh_doubles = 1e-10

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
       self.assertAlmostEqual(mc.e_tot, -40.2495982004973, 5)
       self.assertAlmostEqual(mc.e_cas, -9.85045629332514, 4)

    def _check_values(self, e, p):
        self.assertAlmostEqual(e[0], 293.9668, 3)
        self.assertAlmostEqual(e[1], 323.0314, 3)
        self.assertAlmostEqual(e[2], 323.0619, 3)
        self.assertAlmostEqual(e[3], 324.0186, 3)
        self.assertAlmostEqual(e[4], 324.0186, 3)
        self.assertAlmostEqual(e[5], 324.0186, 3)
        self.assertAlmostEqual(e[6], 324.0186, 3)
        self.assertAlmostEqual(e[7], 324.0186, 3)
        self.assertAlmostEqual(e[8], 324.0186, 3)

        self.assertAlmostEqual(p[0], 1.625587, 2)
        self.assertAlmostEqual(p[1], 0.      , 4)
        self.assertAlmostEqual(p[2], 0.001240, 4)
        self.assertAlmostEqual(p[3], 0.000002, 4)
        self.assertAlmostEqual(p[4], 0.000006, 4)
        self.assertAlmostEqual(p[5], 0.000006, 4)
        self.assertAlmostEqual(p[6], 0.000004, 4)
        self.assertAlmostEqual(p[7], 0.000002, 4)
        self.assertAlmostEqual(p[8], 0.000004, 4)

    def test_prism_opt_einsum(self):
        if importlib.util.find_spec('opt_einsum') is None:
            self.skipTest('opt_einsum is not available')
        e, p, x = mr_adc_1.kernel()
        self._check_values(e, p)

    def test_prism_pytblis(self):
        if importlib.util.find_spec('pytblis') is None:
            self.skipTest('pytblis is not available')
        e, p, x = mr_adc_2.kernel()
        self._check_values(e, p)

    def test_prism_numpy(self):
        e, p, x = mr_adc_3.kernel()
        self._check_values(e, p)

if __name__ == "__main__":
   print("CVS-IP calculations for different CVS-IP-MR-ADC methods")
   unittest.main()
