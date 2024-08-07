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

r = 1.1508

mol = pyscf.gto.Mole()
mol.atom = [
            ['N', (0.0, 0.0, 0.0)],
            ['O', (0.0, 0.0,   r)]]
mol.basis = 'cc-pcvdz'
mol.symmetry = True
#mol.verbose = 4
mol.spin = 0
mol.charge = +1
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.max_cycle = 250
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

mol.spin = 1
mol.charge = 0
mol.build()

# CASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 9, 7)
mc.max_cycle = 150
mc.conv_tol = 1e-8
mc.conv_tol_grad = 1e-5
mc.fcisolver.wfnsym = 'E1x'

emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

# MR-ADC calculation
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True).density_fit('cc-pvdz-ri')
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.ncvs = 2
mr_adc.nroots = 12
mr_adc.max_space = 100
mr_adc.s_thresh_singles = 1e-6
mr_adc.s_thresh_doubles = 1e-10
mr_adc.method_type = "cvs-ip"
mr_adc.method = "mr-adc(2)"

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot, -129.402145048918, 5)
        self.assertAlmostEqual(mc.e_cas, -17.5876339406786, 4)

    def test_prism(self):

        e, p, x = mr_adc.kernel()

        self.assertAlmostEqual(e[0],  413.56404009, 3)
        self.assertAlmostEqual(e[1],  426.5560299 , 3)
        self.assertAlmostEqual(e[2],  426.57477299, 3)
        self.assertAlmostEqual(e[3],  428.31504499, 3)
        self.assertAlmostEqual(e[4],  428.66911999, 3)
        self.assertAlmostEqual(e[5],  428.66911999, 3)

        self.assertAlmostEqual(p[0], 1.61026973, 4)
        self.assertAlmostEqual(p[1], 0.        , 4)
        self.assertAlmostEqual(p[2], 0.00216196, 4)
        self.assertAlmostEqual(p[3], 0.00000012, 4)
        self.assertAlmostEqual(p[4], 0.00000022, 2)
        self.assertAlmostEqual(p[5], 0.00000105, 2)

if __name__ == "__main__":
    print("CVS-IP calculations for different CVS-IP-MR-ADC methods")
    unittest.main()

