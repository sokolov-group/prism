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
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

r = 0.917

mol = pyscf.gto.Mole()
mol.atom = [
            ['H', (0.0, 0.0, -r/2)],
            ['F', (0.0, 0.0,  r/2)]]
mol.basis = 'aug-cc-pvdz'
mol.symmetry = True
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# CASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 4, 4)

mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6
mc.max_cycle_macro = 100

emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

# MR-ADC calculation
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.ncvs = 1
mr_adc.nroots = 4
mr_adc.s_thresh_singles = 1e-6
mr_adc.s_thresh_doubles = 1e-10
mr_adc.method_type = "cvs-ip"
mr_adc.method = "mr-adc(2)"

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot, -100.036765659825, 6)
        self.assertAlmostEqual(mc.e_cas, -7.44248856525863, 6)

    def test_prism(self):

        e, p, x = mr_adc.kernel()

        self.assertAlmostEqual(e[0], 693.67507015, 4)
        self.assertAlmostEqual(e[1], 735.43509592, 4)
        self.assertAlmostEqual(e[2], 735.43509592, 4)
        self.assertAlmostEqual(e[3], 735.43509592, 4)

        self.assertAlmostEqual(p[0], 1.55247094, 4)
        self.assertAlmostEqual(p[1], 0.00000000, 4)
        self.assertAlmostEqual(p[2], 0.00000000, 4)
        self.assertAlmostEqual(p[3], 0.00000000, 4)

if __name__ == "__main__":
    print("CVS-IP calculations for different CVS-IP-MR-ADC methods")
    unittest.main()
