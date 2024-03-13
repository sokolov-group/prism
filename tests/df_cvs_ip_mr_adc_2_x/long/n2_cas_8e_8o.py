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
mc = pyscf.mcscf.CASSCF(mf, 8, 8)
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
mr_adc.method = "mr-adc(2)-x"

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot, -109.118975355417, 6)
        self.assertAlmostEqual(mc.e_cas, -21.4578159085085, 6)

    def test_prism(self):

        e, p, x = mr_adc.kernel()

        self.assertAlmostEqual(e[0],  411.84460446, 4)
        self.assertAlmostEqual(e[1],  411.96122886, 4)
        self.assertAlmostEqual(e[2],  427.05797773, 4)
        self.assertAlmostEqual(e[3],  427.05799353, 4)
        self.assertAlmostEqual(e[4],  427.13933465, 4)
        self.assertAlmostEqual(e[5],  427.13935045, 4)
        self.assertAlmostEqual(e[6],  428.51574555, 4)
        self.assertAlmostEqual(e[7],  428.676993  , 4)
        self.assertAlmostEqual(e[8],  428.67701071, 4)
        self.assertAlmostEqual(e[9],  428.75435317, 4)
        self.assertAlmostEqual(e[10], 428.75437065, 4)
        self.assertAlmostEqual(e[11], 428.82069258, 4)

        self.assertAlmostEqual(p[0],  1.45512798, 4)
        self.assertAlmostEqual(p[1],  1.45174509, 4)
        self.assertAlmostEqual(p[2],  0.00000000, 4)
        self.assertAlmostEqual(p[3],  0.00000000, 4)
        self.assertAlmostEqual(p[4],  0.00000000, 4)
        self.assertAlmostEqual(p[5],  0.00000000, 4)
        self.assertAlmostEqual(p[6],  0.00000000, 4)
        self.assertAlmostEqual(p[7],  0.00000437, 4)
        self.assertAlmostEqual(p[8],  0.00000437, 4)
        self.assertAlmostEqual(p[9],  0.00000254, 4)
        self.assertAlmostEqual(p[10], 0.00000254, 4)
        self.assertAlmostEqual(p[11], 0.00000000, 4)

if __name__ == "__main__":
    print("CVS-IP calculations for different CVS-IP-MR-ADC methods")
    unittest.main()