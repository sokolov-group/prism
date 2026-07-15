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
#

import unittest
import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.nevpt

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
mol.spin = 2
mol.build()

# RHF calculation
mf = pyscf.scf.ROHF(mol).density_fit()
mf.conv_tol = 1e-12
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# NEVPT2 calculation
interface = prism.interface.PYSCF(mf, backend = 'opt_einsum')
nevpt = prism.nevpt.NEVPT(interface)
nevpt.compute_singles_amplitudes = False
nevpt.semi_internal_projector = "gno"
nevpt.s_thresh_singles = 1e-6
nevpt.s_thresh_doubles = 1e-10

# NEVPT2 calculation
df_interface = prism.interface.PYSCF(mf, backend = 'opt_einsum').density_fit()
df_nevpt = prism.nevpt.NEVPT(df_interface)
df_nevpt.compute_singles_amplitudes = False
df_nevpt.semi_internal_projector = "gno"
df_nevpt.s_thresh_singles = 1e-6
df_nevpt.s_thresh_doubles = 1e-10

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mf.e_tot, -75.8159702269242, 5)

    def test_prism(self):

        e_tot, e_corr, osc = nevpt.kernel()

        self.assertAlmostEqual(interface.mc.e_tot, -75.815970226924, 5)

        self.assertAlmostEqual(e_tot[0], -75.993734292602, 6)
        self.assertAlmostEqual(e_corr[0], -0.177764065678, 6)

    def test_df_prism(self):

        e_tot, e_corr, osc = df_nevpt.kernel()

        self.assertAlmostEqual(df_interface.mc.e_tot, -75.815970226924, 5)

        self.assertAlmostEqual(e_tot[0], -75.993727586513, 6)
        self.assertAlmostEqual(e_corr[0], -0.177757359589, 6)

if __name__ == "__main__":
    print("NEVPT2 test")
    unittest.main()
