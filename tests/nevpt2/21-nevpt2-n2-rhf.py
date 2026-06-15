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
import prism.nevpt

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

# NEVPT2 calculation
interface = prism.interface.PYSCF(mf, backend = 'opt_einsum')
nevpt = prism.nevpt.NEVPT(interface)
nevpt.compute_singles_amplitudes = False
nevpt.semi_internal_projector = "gno"
nevpt.s_thresh_singles = 1e-6
nevpt.s_thresh_doubles = 1e-10
nevpt.verbose = 4

# NEVPT2 calculation
df_interface = prism.interface.PYSCF(mf, backend = 'opt_einsum').density_fit('aug-cc-pvdz-ri')
df_nevpt = prism.nevpt.NEVPT(df_interface)
df_nevpt.compute_singles_amplitudes = False
df_nevpt.semi_internal_projector = "gno"
df_nevpt.s_thresh_singles = 1e-6
df_nevpt.s_thresh_doubles = 1e-10
df_nevpt.verbose = 4

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mf.e_tot, -108.960608507245, 6)

    def test_prism(self):

        e_tot, e_corr, osc = nevpt.kernel()

        self.assertAlmostEqual(e_tot[0], -109.282625430938, 6)
        self.assertAlmostEqual(e_corr[0], -0.322016923693, 6)

    def test_df_prism(self):

        e_tot, e_corr, osc = df_nevpt.kernel()

        self.assertAlmostEqual(e_tot[0], -109.282738926072, 6)
        self.assertAlmostEqual(e_corr[0], -0.322130418827, 6)

if __name__ == "__main__":
    print("NEVPT2 test")
    unittest.main()
