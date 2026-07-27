# Copyright 2025 Prism Developers. All Rights Reserved.
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
#          Rajat S. Majumder <majumder.rajat071@gmail.com>
#          Nicholas Y. Chiang <nicholas.yiching.chiang@gmail.com>
#
#

import unittest
import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.nevpt
from pathlib import Path

np.set_printoptions(suppress=True, linewidth=1000)

mol = pyscf.gto.Mole()
mol.atom =[ 
[ 'Ga',  (0, 0, 0)],
]
mol.basis = 'def2-tzvp'
mol.symmetry = False
mol.charge = +1
mol.spin = 0
mol.verbose = 1
mol.build()


# RDFT calculation
mf = pyscf.scf.RKS(mol).x2c()
mf.xc = "bp86"
mf.conv_tol = 1e-12
ehf = mf.scf()
mf.analyze()

# SA-CASSCF calculation
n_states = 5
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 4, (1,1)).state_average_(weights)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6
emc = mc.mc1step()[0]
mc.analyze()


# QD-NEVPT2 with all electrons correlated
interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')
nevpt = prism.nevpt.QDNEVPT(interface)
nevpt.compute_singles_amplitudes = False
nevpt.semi_internal_projector = "gno"
nevpt.s_thresh_singles = 1e-5
nevpt.s_thresh_doubles = 1e-5
nevpt.method = "nevpt2"
nevpt.soc = "Breit-Pauli" # Possible methods: Breit-Pauli (BP), DKH1 (x2c-1)


class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,  -1937.93750755673, 5)
        self.assertAlmostEqual(mc.e_cas,  -1.62124228555217, 5)

    def test_prism(self):
        socutils_dir = Path(prism.__file__).parent / "socutils"
        if (not socutils_dir.exists()) or (not any(socutils_dir.iterdir())):
            print("\nsocutilis is not available. Skip soc test")
            self.skipTest('socutilis is not available')
        e_tot, e_corr, osc = nevpt.kernel()

        self.assertAlmostEqual(e_tot[0] , -1938.424357125316, 5)
        self.assertAlmostEqual(e_tot[1] , -1938.219070869459, 5)
        self.assertAlmostEqual(e_tot[2] , -1938.218011281023, 5)
        self.assertAlmostEqual(e_tot[3] , -1938.217081978226, 5)
        self.assertAlmostEqual(e_tot[4] , -1938.217081977996, 5)
        self.assertAlmostEqual(e_tot[5] , -1938.214327430154, 5)
        self.assertAlmostEqual(e_tot[6] , -1938.214327430154, 5)
        self.assertAlmostEqual(e_tot[7] , -1938.213267559177, 5)
        self.assertAlmostEqual(e_tot[8] , -1938.213267559084, 5)
        self.assertAlmostEqual(e_tot[9] , -1938.213096481312, 5)
        self.assertAlmostEqual(e_tot[10], -1938.106166585678, 5)

        

        self.assertAlmostEqual(osc[0], 0.00000000, 5)
        self.assertAlmostEqual(osc[1], 0.00016580, 5)
        self.assertAlmostEqual(osc[2], 0.00000000, 5)
        self.assertAlmostEqual(osc[3], 0.00000000, 5)
        self.assertAlmostEqual(osc[4], 0.00000000, 5)
        self.assertAlmostEqual(osc[5], 0.00000000, 5)
        self.assertAlmostEqual(osc[6], 0.00000000, 5)
        self.assertAlmostEqual(osc[7], 0.00000000, 5)
        self.assertAlmostEqual(osc[8], 0.00000000, 5)
        self.assertAlmostEqual(osc[9], 0.59272385, 5)

if __name__ == "__main__":
    print("SOC-QD-NEVPT2 test")
    unittest.main()



