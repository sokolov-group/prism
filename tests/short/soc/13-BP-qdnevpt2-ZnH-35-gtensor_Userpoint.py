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

np.set_printoptions(suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [['Zn', (0.0, 0.0, 0.0)],
            ['H', (0.0,  0,  1.595 )]
            ]
mol.basis = 'def2-tzvp'
mol.symmetry = False
mol.spin = 1
mol.verbose = 1
mol.build()


# RDFT calculation
mf = pyscf.scf.RKS(mol).x2c()
mf.xc = "bp86"
ehf = mf.scf()
mf.analyze()

# SA-CASSCF calculation
n_states = 4
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 5, 3).state_average_(weights)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6
emc = mc.mc1step()[0]
mc.analyze()


# QD-NEVPT2 with all electrons correlated
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
nevpt = prism.nevpt.QDNEVPT(interface)
nevpt.compute_singles_amplitudes = False
nevpt.semi_internal_projector = "gno"
nevpt.s_thresh_singles = 1e-8
nevpt.s_thresh_doubles = 1e-8
nevpt.method = "nevpt2"
nevpt.soc = "Breit-Pauli" # Possible methods: Breit-Pauli (BP), DKH1 (x2c-1)
nevpt.verbose = 1
nevpt.gtensor = True
nevpt.gtensor_target_state = 1
nevpt.gtensor_origin_type = [3,0.623445,0.114514]


class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,   -1791.44304512657, 6)
        self.assertAlmostEqual(mc.e_cas,  -2.02048893196798, 6)

    def test_prism(self):

        e_tot, e_corr, osc = nevpt.kernel()

        #Prism
        self.assertAlmostEqual(e_tot[0], -1792.202766317773  , 5)
        self.assertAlmostEqual(e_tot[1], -1792.202766317773  , 5)
        self.assertAlmostEqual(e_tot[2], -1792.099589949302  , 5)
        self.assertAlmostEqual(e_tot[3], -1792.099589949302  , 5)
        self.assertAlmostEqual(e_tot[4], -1792.098209192672  , 5)
        self.assertAlmostEqual(e_tot[5], -1792.098209192671  , 5)
        self.assertAlmostEqual(e_tot[6], -1792.033455602739  , 5)
        self.assertAlmostEqual(e_tot[7], -1792.033455602739  , 5)

        self.assertAlmostEqual(osc[0], 0.0, 5)
        self.assertAlmostEqual(osc[1],  0.05283955, 5)
        self.assertAlmostEqual(osc[2],  0.05283955, 5)
        self.assertAlmostEqual(osc[3],  0.05354203, 5)
        self.assertAlmostEqual(osc[4],  0.05354203, 5)
        self.assertAlmostEqual(osc[5],  0.05010837, 5)
        self.assertAlmostEqual(osc[6],  0.05010837, 5)

        self.assertAlmostEqual(nevpt.g_factor[0], 1.985197, 5)
        self.assertAlmostEqual(nevpt.g_factor[1], 1.985566, 5)
        self.assertAlmostEqual(nevpt.g_factor[2], 2.002583, 5)






if __name__ == "__main__":
    print("SOC-QD-NEVPT2 test")
    unittest.main()

