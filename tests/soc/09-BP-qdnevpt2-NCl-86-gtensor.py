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
mol.atom =[ 
[ 'N',  (0, 0, 0)],
[ 'Cl',  ( 0, 0, 1.643)] 
]
mol.basis = 'def2-tzvp'
mol.symmetry = False
mol.spin = 2
mol.verbose = 1
mol.build()


# RDFT calculation
mf = pyscf.scf.RKS(mol).x2c()
mf.xc = "bp86"
ehf = mf.scf()
mf.analyze()

# SA-CASSCF calculation
n_states = 9
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 6, 8).state_average_(weights)
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
nevpt.gtensor_origin_type = 'charge' 



class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,  -514.996350232067, 6)
        self.assertAlmostEqual(mc.e_cas,  -15.3800081256055, 6)

    def test_prism(self):

        e_tot, e_corr, osc = nevpt.kernel()

        self.assertAlmostEqual(e_tot[0],    -515.619766868831 , 5)
        self.assertAlmostEqual(e_tot[1],    -515.619766533561 , 5)
        self.assertAlmostEqual(e_tot[2],    -515.619766533561 , 5)
        self.assertAlmostEqual(e_tot[3],    -515.429556192766 , 5)
        self.assertAlmostEqual(e_tot[4],    -515.429552658242 , 5)
        self.assertAlmostEqual(e_tot[5],    -515.429015984948 , 5)
        self.assertAlmostEqual(e_tot[6],    -515.429015984943 , 5)
        self.assertAlmostEqual(e_tot[7],    -515.428482321935 , 5)
        self.assertAlmostEqual(e_tot[8],    -515.428482321932 , 5)
        self.assertAlmostEqual(e_tot[9],    -515.418689643097 , 5)
        self.assertAlmostEqual(e_tot[10],    -515.418689643091 , 5)
        self.assertAlmostEqual(e_tot[11],    -515.418467090563 , 5)
        self.assertAlmostEqual(e_tot[12],    -515.418467090541 , 5)
        self.assertAlmostEqual(e_tot[13],    -515.418161757818 , 5)
        self.assertAlmostEqual(e_tot[14],    -515.418131406329 , 5)
        self.assertAlmostEqual(e_tot[15],    -515.406970609959 , 5)
        self.assertAlmostEqual(e_tot[16],    -515.406970609958 , 5)
        self.assertAlmostEqual(e_tot[17],    -515.405389371676 , 5)
        self.assertAlmostEqual(e_tot[18],    -515.405389368705 , 5)
        self.assertAlmostEqual(e_tot[19],    -515.403885463048 , 5)
        self.assertAlmostEqual(e_tot[20],    -515.403885463048 , 5)
        self.assertAlmostEqual(e_tot[21],    -515.401230794648 , 5)
        self.assertAlmostEqual(e_tot[22],    -515.401230794648 , 5)
        self.assertAlmostEqual(e_tot[23],    -515.401151774138 , 5)
        self.assertAlmostEqual(e_tot[24],    -515.385769336790 , 5)
        self.assertAlmostEqual(e_tot[25],    -515.385730126627 , 5)
        self.assertAlmostEqual(e_tot[26],    -515.385730126627 , 5)
        
        self.assertAlmostEqual(osc[2],   0.00198267 ,  5)
        self.assertAlmostEqual(osc[3],   0.00197850 ,  5)
        self.assertAlmostEqual(osc[4],   0.00207758 ,  5)
        self.assertAlmostEqual(osc[5],   0.00207758 ,  5)
        self.assertAlmostEqual(osc[6],   0.00218643 ,  5)
        self.assertAlmostEqual(osc[7],   0.00218643 ,  5)
        self.assertAlmostEqual(osc[8],   0.00462686 ,  5)
        self.assertAlmostEqual(osc[9],   0.00462686 ,  5)
        self.assertAlmostEqual(osc[10],   0.00474121 , 5)
        self.assertAlmostEqual(osc[11],   0.00474121 , 5)
        self.assertAlmostEqual(osc[12],   0.00483669 , 5)
        self.assertAlmostEqual(osc[13],   0.00488747 , 5)
        self.assertAlmostEqual(osc[14],   0.00000000 , 5)
        self.assertAlmostEqual(osc[15],   0.00000000 , 5)
        self.assertAlmostEqual(osc[16],   0.00002976 , 5)
        self.assertAlmostEqual(osc[17],   0.00002976 , 5)
        self.assertAlmostEqual(osc[18],   0.00002469 , 5)
        self.assertAlmostEqual(osc[19],   0.00002469 , 5)
        self.assertAlmostEqual(osc[20],   0.00016620 , 5)
        self.assertAlmostEqual(osc[21],   0.00016620 , 5)
        self.assertAlmostEqual(osc[22],   0.00001925 , 5)
        self.assertAlmostEqual(osc[23],   0.04119364 , 5)
        self.assertAlmostEqual(osc[24],   0.04105223 , 5)
        self.assertAlmostEqual(osc[25],   0.04105223 , 5)

        g_factor_all = nevpt.properties["g-factors"]
        g_factor = g_factor_all[0]
        self.assertAlmostEqual(g_factor[0], 2.002306, 5)
        self.assertAlmostEqual(g_factor[1], 2.007136, 5)
        self.assertAlmostEqual(g_factor[2], 2.007136, 5)

if __name__ == "__main__":
    print("SOC-QD-NEVPT2 test")
    unittest.main()



