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

np.set_printoptions(suppress=True)

mol = pyscf.gto.Mole()
mol.atom =[ 
[ 'N',  (0, 0, 0)],
[ 'Cl',  ( 0, 0, 1.643)] 
]
mol.basis = 'def2-tzvp'
mol.symmetry = False
mol.spin = 2
mol.verbose = 4
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
nevpt = prism.nevpt.NEVPT(interface)
nevpt.compute_singles_amplitudes = False
nevpt.semi_internal_projector = "gno"
nevpt.s_thresh_singles = 1e-8
nevpt.s_thresh_doubles = 1e-8
nevpt.method = "qd-nevpt2"
nevpt.soc = "Breit-Pauli" # Possible methods: Breit-Pauli (BP), DKH1 (x2c-1)
nevpt.verbose = 4
nevpt.gtensor = True
nevpt.origin_type = 'charge' 

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,  -514.996350232067, 6)
        self.assertAlmostEqual(mc.e_cas,  -15.3800081255977, 6)

    def test_prism(self):

        e_tot, e_corr, osc = nevpt.kernel()

        self.assertAlmostEqual(e_tot[0],    -515.619766868838 , 5)
        self.assertAlmostEqual(e_tot[1],    -515.619766533568 , 5)
        self.assertAlmostEqual(e_tot[2],    -515.619766533568 , 5)
        self.assertAlmostEqual(e_tot[3],    -515.429556192768 , 5)
        self.assertAlmostEqual(e_tot[4],    -515.429552658244 , 5)
        self.assertAlmostEqual(e_tot[5],    -515.429015984949 , 5)
        self.assertAlmostEqual(e_tot[6],    -515.429015984946 , 5)
        self.assertAlmostEqual(e_tot[7],    -515.428482321937 , 5)
        self.assertAlmostEqual(e_tot[8],    -515.428482321935 , 5)
        self.assertAlmostEqual(e_tot[9],    -515.418689643078 , 5)
        self.assertAlmostEqual(e_tot[10],    -515.418689643072 , 5)
        self.assertAlmostEqual(e_tot[11],    -515.418467090543 , 5)
        self.assertAlmostEqual(e_tot[12],    -515.418467090523 , 5)
        self.assertAlmostEqual(e_tot[13],    -515.418161757799 , 5)
        self.assertAlmostEqual(e_tot[14],    -515.418131406309 , 5)
        self.assertAlmostEqual(e_tot[15],    -515.406970609960 , 5)
        self.assertAlmostEqual(e_tot[16],    -515.406970609960 , 5)
        self.assertAlmostEqual(e_tot[17],    -515.405389371686 , 5)
        self.assertAlmostEqual(e_tot[18],    -515.405389368698 , 5)
        self.assertAlmostEqual(e_tot[19],    -515.403885463050 , 5)
        self.assertAlmostEqual(e_tot[20],    -515.403885463049 , 5)
        self.assertAlmostEqual(e_tot[21],    -515.401230794650 , 5)
        self.assertAlmostEqual(e_tot[22],    -515.401230794649 , 5)
        self.assertAlmostEqual(e_tot[23],    -515.401151774140 , 5)
        self.assertAlmostEqual(e_tot[24],    -515.385769336788 , 5)
        self.assertAlmostEqual(e_tot[25],    -515.385730126626 , 5)
        self.assertAlmostEqual(e_tot[26],    -515.385730126625 , 5)
        
        self.assertAlmostEqual(osc[2],   0.00198200 , 4)
        self.assertAlmostEqual(osc[3],   0.00197785 , 4)
        self.assertAlmostEqual(osc[4],   0.00207715 , 4)
        self.assertAlmostEqual(osc[5],   0.00207947 , 4)
        #self.assertAlmostEqual(osc[6],   0.00230175 , 4)
        self.assertAlmostEqual(osc[7],   0.00860815 , 4)
        self.assertAlmostEqual(osc[8],   0.00611759 , 4)
        self.assertAlmostEqual(osc[9],   0.00585906 , 4)
        self.assertAlmostEqual(osc[10],   0.00473170 , 4)
        self.assertAlmostEqual(osc[11],   0.00472821 , 4)
        self.assertAlmostEqual(osc[12],   0.00484512 , 4)
        self.assertAlmostEqual(osc[13],   0.00489600 , 4)
        self.assertAlmostEqual(osc[14],   0.00000001 , 4)
        self.assertAlmostEqual(osc[15],   0.00000001 , 4)
        self.assertAlmostEqual(osc[16],   0.00004633 , 4)
        self.assertAlmostEqual(osc[17],   0.00003552 , 4)
        self.assertAlmostEqual(osc[18],   0.00002558 , 4)
        self.assertAlmostEqual(osc[19],   0.00002558 , 4)
        self.assertAlmostEqual(osc[20],   0.00558477 , 4)
        self.assertAlmostEqual(osc[21],   0.00558477 , 4)
        self.assertAlmostEqual(osc[22],   0.00001929 , 4)
        self.assertAlmostEqual(osc[23],   0.04119364 , 4)
        self.assertAlmostEqual(osc[24],   0.03961083 , 4)
        self.assertAlmostEqual(osc[25],   0.03961083 , 4)

        self.assertAlmostEqual(nevpt.g_factor[0], 2.002306, 5)
        self.assertAlmostEqual(nevpt.g_factor[1], 2.007136, 5)
        self.assertAlmostEqual(nevpt.g_factor[2], 2.007136, 5)

if __name__ == "__main__":
    print("SOC-QD-NEVPT2 test")
    unittest.main()



