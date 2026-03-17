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

import prism_beta.interface 
import prism_beta.qd_nevpt 

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
mf.conv_tol = 1e-12
ehf = mf.scf()
mf.analyze()

# SA-CASSCF calculation
n_states = 8
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 6, (4,4)).state_average_(weights)
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
nevpt.origin_type = 'charge' 
#e_tot, e_corr, osc = nevpt.kernel()
#
#
#
## Run QDNEVPT2 computation
#interface4 = prism_beta.interface.PYSCF(mf, mc, True)
#qdnevpt = prism_beta.qd_nevpt.QDNEVPT(interface4)
#qdnevpt.method = "qd-nevpt2"
#qdnevpt.soc = 'bp-somf'
#qdnevpt.soc_order = 1
#qdnevpt.print_level = 6
#qdnevpt.ncasci = n_states
#qdnevpt.s_thresh_singles = 1e-5
#qdnevpt.s_thresh_doubles = 1e-5
#qdnevpt.gtensor_Heff_general = True
#qdnevpt.gtensor_EH = False
#qdnevpt.amp_avg_soc = False 
#en, s ,H_eff_BO, H_eff , g_origin = qdnevpt.kernel()
##en, s = qdnevpt.kernel()
#
#print("E_prism=")
#print(e_tot)
#print("E_beta=")
#print(en)
#
#print("osc_prism=")
#print(osc)
#print("s_beta=")
#print(s)


class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,  -515.064128148086, 6)
        self.assertAlmostEqual(mc.e_cas,  -15.4297042913512, 6)

    def test_prism(self):

        e_tot, e_corr, osc = nevpt.kernel()
        self.assertAlmostEqual(e_tot[0],    -515.619420396824 , 5)
        self.assertAlmostEqual(e_tot[1],    -515.619418128734 , 5)
        self.assertAlmostEqual(e_tot[2],    -515.619418128734 , 5)
        self.assertAlmostEqual(e_tot[3],    -515.574417580481 , 5)
        self.assertAlmostEqual(e_tot[4],    -515.574417578972 , 5)
        self.assertAlmostEqual(e_tot[5],    -515.540018354566 , 5)
        self.assertAlmostEqual(e_tot[6],    -515.429651002740 , 5)
        self.assertAlmostEqual(e_tot[7],    -515.429648167613 , 5)
        self.assertAlmostEqual(e_tot[8],    -515.429115815182 , 5)
        self.assertAlmostEqual(e_tot[9],    -515.429115815179 , 5)
        self.assertAlmostEqual(e_tot[10],    -515.428587404930 , 5)
        self.assertAlmostEqual(e_tot[11],    -515.428587404930 , 5)
        self.assertAlmostEqual(e_tot[12],    -515.420139641206 , 5)
        self.assertAlmostEqual(e_tot[13],    -515.420139641206 , 5)
        self.assertAlmostEqual(e_tot[14],    -515.419885962830 , 5)
        self.assertAlmostEqual(e_tot[15],    -515.419885962795 , 5)
        self.assertAlmostEqual(e_tot[16],    -515.419630037374 , 5)
        self.assertAlmostEqual(e_tot[17],    -515.419628339278 , 5)

        self.assertAlmostEqual(osc[2],   0.00000000 , 5)
        self.assertAlmostEqual(osc[3],   0.00000000 , 5)
        self.assertAlmostEqual(osc[4],   0.00000000 , 5)
        self.assertAlmostEqual(osc[5],   0.00177882 , 5)
        self.assertAlmostEqual(osc[6],   0.00177773 , 5)
        self.assertAlmostEqual(osc[7],   0.00186758 , 5)
        self.assertAlmostEqual(osc[8],   0.00186758 , 5)
        self.assertAlmostEqual(osc[9],   0.00197541 , 5)
        self.assertAlmostEqual(osc[10],   0.00197541 , 5)
        self.assertAlmostEqual(osc[11],   0.00559676 , 5)
        self.assertAlmostEqual(osc[12],   0.00559676 , 5)
        self.assertAlmostEqual(osc[13],   0.00571138 , 5)
        self.assertAlmostEqual(osc[14],   0.00571138 , 5)
        self.assertAlmostEqual(osc[15],   0.00580644 , 5)
        self.assertAlmostEqual(osc[16],   0.00580804 , 5)
        
        self.assertAlmostEqual(nevpt.g_factor[0], 2.002307, 5)
        self.assertAlmostEqual(nevpt.g_factor[1], 2.007252, 5)
        self.assertAlmostEqual(nevpt.g_factor[2], 2.007252, 5)




if __name__ == "__main__":
    print("SOC-QD-NEVPT2 test")
    unittest.main()