#!/usr/bin/env python3
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
from pathlib import Path

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

# SSCASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 6, 8) 
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6
emc = mc.mc1step()[0]
mc.analyze()


# NEVPT2 with all electrons correlated
interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')


class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,  -515.206651213314, 5)
        self.assertAlmostEqual(mc.e_cas,   -16.1552700236644, 5)

    def test_prism(self):
        socutils_dir = Path(prism.__file__).parent / "socutils"
        if (not socutils_dir.exists()) or (not any(socutils_dir.iterdir())):
            print("\nsocutilis is not available. Skip soc test")
            self.skipTest('socutilis is not available')
        e_tot, osc = interface.run_soc("BP")

        self.assertAlmostEqual(e_tot[0],   -515.206651213314      , 5)
        self.assertAlmostEqual(e_tot[1],   -515.206651213314      , 5)
        self.assertAlmostEqual(e_tot[2],   -515.206651213314      , 5)
        
        self.assertAlmostEqual(osc[0],   0 ,  5)
        self.assertAlmostEqual(osc[1],   0 ,  5)
 


if __name__ == "__main__":
    print("SOC-CASSCF test")
    unittest.main()



