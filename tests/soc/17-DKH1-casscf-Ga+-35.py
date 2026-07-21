'''
SOC CASSCF calculation for B
'''

import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc
import prism.nevpt
import time
import unittest
from pathlib import Path

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)
mol = pyscf.gto.Mole()
mol.atom = [
            ['Ga', (0.0, 0.0, 0.0)]
]
mol.basis = 'def2-tzvp'
mol.symmetry = False
mol.spin = 0
mol.charge = +1
mol.verbose = 1
mol.build()


# RDFT calculation
mf = pyscf.scf.RKS(mol).x2c()
mf.xc = "bp86"
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


interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')


class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,  -1937.93750755673, 5)
        self.assertAlmostEqual(mc.e_cas,  -1.62124228555604, 5)

    def test_prism(self):
        socutils_dir = Path(prism.__file__).parent / "socutils"
        if (not socutils_dir.exists()) or (not any(socutils_dir.iterdir())):
            print("\nsocutilis is not available. Skip soc test")
            self.skipTest('socutilis is not available')
        e_tot, osc = interface.run_soc("x2c-1")

        self.assertAlmostEqual(e_tot[0] , -1938.126199965225 , 5)
        self.assertAlmostEqual(e_tot[1] , -1937.929410775354 , 5)
        self.assertAlmostEqual(e_tot[2] , -1937.928799530262 , 5)
        self.assertAlmostEqual(e_tot[3] , -1937.927457869667 , 5)
        self.assertAlmostEqual(e_tot[4] , -1937.927457868521 , 5)
        self.assertAlmostEqual(e_tot[5] , -1937.925283797794 , 5)
        self.assertAlmostEqual(e_tot[6] , -1937.925283797792 , 5)
        self.assertAlmostEqual(e_tot[7] , -1937.921322399549 , 5)
        self.assertAlmostEqual(e_tot[8] , -1937.921322399453 , 5)
        self.assertAlmostEqual(e_tot[9] , -1937.921110107867 , 5)
        self.assertAlmostEqual(e_tot[10], -1937.785498634176 , 5)

        

        self.assertAlmostEqual(osc[0], 0.00000000, 5)
        self.assertAlmostEqual(osc[1], 0.00008858, 5)
        self.assertAlmostEqual(osc[2], 0.00000000, 5)
        self.assertAlmostEqual(osc[3], 0.00000000, 5)
        self.assertAlmostEqual(osc[4], 0.00000000, 5)
        self.assertAlmostEqual(osc[5], 0.00000000, 5)
        self.assertAlmostEqual(osc[6], 0.00000000, 5)
        self.assertAlmostEqual(osc[7], 0.00000000, 5)
        self.assertAlmostEqual(osc[8], 0.00000000, 5)
        self.assertAlmostEqual(osc[9], 0.63477774, 5)

if __name__ == "__main__":
    print("SOC-CASSCF test")
    unittest.main()
