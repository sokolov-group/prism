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
            ['Zn', (0.0, 0.0, 0.0)],
            ['H', (0.0,  0,  1.595 )]]
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
n_states = 3
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 5, 3).state_average_(weights)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6
emc = mc.mc1step()[0]
mc.analyze()


interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')


class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,  -1791.47091104697, 5)
        self.assertAlmostEqual(mc.e_cas,  -2.06221639685918, 5)

    def test_prism(self):
        socutils_dir = Path(prism.__file__).parent / "socutils"
        if (not socutils_dir.exists()) or (not any(socutils_dir.iterdir())):
            print("\nsocutilis is not available. Skip soc test")
            self.skipTest('socutilis is not available')
        e_tot, osc = interface.run_soc("x2c-1")

        self.assertAlmostEqual(e_tot[0] ,  -1791.538028654810, 5)
        self.assertAlmostEqual(e_tot[1] ,  -1791.538028654808, 5)
        self.assertAlmostEqual(e_tot[2] ,  -1791.437972375422, 5)
        self.assertAlmostEqual(e_tot[3] ,  -1791.437972375421, 5)
        self.assertAlmostEqual(e_tot[4] ,  -1791.436732110689, 5)
        self.assertAlmostEqual(e_tot[5] ,  -1791.436732110689, 5)


        
        self.assertAlmostEqual(osc[0], 0, 5)
        self.assertAlmostEqual(osc[1],  0.05092973, 5)
        self.assertAlmostEqual(osc[2],  0.05092973, 5)
        self.assertAlmostEqual(osc[3],  0.05155707, 5)
        self.assertAlmostEqual(osc[4],  0.05155707, 5)


if __name__ == "__main__":
    print("SOC-CASSCF test")
    unittest.main()
