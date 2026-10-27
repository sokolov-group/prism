import numpy as np
import unittest
import pyscf.gto
import prism.interface
from pyscf import adc
from pyscf import scf
import prism.interface


np.set_printoptions(suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [ 
            ['Zn', (0.0, 0.0, 0.0)],
            ['H', (0.0,  0,  1.595 )]
]
mol.basis = 'def2-tzvp' #'6-31g'
mol.symmetry = False
mol.spin = 1
mol.charge = 0
mol.verbose = 1
mol.build()


mf = scf.RHF(mol).x2c()
mf.conv_tol = 1e-12
mf.kernel()
mf.analyze()

myadc = adc.ADC(mf)
myadc.method_type = "ee"
myadc.compute_spin_square = True
E,U,P,X = myadc.kernel(nroots=2)



interface = prism.interface.PYSCF(mf)
adc_prism = interface.import_adc(myadc)
adc_prism.soc = "bp"
adc_prism.gtensor = True

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(E[0],  0.1044581335, 5)
        self.assertAlmostEqual(E[1],  0.1044581335, 5)

    def test_prism(self):

        en_soc, properties = adc_prism.run_soc()

        self.assertAlmostEqual(en_soc[0], -1792.197376841653, 5)
        self.assertAlmostEqual(en_soc[1], -1792.197376841653, 5)
        self.assertAlmostEqual(en_soc[2], -1792.093584007700, 5)
        self.assertAlmostEqual(en_soc[3], -1792.093584007698, 5)
        self.assertAlmostEqual(en_soc[4], -1792.092234646684, 5)
        self.assertAlmostEqual(en_soc[5], -1792.092234646683, 5)

        g_factor_all = adc_prism.properties_soc["g-factors"]
        g_factor = g_factor_all[0]
        self.assertAlmostEqual(g_factor[0], 1.983434, 5)
        self.assertAlmostEqual(g_factor[1], 1.983434, 5)
        self.assertAlmostEqual(g_factor[2], 2.002198, 5)


        


if __name__ == "__main__":
    print("ROADC-ee test")
    unittest.main()