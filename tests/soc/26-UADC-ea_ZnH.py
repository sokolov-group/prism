import unittest
import numpy as np
import pyscf.gto
import pyscf.scf
from pyscf import adc
from pyscf import scf
import prism.interface


np.set_printoptions(suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [
            ['Zn', (0.0, 0.0, 0.0)],
            ['H', (0.0,  0,  1.595 )]]
mol.basis = 'def2-tzvp'
mol.symmetry = False
mol.spin = 0
mol.verbose = 1
mol.charge = +1
mol.build()


mf = scf.UHF(mol).x2c()
mf.conv_tol = 1e-12
mf.kernel()

myadc = adc.ADC(mf)
myadc.method_type = "ea"
E,U,P,X = myadc.kernel(nroots=10)

interface = prism.interface.PYSCF(mf)
adc_prism = interface.import_adc(myadc)
adc_prism.soc = "BP"
adc_prism.gtensor = True




class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(E[0],  -0.2822636116, 5)
        self.assertAlmostEqual(E[1],  -0.2822636074, 5)
        self.assertAlmostEqual(E[2],  -0.1729205553, 5)
        self.assertAlmostEqual(E[3],  -0.1729205553, 5)
        self.assertAlmostEqual(E[4],  -0.1729205541, 5)
        self.assertAlmostEqual(E[5],  -0.1729205541, 5)
        self.assertAlmostEqual(E[6],  -0.0508731510, 5)
        self.assertAlmostEqual(E[7],  -0.0508731493, 5)
        self.assertAlmostEqual(E[8],  -0.0117694232, 5)
        self.assertAlmostEqual(E[9],  -0.0117694232, 5)


        

    def test_prism(self):

        en_soc, osc_str_soc = adc_prism.run_soc()

        self.assertAlmostEqual(en_soc[0],  -0.282269864321, 5)
        self.assertAlmostEqual(en_soc[1],  -0.282269864321, 5)
        self.assertAlmostEqual(en_soc[2],  -0.173599554213, 5)
        self.assertAlmostEqual(en_soc[3],  -0.173599554213, 5)
        self.assertAlmostEqual(en_soc[4],  -0.172235398435, 5)
        self.assertAlmostEqual(en_soc[5],  -0.172235398435, 5)
        self.assertAlmostEqual(en_soc[6],  -0.050873144086, 5)
        self.assertAlmostEqual(en_soc[7],  -0.050873144086, 5)
        self.assertAlmostEqual(en_soc[8],  -0.011769333715, 5)
        self.assertAlmostEqual(en_soc[9],  -0.011769333715, 5)

        g_factor_all = adc_prism.properties_soc["g-factors"]
        g_factor = g_factor_all[0]
        self.assertAlmostEqual(g_factor[0], 1.985757, 5)
        self.assertAlmostEqual(g_factor[1], 1.985757, 5)
        self.assertAlmostEqual(g_factor[2], 2.002204, 5)


        


if __name__ == "__main__":
    print("ADC-ea test")
    unittest.main()
