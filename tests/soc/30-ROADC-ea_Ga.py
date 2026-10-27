import unittest
import numpy as np
import math
import pyscf.gto
from pyscf import adc
from pyscf import scf
import prism.interface



np.set_printoptions(suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [
            ['Ga', (0.0, 0.0, 0.0)]]
mol.basis = 'def2-tzvp'  
mol.symmetry = False
mol.spin = 1
mol.verbose = 1
mol.charge = +2
mol.build()


mf = scf.RHF(mol).x2c()
mf.conv_tol = 1e-12
mf.kernel()

myadc = adc.ADC(mf)
myadc.method_type = "ea"
myadc.conv_tol = 1e-12
myadc.tol_residual = 1e-6
E,U,P,X = myadc.kernel(nroots=7)

interface = prism.interface.PYSCF(mf)
adc_prism = interface.import_adc(myadc)
adc_prism.soc = "bp"
adc_prism.ms_ref_select = 1

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(E[0],   -0.7286186401 , 5)
        self.assertAlmostEqual(E[1],   -0.5228365288 , 5)
        self.assertAlmostEqual(E[2],   -0.5228365287 , 5)
        self.assertAlmostEqual(E[3],   -0.5228365287 , 5)
        self.assertAlmostEqual(E[4],   -0.4796273507 , 5)
        self.assertAlmostEqual(E[5],   -0.4796273507 , 5)
        self.assertAlmostEqual(E[6],   -0.4796273507 , 5)



        

    def test_prism(self):

        en_soc, properties = adc_prism.run_soc()

        self.assertAlmostEqual(en_soc[0],   -0.526415703991 , 5)
        self.assertAlmostEqual(en_soc[1],   -0.524626116371 , 5)
        self.assertAlmostEqual(en_soc[2],   -0.524626116369 , 5)
        self.assertAlmostEqual(en_soc[3],   -0.524626116367 , 5)
        self.assertAlmostEqual(en_soc[4],   -0.521046941128 , 5)
        self.assertAlmostEqual(en_soc[5],   -0.521046941128 , 5)
        self.assertAlmostEqual(en_soc[6],   -0.521046941126 , 5)
        self.assertAlmostEqual(en_soc[7],   -0.521046941124 , 5)
        self.assertAlmostEqual(en_soc[8],   -0.521046941124 , 5)






        


if __name__ == "__main__":
    print("ROADC-ea test")
    unittest.main()
