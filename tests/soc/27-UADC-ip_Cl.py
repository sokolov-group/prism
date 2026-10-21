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
            ['Cl', (0.0, 0.0, 0.0)]]
mol.basis = 'def2-tzvp'
mol.symmetry = False
mol.spin = 0
mol.verbose = 1
mol.charge = -1
mol.build()


mf = scf.UHF(mol).x2c()
mf.conv_tol = 1e-12
mf.kernel()

myadc = adc.ADC(mf)
myadc.method_type = "ip"
myadc.conv_tol = 1e-12
myadc.tol_residual = 1e-6
E,U,P,X = myadc.kernel(nroots=12)

interface = prism.interface.PYSCF(mf)
interface = interface.import_adc(myadc, E,U,P,X)
interface.soc = "BP"



class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(E[0],   0.1057013423  , 5)
        self.assertAlmostEqual(E[1],   0.1057013423  , 5)
        self.assertAlmostEqual(E[2],   0.1057013423  , 5)
        self.assertAlmostEqual(E[3],   0.1057013423  , 5)
        self.assertAlmostEqual(E[4],   0.1057013423  , 5)
        self.assertAlmostEqual(E[5],   0.1057013423  , 5)
        self.assertAlmostEqual(E[6],   0.5696265266  , 5)
        self.assertAlmostEqual(E[7],   0.5696265266  , 5)
        self.assertAlmostEqual(E[8],   1.0361739834  , 5)
        self.assertAlmostEqual(E[9],   1.0361739834  , 5)
        self.assertAlmostEqual(E[10],   1.0361739834 , 5)
        self.assertAlmostEqual(E[11],   1.0361739834 , 5)


        

    def test_prism(self):

        en_soc, properties = interface.run_soc()

        self.assertAlmostEqual(en_soc[0],    0.104604360853   , 5)
        self.assertAlmostEqual(en_soc[1],    0.104604360853   , 5)
        self.assertAlmostEqual(en_soc[2],    0.104604756033   , 5)
        self.assertAlmostEqual(en_soc[3],    0.104604756033   , 5)
        self.assertAlmostEqual(en_soc[4],    0.107894906036   , 5)
        self.assertAlmostEqual(en_soc[5],    0.107894906036   , 5)
        self.assertAlmostEqual(en_soc[6],    0.569626526560   , 5)
        self.assertAlmostEqual(en_soc[7],    0.569626526560   , 5)
        self.assertAlmostEqual(en_soc[8],    1.035763307532   , 5)
        self.assertAlmostEqual(en_soc[9],    1.035763307532   , 5)
        self.assertAlmostEqual(en_soc[10],   1.036584663219   , 5)
        self.assertAlmostEqual(en_soc[11],   1.036584663219   , 5)





        


if __name__ == "__main__":
    print("ADC-ip test")
    unittest.main()
