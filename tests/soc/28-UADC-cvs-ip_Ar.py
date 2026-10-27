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
            ['Ar', (0.0, 0.0, 0.0)]]
mol.basis = 'def2-tzvpd'
mol.symmetry = False
mol.spin = 0
mol.verbose = 1
mol.charge = 0
mol.build()


mf = scf.UHF(mol).x2c()
mf.conv_tol = 1e-12
mf.kernel()

myadc = adc.ADC(mf)
#myadc.conv_tol = 1e-12
#myadc.tol_residual = 1e-6
myadc.method_type = "ip"
myadc.ncvs = 5
E,U,P,X = myadc.kernel(nroots=12)

interface = prism.interface.PYSCF(mf)
adc_prism = interface.import_adc(myadc)
adc_prism.soc = "BP"


class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(E[0],    9.1418971175 , 5)
        self.assertAlmostEqual(E[1],    9.1418971175 , 5)
        self.assertAlmostEqual(E[2],    9.1418971175 , 5)
        self.assertAlmostEqual(E[3],    9.1418971175 , 5)
        self.assertAlmostEqual(E[4],    9.1418971175 , 5)
        self.assertAlmostEqual(E[5],    9.1418971175 , 5)
        self.assertAlmostEqual(E[6],   10.2966486727 , 5)
        self.assertAlmostEqual(E[7],   10.2966486727 , 5)
        self.assertAlmostEqual(E[8],   10.2966486727 , 5)
        self.assertAlmostEqual(E[9],   10.2966486727 , 5)
        self.assertAlmostEqual(E[10],   10.2966486727, 5)
        self.assertAlmostEqual(E[11],   10.2966486727, 5)


        

    def test_prism(self):

        en_soc, properties = adc_prism.run_soc()

        self.assertAlmostEqual(en_soc[0],     9.116761530460   , 5)
        self.assertAlmostEqual(en_soc[1],     9.116761530460   , 5)
        self.assertAlmostEqual(en_soc[2],     9.117054822045   , 5)
        self.assertAlmostEqual(en_soc[3],     9.117054822045   , 5)
        self.assertAlmostEqual(en_soc[4],     9.191874972369   , 5)
        self.assertAlmostEqual(en_soc[5],     9.191874972369   , 5)
        self.assertAlmostEqual(en_soc[6],    10.296187840513   , 5)
        self.assertAlmostEqual(en_soc[7],    10.296187840513   , 5)
        self.assertAlmostEqual(en_soc[8],    10.296463218025   , 5)
        self.assertAlmostEqual(en_soc[9],    10.296463218025   , 5)
        self.assertAlmostEqual(en_soc[10],   10.296648347970   , 5)
        self.assertAlmostEqual(en_soc[11],   10.296648347970   , 5)
        self.assertAlmostEqual(en_soc[12],   10.296649196647   , 5)
        self.assertAlmostEqual(en_soc[13],   10.296649196647   , 5)
        self.assertAlmostEqual(en_soc[14],   10.296834086349   , 5)
        self.assertAlmostEqual(en_soc[15],   10.296834086349   , 5)
        self.assertAlmostEqual(en_soc[16],   10.297109374286   , 5)
        self.assertAlmostEqual(en_soc[17],   10.297109374286   , 5)




        


if __name__ == "__main__":
    print("cvs-ADC-ip test")
    unittest.main()
