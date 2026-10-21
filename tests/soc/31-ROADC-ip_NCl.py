import unittest
import numpy as np
import pyscf.gto
from pyscf import adc
from pyscf import scf
import prism.interface



np.set_printoptions(suppress=True)

mol = pyscf.gto.Mole()
mol.atom =[ 
[ 'N',  (0, 0, 0)],
[ 'Cl',  ( 0, 0, 1.643)] 
]
mol.basis = 'def2-tzvp'  
mol.symmetry = False
mol.spin = 1
mol.verbose = 1
mol.charge = -1
mol.build()


mf = scf.RHF(mol).x2c()
mf.conv_tol = 1e-12
mf.kernel()

myadc = adc.ADC(mf)
myadc.method_type = "ip"
myadc.conv_tol = 1e-12
myadc.tol_residual = 1e-6
E,U,P,X = myadc.kernel(nroots=10)


interface = prism.interface.PYSCF(mf)
interface = interface.import_adc(myadc, E,U,P,X)
interface.soc = "bp"
interface.ms_ref_select = 1


class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(E[0],  -0.0187504073  , 5)
        self.assertAlmostEqual(E[1],   0.0059632215  , 5)
        self.assertAlmostEqual(E[2],   0.0438963931  , 5)
        self.assertAlmostEqual(E[3],   0.1735169962  , 5)
        self.assertAlmostEqual(E[4],   0.2033391336  , 5)
        self.assertAlmostEqual(E[5],   0.2048502448  , 5)
        self.assertAlmostEqual(E[6],   0.2085930783  , 5)
        self.assertAlmostEqual(E[7],   0.2103397589  , 5)
        self.assertAlmostEqual(E[8],   0.2252637826  , 5)
        self.assertAlmostEqual(E[9],   0.4568973540  , 5)



        

    def test_prism(self):

        en_soc, properties = interface.run_soc()

        self.assertAlmostEqual(en_soc[0],    -0.018752216678 , 5)
        self.assertAlmostEqual(en_soc[1],    -0.018751335962 , 5)
        self.assertAlmostEqual(en_soc[2],    -0.018751295277 , 5)
        self.assertAlmostEqual(en_soc[3],     0.173484102349 , 5)
        self.assertAlmostEqual(en_soc[4],     0.173497808105 , 5)
        self.assertAlmostEqual(en_soc[5],     0.173502137214 , 5)
        self.assertAlmostEqual(en_soc[6],     0.202777820489 , 5)
        self.assertAlmostEqual(en_soc[7],     0.202781658344 , 5)
        self.assertAlmostEqual(en_soc[8],     0.203357059762 , 5)        
        self.assertAlmostEqual(en_soc[9],     0.204864789165 , 5)
        self.assertAlmostEqual(en_soc[10],    0.205412445886 , 5)
        self.assertAlmostEqual(en_soc[11],    0.205439906815 , 5)
        self.assertAlmostEqual(en_soc[12],    0.456898597371 , 5)
        self.assertAlmostEqual(en_soc[13],    0.456898615934 , 5)
        self.assertAlmostEqual(en_soc[14],    0.456899870512 , 5)







        


if __name__ == "__main__":
    print("ROADC-ea test")
    unittest.main()
