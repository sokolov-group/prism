import numpy as np
import unittest
import pyscf.gto
import pyscf.scf
import prism.interface
from pyscf import adc
from pyscf import scf



np.set_printoptions(suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [ 
            ['Zn', (0.0, 0.0, 0.0)],
            ['H', (0.0,  0,  1.595 )]
]
mol.basis = 'def2-tzvp' #'6-31g'
#mol.basis = bse.get_basis('ano-rcc-vtzp',fmt='nwchem',header=False) #x2c-tzvpall-2c
mol.symmetry = False
mol.spin = 1
mol.charge = 0
mol.verbose = 1
mol.build()


mf = scf.UHF(mol).x2c()
mf.conv_tol = 1e-12
#mf.max_cycle = 1
mf.kernel()
mf.analyze()

myadc = adc.ADC(mf)
myadc.method_type = "ee"
myadc.compute_spin_square = True
E,U,P,X = myadc.kernel(nroots=10)



interface = prism.interface.PYSCF(mf)
interface = interface.import_adc(myadc, E,U,P,X)
interface.soc = "BP"
interface.gtensor=True


class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(E[0],  0.1083915661 , 5)
        self.assertAlmostEqual(E[1],  0.1083915661 , 5)
        self.assertAlmostEqual(E[2],  0.1742387987 , 5)
        self.assertAlmostEqual(E[3],  0.2260147521 , 5)
        self.assertAlmostEqual(E[4],  0.2260147522 , 5)
        self.assertAlmostEqual(E[5],  0.2338904159 , 5)
        self.assertAlmostEqual(E[6],  0.2502417323 , 5)
        self.assertAlmostEqual(E[7],  0.2502417324 , 5)
        self.assertAlmostEqual(E[8],  0.2672107529 , 5)
        self.assertAlmostEqual(E[9],  0.2728717249 , 5)


        

    def test_prism(self):

        en_soc, properties = interface.run_soc()

        self.assertAlmostEqual(en_soc[0],    -1792.198980907531, 5)
        self.assertAlmostEqual(en_soc[1],    -1792.198980907530, 5)
        self.assertAlmostEqual(en_soc[2],    -1792.091243224217, 5)
        self.assertAlmostEqual(en_soc[3],    -1792.091243224217, 5)
        self.assertAlmostEqual(en_soc[4],    -1792.089912687065, 5)
        self.assertAlmostEqual(en_soc[5],    -1792.089912687064, 5)
        self.assertAlmostEqual(en_soc[6],    -1792.024806575700, 5)
        self.assertAlmostEqual(en_soc[7],    -1792.024806575700, 5)
        self.assertAlmostEqual(en_soc[8],    -1791.973157994226, 5)
        self.assertAlmostEqual(en_soc[9],    -1791.973157994224, 5)
        self.assertAlmostEqual(en_soc[10],   -1791.973028000745, 5)
        self.assertAlmostEqual(en_soc[11],   -1791.973028000743, 5)
        self.assertAlmostEqual(en_soc[12],   -1791.972892648649, 5)
        self.assertAlmostEqual(en_soc[13],   -1791.972892648649, 5)
        self.assertAlmostEqual(en_soc[14],   -1791.972752674596, 5)
        self.assertAlmostEqual(en_soc[15],   -1791.972752674596, 5)
        self.assertAlmostEqual(en_soc[16],   -1791.965337747333, 5)
        self.assertAlmostEqual(en_soc[17],   -1791.965337747332, 5)
        self.assertAlmostEqual(en_soc[18],   -1791.950948342774, 5)
        self.assertAlmostEqual(en_soc[19],   -1791.950948342772, 5)
        self.assertAlmostEqual(en_soc[20],   -1791.946602004077, 5)
        self.assertAlmostEqual(en_soc[21],   -1791.946602004076, 5)
        self.assertAlmostEqual(en_soc[22],   -1791.931375097001, 5)
        self.assertAlmostEqual(en_soc[23],   -1791.931375097000, 5)
        self.assertAlmostEqual(en_soc[24],   -1791.926058610667, 5)
        self.assertAlmostEqual(en_soc[25],   -1791.926058610667, 5)


        g_factor_all = interface.properties_soc["g-factors"]
        g_factor = g_factor_all[0]
        self.assertAlmostEqual(g_factor[0], 1.985173, 5)
        self.assertAlmostEqual(g_factor[1], 1.986004, 5)
        self.assertAlmostEqual(g_factor[2], 2.002212, 5)


        


if __name__ == "__main__":
    print("ADC-ee test")
    unittest.main()