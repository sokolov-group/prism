'''
SOC ROADC calculation for ZnH 
'''

import numpy as np
import math
import unittest
import pyscf.gto
import pyscf.mcscf
import prism.interface
import prism.nevpt
from prism.libsoc import general_somf
from pyscf import adc
from pyscf import scf
import prism.interface
'''
WARNING: Because for now there is no spinsquare function to get S quantum number for IP EA ADC
I just assume all states are triplet states and make some unreasonable result.
EE-ROADC should be fine
'''
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
mol.verbose = 4
mol.build()


mf = scf.RHF(mol).x2c()
mf.conv_tol = 1e-12
mf.kernel()
mf.analyze()

myadc = adc.ADC(mf)
myadc.method_type = "ee" #Can be "ip", "ea", "ee"
E,U,P,X = myadc.kernel(nroots=2)



interface = prism.interface.PYSCF(mf)
adc_prism = interface.import_adc(myadc, E,U,P,X)
adc_prism.soc = "bp" # Possible methods: Breit-Pauli (BP), DKH1 (x2c-1)
adc_prism.run_soc()
