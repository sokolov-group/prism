'''
SOC CVS-IP-UADC calculation for Ar
'''

import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
from pyscf import mcpdft
from pyscf import adc
from pyscf import scf
from pyscf.adc import uadc_ea
import prism.interface_soc


np.set_printoptions(suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [
            ['Ar', (0.0, 0.0, 0.0)]]
mol.basis = 'def2-tzvpd'
mol.symmetry = False
mol.spin = 0
mol.verbose = 4
mol.charge = 0
mol.build()


mf = scf.UHF(mol).x2c()
mf.conv_tol = 1e-12
mf.kernel()

myadc = adc.ADC(mf)
myadc.method_type = "ip"
myadc.ncvs = 5
E,U,P,X = myadc.kernel(nroots=12)

interface = prism.interface_soc.PYSCF_ADC(mf,myadc,E,U,P,X)
interface.soc = 'BP' # Possible methods: Breit-Pauli (BP), DKH1 (x2c-1)
en_soc, properties = interface.kernel()
