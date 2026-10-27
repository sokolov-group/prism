'''
SOC UADC calculation for ZnH magnetic properties
'''

import numpy as np
import math
import pyscf.gto
import pyscf.scf
from pyscf import adc
from pyscf import scf
from pyscf.adc import uadc_ea
import prism.interface


np.set_printoptions(suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [
            ['Zn', (0.0, 0.0, 0.0)],
            ['H', (0.0,  0,  1.595 )]]
mol.basis = 'def2-tzvp'
mol.symmetry = False
mol.spin = 0
mol.verbose = 4
mol.charge = +1
mol.build()


mf = scf.UHF(mol).x2c()
mf.conv_tol = 1e-12
mf.kernel()

myadc = adc.ADC(mf)
myadc.method_type = "ea" #Can be ea, ip, ee
myadc.kernel(nroots=6)

interface = prism.interface.PYSCF(mf)
adc_prism = interface.import_adc(myadc)
adc_prism.soc = "BP" # Possible methods: Breit-Pauli (BP), DKH1 (x2c-1)
adc_prism.analyze_evec_soc = True #Print eigenvector analysis based on spin-free wavefunction
adc_prism.gtensor = True     #Turn on g-tensor calculation
adc_prism.mag_av =     True  #Turn on powder magnetization calculation
adc_prism.sus_av =     True  #Turn on powder susceptibility calculation
adc_prism.mag_vec =    True  #Turn on magnetization calculation under spectific direction of magnetic field
adc_prism.sus_tensor = True  #Turn on susceptibility calculation under spectific direction of magnetic field
adc_prism.run_soc()


