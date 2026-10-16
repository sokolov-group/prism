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
import prism.interface_soc


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

E,U,P,X = myadc.kernel(nroots=6)

interface = prism.interface_soc.PYSCF_ADC(mf,myadc,E,U,P,X)
interface.analyze_evec_soc = True #Print eigenvector analysis based on spin-free wavefunction
interface.gtensor = True     #Turn on g-tensor calculation
interface.mag_av =     True  #Turn on powder magnetization calculation
interface.sus_av =     True  #Turn on powder susceptibility calculation
interface.mag_vec =    True  #Turn on magnetization calculation under spectific direction of magnetic field
interface.sus_tensor = True  #Turn on susceptibility calculation under spectific direction of magnetic field
en_soc, properties = interface.kernel()


