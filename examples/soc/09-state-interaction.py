'''
Loading spin-free information to run state-interaction codes
'''

import numpy as np
import math
import pyscf.gto
import pyscf.scf
from pyscf import adc
from pyscf import scf
from pyscf.adc import uadc_ea
import prism.interface
import prism.libsoc

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

interface = prism.interface.PYSCF(mf)
adc_prism = interface.import_adc(myadc, E,U,P,X)
adc_prism.analyze_evec_soc = True
adc_prism.soc = 'BP'   # Possible methods: Breit-Pauli (BP), DKH1 (x2c-1)
adc_prism.save_sf = True #Save inofrmation for SOC calculation
adc_prism.gtensor =    True #Turn on g-tensor calculation
adc_prism.mag_av =     True #Turn on powder magnetization calculation
adc_prism.sus_av =     True #Turn on powder susceptibility calculation
adc_prism.mag_vec =    True #Turn on magnetization calculation under spectific direction of magnetic field
adc_prism.sus_tensor = True #Turn on susceptibility calculation under spectific direction of magnetic field
adc_prism.run_soc()


E_ref, rdm_aabb_ref, S_ref, ms_ref, mo = prism.libsoc.read_checkpoint("09-state-interaction.h5")
interface_sf = prism.libsoc.SF_OBJECT(mol, E_ref, rdm_aabb_ref, S_ref, mo) #, ms_ref=ms_ref)
interface_sf.spin = 1 #Becasue EA-ADC add one electron
interface_sf.soc = 'BP'   # Possible methods: Breit-Pauli (BP), DKH1 (x2c-1)
interface_sf.gtensor =    True      #Turn on g-tensor calculation
interface_sf.mag_av =     True      #Turn on powder magnetization calculation
interface_sf.sus_av =     True      #Turn on powder susceptibility calculation
interface_sf.mag_vec =    True      #Turn on magnetization calculation under spectific direction of magnetic field
interface_sf.sus_tensor = True      #Turn on susceptibility calculation under spectific direction of magnetic field
interface_sf.kernel()
