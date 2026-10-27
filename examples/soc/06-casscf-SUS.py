'''
SOC CASSCF calculation for ZnH magnetic susceptibility and  magnetization
'''

import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc
import prism.nevpt
import time

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)
mol = pyscf.gto.Mole()
mol.atom = [
            ['Zn', (0.0, 0.0, 0.0)],
            ['H', (0.0,  0,  1.595 )]]
mol.basis = 'def2-tzvp'
mol.symmetry = False
mol.spin = 1
mol.verbose = 4
mol.build()


# RDFT calculation
mf = pyscf.scf.RKS(mol).x2c()
mf.xc = "bp86"
ehf = mf.scf()
mf.analyze()

# SA-CASSCF calculation
n_states = 3
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 5, 3).state_average_(weights)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6
emc = mc.mc1step()[0]
mc.analyze()


interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')
casscf_sf = interface.import_casscf_sf() #Load interface in to SF_OBJECT for SOC calculation
casscf_sf.soc = "bp"
#Below Key is same in 04-qdnevpt2-SUS.py
casscf_sf.mag_av = True
casscf_sf.sus_av = True
casscf_sf.mag_vec = True
casscf_sf.sus_tensor = True
casscf_sf.step_h_s = 0.001 

Bs_list = []
for i in range(15):
    H = i * 0.5
    Bs_list.append(H)

T_list = []
for i in range(21):
    T = 14.75 * i + 5
    T_list.append(T)

casscf_sf.Bs_powder_M = Bs_list 
casscf_sf.T_powder_M = [1.8]
casscf_sf.T_powder_chi = T_list
casscf_sf.Bs_powder_chi = [0.1]
casscf_sf.B_vec_M = [0,0,1]
casscf_sf.Bs_vec_M = Bs_list 
casscf_sf.T_vec_M = [1.8]
casscf_sf.B_vec_chi = [0,0,1]
casscf_sf.Bs_vec_chi = [0.1]
casscf_sf.T_vec_chi = [5,100,200,250]
casscf_sf.run_soc()
