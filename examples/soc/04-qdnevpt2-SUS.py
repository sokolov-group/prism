'''
SOC QD-NEVPT2 calculation for ZnH Magnetization and Magnetic Susceptibility 
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
r = 0.96
x = r * math.sin(104.5 * math.pi/(2 * 180.0))
y = r * math.cos(104.5 * math.pi/(2 * 180.0))

mol = pyscf.gto.Mole()
mol.atom =[ 
[ 'Zn',  (0, 0, 0)],
[ 'H',  ( 0, 0, 1.595)]
]

mol.basis = 'def2-tzvp'
mol.symmetry = False
mol.spin = 1
mol.verbose = 4
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol).x2c()
mf.conv_tol = 1e-12
ehf = mf.scf()
mf.analyze()
print("SCF energy: %f\n" % ehf)

# SA-CASSCF calculation
n_states = 3
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 5, 3).state_average_(weights)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6
emc = mc.mc1step()[0]
mc.analyze()

# QD-NEVPT2 with all electrons correlated
interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')
nevpt = prism.nevpt.QDNEVPT(interface)
nevpt.soc = "breit-pauli"
nevpt.s_thresh_singles = 1e-10
nevpt.s_thresh_doubles = 1e-10

nevpt.mag_av = True
nevpt.sus_av = True
nevpt.mag_vec = True
nevpt.sus_tensor = True
nevpt.step_h_s = 0.001 

Bs_list = []
for i in range(15):
    H = i * 0.5
    Bs_list.append(H)

T_list = []
for i in range(21):
    T = 14.75 * i + 5
    T_list.append(T)

nevpt.Bs_powder_M = Bs_list 
nevpt.T_powder_M = [1.8]
nevpt.T_powder_chi = T_list
nevpt.Bs_powder_chi = [0.1]
nevpt.B_vec_M = [0,0,1]
nevpt.Bs_vec_M = Bs_list 
nevpt.T_vec_M = [1.8]
nevpt.B_vec_chi = [0,0,1]
nevpt.Bs_vec_chi = [0.1]
nevpt.T_vec_chi = [5,100,200,250]
e_tot, e_corr, osc = nevpt.kernel()




