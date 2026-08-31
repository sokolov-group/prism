#!/usr/bin/env python3
import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc
import prism.nevpt
import time
import unittest
from pathlib import Path

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)
r = 0.96
x = r * math.sin(104.5 * math.pi/(2 * 180.0))
y = r * math.cos(104.5 * math.pi/(2 * 180.0))

mol = pyscf.gto.Mole()
mol.atom = """
Cu       0.000000000      0.000000000      0.000000000 
N        2.069792801     -0.000154321      0.000000000 
H        2.505629535      0.929497774      0.002402263 
H        2.449007235     -0.480559296     -0.826605894 
H        2.449000743     -0.484842588      0.824073697 
N        0.000226341     -2.069317025     -0.000084937 
H       -0.480514452     -2.448524769     -0.826508323 
H       -0.483911864     -2.449104555      0.824065649 
H        0.930005537     -2.504881604      0.001655634 
N       -2.069872326      0.000035799     -0.000084931 
H       -2.505703662     -0.929614259      0.000861464 
H       -2.448912402      0.483428698      0.824837149 
H       -2.449124383      0.481833183     -0.825845134 
N        0.000000000      2.069316351      0.000000000 
H        0.480213314      2.448208910      0.826866142 
H        0.484816003      2.449083940     -0.823753797 
H       -0.929693551      2.505065105     -0.002401724
"""

mol.basis = '6-31g'
mol.symmetry = False
mol.spin = 1
mol.charge = 2
mol.verbose = 1
mol.build()

# RHF calculation
mf = pyscf.scf.RKS(mol).x2c()
mf.xc = "bp86"
ehf = mf.scf()
mf.analyze()
print("SCF energy: %f\n" % ehf)

# SA-CASSCF calculation
n_states = 5
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 5, 9).state_average_(weights)
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

nevpt.gtensor = True
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
nevpt.compute_singles_amplitudes = False


class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,   -1874.52736493757, 5)
        self.assertAlmostEqual(mc.e_cas,   -44.5993631435581, 5)

    def test_prism(self):
        socutils_dir = Path(prism.__file__).parent / "socutils"
        if (not socutils_dir.exists()) or (not any(socutils_dir.iterdir())):
            print("\nsocutilis is not available. Skip soc test")
            self.skipTest('socutilis is not available')
        e_tot, e_corr, osc = nevpt.kernel()

        #Prism
        self.assertAlmostEqual(e_tot[0], -1875.255627300661  , 5)
        self.assertAlmostEqual(e_tot[1], -1875.255627300661  , 5)
        self.assertAlmostEqual(e_tot[2], -1875.196872480976  , 5)
        self.assertAlmostEqual(e_tot[3], -1875.196872480975  , 5)
        self.assertAlmostEqual(e_tot[4], -1875.186735324158  , 5)
        self.assertAlmostEqual(e_tot[5], -1875.186735324155  , 5)
        self.assertAlmostEqual(e_tot[6], -1875.185841074099  , 5)
        self.assertAlmostEqual(e_tot[7], -1875.185841074097  , 5)
        self.assertAlmostEqual(e_tot[8], -1875.178070630576  , 5)
        self.assertAlmostEqual(e_tot[9], -1875.178070630574  , 5)


        g_factor_all =  nevpt.properties["g-factors"]
        g_factor = g_factor_all[0]
        self.assertAlmostEqual(g_factor[0], 2.082427, 5)
        self.assertAlmostEqual(g_factor[1], 2.082485, 5)
        self.assertAlmostEqual(g_factor[2], 2.431688, 5)



        M_av = nevpt.properties["M_av"]
        chi_av = nevpt.properties["chi_av"]
        M_xyz_all = nevpt.properties["M_xyz_all"]
        chi_T_eval_all = nevpt.properties["chi_T_eval_all"]
\
        self.assertAlmostEqual(M_av[0,0],   -0.000000 , 5)
        self.assertAlmostEqual(M_av[0,1],    0.223687 , 5)
        self.assertAlmostEqual(M_av[0,2],    0.429555 , 5)
        self.assertAlmostEqual(M_av[0,3],    0.605082 , 5)
        self.assertAlmostEqual(M_av[0,4],    0.745296 , 5)
        self.assertAlmostEqual(M_av[0,5],    0.851618 , 5)
        self.assertAlmostEqual(M_av[0,6],    0.929124 , 5)
        self.assertAlmostEqual(M_av[0,7],    0.984029 , 5)
        self.assertAlmostEqual(M_av[0,8],    1.022147 , 5)
        self.assertAlmostEqual(M_av[0,9],    1.048250 , 5)
        self.assertAlmostEqual(M_av[0,10],   1.065962  , 5)
        self.assertAlmostEqual(M_av[0,11],   1.077910  , 5)
        self.assertAlmostEqual(M_av[0,12],   1.085944  , 5)
        self.assertAlmostEqual(M_av[0,13],   1.091338  , 5)
        self.assertAlmostEqual(M_av[0,14],   1.094959  , 5)

        self.assertAlmostEqual(chi_av[0,0],     0.091247   , 5)
        self.assertAlmostEqual(chi_av[1,0],     0.023155   , 5)
        self.assertAlmostEqual(chi_av[2,0],     0.013284   , 5)
        self.assertAlmostEqual(chi_av[3,0],     0.009326   , 5)
        self.assertAlmostEqual(chi_av[4,0],     0.007192   , 5)
        self.assertAlmostEqual(chi_av[5,0],     0.005857   , 5)
        self.assertAlmostEqual(chi_av[6,0],     0.004944   , 5)
        self.assertAlmostEqual(chi_av[7,0],     0.004279   , 5)
        self.assertAlmostEqual(chi_av[8,0],     0.003774   , 5)
        self.assertAlmostEqual(chi_av[9,0],     0.003377   , 5)
        self.assertAlmostEqual(chi_av[10,0],    0.003057   , 5)
        self.assertAlmostEqual(chi_av[11,0],    0.002793   , 5)
        self.assertAlmostEqual(chi_av[12,0],    0.002572   , 5)
        self.assertAlmostEqual(chi_av[13,0],    0.002384   , 5)
        self.assertAlmostEqual(chi_av[14,0],    0.002223   , 5)
        self.assertAlmostEqual(chi_av[15,0],    0.002082   , 5)
        self.assertAlmostEqual(chi_av[16,0],    0.001959   , 5)
        self.assertAlmostEqual(chi_av[17,0],    0.001850   , 5)
        self.assertAlmostEqual(chi_av[18,0],    0.001752   , 5)
        self.assertAlmostEqual(chi_av[19,0],    0.001665   , 5)
        self.assertAlmostEqual(chi_av[20,0],    0.001587   , 5)

        self.assertAlmostEqual(M_xyz_all[0,0,2],  -0.000000, 5)
        self.assertAlmostEqual(M_xyz_all[0,1,2],   0.271316, 5)
        self.assertAlmostEqual(M_xyz_all[0,2,2],   0.516926, 5)
        self.assertAlmostEqual(M_xyz_all[0,3,2],   0.720030, 5)
        self.assertAlmostEqual(M_xyz_all[0,4,2],   0.875788, 5)
        self.assertAlmostEqual(M_xyz_all[0,5,2],   0.988481, 5)
        self.assertAlmostEqual(M_xyz_all[0,6,2],   1.066633, 5)
        self.assertAlmostEqual(M_xyz_all[0,7,2],   1.119261, 5)
        self.assertAlmostEqual(M_xyz_all[0,8,2],   1.154013, 5)
        self.assertAlmostEqual(M_xyz_all[0,9,2],   1.176673, 5)
        self.assertAlmostEqual(M_xyz_all[0,10,2],  1.191338 , 5)
        self.assertAlmostEqual(M_xyz_all[0,11,2],  1.200791 , 5)
        self.assertAlmostEqual(M_xyz_all[0,12,2],  1.206879 , 5)
        self.assertAlmostEqual(M_xyz_all[0,13,2],  1.210808 , 5)
        self.assertAlmostEqual(M_xyz_all[0,14,2],  1.213355 , 5)


       #self.assertAlmostEqual(chi_T_eval_all[0,0,0],  0.369609    , 5)
       #self.assertAlmostEqual(chi_T_eval_all[1,0,0],  0.371271    , 5)
       #self.assertAlmostEqual(chi_T_eval_all[2,0,0],  0.372986    , 5)
       #self.assertAlmostEqual(chi_T_eval_all[3,0,0],  0.373810    , 5)
       #self.assertAlmostEqual(chi_T_eval_all[0,0,1],  0.369642    , 5)
       #self.assertAlmostEqual(chi_T_eval_all[1,0,1],  0.371932    , 5)
       #self.assertAlmostEqual(chi_T_eval_all[2,0,1],  0.374327    , 5)
       #self.assertAlmostEqual(chi_T_eval_all[3,0,1],  0.375500    , 5)
       #self.assertAlmostEqual(chi_T_eval_all[0,0,2],  0.375909    , 5)
       #self.assertAlmostEqual(chi_T_eval_all[1,0,2],  0.375991    , 5)
       #self.assertAlmostEqual(chi_T_eval_all[2,0,2],  0.376018    , 5)
       #self.assertAlmostEqual(chi_T_eval_all[3,0,2],  0.376059    , 5)



if __name__ == "__main__":
    print("SOC-QD-NEVPT2 test")
    unittest.main()