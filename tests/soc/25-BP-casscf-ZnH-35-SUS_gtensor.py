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
mol = pyscf.gto.Mole()
mol.atom = [
            ['Zn', (0.0, 0.0, 0.0)],
            ['H', (0.0,  0,  1.595 )]]
mol.basis = 'def2-tzvp'
mol.symmetry = False
mol.spin = 1
mol.verbose = 1
mol.build()


# RHF calculation
mf = pyscf.scf.RHF(mol) #.x2c()
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
interface.gtensor = True
interface.soc = "bp"
interface.mag_av = True
interface.sus_av = True
interface.mag_vec = True
interface.sus_tensor = True
interface.step_h_s = 0.001 

Bs_list = []
for i in range(15):
    H = i * 0.5
    Bs_list.append(H)

T_list = []
for i in range(5):
    T = i + 1
    T_list.append(T)

interface.Bs_powder_M = Bs_list 
interface.T_powder_M = [4]
interface.T_powder_chi = T_list
interface.Bs_powder_chi = [0.0001]
interface.B_vec_M = [0,0,1]
interface.Bs_vec_M = Bs_list 
interface.T_vec_M = [1.8]
interface.B_vec_chi = [1,0,0]
interface.Bs_vec_chi = [1]
interface.T_vec_chi = [1,2,3,4]


class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,    -1778.28856078445, 5)
        self.assertAlmostEqual(mc.e_cas,   -2.04925258542517, 5)

    def test_prism(self):
        socutils_dir = Path(prism.__file__).parent / "socutils"
        if (not socutils_dir.exists()) or (not any(socutils_dir.iterdir())):
            print("\nsocutilis is not available. Skip soc test")
            self.skipTest('socutilis is not available')
        e_tot, osc =  interface.run_soc()

        #Prism
        self.assertAlmostEqual(e_tot[0], -1778.354366943675  , 5)
        self.assertAlmostEqual(e_tot[1], -1778.354366943675  , 5)
        self.assertAlmostEqual(e_tot[2], -1778.256305429052  , 5)
        self.assertAlmostEqual(e_tot[3], -1778.256305429051  , 5)
        self.assertAlmostEqual(e_tot[4], -1778.255009980629  , 5)
        self.assertAlmostEqual(e_tot[5], -1778.255009980629  , 5)

        g_factor_all = interface.properties_cas["g-factors"]
        g_factor = g_factor_all[0]
        self.assertAlmostEqual(g_factor[0], 1.983961, 5)
        self.assertAlmostEqual(g_factor[1], 1.983961, 5)
        self.assertAlmostEqual(g_factor[2], 2.002193, 5)

        M_av = interface.properties_cas["M_av"]
        chi_av = interface.properties_cas["chi_av"]
        M_xyz_all = interface.properties_cas["M_xyz_all"]
        chi_T_eval_all = interface.properties_cas["chi_T_eval_all"]

        self.assertAlmostEqual(M_av[0,0],   -0.000000  , 5)
        self.assertAlmostEqual(M_av[0,1],    0.082948  , 5)
        self.assertAlmostEqual(M_av[0,2],    0.164752  , 5)
        self.assertAlmostEqual(M_av[0,3],    0.244328  , 5)
        self.assertAlmostEqual(M_av[0,4],    0.320714  , 5)
        self.assertAlmostEqual(M_av[0,5],    0.393103  , 5)
        self.assertAlmostEqual(M_av[0,6],    0.460878  , 5)
        self.assertAlmostEqual(M_av[0,7],    0.523615  , 5)
        self.assertAlmostEqual(M_av[0,8],    0.581081  , 5)
        self.assertAlmostEqual(M_av[0,9],    0.633214  , 5)
        self.assertAlmostEqual(M_av[0,10],   0.680096   , 5)
        self.assertAlmostEqual(M_av[0,11],   0.721926   , 5)
        self.assertAlmostEqual(M_av[0,12],   0.758987   , 5)
        self.assertAlmostEqual(M_av[0,13],   0.791619   , 5)
        self.assertAlmostEqual(M_av[0,14],   0.820193   , 5)

        self.assertAlmostEqual(chi_av[0,0],     0.371438   , 5)
        self.assertAlmostEqual(chi_av[1,0],     0.185724   , 5)
        self.assertAlmostEqual(chi_av[2,0],     0.123819   , 5)
        self.assertAlmostEqual(chi_av[3,0],     0.092867   , 5)
        self.assertAlmostEqual(chi_av[4,0],     0.074296   , 5)

        self.assertAlmostEqual(M_xyz_all[0,0,2] ,  0.000000 , 5)
        self.assertAlmostEqual(M_xyz_all[0,1,2] ,  0.184852 , 5)
        self.assertAlmostEqual(M_xyz_all[0,2,2] ,  0.357514 , 5)
        self.assertAlmostEqual(M_xyz_all[0,3,2] ,  0.508813 , 5)
        self.assertAlmostEqual(M_xyz_all[0,4,2] ,  0.634151 , 5)
        self.assertAlmostEqual(M_xyz_all[0,5,2] ,  0.733238 , 5)
        self.assertAlmostEqual(M_xyz_all[0,6,2] ,  0.808716 , 5)
        self.assertAlmostEqual(M_xyz_all[0,7,2] ,  0.864600 , 5)
        self.assertAlmostEqual(M_xyz_all[0,8,2] ,  0.905111 , 5)
        self.assertAlmostEqual(M_xyz_all[0,9,2] ,  0.934031 , 5)
        self.assertAlmostEqual(M_xyz_all[0,10,2],  0.954451 , 5)
        self.assertAlmostEqual(M_xyz_all[0,11,2],  0.968757 , 5)
        self.assertAlmostEqual(M_xyz_all[0,12,2],  0.978726 , 5)
        self.assertAlmostEqual(M_xyz_all[0,13,2],  0.985646 , 5)
        self.assertAlmostEqual(M_xyz_all[0,14,2],  0.990437 , 5)






if __name__ == "__main__":
    print("SOC-CASSCF test")
    unittest.main()
