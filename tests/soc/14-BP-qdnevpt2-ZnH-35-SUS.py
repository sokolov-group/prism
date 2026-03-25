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
mol.verbose = 1
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
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
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


nevpt.compute_singles_amplitudes = True


class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,   -1791.47091104697, 6)
        self.assertAlmostEqual(mc.e_cas,  -2.06221691833025, 6)

    def test_prism(self):

        e_tot, e_corr, osc = nevpt.kernel()

        #Prism
        self.assertAlmostEqual(e_tot[0],-1792.202314822049  , 5)
        self.assertAlmostEqual(e_tot[1],-1792.202314822048  , 5)
        self.assertAlmostEqual(e_tot[2],-1792.098920578952  , 5)
        self.assertAlmostEqual(e_tot[3],-1792.098920578951  , 5)
        self.assertAlmostEqual(e_tot[4],-1792.097629143424  , 5)
        self.assertAlmostEqual(e_tot[5],-1792.097629143423  , 5)




        M_av = nevpt.properties["M_av"]
        chi_av = nevpt.properties["chi_av"]
        M_xyz_all = nevpt.properties["M_xyz_all"]
        chi_T_eval_all = nevpt.properties["chi_T_eval_all"]
\
        self.assertAlmostEqual(M_av[0,0],  -0.000000 , 5)
        self.assertAlmostEqual(M_av[0,1],   0.182779 , 5)
        self.assertAlmostEqual(M_av[0,2],   0.353635 , 5)
        self.assertAlmostEqual(M_av[0,3],   0.503566 , 5)
        self.assertAlmostEqual(M_av[0,4],   0.628012 , 5)
        self.assertAlmostEqual(M_av[0,5],   0.726620 , 5)
        self.assertAlmostEqual(M_av[0,6],   0.801921 , 5)
        self.assertAlmostEqual(M_av[0,7],   0.857818 , 5)
        self.assertAlmostEqual(M_av[0,8],   0.898445 , 5)
        self.assertAlmostEqual(M_av[0,9],   0.927523 , 5)
        self.assertAlmostEqual(M_av[0,10],   0.948108 , 5)
        self.assertAlmostEqual(M_av[0,11],   0.962567 , 5)
        self.assertAlmostEqual(M_av[0,12],   0.972669 , 5)
        self.assertAlmostEqual(M_av[0,13],   0.979700 , 5)
        self.assertAlmostEqual(M_av[0,14],   0.984582 , 5)

        self.assertAlmostEqual(chi_av[0,0],    0.074335  , 5)
        self.assertAlmostEqual(chi_av[1,0],    0.018830  , 5)
        self.assertAlmostEqual(chi_av[2,0],    0.010784  , 5)
        self.assertAlmostEqual(chi_av[3,0],    0.007557  , 5)
        self.assertAlmostEqual(chi_av[4,0],    0.005818  , 5)
        self.assertAlmostEqual(chi_av[5,0],    0.004730  , 5)
        self.assertAlmostEqual(chi_av[6,0],    0.003986  , 5)
        self.assertAlmostEqual(chi_av[7,0],    0.003444  , 5)
        self.assertAlmostEqual(chi_av[8,0],    0.003032  , 5)
        self.assertAlmostEqual(chi_av[9,0],    0.002709  , 5)
        self.assertAlmostEqual(chi_av[10,0],   0.002448  , 5)
        self.assertAlmostEqual(chi_av[11,0],   0.002233  , 5)
        self.assertAlmostEqual(chi_av[12,0],   0.002053  , 5)
        self.assertAlmostEqual(chi_av[13,0],   0.001900  , 5)
        self.assertAlmostEqual(chi_av[14,0],   0.001768  , 5)
        self.assertAlmostEqual(chi_av[15,0],   0.001653  , 5)
        self.assertAlmostEqual(chi_av[16,0],   0.001553  , 5)
        self.assertAlmostEqual(chi_av[17,0],   0.001464  , 5)
        self.assertAlmostEqual(chi_av[18,0],   0.001384  , 5)
        self.assertAlmostEqual(chi_av[19,0],   0.001313  , 5)
        self.assertAlmostEqual(chi_av[20,0],   0.001249  , 5)

        self.assertAlmostEqual(M_xyz_all[0,0,2],  0.00000000, 5)
        self.assertAlmostEqual(M_xyz_all[0,1,2],  0.18485319, 5)
        self.assertAlmostEqual(M_xyz_all[0,2,2],  0.35751669, 5)
        self.assertAlmostEqual(M_xyz_all[0,3,2],  0.50881722, 5)
        self.assertAlmostEqual(M_xyz_all[0,4,2],  0.63415541, 5)
        self.assertAlmostEqual(M_xyz_all[0,5,2],  0.73324311, 5)
        self.assertAlmostEqual(M_xyz_all[0,6,2],  0.80872181, 5)
        self.assertAlmostEqual(M_xyz_all[0,7,2],  0.86460576, 5)
        self.assertAlmostEqual(M_xyz_all[0,8,2],  0.90511724, 5)
        self.assertAlmostEqual(M_xyz_all[0,9,2],  0.93403744, 5)
        self.assertAlmostEqual(M_xyz_all[0,10,2], 0.95445736 , 5)
        self.assertAlmostEqual(M_xyz_all[0,11,2], 0.96876385 , 5)
        self.assertAlmostEqual(M_xyz_all[0,12,2], 0.97873270 , 5)
        self.assertAlmostEqual(M_xyz_all[0,13,2], 0.98565273 , 5)
        self.assertAlmostEqual(M_xyz_all[0,14,2], 0.99044370 , 5)


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