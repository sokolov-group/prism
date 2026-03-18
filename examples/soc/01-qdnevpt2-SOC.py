import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.nevpt

np.set_printoptions(suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [
            ['B', (0.0, 0.0, 0.0)]
            ]
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
mc = pyscf.mcscf.CASSCF(mf, 4, 3).state_average_(weights)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6
emc = mc.mc1step()[0]
mc.analyze()


# QD-NEVPT2 with all electrons correlated
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
nevpt = prism.nevpt.NEVPT(interface)
nevpt.method = "qd-nevpt2"
nevpt.soc = "DKH1" # Possible methods: Breit-Pauli (BP), DKH1 (x2c-1)
nevpt.verbose = 4
nevpt.kernel()

