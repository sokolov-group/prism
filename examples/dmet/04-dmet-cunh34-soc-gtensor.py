#!/usr/bin/env python

'''
DMET + QD-NEVPT2 with spin-orbit coupling for [Cu(NH3)4]2+

The Cu atom is the only fragment. The ammonia ligands enter through the Schmidt
bath, and the part of the ligand density left outside the cluster is frozen into
the core. The spin-orbit and magnetic integrals are built over the whole
molecule, and the spin-orbit mean field adds that frozen density back, so the
g-tensor, the powder magnetization and the powder susceptibility are in the
molecular frame. The g-tensor is also computed on the whole molecule.
'''

import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.nevpt
from prism.dmet import DMET, LocalIntegrals, make_fragments

np.set_printoptions(suppress=True)

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
mol.spin = 1
mol.charge = 2
mol.verbose = 4
mol.build()

# Scalar-relativistic RKS calculation on the whole molecule
mf = pyscf.scf.RKS(mol).x2c()
mf.xc = "bp86"
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0]])

# Five orbitals and nine electrons from around the Fermi level of the embedded
# cluster. print_bath_spectrum reports how much of the ligand shell the bath keeps.
dmet = DMET(ints, frags, False, method='QD-NEVPT2',
            ncas=5, nelecas=9, sa_nstates=5, cas_select='energy',
            print_bath_spectrum=True,
            qdnevpt2_kwargs={'soc': 'breit-pauli', 'gtensor': True,
                             'mag_av': True, 'sus_av': True,
                             'Bs_powder_M': [0.5, 1.0, 2.0],
                             'T_powder_M': [1.8],
                             'Bs_powder_chi': [0.1],
                             'T_powder_chi': [5.0, 100.0, 300.0],
                             's_thresh_singles': 1e-10,
                             's_thresh_doubles': 1e-10})
dmet.oneshot()

# The same active space and method on the whole molecule, for comparison
mc_ref = pyscf.mcscf.CASSCF(mf, 5, 9).state_average_([0.2] * 5)
mc_ref.verbose = 0
mc_ref.kernel()
nevpt_ref = prism.nevpt.QDNEVPT(prism.interface.PYSCF(mf, mc_ref, backend='opt_einsum'))
nevpt_ref.verbose = 0
nevpt_ref.soc = 'breit-pauli'
nevpt_ref.gtensor = True
nevpt_ref.s_thresh_singles = 1e-10
nevpt_ref.s_thresh_doubles = 1e-10
nevpt_ref.kernel()

props = dmet.qdnevpt2_results[0]['nevpt'].properties
print("\ng-factors of the lowest Kramers doublet")
print("  DMET:          %s" % np.round(props['g-factors'][0], 6))
print("  full molecule: %s" % np.round(nevpt_ref.properties['g-factors'][0], 6))
print("Powder magnetization at 1.8 K, 0.5/1.0/2.0 T (Bohr magneton): %s"
      % np.round(props['M_av'][0], 6))
print("Powder susceptibility at 0.1 T, 5/100/300 K (cm3/mol): %s"
      % np.round(props['chi_av'][:, 0], 6))
