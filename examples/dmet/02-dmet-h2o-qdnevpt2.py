#!/usr/bin/env python

'''
DMET + QD-NEVPT2 for H2O: excitation energies, oscillator strengths, and
eigenvector analysis, checked against a direct (non-embedded) QD-NEVPT2.

The whole molecule is one fragment, so the embedding is exact and DMET
reproduces the direct result; both are printed side by side. Embedded total
energies lack nuclear repulsion, but excitations and oscillator strengths are
unaffected.
'''

import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.nevpt
from prism.dmet import DMET, LocalIntegrals, make_fragments

mol = pyscf.gto.Mole()
mol.atom = [['O', (0.0, 0.0, 0.0)],
            ['H', (0.0, 0.757, 0.587)],
            ['H', (0.0, -0.757, 0.587)]]
mol.basis = 'cc-pvdz'
mol.verbose = 0
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# Direct QD-NEVPT2 on the whole molecule (no embedding), for reference
mc = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_([0.25, 0.25, 0.25, 0.25])
mc.verbose = 0
mc.kernel()
qd = prism.nevpt.QDNEVPT(prism.interface.PYSCF(mf, mc))
qd.verbose = 0
e_direct = qd.kernel()[0]
qd.compute_properties()
osc_direct = qd.properties['osc_strengths']

# DMET with the whole molecule as one fragment, same active space
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0, 1, 2]])
dmet = DMET(ints, frags, False, method='QD-NEVPT2',
            ncas=6, nelecas=6, sa_nstates=4, cas_select='energy')
dmet.oneshot()
res = dmet.qdnevpt2_results[0]
e_dmet = res['e_tot']
osc_dmet = res['nevpt'].properties['osc_strengths']

# Excitation energies (eV) and oscillator strengths: DMET vs direct
print("\n  transition   dE_direct  dE_DMET   f_direct  f_DMET")
for i in range(1, 4):
    print("  0 -> %d      %8.4f  %8.4f  %8.5f  %8.5f"
          % (i, (e_direct[i] - e_direct[0]) * 27.21138602,
                (e_dmet[i] - e_dmet[0]) * 27.21138602,
                osc_direct[i - 1], osc_dmet[i - 1]))

# Dominant CI configurations and active natural occupations of the DMET states
res['nevpt'].verbose = 4
res['nevpt'].analyze()
