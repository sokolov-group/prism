#!/usr/bin/env python

'''
DMET with the Prism CASSCF, PC-NEVPT2 and CVS-IP-MR-ADC solvers for N2

Each N atom is a fragment; the Schmidt bath completes the embedding cluster and
the remaining orbitals are frozen into the core. Every solver is run again on the
whole molecule for comparison.
'''

import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc
import prism.nevpt
from prism.dmet import DMET, LocalIntegrals, make_fragments

_eV = 27.21138602

mol = pyscf.gto.Mole()
mol.atom = [['N', (0, 0, 0)], ['N', (0, 0, 1.5)]]
mol.basis = 'sto-3g'
mol.verbose = 4
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0], [1]])

# Embedded CASSCF. print_bath_spectrum lists the occupation, the deviation from
# 0 or 2, and the entropy of the orbitals on either side of the bath cut.
dmet = DMET(ints, frags, False, method='CASSCF', ncas=4, nelecas=4,
            print_bath_spectrum=True)
e_dmet = dmet.oneshot()

mc_ref = pyscf.mcscf.CASSCF(mf, 4, 4)
mc_ref.verbose = 0
mc_ref.kernel()
print("\nCASSCF(4,4) total energy   DMET %.10f   full molecule %.10f"
      % (e_dmet, mc_ref.e_tot))

# The first fragment's embedded 1-RDM mapped back to the AO basis of the
# molecule, plus the density frozen into the core.
dm_ao = dmet.to_ao(dmet.imp_rdm1[0]) + dmet.core_dm_ao(0)
print("Electrons in that density: %.6f of %d"
      % (np.trace(dm_ao @ mol.intor('int1e_ovlp')), mol.nelectron))

# PC-NEVPT2 (Prism fully internally contracted NEVPT2) on a state-averaged
# reference; natorb picks the active space from a 4-orbital superset window.
# For the quasidegenerate variant use method='QD-NEVPT2'.
dmet = DMET(ints, frags, False, method='PC-NEVPT2',
            ncas=4, nelecas=4, sa_nstates=2, cas_select='natorb',
            scf_stability=True)
dmet.oneshot()
e_states = dmet.pcnevpt2_results[0]['e_tot']

mc_sa = pyscf.mcscf.CASSCF(mf, 4, 4).state_average_([0.5, 0.5])
mc_sa.verbose = 0
mc_sa.kernel()
nevpt_ref = prism.nevpt.NEVPT(prism.interface.PYSCF(mf, mc_sa))
nevpt_ref.verbose = 0
e_ref = nevpt_ref.kernel()[0]

# Embedded total energies leave out the nuclear repulsion, so compare excitations.
print("\nPC-NEVPT2 excitation 0 -> 1   DMET %.4f eV   full molecule %.4f eV"
      % ((e_states[1] - e_states[0]) * _eV, (e_ref[1] - e_ref[0]) * _eV))

# CVS-IP-MR-ADC(2) for the N 1s of the fragment atom. MR-ADC takes a
# single-state reference, so the roots are requested through mradc_kwargs.
dmet = DMET(ints, frags, False, method='MR-ADC', ncas=4, nelecas=4,
            mradc_kwargs={'method_type': 'cvs-ip', 'ncvs': 1, 'nroots': 3})
dmet.oneshot()
e_ip = dmet.mradc_results[0]['e_exc']

# The molecule has two N 1s orbitals, which mix into a close pair; the fragment
# has the one localized on its own atom.
adc_ref = prism.mr_adc.MRADC(prism.interface.PYSCF(mf, mc_ref))
adc_ref.verbose = 0
adc_ref.method_type = 'cvs-ip'
adc_ref.ncvs = 2
adc_ref.nroots = 3
e_ip_ref = np.atleast_1d(adc_ref.kernel()[0])

print("\nCVS-IP-MR-ADC(2) N 1s ionization energies (eV):")
print("  DMET fragment: %s" % np.round(e_ip, 4))
print("  full molecule: %s" % np.round(e_ip_ref, 4))
