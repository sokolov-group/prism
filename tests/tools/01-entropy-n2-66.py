# Copyright 2026 Prism Developers. All Rights Reserved.
#
# Licensed under the GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.
#
# See the License file for the specific language governing
# permissions and limitations.
#
# Available at https://github.com/sokolov-group/prism
#
# Authors: Bryce Pickett <brycepickett22@outlook.com>
#

import unittest
import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
from prism.tools import entropy

mol = pyscf.gto.Mole()
mol.atom = [
            ['N', (0.0, 0.0, 0.0)],
            ['N', (0.0, 0.0, 1.8)]]
mol.basis = 'sto-3g'
mol.symmetry = True
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# SS-CASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 6, 6)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6

emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

# SA-CASSCF calculation
weights = [0.7, 0.3]
mc_sa = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights)
mc_sa.fix_spin_(ss = 0)
mc_sa.conv_tol = 1e-11
mc_sa.conv_tol_grad = 1e-6

emc_sa = mc_sa.mc1step()[0]
print("SA-CASSCF energy: %f\n" % emc_sa)

def s1_from_rdms(interface, states, weights, u):
    '''s1 from weight-averaged RDMs in the active orbitals rotated by u.'''
    ncas, nelecas = interface.ncas, interface.ref_nelecas[0]
    rdm1a, rdm1b, d = 0, 0, 0
    for w, c in zip(weights, states):
        r1a, r1b = interface.trans_rdm1s(c, c, ncas, nelecas)
        r2 = interface.compute_rdm123(c, c, nelecas)[1]
        rdm1a = rdm1a + w * u.T @ r1a @ u
        rdm1b = rdm1b + w * u.T @ r1b @ u
        # <n_alpha n_beta> per orbital; the same-spin parts of the diagonal vanish
        d = d + w * np.einsum('pi,qi,ri,si,pqrs->i', u, u, u, u, r2) / 2
    rdm2ab = np.zeros((ncas,) * 4)
    rdm2ab[(np.arange(ncas),) * 4] = d
    return entropy.orbital_entropies_from_rdm((rdm1a, rdm1b), rdm2ab)

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot, -107.477475114900, 5)

    def test_casscf(self):

        interface = prism.interface.PYSCF(mf, mc)
        s1, s2, mi = entropy.analyze(interface)

        s1_ref = [0.604600880800, 1.088016877500, 1.088016877500,
                  1.088008068300, 1.088008068300, 0.604644967900]
        for i in range(interface.ncas):
            self.assertAlmostEqual(s1[0, i], s1_ref[i], 5)
        self.assertAlmostEqual(mi[0, 1, 4], 0.714070831400, 5)
        self.assertAlmostEqual(mi[0, 1, 4], mi.max(), 10)

        s1_rdm = s1_from_rdms(interface, interface.ref_wfn, [1.0], np.eye(interface.ncas))
        self.assertAlmostEqual(abs(s1[0] - s1_rdm).max(), 0.0, 10)

    def test_sa_casscf(self):

        interface = prism.interface.PYSCF(mf, mc_sa)
        states = interface.ref_wfn
        eye = np.eye(interface.ncas)

        # Each state
        s1 = entropy.analyze(interface)[0]
        for k in range(2):
            self.assertAlmostEqual(abs(s1[k] - s1_from_rdms(interface, [states[k]], [1.0], eye)).max(), 0.0, 10)

        # Ensemble with the SA weights
        s1 = entropy.analyze(interface, ensemble = True)[0]
        self.assertAlmostEqual(abs(s1 - s1_from_rdms(interface, states, weights, eye)).max(), 0.0, 10)

        # State 2 in the natural orbitals of state 2 and in its own natural orbitals
        s1_two = entropy.analyze(interface, basis = 'natural', basis_state = 2)[0]
        s1_own = entropy.analyze(interface, basis = 'natural', basis_state = 'own')[0]
        self.assertAlmostEqual(abs(s1_two[1] - s1_own[1]).max(), 0.0, 10)

if __name__ == "__main__":
    print("Orbital entropy test")
    unittest.main()
