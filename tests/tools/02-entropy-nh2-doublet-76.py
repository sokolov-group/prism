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
            ['N', (0.0,  0.0, 0.0)],
            ['H', (0.0,  1.0, 0.3)],
            ['H', (0.0, -1.0, 0.3)]]
mol.basis = 'sto-3g'
mol.spin = 1
mol.build()

# ROHF calculation
mf = pyscf.scf.ROHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# CASCI calculation with two states
mc = pyscf.mcscf.CASCI(mf, 6, 7)
mc.fcisolver.nroots = 2

emc = mc.kernel()[0]
print("CASCI energies: %f %f\n" % tuple(emc))

# PRISM interface
interface = prism.interface.PYSCF(mf, mc)

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
        self.assertAlmostEqual(mc.e_tot[0], -54.823403805500, 5)

    def test_casci(self):

        s1, s2, mi = entropy.analyze(interface)

        s1_ref = [0.103090838600, 0.111997047400, 0.062445697600,
                  0.033246424000, 0.152202133400, 0.093920878100]
        for i in range(interface.ncas):
            self.assertAlmostEqual(s1[0, i], s1_ref[i], 5)
        self.assertAlmostEqual(mi[0, 1, 5], 0.050554885400, 5)
        self.assertAlmostEqual(mi[0, 1, 5], mi[0].max(), 10)

    def test_natural(self):

        # Both states in the natural orbitals of state 1
        s1 = entropy.analyze(interface, basis = 'natural', basis_state = 1)[0]

        states = interface.ref_wfn
        noons, u = np.linalg.eigh(interface.compute_rdm1(states[0], states[0], interface.ref_nelecas[0]))
        u = u[:, np.argsort(-noons)]
        for k in range(2):
            self.assertAlmostEqual(abs(s1[k] - s1_from_rdms(interface, [states[k]], [1.0], u)).max(), 0.0, 10)

if __name__ == "__main__":
    print("Orbital entropy test")
    unittest.main()
