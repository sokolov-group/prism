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
# Authors: Bryce Pickett <pickettosu@gmail.com>

import unittest
import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.nevpt
from prism.dmet import DMET, LocalIntegrals, make_fragments

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [['B', (0.0, 0.0, 0.0)]]
mol.basis = 'sto-3g'
mol.spin = 1
mol.build()

# ROHF calculation
mf = pyscf.scf.ROHF(mol)
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# Direct SA-CASSCF and QD-NEVPT2 with spin-orbit coupling on the whole atom
mc = pyscf.mcscf.CASSCF(mf, 4, 3).state_average_([1/3., 1/3., 1/3.])
mc.conv_tol = 1e-11
emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

nevpt = prism.nevpt.QDNEVPT(prism.interface.PYSCF(mf, mc))
nevpt.soc = "breit-pauli"
nevpt.gtensor = True

# DMET with every orbital in the impurity, which makes the embedding exact.
# The spin-orbit integrals are built over the molecule, not over the cluster.
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0]])
dmet = DMET(ints, frags, False, method='QD-NEVPT2', ncas=4, nelecas=3,
            sa_nstates=3, sc_method='NONE', casscf_kwargs={'conv_tol': 1e-11},
            qdnevpt2_kwargs={'soc': 'breit-pauli', 'gtensor': True})

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mf.e_tot, -24.148988598854, 5)
        self.assertAlmostEqual(mc.e_tot, -24.189236636954, 5)

    def test_prism(self):

        e_direct = nevpt.kernel()[0]
        g_direct = nevpt.properties['g-factors'][0]

        dmet.oneshot()
        e_dmet = dmet.qdnevpt2_results[0]['e_tot']
        g_dmet = dmet.qdnevpt2_results[0]['nevpt'].properties['g-factors'][0]

        self.assertAlmostEqual(e_direct[0], -24.189316763795, 5)
        self.assertAlmostEqual(e_dmet[0],   -24.189316763795, 5)

        self.assertAlmostEqual(g_direct[0], 0.665893666529, 5)
        self.assertAlmostEqual(g_dmet[0],   0.665893666413, 5)

        for state in range(6):
            self.assertAlmostEqual(e_dmet[state], e_direct[state], 8)
        for axis in range(3):
            self.assertAlmostEqual(g_dmet[axis], g_direct[axis], 8)

if __name__ == "__main__":
    print("DMET-SOC-QD-NEVPT2 test")
    unittest.main()
