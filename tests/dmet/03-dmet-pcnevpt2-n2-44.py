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
from prism.dmet import DMET, LocalIntegrals, make_fragments

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [['N', (0, 0, 0)], ['N', (0, 0, 1.5)]]
mol.basis = 'sto-3g'
mol.verbose = 4
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# DMET with one N atom per fragment, PC-NEVPT2 on a two-state reference.
# The state energies are for the embedded cluster and leave out the nuclear repulsion.
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0], [1]])
dmet = DMET(ints, frags, False, method='PC-NEVPT2', ncas=4, nelecas=4,
            sa_nstates=2, sc_method='NONE', casscf_kwargs={'conv_tol': 1e-11})

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mf.e_tot, -107.272448501206, 5)

    def test_prism(self):

        dmet.oneshot()
        res = dmet.pcnevpt2_results[0]

        self.assertAlmostEqual(res['e_tot'][0],  -65.569537575318, 5)
        self.assertAlmostEqual(res['e_tot'][1],  -65.492655224603, 5)
        self.assertAlmostEqual(res['e_corr'][0],  -0.052986669482, 5)
        self.assertAlmostEqual(res['e_corr'][1],  -0.039576578599, 5)

if __name__ == "__main__":
    print("DMET-PC-NEVPT2 test")
    unittest.main()
