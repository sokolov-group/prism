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


def setup(natom, spin, scf_type):
    mol = pyscf.gto.Mole()
    mol.atom = '; '.join('H 0 0 %.2f' % (1.0 * i) for i in range(natom))
    mol.basis = 'sto-3g'
    mol.spin = spin
    mol.build(verbose=0)
    mf = getattr(pyscf.scf, scf_type)(mol)
    mf.verbose = 0
    mf.scf()
    ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'lowdin')
    dmet = DMET(ints, make_fragments(mol, ints, [[0]]), False, method='CASSCF',
                ncas=2, nelecas=natom % 2 + 2, sc_method='NONE', verbose=0)
    return mf, ints, dmet


def bath_density_error(ints, dmet):
    # Density the bath is built from, against the reference density.
    dm_bath = dmet.helper.construct_1rdm_loc(np.zeros((ints.norb, ints.norb)))
    s_loc = ints.ao2loc.T @ ints.ovlp
    return np.abs(dm_bath - s_loc @ ints.full_dm_ao @ s_loc.T).max()


class KnownValues(unittest.TestCase):

    def test_reference_spin(self):
        for natom, spin, scf_type in ((4, 0, 'RHF'), (3, 1, 'ROHF'), (4, 2, 'ROHF'),
                                      (3, 1, 'UHF'), (4, 2, 'UHF')):
            ints = setup(natom, spin, scf_type)[1]
            self.assertEqual(ints.spin, spin)

    def test_bath_density_matches_the_reference(self):
        for natom, spin in ((4, 0), (3, 1), (4, 2)):
            _, ints, dmet = setup(natom, spin, 'ROHF' if spin else 'RHF')
            self.assertLess(bath_density_error(ints, dmet), 1e-10)

    def test_solver_spin_leaves_the_bath(self):
        # A triplet solver on a singlet reference keeps the singlet bath.
        _, ints, dmet = setup(4, 0, 'RHF')
        ints.mol.spin = 2
        try:
            self.assertEqual(ints.spin, 0)
            self.assertLess(bath_density_error(ints, dmet), 1e-10)
        finally:
            ints.mol.spin = 0


if __name__ == "__main__":
    print("Bath density of open-shell references\n")
    unittest.main()
