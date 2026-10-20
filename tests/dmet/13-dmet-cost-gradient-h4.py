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

np.set_printoptions(linewidth=150, suppress=True)

H4 = 'H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22'


def embed(groups, **kw):
    mol = pyscf.gto.Mole()
    mol.atom = H4
    mol.basis = 'sto-3g'
    mol.build(verbose=0)
    mf = pyscf.scf.RHF(mol)
    mf.verbose = 0
    mf.scf()
    ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
    dmet = DMET(ints, make_fragments(mol, ints, groups), False,
                method='FCI', sc_method='NONE', **kw)
    dmet.oneshot()
    return dmet


def jacobian_gradient(dmet, umat_flat):
    # The cost gradient through the explicit per-element RDM response.
    return 2.0 * (dmet.rdm_differences_derivative(umat_flat).T
                  @ dmet.rdm_differences(umat_flat))


def numerical_gradient(dmet, umat_flat, step=1e-6):
    grad = np.zeros(len(umat_flat))
    for k in range(len(umat_flat)):
        plus, minus = umat_flat.copy(), umat_flat.copy()
        plus[k] += step
        minus[k] -= step
        grad[k] = (dmet.cost_function(plus) - dmet.cost_function(minus)) / (2.0 * step)
    return grad


CASES = (([[0, 1], [2, 3]], {}),
         ([[0, 1]], {}),
         ([[0, 1], [2, 3]], {'fit_impurity_and_bath': True}),
         ([[0, 1], [2, 3]], {'use_density_embedding': True}),
         # The only configuration that rotates the one-electron integrals.
         ([[0, 1], [2, 3]], {'use_density_embedding': True,
                             'use_density_embedding_no': True}))


class KnownValues(unittest.TestCase):

    def test_gradient_matches_the_explicit_jacobian(self):
        for groups, kw in CASES:
            dmet = embed(groups, **kw)
            umat_flat = dmet.square2flat(dmet.umat)
            rng = np.random.RandomState(7)
            for scale in (0.0, 0.02):
                flat = umat_flat + scale * rng.randn(len(umat_flat))
                np.testing.assert_allclose(dmet.cost_function_derivative(flat),
                                           jacobian_gradient(dmet, flat),
                                           rtol=0, atol=1e-10)

    def test_gradient_matches_finite_differences(self):
        for groups, kw in CASES:
            dmet = embed(groups, **kw)
            umat_flat = dmet.square2flat(dmet.umat)
            analytic = dmet.cost_function_derivative(umat_flat)
            numerical = numerical_gradient(dmet, umat_flat)
            np.testing.assert_allclose(analytic, numerical, rtol=1e-5,
                                       atol=1e-8)


if __name__ == "__main__":
    print("Cost-function gradient of the u-matrix fit\n")
    unittest.main()
