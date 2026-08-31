#!/usr/bin/env python3
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
# Authors: Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#          Carlos E. V. de Moura <carlosevmoura@gmail.com>
#          James D. Serna <jserna456@gmail.com>

import unittest
import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.nevpt

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

r = 0.96
x = r * math.sin(104.5 * math.pi/(2 * 180.0))
y = r * math.cos(104.5 * math.pi/(2 * 180.0))

mol = pyscf.gto.Mole()
mol.atom = [
            ['O', (0.0, 0.0, 0.0)],
            ['H', (0.0,  -x,   y)],
            ['H', (0.0,   x,   y)]]
mol.basis = 'aug-cc-pvdz'
mol.symmetry = True
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)


# NEVPT2 calculation
interface = prism.interface.PYSCF(mf, backend = 'opt_einsum')
nevpt = prism.nevpt.NEVPT(interface)
nevpt.compute_singles_amplitudes = False
nevpt.semi_internal_projector = "gno"
nevpt.s_thresh_singles = 1e-6
nevpt.s_thresh_doubles = 1e-6
nevpt.keep_amplitudes = True

# Correlated 1RDM
nevpt.rdm_order = 2 

def rdms_test(dm):
    r2_int = nevpt.interface.mol.intor('int1e_r2')  
    dm_ao = nevpt.interface.einsum('pi,ij,qj->pq', nevpt.mo, dm, nevpt.mo.conj())  
    r2 = nevpt.interface.einsum('pq,pq->', r2_int, dm_ao)  
    return r2

class KnownValues(unittest.TestCase):

    def test_1rdm(self):

        e_tot, e_corr, osc = nevpt.kernel()

        # Using L,R
        # Ground state
        gs_1rdm = nevpt.make_rdm1(L = 0, R = 0)
        
        self.assertAlmostEqual(np.trace(gs_1rdm), nevpt.nelec, 6)
        self.assertAlmostEqual(rdms_test(gs_1rdm), 19.90404269, 6)

        # Using 'all' flag
        # Ground state
        rdms = nevpt.make_rdm1(type = 'all')

        # Ground state
        gs_1rdm = rdms[0,0]

        self.assertAlmostEqual(np.trace(gs_1rdm), nevpt.nelec, 6)
        self.assertAlmostEqual(rdms_test(gs_1rdm), 19.90404269, 6)

        # Using ss flag
        rdms = nevpt.make_rdm1(type = 'ss')

        # Ground state
        gs_1rdm = rdms[0]

        self.assertAlmostEqual(np.trace(gs_1rdm), nevpt.nelec, 6)
        self.assertAlmostEqual(rdms_test(gs_1rdm), 19.90404269, 6)

if __name__ == "__main__":
    print("NEVPT2 test")
    unittest.main()
