# Copyright 2025 Prism Developers. All Rights Reserved.
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

# CASSCF calculation
n_states = 4 
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6
emc = mc.mc1step()[0]

# NEVPT2 calculation
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
nevpt = prism.nevpt.NEVPT(interface)
nevpt.compute_singles_amplitudes = False
nevpt.semi_internal_projector = "gno"
nevpt.s_thresh_singles = 1e-6
nevpt.s_thresh_doubles = 1e-6
nevpt.method = "qd-nevpt2"
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

        # Excited States
        es1_1rdm = nevpt.make_rdm1(L = 1, R = 1)
        es2_1rdm = nevpt.make_rdm1(L = 2, R = 2)
        es3_1rdm = nevpt.make_rdm1(L = 3, R = 3)

        # Transition 1RDMS
        tr1_1rdm = nevpt.make_rdm1(L = 0, R = 1)
        tr2_1rdm = nevpt.make_rdm1(L = 0, R = 2)
        tr3_1rdm = nevpt.make_rdm1(L = 0, R = 3)
        
        # Store norms for different type check
        tr1_norm = np.linalg.norm(tr1_1rdm)
        tr2_norm = np.linalg.norm(tr2_1rdm)
        tr3_norm = np.linalg.norm(tr3_1rdm)
        
        self.assertAlmostEqual(np.trace(gs_1rdm), nevpt.nelec, 6)
        self.assertAlmostEqual(np.trace(es1_1rdm), nevpt.nelec, 6)
        self.assertAlmostEqual(np.trace(es2_1rdm), nevpt.nelec, 6)
        self.assertAlmostEqual(np.trace(es3_1rdm), nevpt.nelec, 6)
        
        self.assertAlmostEqual(np.trace(tr1_1rdm), 0, 6)
        self.assertAlmostEqual(np.trace(tr2_1rdm), 0, 6)
        self.assertAlmostEqual(np.trace(tr3_1rdm), 0, 6)
        
        self.assertAlmostEqual(rdms_test(gs_1rdm), 20.0268093354653, 6)
        self.assertAlmostEqual(rdms_test(es1_1rdm), 37.7195898465653, 6)
        self.assertAlmostEqual(rdms_test(es2_1rdm), 39.78540440455074, 6)
        self.assertAlmostEqual(rdms_test(es3_1rdm), 38.61663768801751, 6)
        
        # Using 'all' flag
        # Ground state
        rdms = nevpt.make_rdm1(type = 'all')
        
        # Ground state
        gs_1rdm = rdms[0,0]
        
        # Excited States
        es1_1rdm = rdms[1,1]
        es2_1rdm = rdms[2,2]
        es3_1rdm = rdms[3,3]

        # Transition 1RDMS
        tr1_1rdm = rdms[0,1]
        tr2_1rdm = rdms[0,2]
        tr3_1rdm = rdms[0,3]
        
        self.assertAlmostEqual(np.trace(gs_1rdm), nevpt.nelec, 6)
        self.assertAlmostEqual(np.trace(es1_1rdm), nevpt.nelec, 6)
        self.assertAlmostEqual(np.trace(es2_1rdm), nevpt.nelec, 6)
        self.assertAlmostEqual(np.trace(es3_1rdm), nevpt.nelec, 6)
        
        self.assertAlmostEqual(np.trace(tr1_1rdm), 0, 6)
        self.assertAlmostEqual(np.trace(tr2_1rdm), 0, 6)
        self.assertAlmostEqual(np.trace(tr3_1rdm), 0, 6)
        
        # Check different type returns same transition 1RDMS
        self.assertAlmostEqual(np.linalg.norm(tr1_1rdm), tr1_norm, 6)
        self.assertAlmostEqual(np.linalg.norm(tr2_1rdm), tr2_norm, 6)
        self.assertAlmostEqual(np.linalg.norm(tr3_1rdm), tr3_norm, 6)
        
        self.assertAlmostEqual(rdms_test(gs_1rdm), 20.0268093354653, 6)
        self.assertAlmostEqual(rdms_test(es1_1rdm), 37.7195898465653, 6)
        self.assertAlmostEqual(rdms_test(es2_1rdm), 39.78540440455074, 6)
        self.assertAlmostEqual(rdms_test(es3_1rdm), 38.61663768801751, 6)
        
        # Using ss flag
        rdms = nevpt.make_rdm1(type = 'ss')
        
        # Ground state
        gs_1rdm = rdms[0]

        # Excited States
        es1_1rdm = rdms[1]
        es2_1rdm = rdms[2]
        es3_1rdm = rdms[3]
        
        self.assertAlmostEqual(np.trace(gs_1rdm), nevpt.nelec, 6)
        self.assertAlmostEqual(np.trace(es1_1rdm), nevpt.nelec, 6)
        self.assertAlmostEqual(np.trace(es2_1rdm), nevpt.nelec, 6)
        self.assertAlmostEqual(np.trace(es3_1rdm), nevpt.nelec, 6)
        
        self.assertAlmostEqual(rdms_test(gs_1rdm), 20.0268093354653, 6)
        self.assertAlmostEqual(rdms_test(es1_1rdm), 37.7195898465653, 6)
        self.assertAlmostEqual(rdms_test(es2_1rdm), 39.78540440455074, 6)
        self.assertAlmostEqual(rdms_test(es3_1rdm), 38.61663768801751, 6)

if __name__ == "__main__":
    print("QD-NEVPT2 test")
    unittest.main()
