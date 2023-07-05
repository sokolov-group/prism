# Copyright 2023 Prism Developers. All Rights Reserved.
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
#

import sys
import numpy as np

class PYSCF:

    def __init__(self, mf, mc = None, opt_einsum = False):

        print_header()

        print("\nImporting Pyscf objects...\n")
        sys.stdout.flush()

        self.type = "pyscf"

        # General info
        self.nelec = mf.mol.nelectron
        self.enuc = mf.mol.energy_nuc()
        self.e_scf = mf.e_tot
        self.mf = mf
        # Maximum S^2 value of CASCI roots to keep; default to only singlet calculations
        self.spin_sq_thresh = 0

        if mc is None:
            self.reference = "scf"
            self.mo = mf.mo_coeff.copy()
            self.nmo = self.mo.shape[1]
            self.mo_energy = mf.mo_energy.copy()
        else:
            self.reference = "casscf"
            self.mo = mc.mo_coeff.copy()
            self.mo_hf = mf.mo_coeff.copy()
            self.ovlp = mf.get_ovlp(mf.mol)
            self.nmo = self.mo.shape[1]
            self.ncore = mc.ncore
            self.ncas = mc.ncas
            self.nextern = self.nmo - self.ncore - self.ncas
            self.nelecas = mc.nelecas
            self.e_casscf = mc.e_tot
            self.e_cas = mc.e_cas
            self.print_level = mc.verbose
            self.davidson_only = mc.fcisolver.davidson_only
            self.pspace_size = mc.fcisolver.pspace_size
            self.enforce_degeneracy = True

            # Make sure that the orbitals are canonicalized
            mo, ci, mo_energy = mc.canonicalize(mo_coeff=mc.mo_coeff, ci=mc.ci)
            self.mo = mo.copy()
            self.wfn_casscf = ci.copy()
            self.mo_energy = mo_energy.copy()

            from pyscf import ao2mo
            self.transform_2e_chem_incore = ao2mo.general

        #    from pyscf import fci
        #    self.cre_a = fci.addons.cre_a
        #    self.cre_b = fci.addons.cre_b
        #    self.des_a = fci.addons.des_a
        #    self.des_b = fci.addons.des_b
        #    self.trans_rdm1s = fci.direct_spin1.trans_rdm1s
        #    self.trans_rdm12s = fci.direct_spin1.trans_rdm12s

            from pyscf import lib
            self.davidson = lib.linalg_helper.davidson1

            # If set to a list, can be used to select certain CASCI states during MR-ADC computations
            self.select_casci = None

        # Integrals
        self.h1e_ao = mf.get_hcore()

        # TODO: replace exact 2e integrals with the DF integrals
        self.v2e_ao = None
        if mf._eri is None:
            raise Exception("Out-of-core algorithm is not implemented for Pyscf")
        else:
            self.v2e_ao = mf._eri.copy()

        # Dipole moments
        self.dip_mom_ao    = mf.mol.intor_symmetric("int1e_r", comp = 3)

        # Whether to use opt_einsum
        if opt_einsum:
            from opt_einsum import contract
            self.einsum = contract
            self.einsum_type = "greedy"
        else:
            self.einsum = np.einsum
            self.einsum_type = "greedy"


    def compute_rdm123(self, bra, ket, nelecas):

        from pyscf import fci

        rdm1, rdm2, rdm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', bra, ket, self.ncas, nelecas)
        rdm1, rdm2, rdm3 = fci.rdm.reorder_dm123(rdm1, rdm2, rdm3)

        # rdm2[p,q,r,s] = \langle p^\dagger q^\dagger s r\rangle
        rdm2 = np.ascontiguousarray(rdm2.transpose(0, 2, 1, 3))

        # rdm3[p,q,r,s,t,u] = \langle p^\dagger q^\dagger r^\dagger u t s\rangle
        rdm3 = np.ascontiguousarray(rdm3.transpose(0, 2, 4, 1, 3, 5))

        return rdm1, rdm2, rdm3

    def compute_rdm1234(self, bra, ket, nelecas):

        from pyscf import fci

        rdm1, rdm2, rdm3, rdm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', bra, ket, self.ncas, nelecas)
        rdm1, rdm2, rdm3, rdm4 = fci.rdm.reorder_dm1234(rdm1, rdm2, rdm3, rdm4)

        # rdm2[p,q,r,s] = \langle p^\dagger q^\dagger s r\rangle
        rdm2 = np.ascontiguousarray(rdm2.transpose(0, 2, 1, 3))

        # rdm3[p,q,r,s,t,u] = \langle p^\dagger q^\dagger r^\dagger u t s\rangle
        rdm3 = np.ascontiguousarray(rdm3.transpose(0, 2, 4, 1, 3, 5))

        # rdm4[p,q,r,s,t,u,v,w] = \langle p^\dagger q^\dagger r^\dagger w v u t\rangle
        rdm4 = np.ascontiguousarray(rdm4.transpose(0, 2, 4, 6, 1, 3, 5, 7))

        return rdm1, rdm2, rdm3, rdm4

def print_header():

    print("""\n
----------------------------------------------------------------------
        PRISM: Open-Source implementation of ab initio methods
                for excited states and spectroscopy

                           Version 0.2.1

               Copyright (C) 2023 Alexander Sokolov
                                  Carlos E. V. de Moura

        Unless required by applicable law or agreed to in
        writing, software distributed under the GNU General
        Public License v3.0 and is distributed on an "AS IS"
        BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
        either express or implied.

        See the License for the specific language governing
        permissions and limitations.

        Available at https://github.com/sokolov-group/prism

----------------------------------------------------------------------""")
