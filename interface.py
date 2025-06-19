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
# Authors: Carlos E. V. de Moura <carlosevmoura@gmail.com>
#          Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#

import os
import tempfile
import numpy as np

import prism.lib.logger as logger
class PYSCF:

    def __init__(self, mf, mc = None, opt_einsum = False, select_reference = None):

        if mc is None:
            self.stdout = mf.stdout
            self.verbose = mf.verbose
        else:
            self.stdout = mf.stdout
            self.verbose = mc.verbose

        log = logger.Logger(self.stdout, self.verbose)
        log.prism_header()

        log.info("Importing Pyscf objects...\n")

        from pyscf import lib
        self.type = "pyscf"

        # General info
        self.mol = mf.mol
        self.nelec = mf.mol.nelectron
        self.enuc = mf.mol.energy_nuc()
        self.e_scf = mf.e_tot
        self.mf = mf
        self.log = log

        # Unit conversions
        self.hartree_to_ev = 27.2113862459817
#        self.hartree_to_ev = 27.2114
        self.hartree_to_inv_cm = 219474.63136314

        log.info("Collecting reference wavefunction information...")
        if mc is None:
            self.reference = "scf"
            log.info("Reference wavefunction: %s\n" % self.reference)

            self.max_memory = mf.max_memory

            self.mo = mf.mo_coeff.copy()
            self.nmo = self.mo.shape[1]
            self.mo_energy = mf.mo_energy.copy()
            self.symmetry = mf.mol.symmetry
            self.e_ref = [mf.e_tot]

            if getattr(mf, 'with_df', None):
                self.reference_df = mf.with_df
            else:
                self.reference_df = None

            if self.symmetry:
                from pyscf import symm
                if hasattr(mf._scf.mo_coeff, 'orbsym'):
                    self.group_repr_symm = [symm.irrep_id2name(mf.mol.groupname, x) for x in mf._scf.mo_coeff.orbsym]
                else:
                    self.group_repr_symm = None
            else:
                self.group_repr_symm = None
        else:
            # Determine reference type
            from pyscf.mcscf.casci import CASCI

            e_ref = mc.e_tot.copy() 
            e_cas = mc.e_cas.copy()
            ci = mc.ci.copy()

            if isinstance(mc, CASCI):
                if isinstance(ci, (list)):
                    self.reference = "ms-casci"
                else:
                    self.reference = "casci"
            else:
                if(hasattr(mc, 'weights') == True and isinstance(mc.ci, (list))):
                    self.weights = mc.weights
                    self.reference = "sa-casscf"
                else:
                    self.reference = "casscf"

            log.info("Reference wavefunction: %s" % self.reference)

            if self.reference == "sa-casscf":
                e_fzc = e_ref - e_cas
                e_ref = mc.e_states
                e_cas = []
                for p in range(len(e_ref)):
                    e_cas.append(e_ref[p]-e_fzc)
            elif self.reference in ("casscf", "casci"):
                e_ref = [e_ref]
                e_cas = [e_cas]
                ci = [ci]

            if select_reference is not None:
                e_ref_copy = []
                e_cas_copy = []
                ci_copy = []
                for state in select_reference:
                    e_ref_copy.append(e_ref[state-1])
                    e_cas_copy.append(e_cas[state-1])
                    ci_copy.append(ci[state-1])
                e_ref = e_ref_copy
                e_cas = e_cas_copy
                ci = ci_copy
                log.info("\nReference states selected: %s" % str(select_reference))

            self.max_memory = mc.max_memory
            self.mo = mc.mo_coeff.copy()
            self.mo_scf = mf.mo_coeff.copy()
            self.ovlp = mf.get_ovlp(mf.mol)
            self.nmo = self.mo.shape[1]
            self.ncore = mc.ncore
            self.ncas = mc.ncas
            self.nextern = self.nmo - self.ncore - self.ncas
            self.e_ref = e_ref
            self.e_ref_cas = e_cas
            self.davidson_only = mc.fcisolver.davidson_only
            self.pspace_size = mc.fcisolver.pspace_size
            self.enforce_degeneracy = True

            if getattr(mc, 'with_df', None):
                self.reference_df = mc.with_df
            else:
                self.reference_df = None

            # Compute state-averaged 1-RDM with respect to the reference manifold
            ref_rdm1 = np.zeros((mc.ncas, mc.ncas))
            if self.reference == "sa-casscf":
                mc_casci = CASCI(mf, mc.ncas, mc.nelecas)
                for p in range(len(ci)):
                    ref_rdm1 += mc_casci.fcisolver.make_rdm1(ci[p], mc.ncas, mc.nelecas)
                ref_rdm1 /= len(ci)
            else:
                for p in range(len(ci)):
                    ref_rdm1 += mc.fcisolver.make_rdm1(ci[p], mc.ncas, mc.nelecas)
                ref_rdm1 /= len(ci)

            # Canonicalize the orbitals
            log.info("Canonicalizing molecular orbitals using reference wavefunction manifold...\n")
            mo, ci, mo_energy = mc.canonicalize(casdm1 = ref_rdm1, ci = ci, cas_natorb = False)
            del ref_rdm1

            self.mo = mo.copy()
            self.mo_energy = mo_energy.copy()

# Removing the spin manifold code for now
####            # Check spin symmetry of the reference wavefunction and, if necessary, generate complete reference spin manifold
####            # TODO: make sure this is working with SA-CASSCF and MS-CASCI references that have states with different symmetries
####            ref_ci, ref_nelecas, ref_spin_degeneracy = self.compute_ref_spin_manifold(ci, self.ncas, self.ref_nelecas, e_ref, e_cas)
####
####            # Store wavefunctions and their degeneracy for all microstate 
####            self.ref_wfn = ref_ci
####            self.ref_wfn_spin_mult = ref_spin_degeneracy
####            self.ref_nelecas = ref_nelecas
# Removing the spin manifold code for now

            # Print reference info
            self.ref_wfn_spin_mult = self.print_reference_info(ci, self.ncas, mc.nelecas, e_ref, e_cas)

            self.ref_wfn = ci
            self.ref_nelecas = len(ci) * [mc.nelecas, ]
            self.ref_wfn_deg = len(ci) * [1, ]

            # TODO: Check if this is done correctly when canonicalization changes the order of orbitals
            self.symmetry = mc.mol.symmetry
            if self.symmetry:
                from pyscf import symm
                if hasattr(mc._scf.mo_coeff, 'orbsym'):
                    self.group_repr_symm = [symm.irrep_id2name(mc.mol.groupname, x) for x in mc._scf.mo_coeff.orbsym]
                else:
                    self.group_repr_symm = None
            else:
                self.group_repr_symm = None

            from pyscf import ao2mo
            self.transform_2e_chem_incore = ao2mo.general
            self.transform_2e_pair_chem_incore = ao2mo._ao2mo.nr_e2

            self.davidson = lib.linalg_helper.davidson1

            # If set to a list, can be used to select certain CASCI states during MR-ADC computations
            self.select_casci = None

        # Current Memory
        self.current_memory = lib.current_memory

        # HDF5 Files
        if os.environ.get('PYSCF_TMPDIR'):
            self.temp_dir = os.environ.get('PYSCF_TMPDIR', tempfile.gettempdir())
        else:
            self.temp_dir = os.environ.get('TMPDIR', tempfile.gettempdir())

        # Integrals
        self.h1e_ao = mf.get_hcore()
        if isinstance(mf._eri, np.ndarray):
            self.v2e_ao = mf._eri
        else:
            self.v2e_ao = mf.mol

        self.with_df = None
        self.naux = None

        # Dipole moments
        self.dip_mom_ao = mf.mol.intor_symmetric("int1e_r", comp = 3)

        # Whether to use opt_einsum
        if opt_einsum:
            from opt_einsum import contract
            self.einsum = contract
            self.einsum_type = "greedy"
        else:
            self.einsum = np.einsum
            self.einsum_type = "greedy"

    @property
    def with_df(self):
        return self._with_df
    @with_df.setter
    def with_df(self, obj):
        self._with_df = obj
        if obj:
            self.get_naux = obj.get_naoaux


    def print_reference_info(self, mc_ci, ncas, nelecas, e_ref, e_cas):

        ref_wfn_spin_square = []
        ref_wfn_spin = []
        ref_wfn_spin_mult = []

        # Check spin symmetry of the reference wavefunction and, if necessary, generate complete reference spin manifold
        for p in range(len(mc_ci)):
            ref_wfn_spin_square.append(abs(self.compute_spin_square(mc_ci[p], ncas, nelecas)))
            ref_wfn_spin.append(round((-1) + (np.sqrt(1 + 4 * ref_wfn_spin_square[p]))) / 2)
            ref_wfn_spin_mult.append(int(round((2 * ref_wfn_spin[p]) + 1)))

        self.log.info("Summary of reference wavefunction manifold:")
        self.log.info("Total number of reference states: %d" % len(mc_ci))

        self.log.info("  State       S^2        S      (2S+1)        E(total)              E(CAS)")
        self.log.info("-----------------------------------------------------------------------------------")

        for p in range(len(ref_wfn_spin_square)):
            self.log.info("%5d       %2.5f     %3.1f       %2d  %20.12f %20.12f" % ((p+1), ref_wfn_spin_square[p], ref_wfn_spin[p], ref_wfn_spin_mult[p], e_ref[p], e_cas[p]))

        self.log.info("-----------------------------------------------------------------------------------\n")

        return ref_wfn_spin_mult


    def compute_ref_spin_manifold(self, mc_ci, ncas, nelecas, e_ref, e_cas):

        ref_wfn_spin_square = []
        ref_wfn_spin = []
        ref_wfn_spin_mult = []

        # Check spin symmetry of the reference wavefunction and, if necessary, generate complete reference spin manifold
        for p in range(len(mc_ci)):
            ref_wfn_spin_square.append(abs(self.compute_spin_square(mc_ci[p], ncas, nelecas)))
            ref_wfn_spin.append(round((-1) + (np.sqrt(1 + 4 * ref_wfn_spin_square[p]))) / 2)
            ref_wfn_spin_mult.append(int(round((2 * ref_wfn_spin[p]) + 1)))

        # Compute all CASCI states for the reference spin manifold
        ref_ci = []
        ref_nelecas = []
        ref_spin_degeneracy = []
        for p in range(len(ref_wfn_spin_square)):
            wfn_ci, wfn_nelecas = self.compute_state_spin_manifold(mc_ci[p], ncas, nelecas, ref_wfn_spin_square[p], ref_wfn_spin[p], ref_wfn_spin_mult[p])
            ref_ci += wfn_ci
            ref_nelecas += wfn_nelecas
            ref_spin_degeneracy.append(len(wfn_ci))

        self.log.info("\nSummary of reference wavefunction manifold:")
        self.log.info("Total number of reference states: %d" % len(ref_spin_degeneracy))
        self.log.info("Total number of reference microstates: %d\n" % sum(ref_spin_degeneracy))

        self.log.info("  State       S^2        S      (2S+1)        E(total)              E(CAS)")
        self.log.info("-----------------------------------------------------------------------------------")

        for p in range(len(ref_wfn_spin_square)):
            self.log.info("%5d       %2.5f     %3.1f       %2d  %20.12f %20.12f" % ((p+1), ref_wfn_spin_square[p], ref_wfn_spin[p], ref_wfn_spin_mult[p], e_ref[p], e_cas[p]))

        self.log.info("-----------------------------------------------------------------------------------\n")

        return ref_ci, ref_nelecas, ref_spin_degeneracy


    def compute_spin_square(self, wfn, ncas, nelecas):

        from pyscf import fci
        spin_sq = fci.spin_op.spin_square(wfn, ncas, nelecas)[0]

        return spin_sq


    def compute_state_spin_manifold(self, wfn, ncas, nelecas, spin_sq, s_value, multiplicity):

        msz_value = 0
        if sum(nelecas) > 0:
            msz_wfn = self.apply_S_z(wfn, ncas, nelecas)
            msz_value = np.dot(wfn.ravel(), msz_wfn.ravel())

        ms = []
        for I in range(multiplicity):
            if not np.isclose(s_value-I, msz_value, rtol=1e-05):
                ms.append(s_value-I)

        plus_op_list = [x for x in ms if x > msz_value]
        minus_op_list = [x for x in ms if x < msz_value]

        #Initialize spin up and spin down projection generators:
        spin_multiplet = []
        spin_multiplet_ne = []
        spin_multiplet.append(wfn)
        spin_multiplet_ne.append(nelecas)

        spin_wf_plus = wfn.copy()
        spin_wf_minus = wfn.copy()
        spin_nelec_plus = nelecas
        spin_nelec_minus = nelecas

        for I in range(len(plus_op_list)):
            # Apply spin operators for finding ms values
            sz_plus = self.apply_S_z(spin_wf_plus, ncas, spin_nelec_plus)
            msz_plus = np.dot(spin_wf_plus.ravel(), sz_plus.ravel())
            # Apply Raising operator:
            spin_wf_plus, spin_nelec_plus = self.apply_S_plus(spin_wf_plus, ncas, spin_nelec_plus)
            # Normalize the wfn
            spin_wf_plus = spin_wf_plus/(np.sqrt(spin_sq - msz_plus*(msz_plus + 1)))
            # Add spin states to list
            spin_multiplet.append(spin_wf_plus)
            spin_multiplet_ne.append(spin_nelec_plus)
        
        for I in range(len(minus_op_list)):
            # Apply spin operators for finding ms values
            sz_minus = self.apply_S_z(spin_wf_minus, ncas, spin_nelec_minus)
            msz_minus = np.dot(spin_wf_minus.ravel(), sz_minus.ravel()) 
            # Apply lowering operator:
            spin_wf_minus, spin_nelec_minus = self.apply_S_minus(spin_wf_minus, ncas, spin_nelec_minus)
            # Normalize the wfn
            spin_wf_minus = spin_wf_minus/(np.sqrt(spin_sq - msz_minus*(msz_minus - 1)))
            # Add spin states to list
            spin_multiplet.append(spin_wf_minus)
            spin_multiplet_ne.append(spin_nelec_minus)

        assert(len(spin_multiplet) == multiplicity), 'The number of reference states is not equal to the spin multiplicity'

        return spin_multiplet, spin_multiplet_ne


    # Apply S+ (spin raising) operator
    def apply_S_plus(self, psi, ncas, nelecas):

        Sp_psi = []
        Sp_ne = None

        for y in range(ncas):
            a_psi, a_psi_ne = self.act_des_b(psi, ncas, nelecas, y)
            ca_psi, ca_psi_ne = self.act_cre_a(a_psi, ncas, a_psi_ne, y)

            Sp_psi.append(ca_psi)
            Sp_ne = ca_psi_ne

        Sp_psi = sum(Sp_psi)

#        # Fix the phase for the CI coefficients
#        i, j = np.unravel_index(np.absolute(Sp_psi).argmax(), Sp_psi.shape)
#        if Sp_psi[i, j] < 0.0:
#            Sp_psi *= -1.0

        return Sp_psi, Sp_ne


    # Apply S- (spin lowering) operator
    def apply_S_minus(self, psi, ncas, nelecas):

        Sp_psi = []
        Sp_ne = None

        for y in range(ncas):
            a_psi, a_psi_ne = self.act_des_a(psi, ncas, nelecas, y)
            ca_psi, ca_psi_ne = self.act_cre_b(a_psi, ncas, a_psi_ne, y)

            Sp_psi.append(ca_psi)
            Sp_ne = ca_psi_ne

        Sp_psi = sum(Sp_psi)

#        # Fix the phase for the CI coefficients
#        i, j = np.unravel_index(np.absolute(Sp_psi).argmax(), Sp_psi.shape)
#        if Sp_psi[i, j] < 0.0:
#            Sp_psi *= -1.0

        return Sp_psi, Sp_ne


    # Apply Sz (z-projection of S) operator:
    def apply_S_z(self, psi, ncas, nelecas):

        Sp_psi = []

        for y in range(ncas):
            a_psi, a_psi_ne = self.act_des_a(psi, ncas, nelecas, y)
            a_psi2, a_psi_ne2 = self.act_cre_a(a_psi, ncas, a_psi_ne, y)
            b_psi, b_psi_ne = self.act_des_b(psi, ncas, nelecas, y)
            b_psi2, b_psi_ne2 = self.act_cre_b(b_psi, ncas, b_psi_ne, y)

            if a_psi2 is None:
                Sp_psi.append(0.5*(-b_psi2))
            elif b_psi2 is None:
                Sp_psi.append(0.5*a_psi2)
            else:
                Sp_psi.append(0.5*(a_psi2 - b_psi2))
        Sp_psi = sum(Sp_psi)
        ms = np.dot(psi.ravel(), Sp_psi.ravel())

        return Sp_psi


    # Act annihilation operator (alpha spin)
    def act_cre_a(self, wfn, ncas, nelec, orb):

        from pyscf import fci
        self.cre_a = fci.addons.cre_a

        if (wfn is not None) and (ncas - nelec[0] > 0):
            c_wfn = self.cre_a(wfn, ncas, nelec, orb)
            c_wfn_ne = (nelec[0] + 1, nelec[1])
        else:
            c_wfn = None
            c_wfn_ne = None

        return c_wfn, c_wfn_ne


    # Act annihilation operator (beta spin)
    def act_cre_b(self, wfn, ncas, nelec, orb):

        from pyscf import fci
        self.cre_b = fci.addons.cre_b

        if (wfn is not None) and (ncas - nelec[1] > 0):
            c_wfn = self.cre_b(wfn, ncas, nelec, orb)
            c_wfn_ne = (nelec[0], nelec[1] + 1)
        else:
            c_wfn = None
            c_wfn_ne = None

        return c_wfn, c_wfn_ne


    # Act annihilation operator (alpha spin)
    def act_des_a(self, wfn, ncas, nelec, orb):

        from pyscf import fci
        self.des_a = fci.addons.des_a

        if (wfn is not None) and (nelec[0] > 0):
            a_wfn = self.des_a(wfn, ncas, nelec, orb)
            a_wfn_ne = (nelec[0] - 1, nelec[1])
        else:
            a_wfn = None
            a_wfn_ne = None

        return a_wfn, a_wfn_ne


    # Act annihilation operator (beta spin)
    def act_des_b(self, wfn, ncas, nelec, orb):

        from pyscf import fci
        self.des_b = fci.addons.des_b

        if (wfn is not None) and (nelec[1] > 0):
            a_wfn = self.des_b(wfn, ncas, nelec, orb)
            a_wfn_ne = (nelec[0], nelec[1] - 1)
        else:
            a_wfn = None
            a_wfn_ne = None

        return a_wfn, a_wfn_ne


    def density_fit(self, auxbasis=None, with_df = None):
        if with_df is None:
            self.log.info("Importing Pyscf density-fitting objects...")
            from pyscf import df

            self.with_df = df.DF(self.mol, auxbasis)
            self.get_naux = self.with_df.get_naoaux
        else:
            self.with_df = with_df

        return self

    def compute_rdm1(self, bra, ket, nelecas):

        from pyscf.fci.direct_spin1 import trans_rdm1

        rdm1 = (None, )

        if isinstance(nelecas, (list)):
            rdm1 = trans_rdm1(bra, ket, self.ncas, nelecas)
            for p in range(1, len(nelecas)):
                rdm1_p = trans_rdm1(bra, ket, self.ncas, nelecas[p])
                rdm1 += rdm1_p

            rdm1 /= len(nelecas)
        
        else:
           rdm1 = trans_rdm1(bra, ket, self.ncas, nelecas)

        return rdm1  

    def compute_rdm123(self, bra, ket, nelecas):

        from pyscf import fci

        rdm1, rdm2, rdm3 = 3 * (None,)
        if isinstance(nelecas, (list)):
            rdm1, rdm2, rdm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', bra[0], ket[0], self.ncas, nelecas[0])
            for p in range(1, len(nelecas)):
                rdm1_p, rdm2_p, rdm3_p = fci.rdm.make_dm123('FCI3pdm_kern_sf', bra[p], ket[p], self.ncas, nelecas[p])
                rdm1 += rdm1_p
                rdm2 += rdm2_p
                rdm3 += rdm3_p
            rdm1 /= len(nelecas)
            rdm2 /= len(nelecas)
            rdm3 /= len(nelecas)
        else:
            rdm1, rdm2, rdm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', bra, ket, self.ncas, nelecas)

        rdm1, rdm2, rdm3 = fci.rdm.reorder_dm123(rdm1, rdm2, rdm3)

        # This transpose is necessary to get the correct index order for transition 1-RDM
        rdm1 = rdm1.T

        # rdm2[p,q,r,s] = \langle p^\dagger q^\dagger s r\rangle
        rdm2 = np.ascontiguousarray(rdm2.transpose(0, 2, 1, 3))

        # rdm3[p,q,r,s,t,u] = \langle p^\dagger q^\dagger r^\dagger u t s\rangle
        rdm3 = np.ascontiguousarray(rdm3.transpose(0, 2, 4, 1, 3, 5))

        return rdm1, rdm2, rdm3


    def compute_rdm1234(self, bra, ket, nelecas):

        from pyscf import fci

        rdm1, rdm2, rdm3, rdm4 = 4 * (None,)
        if isinstance(nelecas, (list)):
            rdm1, rdm2, rdm3, rdm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', bra[0], ket[0], self.ncas, nelecas[0])
            for p in range(1, len(nelecas)):
                rdm1_p, rdm2_p, rdm3_p, rdm4_p = fci.rdm.make_dm1234('FCI4pdm_kern_sf', bra[p], ket[p], self.ncas, nelecas[p])
                rdm1 += rdm1_p
                rdm2 += rdm2_p
                rdm3 += rdm3_p
                rdm4 += rdm4_p
            rdm1 /= len(nelecas)
            rdm2 /= len(nelecas)
            rdm3 /= len(nelecas)
            rdm4 /= len(nelecas)
        else:
            rdm1, rdm2, rdm3, rdm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', bra, ket, self.ncas, nelecas)

        rdm1, rdm2, rdm3, rdm4 = fci.rdm.reorder_dm1234(rdm1, rdm2, rdm3, rdm4)

        # This transpose is necessary to get the correct index order for transition 1-RDM
        rdm1 = rdm1.T

        # rdm2[p,q,r,s] = \langle p^\dagger q^\dagger s r\rangle
        rdm2 = np.ascontiguousarray(rdm2.transpose(0, 2, 1, 3))

        # rdm3[p,q,r,s,t,u] = \langle p^\dagger q^\dagger r^\dagger u t s\rangle
        rdm3 = np.ascontiguousarray(rdm3.transpose(0, 2, 4, 1, 3, 5))

        # rdm4[p,q,r,s,t,u,v,w] = \langle p^\dagger q^\dagger r^\dagger w v u t\rangle
        rdm4 = np.ascontiguousarray(rdm4.transpose(0, 2, 4, 6, 1, 3, 5, 7))

        return rdm1, rdm2, rdm3, rdm4
