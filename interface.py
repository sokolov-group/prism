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
        self.spin_sq_thresh = 0 # Maximum S^2 value of CASCI roots to keep; default to only singlet calculations

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

#            from pyscf import fci
#            self.cre_a = fci.addons.cre_a
#            self.cre_b = fci.addons.cre_b
#            self.des_a = fci.addons.des_a
#            self.des_b = fci.addons.des_b
#            self.trans_rdm1s = fci.direct_spin1.trans_rdm1s
#            self.trans_rdm12s = fci.direct_spin1.trans_rdm12s

            from pyscf import lib
            self.davidson = lib.linalg_helper.davidson1

            self.select_casci = None # If set to a list, can be used to select certain CASCI states during MR-ADC computations

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

        rdm2 = np.ascontiguousarray(rdm2.transpose(0, 2, 1, 3))               # rdm2[p,q,r,s] = \langle p^\dagger q^\dagger s r\rangle
        rdm3 = np.ascontiguousarray(rdm3.transpose(0, 2, 4, 1, 3, 5))         # rdm3[p,q,r,s,t,u] = \langle p^\dagger q^\dagger r^\dagger u t s\rangle

        return rdm1, rdm2, rdm3

    def compute_rdm1234(self, bra, ket, nelecas):

        from pyscf import fci

        rdm1, rdm2, rdm3, rdm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', bra, ket, self.ncas, nelecas)
        rdm1, rdm2, rdm3, rdm4 = fci.rdm.reorder_dm1234(rdm1, rdm2, rdm3, rdm4)

        rdm2 = np.ascontiguousarray(rdm2.transpose(0, 2, 1, 3))               # rdm2[p,q,r,s] = \langle p^\dagger q^\dagger s r\rangle
        rdm3 = np.ascontiguousarray(rdm3.transpose(0, 2, 4, 1, 3, 5))         # rdm3[p,q,r,s,t,u] = \langle p^\dagger q^\dagger r^\dagger u t s\rangle
        rdm4 = np.ascontiguousarray(rdm4.transpose(0, 2, 4, 6, 1, 3, 5, 7))   # rdm3[p,q,r,s,t,u,v,w] = \langle p^\dagger q^\dagger r^\dagger w v u t\rangle

        return rdm1, rdm2, rdm3, rdm4

#
#    def compute_casci_ip_ea(self, ncasci, method_type):
#
#        from pyscf import mcscf
#        from pyscf import fci
#
#        if method_type == "ip": 
#            print("Running CASCI computation for %d IP roots...\n" % ncasci)
#        elif method_type == "ea": 
#            print("Running CASCI computation for %d EA roots...\n" % ncasci)
#        else:
#            raise Exception ("This function should be used only for IP or EA")
#
#        # Set up CASCI computation for IP in alpha
#        nalpha, nbeta = self.nelecas
#        if method_type == "ip": 
#            nalpha -= 1
#        else: 
#            nalpha += 1
#
#        mc_casci = mcscf.CASCI(self.mf, self.ncas, (nalpha, nbeta), ncore = self.ncore)
#        mc_casci.verbose = self.print_level
#        mc_casci.canonicalization = False
#        mc_casci.fcisolver = fci.direct_spin1.FCISolver(self.mf.mol)
#        mc_casci.fcisolver.davidson_only = self.davidson_only
#        mc_casci.fcisolver.pspace_size = self.pspace_size
#        mc_casci.fcisolver.conv_tol = 1e-12
#        mc_casci.fcisolver.max_cycle = 2000
#        mc_casci.fcisolver.max_space = ncasci * 20
#
#        # Increase the number of requested CASCI states to make sure we don't break spatial symmetry
#        ncasci_extra = max(int(1.8 * ncasci), 10)
#        if self.enforce_degeneracy:
#            print("Adding %d CASCI states to enforce degeneracy...\n" % ncasci_extra)
#            mc_casci.fcisolver.nroots = ncasci + ncasci_extra
#        else:
#            mc_casci.fcisolver.nroots = ncasci
#
#        # Run CASCI computation
#        mc_casci_results = mc_casci.casci(mo_coeff=self.mo)
#
#        e_cas_ci = mc_casci_results[1]
#        wfn_casci = mc_casci_results[2]
#
#        # Remove ionized states other than those with S^2 = 0.75
#        spin_sq_thresh = 0.75
#        e_cas_roots_spinstate = []
#        psi_roots_spinstate = []
#        for root in range(len(e_cas_ci)):
#            spin_sq = fci.spin_op.spin_square0(wfn_casci[root], mc_casci.ncas, (nalpha, nbeta))[0]
#            if np.around([spin_sq], decimals=2) == spin_sq_thresh:
#                print("Keeping CASCI state %d with S^2 = %f" % (root, spin_sq))
#                e_cas_roots_spinstate.append(e_cas_ci[root])
#                psi_roots_spinstate.append(wfn_casci[root])
#            else:
#                print("Discarding CASCI state %d with S^2 = %f" % (root, spin_sq))
#        e_cas_ci = np.array(e_cas_roots_spinstate)
#        wfn_casci = psi_roots_spinstate
#
#        # Check if the number of states in CASCI is smaller than requested
#        if len(e_cas_ci) < ncasci:
#            ncasci = len(e_cas_ci)
#            print("\nWARNING: The number of CASCI states is smaller than requested... Reducing the number of states to ", ncasci)
#
#        # Make sure that we do not break spatial symmetry
#        if self.enforce_degeneracy:
#            ncasci_old = ncasci
#            max_e = e_cas_ci[ncasci - 1]
#            e_diff = e_cas_ci[ncasci:] - max_e
#
#            for de in e_diff:
#                if abs(de) < 1e-6:
#                    ncasci += 1
#
#            if (ncasci > ncasci_old):
#                print("\nIncreased the number of CASCI states by %d to enforce degeneracy" % (ncasci - ncasci_old))
#
#            e_cas_ci = e_cas_ci[:ncasci]
#            wfn_casci = wfn_casci[:ncasci]
#
#        if self.select_casci is not None:
#            print("\nSelecting CASCI states using the user-provided list...")
#            print("List:")
#            print(str(self.select_casci))
#            selected_e_cas_ci = [e_cas_ci[i] for i in self.select_casci]
#            selected_wfn_casci = [wfn_casci[i] for i in self.select_casci]
#            e_cas_ci = selected_e_cas_ci
#            wfn_casci = selected_wfn_casci
#            ncasci = len(e_cas_ci)
#
#        # Fix the phase for the CI coefficients
#        for I in range(len(e_cas_ci)):
#            psi_I = wfn_casci[I]
#            i, j = np.unravel_index(np.absolute(psi_I).argmax(), psi_I.shape)
#            if psi_I[i, j] < 0.0:
#                wfn_casci[I] *= -1.0
#
#        sys.stdout.flush()
#
#        return e_cas_ci, wfn_casci
#
#
#    def compute_casci_ee(self, ncasci):
#
#        from pyscf import mcscf
#        from pyscf import fci
#
#        # Increase the number of CASCI states by 1 to account for the ground state
#        ncasci += 1
#
#        print("Running CASCI computation for %d EE roots...\n" % ncasci)
#
#        # Set up CASCI computation for EE in alpha
#        nalpha, nbeta = self.nelecas
#
#        mc_casci = mcscf.CASCI(self.mf, self.ncas, (nalpha, nbeta), ncore = self.ncore)
#        mc_casci.verbose = self.print_level
#        mc_casci.canonicalization = False
#        mc_casci.fcisolver = fci.direct_spin1.FCISolver(self.mf.mol)
#        mc_casci.fcisolver.davidson_only = self.davidson_only
#        mc_casci.fcisolver.pspace_size = self.pspace_size
#        mc_casci.fcisolver.conv_tol = 1e-12
#        mc_casci.fcisolver.max_cycle = 2000
#        mc_casci.fcisolver.max_space = ncasci * 20
#
#        # Increase the number of requested CASCI states to make sure we don't break spatial symmetry
#        ncasci_extra = max(int(1.5 * ncasci), 10)
#
#        if self.enforce_degeneracy:
#            print("Adding %d CASCI states to enforce degeneracy...\n" % ncasci_extra)
#            mc_casci.fcisolver.nroots = ncasci + ncasci_extra
#        else:
#            mc_casci.fcisolver.nroots = ncasci
#
#        # Run CASCI computation
#        mc_casci_results = mc_casci.casci(mo_coeff=self.mo)
#
#        e_cas_ci = mc_casci_results[1][1:]
#        wfn_casci = mc_casci_results[2][1:]
#
#        ncasci -= 1
#
#        # Keep all CASCI states w/ desired maximum S^2 values
#        e_cas_roots_spinstate = []
#        psi_roots_spinstate = []
#        for root in range(len(e_cas_ci)):
#            spin_sq = fci.spin_op.spin_square0(wfn_casci[root], mc_casci.ncas, (nalpha, nbeta))[0]
#
#            if np.around([spin_sq], decimals=2) <= self.spin_sq_thresh:
#                print("Keeping CASCI state %d with S^2 = %f" % (root, spin_sq))
#                e_cas_roots_spinstate.append(e_cas_ci[root])
#                psi_roots_spinstate.append(wfn_casci[root])
#            else:
#                print("Discarding CASCI state %d with S^2 = %f" % (root, spin_sq))
#
#        e_cas_ci = np.array(e_cas_roots_spinstate)
#        wfn_casci = psi_roots_spinstate
#
#        # Check if the number of states in CASCI is smaller than requested
#        if len(e_cas_ci) < ncasci:
#            ncasci = len(e_cas_ci)
#            print("\nWARNING: The number of CASCI states is smaller than requested... Reducing the number of states to ", ncasci)
#
#        # Make sure that we do not break spatial symmetry
#        if self.enforce_degeneracy:
#            ncasci_old = ncasci
#            max_e = e_cas_ci[ncasci - 1]
#            e_diff = e_cas_ci[ncasci:] - max_e
#
#            for de in e_diff:
#                if abs(de) < 1e-6:
#                    ncasci += 1
#
#            if (ncasci > ncasci_old):
#                print("\nIncreased the number of CASCI states by %d to enforce degeneracy" % (ncasci - ncasci_old))
#
#            e_cas_ci = e_cas_ci[:ncasci]
#            wfn_casci = wfn_casci[:ncasci]
#        
#        # Allow user to manually keep requested CASCI states
#        if self.select_casci is not None:
#            print("\nSelecting CASCI states using the user-provided list...")
#            print("List:")
#            print(str(self.select_casci))
#            selected_e_cas_ci = [e_cas_ci[i] for i in self.select_casci]
#            selected_wfn_casci = [wfn_casci[i] for i in self.select_casci]
#            e_cas_ci = selected_e_cas_ci
#            wfn_casci = selected_wfn_casci
#            ncasci = len(e_cas_ci)
#
#
#        # Fix the phase for the CI coefficients
#        for I in range(len(e_cas_ci)):
#            psi_I = wfn_casci[I]
#            i, j = np.unravel_index(np.absolute(psi_I).argmax(), psi_I.shape)
#            if psi_I[i, j] < 0.0:
#                wfn_casci[I] *= -1.0
#
#        sys.stdout.flush()
#
#        return e_cas_ci, wfn_casci
#
#
#    def compute_rdm_ca_si(self, bra, ket, nelecas):
#
#        ncas = self.ncas
#
#        rdm_a, rdm_b = self.trans_rdm1s(bra, ket, ncas, nelecas)
#        rdm_a = rdm_a.T.copy()
#        rdm_b = rdm_b.T.copy()
#
#        return rdm_a, rdm_b
#
#
#    def compute_rdm_ca_general_si(self, bra, ket, bra_ne, ket_ne):
#
#        ncas = self.ncas
#
#        rdm_aa = np.zeros((ncas, ncas))
#        rdm_ab = np.zeros((ncas, ncas))
#        rdm_ba = np.zeros((ncas, ncas))
#        rdm_bb = np.zeros((ncas, ncas))
#
#        # AA and BB
#        if (bra_ne == ket_ne):
#            rdm_aa, rdm_bb = self.trans_rdm1s(bra, ket, ncas, bra_ne)
#            rdm_aa = rdm_aa.T.copy()
#            rdm_bb = rdm_bb.T.copy()
#
#        # AB
#        if ((bra_ne[0] - 1, bra_ne[1]) == (ket_ne[0], ket_ne[1] - 1)):
#            a_bra, a_bra_ne = self.apply_a(bra, ncas, bra_ne, "a")
#            b_ket, b_ket_ne = self.apply_a(ket, ncas, ket_ne, "b")
#            if a_bra is not None:
#                a_bra = a_bra.reshape(ncas, -1)
#                b_ket = b_ket.reshape(ncas, -1)
#                rdm_ab = np.dot(a_bra, b_ket.T)
#
#        # BA
#        if ((bra_ne[0], bra_ne[1] - 1) == (ket_ne[0] - 1, ket_ne[1])):
#            b_bra, b_bra_ne = self.apply_a(bra, ncas, bra_ne, "b")
#            a_ket, a_ket_ne = self.apply_a(ket, ncas, ket_ne, "a")
#            if b_bra is not None:
#                b_bra = b_bra.reshape(ncas, -1)
#                a_ket = a_ket.reshape(ncas, -1)
#                rdm_ba = np.dot(b_bra, a_ket.T)
#
#        return rdm_aa, rdm_ab, rdm_ba, rdm_bb
#
#
#    def compute_rdm_aa_general_si(self, bra, ket, bra_ne, ket_ne):
#
#        ncas = self.ncas
#
#        rdm_aa = np.zeros((ncas, ncas))
#        rdm_ab = np.zeros((ncas, ncas))
#        rdm_ba = np.zeros((ncas, ncas))
#        rdm_bb = np.zeros((ncas, ncas))
#
#        # AA
#        if ((bra_ne[0] + 1, bra_ne[1]) == (ket_ne[0] - 1, ket_ne[1])):
#            a_bra, a_bra_ne = self.apply_c(bra, ncas, bra_ne, "a")
#            a_ket, a_ket_ne = self.apply_a(ket, ncas, ket_ne, "a")
#            if a_bra is not None:
#                a_bra = a_bra.reshape(ncas, -1)
#                a_ket = a_ket.reshape(ncas, -1)
#                rdm_aa = np.dot(a_bra, a_ket.T)
#
#        # BB
#        if ((bra_ne[0], bra_ne[1] + 1) == (ket_ne[0], ket_ne[1] - 1)):
#            b_bra, b_bra_ne = self.apply_c(bra, ncas, bra_ne, "b")
#            b_ket, b_ket_ne = self.apply_a(ket, ncas, ket_ne, "b")
#            if b_bra is not None:
#                b_bra = b_bra.reshape(ncas, -1)
#                b_ket = b_ket.reshape(ncas, -1)
#                rdm_bb = np.dot(b_bra, b_ket.T)
#
#        # AB
#        if ((bra_ne[0] + 1, bra_ne[1]) == (ket_ne[0], ket_ne[1] - 1)):
#            a_bra, a_bra_ne = self.apply_c(bra, ncas, bra_ne, "a")
#            b_ket, b_ket_ne = self.apply_a(ket, ncas, ket_ne, "b")
#            if a_bra is not None:
#                a_bra = a_bra.reshape(ncas, -1)
#                b_ket = b_ket.reshape(ncas, -1)
#                rdm_ab = np.dot(a_bra, b_ket.T)
#
#        # BA
#        if ((bra_ne[0], bra_ne[1] + 1) == (ket_ne[0] - 1, ket_ne[1])):
#            b_bra, b_bra_ne = self.apply_c(bra, ncas, bra_ne, "b")
#            a_ket, a_ket_ne = self.apply_a(ket, ncas, ket_ne, "a")
#            if b_bra is not None:
#                b_bra = b_bra.reshape(ncas, -1)
#                a_ket = a_ket.reshape(ncas, -1)
#                rdm_ba = np.dot(b_bra, a_ket.T)
#
#        return rdm_aa, rdm_ab, rdm_ba, rdm_bb
#
#
#    def compute_rdm_ccaa_si(self, bra, ket, nelecas):
#
#        ncas = self.ncas
#
#        rdm1, rdm2 = self.trans_rdm12s(bra, ket, ncas, nelecas)
#        rdm_aa, rdm_ab, rdm_ba, rdm_bb = rdm2
#        rdm_aa = rdm_aa.transpose(0, 2, 3, 1).copy() # <p+q+rs> (AAAA)
#        rdm_ab = rdm_ab.transpose(0, 2, 3, 1).copy() # <p+q+rs> (ABBA)
#        rdm_bb = rdm_bb.transpose(0, 2, 3, 1).copy() # <p+q+rs> (BBBB)
#
#        return rdm_aa, rdm_ab, rdm_bb
#
#
#    def compute_rdm_cccaaa_si(self, bra, ket, nelecas):
#
#        ncas = self.ncas
#
#        rdm_aaa = np.zeros((ncas, ncas, ncas, ncas, ncas, ncas))
#        rdm_aab = np.zeros((ncas, ncas, ncas, ncas, ncas, ncas))
#        rdm_abb = np.zeros((ncas, ncas, ncas, ncas, ncas, ncas))
#        rdm_bbb = np.zeros((ncas, ncas, ncas, ncas, ncas, ncas))
#
#        # AAAAAA
#        wfns, wfn_ne = self.apply_aaa(ket, ncas, nelecas, "aaa")
#        if wfns is not None:
#            wfns = wfns.reshape(ncas**3, -1)
#            wfns_I, wfn_I = self.apply_aaa(bra, ncas, nelecas, "aaa")
#            wfns_I = wfns_I.reshape(ncas**3, -1)
#            rdm_aaa = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas)).transpose(2,1,0,3,4,5).copy()
#
#        # BAAAAB
#        wfns, wfn_ne = self.apply_aaa(ket, ncas, nelecas, "aab")
#        if wfns is not None:
#            wfns = wfns.reshape(ncas**3, -1)
#            wfns_I, wfn_I = self.apply_aaa(bra, ncas, nelecas, "aab")
#            wfns_I = wfns_I.reshape(ncas**3, -1)
#            rdm_aab = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas)).transpose(2,1,0,3,4,5).copy()
#
#        # BBAABB
#        wfns, wfn_ne = self.apply_aaa(ket, ncas, nelecas, "abb")
#        if wfns is not None:
#            wfns = wfns.reshape(ncas**3, -1)
#            wfns_I, wfn_I = self.apply_aaa(bra, ncas, nelecas, "abb")
#            wfns_I = wfns_I.reshape(ncas**3, -1)
#            rdm_abb = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas)).transpose(2,1,0,3,4,5).copy()
#
#        # BBBBBB
#        wfns, wfn_ne = self.apply_aaa(ket, ncas, nelecas, "bbb")
#        if wfns is not None:
#            wfns = wfns.reshape(ncas**3, -1)
#            wfns_I, wfn_I = self.apply_aaa(bra, ncas, nelecas, "bbb")
#            wfns_I = wfns_I.reshape(ncas**3, -1)
#            rdm_bbb = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas)).transpose(2,1,0,3,4,5).copy()
#
#        return rdm_aaa, rdm_aab, rdm_abb, rdm_bbb
#
#
#    def compute_rdm_ccccaaaa_si(self, bra, ket, nelecas):
#
#        ncas = self.ncas
#
#        rdm_aaaa = np.zeros((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas))
#        rdm_aaab = np.zeros((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas))
#        rdm_aabb = np.zeros((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas))
#        rdm_abbb = np.zeros((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas))
#        rdm_bbbb = np.zeros((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas))
#
#        # AAAAAAAA
#        wfns, wfn_ne = self.apply_aaaa(ket, ncas, nelecas, "aaaa")
#        if wfns is not None:
#            wfns = wfns.reshape(ncas**4, -1)
#            wfns_I, wfn_I = self.apply_aaaa(bra, ncas, nelecas, "aaaa")
#            wfns_I = wfns_I.reshape(ncas**4, -1)
#            rdm_aaaa = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas)).transpose(3,2,1,0,4,5,6,7).copy()
#
#        # BAAAAAAB
#        wfns, wfn_ne = self.apply_aaaa(ket, ncas, nelecas, "aaab")
#        if wfns is not None:
#            wfns = wfns.reshape(ncas**4, -1)
#            wfns_I, wfn_I = self.apply_aaaa(bra, ncas, nelecas, "aaab")
#            wfns_I = wfns_I.reshape(ncas**4, -1)
#            rdm_aaab = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas)).transpose(3,2,1,0,4,5,6,7).copy()
#
#        # BBAAAABB
#        wfns, wfn_ne = self.apply_aaaa(ket, ncas, nelecas, "aabb")
#        if wfns is not None:
#            wfns = wfns.reshape(ncas**4, -1)
#            wfns_I, wfn_I = self.apply_aaaa(bra, ncas, nelecas, "aabb")
#            wfns_I = wfns_I.reshape(ncas**4, -1)
#            rdm_aabb = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas)).transpose(3,2,1,0,4,5,6,7).copy()
#
#        # BBBAABBB
#        wfns, wfn_ne = self.apply_aaaa(ket, ncas, nelecas, "abbb")
#        if wfns is not None:
#            wfns = wfns.reshape(ncas**4, -1)
#            wfns_I, wfn_I = self.apply_aaaa(bra, ncas, nelecas, "abbb")
#            wfns_I = wfns_I.reshape(ncas**4, -1)
#            rdm_abbb = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas)).transpose(3,2,1,0,4,5,6,7).copy()
#
#        # BBBBBBBB
#        wfns, wfn_ne = self.apply_aaaa(ket, ncas, nelecas, "bbbb")
#        if wfns is not None:
#            wfns = wfns.reshape(ncas**4, -1)
#            wfns_I, wfn_I = self.apply_aaaa(bra, ncas, nelecas, "bbbb")
#            wfns_I = wfns_I.reshape(ncas**4, -1)
#            rdm_bbbb = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas)).transpose(3,2,1,0,4,5,6,7).copy()
#
#        return rdm_aaaa, rdm_aaab, rdm_aabb, rdm_abbb, rdm_bbbb
#
#
#    def compute_rdm_c_si(self, bra, ket, nelecas_bra, nelecas_ket):
#
#        ncas = self.ncas
#
#        rdm_a = np.zeros((ncas))
#        rdm_b = np.zeros((ncas))
#
#        # A
#        if (nelecas_bra[0] - 1) == nelecas_ket[0] and nelecas_bra[1] == nelecas_ket[1]:
#            wfns, wfn_ne = self.apply_a(bra, ncas, nelecas_bra, "a")
#            if wfns is not None:
#                wfns = wfns.reshape(ncas, -1)
#                ket = ket.reshape(-1)
#                rdm_a = np.dot(wfns, ket)
#        # B
#        elif nelecas_bra[0] == nelecas_ket[0] and (nelecas_bra[1] - 1) == nelecas_ket[1]:
#            wfns, wfn_ne = self.apply_a(bra, ncas, nelecas_bra, "b")
#            if wfns is not None:
#                wfns = wfns.reshape(ncas, -1)
#                ket = ket.reshape(-1)
#                rdm_b = np.dot(wfns, ket)
#        else:
#            raise Exception("Number of electrons doesn't match in bra and ket")
#
#        return rdm_a, rdm_b
#
#
#    # Apply S+ (spin raising) operator
#    def apply_S_plus(self, psi, ncas, nelecas):
#
#        Sp_psi = []
#        Sp_ne = None
#
#        for y in range(ncas):
#            a_psi, a_psi_ne = self.act_des_b(psi, ncas, nelecas, y)
#            ca_psi, ca_psi_ne = self.act_cre_a(a_psi, ncas, a_psi_ne, y)
#
#            Sp_psi.append(ca_psi)
#            Sp_ne = ca_psi_ne
#
#        Sp_psi = sum(Sp_psi)
#
##        # Fix the phase for the CI coefficients
##        i, j = np.unravel_index(np.absolute(Sp_psi).argmax(), Sp_psi.shape)
##        if Sp_psi[i, j] < 0.0:
##            Sp_psi *= -1.0
#
#        return Sp_psi, Sp_ne
#
#
#    # Apply S- (spin lowering) operator
#    def apply_S_minus(self, psi, ncas, nelecas):
#
#        Sp_psi = []
#        Sp_ne = None
#
#        for y in range(ncas):
#            a_psi, a_psi_ne = self.act_des_a(psi, ncas, nelecas, y)
#            ca_psi, ca_psi_ne = self.act_cre_b(a_psi, ncas, a_psi_ne, y)
#
#            Sp_psi.append(ca_psi)
#            Sp_ne = ca_psi_ne
#
#        Sp_psi = sum(Sp_psi)
#
##        # Fix the phase for the CI coefficients
##        i, j = np.unravel_index(np.absolute(Sp_psi).argmax(), Sp_psi.shape)
##        if Sp_psi[i, j] < 0.0:
##            Sp_psi *= -1.0
#
#        return Sp_psi, Sp_ne
#
#
#
#    def apply_c(self, psi, ncas, nelecas, spin):
#
#        psi_list = []
#        psi_ne = None
#
#        for y in range(ncas):
#            y_psi, y_psi_ne = None, None
#            if spin[0] == "a":
#                y_psi, y_psi_ne = self.act_cre_a(psi, ncas, nelecas, y)
#            else:
#                y_psi, y_psi_ne = self.act_cre_b(psi, ncas, nelecas, y)
#            psi_list.append(y_psi)
#            psi_ne = y_psi_ne
#
#        psi_list = self.extract_vectors(psi_list, ncas, 1)
#
#        return psi_list, psi_ne
#
#
#    def apply_a(self, psi, ncas, nelecas, spin):
#
#        psi_list = []
#        psi_ne = None
#
#        for y in range(ncas):
#            y_psi, y_psi_ne = None, None
#            if spin[0] == "a":
#                y_psi, y_psi_ne = self.act_des_a(psi, ncas, nelecas, y)
#            else:
#                y_psi, y_psi_ne = self.act_des_b(psi, ncas, nelecas, y)
#            psi_list.append(y_psi)
#            psi_ne = y_psi_ne
#
#        psi_list = self.extract_vectors(psi_list, ncas, 1)
#
#        return psi_list, psi_ne
#
#
#    def apply_cc(self, psi, ncas, nelecas, spin):
#
#        psi_list = []
#        psi_ne = None
#
#        for y in range(ncas):
#            y_psi, y_psi_ne = None, None
#            if spin[1] == "a":
#                y_psi, y_psi_ne = self.act_cre_a(psi, ncas, nelecas, y)
#            else:
#                y_psi, y_psi_ne = self.act_cre_b(psi, ncas, nelecas, y)
#            for z in range(ncas): 
#                zy_psi, zy_psi_ne = None, None
#                if spin[0] == "a":
#                    zy_psi, zy_psi_ne = self.act_cre_a(y_psi, ncas, y_psi_ne, z)
#                else:
#                    zy_psi, zy_psi_ne = self.act_cre_b(y_psi, ncas, y_psi_ne, z)
#                psi_list.append(zy_psi) 
#                psi_ne = zy_psi_ne
#                    
#        psi_list = self.extract_vectors(psi_list, ncas, 2)
#
#        return psi_list, psi_ne
#
#
#    def apply_ca(self, psi, ncas, nelecas, spin):
#
#        psi_list = []
#        psi_ne = None
#
#        for y in range(ncas):
#            y_psi, y_psi_ne = None, None
#            if spin[1] == "a":
#                y_psi, y_psi_ne = self.act_des_a(psi, ncas, nelecas, y)
#            else:
#                y_psi, y_psi_ne = self.act_des_b(psi, ncas, nelecas, y)
#            for z in range(ncas):
#                zy_psi, zy_psi_ne = None, None
#                if spin[0] == "a":
#                    zy_psi, zy_psi_ne = self.act_cre_a(y_psi, ncas, y_psi_ne, z)
#                else:
#                    zy_psi, zy_psi_ne = self.act_cre_b(y_psi, ncas, y_psi_ne, z)
#                psi_list.append(zy_psi)
#                psi_ne = zy_psi_ne
#
#        psi_list = self.extract_vectors(psi_list, ncas, 2)
#
#        return psi_list, psi_ne
#
#
#    def apply_aa(self, psi, ncas, nelecas, spin):
#
#        psi_list = []
#        psi_ne = None
#
#        for y in range(ncas):
#            y_psi, y_psi_ne = None, None
#            if spin[1] == "a":
#                y_psi, y_psi_ne = self.act_des_a(psi, ncas, nelecas, y)
#            else:
#                y_psi, y_psi_ne = self.act_des_b(psi, ncas, nelecas, y)
#            for z in range(ncas):
#                zy_psi, zy_psi_ne = None, None
#                if spin[0] == "a":
#                    zy_psi, zy_psi_ne = self.act_des_a(y_psi, ncas, y_psi_ne, z)
#                else:
#                    zy_psi, zy_psi_ne = self.act_des_b(y_psi, ncas, y_psi_ne, z)
#                psi_list.append(zy_psi)
#                psi_ne = zy_psi_ne
#
#        psi_list = self.extract_vectors(psi_list, ncas, 2)
#
#        return psi_list, psi_ne
#
#
#    def apply_cca(self, psi, ncas, nelecas, spin):
#
#        psi_list = []
#        psi_ne = None
#
#        for y in range(ncas):
#            y_psi, y_psi_ne = None, None
#            if spin[2] == "a":
#                y_psi, y_psi_ne = self.act_des_a(psi, ncas, nelecas, y)
#            else:
#                y_psi, y_psi_ne = self.act_des_b(psi, ncas, nelecas, y)
#            for z in range(ncas):
#                zy_psi, zy_psi_ne = None, None
#                if spin[1] == "a":
#                    zy_psi, zy_psi_ne = self.act_cre_a(y_psi, ncas, y_psi_ne, z)
#                else:
#                    zy_psi, zy_psi_ne = self.act_cre_b(y_psi, ncas, y_psi_ne, z)
#                for w in range(ncas):
#                    wzy_psi, wzy_psi_ne = None, None
#                    if spin[0] == "a":
#                        wzy_psi, wzy_psi_ne = self.act_cre_a(zy_psi, ncas, zy_psi_ne, w)
#                    else:
#                        wzy_psi, wzy_psi_ne = self.act_cre_b(zy_psi, ncas, zy_psi_ne, w)
#                    psi_list.append(wzy_psi)
#                    psi_ne = wzy_psi_ne
#
#        psi_list = self.extract_vectors(psi_list, ncas, 3)
#
#        return psi_list, psi_ne
#
#
#    def apply_caa(self, psi, ncas, nelecas, spin):
#
#        psi_list = []
#        psi_ne = None
#
#        for y in range(ncas):
#            y_psi, y_psi_ne = None, None
#            if spin[2] == "a":
#                y_psi, y_psi_ne = self.act_des_a(psi, ncas, nelecas, y)
#            else:
#                y_psi, y_psi_ne = self.act_des_b(psi, ncas, nelecas, y)
#            for z in range(ncas):
#                zy_psi, zy_psi_ne = None, None
#                if spin[1] == "a":
#                    zy_psi, zy_psi_ne = self.act_des_a(y_psi, ncas, y_psi_ne, z)
#                else:
#                    zy_psi, zy_psi_ne = self.act_des_b(y_psi, ncas, y_psi_ne, z)
#                for w in range(ncas):
#                    wzy_psi, wzy_psi_ne = None, None
#                    if spin[0] == "a":
#                        wzy_psi, wzy_psi_ne = self.act_cre_a(zy_psi, ncas, zy_psi_ne, w)
#                    else:
#                        wzy_psi, wzy_psi_ne = self.act_cre_b(zy_psi, ncas, zy_psi_ne, w)
#                    psi_list.append(wzy_psi)
#                    psi_ne = wzy_psi_ne
#
#        psi_list = self.extract_vectors(psi_list, ncas, 3)
#
#        return psi_list, psi_ne
#
#
#    def apply_aaa(self, psi, ncas, nelecas, spin):
#
#        psi_list = []
#        psi_ne = None
#
#        for y in range(ncas):
#            y_psi, y_psi_ne = None, None
#            if spin[2] == "a":
#                y_psi, y_psi_ne = self.act_des_a(psi, ncas, nelecas, y)
#            else:
#                y_psi, y_psi_ne = self.act_des_b(psi, ncas, nelecas, y)
#            for z in range(ncas):
#                zy_psi, zy_psi_ne = None, None
#                if spin[1] == "a":
#                    zy_psi, zy_psi_ne = self.act_des_a(y_psi, ncas, y_psi_ne, z)
#                else:
#                    zy_psi, zy_psi_ne = self.act_des_b(y_psi, ncas, y_psi_ne, z)
#                for w in range(ncas):
#                    wzy_psi, wzy_psi_ne = None, None
#                    if spin[0] == "a":
#                        wzy_psi, wzy_psi_ne = self.act_des_a(zy_psi, ncas, zy_psi_ne, w)
#                    else:
#                        wzy_psi, wzy_psi_ne = self.act_des_b(zy_psi, ncas, zy_psi_ne, w)
#                    psi_list.append(wzy_psi)
#                    psi_ne = wzy_psi_ne
#
#        psi_list = self.extract_vectors(psi_list, ncas, 3)
#
#        return psi_list, psi_ne
#
#
#    def apply_aaaa(self, psi, ncas, nelecas, spin):
#
#        psi_list = []
#        psi_ne = None
#
#        for z in range(ncas):
#            z_psi, z_psi_ne = None, None
#            if spin[3] == "a":
#                z_psi, z_psi_ne = self.act_des_a(psi, ncas, nelecas, z)
#            else:
#                z_psi, z_psi_ne = self.act_des_b(psi, ncas, nelecas, z)
#            for y in range(ncas):
#                yz_psi, yz_psi_ne = None, None
#                if spin[2] == "a":
#                    yz_psi, yz_psi_ne = self.act_des_a(z_psi, ncas, z_psi_ne, y)
#                else:
#                    yz_psi, yz_psi_ne = self.act_des_b(z_psi, ncas, z_psi_ne, y)
#                for w in range(ncas):
#                    wyz_psi, wyz_psi_ne = None, None
#                    if spin[1] == "a":
#                        wyz_psi, wyz_psi_ne = self.act_des_a(yz_psi, ncas, yz_psi_ne, w)
#                    else:
#                        wyz_psi, wyz_psi_ne = self.act_des_b(yz_psi, ncas, yz_psi_ne, w)
#                    for x in range(ncas):
#                        xwyz_psi, xwyz_psi_ne = None, None
#                        if spin[0] == "a":
#                            xwyz_psi, xwyz_psi_ne = self.act_des_a(wyz_psi, ncas, wyz_psi_ne, x)
#                        else:
#                            xwyz_psi, xwyz_psi_ne = self.act_des_b(wyz_psi, ncas, wyz_psi_ne, x)
#                        psi_list.append(xwyz_psi)
#                        psi_ne = xwyz_psi_ne
#
#        psi_list = self.extract_vectors(psi_list, ncas, 4)
#
#        return psi_list, psi_ne
#
#
#    # Act annihilation operator (alpha spin)
#    def act_cre_a(self, wfn, ncas, nelec, orb):
#
#        if (wfn is not None) and (ncas - nelec[0] > 0):
#            c_wfn = self.cre_a(wfn, ncas, nelec, orb)
#            c_wfn_ne = (nelec[0] + 1, nelec[1])
#        else:
#            c_wfn = None
#            c_wfn_ne = None
#
#        return c_wfn, c_wfn_ne
#
#
#    # Act annihilation operator (beta spin)
#    def act_cre_b(self, wfn, ncas, nelec, orb):
#
#        if (wfn is not None) and (ncas - nelec[1] > 0):
#            c_wfn = self.cre_b(wfn, ncas, nelec, orb)
#            c_wfn_ne = (nelec[0], nelec[1] + 1)
#        else:
#            c_wfn = None
#            c_wfn_ne = None
#
#        return c_wfn, c_wfn_ne
#
#
#    # Act annihilation operator (alpha spin)
#    def act_des_a(self, wfn, ncas, nelec, orb):
#
#        if (wfn is not None) and (nelec[0] > 0):
#            a_wfn = self.des_a(wfn, ncas, nelec, orb)
#            a_wfn_ne = (nelec[0] - 1, nelec[1])
#        else:
#            a_wfn = None
#            a_wfn_ne = None
#
#        return a_wfn, a_wfn_ne
#
#
#    # Act annihilation operator (beta spin)
#    def act_des_b(self, wfn, ncas, nelec, orb):
#
#        if (wfn is not None) and (nelec[1] > 0):
#            a_wfn = self.des_b(wfn, ncas, nelec, orb)
#            a_wfn_ne = (nelec[0], nelec[1] - 1)
#        else:
#            a_wfn = None
#            a_wfn_ne = None
#
#        return a_wfn, a_wfn_ne
#
#
#    # Convert list of CI vectors into a numpy array
#    def extract_vectors(self, vec_list, ncas, dim):
#
#        if (vec_list[0] is not None):
#            ndim_a = vec_list[0].shape[0]
#            ndim_b = vec_list[0].shape[1]
#            vec_list = np.array(vec_list)
#            if dim == 1:
#                vec_list = vec_list.reshape(ncas, ndim_a, ndim_b).copy()
#            elif dim == 2:
#                vec_list = vec_list.reshape(ncas, ncas, ndim_a, ndim_b).transpose(1,0,2,3).copy()
#            elif dim == 3:
#                vec_list = vec_list.reshape(ncas, ncas, ncas, ndim_a, ndim_b).transpose(2,1,0,3,4).copy()
#            elif dim == 4:
#                vec_list = vec_list.reshape(ncas, ncas, ncas, ncas, ndim_a, ndim_b).transpose(3,2,1,0,4,5).copy()
#            else:
#                raise Exception("Unknown dimension")
#        else:
#            vec_list = None
#
#        return vec_list
#
#
#    def compute_overlap_braket_so(self, bra, ket, bra_ne, ket_ne):
#    
#        dim_bra = bra.shape
#        dim_ket = ket.shape
#    
#        dim = dim_bra[0 : -2] + dim_ket[0 : -2]
#    
#        overlap = np.zeros(dim)
#    
#        if (bra_ne[0] == ket_ne[0]) and (bra_ne[1] == ket_ne[1]):
#           bra = bra.reshape(-1, bra.shape[-2], bra.shape[-1])
#           bra = bra.reshape(bra.shape[0], -1).copy()
#    
#           ket = ket.reshape(-1, ket.shape[-2], ket.shape[-1])
#           ket = ket.reshape(ket.shape[0], -1).copy()
#    
#           overlap = np.dot(bra, ket.T).reshape(dim)
#    
#        return overlap
#
#
#    # Apply Hamiltonian on a vector: (H - E_0) |vec>
#    def apply_H(self, h1eff_act, eri, nelectron, e_zero, vec):
#
#        ncas = eri.shape[0]
#
#        h_eri = self.absorb_h1e(h1eff_act.copy(), eri, ncas, nelectron, 0.5)
#        temp = self.apply_2e_operator(h_eri, vec, ncas, nelectron)
#        ham_vec = temp - e_zero*vec
#
#        return ham_vec


def print_header():

    print("""\n--------------------------------------------------------------
    PRISM: Open-Source implementation of ab initio methods
            for excited states and spectroscopy

                       Version 0.1

           Copyright (C) 2019  Alexander Sokolov

    This program is distributed in the hope that it will
    be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License
    for more details.
--------------------------------------------------------------""")
