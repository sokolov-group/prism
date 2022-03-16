import sys
import time
import numpy as np
import prism_beta.mr_adc_compute as mr_adc_compute
import prism_beta.mr_adc_integrals as mr_adc_integrals
import prism_beta.mr_adc_rdms as mr_adc_rdms

class MRADC:
    def __init__(self, interface):

        print ("Initializing MR-ADC...\n")
        sys.stdout.flush()

        if (interface.reference != "casscf"):
            raise Exception("MR-ADC requires CASSCF reference")

        # General info
        self.mo = interface.mo
        self.mo_hf = interface.mo_hf
        self.ovlp = interface.ovlp
        self.nmo = interface.nmo
        self.nelec = interface.nelec
        self.enuc = interface.enuc
        self.e_scf = interface.e_scf
        self.interface = interface
        self.print_level = interface.print_level

        # CASSCF specific
        self.ncore = interface.ncore
        self.ncas = interface.ncas
        self.nextern = interface.nextern
        self.nocc = self.ncas + self.ncore
        self.nelecas = interface.nelecas
        self.e_casscf = interface.e_casscf # Total CASSCF energy
        self.e_cas = interface.e_cas # Active-space CASSCF energy
        self.wfn_casscf = interface.wfn_casscf # Ground-state CASSCF wavefunction
        self.mo_energy = interface.mo_energy # Diagonal elements of the generalized Fock operator

        # Spin-orbital
        self.nmo_so = 2 * self.nmo
        self.ncore_so = 2 * self.ncore
        self.ncas_so = 2 * self.ncas
        self.nocc_so = 2 * self.nocc
        self.nextern_so = 2 * self.nextern

        # MR-ADC specific variables
        self.method = "mr-adc(2)" # Possible methods: mr-adc(0), mr-adc(1), mr-adc(2), mr-adc(2)-x
        self.method_type = "ip" # Possible method types: ee, ip, ea
        self.max_t_order = 1 # Maximum order of t amplitudes to compute
        self.ncasci = 10 # Number of CASCI roots requested
        self.nroots = 20 # Number of MR-ADC roots requested
        self.max_space = 200 # Maximum size of the Davidson trial space
        self.max_cycle = 80 # Maximum number of iterations in the Davidson procedure
        self.tol_davidson = 1e-5 # Tolerance for the residual in the Davidson procedure
        self.s_thresh_singles = 1e-6
        self.s_thresh_singles_t2 = 1e-3
        self.s_thresh_doubles = 1e-10
        self.s_damping_strength = None # If set to a positive value defines the range (log scale) of overlap matrix eigenvalues damped by a sigmoid function
        self.exact_semiinternals = True
        self.disk = False
        self.e_cas_ci = None # Active-space energies of CASCI states
        self.wfn_casci = None # Active-space wavefunctions of CASCI states
        self.nelecasci = None # Active-space number of electrons of CASCI states
        self.h0 = lambda:None # Information about h0 excitation manifold
        self.h1 = lambda:None # Information about h1 excitation manifold
        self.h_orth = lambda:None # Information about orthonormalized excitation manifold
        self.S12 = lambda:None # Matrices for orthogonalization of excitation spaces

        # Parameters for imaginary-time propagation
        self.tol_it = 1e-7
        self.delta_t = 0.001
        self.tmax = 20.0
        self.rk4_fix_step = False

        # Parameters for the CVS implementation
        self.ncvs = None
        self.ncvs_so = None
        self.nval_so = None
        self.cvs_relaxed = False
        self.cvs_mom = False
        self.cvs_npick = False

        # Integrals (spin-orbital)
        self.h1eff_act = None
        self.v2e_act = None
        self.h1e_so = None
        self.h1eff_so = None
        self.h1eff_act_so = None
        self.dm_so = None
        self.v2e_so = lambda:None
        self.mo_energy_so = lambda:None
        self.rdm_so = lambda:None


    def kernel(self):

        self.method = self.method.lower()
        self.method_type = self.method_type.lower()

        if self.method not in ("mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            raise Exception("Unknown method %s" % self.method)

        if self.method_type not in ("ee", "ip", "ea", "cvs-ip", "cvs-ee"):
            raise Exception("Unknown method type %s" % self.method_type)

        if self.nelecas[0] != self.nelecas[1]:
            raise Exception("This program currently does not work for open-shell molecules")

        if self.method_type == "cvs-ip" and self.ncvs is None:
            raise Exception("Method type %s requires setting the ncvs parameter" % self.method_type)

        if self.method_type in ("cvs-ip", "cvs-ee"):

            if isinstance (self.ncvs, int):
                if self.ncvs < 1 or self.ncvs > self.ncore:
                    raise Exception("Method type %s requires setting the ncvs parameter as a positive integer that is smaller than ncore" % self.method_type)

                self.ncvs_so = 2 * self.ncvs
                self.nval_so = self.ncore_so - self.ncvs_so

            else:
                raise Exception("Method type %s requires setting the ncvs parameter as a positive integer" % self.method_type)

        # Transform one- and two-electron integrals
        self.transform_integrals_to_so()

        # Compute CASCI energies and reduced density matrices
        self.compute_rdms_so()

        # Run MR-ADC computation
        ee, spec_factors = mr_adc_compute.kernel(self)

        if self.disk:
            self.clean_up_disk()

        return ee, spec_factors


    def transform_integrals_to_so(self):

        start_time = time.time()

        print ("Transforming integrals to spin-orbital basis...\n")
        sys.stdout.flush()

        mo = self.mo

        # One-electron integrals
        self.h1e_so = mr_adc_integrals.transform_1e_integrals_so(self.interface, mo)

        if self.method_type in ('ee','cvs-ee'):

            print ("Transforming dipole moments to spin-orbital basis...\n")
            sys.stdout.flush()
            self.dm_so = np.zeros((3, self.nmo_so, self.nmo_so))

            # Dipole moments
            for i in range(3):
                self.dm_so[i] = mr_adc_integrals.transform_1e_integrals_so(self.interface, mo, self.interface.dm_ao[i])

        # Two-electron integrals
        mo_c = mo[:, :self.ncore].copy()
        mo_a = mo[:, self.ncore:self.nocc].copy()
        mo_e = mo[:, self.nocc:].copy()

        self.v2e_act = self.interface.transform_2e_integrals(mo_a, mo_a, mo_a, mo_a)
        self.v2e_act = self.v2e_act.reshape(self.ncas, self.ncas, self.ncas, self.ncas).copy()
        self.v2e_so.aaaa = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_a, mo_a, mo_a, mo_a)

        if self.method_type == "ip" or self.method_type == "ea" or self.method_type == "cvs-ip":
            if self.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
                self.v2e_so.caea = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_e, mo_a)
                self.v2e_so.caaa = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_a, mo_a)
                self.v2e_so.aaea = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_a, mo_a, mo_e, mo_a)
                self.v2e_so.caca = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_c, mo_a)
    
            if self.method in ("mr-adc(2)", "mr-adc(2)-x"):
                self.v2e_so.ccee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_e, mo_e)
                self.v2e_so.ccea = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_e, mo_a)
                self.v2e_so.caee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_e, mo_e)
                self.v2e_so.ccaa = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_a, mo_a)
                self.v2e_so.aaee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_a, mo_a, mo_e, mo_e)
                self.v2e_so.ccca = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_c, mo_a)
                self.v2e_so.ccce = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_c, mo_e)
                self.v2e_so.cace = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_c, mo_e)
                self.v2e_so.ceaa = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_e, mo_a, mo_a)
                self.v2e_so.cece = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_e, mo_c, mo_e)
                self.v2e_so.ceee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_e, mo_e, mo_e)
                self.v2e_so.aeee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_a, mo_e, mo_e, mo_e)
                self.v2e_so.ceae = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_e, mo_a, mo_e)
                self.v2e_so.aeae = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_a, mo_e, mo_a, mo_e)

                # Need for mr-adc(2)-x
                if self.method == "mr-adc(2)-x":
                    self.v2e_so.cccc = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_c, mo_c)

                if (self.method == "mr-adc(2)-x" and self.max_t_order > 1) or (self.method == "mr-adc(2)-x" and self.method_type == "ea"):
                    if self.disk:
                        self.v2e_so.eeee = mr_adc_integrals.transform_asym_integrals_so_disk(self.interface, mo_e, mo_e, mo_e, mo_e)
                    else:
                        self.v2e_so.eeee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_e, mo_e, mo_e, mo_e)

        elif self.method_type == "ee":
            if self.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
                self.v2e_so.caea = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_e, mo_a)
                self.v2e_so.caaa = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_a, mo_a)
                self.v2e_so.aaea = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_a, mo_a, mo_e, mo_a)
                self.v2e_so.caca = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_c, mo_a)
                self.v2e_so.ccee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_e, mo_e)
                self.v2e_so.aaee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_a, mo_a, mo_e, mo_e)
                self.v2e_so.caee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_e, mo_e)
                self.v2e_so.cace = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_c, mo_e)
                self.v2e_so.cece = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_e, mo_c, mo_e)
                self.v2e_so.aeae = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_a, mo_e, mo_a, mo_e)
                self.v2e_so.ceae = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_e, mo_a, mo_e)
                self.v2e_so.ccea = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_e, mo_a)
                self.v2e_so.ceaa = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_e, mo_a, mo_a)
                self.v2e_so.ccaa = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_a, mo_a)

            if self.method in ("mr-adc(2)", "mr-adc(2)-x"): 
                self.v2e_so.ccca = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_c, mo_a)
                self.v2e_so.ccce = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_c, mo_e)
                self.v2e_so.ceee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_e, mo_e, mo_e)
                self.v2e_so.aeee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_a, mo_e, mo_e, mo_e)
                
                if self.method == "mr-adc(2)-x":
                    self.v2e_so.cccc = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_c, mo_c)

                    if self.disk:
                        self.v2e_so.eeee = mr_adc_integrals.transform_asym_integrals_so_disk(self.interface, mo_e, mo_e, mo_e, mo_e)
                    else:
                        self.v2e_so.eeee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_e, mo_e, mo_e, mo_e)
                
        #TODO: remove redundant integrals
        elif self.method_type == "cvs-ee":
            if self.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
                self.v2e_so.caea = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_e, mo_a)
                self.v2e_so.caaa = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_a, mo_a)
                self.v2e_so.aaea = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_a, mo_a, mo_e, mo_a)
                self.v2e_so.caca = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_c, mo_a)
                self.v2e_so.ccee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_e, mo_e)
                self.v2e_so.aaee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_a, mo_a, mo_e, mo_e)
                self.v2e_so.caee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_e, mo_e)
                self.v2e_so.cace = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_a, mo_c, mo_e)
                self.v2e_so.cece = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_e, mo_c, mo_e)
                self.v2e_so.aeae = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_a, mo_e, mo_a, mo_e)
                self.v2e_so.ceae = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_e, mo_a, mo_e)
                self.v2e_so.ccea = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_e, mo_a)
                self.v2e_so.ceaa = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_e, mo_a, mo_a)
                self.v2e_so.ccaa = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_a, mo_a)

            if self.method in ("mr-adc(2)", "mr-adc(2)-x"): 
                self.v2e_so.ccca = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_c, mo_a)
                self.v2e_so.ccce = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_c, mo_e)
                self.v2e_so.ceee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_e, mo_e, mo_e)
                self.v2e_so.aeee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_a, mo_e, mo_e, mo_e)
                
                if self.method == "mr-adc(2)-x":
                    self.v2e_so.cccc = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_c, mo_c, mo_c, mo_c)

                    if self.disk:
                        self.v2e_so.eeee = mr_adc_integrals.transform_asym_integrals_so_disk(self.interface, mo_e, mo_e, mo_e, mo_e)
                    else:
                        self.v2e_so.eeee = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo_e, mo_e, mo_e, mo_e)
                
        # Effective one-electron integrals
        gcgc = mr_adc_integrals.transform_asym_integrals_so(self.interface, mo, mo_c, mo, mo_c)
        self.h1eff_so = self.h1e_so + np.einsum('prqr->pq', gcgc)
        self.h1eff_act_so = self.h1eff_so[self.ncore_so:self.nocc_so, self.ncore_so:self.nocc_so].copy()
        self.h1eff_act = self.h1eff_act_so[::2, ::2].copy()

        # Store diagonal elements of the generalized Fock operator in spin-orbital basis
        self.mo_energy_so.c = np.zeros(self.ncore_so)
        self.mo_energy_so.c[::2] = self.mo_energy[:self.ncore]
        self.mo_energy_so.c[1::2] = self.mo_energy[:self.ncore]
        self.mo_energy_so.e = np.zeros(self.nextern_so)
        self.mo_energy_so.e[::2] = self.mo_energy[(self.ncore + self.ncas):]
        self.mo_energy_so.e[1::2] = self.mo_energy[(self.ncore + self.ncas):]

        print ("Time for transforming integrals:                  %f sec\n" % (time.time() - start_time))


    def compute_rdms_so(self):

        start_time = time.time()

        wfn_casci = None
        e_cas_ci = None

        print ("Computing ground-state RDMs in the spin-orbital basis...\n")
        sys.stdout.flush()

        # Compute ground-state RDMs
        if self.ncas != 0:
            self.rdm_so.ca = mr_adc_rdms.compute_rdm_ca_so(self.interface)
            self.rdm_so.ccaa = mr_adc_rdms.compute_rdm_ccaa_so(self.interface)
            if self.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
                self.rdm_so.cccaaa = mr_adc_rdms.compute_rdm_cccaaa_so(self.interface)
            if (self.method in ("mr-adc(2)", "mr-adc(2)-x") and self.max_t_order > 1) \
            or self.method_type in ("ee", "cvs-ee"):
                self.rdm_so.ccccaaaa = mr_adc_rdms.compute_rdm_ccccaaaa_so(self.interface)
            else:
                self.rdm_so.ccccaaaa = None

        else:
            self.rdm_so.ca = np.zeros((self.ncas, self.ncas))
            self.rdm_so.ccaa =  np.zeros((self.ncas, self.ncas, self.ncas, self.ncas))
            if self.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
                self.rdm_so.cccaaa =  np.zeros((self.ncas, self.ncas, self.ncas, self.ncas, self.ncas, self.ncas))
            if (self.method in ("mr-adc(2)", "mr-adc(2)-x") and self.max_t_order > 1) \
            or self.method_type in ("ee", "cvs-ee"):
                self.rdm_so.ccccaaaa =  np.zeros((self.ncas, self.ncas, self.ncas, self.ncas, self.ncas, self.ncas, self.ncas, self.ncas))
            else:
                self.rdm_so.ccccaaaa = None

        print ("Computing excited-state CASCI wavefunctions...\n")
        sys.stdout.flush()

        # Compute CASCI wavefunctions for excited states in the active space
        if self.method_type == "ip":

            self.nelecasci = (self.nelecas[0] - 1, self.nelecas[1])

            if (0 <= self.nelecasci[0] <= self.ncas and 0 <= self.nelecasci[1] <= self.ncas):
                e_cas_ci, wfn_casci = self.interface.compute_casci_ip_ea(self.ncasci, self.method_type)
            else:
                self.nelecasci = None

        elif self.method_type == "ea":

            self.nelecasci = (self.nelecas[0] + 1, self.nelecas[1])

            if (0 <= self.nelecasci[0] <= self.ncas and 0 <= self.nelecasci[1] <= self.ncas):
                e_cas_ci, wfn_casci = self.interface.compute_casci_ip_ea(self.ncasci, self.method_type)
            else:
                self.nelecasci = None

        elif self.method_type == "ee":

            self.nelecasci = (self.nelecas[0], self.nelecas[1])
            
            if (self.nelecasci[0] != 0 or self.nelecasci[1] != 0) and (self.nelecasci[0] != self.ncas or self.nelecasci[1] != self.ncas): 
                e_cas_ci, wfn_casci = self.interface.compute_casci_ee(self.ncasci)
            else:
                self.nelecasci = None

        elif self.method_type in ("cvs-ip", "cvs-ee"):

            self.nelecasci = None

        else:
            raise Exception("MR-ADC is not implemented for %s" % self.method_type)

        if self.nelecasci is not None:
            self.ncasci = len(e_cas_ci)
            self.wfn_casci = wfn_casci
            self.e_cas_ci = e_cas_ci
        else:
            if self.method_type in ("cvs-ip", "cvs-ee"):
                print ("Requested method type %s does not require running a CASCI calculation..." % self.method_type)
            else:
                print ("WARNING: active orbitals are either empty of completely filled...")
            print ("Skipping the CASCI calculation...")
            self.ncasci = 0
            self.wfn_casci = None
            self.e_cas_ci = None

        print ("\nFinal number of excited CASCI states: %d\n" % self.ncasci)

        if self.ncasci > 0:

            # Compute transition RDMs between the ground (reference) state and target CASCI states
            print ("Computing transition RDMs between reference and target CASCI states...\n")
            sys.stdout.flush()

            if self.method_type == "ip":
                # Compute CASCI states with higher MS
                Sp_wfn_casci = []
                Sp_wfn_ne = None
                for wfn in wfn_casci:
                    Sp_wfn, Sp_wfn_ne = self.interface.apply_S_plus(wfn, self.ncas, self.nelecasci)
                    Sp_wfn_casci.append(Sp_wfn)

                self.rdm_so.ct = np.zeros((2 * self.ncasci, self.ncas_so))
                self.rdm_so.ct[:self.ncasci] = mr_adc_rdms.compute_rdm_ct_so(self.interface, wfn_casci, self.nelecasci).copy()
                self.rdm_so.ct[self.ncasci:] = mr_adc_rdms.compute_rdm_ct_so(self.interface, Sp_wfn_casci, Sp_wfn_ne).copy()

                self.rdm_so.ccat = np.zeros((2 * self.ncasci, self.ncas_so, self.ncas_so, self.ncas_so))
                self.rdm_so.ccat[:self.ncasci] = mr_adc_rdms.compute_rdm_ccat_so(self.interface, wfn_casci, self.nelecasci).copy()
                self.rdm_so.ccat[self.ncasci:] = mr_adc_rdms.compute_rdm_ccat_so(self.interface, Sp_wfn_casci, Sp_wfn_ne).copy()

                if self.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
                    self.rdm_so.cccaat = np.zeros((2 * self.ncasci, self.ncas_so, self.ncas_so, self.ncas_so, self.ncas_so, self.ncas_so))
                    self.rdm_so.cccaat[:self.ncasci] = mr_adc_rdms.compute_rdm_cccaat_so(self.interface, wfn_casci, self.nelecasci).copy()
                    self.rdm_so.cccaat[self.ncasci:] = mr_adc_rdms.compute_rdm_cccaat_so(self.interface, Sp_wfn_casci, Sp_wfn_ne).copy()

            elif self.method_type == "ea":
                # Compute CASCI states with lower MS
                Sm_wfn_casci = []
                Sm_wfn_ne = None
                for wfn in wfn_casci:
                    Sm_wfn, Sm_wfn_ne = self.interface.apply_S_minus(wfn, self.ncas, self.nelecasci)
                    Sm_wfn_casci.append(Sm_wfn)

                self.rdm_so.tc = np.zeros((2 * self.ncasci, self.ncas_so))
                self.rdm_so.tc[:self.ncasci] = mr_adc_rdms.compute_rdm_tc_so(self.interface, wfn_casci, self.nelecasci).copy()
                self.rdm_so.tc[self.ncasci:] = mr_adc_rdms.compute_rdm_tc_so(self.interface, Sm_wfn_casci, Sm_wfn_ne).copy()

                self.rdm_so.tcca = np.zeros((2 * self.ncasci, self.ncas_so, self.ncas_so, self.ncas_so))
                self.rdm_so.tcca[:self.ncasci] = mr_adc_rdms.compute_rdm_tcca_so(self.interface, wfn_casci, self.nelecasci).copy()
                self.rdm_so.tcca[self.ncasci:] = mr_adc_rdms.compute_rdm_tcca_so(self.interface, Sm_wfn_casci, Sm_wfn_ne).copy()

                if self.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
                    self.rdm_so.tcccaa = np.zeros((2 * self.ncasci, self.ncas_so, self.ncas_so, self.ncas_so, self.ncas_so, self.ncas_so))
                    self.rdm_so.tcccaa[:self.ncasci] = mr_adc_rdms.compute_rdm_tcccaa_so(self.interface, wfn_casci, self.nelecasci).copy()
                    self.rdm_so.tcccaa[self.ncasci:] = mr_adc_rdms.compute_rdm_tcccaa_so(self.interface, Sm_wfn_casci, Sm_wfn_ne).copy()

            elif self.method_type == "ee":
                self.rdm_so.tca = mr_adc_rdms.compute_rdm_tca_so(self.interface, wfn_casci, self.nelecasci).copy()
                self.rdm_so.tccaa = mr_adc_rdms.compute_rdm_tccaa_so(self.interface, wfn_casci, self.nelecasci).copy()

                if self.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
                    self.rdm_so.tcccaaa = mr_adc_rdms.compute_rdm_tcccaaa_so(self.interface, wfn_casci, self.nelecasci).copy()

            print ("Computing transition RDMs between target CASCI states...\n")
            sys.stdout.flush()

            # Compute transition RDMs between two target CASCI states
            self.rdm_so.tcat = mr_adc_rdms.compute_rdm_tcat_so(self.interface, wfn_casci, self.nelecasci)
            self.rdm_so.tccaat = mr_adc_rdms.compute_rdm_tccaat_so(self.interface, wfn_casci, self.nelecasci)

        print ("Time for computing RDMs:                          %f sec\n" % (time.time() - start_time))


    def clean_up_disk(self):

        if self.method_type in ("ee","cvs-ee","ea") and self.method == "mr-adc(2)-x":
            mr_adc_integrals.remove_dataset(self.v2e_so.eeee)
