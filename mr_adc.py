import sys
import time
import numpy as np
import prism.mr_adc_integrals as mr_adc_integrals
import prism.mr_adc_rdms as mr_adc_rdms
import prism.mr_adc_compute as mr_adc_compute

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
        self.e_casscf = interface.e_casscf      # Total CASSCF energy
        self.e_cas = interface.e_cas            # Active-space CASSCF energy
        self.wfn_casscf = interface.wfn_casscf  # Ground-state CASSCF wavefunction

        # MR-ADC specific variables
        self.method = "mr-adc(2)"       # Possible methods: mr-adc(0), mr-adc(1), mr-adc(2), mr-adc(2)-x
        self.method_type = "ip"         # Possible method types: ee, ip, ea
        # self.max_t_order = 1          # Maximum order of t amplitudes to compute
        self.ncasci = 6                 # Number of CASCI roots requested
        self.nroots = 6                 # Number of MR-ADC roots requested
        self.max_space = 100            # Maximum size of the Davidson trial space
        self.max_cycle = 50             # Maximum number of iterations in the Davidson procedure
        self.tol_davidson = 1e-5        # Tolerance for the residual in the Davidson procedure
        self.s_thresh_singles = 1e-5
        self.s_thresh_singles_t2 = 1e-3
        self.s_thresh_doubles = 1e-10

        self.s_damping_strength = None  # If set to a positive value defines the range (log scale) of overlap matrix
                                        # eigenvalues damped by a sigmoid function

        self.e_cas_ci = None            # Active-space energies of CASCI states
        self.wfn_casci = None           # Active-space wavefunctions of CASCI states
        self.nelecasci = None           # Active-space number of electrons of CASCI states
        self.h0 = lambda:None           # Information about h0 excitation manifold
        self.h1 = lambda:None           # Information about h1 excitation manifold
        self.h_orth = lambda:None       # Information about orthonormalized excitation manifold
        self.S12 = lambda:None          # Matrices for orthogonalization of excitation spaces
        self.debug_mode = False         # Debug printing statements

        # Parameters for the CVS implementation
        self.ncvs = None
        self.nval = None

        # Integrals (spin-orbital)
        self.h1eff = None
        self.h1eff_act = None
        self.dip_mom = None
        self.v2e = lambda:None
        self.mo_energy = lambda:None
        self.rdm = lambda:None

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

                self.nval = self.ncore - self.ncvs

            else:
                raise Exception("Method type %s requires setting the ncvs parameter as a positive integer" % self.method_type)

        # TODO: Temporary check of what methods are implemented in this version
        if self.method_type not in ("ip", "cvs-ip"):
            raise Exception("This spin-adapted version does not currently support method type %s" % self.method_type)

        # Transform one- and two-electron integrals
        # TODO: implement DF integral transformation
        mr_adc_integrals.transform_integrals_1e(self)
        mr_adc_integrals.transform_integrals_2e_incore(self)

        # Compute CASCI energies and reduced density matrices
        mr_adc_rdms.compute_gs_rdms(self)

        ### DEBUG
        # if self.debug_mode:
        #     dyall_hamiltonian(self)

        #     import prism.mr_adc_amplitudes as mr_adc_amplitudes
        #     e_0, t1_ccee = mr_adc_amplitudes.compute_t1_0(self)
        #     e_p1, t1_ccea = mr_adc_amplitudes.compute_t1_p1(self)
        #     e_m1, t1_caee = mr_adc_amplitudes.compute_t1_m1(self)
        #     e_p2, t1_ccaa = mr_adc_amplitudes.compute_t1_p2(self)
        #     e_m2, t1_aaee = mr_adc_amplitudes.compute_t1_m2(self)
        #     mr_adc_amplitudes.compute_t1_0p(self)
        #     mr_adc_amplitudes.compute_t1_p1p(self)
        #     e_0p, t1_ce, t1_caea, t1_caae = mr_adc_amplitudes.compute_t1_0p_sanity_check(self)
        #     e_p1p, t1_ca, t1_caaa = mr_adc_amplitudes.compute_t1_p1p_sanity_check(self)
        #     e_m1p, t1_ae, t1_aaea, t1_aaae = mr_adc_amplitudes.compute_t1_m1p_sanity_check(self)
        #     mr_adc_amplitudes.compute_t1_p1p_singles_sanity_check(self)
        #     mr_adc_amplitudes.compute_t1_p1p_singles(self)

        #     t1_amp = (t1_ce, t1_ca, t1_ae, t1_caea, t1_caae, t1_caaa, t1_aaea, t1_aaae, t1_ccee, t1_ccea, t1_caee, t1_ccaa, t1_aaee)
        #     mr_adc_amplitudes.compute_t2_0p_singles(self, t1_amp)

        # return "ee", "spec_factors"
        ### DEBUG

        # TODO: Compute CASCI wavefunctions for excited states in the active space
        # mr_adc_rdms.compute_es_rdms(self)

        # Run MR-ADC computation
        ee, spec_factors = mr_adc_compute.kernel(self)

        return ee, spec_factors


def dyall_hamiltonian(mr_adc):
    """Zeroth Order Dyall Hamiltonian"""

    # Testing Dyall Hamiltonian expected value
    print ("Calculating the Spin-Adapted Dyall Hamiltonian...")

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables needed
    h_aa = mr_adc.h1eff[mr_adc.ncore:mr_adc.nocc, mr_adc.ncore:mr_adc.nocc].copy()
    rdm_ca = mr_adc.rdm.ca
    v_aaaa = mr_adc.v2e.aaaa
    rdm_ccaa = mr_adc.rdm.ccaa
    mo_c = mr_adc.mo[:, :mr_adc.ncore].copy()
    mo_a = mr_adc.mo[:, mr_adc.ncore:mr_adc.nocc].copy()

    # Calculating E_fc
    ## Calculating h_cc term
    h_cc = 2.0 * mr_adc.h1e[:mr_adc.ncore, :mr_adc.ncore].copy()

    ## Calculating v_cccc term
    v_cccc = mr_adc_integrals.transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_c)

    # Calculating temp_E_fc
    temp_E_fc  = einsum('ii', h_cc, optimize = True)
    temp_E_fc += 2.0 * einsum('ijij', v_cccc, optimize = True)
    temp_E_fc -= einsum('jiij', v_cccc, optimize = True)

    # Calculating H_act
    temp  = einsum('xy,xy', h_aa, rdm_ca, optimize = einsum_type)
    temp += 1/2 * einsum('xyzw,xyzw', v_aaaa, rdm_ccaa, optimize = einsum_type)

    print ("\n>>> SA Expected value of Zeroth-order Dyall Hamiltonian: {:}".format(temp + temp_E_fc + mr_adc.interface.enuc))
