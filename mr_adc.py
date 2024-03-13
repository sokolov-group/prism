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

import prism.mr_adc_integrals as mr_adc_integrals
import prism.mr_adc_rdms as mr_adc_rdms
import prism.mr_adc_compute as mr_adc_compute

class MRADC:
    def __init__(self, interface):

        # General info
        self.interface = interface
        self.log = interface.log
        log = self.log

        log.info("\nInitializing MR-ADC...")

        if (interface.reference != "casscf"):
            log.info("MR-ADC requires CASSCF reference")
            raise Exception("MR-ADC requires CASSCF reference")

        self.stdout = interface.stdout
        self.verbose = interface.verbose
        self.max_memory = interface.max_memory
        self.current_memory = interface.current_memory

        self.temp_dir = interface.temp_dir
        self.tmpfile = lambda:None

        self.mo = interface.mo
        self.mo_hf = interface.mo_hf
        self.ovlp = interface.ovlp
        self.nmo = interface.nmo
        self.nelec = interface.nelec
        self.enuc = interface.enuc
        self.e_scf = interface.e_scf
        self.spin = interface.spin

        self.symmetry = interface.symmetry
        self.group_repr_symm = interface.group_repr_symm

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
        self.tol_e = 1e-8               # Tolerance for the energy in the Davidson procedure
        self.tol_davidson = 1e-5        # Tolerance for the residual in the Davidson procedure
        self.s_thresh_singles = 1e-5
        self.s_thresh_singles_t2 = 1e-3
        self.s_thresh_doubles = 1e-10

        self.analyze_spec_factor = False
        self.spec_factor_print_tol = 0.1

        self.e_cas_ci = None            # Active-space energies of CASCI states
        self.wfn_casci = None           # Active-space wavefunctions of CASCI states
        self.nelecasci = None           # Active-space number of electrons of CASCI states
        self.h0 = lambda:None           # Information about h0 excitation manifold
        self.h1 = lambda:None           # Information about h1 excitation manifold
        self.h_orth = lambda:None       # Information about orthonormalized excitation manifold
        self.S12 = lambda:None          # Matrices for orthogonalization of excitation spaces

        self.outcore_expensive_tensors = True # Store expensive (ooee) integrals and amplitudes on disk

        # Approximations
        self.approx_trans_moments = False

        # Parameters for the CVS implementation
        self.ncvs = None
        self.nval = None

        # Integrals
        self.mo_energy = lambda:None
        self.h1eff = lambda:None
        self.v2e = lambda:None
        self.rdm = lambda:None
        self.t1 = lambda:None
        self.t2 = lambda:None
        self.dip_mom = None

        # Matrix blocks
        self.M_00 = None
        self.M_01 = lambda:None

    def kernel(self):

        log = self.log
        self.method = self.method.lower()
        self.method_type = self.method_type.lower()

        if self.method not in ("mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            msg = "Unknown method %s" % self.method
            log.error(msg)
            raise Exception(msg)

        if self.method_type not in ("ee", "ip", "ea", "cvs-ip", "cvs-ee"):
            msg = "Unknown method type %s" % self.method_type
            log.error(msg)
            raise Exception(msg)

        if self.interface.with_df and self.method_type not in ('cvs-ip'):
            msg = "Density-fitting currently only compatible with CVS-IP method type."
            log.error(msg)
            raise Exception(msg)

        if self.spin > 0:
            msg = "This program currently does not work for open-shell molecules"
            log.error(msg)
            raise Exception(msg)

        if self.method_type == "cvs-ip" and self.ncvs is None:
            msg = "Method type %s requires setting the ncvs parameter" % self.method_type
            log.error(msg)
            raise Exception(msg)

        if self.method_type in ("cvs-ip", "cvs-ee"):

            if isinstance (self.ncvs, int):
                if self.ncvs < 1 or self.ncvs > self.ncore:
                    msg = '''Method type %s requires setting the ncvs parameter as a
                             positive integer that is smaller than ncore''' % self.method_type
                    log.error(msg)
                    raise Exception(msg)

                self.nval = self.ncore - self.ncvs

            else:
                msg = "Method type %s requires setting the ncvs parameter as a positive integer" % self.method_type
                log.error(msg)
                raise Exception(msg)

        # TODO: Temporary check of what methods are implemented in this version
        if self.method_type not in ("cvs-ip"):
            msg = "This spin-adapted version does not currently support method type %s" % self.method_type
            log.error(msg)
            raise Exception(msg)

        # Transform one- and two-electron integrals
        log.info("\nTransforming integrals to MO basis...")
        mr_adc_integrals.transform_integrals_1e(self)
        if self.interface.with_df:
            mr_adc_integrals.transform_Heff_integrals_2e_df(self)
            mr_adc_integrals.transform_integrals_2e_df(self)
        elif self.interface.v2e_ao is not None:
            mr_adc_integrals.transform_integrals_2e_incore(self)
        else:
            msg = "Out-of-core algorithm is not implemented in Prism."
            log.error(msg)
            raise Exception(msg)

        # Compute CASCI energies and reduced density matrices
        mr_adc_rdms.compute_gs_rdms(self)

        # TODO: Compute CASCI wavefunctions for excited states in the active space
        # mr_adc_rdms.compute_es_rdms(self)

        # Run MR-ADC computation
        ee, spec_factors, X = mr_adc_compute.kernel(self)

        return ee, spec_factors, X

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self, obj):
        self._verbose = obj
        self.log.verbose = obj
