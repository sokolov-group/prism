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

from prism.mr_adc import compute

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
        self.mo_scf = interface.mo_scf
        self.ovlp = interface.ovlp
        self.nmo = interface.nmo
        self.nelec = interface.nelec
        self.enuc = interface.enuc
        self.e_scf = interface.e_scf

        self.symmetry = interface.symmetry
        self.group_repr_symm = interface.group_repr_symm

        # CASSCF specific
        self.ncore = interface.ncore
        self.ncas = interface.ncas
        self.nextern = interface.nextern
        self.nocc = self.ncas + self.ncore
        self.ref_nelecas = interface.ref_nelecas
        self.e_ref = interface.e_ref           # Total reference energy
        self.e_ref_cas = interface.e_ref_cas   # Reference active-space energy
        self.ref_wfn = interface.ref_wfn       # Reference wavefunction
        self.ref_wfn_spin_mult = interface.ref_wfn_spin_mult
        self.ref_wfn_deg = interface.ref_wfn_deg

        # MR-ADC specific variables
        self.method = "mr-adc(2)"       # Possible methods: mr-adc(0), mr-adc(1), mr-adc(2), mr-adc(2)-x
        self.method_type = "cvs-ip"     # Possible method types: cvs-ip
        # self.max_t_order = 1          # Maximum order of t amplitudes to compute
        self.ncasci = 6                 # Number of CASCI roots requested
        self.nroots = 6                 # Number of MR-ADC roots requested
        self.max_space = 100            # Maximum size of the Davidson trial space
        self.max_cycle = 50             # Maximum number of iterations in the Davidson procedure
        self.tol_e = 1e-8               # Tolerance for the energy in the Davidson procedure
        self.tol_r = 1e-5        # Tolerance for the residual in the Davidson procedure
        self.s_thresh_singles = 1e-5
        self.s_thresh_singles_t2 = 1e-3
        self.s_thresh_doubles = 1e-10
        self.semi_internal_projector = "gno" # Possible values: gno, gs

        self.analyze_spec_factor = False
        self.spec_factor_print_tol = 0.01

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

        # Integrals
        self.mo_energy = lambda:None
        self.h1eff = lambda:None
        self.v2e = lambda:None
        self.rdm = lambda:None
        self.t1 = lambda:None
        self.t2 = lambda:None
        self.dip_mom = None

        self.mo_energy.c = interface.mo_energy[:self.ncore]
        self.mo_energy.e = interface.mo_energy[self.nocc:]

        # Matrix blocks
        self.M_00 = None
        self.M_01 = lambda:None


    def _make_method_instance(self):
        cls_map = {
            "cvs-ip": CVSIPMRADC,
        }

        try:
            cls = cls_map[self.method_type]
        except KeyError:
            raise ValueError(f"Unknown method_type: {self.method_type}")

        # Create child object without calling its __init__
        method = cls.__new__(cls)

        # Copy all current state from parent
        method.__dict__ = self.__dict__.copy()

        # Optional subclass-specific post-init
        if hasattr(method, "_init_method"):
            method._init_method()

        return method

    def kernel(self):
        method = self._make_method_instance()
        return compute.kernel(method)

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self, obj):
        self._verbose = obj
        self.log.verbose = obj


# Classes for specific MRADC flavors go below
# Only attributes unique to each class should be added
class CVSIPMRADC(MRADC):

    def __init__(self, interface):
        super().__init__(interface)
        self._init_method()

    def _init_method(self):
        self.method_type = "cvs-ip"
        self.nval = None

