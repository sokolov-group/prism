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
# Authors: Carlos E. V. de Moura <carlosevmoura@gmail.com>
#          Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#


from prism.nevpt import compute
from prism.nevpt import nevpt
from prism.nevpt import qd_nevpt


class NEVPT:
    def __init__(self, interface):

        # General info
        self.interface = interface
        self.log = interface.log
        log = self.log

        log.info("Initializing state-specific fully internally contracted NEVPT...")

        if (interface.reference not in ("casscf", "casci", "sa-casscf", "ms-casci")):
            log.info("The NEVPT code does not support %s reference" % interface.reference)
            raise Exception("The NEVPT code does not support %s reference" % interface.reference)

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
        self.e_ref = interface.e_ref              # Total reference energy
        self.e_ref_cas = interface.e_ref_cas      # Reference active-space energy
        self.ref_wfn = interface.ref_wfn          # Reference wavefunction
        self.ref_wfn_spin_mult = interface.ref_wfn_spin_mult
        self.ref_wfn_deg = interface.ref_wfn_deg

        # NEVPT specific variables
        self.method = "nevpt2"                    # Possible methods: nevpt2
        self.method_type = "ss"                   # Possible methods: ss (state-specific) and qd (quasidegenerate)
        self.nfrozen = None                       # Number of lowest-energy (core) orbitals that will be left uncorrelated ("frozen core")
        self.compute_singles_amplitudes = False   # Include singles amplitudes in the NEVPT2 energy?
        self.semi_internal_projector = "gno"      # Possible values: gno, gs, only matters when compute_singles_amplitudes is True
        self.s_thresh_singles = 1e-8
        self.s_thresh_doubles = 1e-8
        
        self.shift_type_p1p = None                # Level shift type for [+1']: imaginary, DSRG
        self.shift_type_m1p = None                # Level shift type for [-1']: imaginary, DSRG
        self.shift_type_0p = None                 # Level shift type for [0']: imaginary, DSRG
        self.shift_epsilon = 0.01                 # Level shift value (in Hartree)

        self.S12 = lambda:None                    # Matrices for orthogonalization of excitation spaces
        
        self.outcore_expensive_tensors = True     # Store expensive (ooee) integrals and amplitudes on disk   
        self.e_tot = None                         # Total energies
        self.e_corr = None                        # Correlation energies
        self.spin_mult = self.ref_wfn_spin_mult   # Spin multiplicities
        self.properties = {}                      # Dictionary to store computed properties
        self.compute_ntos = False                 # Option for NTO computation

        # Integrals
        self.mo_energy = lambda:None
        self.h1eff = lambda:None
        self.v2e = lambda:None

        # Amplitudes
        self.t1 = None
        self.t1_0 = None 
        self.keep_amplitudes = True

        # Compute correlated RDMs
        self.rdm_order = 0                         # Default value of 0 (uncorrelated), 2 for correlated

        # For SOC
        self.soc = None                            # Spin–orbit coupling. Possible methods: Breit-Pauli (BP), DKH1 (x2c-1)
        self.gtensor = False                       # Enable calculating g-tensors (requires soc)
        self.gtensor_origin_type = 'charge'        # Origin of coordinate system for g-tensor calculations. Possible values: charge, GIAO, atom1 or user-defined point (list)
        self.gtensor_target_state = 1              # Target state for g-tensor calculation. Default is the ground state (target_state = 1).
        self.h_evec_soc = None

    def _make_method_instance(self):
        cls_map = {
            "qd": QDNEVPT,
        }

        try:
            cls = cls_map[self.method_type]
        except KeyError:
            raise ValueError(f"Unknown method_type: {self.method_type}")

        # Create child object without calling its __init__
        method = cls.__new__(cls)

        # Copy all current state from parent
        method.__dict__ = self.__dict__

        # Optional subclass-specific post-init
        if hasattr(method, "_init_method"):
            method._init_method()

        return method

    def kernel(self):

        self.method_type = self.method_type.lower()

        method = None
        if (self.method_type != "ss"):
            method = self._make_method_instance()
        else:
            method = self

        # Run NEVPT computation
        e_tot, e_corr, osc = compute.kernel(method)

        if not self.keep_amplitudes:
            del(self.t1)
            del(self.t1_0)

        return e_tot, e_corr, osc 

    def compute_energy(self):

        e_tot, e_corr = nevpt.compute_energy(self)

        # State-interaction spin–orbit coupling
        if self.soc:
            from prism.nevpt import soc
            e_tot, e_corr = soc.state_interaction_soc(self)

        return e_tot, e_corr

    def make_rdm1(self, L = None, R = None, type = 'all'):

        # For state-interaction spin–orbit coupling, transform to microstate basis
        if self.soc:
            rdm1 = nevpt.make_rdm1(self)

            from prism.nevpt import soc
            rdm1 = soc.transform_rdm1(self, rdm1, L, R, type)

        else:
            rdm1 = nevpt.make_rdm1(self, L, R, type)

        return rdm1

    def make_rdm1s(self, wfn=None, wfn_ref_nelecas=None, L = None, R = None, type = 'all'):

        rdm1s = nevpt.make_rdm1s(self, wfn, wfn_ref_nelecas, L, R, type)

        return rdm1s

    def compute_properties(self):

        return nevpt.compute_properties(self)

    def analyze(self):

        self.method_type = self.method_type.lower()

        if self.method_type == "ss":
            method = self
        else:
            method = self._make_method_instance()
        self.__dict__.update(method.__dict__)

        return compute.analyze(method)

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, obj):
        self._verbose = obj
        self.log.verbose = obj


# Quasidegenerate NEVPT
# Only attributes unique to each class should be added
class QDNEVPT(NEVPT):

    def __init__(self, interface):

        super().__init__(interface)
        self._init_method()

    def _init_method(self):
        self.method_type = "qd"
        self.h_evec = None # Eigenvectors of effective Hamiltonian

    def compute_energy(self):

        e_tot, e_corr = qd_nevpt.compute_energy(self)

        # State-interaction spin–orbit coupling
        if self.soc:
            from prism.nevpt import soc
            e_tot, e_corr = soc.state_interaction_soc(self)

        return e_tot, e_corr

    def make_rdm1(self, L = None, R = None, type = 'all'):

        if self.soc:
            rdm1 = qd_nevpt.make_rdm1(self)

            from prism.nevpt import soc
            rdm1 = soc.transform_rdm1(self, rdm1, L, R, type)

        else:
            rdm1 = qd_nevpt.make_rdm1(self, L, R, type)

        return rdm1

    def make_rdm1s(self, wfn=None, wfn_ref_nelecas=None, L = None, R = None, type = 'all'):

        rdm1s = qd_nevpt.make_rdm1s(self, wfn, wfn_ref_nelecas, L, R, type)

        return rdm1s

    def compute_properties(self):

        return qd_nevpt.compute_properties(self)


