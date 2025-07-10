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
#                  Ilia M. Mazin <ilia.mazin@gmail.com>
#              Donna H. Odhiambo <donna.odhiambo@proton.me>

from typing import Optional, Tuple, List
from dataclasses import dataclass

import prism.mr_adc_integrals as mr_adc_integrals
import prism.mr_adc_rdms as mr_adc_rdms
import prism.mr_adc_compute as mr_adc_compute

##TODO: Add log.extra statements for initialization and configuration

@dataclass
class MRADCConfig:
    """ General configuration for MR-ADC calculations."""
    
    # Method settings
    method: str = "mr-adc(2)"               # Possible methods: mr-adc(0), mr-adc(1), mr-adc(2), mr-adc(2)-sx, mr-adc(2)-x
    method_type: str = "ip"                 # Possible method types: ee, ip, ea

    ncasci: int = 6                         # Number of CASCI roots requested
    nroots: int = 6                         # Number of MR-ADC roots requested
    
    # Davidson solver settings
    max_space: int = 100                    # Maximum size of the Davidson trial space
    max_cycle: int = 50                     # Maximum number of iterations in the Davidson procedure
    tol_e: float = 1e-8                     # Tolerance for the energy in the Davidson procedure
    tol_r: float = 1e-5                     # Tolerance for the residual in the Davidson procedure
    
    # Overlap settings
    s_thresh_singles: float = 1e-5          # Singles truncation threshold    
    s_thresh_doubles: float = 1e-10         # Doubles truncation threshold
    semi_internal_projector: str = "gno"    # Possible projection technique for amplitudes: gno, gs
    
    # Other settings
    analyze_spec_factor: bool = False       # Spectroscopic factors analysis
    spec_factor_print_tol: float = 0.01     # Tolerance for spectroscopic factors analysis
    outcore_expensive_tensors: bool = True  # Store expensive (ooee) integrals and amplitudes on disk 
    approx_trans_moments: bool = False      # Approximate transition moments
    
    # CVS parameters
    ncvs: Optional[int] = None              # Number of core orbitals selected for CVS (if None, no CVS)
    nval: Optional[int] = None              # Number of valence orbitals (ncore - ncvs)

    ## New feature
    use_mradc1_guess: bool = False          # Use MR-ADC(1) guess vectors for MR-ADC(2)-/-SX/-X calculations


class MRADCError(Exception):
    """Custom exception for MR-ADC errors."""
    pass


class MRADC:
    """MR-ADC class for performing multi-reference algebraic diagrammatic construction calculations."""

    # Supported methods and method types
    SUPPORTED_METHODS = {"mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-sx", "mr-adc(2)-x"}
    SUPPORTED_METHOD_TYPES = {"ee", "ip", "ea", "cvs-ip", "cvs-ee"}
    DF_COMPATIBLE_TYPES = {"cvs-ip", "cvs-ee"}
    IMPLEMENTED_TYPES = {"cvs-ip", "cvs-ee"}
    CVS_TYPES = {"cvs-ip", "cvs-ee"}

    def __init__(self, interface, config: Optional[MRADCConfig] = None):
        """ Initialize the MRADC object."""
        self.interface = interface
        self.log = interface.log
        self.config = config or MRADCConfig()
        
        self._validate_interface()
        self._initialize_basic_attributes()
        self._initialize_casscf_attributes()
        self._initialize_config_attributes()
        self._initialize_matrices()
        
        self.log.info("\nMR-ADC initialized successfully")
    
    def _validate_interface(self) -> None:
        """Validate the interface and ensure it is compatible with MR-ADC."""
        if self.interface.reference_type != "casscf":
            msg = "MR-ADC requires CASSCF reference"
            self.log.error(msg)
            raise MRADCError(msg)
    
    def _initialize_basic_attributes(self) -> None:
        """Initialize basic attributes from the interface."""

        # General info
        self.stdout = self.interface.stdout
        self.max_memory = self.interface.max_memory
        self.current_memory = self.interface.current_memory
        self.temp_dir = self.interface.temp_dir
        
        # Molecular orbital and system info
        self.mo = self.interface.mo
        self.mo_scf = self.interface.mo_scf
        self.ovlp = self.interface.ovlp
        self.nmo = self.interface.nmo
        self.nelec = self.interface.nelec
        self.enuc = self.interface.enuc
        self.e_scf = self.interface.e_scf
        
        # Symmetry information
        self.symmetry = self.interface.symmetry
        self.group_repr_symm = self.interface.group_repr_symm
    
    def _initialize_casscf_attributes(self) -> None:
        """Initialize attributes specific to the CASSCF reference."""

        self.ncore = self.interface.ncore
        self.ncas = self.interface.ncas
        self.nextern = self.interface.nextern
        self.nocc = self.ncas + self.ncore
        self.ref_nelecas = self.interface.ref_nelecas

        # Reference energies and wavefunctions
        self.e_ref = self.interface.e_ref
        self.e_ref_cas = self.interface.e_ref_cas
        self.ref_wfn = self.interface.ref_wfn
        self.ref_wfn_spin_mult = self.interface.ref_wfn_spin_mult
        self.ref_wfn_deg = self.interface.ref_wfn_deg
           
    def _initialize_config_attributes(self) -> None:
        """Initialize attributes from MRADCConfig."""

        # Method settings
        self.method = self.config.method
        self.method_type = self.config.method_type
        
        # CASCI and root settings
        self.ncasci = self.config.ncasci
        self.nroots = self.config.nroots
        
        # Davidson solver settings
        self.max_space = self.config.max_space
        self.max_cycle = self.config.max_cycle
        self.tol_e = self.config.tol_e
        self.tol_r = self.config.tol_r
        
        # Overlap settings
        self.s_thresh_singles = self.config.s_thresh_singles
        self.s_thresh_doubles = self.config.s_thresh_doubles
        self.semi_internal_projector = self.config.semi_internal_projector
        
        # Other settings
        self.analyze_spec_factor = self.config.analyze_spec_factor
        self.spec_factor_print_tol = self.config.spec_factor_print_tol
        self.outcore_expensive_tensors = self.config.outcore_expensive_tensors
        self.approx_trans_moments = self.config.approx_trans_moments

        self.use_mradc1_guess = self.config.use_mradc1_guess
   
    def _initialize_matrices(self) -> None:
        """Initialize matrices and attributes used in MR-ADC calculations."""
        
        # CASCI states
        self.e_cas_ci = None            # Active-space energies of CASCI states
        self.wfn_casci = None           # Active-space wavefunctions of CASCI states
        self.nelecasci = None           # Active-space number of electrons of CASCI states
        
        # Excitation manifolds
        self.h0 = lambda:None           # Information about h0 excitation manifold 
        self.h1 = lambda:None           # Information about h1 excitation manifold
        self.h_orth = lambda:None       # Information about orthonormalized excitation manifold
        self.S12 = lambda:None          # Matrices for orthogonalization of excitation spaces
        
        # Integrals and amplitudes
        self.h1eff = lambda:None        # Effective one-electron integrals
        self.v2e = lambda:None          # Two-electron integrals
        self.rdm = lambda:None          # Reduced density matrices
        self.t1 = lambda:None           # Amplitudes for singles excitations
        self.t2 = lambda:None           # Amplitudes for doubles excitations
        self.dip_mom = None             # Dipole moment
        
        self.mo_energy = lambda:None    # Diagonal elements of generalized Fock operator 
        
        # Matrix blocks
        self.M_00 = None
        self.M_01 = lambda:None
        
        # Temporary file placeholder
        self.tmpfile = lambda:None
    
    def _validate_method_parameters(self) -> None:
        """Validate method and method type parameters."""

        method = self.method.lower()
        method_type = self.method_type.lower()
        
        if method not in self.SUPPORTED_METHODS:
            raise MRADCError(f"Unknown method: {method}")
        
        if method_type not in self.SUPPORTED_METHOD_TYPES:
            raise MRADCError(f"Unknown method type: {method_type}")
        
        if self.interface.with_df and method_type not in self.DF_COMPATIBLE_TYPES:
            raise MRADCError(f"Density-fitting only compatible with {self.DF_COMPATIBLE_TYPES} method types")
        
        if method_type not in self.IMPLEMENTED_TYPES:
            raise MRADCError(f"Spin-adapted version does not support method type: {method_type}")

        ### TODO: Temporary check for new MR-ADC(2)-SX method
        if method == 'mr-adc(2)-sx' and  method_type != 'cvs-ee':
            raise MRADCError(f"Semi-internal extended method currently only compatible with CVS-EE method type.")
 
    def _validate_cvs_parameters(self) -> None:
        """Validate CVS parameters."""

        method_type = self.method_type.lower()
        
        if method_type not in self.CVS_TYPES:
            return
        
        if self.ncvs is None:
            raise MRADCError(f"Method type {method_type} requires setting ncvs parameter")
        
        if not isinstance(self.ncvs, int):
            raise MRADCError(f"ncvs parameter must be a positive integer")
        
        if self.ncvs < 1 or self.ncvs > self.ncore:
            raise MRADCError(f"ncvs must be integer between 1 and ncore (ncore = {self.ncore},  got ncvs = {self.ncvs})")
        
        self.nval = self.ncore - self.ncvs
        self.ncasci = 0
    
    def _transform_integrals(self) -> None:
        """Transform integrals to molecular orbital basis."""

        self.log.info("\nTransforming integrals to MO basis...")
        
        # Transform one-electron integrals
        mr_adc_integrals.transform_integrals_1e(self)
        
        # Transform two-electron integrals
        if self.interface.with_df:
            mr_adc_integrals.transform_Heff_integrals_2e_df(self)
            mr_adc_integrals.transform_integrals_2e_df(self)
        else:
            mr_adc_integrals.transform_integrals_2e_incore(self)
    
    def _compute_reference_data(self) -> None:
        """Compute CASCI energies and reduced density matrices."""

        mr_adc_rdms.compute_reference_rdms(self)

        ## TODO: Compute CASCI wavefunctions for excited states in the active space
        # mr_adc_rdms.compute_es_rdms(self)
    
    def kernel(self) -> Tuple[List, List, List]:
        """Main kernel for MR-ADC calculations."""

        self.log.info("Starting MR-ADC calculation...")
        
        # Validate parameters (uses current instance attributes)
        self._validate_method_parameters()
        self._validate_cvs_parameters()
        
        # Perform calculation steps
        self._transform_integrals()
        self._compute_reference_data()
        
        # Run MR-ADC computation
        self.log.info("Running MR-ADC computation...")
        excitation_energies, spec_factors, eigenvectors = mr_adc_compute.kernel(self)
        
        self.log.info("MR-ADC calculation completed successfully")
        return excitation_energies, spec_factors, eigenvectors
    
    @property
    def verbose(self) -> int:
        """Get verbosity level."""
        return self._verbose
    
    @verbose.setter
    def verbose(self, value: int) -> None:
        """Set verbosity level."""
        self._verbose = value
        self.log.verbose = value
