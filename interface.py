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
#

from typing import Optional, List, Tuple, Union, Any
from dataclasses import dataclass, field

import os
import tempfile
import numpy as np

import prism.lib.logger as logger

@dataclass
class BasicProperties:
    """Basic properties of the PySCF interface."""

    type: str = "pyscf"
    mol: Any = None
    nelec: int = 0
    enuc: float = 0.0
    e_scf: float = 0.0
    symmetry: bool = False
    group_repr_symm: Optional[List[str]] = None


@dataclass
class ReferenceWavefunction:
    """Reference wavefunction information."""

    reference: str = ""
    mo: Optional[np.ndarray] = None
    mo_scf: Optional[np.ndarray] = None
    mo_energy: Optional[np.ndarray] = None
    nmo: int = 0
    ncore: int = 0
    ncas: int = 0
    nextern: int = 0
    e_ref: List[float] = field(default_factory=list)
    e_ref_cas: List[float] = field(default_factory=list)
    ref_wfn: Optional[List] = None
    ref_nelecas: List[Tuple[int, int]] = field(default_factory=list)
    ref_wfn_spin_mult: List[int] = field(default_factory=list)
    ref_wfn_deg: List[int] = field(default_factory=list)
    weights: Optional[List[float]] = None
    ovlp: Optional[np.ndarray] = None


@dataclass
class CalculationSettings:
    """Settings for the PySCF calculation."""

    max_memory: int = 0
    verbose: int = 0
    davidson_only: bool = False
    pspace_size: int = 0
    enforce_degeneracy: bool = True
    opt_einsum: bool = False
    einsum_type: str = "greedy"
    temp_dir: str = ""


@dataclass
class Integrals:
    """One and two-electron integrals."""

    h1e_ao: Optional[np.ndarray] = None
    v2e_ao: Optional[Union[np.ndarray, Any]] = None
    dip_mom_ao: Optional[np.ndarray] = None
    reference_df: Optional[Any] = None
    with_df: Optional[Any] = None
    naux: Optional[int] = None


@dataclass
class SpinOperatorResult:
    """Result from spin operator application."""
    wfn: Optional[np.ndarray]
    nelec: Optional[Tuple[int, int]]

class PYSCF:
    """PySCF interface for Prism."""

    # Unit conversions
    HARTREE_TO_EV = 27.2113862459817
    HARTREE_TO_INV_CM = 219474.63136314
    HARTREE_TO_J = 4.359744722206e-18

    def convert_energies(self, value, unit: str):
        """Convert Hartree"""
        unit = unit.lower()
        if unit == "ev":
            return value * self.HARTREE_TO_EV
        elif unit == "inv_cm":
            return value * self.HARTREE_TO_INV_CM
        elif unit == "nm":
            return 1e7 / (value * self.HARTREE_TO_INV_CM)
        elif unit == "j":
            return value * self.HARTREE_TO_J
        else:
            raise ValueError(f"Unsupported unit: {unit}. Supported units are 'eV', 'inv_cm', 'nm', and 'J'.")
    
    def __init__(self, mf, mc: Optional[Any] = None, opt_einsum: bool = False, select_reference: Optional[List[int]] = None):

        # Initialize dataclass containers
        self.system =  BasicProperties()
        self.reference = ReferenceWavefunction()
        self.settings = CalculationSettings()
        self.integrals = Integrals()
        
        # Store original objects
        self.mf = mf
        self.mc = mc

        # Setup the PySCF interface
        self._setup_logging(mf, mc)
        self._setup_basic_properties(mf)
        self._setup_reference_wavefunction(mf, mc, select_reference)
        self._setup_integrals_and_memory(mf)
        self._setup_einsum(opt_einsum)
        
    def _setup_logging(self, mf, mc: Optional[Any]) -> None:
        """Setup logging and output stream."""

        if mc is None:
            self.stdout = mf.stdout
            self.verbose = mf.verbose
        else:
            self.stdout = mf.stdout
            self.verbose = mc.verbose
            
        self.log = logger.Logger(self.stdout, self.verbose)
        self.log.prism_header()

        self.log.info("Importing Pyscf objects...\n")
        
    def _setup_basic_properties(self, mf) -> None:
        """Setup basic properties of the PySCF interface."""

        from pyscf import lib
        
        self.system.type = "pyscf"
        self.system.mol = mf.mol
        self.system.nelec = mf.mol.nelectron
        self.system.enuc = mf.mol.energy_nuc()
        self.system.e_scf = mf.e_tot
        self.system.symmetry = mf.mol.symmetry
        
        self.current_memory = lib.current_memory
        
    def _setup_reference_wavefunction(self, mf, mc: Optional[Any], select_reference: Optional[List[int]]) -> None:
        """Collect reference wavefunction information."""

        self.log.info("Collecting reference wavefunction information...")
        
        if mc is None:
            self._setup_scf_reference(mf)
        else:
            self._setup_mc_reference(mf, mc, select_reference)
            
    def _setup_scf_reference(self, mf) -> None:
        """Setup reference wavefunction for SCF calculations."""

        self.reference.reference = "scf"
        self.log.info(f"Reference wavefunction: {self.reference.reference}")
        
        self.settings.max_memory = mf.max_memory
        self.reference.mo = mf.mo_coeff.copy()
        self.reference.nmo = self.reference.mo.shape[1]
        self.reference.mo_energy = mf.mo_energy.copy()
        self.reference.e_ref = [mf.e_tot]
        
        self.integrals.reference_df = getattr(mf, 'with_df', None)
        self._setup_symmetry_info(mf)
        
    def _setup_mc_reference(self, mf, mc, select_reference: Optional[List[int]]) -> None:
        """Setup reference wavefunction for multi-configurational calculations."""

        from pyscf import ao2mo, lib
        
        # Determine reference type
        reference_type, e_ref, e_cas, ci = self._determine_reference_type(mc)
        self.reference.reference = reference_type
        self.log.info(f"Reference wavefunction: {self.reference.reference}")
        
        # Select reference states
        if select_reference is not None:
            e_ref, e_cas, ci = self._select_reference_states(e_ref, e_cas, ci, select_reference)
            
        # Setup MC properties
        self._setup_mc_properties(mf, mc, e_ref, e_cas)
        
        # Canonicalize orbitals
        self._canonicalize_orbitals(mc, ci)
        
        # Print and store reference information
        self.reference.ref_wfn_spin_mult = self.print_reference_info(ci, self.reference.ncas, mc.nelecas, e_ref, e_cas)
        self.reference.ref_wfn = ci
        self.reference.ref_nelecas = len(ci) * [mc.nelecas]
        self.reference.ref_wfn_deg = len(ci) * [1]
        
        # Setup additional MC properties
        self._setup_symmetry_info(mc)
        self.transform_2e_chem_incore = ao2mo.general
        self.transform_2e_pair_chem_incore = ao2mo._ao2mo.nr_e2
        self.davidson = lib.linalg_helper.davidson1
        self.select_casci = None
        
    def _determine_reference_type(self, mc) -> Tuple[str, List[float], List[float], List]:
        """Determine the type of reference wavefunction and collect energies."""

        from pyscf.mcscf.casci import CASCI
        
        e_ref = mc.e_tot.copy()
        e_cas = mc.e_cas.copy()
        ci = mc.ci.copy()
        
        if isinstance(mc, CASCI):
            reference_type = "ms-casci" if isinstance(ci, list) else "casci"
        else:
            if hasattr(mc, 'weights') and isinstance(mc.ci, list):
                self.weights = mc.weights
                reference_type = "sa-casscf"
            else:
                reference_type = "casscf"
                
        # Store energies
        if reference_type == "sa-casscf":
            e_fzc = e_ref - e_cas
            e_ref = mc.e_states
            e_cas = [e - e_fzc for e in e_ref]
        else:
            e_ref = [e_ref]
            e_cas = [e_cas]
            ci = [ci]
            
        return reference_type, e_ref, e_cas, ci
    
    def _select_reference_states(self, e_ref: List[float], e_cas: List[float], ci: List, select_reference: List[int]) -> Tuple[List, List, List]:
        """Select specific reference states based on user input."""

        selected_e_ref = [e_ref[state - 1] for state in select_reference]
        selected_e_cas = [e_cas[state - 1] for state in select_reference]
        selected_ci = [ci[state - 1] for state in select_reference]
        
        self.log.info(f"\nReference states selected: {str(select_reference)}")
        return selected_e_ref, selected_e_cas, selected_ci
    
    def _setup_mc_properties(self, mf, mc, e_ref: List[float], e_cas: List[float]) -> None:
        """Setup properties specific to multi-configurational calculations."""

        self.settings.max_memory = mc.max_memory
        self.reference.mo = mc.mo_coeff.copy()
        self.reference.mo_scf = mf.mo_coeff.copy()
        self.reference.ovlp = mf.get_ovlp(mf.mol)
        self.reference.nmo = self.reference.mo.shape[1]
        self.reference.ncore = mc.ncore
        self.reference.ncas = mc.ncas
        self.reference.nextern = self.reference.nmo - self.reference.ncore - self.reference.ncas
        self.reference.e_ref = e_ref
        self.reference.e_ref_cas = e_cas
        self.settings.davidson_only = mc.fcisolver.davidson_only
        self.settings.pspace_size = mc.fcisolver.pspace_size
        self.settings.enforce_degeneracy = True
        self.integrals.reference_df = getattr(mc, 'with_df', None)
        
    def _canonicalize_orbitals(self, mc, ci: List) -> None:
        """Canonicalize molecular orbitals using the reference wavefunction manifold."""

        from pyscf.mcscf.casci import CASCI
        
        # Compute state-averaged 1-RDM
        ref_rdm1 = np.zeros((mc.ncas, mc.ncas))
        if self.reference.reference == "sa-casscf":
            mc_casci = CASCI(self.mf, mc.ncas, mc.nelecas)
            for ci_state in ci:
                ref_rdm1 += mc_casci.fcisolver.make_rdm1(ci_state, mc.ncas, mc.nelecas)
        else:
            for ci_state in ci:
                ref_rdm1 += mc.fcisolver.make_rdm1(ci_state, mc.ncas, mc.nelecas)
        ref_rdm1 /= len(ci)
        
        self.log.info("Canonicalizing molecular orbitals using reference wavefunction manifold...\n")
        mo, ci, mo_energy = mc.canonicalize(casdm1=ref_rdm1, ci=ci, cas_natorb=False)
        
        self.reference.mo = mo.copy()
        self.reference.mo_energy = mo_energy.copy()
        
    def _setup_symmetry_info(self, obj) -> None:
        """Setup symmetry information for the molecular system."""

        self.system.symmetry = obj.mol.symmetry
        if self.system.symmetry:
            from pyscf import symm
            scf_obj = getattr(obj, '_scf', obj)
            if hasattr(scf_obj.mo_coeff, 'orbsym'):
                self.system.group_repr_symm = [symm.irrep_id2name(obj.mol.groupname, x) for x in scf_obj.mo_coeff.orbsym]
            else:
                self.system.group_repr_symm = None
        else:
            self.system.group_repr_symm = None

    def _setup_integrals_and_memory(self, mf) -> None:
        """Setup integrals and memory directories for the PySCF interface."""

        # Memory and temporary directories
        if os.environ.get('PYSCF_TMPDIR'):
            self.settings.temp_dir = os.environ.get('PYSCF_TMPDIR', tempfile.gettempdir())
        else:
            self.settings.temp_dir = os.environ.get('TMPDIR', tempfile.gettempdir())
            
        # Integrals
        self.integrals.h1e_ao = mf.get_hcore()
        if isinstance(mf._eri, np.ndarray):
            self.integrals.v2e_ao = mf._eri
        else:
            self.integrals.v2e_ao = mf.mol
            
        self.integrals.dip_mom_ao = mf.mol.intor_symmetric("int1e_r", comp=3)
        
    def _setup_einsum(self, opt_einsum: bool) -> None:
        """Setup einsum functionality for tensor contractions."""

        self.settings.opt_einsum = opt_einsum
        self.settings.einsum_type = "greedy"

        if opt_einsum:
            from opt_einsum import contract
            self.einsum = contract
        else:
            self.einsum = np.einsum

        self.einsum_type = self.settings.einsum_type
        
    @property
    def with_df(self):
        return self.integrals.with_df
        
    @with_df.setter
    def with_df(self, obj):
        self.integrals.with_df = obj
        if obj:
            self.get_naux = obj.get_naoaux
            
    def density_fit(self, auxbasis: Optional[str] = None, with_df: Optional[Any] = None) -> 'PYSCF':
        """Setup density-fitting objects for the PySCF interface."""
        
        if with_df is None:
            self.log.info("Importing Pyscf density-fitting objects...")
            from pyscf import df
            self.integrals.with_df = df.DF(self.system.mol, auxbasis)
            self.get_naux = self.integrals.with_df.get_naoaux
        else:
            self.integrals.with_df = with_df
        return self
           
    #========== SPIN INFORMATION METHODS ==========#
    def _compute_spin_manifold(self, mc_ci: List, ncas: int, nelecas: Tuple[int, int]) -> List[Tuple[float, float, int]]:
        """Compute spin information for the reference wavefunction manifold."""
        
        spin_manifold = []
        for ci_state in mc_ci:
            s2 = abs(self.compute_spin_square(ci_state, ncas, nelecas))
            s = round((-1 + np.sqrt(1 + 4 * s2)) / 2)
            mult = int(round(2 * s + 1))
            spin_manifold.append((s2, s, mult))
        return spin_manifold
        
    def compute_spin_square(self, wfn, ncas: int, nelecas: Tuple[int, int]) -> float:
        """Compute the square of the total spin operator S^2."""
        
        from pyscf import fci
        return fci.spin_op.spin_square(wfn, ncas, nelecas)[0]
    
    ##TODO: Make this method more general for SA-CASSCF and MS-CASCI references (microstate, degeneracy handling)
    def print_reference_info(self, mc_ci: List, ncas: int, nelecas: Tuple[int, int], e_ref: List[float], e_cas: List[float]) -> List[int]:
        """Print summary of reference wavefunction manifold."""
        
        spin_manifold = self._compute_spin_manifold(mc_ci, ncas, nelecas)
        
        self.log.info("Summary of reference wavefunction manifold:")
        self.log.info(f"Total number of reference states: {len(mc_ci)}")
        self.log.info("  State       S^2        S      (2S+1)        E(total)              E(CAS)")
        self.log.info("-----------------------------------------------------------------------------------")
        
        for i, (s2, s, mult) in enumerate(spin_manifold):
            self.log.info(f"{i + 1:5d}       {s2:2.5f}     {s:3.1f}       {mult:2d}  {e_ref[i]:20.12f} {e_cas[i]:20.12f}")
            
        self.log.info("-----------------------------------------------------------------------------------\n")
        return [info[2] for info in spin_manifold]  # Return multiplicities
    
    #========== SPIN OPERATOR METHODS ==========#
    def apply_S_plus(self, psi, ncas: int, nelecas: Tuple[int, int]) -> SpinOperatorResult:
        """Apply spin raising operator S+."""

        sp_psi_components = []
        sp_ne = None
        
        for y in range(ncas):
            a_result = self.act_des_b(psi, ncas, nelecas, y)
            if a_result.wfn is not None:
                ca_result = self.act_cre_a(a_result.wfn, ncas, a_result.nelec, y)
                if ca_result.wfn is not None:
                    sp_psi_components.append(ca_result.wfn)
                    sp_ne = ca_result.nelec
                    
        result_wfn = sum(sp_psi_components) if sp_psi_components else None
        
#        # Fix the phase for the CI coefficients
#        if result_wfn is not None:
#            i, j = np.unravel_index(np.argmax(result_wfn).argmax(), result_wfn.shape)
#            if result_wfn[i, j] < 0.0:
#                result_wfn *= -1.0

        return SpinOperatorResult(result_wfn, sp_ne)
        
    def apply_S_minus(self, psi, ncas: int, nelecas: Tuple[int, int]) -> SpinOperatorResult:
        """Apply spin lowering operator S-."""

        sp_psi_components = []
        sp_ne = None
        
        for y in range(ncas):
            a_result = self.act_des_a(psi, ncas, nelecas, y)
            if a_result.wfn is not None:
                ca_result = self.act_cre_b(a_result.wfn, ncas, a_result.nelec, y)
                if ca_result.wfn is not None:
                    sp_psi_components.append(ca_result.wfn)
                    sp_ne = ca_result.nelec
                    
        result_wfn = sum(sp_psi_components) if sp_psi_components else None

#        # Fix the phase for the CI coefficients
#        if result_wfn is not None:
#            i, j = np.unravel_index(np.argmax(result_wfn).argmax(), result_wfn.shape)
#            if result_wfn[i, j] < 0.0:
#                result_wfn *= -1.0

        return SpinOperatorResult(result_wfn, sp_ne)
        
    def apply_S_z(self, psi, ncas: int, nelecas: Tuple[int, int]) -> Optional[np.ndarray]:
        """Apply z-projection of spin operator Sz."""

        sz_components = []
        for y in range(ncas):

            # Alpha contribution
            a_result = self.act_des_a(psi, ncas, nelecas, y)
            if a_result.wfn is not None:
                a_result2 = self.act_cre_a(a_result.wfn, ncas, a_result.nelec, y)
                alpha_contrib = a_result2.wfn
            else:
                alpha_contrib = None
            
            # Beta contribution
            b_result = self.act_des_b(psi, ncas, nelecas, y)
            if b_result.wfn is not None:
                b_result2 = self.act_cre_b(b_result.wfn, ncas, b_result.nelec, y)
                beta_contrib = b_result2.wfn
            else:
                beta_contrib = None
            
            # Combine contributions
            if alpha_contrib is None and beta_contrib is not None:
                sz_components.append(0.5 * (-beta_contrib))
            elif alpha_contrib is not None and beta_contrib is None:
                sz_components.append(0.5 * alpha_contrib)
            elif alpha_contrib is not None and beta_contrib is not None:
                sz_components.append(0.5 * (alpha_contrib - beta_contrib))
                
        return sum(sz_components) if sz_components else None
        
    #========== CREATION & ANNIHILATION OPERATORS ==========#
    def act_cre_a(self, wfn, ncas: int, nelec: Tuple[int, int], orb: int) -> SpinOperatorResult:
        """Apply alpha spin creation operator."""

        from pyscf import fci
        
        if wfn is not None and (ncas - nelec[0] > 0):
            c_wfn = fci.addons.cre_a(wfn, ncas, nelec, orb)
            c_wfn_ne = (nelec[0] + 1, nelec[1])
            return SpinOperatorResult(c_wfn, c_wfn_ne)
        
        return SpinOperatorResult(None, None)
        
    def act_cre_b(self, wfn, ncas: int, nelec: Tuple[int, int], orb: int) -> SpinOperatorResult:
        """Apply beta spin creation operator."""
        
        from pyscf import fci
        
        if wfn is not None and (ncas - nelec[1] > 0):
            c_wfn = fci.addons.cre_b(wfn, ncas, nelec, orb)
            c_wfn_ne = (nelec[0] + 1, nelec[1])
            return SpinOperatorResult(c_wfn, c_wfn_ne)
        
        return SpinOperatorResult(None, None)
        
    def act_des_a(self, wfn, ncas: int, nelec: Tuple[int, int], orb: int) -> SpinOperatorResult:
        """Apply alpha spin annihilation operator."""
        
        from pyscf import fci
        
        if wfn is not None and nelec[0] > 0:
            a_wfn = fci.addons.des_a(wfn, ncas, nelec, orb)
            a_wfn_ne = (nelec[0] - 1, nelec[1])
            return SpinOperatorResult(a_wfn, a_wfn_ne)
        
        return SpinOperatorResult(None, None)
        
    def act_des_b(self, wfn, ncas: int, nelec: Tuple[int, int], orb: int) -> SpinOperatorResult:
        """Apply beta spin annihilation operator."""
        
        from pyscf import fci
        
        if wfn is not None and nelec[1] > 0:
            a_wfn = fci.addons.des_b(wfn, ncas, nelec, orb)
            a_wfn_ne = (nelec[0], nelec[1] - 1)
            return SpinOperatorResult(a_wfn, a_wfn_ne)
        
        return SpinOperatorResult(None, None)
        
    #========== REDUCED DENSITY MATRIX METHODS ==========#
    def compute_rdm123(self, bra, ket, nelecas: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spin traced 1, 2 and 3-particle reduced density matrices."""

        from pyscf import fci
        
        rdm1, rdm2, rdm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', bra, ket, self.ncas, nelecas)
        rdm1, rdm2, rdm3 = fci.rdm.reorder_dm123(rdm1, rdm2, rdm3)
        
        # rdm2[p,q,r,s] = \langle p^\dagger q^\dagger s r\rangle
        rdm2 = np.ascontiguousarray(rdm2.transpose(0, 2, 1, 3))
        # rdm3[p,q,r,s,t,u] = \langle p^\dagger q^\dagger r^\dagger u t s\rangle
        rdm3 = np.ascontiguousarray(rdm3.transpose(0, 2, 4, 1, 3, 5))
        
        return rdm1, rdm2, rdm3
        
    def compute_rdm1234(self, bra, ket, nelecas: Union[Tuple[int, int], List[Tuple[int, int]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute spin traced 1, 2, 3 and 4-particle reduced density matrices."""

        from pyscf import fci
        
        if isinstance(nelecas, list):
            # Compute averaged RDMs
            rdm1, rdm2, rdm3, rdm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', bra[0], ket[0], self.ncas, nelecas[0])
            for i in range(1, len(nelecas)):
                rdm1_i, rdm2_i, rdm3_i, rdm4_i = fci.rdm.make_dm1234('FCI4pdm_kern_sf', bra[i], ket[i], self.ncas, nelecas[i])
                rdm1 += rdm1_i
                rdm2 += rdm2_i
                rdm3 += rdm3_i
                rdm4 += rdm4_i
            rdm1 /= len(nelecas)
            rdm2 /= len(nelecas)
            rdm3 /= len(nelecas)
            rdm4 /= len(nelecas)
        else:
            rdm1, rdm2, rdm3, rdm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', bra, ket, self.ncas, nelecas)
            
        rdm1, rdm2, rdm3, rdm4 = fci.rdm.reorder_dm1234(rdm1, rdm2, rdm3, rdm4)
        
        # rdm2[p,q,r,s] = \langle p^\dagger q^\dagger s r\rangle
        rdm2 = np.ascontiguousarray(rdm2.transpose(0, 2, 1, 3))
        # rdm3[p,q,r,s,t,u] = \langle p^\dagger q^\dagger r^\dagger u t s\rangle
        rdm3 = np.ascontiguousarray(rdm3.transpose(0, 2, 4, 1, 3, 5))
        # rdm4[p,q,r,s,t,u,v,w] = \langle p^\dagger q^\dagger r^\dagger w v u t\rangle
        rdm4 = np.ascontiguousarray(rdm4.transpose(0, 2, 4, 6, 1, 3, 5, 7))
        
        return rdm1, rdm2, rdm3, rdm4

class InterfaceMixin:
    """Mixin to provide backward compatibility for accessing attributes."""

    # Map legacy attributes to dataclass fields
    _legacy_map = {
        # attribute: (dataclass_name, field_name)
        "max_memory": ("settings", "max_memory"),
        "temp_dir": ("settings", "temp_dir"),
        "reference_type": ("reference", "reference"),
        "mo": ("reference", "mo"),
        "mo_scf": ("reference", "mo_scf"),
        "ovlp": ("reference", "ovlp"),
        "nmo": ("reference", "nmo"),
        "nelec": ("system", "nelec"),
        "enuc": ("system", "enuc"),
        "e_scf": ("system", "e_scf"),
        "symmetry": ("system", "symmetry"),
        "group_repr_symm": ("system", "group_repr_symm"),
        "ncore": ("reference", "ncore"),
        "ncas": ("reference", "ncas"),
        "nextern": ("reference", "nextern"),
        "e_ref": ("reference", "e_ref"),
        "e_ref_cas": ("reference", "e_ref_cas"),
        "ref_nelecas": ("reference", "ref_nelecas"),
        "ref_wfn": ("reference", "ref_wfn"),
        "ref_wfn_spin_mult": ("reference", "ref_wfn_spin_mult"),
        "ref_wfn_deg": ("reference", "ref_wfn_deg"),
        "mo_energy": ("reference", "mo_energy"),
        "h1e_ao": ("integrals", "h1e_ao"),
        "v2e_ao": ("integrals", "v2e_ao"),
        "dip_mom_ao": ("integrals", "dip_mom_ao"),
        "reference_df": ("integrals", "reference_df"),
    }

    def __getattr__(self, name):
        if name in self._legacy_map:
            obj, field = self._legacy_map[name]
            return getattr(getattr(self, obj), field)
        raise AttributeError(f"{type(self).__name__} object has no attribute {name}")

# Add backward compatibility to the main class
class PYSCF(PYSCF, InterfaceMixin):
    pass
