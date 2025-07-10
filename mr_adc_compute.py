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

from typing import Tuple, List, Callable
import numpy as np

import prism.mr_adc_amplitudes as mr_adc_amplitudes
import prism.mr_adc_integrals as mr_adc_integrals
#import prism.mr_adc_testing_block as testing_block

## compute_guess_vector -> use MR-ADC(1) eigvec for guess vectors

import prism.lib.logger as logger

def kernel(mr_adc) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Main computation kernel for MR-ADC calculations."""

    cput0 = (logger.process_clock(), logger.perf_counter())
    
    # Setup and logging
    _log_calculation_info(mr_adc)
    
    # Core computation pipeline
    _compute_amplitudes(mr_adc)
    _compute_cvs_integrals(mr_adc)
    _compute_excitation_manifolds(mr_adc)
    
    # Solve eigenvalue problem
    conv, E, U = _solve_eigenvalue_problem(mr_adc)
    
    # Compute spectroscopic properties
    spec_intensity, X = _compute_spectroscopic_properties(mr_adc, E, U)
    
    # Results and cleanup 
    _log_results_summary(mr_adc, E, spec_intensity, conv)
    mr_adc.log.timer0(f"total {mr_adc.method_type.upper()}-{mr_adc.method.upper()} calculation", *cput0)
    
    HARTREE_TO_EV = mr_adc.interface.HARTREE_TO_EV
    return E * HARTREE_TO_EV, spec_intensity, X


def _log_calculation_info(mr_adc):
    """Print comprehensive calculation setup information."""

    log = mr_adc.log
    log.info("\nComputing MR-ADC excitation energies...\n")
    
    # General parameters
    log.info(f"Method:                                            {mr_adc.method_type}-{mr_adc.method}")
    log.info(f"Nuclear repulsion energy:                    {mr_adc.enuc:20.12f}")
    log.info(f"Number of electrons:                               {mr_adc.nelec}")
    log.info(f"Number of basis functions:                         {mr_adc.nmo}")
    log.info(f"Reference wavefunction type:                       {mr_adc.interface.reference_type}")
    log.info(f"Number of core orbitals:                           {mr_adc.ncore}")
    log.info(f"Number of active orbitals:                         {mr_adc.ncas}")
    log.info(f"Number of external orbitals:                       {mr_adc.nextern}")
    log.info(f"Number of active electrons:                        {str(mr_adc.ref_nelecas)}")
    log.info(f"Reference state active-space energy:         {mr_adc.e_ref_cas[0]:20.12f}")
    log.info(f"Reference state spin multiplicity:                 {str(mr_adc.ref_wfn_spin_mult)}")
    log.info(f"Number of MR-ADC roots requested:                  {mr_adc.nroots}")
    if mr_adc.ncvs is not None:
        log.info(f"Number of CVS orbitals:                            {mr_adc.ncvs}")
        log.info(f"Number of valence (non-CVS) orbitals:              {mr_adc.nval}")
    
    # Other parameters
    log.info(f"Reference density fitting?                         {bool(mr_adc.interface.reference_df)}")
    log.info(f"Correlation density fitting?                       {bool(mr_adc.interface.with_df)}")
    log.info(f"Temporary directory path:                          {mr_adc.temp_dir}")
    
    log.info(f"\nInternal contraction:                              Full")
    log.info(f"Overlap truncation parameter (singles):            {mr_adc.s_thresh_singles:e}")
    log.info(f"Overlap truncation parameter (doubles):            {mr_adc.s_thresh_doubles:e}")
    log.info(f"Projector for the semi-internal amplitudes:        {mr_adc.semi_internal_projector}")
    
    # CASCI states information
    if mr_adc.ncasci > 0:
        log.info(f"Number of CASCI states:                            {mr_adc.ncasci}")
    
    if mr_adc.e_cas_ci is not None:
        HARTREE_TO_EV = mr_adc.interface.HARTREE_TO_EV
        excitation_energies = HARTREE_TO_EV * (mr_adc.e_cas_ci - mr_adc.e_cas)
        log.extra(f"CASCI excitation energies (eV):                    {str(excitation_energies)}")


def _compute_amplitudes(mr_adc):
    """Compute amplitudes."""
    mr_adc_amplitudes.compute_amplitudes(mr_adc)

def _compute_cvs_integrals(mr_adc):
    """Compute CVS integrals."""
    if mr_adc.method_type in mr_adc.CVS_TYPES:
        if mr_adc.interface.with_df:
            mr_adc_integrals.compute_cvs_integrals_2e_df(mr_adc)
        else:
            mr_adc_integrals.compute_cvs_integrals_2e_incore(mr_adc)


def _compute_excitation_manifolds(mr_adc):
    """Define function for the matrix-vector product S^(-1/2) M S^(-1/2) vec"""
    method_type = mr_adc.method_type
    mr_adc = method_registry.call(method_type, "compute_excitation_manifolds", mr_adc)


def _solve_eigenvalue_problem(mr_adc) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Solve eigenvalue problem using Davidson algorithm."""

    # Setup Davidson algorithm parameters
    apply_M, precond, x0 = _setup_davidson_solver(mr_adc)

    # Verbose logging
    davidson_verbose = 6 if mr_adc.verbose > 3 else 3
    
    # Using Davidson algorithm, solve the [S^(-1/2) M S^(-1/2) C = C E] eigenvalue problem
    cput1 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.info("")
        
    conv, E, U = mr_adc.interface.davidson(
        lambda xs: [apply_M(x) for x in xs], 
        x0, 
        precond,
        nroots=mr_adc.nroots,
        verbose=davidson_verbose,
        max_space=mr_adc.max_space,
        max_cycle=mr_adc.max_cycle,
        tol=mr_adc.tol_e,
        tol_residual=mr_adc.tol_r
    )
    
    mr_adc.log.timer("solving eigenvalue problem", *cput1)
    return conv, E, np.array(U)


def _setup_davidson_solver(mr_adc) -> Tuple[Callable, np.ndarray, List[np.ndarray]]:
    """Setup all Davidson algorithm components."""

    # Compute M matrix sectors
    _compute_effective_hamiltonian_blocks(mr_adc)
    
    # Compute diagonal of the M matrix (preconditioner)
    precond = _compute_preconditioner(mr_adc)
    
    # Compute initial guess vectors
    x0 = _compute_guess_vectors(precond, mr_adc.nroots)
 
    # Create matrix-vector product function (M * vec)
    apply_M = _create_matrix_vector_product(mr_adc)
    
    return apply_M, precond, x0

def _compute_effective_hamiltonian_blocks(mr_adc):
    """Compute effective Hamiltonian matrix blocks."""

    method_type = mr_adc.method_type
    method = mr_adc.method
    
    # Standard methods
    if method_type in ("ip", "ea", "ee"):
        module = method_registry.get_module(method_type)

        # Compute h0-h0 block of the effective Hamiltonian matrix
        module.compute_M_00(mr_adc)

        if method in ("mr-adc(2)", "mr-adc(2)-x"):
            # Compute h0-h1 block of the effective Hamiltonian matrix
            module.compute_M_01(mr_adc)

            # Compute h1-h1 block of the effective Hamiltonian matrix
            if method_type in ("ip", "ea") and hasattr(module, 'compute_M_11'):
                module.compute_M_11(mr_adc)

    # CVS methods
    elif method_type in mr_adc.CVS_TYPES:
        module = method_registry.get_module(method_type)

        # Compute h0-h0 block of the effective Hamiltonian matrix
        module.compute_M_00(mr_adc)

        # Compute parts of the h0-h1 block of the effective Hamiltonian matrix
        if method in ("mr-adc(2)", "mr-adc(2)-x") and method_type == 'cvs-ip':
            module.compute_M_01(mr_adc)

    else:
        raise ValueError(f"Unknown method type: {method_type}")

def _compute_preconditioner(mr_adc) -> np.ndarray:
    """Compute preconditioner for Davidson algorithm."""

    method_type = mr_adc.method_type
    if method_type in mr_adc.SUPPORTED_METHOD_TYPES:
        return method_registry.call(method_type, "compute_preconditioner", mr_adc)
    else:
        raise ValueError(f"Unknown method type: {method_type}")


def _create_matrix_vector_product(mr_adc) -> Callable:
    """Create matrix-vector product function for Davidson algorithm."""

    method_type = mr_adc.method_type
    if method_type in mr_adc.SUPPORTED_METHOD_TYPES:
        return method_registry.call(method_type, "define_effective_hamiltonian", mr_adc)
    else:
        raise ValueError(f"Unknown method type: {method_type}")

def _compute_guess_vectors(precond: np.ndarray, nroots: int, ascending: bool = True) -> List[np.ndarray]:
    """Compute initial guess vectors for Davidson algorithm."""

    sort_ind = np.argsort(precond)
    if not ascending:
        sort_ind = sort_ind[::-1]
    
    x0s = np.zeros((precond.shape[0], nroots))
    min_shape = min(precond.shape[0], nroots)
    x0s[:min_shape, :min_shape] = np.identity(min_shape)
    
    x0 = np.zeros_like(x0s)
    x0[sort_ind] = x0s.copy()
    
    # Return list of vectors
    return [x0[:, p] for p in range(x0.shape[1])]

def _compute_transition_moments(mr_adc, eigenvectors: np.ndarray) -> np.ndarray:
    """Compute transition moments."""
    method_type = mr_adc.method_type
    return method_registry.call(method_type, "compute_trans_moments", mr_adc, eigenvectors)

def _compute_spectroscopic_properties(mr_adc, energies: np.ndarray, eigenvectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute transition moments and spectroscopic intensities."""

    # Compute transition moments
    X = _compute_transition_moments(mr_adc, eigenvectors)

    # X is a tuple for CVS-EE
    if isinstance(X, tuple):
        X = X[1] 

    # Compute spectroscopic intensities
    spec_intensity = 2.0 * np.sum(X**2, axis=0)
    
    # Compute oscillator strengths for EE methods
    if mr_adc.method_type in ("cvs-ee", "ee"):
        osc_strength = (2.0/3.0) * energies * spec_intensity
        #analyze_eigenvector(mr_adc, U, E, osc_strength)
    
    # Analyze spectroscopic factors if requested
    #if mr_adc.analyze_spec_factor or (mr_adc.verbose > 4): 
    if mr_adc.analyze_spec_factor: 
        analyze_mo_overlap(mr_adc)
        analyze_spec_factor(mr_adc, X, spec_intensity)
    
    return spec_intensity, X

## TODO: Include Davidson convergence info
def _log_results_summary(mr_adc, energies: np.ndarray, intensities: np.ndarray, conv: np.ndarray):
    """Print formatted results summary table."""

    log = mr_adc.log
    HARTREE_TO_EV = mr_adc.interface.HARTREE_TO_EV
    HARTREE_TO_INV_CM = mr_adc.interface.HARTREE_TO_INV_CM

    log.info(f"\nSummary of results for the {mr_adc.method_type.upper()}-{mr_adc.method.upper()} calculation with the {mr_adc.interface.reference_type.upper()} reference:")
    
    log.info("-" * 96)
    log.info("  State        dE(a.u.)        dE(eV)      dE(nm)       dE(cm-1)       Intensity")
    log.info("-" * 96)
    
    energies_ev = energies * HARTREE_TO_EV
    energies_cm = energies * HARTREE_TO_INV_CM
    
    for i, (e_au, e_ev, e_cm, intensity) in enumerate(zip(energies, energies_ev, energies_cm, intensities), 1):
        wavelength_nm = 10000000 / e_cm
        log.info(f"{i:5d}     {e_au:14.8f}  {e_ev:12.4f} {wavelength_nm:10.4f}  {e_cm:14.4f}    {intensity:10.6f}")
    
    log.info("-" * 96)

    if all(conv):
        log.info('Convergence reached.')
    elif any(conv):
        not_converged = np.where(conv == False)[0]
        log.warn(f'States {not_converged+1} did not converge!')
    else:
        log.warn('No convergence reached for Davidson iterations!')

#========== METHOD REGISTRY FOR MR-ADC METHOD MODULES AND FUNCTIONS ==========#
import importlib

class MethodRegistry:
    """Registry for MR-ADC method modules and function dispatch."""

    _module_map = {
        "ip": "prism.mr_adc_ip",
        "ea": "prism.mr_adc_ea",
        "ee": "prism.mr_adc_ee",
        "cvs-ip": "prism.mr_adc_cvs_ip",
        "cvs-ee": "prism.mr_adc_cvs_ee",
    }

    def __init__(self):
        self._cache = {}

    def get_module(self, method_type):
        if method_type not in self._cache:
            module_name = self._module_map[method_type]
            self._cache[method_type] = importlib.import_module(module_name)
        return self._cache[method_type]

    def call(self, method_type, func_name, *args, **kwargs):
        module = self.get_module(method_type)
        func = getattr(module, func_name)
        return func(*args, **kwargs)

method_registry = MethodRegistry()

#========== UTILITY FUNCTIONS FOR SPECIAL ANALYSES ==========#
## TODO: Is this needed? Also, transform_2e_phys_incore is not defined anywhere
def dyall_hamiltonian(mr_adc) -> float:
    """Compute expected value of zeroth-order Dyall Hamiltonian."""
    
    mr_adc.log.info("Calculating the Spin-Adapted Dyall Hamiltonian...")
    
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type
    
    # Extract required quantities
    h_aa = mr_adc.h1eff.aa
    rdm_ca = mr_adc.rdm.ca
    v_aaaa = mr_adc.v2e.aaaa
    rdm_ccaa = mr_adc.rdm.ccaa
    mo_c = mr_adc.mo_energy.c
    
    # Compute frozen core energy
    h_cc = 2.0 * mr_adc.h1e[:mr_adc.ncore, :mr_adc.ncore].copy()
    v_cccc = mr_adc_integrals.transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_c)
    
    e_fc = einsum('ii', h_cc, optimize=True)
    e_fc += 2.0 * einsum('ijij', v_cccc, optimize=True)
    e_fc -= einsum('jiij', v_cccc, optimize=True)

    # Compute active space energy
    h_act = einsum('xy,xy', h_aa, rdm_ca, optimize=True)
    h_act += 0.5 * einsum('xyzw,xyzw', v_aaaa, rdm_ccaa, optimize=einsum_type)
    
    total_energy = h_act + e_fc + mr_adc.interface.enuc
    print(f"\n>>> SA Expected value of Zeroth-order Dyall Hamiltonian: {total_energy}")
    
    return total_energy


def analyze_spec_factor(mr_adc, X, spec_intensity):
    """Analyze and print spectroscopic factors for each state above a given threshold."""

    print_thresh = mr_adc.spec_factor_print_tol
    mr_adc.log.info(f"\nAnalyzing spectroscopic factors (threshold = {print_thresh:e}) ...")

    X_2 = np.empty_like(X.T)
    np.square(X.T, out=X_2, order='C')
    
    if mr_adc.method_type == 'cvs-ee':
        X_2 *= 2.0

    for i in range(X_2.shape[0]):

        sort = np.argsort(-X_2[i,:])
        X_2_row = X_2[i,:]
        X_2_row = X_2_row[sort]
        
        # If symmetry is not used or symmetry labels are unavailable, assign all orbitals to 'A'
        if not mr_adc.symmetry:
            group_repr_symm = np.repeat(['A'], X_2_row.shape[0])

        else:
            group_repr_symm = mr_adc.group_repr_symm
            group_repr_symm = np.array(group_repr_symm)[sort]

        spec_contribution = X_2_row[X_2_row > print_thresh]
        index_orb = sort[X_2_row > print_thresh] + 1

        if np.sum(spec_contribution) <= print_thresh:
            continue

        partial_Contribution = spec_contribution / spec_intensity[i]

        filtered_spec_contribution = spec_contribution[partial_Contribution > 1e-6]
        index_orb = index_orb[partial_Contribution > 1e-6]
        partial_Contribution = partial_Contribution[partial_Contribution > 1e-6]

        print(f"\n{mr_adc.method} | state {i+1} \n")
        print("  MO          Spec. Contribution       Partial Contribution")
        print("-------------------------------------------------------------")

        for c in range(index_orb.shape[0]):
            print(f" {index_orb[c]:>5d}   ({group_repr_symm[c]:<2})   {filtered_spec_contribution[c]:>20.8f}   {partial_Contribution[c]:>20.8f}")

def analyze_mo_overlap(mr_adc):
    """Analyze and print overlap of CASSCF and HF spatial MOs."""

    from functools import reduce
    print_thresh = 0.01
    mr_adc.log.info(f"\nAnalyzing overlap of CASSCF and HF spatial MO's (threshold = {print_thresh:e}) ...")

    cas_hf_ovlp = reduce(np.dot, (mr_adc.mo.T, mr_adc.ovlp, mr_adc.mo_scf))
    for p in range(mr_adc.mo.shape[1]):

        hf_ovlp = cas_hf_ovlp[p]**2
        hf_ovlp_ind = np.argsort(hf_ovlp)[::-1]
        hf_ovlp_sorted = hf_ovlp[hf_ovlp_ind]

        print (f"\nCASSCF MO #{p + 1}:")

        for hf_p in range(mr_adc.mo_scf.shape[1]):
            if (hf_ovlp_sorted[hf_p] > print_thresh):
                print (f"{hf_ovlp_sorted[hf_p]:.3f} HF MO #{hf_ovlp_ind[hf_p] + 1}")
