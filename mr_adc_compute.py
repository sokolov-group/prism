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
#                  Ilia M. Mazin <ilia.mazin@gmail.com>
#              Donna H. Odhiambo <donna.odhiambo@proton.me>
#

import numpy as np

import prism.mr_adc_amplitudes as mr_adc_amplitudes
import prism.mr_adc_integrals as mr_adc_integrals
import prism.mr_adc_cvs_ip as mr_adc_cvs_ip
import prism.mr_adc_cvs_ee as mr_adc_cvs_ee

import prism.lib.logger as logger

def kernel(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.info("\nComputing MR-ADC excitation energies...\n")

    h2ev = mr_adc.interface.hartree_to_ev
    h2cm = mr_adc.interface.hartree_to_inv_cm

    ref_df = False
    df = False
    if mr_adc.interface.reference_df:
        ref_df = True
    if mr_adc.interface.with_df:
        df = True

    # Print general information
    mr_adc.log.info("Method:                                            %s-%s" % (mr_adc.method_type, mr_adc.method))
    mr_adc.log.info("Nuclear repulsion energy:                    %20.12f" % mr_adc.enuc)
    mr_adc.log.info("Number of electrons:                               %d" % mr_adc.nelec)
    mr_adc.log.info("Number of basis functions:                         %d" % mr_adc.nmo)
    mr_adc.log.info("Reference wavefunction type:                       %s" % mr_adc.interface.reference)
    mr_adc.log.info("Number of core orbitals:                           %d" % mr_adc.ncore)
    mr_adc.log.info("Number of active orbitals:                         %d" % mr_adc.ncas)
    mr_adc.log.info("Number of external orbitals:                       %d" % mr_adc.nextern)
    mr_adc.log.info("Number of active electrons:                        %s" % str(mr_adc.ref_nelecas))
    mr_adc.log.info("Reference state active-space energy:         %20.12f" % mr_adc.e_ref_cas[0])
    mr_adc.log.info("Reference state spin multiplicity:                 %s" % str(mr_adc.ref_wfn_spin_mult))
    mr_adc.log.info("Number of MR-ADC roots requested:                  %d" % mr_adc.nroots)
    if mr_adc.ncvs is not None:
        mr_adc.log.info("Number of CVS orbitals:                            %d" % mr_adc.ncvs)
        mr_adc.log.info("Number of valence (non-CVS) orbitals:              %d" % (mr_adc.ncore - mr_adc.ncvs))

    mr_adc.log.info("Reference density fitting?                         %s" % ref_df)
    mr_adc.log.info("Correlation density fitting?                       %s" % df)
    mr_adc.log.info("Temporary directory path:                          %s" % mr_adc.temp_dir)

    mr_adc.log.info("\nInternal contraction:                              %s" % "Full")
    mr_adc.log.info("Overlap truncation parameter (singles):            %e" % mr_adc.s_thresh_singles)
    mr_adc.log.info("Overlap truncation parameter (doubles):            %e" % mr_adc.s_thresh_doubles)
    mr_adc.log.info("Projector for the semi-internal amplitudes:        %s" % mr_adc.semi_internal_projector)

    # Print info about CASCI states
    if mr_adc.ncasci > 0:
        mr_adc.log.info("Number of CASCI states:                            %d" % mr_adc.ncasci)

    if mr_adc.e_cas_ci is not None:
        mr_adc.log.extra("CASCI excitation energies (eV):                    %s" % str(h2ev*(mr_adc.e_cas_ci - mr_adc.e_cas)))
    
    # Compute amplitudes
    mr_adc_amplitudes.compute_amplitudes(mr_adc)

    # Compute CVS integrals
    if mr_adc.method_type in ("cvs-ip", "cvs-ee"):
        if mr_adc.interface.with_df:
            mr_adc_integrals.compute_cvs_integrals_2e_df(mr_adc)
        else:
            mr_adc_integrals.compute_cvs_integrals_2e_incore(mr_adc)

    # Define function for the matrix-vector product S^(-1/2) M S^(-1/2) vec
    if mr_adc.method_type == "ip":
        mr_adc = mr_adc_ip.compute_excitation_manifolds(mr_adc)

    elif mr_adc.method_type == "ea":
        mr_adc = mr_adc_ea.compute_excitation_manifolds(mr_adc)

    elif mr_adc.method_type == "ee":
        mr_adc = mr_adc_ee.compute_excitation_manifolds(mr_adc)

    elif mr_adc.method_type == "cvs-ip":
        mr_adc = mr_adc_cvs_ip.compute_excitation_manifolds(mr_adc)

    elif mr_adc.method_type == "cvs-ee":
        mr_adc = mr_adc_cvs_ee.compute_excitation_manifolds(mr_adc)

    # Setup Davidson algorithm parameters
    davidson_verbose = 6 if mr_adc.verbose > 3 else 3
    apply_M, precond, x0 = setup_davidson(mr_adc)

    # Using Davidson algorithm, solve the [S^(-1/2) M S^(-1/2) C = C E] eigenvalue problem
    cput1 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.info("")
    conv, de, U = mr_adc.interface.davidson(lambda xs: [apply_M(x) for x in xs], x0, precond,
                                           nroots = mr_adc.nroots,
                                           verbose = davidson_verbose,
                                           max_space = mr_adc.max_space,
                                           max_cycle = mr_adc.max_cycle,
                                           tol = mr_adc.tol_e,
                                           tol_residual = mr_adc.tol_r)
    mr_adc.log.timer("solving eigenvalue problem", *cput1)

    # Compute transition moments and spectroscopic factors
    U = np.array(U)
    spec_intensity, X = compute_trans_properties(mr_adc, de, U)

    # Conversions for energy
    de_ev = de * h2ev
    de_cm = de * h2cm

    mr_adc.log.info("\nSummary of results for the %s-%s calculation with the %s reference:" % (mr_adc.method_type.upper(), mr_adc.method.upper(), mr_adc.interface.reference.upper()))

    ## Oscillator Strength
    if mr_adc.method_type == "cvs-ee":
        mr_adc.log.info("-"*100)
        mr_adc.log.info("  State        dE(a.u.)        dE(eV)      dE(nm)       dE(cm-1)       Intensity       Osc Str(f)")
        mr_adc.log.info("-"*100)

        for p, (e_au, e_ev, e_cm, intensity) in enumerate(zip(de, de_ev, de_cm, spec_intensity), 1):
            e_nm = 10000000 / e_cm
            osc_str = (2.0/3.0) * e_au * intensity
            mr_adc.log.info(f"{p:5d}     {e_au:14.8f}  {e_ev:12.4f} {e_nm:10.4f}  {e_cm:14.4f}    {intensity:10.6f}      {osc_str:10.6f} ")

        mr_adc.log.info("-"*100)
    else: 
        mr_adc.log.info("-"*96)
        mr_adc.log.info("  State        dE(a.u.)        dE(eV)      dE(nm)       dE(cm-1)       Intensity")
        mr_adc.log.info("-"*96)

        for p, (e_au, e_ev, e_cm, intensity) in enumerate(zip(de, de_ev, de_cm, spec_intensity), 1):
            e_nm = 10000000 / e_cm
            mr_adc.log.info(f"{p:5d}     {e_au:14.8f}  {e_ev:12.4f} {e_nm:10.4f}  {e_cm:14.4f}    {intensity:10.6f}")

        mr_adc.log.info("-"*96)

    if all(conv):
        mr_adc.log.info('Full convergence reached.')
    elif any(conv):
        not_converged = np.where(conv == False)[0]
        mr_adc.log.warn(f'States {not_converged+1} did not converge!')
    else:
        mr_adc.log.warn('No convergence reached for Davidson iterations!')

    mr_adc.log.timer0("total %s-%s calculation" % (mr_adc.method_type.upper(), mr_adc.method.upper()), *cput0)

    return de_ev, spec_intensity, X

def setup_davidson(mr_adc):

    precond = None

    # Compute M matrix sectors to be stored
    if mr_adc.method_type == "ip":
        # Compute h0-h0 block of the effective Hamiltonian matrix
        M_00 = mr_adc_ip.compute_M_00(mr_adc)

        # Compute parts of the h0-h1 block of the effective Hamiltonian matrix
        if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
            M_01 = mr_adc_ip.compute_M_01(mr_adc)

    elif mr_adc.method_type == "ea":
        # Compute h0-h0 block of the effective Hamiltonian matrix
        M_00 = mr_adc_ea.compute_M_00(mr_adc)

        # Compute parts of the h0-h1 block of the effective Hamiltonian matrix
        if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
            M_01 = mr_adc_ea.compute_M_01(mr_adc)

    elif mr_adc.method_type == "ee":
        # Compute h0-h0 block of the effective Hamiltonian matrix
        M_00   = mr_adc_ee.compute_M_00(mr_adc)

        # Compute parts of the h0-h1 and h1-h1 blocks of the effective Hamiltonian matrix
        if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
            M_01 = mr_adc_ee.compute_M_01(mr_adc)

    elif mr_adc.method_type == "cvs-ip":
        # Compute h0-h0 block of the effective Hamiltonian matrix
        mr_adc_cvs_ip.compute_M_00(mr_adc)

        # Compute parts of the h0-h1 block of the effective Hamiltonian matrix
        if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
            mr_adc_cvs_ip.compute_M_01(mr_adc)

    elif mr_adc.method_type == "cvs-ee":
        # Compute h0-h0 block of the effective Hamiltonian matrix
        M_00 = mr_adc_cvs_ee.compute_M_00(mr_adc)

    # Compute diagonal of the M matrix
    if mr_adc.method_type == "ip":
        precond = mr_adc_ip.compute_preconditioner(mr_adc, M_00)
    elif mr_adc.method_type == "ea":
        precond = mr_adc_ea.compute_preconditioner(mr_adc, M_00)
    elif mr_adc.method_type == "ee":
        precond = mr_adc_ee.compute_preconditioner(mr_adc, M_00)

    # Apply Core-Valence Separation Approximation (CVS)
    elif mr_adc.method_type == "cvs-ip":
        precond = mr_adc_cvs_ip.compute_preconditioner(mr_adc)
    elif mr_adc.method_type == "cvs-ee":
        precond = mr_adc_cvs_ee.compute_preconditioner(mr_adc)

    # Compute guess vectors
    x0 = compute_guess_vectors(mr_adc, precond)

    # Define M * vec
    apply_M = None
    if mr_adc.method_type == "ip":
        apply_M = mr_adc_ip.define_effective_hamiltonian(mr_adc, M_00, M_01, M_11)
    elif mr_adc.method_type == "ea":
        apply_M = mr_adc_ea.define_effective_hamiltonian(mr_adc, M_00, M_01, M_11)
    elif mr_adc.method_type == "ee":
        apply_M = mr_adc_ee.define_effective_hamiltonian(mr_adc, M_00, M_01)
    elif mr_adc.method_type == "cvs-ip":
        apply_M = mr_adc_cvs_ip.define_effective_hamiltonian(mr_adc)
    elif mr_adc.method_type == "cvs-ee":
        apply_M = mr_adc_cvs_ee.define_effective_hamiltonian(mr_adc)

    return apply_M, precond, x0

def compute_guess_vectors(mr_adc, precond, ascending = True):

    sort_ind = np.argsort(precond)
    if not ascending:
        sort_ind = sort_ind[::-1]

    x0s = np.zeros((precond.shape[0], mr_adc.nroots))
    min_shape = min(precond.shape[0], mr_adc.nroots)
    x0s[:min_shape,:min_shape] = np.identity(min_shape)

    x0 = np.zeros((precond.shape[0], mr_adc.nroots))
    x0[sort_ind] = x0s.copy()

    return [x0[:, p] for p in range(x0.shape[1])]

## TODO: add in NTOs!
def compute_trans_properties(mr_adc, E, U):

    X = None

    if mr_adc.method_type == "ip":
        X = mr_adc_ip.compute_trans_moments(mr_adc, U)
    elif mr_adc.method_type == "cvs-ip":
        X = mr_adc_cvs_ip.compute_trans_moments(mr_adc, U)
    elif mr_adc.method_type == "ea":
        X = mr_adc_ea.compute_trans_moments(mr_adc, U)
    elif mr_adc.method_type == "ee":
        X = mr_adc_ee.compute_trans_moments(mr_adc, U)
    elif mr_adc.method_type == "cvs-ee":
        X = mr_adc_cvs_ee.compute_trans_moments(mr_adc, U)
    else:
        msg = "Unknown Method Type ..."
        mr_adc.log.error(msg)
        raise Exception(msg)

    # X is a tuple for CVS-EE
    if isinstance(X, tuple):
        X = X[1]
        spec_intensity = np.sum(X**2, axis=0)
    else:
        spec_intensity = 2.0 * np.sum(X**2, axis=0)
   
    # Analyze spectroscopic factors if requested
    #if mr_adc.analyze_spec_factor or (mr_adc.verbose > 4): 
    if mr_adc.analyze_spec_factor: 
        analyze_mo_overlap(mr_adc)
        analyze_spec_factor(mr_adc, X, spec_intensity)
    
    return spec_intensity, X

#========== UTILITY FUNCTIONS FOR SPECIAL ANALYSES ==========#

## TODO: either remove or restore functionality to dyall_hamiltonian
def dyall_hamiltonian(mr_adc):
    """Zeroth Order Dyall Hamiltonian"""

    from prism.mr_adc_integrals import mr_adc_integrals

    # Testing Dyall Hamiltonian expected value
    print("Calculating the Spin-Adapted Dyall Hamiltonian...")

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables needed
    h_aa = mr_adc.h1eff[mr_adc.ncore:mr_adc.nocc, mr_adc.ncore:mr_adc.nocc].copy()
    rdm_ca = mr_adc.rdm.ca
    v_aaaa = mr_adc.v2e.aaaa
    rdm_ccaa = mr_adc.rdm.ccaa
    mo_c = mr_adc.mo[:, :mr_adc.ncore].copy()

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

    print("\n>>> SA Expected value of Zeroth-order Dyall Hamiltonian: {:}".format(temp + temp_E_fc + mr_adc.interface.enuc))

def analyze_spec_factor(mr_adc, X, spec_intensity):
    """Analyze and print spectroscopic factors for each state above a given threshold."""

    print_thresh = mr_adc.spec_factor_print_tol
    mr_adc.log.info(f"\nSpectroscopic factors analysis (threshold = {print_thresh:.2e}): ")

    X_squared = np.square(X.T, order='C')
    
    if mr_adc.method_type == 'cvs-ip':
        X_squared *= 2.0

    results = []

    # Analyze states
    for state in range(X_squared.shape[0]):

        sort_ind = np.argsort(-X_squared[state])
        sorted_cont = X_squared[state][sort_ind]
       
        # If symmetry is not used or symmetry labels are unavailable, assign all orbitals to 'A'
        if not mr_adc.symmetry or not hasattr(mr_adc, 'group_repr_symm'):
            orb_symm_labels = np.repeat(['A'], sorted_cont.shape[0])
        else:
            orb_symm_labels = np.array(mr_adc.group_repr_symm)[sort_ind]

        # check if any contributions meet given threshold
        if not np.any(sorted_cont > print_thresh):
            continue

        # apply threshold to contributions
        spec_contribution = sorted_cont[sorted_cont > print_thresh]
        orb_index = sort_ind[sorted_cont > print_thresh] + 1
        orb_symm_labels = orb_symm_labels[sorted_cont > print_thresh]

        # compute contribution by state
        partial_cont = spec_contribution / spec_intensity[state]

        # apply another threshold (is this necessary?)
        filtered_spec_contribution = spec_contribution[partial_cont > 1e-6]
        orb_index = orb_index[partial_cont > 1e-6]
        partial_cont = partial_cont[partial_cont > 1e-6]
        orb_symm_labels = orb_symm_labels[partial_cont > 1e-6]

        # Store results for this state
        for idx, sym, contrib, partial in zip(orb_index, orb_symm_labels, filtered_spec_contribution, partial_cont):
            results.append((state + 1, idx, sym, contrib, partial))

    # Print results
    if results:
        current_state = None
        mr_adc.log.info("="*60)
        mr_adc.log.info("                             Spectral            Partial")
        mr_adc.log.info("  State       MO         Contribution       Contribution")
        mr_adc.log.info("="*60)
        
        for state_num, mo_idx, symmetry, spec_contrib, partial_contrib in results:
            if current_state is not None and state_num != current_state:
                mr_adc.log.info("") ## line break for new states
            mr_adc.log.info(f"  {state_num:>3d}    {mo_idx:>4d} ({symmetry:<2})       {spec_contrib:8e}       {partial_contrib:8e}")
            current_state = state_num
        mr_adc.log.info("="*60)
    else:
        mr_adc.log.info(f"No significant spectroscopic contributions found above threshold = {print_thresh:.2e}.")

def analyze_mo_overlap(mr_adc):
    """Analyze and print overlap of CASSCF and HF spatial MOs."""
    from functools import reduce

    print_thresh = mr_adc.spec_factor_print_tol
    mr_adc.log.info(f"\nOverlap of CASSCF and HF spatial MO's:")

    # Overlap Matrix
    cas_hf_ovlp = reduce(np.dot, (mr_adc.mo.T, mr_adc.ovlp, mr_adc.mo_scf))
    ovlp_squared = np.square(cas_hf_ovlp, order='C')

    results = []

    # Analyze CAS MOs
    for cas_mo in range(mr_adc.mo.shape[1]):

        hf_ovlp = ovlp_squared[cas_mo]
        sort_ind = np.argsort(hf_ovlp)[::-1]
        hf_ovlp_sorted = hf_ovlp[sort_ind]

        if np.any(hf_ovlp_sorted > print_thresh):
            idx = np.where(hf_ovlp_sorted > print_thresh) 

            hf_ovlp_sorted = hf_ovlp_sorted[idx]
            sort_ind = sort_ind[idx]

            for overlap_val, hf_idx in zip(hf_ovlp_sorted, sort_ind):
                results.append((cas_mo + 1, hf_idx + 1, overlap_val))

    # Print results
    if results:
        current_cas = None
        mr_adc.log.info("="*40)
        mr_adc.log.info("   CASSCF MO    HF MO     Overlap")
        mr_adc.log.info("="*40)
        for cas_mo, hf_mo, overlap in results:
            if current_cas is not None and cas_mo != current_cas:
                mr_adc.log.info("")
            
            mr_adc.log.info(f"     {cas_mo:>3d}        {hf_mo:>3d}      {overlap:>8.3%}")
            current_cas = cas_mo
        mr_adc.log.info("="*40)
    else:
        mr_adc.log.info(f"\nNo significant MO overlaps found above threshold = {print_thresh:.2e}.")

