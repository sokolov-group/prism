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

import sys
import numpy as np
from functools import reduce

import prism.mr_adc_amplitudes as mr_adc_amplitudes
import prism.mr_adc_integrals as mr_adc_integrals
import prism.mr_adc_cvs_ip as mr_adc_cvs_ip
import prism.mr_adc_cvs_ee as mr_adc_cvs_ee

import prism.lib.logger as logger

def kernel(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.info("\nComputing MR-ADC excitation energies...\n")

    # Print general information
    mr_adc.log.info("Method:                                            %s-%s" % (mr_adc.method_type, mr_adc.method))
    mr_adc.log.info("Number of MR-ADC roots requested:                  %d" % mr_adc.nroots)
    mr_adc.log.info("Ground-state active-space energy:            %20.12f" % mr_adc.e_cas)
    mr_adc.log.info("Nuclear repulsion energy:                    %20.12f" % mr_adc.enuc)
    mr_adc.log.info("Number of basis functions:                         %d" % mr_adc.nmo)
    mr_adc.log.info("Number of core orbitals:                           %d" % mr_adc.ncore)
    mr_adc.log.info("Number of active orbitals:                         %d" % mr_adc.ncas)
    mr_adc.log.info("Number of external orbitals:                       %d" % mr_adc.nextern)
    mr_adc.log.info("Number of electrons:                               %d" % mr_adc.nelec)
    mr_adc.log.info("Number of active electrons:                        %s" % str(mr_adc.nelecas))
    if mr_adc.ncvs is not None:
        mr_adc.log.info("Number of CVS orbitals:                            %d" % mr_adc.ncvs)
        mr_adc.log.info("Number of valence (non-CVS) orbitals:              %d" % (mr_adc.ncore - mr_adc.ncvs))

    mr_adc.log.extra("Overlap truncation parameter (singles):            %e" % mr_adc.s_thresh_singles)
    mr_adc.log.extra("Overlap truncation parameter (doubles):            %e" % mr_adc.s_thresh_doubles)

    # Print info about CASCI states
    mr_adc.log.info("Number of CASCI states:                            %d" % mr_adc.ncasci)

    if mr_adc.e_cas_ci is not None:
        mr_adc.log.extra("CASCI excitation energies (eV):                    %s" % str(27.2114*(mr_adc.e_cas_ci - mr_adc.e_cas)))

    mr_adc.log.debug("Temporary directory path: %s" % mr_adc.temp_dir)

    davidson_verbose = 3
    if mr_adc.verbose > 3:
        davidson_verbose = 6

    # Compute amplitudes
    mr_adc_amplitudes.compute_amplitudes(mr_adc)

    # Compute CVS integrals
    if mr_adc.method_type == "cvs-ip" or mr_adc.method_type == "cvs-ee":
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
    apply_M, precond, x0 = setup_davidson(mr_adc)

    # Using Davidson algorithm, solve the [S^(-1/2) M S^(-1/2) C = C E] eigenvalue problem
    cput1 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.info("")
    conv, E, U = mr_adc.interface.davidson(lambda xs: [apply_M(x) for x in xs], x0, precond,
                                           nroots = mr_adc.nroots,
                                           verbose = davidson_verbose,
                                           max_space = mr_adc.max_space,
                                           max_cycle = mr_adc.max_cycle,
                                           tol = mr_adc.tol_e,
                                           tol_residual = mr_adc.tol_davidson)

    mr_adc.log.timer("solving eigenvalue problem", *cput1)

    mr_adc.log.note("\n%s-%s excitation energies (a.u.):" % (mr_adc.method_type, mr_adc.method))
    print(E.reshape(-1, 1))
    mr_adc.log.note("\n%s-%s excitation energies (eV):" % (mr_adc.method_type, mr_adc.method))
    E_ev = E * 27.2114
    print(E_ev.reshape(-1, 1))
    sys.stdout.flush()

    # Compute transition moments and spectroscopic factors
    U = np.array(U)

    spec_intensity, X = compute_trans_properties(mr_adc, E, U)

    mr_adc.log.info("\n------------------------------------------------------------------------------")
    mr_adc.log.timer0("total MR-ADC calculation", *cput0)

    return E_ev, spec_intensity, X

def setup_davidson(mr_adc):

    precond = None

    # Compute M matrix sectors to be stored
    if mr_adc.method_type == "ip":
        # Compute h0-h0 block of the effective Hamiltonian matrix
        M_00 = mr_adc_ip.compute_M_00(mr_adc)

        # Compute parts of the h0-h1 block of the effective Hamiltonian matrix
        if mr_adc.method == "mr-adc(2)" or mr_adc.method == "mr-adc(2)-x":
            M_01 = mr_adc_ip.compute_M_01(mr_adc)

    elif mr_adc.method_type == "ea":
        # Compute h0-h0 block of the effective Hamiltonian matrix
        M_00 = mr_adc_ea.compute_M_00(mr_adc)

        # Compute parts of the h0-h1 block of the effective Hamiltonian matrix
        if mr_adc.method == "mr-adc(2)" or mr_adc.method == "mr-adc(2)-x":
            M_01 = mr_adc_ea.compute_M_01(mr_adc)

    elif mr_adc.method_type == "ee":
        # Compute h0-h0 block of the effective Hamiltonian matrix
        M_00   = mr_adc_ee.compute_M_00(mr_adc)

        # Compute parts of the h0-h1 and h1-h1 blocks of the effective Hamiltonian matrix
        if mr_adc.method == "mr-adc(2)" or mr_adc.method == "mr-adc(2)-x":
            M_01 = mr_adc_ee.compute_M_01(mr_adc)

    elif mr_adc.method_type == "cvs-ip":
        # Compute h0-h0 block of the effective Hamiltonian matrix
        mr_adc_cvs_ip.compute_M_00(mr_adc)

        # Compute parts of the h0-h1 block of the effective Hamiltonian matrix
        if mr_adc.method == "mr-adc(2)" or mr_adc.method == "mr-adc(2)-x":
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

    sort_ind = None
    if ascending:
        sort_ind = np.argsort(precond)
    else:
        sort_ind = np.argsort(precond)[::-1]

    x0s = np.zeros((precond.shape[0], mr_adc.nroots))
    min_shape = min(precond.shape[0], mr_adc.nroots)
    x0s[:min_shape,:min_shape] = np.identity(min_shape)

    x0 = np.zeros((precond.shape[0], mr_adc.nroots))
    x0[sort_ind] = x0s.copy()

    x0s = []
    for p in range(x0.shape[1]):
        x0s.append(x0[:,p])

    return x0s

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
        X, dX = mr_adc_cvs_ee.compute_trans_moments(mr_adc, U)
    else:
        msg = "Unknown Method Type ..."
        mr_adc.log.error(msg)
        raise Exception(msg)

    if mr_adc.method_type == "cvs-ip":
        spec_intensity = 2.0 * np.sum(X**2, axis=0)
    elif mr_adc.method_type == "ee" or mr_adc.method_type == "cvs-ee":
        spec_intensity = np.sum(dX**2, axis=0)
        
    mr_adc.log.note("\n%s-%s spectroscopic intensity:" % (mr_adc.method_type, mr_adc.method))
    print(spec_intensity.reshape(-1, 1))

    if mr_adc.method_type == "ee" or mr_adc.method_type == "cvs-ee":
        osc_strength = (2.0/3.0) * E * spec_intensity

        mr_adc.log.note("\n%s-%s oscillator strength:" % (mr_adc.method_type, mr_adc.method))
        print(osc_strength.reshape(-1, 1))

        #mr_adc_cvs_ee.analyze_eigenvector(mr_adc, U, E, osc_strength)

    if mr_adc.analyze_spec_factor or (mr_adc.verbose > 4):
        analyze_trans_properties(mr_adc)
        if mr_adc.method_type == "cvs-ip":
            analyze_spec_factor(mr_adc, X, spec_intensity)
        elif mr_adc.method_type == "cvs-ee":
            analyze_spec_factor(mr_adc, dX, spec_intensity)

    return spec_intensity, X


def analyze_spec_factor(mr_adc, X, spec_intensity):

    print("\nSpectroscopic Factors Analysis:")

    print_thresh = mr_adc.spec_factor_print_tol
    mr_adc.log.extra("Print spectroscopic factors > %e" %  print_thresh)

    if mr_adc.method_type == "cvs-ee":
        X_2 = (X.T)**2
    elif mr_adc.method_type == "cvs-ip":
        X_2 = 2.0 * (X.T)**2

    for i in range(X_2.shape[0]):

        sort = np.argsort(-X_2[i,:])
        X_2_row = X_2[i,:]
        X_2_row = X_2_row[sort]

        if not mr_adc.symmetry:
            group_repr_symm = np.repeat(['A'], X_2_row.shape[0])
        else:
            group_repr_symm = mr_adc.group_repr_symm
            group_repr_symm = np.array(group_repr_symm)

            group_repr_symm = group_repr_symm[sort]

        spec_Contribution = X_2_row[X_2_row > print_thresh]
        index_orb = sort[X_2_row > print_thresh] + 1

        if np.sum(spec_Contribution) <= print_thresh:
            continue

        partial_Contribution = spec_Contribution / spec_intensity[i]

        spec_Contribution = spec_Contribution[partial_Contribution > 1e-6]
        index_orb = index_orb[partial_Contribution > 1e-6]
        partial_Contribution = partial_Contribution[partial_Contribution > 1e-6]

        print("\n%s | root %d \n" % (mr_adc.method, i))
        print("  MO          Spec. Contribution       Partial Contribution")
        print("-------------------------------------------------------------")

        for c in range(index_orb.shape[0]):
            print(" %3.d (%s)             %10.8f                 %10.8f" % (index_orb[c], group_repr_symm[c],
                                                                            spec_Contribution[c],
                                                                            partial_Contribution[c]))

def analyze_trans_properties(mr_adc):

    print ("\nOverlap of CASSCF and HF spatial MO's:")

    print_thresh = 0.01
    mr_adc.log.extra("Print HF MO contributions > %e" %  print_thresh)

    cas_hf_ovlp = reduce(np.dot, (mr_adc.mo.T, mr_adc.ovlp, mr_adc.mo_hf))
    for p in range(mr_adc.mo.shape[1]):

        hf_ovlp = cas_hf_ovlp[p]**2
        hf_ovlp_ind = np.argsort(hf_ovlp)[::-1]
        hf_ovlp_sorted = hf_ovlp[hf_ovlp_ind]

        print ("\nCASSCF MO #%d:" % (p + 1))

        for hf_p in range(mr_adc.mo_hf.shape[1]):
            if (hf_ovlp_sorted[hf_p] > print_thresh):
                print ("%.3f HF MO #%d" % (hf_ovlp_sorted[hf_p], hf_ovlp_ind[hf_p] + 1))

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
