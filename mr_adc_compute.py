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

import sys
import time
import numpy as np
import prism.mr_adc_amplitudes as mr_adc_amplitudes
import prism.mr_adc_cvs_ip as mr_adc_cvs_ip
from functools import reduce

def kernel(mr_adc):

    start_time = time.time()

    print("Computing MR-ADC excitation energies...\n")

    # Print general information
    print("Method:                                            %s-%s" % (mr_adc.method_type, mr_adc.method))
    print("Number of MR-ADC roots requested:                  %d" % mr_adc.nroots)
    print("Ground-state active-space energy:            %20.12f" % mr_adc.e_cas)
    print("Nuclear repulsion energy:                    %20.12f" % mr_adc.enuc)
    print("Number of basis functions:                         %d" % mr_adc.nmo)
    print("Number of core orbitals:                           %d" % mr_adc.ncore)
    print("Number of active orbitals:                         %d" % mr_adc.ncas)
    print("Number of external orbitals:                       %d" % mr_adc.nextern)
    print("Number of electrons:                               %d" % mr_adc.nelec)
    print("Number of active electrons:                        %s" % str(mr_adc.nelecas))
    if mr_adc.ncvs is not None:
        print("Number of CVS orbitals:                            %d" % mr_adc.ncvs)
        print("Number of valence (non-CVS) orbitals:              %d" % (mr_adc.ncore - mr_adc.ncvs))

    print("Overlap truncation parameter (singles):            %e" % mr_adc.s_thresh_singles)
    # if mr_adc.s_damping_strength is None:
    #     print("Overlap truncation parameter (singles):            %e" % mr_adc.s_thresh_singles)
    # else:
    #     print("Overlap damping width:                             %f" % mr_adc.s_damping_strength)
    #     print("Overlap truncation parameter (singles):            %e" % (mr_adc.s_thresh_singles * 10**(- mr_adc.s_damping_strength / 2)))

    # Print info about CASCI states
    print("Overlap truncation parameter (doubles):            %e" % mr_adc.s_thresh_doubles)
    print("Number of CASCI states:                            %d\n" % mr_adc.ncasci)

    if mr_adc.e_cas_ci is not None:
        print("CASCI excitation energies (eV):                    %s\n" % str(27.2114*(mr_adc.e_cas_ci - mr_adc.e_cas)))
    sys.stdout.flush()

    # Compute amplitudes
    mr_adc_amplitudes.compute_amplitudes(mr_adc)

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
    conv, E, U = mr_adc.interface.davidson(lambda xs: [apply_M(x) for x in xs], x0, precond,
                                           nroots = mr_adc.nroots,
                                           verbose = 6,
                                           max_space = mr_adc.max_space,
                                           max_cycle = mr_adc.max_cycle,
                                           tol_residual = mr_adc.tol_davidson)

    print("\n%s-%s excitation energies (a.u.):" % (mr_adc.method_type, mr_adc.method))
    print(E.reshape(-1, 1))
    print("\n%s-%s excitation energies (eV):" % (mr_adc.method_type, mr_adc.method))
    E_ev = E * 27.2114
    print(E_ev.reshape(-1, 1))
    sys.stdout.flush()

    # Compute transition moments and spectroscopic factors
    # spec_intensity = compute_trans_properties(mr_adc, E, U)
    spec_intensity = 'spec_intensity'

    print("\nTotal time:                                       %f sec" % (time.time() - start_time))

    return E_ev, spec_intensity

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
        precond = mr_adc_cvs_ee.compute_preconditioner(mr_adc, M_00)

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
        apply_M = mr_adc_cvs_ee.define_effective_hamiltonian(mr_adc, M_00)

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

    start_time = time.time()

    print("\nComputing transition moments matrix...\n")
    sys.stdout.flush()

    T = None

    if mr_adc.method_type == "ip":
        T = mr_adc_ip.compute_trans_moments(mr_adc)
    elif mr_adc.method_type == "cvs-ip":
        T = mr_adc_cvs_ip.compute_trans_moments(mr_adc)
    elif mr_adc.method_type == "ea":
        T = mr_adc_ea.compute_trans_moments(mr_adc)
    elif mr_adc.method_type == "ee":
        T = mr_adc_ee.compute_trans_moments(mr_adc)
    elif mr_adc.method_type == "cvs-ee":
        T = mr_adc_cvs_ee.compute_trans_moments(mr_adc)
    else:
        raise Exception("Unknown Method Type ...")

    U = np.array(U)
    T = np.dot(T, U.T)

    spec_intensity = np.sum(T**2, axis=0)

    print("%s-%s spectroscopic intensity:" % (mr_adc.method_type, mr_adc.method))
    print(spec_intensity.reshape(-1, 1))

    if mr_adc.method_type in ("cvs-ee", "ee"):
        osc_strength = (2.0/3.0) * E * spec_intensity

        print("\n%s-%s oscillator strength:" % (mr_adc.method_type, mr_adc.method))
        print(osc_strength.reshape(-1, 1))

    #TODO: Change to external functions
    if mr_adc.print_level > 5:
        analyze_trans_properties(mr_adc, T)
        analyze_spec_factor(mr_adc, T)

    print("Time for computing transition moments matrix:     %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return spec_intensity

def analyze_trans_properties(mr_adc, T):

    print("Overlap of CASSCF and HF spatial MO's:")

    cas_hf_ovlp = reduce(np.dot, (mr_adc.mo.T, mr_adc.ovlp, mr_adc.mo_hf))

    print_thresh = 0.1

    for p in range(mr_adc.mo.shape[1]):

        hf_ovlp = cas_hf_ovlp[p]**2
        hf_ovlp_ind = np.argsort(hf_ovlp)[::-1]
        hf_ovlp_sorted = hf_ovlp[hf_ovlp_ind]

        print("\nCASSCF MO #%d:" % (p + 1))

        for hf_p in range(mr_adc.mo_hf.shape[1]):
            if (hf_ovlp_sorted[hf_p] > print_thresh):
                print("%.3f HF MO #%d" % (hf_ovlp_sorted[hf_p], hf_ovlp_ind[hf_p] + 1))

    print("\n")

def analyze_spec_factor(mr_adc, T):

    # Einsum
    einsum = mr_adc.interface.einsum

    print("\nSpectroscopic Factors Analysis:\n")

    if  mr_adc.print_level == 6:
        print_thresh = 0.1
    else:
        print_thresh = 0.00000001

    print("Printing threshold: %e" %  print_thresh)

    X = (T.T).copy()
    X_2 = X.copy()**2

    for i in range(X_2.shape[0]):

        sort = np.argsort(-X_2[i,:])
        X_2_row = X_2[i,:]
        X_2_row = X_2_row[sort]

        spec_Contribution = X_2_row[X_2_row > print_thresh]
        index_orb = sort[X_2_row > print_thresh]

        if np.sum(spec_Contribution) == 0.0:
            continue

        print("\nRoot %d:\n" % (i))
        print("  MO (spin)    Spec. Contribution")
        print("---------------------------------")

        for c in range(index_orb.shape[0]):
            index_mo = index_orb[c] // 2 + 1

            spin = "A"
            if (index_orb[c] % 2 == 1):
                spin = "B"

            print(" %3.d (%s)               %10.8f" % (index_mo, spin, spec_Contribution[c]))

    print("\n")

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
