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

import sys
import numpy as np
from functools import reduce

from prism.mr_adc import amplitudes
from prism.mr_adc import integrals
from prism.mr_adc import rdms
from prism.mr_adc import cvs_ip

import prism.lib.logger as logger

def kernel(mr_adc):

    # Initial checks
    log = mr_adc.log
    mr_adc.method = mr_adc.method.lower()
    mr_adc.method_type = mr_adc.method_type.lower()

    if mr_adc.method not in ("mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
        msg = "Unknown method %s" % mr_adc.method
        log.error(msg)
        raise Exception(msg)

    if mr_adc.method_type not in ("cvs-ip"):
        msg = "Unknown method type %s" % mr_adc.method_type
        log.error(msg)
        raise Exception(msg)

    if mr_adc.interface.with_df and mr_adc.method_type not in ('cvs-ip'):
        msg = "Density-fitting currently only compatible with CVS-IP method type."
        log.error(msg)
        raise Exception(msg)

    if mr_adc.method_type == "cvs-ip" and mr_adc.ncvs is None:
        msg = "Method type %s requires setting the ncvs parameter" % mr_adc.method_type
        log.error(msg)
        raise Exception(msg)

    if mr_adc.method_type in ("cvs-ip", "cvs-ee"):

        if isinstance (mr_adc.ncvs, int):
            if mr_adc.ncvs < 1 or mr_adc.ncvs > mr_adc.ncore:
                msg = '''Method type %s requires setting the ncvs parameter as a
                         positive integer that is smaller than ncore''' % mr_adc.method_type
                log.error(msg)
                raise Exception(msg)

            mr_adc.nval = mr_adc.ncore - mr_adc.ncvs

        else:
            msg = "Method type %s requires setting the ncvs parameter as a positive integer" % mr_adc.method_type
            log.error(msg)
            raise Exception(msg)

        mr_adc.ncasci = 0

    # TODO: Temporary check of what methods are implemented in this version
    if mr_adc.method_type not in ("cvs-ip"):
        msg = "This spin-adapted version does not currently support method type %s" % mr_adc.method_type
        log.error(msg)
        raise Exception(msg)

    # Transform one- and two-electron integrals
    log.info("\nTransforming integrals to MO basis...")
    integrals.transform_integrals_1e(mr_adc)
    if mr_adc.interface.with_df:
        integrals.transform_Heff_integrals_2e_df(mr_adc)
        integrals.transform_integrals_2e_df(mr_adc)
    else: 
        # TODO: this actually handles out-of-core integrals too, rename the function
        integrals.transform_integrals_2e_incore(mr_adc)

    # Compute CASCI energies and reduced density matrices
    rdms.compute_reference_rdms(mr_adc)

    # TODO: Compute CASCI wavefunctions for excited states in the active space
    # rdms.compute_es_rdms(mr_adc)
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

    davidson_verbose = 3
    if mr_adc.verbose > 3:
        davidson_verbose = 6

    # Compute amplitudes
    e_tot, e_corr = amplitudes.compute_reference_energy(mr_adc)

    # Compute CVS integrals
    if mr_adc.method_type == "cvs-ip":
        if mr_adc.interface.with_df:
            integrals.compute_cvs_integrals_2e_df(mr_adc)
        else:
            integrals.compute_cvs_integrals_2e_incore(mr_adc)

    # Define function for the matrix-vector product S^(-1/2) M S^(-1/2) vec
    mr_adc.compute_excitation_manifolds()

    # Setup Davidson algorithm parameters
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

    mr_adc.log.info("\nSummary of results for the %s-%s calculation with the %s reference:" % (mr_adc.method_type.upper(), mr_adc.method.upper(), mr_adc.interface.reference.upper()))

    mr_adc.log.info("------------------------------------------------------------------------------------------------")
    mr_adc.log.info("  State        dE(a.u.)        dE(eV)      dE(nm)       dE(cm-1)       Intensity")
    mr_adc.log.info("------------------------------------------------------------------------------------------------")

    de_ev = de * h2ev
    de_cm = de * h2cm

    for p in range(len(de)):
        de_nm = 10000000 / de_cm[p]
        mr_adc.log.info("%5d     %14.8f  %12.4f %10.4f  %14.4f    %10.6f" % ((p+1), de[p], de_ev[p], de_nm, de_cm[p], spec_intensity[p]))

    mr_adc.log.info("------------------------------------------------------------------------------------------------")

    mr_adc.log.timer0("total %s-%s calculation" % (mr_adc.method_type.upper(), mr_adc.method.upper()), *cput0)

    return de_ev, spec_intensity, X

def setup_davidson(mr_adc):

    precond = None

    mr_adc.compute_M_00()
    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        mr_adc.compute_M_01()

    # Compute diagonal of the M matrix
    if mr_adc.method_type == "ip":
        precond = mr_adc_ip.compute_preconditioner(mr_adc, M_00)
    elif mr_adc.method_type == "ea":
        precond = mr_adc_ea.compute_preconditioner(mr_adc, M_00)
    elif mr_adc.method_type == "ee":
        precond = mr_adc_ee.compute_preconditioner(mr_adc, M_00)

    # Apply Core-Valence Separation Approximation (CVS)
    elif mr_adc.method_type == "cvs-ip":
        precond = cvs_ip.compute_preconditioner(mr_adc)
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
        apply_M = cvs_ip.define_effective_hamiltonian(mr_adc)
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

    X = None

    if mr_adc.method_type == "ip":
        X = mr_adc_ip.compute_trans_moments(mr_adc, U)
    elif mr_adc.method_type == "cvs-ip":
        X = cvs_ip.compute_trans_moments(mr_adc, U)
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

    spec_intensity = 2.0 * np.sum(X**2, axis=0)

    if mr_adc.method_type in ("cvs-ee", "ee"):
        osc_strength = (2.0/3.0) * E * spec_intensity

    if (mr_adc.analyze_spec_factor or mr_adc.verbose > 4) and (mr_adc.method_type == "cvs-ip"):
        cvs_ip.analyze_spec_factor(mr_adc, X, spec_intensity)

    return spec_intensity, X

def dyall_hamiltonian(mr_adc):
    """Zeroth Order Dyall Hamiltonian"""

    from prism.integrals import integrals

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
    v_cccc = integrals.transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_c)

    # Calculating temp_E_fc
    temp_E_fc  = einsum('ii', h_cc, optimize = True)
    temp_E_fc += 2.0 * einsum('ijij', v_cccc, optimize = True)
    temp_E_fc -= einsum('jiij', v_cccc, optimize = True)

    # Calculating H_act
    temp  = einsum('xy,xy', h_aa, rdm_ca, optimize = einsum_type)
    temp += 1/2 * einsum('xyzw,xyzw', v_aaaa, rdm_ccaa, optimize = einsum_type)

    print("\n>>> SA Expected value of Zeroth-order Dyall Hamiltonian: {:}".format(temp + temp_E_fc + mr_adc.interface.enuc))
