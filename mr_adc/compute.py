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
#          Donna H. Odhiambo <donna.odhiambo@proton.me>
#

import numpy as np

from prism.mr_adc import integrals
from prism.mr_adc import rdms
from prism.tools import trans_prop

import prism.lib.logger as logger

def kernel(mr_adc):

    log = mr_adc.log
    mr_adc.method = mr_adc.method.lower()
    mr_adc.method_type = mr_adc.method_type.lower()

    cput0 = (logger.process_clock(), logger.perf_counter())
    log.info("\nComputing MR-ADC excitation energies...\n")

    # Initial checks
    initialize(mr_adc)

    # Print calculation info
    print_header(mr_adc)

    # Transform one- and two-electron integrals
    integrals.transform_integrals(mr_adc)

    # Compute CASCI energies and reduced density matrices
    rdms.compute_reference_rdms(mr_adc)

    # Compute amplitudes
    e_tot, e_corr = mr_adc.compute_reference_energy()

    # Compute CVS integrals
    if mr_adc.method_type in ("cvs-ip", "cvs-ee"):
        integrals.transform_cvs_integrals(mr_adc)

    e_tot, de = mr_adc.compute_energy()

    h2ev = mr_adc.interface.hartree_to_ev
    de_ev = de * h2ev

    # Compute transition moments and spectroscopic factors
    spec_intensity, X = mr_adc.compute_properties()

    print_results(mr_adc)

    log.timer0("total %s-%s calculation" % (mr_adc.method_type.upper(), mr_adc.method.upper()), *cput0)

    return de_ev, spec_intensity, X


def initialize(mr_adc):

    # Supported methods and method types
    SUPPORTED_METHODS = {"mr-adc(0)", "mr-adc(1)", "mr-adc(2)", "mr-adc(2)-sx", "mr-adc(2)-x"}
    SUPPORTED_METHOD_TYPES = {"ee", "ip", "ea", "cvs-ip", "cvs-ee"}
    CVS_TYPES = {"cvs-ip", "cvs-ee"}
    DF_COMPATIBLE_TYPES = CVS_TYPES

    log = mr_adc.log

    if mr_adc.method not in SUPPORTED_METHODS:
        msg = "Unknown method %s" % mr_adc.method
        log.error(msg)
        raise Exception(msg)

    if mr_adc.method_type not in SUPPORTED_METHOD_TYPES:
        msg = "Unknown method type %s" % mr_adc.method_type
        log.error(msg)
        raise Exception(msg)

    if mr_adc.interface.with_df and mr_adc.method_type not in DF_COMPATIBLE_TYPES:
        msg = "Density-fitting currently only compatible with CVS method types"
        log.error(msg)
        raise Exception(msg)

    if mr_adc.method_type in CVS_TYPES:

        if mr_adc.ncvs is None or not isinstance(mr_adc.ncvs, int):
            msg = "Method type %s requires ncvs to be a positive integer" % mr_adc.method_type
            log.error(msg)
            raise Exception(msg)

        if mr_adc.ncvs < 1 or mr_adc.ncvs > mr_adc.ncore:
            msg = "Method type %s requires ncvs to be a positive integer less than ncore" % mr_adc.method_type
            log.error(msg)
            raise Exception(msg)

        mr_adc.nval = mr_adc.ncore - mr_adc.ncvs


def print_header(mr_adc):

    ref_df = bool(mr_adc.interface.reference_df)
    df = bool(mr_adc.interface.with_df)

    h2ev = mr_adc.interface.hartree_to_ev

    # Print general information
    mr_adc.log.info("Method:                                            %s-%s" % (mr_adc.method_type, mr_adc.method))
    mr_adc.log.info("Nuclear repulsion energy:                    %20.12f" % mr_adc.enuc)
    mr_adc.log.info("Number of electrons:                               %d" % mr_adc.nelec)
    mr_adc.log.info("Number of basis functions:                         %d" % mr_adc.nmo)
    mr_adc.log.info("Reference wavefunction type:                       %s" % mr_adc.interface.reference)
    mr_adc.log.info("Number of core orbitals:                           %d" % mr_adc.ncore)
    mr_adc.log.info("Number of active orbitals:                         %d" % mr_adc.ncas)
    mr_adc.log.info("Number of external orbitals:                       %d" % mr_adc.nextern)
    mr_adc.log.info("Number of active electrons:                        %s" % mr_adc.ref_nelecas)
    mr_adc.log.info("Reference state active-space energy:         %20.12f" % mr_adc.e_ref_cas[0])
    mr_adc.log.info("Reference state spin multiplicity:                 %s" % mr_adc.ref_wfn_spin_mult)
    mr_adc.log.info("Number of MR-ADC roots requested:                  %d" % mr_adc.nroots)
    if mr_adc.ncvs is not None:
        mr_adc.log.info("Number of CVS orbitals:                            %d" % mr_adc.ncvs)
        mr_adc.log.info("Number of valence (non-CVS) orbitals:              %d" % (mr_adc.ncore - mr_adc.ncvs))

    mr_adc.log.info("Reference density fitting?                         %s" % ref_df)
    mr_adc.log.info("Correlation density fitting?                       %s" % df)
    mr_adc.log.info("Temporary directory path:                          %s" % mr_adc.temp_dir)

    mr_adc.log.info("\nInternal contraction:                              %s" % "Full")
    mr_adc.log.info("Overlap truncation parameter (singles):            %.2e" % mr_adc.s_thresh_singles)
    mr_adc.log.info("Overlap truncation parameter (doubles):            %.2e" % mr_adc.s_thresh_doubles)
    mr_adc.log.info("Projector for the semi-internal amplitudes:        %s" % mr_adc.semi_internal_projector)

    mr_adc.log.info("\nEinsum Backend:                                    %s" % mr_adc.interface.einsum_backend)

    # Print info about CASCI states
    if mr_adc.ncasci > 0:
        mr_adc.log.info("Number of CASCI states:                            %d" % mr_adc.ncasci)

    if mr_adc.e_cas_ci is not None:
        mr_adc.log.extra("CASCI excitation energies (eV):                    %s" % h2ev*(mr_adc.e_cas_ci - mr_adc.e_cas))

def compute_energy(mr_adc):

    # Define function for the matrix-vector product S^(-1/2) M S^(-1/2) vec
    mr_adc.compute_excitation_manifolds()

    # Setup Davidson algorithm parameters
    apply_M, precond, x0 = setup_davidson(mr_adc)

    davidson_verbose = 6 if mr_adc.verbose > 3 else 3

    # Using Davidson algorithm, solve the [S^(-1/2) M S^(-1/2) C = C E] eigenvalue problem
    cput1 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.info("")
    conv, de, U = mr_adc.davidson(lambda xs: [apply_M(x) for x in xs], x0, precond,
                                           nroots = mr_adc.nroots,
                                           verbose = davidson_verbose,
                                           max_space = mr_adc.max_space,
                                           max_memory = mr_adc.max_memory,
                                           max_cycle = mr_adc.max_cycle,
                                           tol = mr_adc.tol_e,
                                           tol_residual = mr_adc.tol_r)

    mr_adc.h_evec = np.array(U)
    mr_adc.e_tot = mr_adc.e_ref_nevpt2 + de
    mr_adc.e_diff = de

    mr_adc.log.timer("solving eigenvalue problem", *cput1)

    if all(conv):
        mr_adc.log.extra('Full convergence reached.')
    elif any(conv):
        not_converged = np.where(~conv)[0]
        mr_adc.log.warn(f'State(s) {not_converged+1} did not converge!')
    else:
        mr_adc.log.warn('No convergence reached for Davidson iterations!')

    return mr_adc.e_tot, mr_adc.e_diff


def setup_davidson(mr_adc):

    precond = None

    mr_adc.compute_M_00()

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        if hasattr(mr_adc, "compute_M_01"):
            mr_adc.compute_M_01()

    # Compute diagonal of the M matrix
    precond = mr_adc.compute_preconditioner()

    # Compute guess vectors
    x0 = compute_guess_vectors(mr_adc, precond)

    # Define M * vec
    apply_M = mr_adc.define_effective_hamiltonian()

    return apply_M, precond, x0


def compute_guess_vectors(mr_adc, precond, ascending = True):

    nroots = mr_adc.nroots
    dim = mr_adc.h_orth.dim

    sort_ind = np.argsort(precond)
    if not ascending:
        sort_ind = sort_ind[::-1]

    x0s = np.zeros((dim, nroots))
    min_shape = min(dim, nroots)
    x0s[:min_shape,:min_shape] = np.identity(min_shape)

    x0 = np.zeros((dim, nroots))
    x0[sort_ind] = x0s.copy()

    return [x0[:, p] for p in range(nroots)]


def compute_properties(mr_adc):

    # Spectrocopic amplitudes
    X = mr_adc.compute_trans_moments()
    mr_adc.properties["spec_amplitudes"] = X

    # Spectrocopic factors
    # spec_factors = 2.0 * np.sum(X**2, axis=0)

    if mr_adc.method_type == 'cvs-ee':
        DX = np.einsum("Rpq,Kpq->RK", mr_adc.dip_mom, X)
        spec_factors = 2.0 * np.sum(DX**2, axis=0)
    else:
        spec_factors = 2.0 * np.sum(X**2, axis=0)

    mr_adc.properties["spec_probabilities"] = spec_factors

    return spec_factors, X


def analyze(mr_adc):

    if "spec_amplitudes" in mr_adc.properties:
        if hasattr(mr_adc, "analyze_spec_factor"):
            mr_adc.analyze_spec_factor()

        if mr_adc.compute_dyson:
            trans_prop.compute_dyson(mr_adc.interface, mr_adc.properties["spec_amplitudes"])

    if mr_adc.h_evec is not None:
        if hasattr(mr_adc, "analyze_eigenvector"):
            mr_adc.analyze_eigenvector()


def print_results(mr_adc):

    de = mr_adc.e_diff

    h2ev = mr_adc.interface.hartree_to_ev
    h2cm = mr_adc.interface.hartree_to_inv_cm

    mr_adc.log.info("\nSummary of results for the %s-%s calculation with the %s reference:" % (mr_adc.method_type.upper(), mr_adc.method.upper(), mr_adc.interface.reference.upper()))

    mr_adc.log.info("------------------------------------------------------------------------------------------------")
    mr_adc.log.info("  State        dE(a.u.)        dE(eV)      dE(nm)       dE(cm-1)       Intensity")
    mr_adc.log.info("------------------------------------------------------------------------------------------------")

    de_ev = de * h2ev
    de_cm = de * h2cm

    spec_intensity = mr_adc.properties["spec_probabilities"]

    for p in range(len(de)):
        de_nm = 10000000 / de_cm[p]
        mr_adc.log.info("%5d     %14.8f  %12.4f %10.4f  %14.4f    %10.6f" % ((p+1), de[p], de_ev[p], de_nm, de_cm[p], spec_intensity[p]))

    mr_adc.log.info("------------------------------------------------------------------------------------------------")
