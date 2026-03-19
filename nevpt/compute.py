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
#          James D. Serna <jserna456@gmail.com>
#          Nicholas Y. Chiang <nicholas.yiching.chiang@gmail.com>
#

from prism.nevpt import integrals
#from prism.nevpt import soc

import prism.lib.logger as logger

def kernel(nevpt):

    # Initial checks
    nevpt.method = nevpt.method.lower()

    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.info("\nComputing NEVPT energies...\n")

    # Initial checks
    initialize(nevpt)

    # Print calculation info
    print_header(nevpt)

    # Transform one- and two-electron integrals
    integrals.transform_integrals(nevpt)

    # Compute state-specific or quasidegenerate NEVPT energy
    nevpt.compute_energy()

    # Compute properties and spin multiplicity
    osc_str = nevpt.compute_properties()

    # Print results
    print_results(nevpt, osc_str)

    if nevpt.soc:
        nevpt.log.timer0("total %s calculation" % ("SOC-"+nevpt.method.upper()), *cput0)
    else:
        nevpt.log.timer0("total %s calculation" % nevpt.method.upper(), *cput0)

    return nevpt.e_tot, nevpt.e_corr, osc_str


def initialize(nevpt):

    if nevpt.method not in ("nevpt2"):
        msg = "Unknown method %s" % nevpt.method
        nevpt.log.info(msg)
        raise Exception(msg)

    if nevpt.nfrozen is None:
        nevpt.nfrozen = 0

    if nevpt.nfrozen > nevpt.ncore:
        msg = "The number of frozen orbitals cannot exceed the number of core orbitals"
        nevpt.log.error(msg)
        raise Exception(msg)
    
    if nevpt.rdm_order not in [0,2]:
         raise ValueError(f"Invalid {'rdm_order'}: '{nevpt.rdm_order}'. Available options are {0,2}.")
     
    avail_shifts = ['imaginary', 'DSRG']
    
    if nevpt.shift_type_m1p is not None and nevpt.shift_type_m1p not in avail_shifts:
        raise ValueError(f"Invalid {'shift_type_m1p'}: '{nevpt.shift_type_m1p}'. Available options are {avail_shifts}.")

    if nevpt.shift_type_p1p is not None and nevpt.shift_type_p1p not in avail_shifts:
        raise ValueError(f"Invalid {'shift_type_p1p'}: '{nevpt.shift_type_p1p}'. Available options are {avail_shifts}.")
    
    if nevpt.shift_type_0p is not None and nevpt.shift_type_0p not in avail_shifts:
        raise ValueError(f"Invalid {'shift_type_0p'}: '{nevpt.shift_type_0p}'. Available options are {avail_shifts}.")

def analyze(nevpt):

    from prism.tools import trans_prop

    n_micro_states = sum(nevpt.ref_wfn_deg)
    if nevpt.compute_ntos:
        if n_micro_states == 1:
            nevpt.log.warn('Only one state provided for NTO analysis.')
        else:
            # GS -> ES only
            trdm = nevpt.make_rdm1(L=0)[1:]
            for state, trdm_state in enumerate(trdm):
                trans_prop.compute_ntos(nevpt.interface, trdm_state, initial_state=0, target_state=state+1)

def print_header(nevpt):

    n_states = len(nevpt.ref_wfn_deg)
    n_micro_states = sum(nevpt.ref_wfn_deg)

    ref_df = bool(nevpt.interface.reference_df)
    df = bool(nevpt.interface.with_df)

    # Print general information
    nevpt.log.info("Method:                                            %s" % nevpt.method)
    nevpt.log.info("Nuclear repulsion energy:                    %20.12f" % nevpt.enuc)
    nevpt.log.info("Number of electrons:                               %d" % nevpt.nelec)
    nevpt.log.info("Number of basis functions:                         %d" % nevpt.nmo)
    nevpt.log.info("Reference wavefunction type:                       %s" % nevpt.interface.reference)
    nevpt.log.info("Number of reference states:                        %d" % n_states)
    nevpt.log.info("Number of reference microstates:                   %d" % n_micro_states)
    nevpt.log.info("Number of frozen orbitals:                         %d" % nevpt.nfrozen)
    nevpt.log.info("Number of core orbitals:                           %d" % nevpt.ncore)
    nevpt.log.info("Number of active orbitals:                         %d" % nevpt.ncas)
    nevpt.log.info("Number of external orbitals:                       %d" % nevpt.nextern)

    nevpt.log.info("Reference density fitting?                         %s" % ref_df)
    nevpt.log.info("Correlation density fitting?                       %s" % df)
    nevpt.log.info("Temporary directory path:                          %s" % nevpt.temp_dir)

    nevpt.log.info("\nInternal contraction:                              %s" % "Full (= Partial)")
    nevpt.log.info("Compute singles amplitudes?                        %s" % str(nevpt.compute_singles_amplitudes))
    nevpt.log.info("Overlap truncation parameter (singles):            %e" % nevpt.s_thresh_singles)
    nevpt.log.info("Overlap truncation parameter (doubles):            %e" % nevpt.s_thresh_doubles)
    if nevpt.compute_singles_amplitudes:
        nevpt.log.info("Projector for the semi-internal amplitudes:        %s" % nevpt.semi_internal_projector)


def print_results(nevpt, osc_str):

    h2ev = nevpt.interface.hartree_to_ev
    h2cm = nevpt.interface.hartree_to_inv_cm

    if nevpt.soc:
        nevpt.log.info("\nSummary of results for the %s calculation with the %s reference:" % (nevpt.soc.upper()+"-"+nevpt.method.upper(), nevpt.interface.reference.upper()))
    else:
        nevpt.log.info("\nSummary of results for the %s calculation with the %s reference:" % (nevpt.method.upper(), nevpt.interface.reference.upper()))

    nevpt.log.info("------------------------------------------------------------------------------------------------------------------")
    nevpt.log.info("  State    Degen.        E(total)            dE(a.u.)        dE(eV)      dE(nm)       dE(cm-1)      Osc Str.  ")
    nevpt.log.info("------------------------------------------------------------------------------------------------------------------")

    e_gs = nevpt.e_tot[0]
    e_tot = nevpt.e_tot

    n_states = len(e_tot)

    for p in range(n_states):
        deg = 1
        if not nevpt.soc:
            deg = nevpt.spin_mult[p]
        de = nevpt.e_tot[p] - e_gs
        de_ev = de * h2ev
        de_cm = de * h2cm
        if p == 0 or abs(de) < 1e-5:
            nevpt.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12s %14.4f   %12s" % ((p+1), deg, e_tot[p], de, de_ev, " ", de_cm, " "))
        else:
            de_nm = 10000000 / de_cm
            nevpt.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12.4f %14.4f   %12.8f" % ((p+1), deg, e_tot[p], de, de_ev, de_nm, de_cm, osc_str[p-1]))

    nevpt.log.info("----------------------------------------------------------------------------------------------------------------")

