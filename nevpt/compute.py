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

import numpy as np
from prism.nevpt import integrals
from prism.tools import trans_prop
import prism.lib.logger as logger

def kernel(nevpt):

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
    nevpt.compute_properties()

    # Print results
    print_results(nevpt)

    if nevpt.soc:
        nevpt.log.timer0("total %s calculation" % ("SOC-"+nevpt.method.upper()), *cput0)
    else:
        nevpt.log.timer0("total %s calculation" % nevpt.method.upper(), *cput0)

    osc_str = None
    if "osc_strengths" in nevpt.properties:
        osc_str = nevpt.properties["osc_strengths"]

    return nevpt.e_tot, nevpt.e_corr, osc_str


def initialize(nevpt):

    # Initial checks
    nevpt.method = nevpt.method.lower()
    nevpt.method_type = nevpt.method_type.lower()
    if nevpt.soc is not None:
        nevpt.soc = nevpt.soc.lower()

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
    nevpt.log.info("Method type:                                       %s" % nevpt.method_type)
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
    nevpt.log.info("Spin–orbit coupling:                               %s" % str(nevpt.soc))
    nevpt.log.info("G-tensor:                                          %s" % str(nevpt.gtensor))
    if nevpt.gtensor:
        nevpt.log.info("G-tensor origin:                                   %s" % str(nevpt.gtensor_origin_type))
        nevpt.log.info("G-tensor target state:                             %s" % str(nevpt.gtensor_target_state))

    if nevpt.shift_type_p1p is not None:
        nevpt.log.info("Level shift [+1']:                                 %s" % str(nevpt.shift_type_p1p))
    if nevpt.shift_type_m1p is not None:
        nevpt.log.info("Level shift [-1']:                                 %s" % str(nevpt.shift_type_m1p))
    if nevpt.shift_type_0p is not None:
        nevpt.log.info("Level shift [0']:                                  %s" % str(nevpt.shift_type_0p))
    if nevpt.shift_type_0p is not None or nevpt.shift_type_m1p is not None or nevpt.shift_type_p1p is not None:
        nevpt.log.info("Level shift value (a.u.):                          %s" % nevpt.shift_epsilon)

    nevpt.log.info("Reference density fitting?                         %s" % ref_df)
    nevpt.log.info("Correlation density fitting?                       %s" % df)
    nevpt.log.info("Temporary directory path:                          %s" % nevpt.temp_dir)

    nevpt.log.info("\nInternal contraction:                              %s" % "Full (= Partial)")
    nevpt.log.info("Compute singles amplitudes?                        %s" % str(nevpt.compute_singles_amplitudes))
    nevpt.log.info("Overlap truncation parameter (singles):            %e" % nevpt.s_thresh_singles)
    nevpt.log.info("Overlap truncation parameter (doubles):            %e" % nevpt.s_thresh_doubles)
    if nevpt.compute_singles_amplitudes:
        nevpt.log.info("Projector for the semi-internal amplitudes:        %s" % nevpt.semi_internal_projector)


def print_results(nevpt):

    h2ev = nevpt.interface.hartree_to_ev
    h2cm = nevpt.interface.hartree_to_inv_cm

    if nevpt.soc:
        nevpt.log.info("\nSummary of results for the %s calculation with the %s reference:" % (nevpt.soc.upper()+"-"+nevpt.method_type.upper()+"-"+nevpt.method.upper(), nevpt.interface.reference.upper()))
    else:
        nevpt.log.info("\nSummary of results for the %s calculation with the %s reference:" % (nevpt.method_type.upper()+"-"+nevpt.method.upper(), nevpt.interface.reference.upper()))

    nevpt.log.info("------------------------------------------------------------------------------------------------------------------")
    nevpt.log.info("  State    Degen.        E(total)            dE(a.u.)        dE(eV)      dE(nm)       dE(cm-1)      Osc Str.  ")
    nevpt.log.info("------------------------------------------------------------------------------------------------------------------")

    e_gs = nevpt.e_tot[0]
    e_tot = nevpt.e_tot

    n_states = len(e_tot)

    osc_str = None
    if n_states > 1:
        osc_str = nevpt.properties["osc_strengths"]

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

    if "osc_strengths_full" in nevpt.properties:
        trans_prop.print_osc_strength(nevpt.interface, nevpt.properties["osc_strengths_full"])

    if "g-eigenvectors" in nevpt.properties:
        G_evecs = nevpt.properties["g-eigenvectors"]
        nevpt.interface.log.info("\nMagnetic g-tensor principal axes:")
        for G_evec in G_evecs:
            nevpt.interface.log.info("%s", np.array2string(G_evec, precision=6, suppress_small=True))
    if "g-factors" in nevpt.properties:
        ge = nevpt.interface.g_free_elec
        G_sq = nevpt.properties["g-factors"]
        nevpt.interface.log.info("\nMagnetic g-factors (ge = %s):" % ge)
        for G_sq_en in G_sq:
            nevpt.interface.log.info("%14.6f, %14.6f, %14.6f" % (G_sq_en[0], G_sq_en[1], G_sq_en[2]))
            nevpt.interface.log.info("%14.6f, %14.6f, %14.6f (g-shift)" % (G_sq_en[0] - ge, G_sq_en[1] - ge, G_sq_en[2] - ge))
            nevpt.interface.log.info("%14.3f, %14.3f, %14.3f (g-shift, ppt)" % (1000 * (G_sq_en[0] - ge), 1000 * (G_sq_en[1] - ge), 1000 * (G_sq_en[2] - ge)))

