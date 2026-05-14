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
#

import numpy as np
import prism.lib.logger as logger

# Reduced density matrices of the reference wavefunction
def compute_reference_rdms(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.info("\nComputing reference wavefunction RDMs...")

    # Compute reference-state RDMs
    if mr_adc.ncas != 0:
        mr_adc.rdm.ca, mr_adc.rdm.ccaa, mr_adc.rdm.cccaaa, mr_adc.rdm.ccccaaaa = mr_adc.interface.compute_rdm1234(mr_adc.ref_wfn,
                                                                                                                  mr_adc.ref_wfn,
                                                                                                                  mr_adc.ref_nelecas)
    else:
        mr_adc.rdm.ca = np.zeros((mr_adc.ncas, mr_adc.ncas))
        mr_adc.rdm.ccaa =  np.zeros((mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas))
        mr_adc.rdm.cccaaa =  np.zeros((mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas))
        mr_adc.rdm.ccccaaaa =  np.zeros((mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas))

    mr_adc.log.timer("reference RDMs", *cput0)


# Transition reduced density matrices for the IP calculations (e.g., <Psi_0^N|a^+_p a^+_q a_r|Psi_I^N-1>)
def compute_ip_transition_rdms(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.info("\nComputing IP transition RDMs...")

    ncasci = mr_adc.ncasci
    ncas = mr_adc.ncas
    nelecasci = mr_adc.nelecasci
    ref_nelecas = mr_adc.ref_nelecas[0]

    ref_wfn = mr_adc.ref_wfn[0]
    wfn_casci = mr_adc.wfn_casci

    mr_adc.rdm.c_a = np.zeros((ncasci, ncas))
    mr_adc.rdm.cca_aaa = np.zeros((ncasci, ncas, ncas, ncas))
    mr_adc.rdm.cca_abb = np.zeros((ncasci, ncas, ncas, ncas))
    mr_adc.rdm.cccaa_aaaaa = np.zeros((ncasci, ncas, ncas, ncas, ncas, ncas))
    mr_adc.rdm.cccaa_aabab = np.zeros((ncasci, ncas, ncas, ncas, ncas, ncas))
    mr_adc.rdm.cccaa_abbbb = np.zeros((ncasci, ncas, ncas, ncas, ncas, ncas))

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        #TODO
        mr_adc.rdm.ccccaaa =  np.zeros((ncasci, ncas, ncas, ncas, ncas, ncas, ncas, ncas))

    # Compute a_p 
    for p in range(ncas):
        bra, bra_ne = None, None

        if ref_nelecas[0] == (nelecasci[0] + 1) and ref_nelecas[1] == nelecasci[1]:
            bra, bra_ne = mr_adc.interface.act_des_a(ref_wfn, ncas, ref_nelecas, p)
        # TODO: Do we need this case?
##        elif ref_nelecas[0] == nelecasci[0] and ref_nelecas[1] == (nelecasci[1] + 1):
##            bra, bra_ne = mr_adc.interface.act_des_b(ref_wfn, ncas, ref_nelecas, p)
        else:
            raise Exception("IP CASCI states must have N - 1 electrons for a given N-electron reference state")

        for I in range(ncasci):
            mr_adc.rdm.c_a[I, p] = np.dot(bra.reshape(-1), wfn_casci[I].reshape(-1))

            rdm1, rdm2 = mr_adc.interface.trans_rdm12s(bra, wfn_casci[I], ncas, nelecasci)
            mr_adc.rdm.cca_aaa[I, p] = rdm1[0].T
            mr_adc.rdm.cca_abb[I, p] = rdm1[1].T
            mr_adc.rdm.cccaa_aaaaa[I, p] = rdm2[0].transpose(0,2,3,1)
            mr_adc.rdm.cccaa_aabab[I, p] = -rdm2[1].transpose(0,2,1,3) # Transpose to the ABAB order
            mr_adc.rdm.cccaa_abbbb[I, p] = rdm2[3].transpose(0,2,3,1)

####            if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):

    mr_adc.log.timer("IP transition RDMs", *cput0)