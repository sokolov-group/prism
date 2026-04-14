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

def compute_reference_rdms(nevpt, ref_wfn_list = None, ref_nelecas_list = None):

    rdm = lambda:None

    if (ref_wfn_list is None or ref_nelecas_list is None):
        ref_wfn_list = nevpt.ref_wfn
        ref_nelecas_list = nevpt.ref_nelecas

    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.extra("Computing reference wavefunction RDMs...")

    # Compute reference-state RDMs
    if nevpt.ncas > 0:
        rdm.ca, rdm.ccaa, rdm.cccaaa, rdm.ccccaaaa = nevpt.interface.compute_rdm1234(ref_wfn_list,
                                                                                                                  ref_wfn_list,
                                                                                                                  ref_nelecas_list)
    else:
        rdm.ca = np.zeros((nevpt.ncas, nevpt.ncas))
        rdm.ccaa =  np.zeros((nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas))
        rdm.cccaaa =  np.zeros((nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas))
        mr_adc.rdm.ccccaaaa =  np.zeros((mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas))

    nevpt.log.timer("transforming RDMs", *cput0)

    return rdm


def compute_reference_rdms_1s(nevpt, ref_wfn_list = None, ref_nelecas_list = None):

    rdm_so = lambda:None
    rdm_so.ca = np.zeros((2 * nevpt.ncas, 2 * nevpt.ncas))
    rdm_so.ccaa = np.zeros((2 * nevpt.ncas, 2 * nevpt.ncas, 2 * nevpt.ncas, 2 * nevpt.ncas))

    if (ref_wfn_list is None or ref_nelecas_list is None):
        ref_wfn_list = nevpt.ref_wfn
        ref_nelecas_list = nevpt.ref_nelecas

    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.extra("Computing reference wavefunction RDMs...")

    # Compute reference-state RDMs
    if nevpt.ncas > 0:
        ref_nelecas_list = ref_nelecas_list[0]
        print("ref_nelecas_list=",ref_nelecas_list)
        rdm1, rdm2  = nevpt.interface.trans_rdm12s(ref_wfn_list, ref_wfn_list, nevpt.ncas, ref_nelecas_list)
        #rdm1a, rdm1b = nevpt.interface.trans_rdm1s(ref_wfn_list, ref_wfn_list, nevpt.ncas, ref_nelecas_list)
        #rdm_aaaa, rdm_abab, rdm_bbbb, rdm_bbab, rdm_abaa, rdm_aaab, rdm_abbb, rdm_bbaa, rdm_aabb = nevpt.interface.compute_rdm_ccaa_si(ref_wfn_list, ref_wfn_list, [(2,1)])#ref_nelecas_list)
        #rdm.ca = (rdm1a, rdm1b) 
        #rdm.ccaa = (rdm_aaaa, rdm_abab, rdm_bbbb, rdm_bbab, rdm_abaa, rdm_aaab, rdm_abbb, rdm_bbaa, rdm_aabb)


    else:
        raise Exception("Not consider about ncas=0 in 2nd soc")
        #rdm.ca = np.zeros((nevpt.ncas, nevpt.ncas))
        #rdm.ccaa =  np.zeros((nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas))
        #rdm.cccaaa =  np.zeros((nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas))
        #mr_adc.rdm.ccccaaaa =  np.zeros((mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas))

    #arrange rdm
    #1st rdm
    rdm_aa, rdm_bb =  rdm1
    #AA
    rdm_so.ca[::2,::2] = rdm_aa.copy()
    #BB
    rdm_so.ca[::2,::2] = rdm_bb.copy()

    #2nd rdm
    rdm_aaaa, rdm_abab, rdm_ba, rdm_bbbb = rdm2
    # AAAA
    rdm_aaaa = rdm_aaaa.transpose(0, 2, 3, 1).copy() # <p+q+rs> (AAAA)
    rdm_so.ccaa[::2,::2,::2,::2] = rdm_aaaa
    # ABAB
    rdm_abab = rdm_abab.transpose(0, 2, 3, 1).copy() # <p+q+rs> (ABBA)
    rdm_so.ccaa[::2,1::2,1::2,::2] = rdm_abab
    rdm_so.ccaa[1::2,::2,1::2,::2] = -rdm_abab.transpose(1,0,2,3)
    rdm_so.ccaa[::2,1::2,::2,1::2] = -rdm_abab.transpose(0,1,3,2)
    rdm_so.ccaa[1::2,::2,::2,1::2] = rdm_abab.transpose(1,0,3,2)
    # BBBB
    rdm_bbbb = rdm_bbbb.transpose(0, 2, 3, 1).copy() # <p+q+rs> (BBBB)
    rdm_so.ccaa[1::2,1::2,1::2,1::2] = rdm_bbbb
    


    nevpt.log.timer("transforming RDMs", *cput0)
    return rdm_so