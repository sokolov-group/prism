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

import numpy as np
import prism.lib.logger as logger

def compute_reference_rdms(nevpt, ref_wfn_list = None, ref_nelecas_list = None):

    if (ref_wfn_list is None or ref_nelecas_list is None):
        ref_wfn_list = nevpt.ref_wfn
        ref_nelecas_list = nevpt.ref_nelecas

    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.extra("Computing reference wavefunction RDMs...")

    # Compute reference-state RDMs
    if nevpt.ncas > 0:
        nevpt.rdm.ca, nevpt.rdm.ccaa, nevpt.rdm.cccaaa, nevpt.rdm.ccccaaaa = nevpt.interface.compute_rdm1234(ref_wfn_list,
                                                                                                                  ref_wfn_list,
                                                                                                                  ref_nelecas_list)
    else:
        nevpt.rdm.ca = np.zeros((nevpt.ncas, nevpt.ncas))
        nevpt.rdm.ccaa =  np.zeros((nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas))
        nevpt.rdm.cccaaa =  np.zeros((nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas))
        mr_adc.rdm.ccccaaaa =  np.zeros((mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas))

    nevpt.log.timer("transforming RDMs", *cput0)