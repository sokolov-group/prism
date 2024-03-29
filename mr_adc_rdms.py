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

import numpy as np
import prism.lib.logger as logger

def compute_gs_rdms(mr_adc):

    cput0 = (logger.process_clock(), logger.perf_counter())
    mr_adc.log.info("\nComputing ground-state RDMs...")

    # TODO: for open-shells, this needs to perform state-averaging
    # Compute ground-state RDMs
    if mr_adc.ncas != 0:
        mr_adc.rdm.ca, mr_adc.rdm.ccaa, mr_adc.rdm.cccaaa, mr_adc.rdm.ccccaaaa = mr_adc.interface.compute_rdm1234(mr_adc.wfn_casscf,
                                                                                                                  mr_adc.wfn_casscf,
                                                                                                                  mr_adc.nelecas)
    else:
        mr_adc.rdm.ca = np.zeros((mr_adc.ncas, mr_adc.ncas))
        mr_adc.rdm.ccaa =  np.zeros((mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas))
        mr_adc.rdm.cccaaa =  np.zeros((mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas))

    mr_adc.log.timer("transforming RDMs", *cput0)