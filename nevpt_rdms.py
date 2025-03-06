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

def compute_reference_rdms(nevpt):

    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.info("\nComputing ground-state RDMs...")

    # Compute reference-state RDMs
    if nevpt.ncas != 0:
        nevpt.rdm.ca, nevpt.rdm.ccaa, nevpt.rdm.cccaaa, nevpt.rdm.ccccaaaa = nevpt.interface.compute_rdm1234(nevpt.wfn_casscf,
                                                                                                                  nevpt.wfn_casscf,
                                                                                                                  nevpt.nelecas)
    else:
        nevpt.rdm.ca = np.zeros((nevpt.ncas, nevpt.ncas))
        nevpt.rdm.ccaa =  np.zeros((nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas))
        nevpt.rdm.cccaaa =  np.zeros((nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas, nevpt.ncas))

    nevpt.log.timer("transforming RDMs", *cput0)