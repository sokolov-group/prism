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

from prism.nevpt import integrals
from prism.nevpt import compute
from prism.nevpt import nevpt2
from prism.nevpt import qd_nevpt2

def kernel(self):

    log = self.log
    self.method = self.method.lower()

    if self.method == "qdnevpt2":
        self.method = "qd-nevpt2"

    if self.method not in ("nevpt2", "qd-nevpt2"):
        msg = "Unknown method %s" % self.method
        log.info(msg)
        raise Exception(msg)

    if self.nfrozen is None:
        self.nfrozen = 0

    if self.nfrozen > self.ncore:
        msg = "The number of frozen orbitals cannot exceed the number of core orbitals"
        log.error(msg)
        raise Exception(msg)
    
    if self.rdm_order not in [0,2]:
         raise ValueError(f"Invalid {'rdm_order'}: '{self.rdm_order}'. Available options are {0,2}.")
     
    avail_shifts = ['imaginary', 'DSRG']
    
    if self.shift_type_m1p is not None and self.shift_type_m1p not in avail_shifts:
        raise ValueError(f"Invalid {'shift_type_m1p'}: '{self.shift_type_m1p}'. Available options are {avail_shifts}.")

    if self.shift_type_p1p is not None and self.shift_type_p1p not in avail_shifts:
        raise ValueError(f"Invalid {'shift_type_p1p'}: '{self.shift_type_p1p}'. Available options are {avail_shifts}.")
    
    if self.shift_type_0p is not None and self.shift_type_0p not in avail_shifts:
        raise ValueError(f"Invalid {'shift_type_0p'}: '{self.shift_type_0p}'. Available options are {avail_shifts}.")
    
    # Transform one- and two-electron integrals
    log.info("\nTransforming integrals to MO basis...")
    integrals.transform_integrals_1e(self)
    if self.interface.with_df:
        integrals.transform_Heff_integrals_2e_df(self)
        integrals.transform_integrals_2e_df(self)
    else:
        # TODO: this actually handles out-of-core integrals too, rename the function
        integrals.transform_integrals_2e_incore(self)

    # Run NEVPT computation
    e_tot, e_corr, osc = compute.kernel(self)
    self.en_tot = e_tot

    if self.keep_amplitudes is False:
        del(self.t1)
        del(self.t1_0)
        

    #Calculate SOC properties
    if self.soc: 
      from prism import general_somf
      import numpy as np
      print("\nInitialize SOC program...")
      # Rotate CAS Wavefunction:
      if self.method == "qd-nevpt2":
          wfn = np.einsum('ij,iab->jab',self.h_evec,self.ref_wfn)
          wfn = list(wfn)
      else:
          wfn = list(self.ref_wfn)

      # Calculate method's S, Ms: 
      S  = []
      ms = []
      nstate = len(self.en_tot)
      for I in range(nstate):
        sz = self.interface.apply_S_z(wfn[I],self.ncas,self.ref_nelecas[I])
        ms.append(np.dot(wfn[I].ravel(), sz.ravel()))

        SS = self.interface.compute_spin_square(wfn[I], self.ncas, self.ref_nelecas[I])
        S.append((-1+np.sqrt(1+4*SS))/2)

      ms = [round(elem,2) for elem in ms]
      S  = [round(elem,2) for elem in S]

      # Calculate RDM_aabb
      rdm_aabb = nevpt2.make_rdm1s(self)

      en_soc, evec_soc, osc_str_soc = general_somf.state_interaction_SOC(self, self.en_tot, rdm_aabb, S, ms)
      
      if self.gtensor is True:
        rdm_sf = rdm_aabb[0] + rdm_aabb[1]
        self.g_factor, G_evec = general_somf.gtensor(self,evec_soc,rdm_sf, S, target_index = self.target_index, origin_type=self.origin_type)
      
      #Calculate SOC e_corr respect with CASSCF energy
      e_ref_spinstate = []
      for i in range(nstate):
        n = int(S[i]*2 + 1)
        for j in range(n):
            e_ref_spinstate.append(self.e_ref[i])
    
      e_corr_soc = en_soc - e_ref_spinstate

      (e_tot, e_corr, osc) = (en_soc, e_corr_soc, osc_str_soc)

    
    return e_tot, e_corr, osc 

@property
def verbose(self):
    return self._verbose
@verbose.setter
def verbose(self, obj):
    self._verbose = obj
    self.log.verbose = obj
