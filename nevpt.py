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

import prism.nevpt_integrals as nevpt_integrals
import prism.nevpt_compute as nevpt_compute
import prism.nevpt2 as nevpt2
import prism.qd_nevpt2 as qdnevpt2

class NEVPT:
    def __init__(self, interface):

        # General info
        self.interface = interface
        self.log = interface.log
        log = self.log

        log.info("Initializing state-specific fully internally contracted NEVPT...")

        if (interface.reference not in ("casscf", "casci", "sa-casscf", "ms-casci")):
            log.info("The NEVPT code does not support %s reference" % interface.reference)
            raise Exception("The NEVPT code does not support %s reference" % interface.reference)

        self.stdout = interface.stdout
        self.verbose = interface.verbose
        self.max_memory = interface.max_memory
        self.current_memory = interface.current_memory

        self.temp_dir = interface.temp_dir
        self.tmpfile = lambda:None

        self.mo = interface.mo
        self.mo_scf = interface.mo_scf
        self.ovlp = interface.ovlp
        self.nmo = interface.nmo
        self.nelec = interface.nelec
        self.enuc = interface.enuc
        self.e_scf = interface.e_scf

        self.symmetry = interface.symmetry
        self.group_repr_symm = interface.group_repr_symm

        # CASSCF specific
        self.ncore = interface.ncore
        self.ncas = interface.ncas
        self.nextern = interface.nextern
        self.nocc = self.ncas + self.ncore
        self.ref_nelecas = interface.ref_nelecas
        self.e_ref = interface.e_ref              # Total reference energy
        self.e_ref_cas = interface.e_ref_cas      # Reference active-space energy
        self.ref_wfn = interface.ref_wfn          # Reference wavefunction
        self.ref_wfn_spin_mult = interface.ref_wfn_spin_mult
        self.ref_wfn_deg = interface.ref_wfn_deg

        # NEVPT specific variables
        self.method = "nevpt2"                    # Possible methods: nevpt2, qd-nevpt2 (qdnevpt2)
        self.nfrozen = None                       # Number of lowest-energy (core) orbitals that will be left uncorrelated ("frozen core")
        self.compute_singles_amplitudes = False   # Include singles amplitudes in the NEVPT2 energy?
        self.semi_internal_projector = "gno"      # Possible values: gno, gs, only matters when compute_singles_amplitudes is True
        self.s_thresh_singles = 1e-8
        self.s_thresh_doubles = 1e-8
        
        self.shift_type_p1p = None                # Possible shift types: imaginary, DSRG
        self.shift_type_m1p = None                
        self.shift_type_0p = None
        
        self.shift_epsilon = 0.01                 # Level shift value, default 0.01 Hartree

        self.S12 = lambda:None                    # Matrices for orthogonalization of excitation spaces
        
        self.outcore_expensive_tensors = True     # Store expensive (ooee) integrals and amplitudes on disk   

        # Integrals
        self.mo_energy = lambda:None
        self.h1eff = lambda:None
        self.v2e = lambda:None

        self.mo_energy.c = interface.mo_energy[:self.ncore]
        self.mo_energy.e = interface.mo_energy[self.nocc:]
        
        # Correlated 1rdm
        self.rdm_order = 0                         # Default value of 0 (uncorrelated), 2 for correlated
        
        # Amplitudes
        self.t1 = None
        self.t1_0 = None 
        self.keep_amplitudes = False
        
        #Eigenvectors
        self.h_evec = None
        
        if self.method == "nevpt2":
            make_rdm1 = nevpt2.make_rdm1
        else:
            make_rdm1 = qdnevpt2.make_rdm1

        self.make_rdm1 = lambda *args, **kwargs: make_rdm1(self, *args, **kwargs)
            
        #For SOC
        self.en_tot = None 
        self.gtensor = False
        self.soc = None  # Possible methods: Breit-Pauli (BP), DKH1 (x2c-1)
        self.origin_type = 'charge'  # Possible methods: charge, GIAO, atom1 or User define point(list)
        self.target_index = 0  #target state for gtensor calculation. Default is the ground state.



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
        nevpt_integrals.transform_integrals_1e(self)
        if self.interface.with_df:
            nevpt_integrals.transform_Heff_integrals_2e_df(self)
            nevpt_integrals.transform_integrals_2e_df(self)
        else:
            # TODO: this actually handles out-of-core integrals too, rename the function
            nevpt_integrals.transform_integrals_2e_incore(self)

        # Run NEVPT computation
        e_tot, e_corr, osc = nevpt_compute.kernel(self)
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
