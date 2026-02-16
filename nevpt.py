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
import numpy as np

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
        
        self.p1p_shift_type = None                # Possible shift types: real, imaginary, DSRG
        self.m1p_shift_type = None                # Possible shift types: real, imaginary, DSRG
        
        self.p1p_shift_epsilon = None             # Level shift value
        self.m1p_shift_epsilon = None
        
        if self.p1p_shift_epsilon is None:
            self.p1p_shift_epsilon = 0.01         # Default level shift value
            
        if self.m1p_shift_epsilon is None:
            self.m1p_shift_epsilon = 0.01         # Default level shift value
        
        self.S12 = lambda:None                    # Matrices for orthogonalization of excitation spaces
        

        self.outcore_expensive_tensors = True     # Store expensive (ooee) integrals and amplitudes on disk
        self.rdm_order = 2

        # Integrals
        self.mo_energy = lambda:None
        self.h1eff = lambda:None
        self.v2e = lambda:None

        self.mo_energy.c = interface.mo_energy[:self.ncore]
        self.mo_energy.e = interface.mo_energy[self.nocc:]
        
        # Correlated 1rdm
        self.compute_corr_1rdm = False            # Explicitly compute SS-1RDM(s) (multiple for multistate calculations)
        self.compute_trans_corr_1rdm = False      # Explicitly compute transition 1RDMs for multistate calculations
        
    def make_rdm1(self, t1, t1_0, m = None, n = None, evec = None):
        method = self.method
        ncore = self.ncore
        ncas = self.ncas
        nextern = self.nextern
        n_micro_states = sum(self.ref_wfn_deg)
        einsum = self.interface.einsum
        
        einsum_type = self.interface.einsum_type
        nmo = self.nmo

        if m is None:
            m = 0
        
        if n is None:
            n = 0       
        
        if self.method != "qd-nevpt2":
            evec = np.identity(n_micro_states)

        rdm_qd = np.zeros((nmo, nmo))

        # Looping over states I,J
        for I in range(n_micro_states):
            L_t1_caea = t1[I].caea
            L_t1_caae = t1[I].caae
            L_t1_caaa = t1[I].caaa
            L_t1_aaae = t1[I].aaae
            L_t1_ccae = t1[I].ccae
            L_t1_ccaa = t1[I].ccaa
            L_t1_caee = t1[I].caee 
            L_t1_aaee = t1[I].aaee
            L_t1_aaea = t1[I].aaae.transpose(1,0,3,2)

            for J in range(n_micro_states): 
                R_t1_caea = t1[J].caea
                R_t1_caae = t1[J].caae
                R_t1_caaa = t1[J].caaa
                R_t1_aaae = t1[J].aaae
                R_t1_ccae = t1[J].ccae
                R_t1_ccaa = t1[J].ccaa
                R_t1_caee = t1[J].caee 
                R_t1_aaee = t1[J].aaee
                R_t1_aaea = t1[J].aaae.transpose(0,1,3,2)
                
                t1_ccee = t1_0
                
                rdm_mo = np.zeros((nmo, nmo)) 
                trdm_ca, trdm_ccaa, trdm_cccaaa, trdm_ccccaaaa = self.interface.compute_rdm1234(self.ref_wfn[I], self.ref_wfn[J], self.ref_nelecas[I])
                rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] = trdm_ca
                
                if I == J:
                    if self.rdm_order == 2:
                        #uncorrelated diagonal terms
                        rdm_mo[:ncore, :ncore] = 2 * np.identity(ncore)
                    
                        # CORE-CORE #
                        rdm_mo[:ncore, :ncore] -= 4 * einsum('Iiab,Jiab->IJ', t1_ccee, t1_ccee, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += 2 * einsum('Iiab,Jiba->IJ', t1_ccee, t1_ccee, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= 4 * einsum('Iixa,Jixa->IJ', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += 2 * einsum('Iixa,iJxa->IJ', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= 4 * einsum('Iixy,Jixy->IJ', L_t1_ccaa, R_t1_ccaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += 2 * einsum('Iixy,Jiyx->IJ', L_t1_ccaa, R_t1_ccaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += 2 * einsum('iIxa,Jixa->IJ', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= 4 * einsum('iIxa,iJxa->IJ', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += 2 * einsum('Iixa,Jiya,xy->IJ', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= einsum('Iixa,iJya,xy->IJ', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += 2 * einsum('Iixy,Jixz,yz->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= einsum('Iixy,Jiyz,xz->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= einsum('Iixy,Jizw,xyzw->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= einsum('Iixy,Jizx,yz->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += 2 * einsum('Iixy,Jizy,xz->IJ', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= 2 * einsum('Ixab,Jyab,xy->IJ', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += einsum('Ixab,Jyba,xy->IJ', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= 2 * einsum('Ixay,Jzaw,xwyz->IJ', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= 2 * einsum('Ixay,Jzay,xz->IJ', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += einsum('Ixay,Jzwa,xwyz->IJ', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += einsum('Ixay,Jzya,xz->IJ', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += einsum('Ixya,Jzaw,xwyz->IJ', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += einsum('Ixya,Jzay,xz->IJ', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += einsum('Ixya,Jzwa,xwzy->IJ', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= 2 * einsum('Ixya,Jzya,xz->IJ', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvwyz->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvwzy->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvywz->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvyzw->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= 1/3 * einsum('Ixyz,Jwuv,xuvzwy->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += 2/3 * einsum('Ixyz,Jwuv,xuvzyw->IJ', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += einsum('Ixyz,Jwuy,xuzw->IJ', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += einsum('Ixyz,Jwuz,xuwy->IJ', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= 2 * einsum('Ixyz,Jwyu,xuzw->IJ', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= 2 * einsum('Ixyz,Jwyz,xw->IJ', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += einsum('Ixyz,Jwzu,xuyw->IJ', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += einsum('Ixyz,Jwzy,xw->IJ', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] -= einsum('iIxa,Jiya,xy->IJ', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, :ncore] += 2 * einsum('iIxa,iJya,xy->IJ', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                        
                        # ACT-ACT # 
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 4 * einsum('ijXa,ijYa->XY', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 2 * einsum('ijXa,jiYa->XY', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 4 * einsum('ijXx,ijYx->XY', L_t1_ccaa, R_t1_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 2 * einsum('ijXx,jiYx->XY', L_t1_ccaa, R_t1_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Xxab,yzab,Yxyz->XY', L_t1_aaee, R_t1_aaee, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Xxya,zwua,Yxuywz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Xxya,zwya,Yxzw->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Yxab,yzab,Xxyz->XY', L_t1_aaee, R_t1_aaee, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Yxya,zwua,Xxuywz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('Yxya,zwya,Xxzw->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXab,ixab,Yx->XY', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXab,ixba,Yx->XY', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXax,iyax,Yy->XY', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXax,iyaz,Yzxy->XY', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXax,iyxa,Yy->XY', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXax,iyza,Yzxy->XY', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxa,iyax,Yy->XY', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxa,iyaz,Yzxy->XY', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXxa,iyxa,Yy->XY', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxa,iyza,Yzyx->XY', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuxyz->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuxzy->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('iXxy,izwu,Ywuyxz->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuyzx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuzxy->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iXxy,izwu,Ywuzyx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izwx,Ywyz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izwy,Ywzx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXxy,izxw,Ywyz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iXxy,izxy,Yz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izyw,Ywxz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iXxy,izyx,Yz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYab,ixab,Xx->XY', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYab,ixba,Xx->XY', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYax,iyax,Xy->XY', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYax,iyaz,Xzxy->XY', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYax,iyxa,Xy->XY', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYax,iyza,Xzxy->XY', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxa,iyax,Xy->XY', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxa,iyaz,Xzxy->XY', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYxa,iyxa,Xy->XY', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxa,iyza,Xzyx->XY', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuxyz->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuxzy->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('iYxy,izwu,Xwuyxz->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuyzx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuzxy->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('iYxy,izwu,Xwuzyx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izwx,Xwyz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izwy,Xwzx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYxy,izxw,Xwyz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('iYxy,izxy,Xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izyw,Xwxz->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('iYxy,izyx,Xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijXa,ijxa,Yx->XY', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijXa,jixa,Yx->XY', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 2 * einsum('ijXx,ijYy,xy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijXx,ijxy,Yy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijXx,ijyz,Yxyz->XY', L_t1_ccaa, R_t1_ccaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ijXx,jiYy,xy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijXx,jixy,Yy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijYa,ijxa,Xx->XY', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijYa,jixa,Xx->XY', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijYx,ijxy,Xy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/2 * einsum('ijYx,ijyz,Xxyz->XY', L_t1_ccaa, R_t1_ccaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ijYx,jixy,Xy->XY', L_t1_ccaa, R_t1_ccaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixXa,iyYa,xy->XY', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixXa,iyaY,xy->XY', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXa,iyaz,Yyxz->XY', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXa,iyza,Yyzx->XY', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixXy,izYw,yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixXy,izYy,xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixXy,izwY,yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzuwx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzuxw->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzwux->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('ixXy,izwu,Yyzwxu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzxuw->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixXy,izwu,Yyzxwu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXy,izwy,Yzwx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixXy,izyY,xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixXy,izyw,Yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYa,iyaz,Xyxz->XY', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYa,iyza,Xyzx->XY', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzuwx->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzuxw->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzwux->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('ixYy,izwu,Xyzwxu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzxuw->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('ixYy,izwu,Xyzxwu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYy,izwy,Xzwx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixYy,izyw,Xzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixaX,iyYa,xy->XY', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixaX,iyaY,xy->XY', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixaX,iyaz,Yyxz->XY', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixaX,iyza,Yyxz->XY', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixaY,iyaz,Xyxz->XY', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixaY,iyza,Xyxz->XY', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixyX,izYw,yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixyX,izYy,xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= einsum('ixyX,izwY,yzwx->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyX,izwu,Yyzxwu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyX,izwy,Yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 2 * einsum('ixyX,izyY,xz->XY', L_t1_caaa, R_t1_caaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixyX,izyw,Yzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyY,izwu,Xyzxwu->XY', L_t1_caaa, R_t1_caaa, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('ixyY,izwy,Xzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('ixyY,izyw,Xzxw->XY', L_t1_caaa, R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('xXya,zwua,Yxuwyz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuwzy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuywz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuyzw->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuzwy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xXya,zwua,Yxuzyw->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('xXya,zwya,Yxwz->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/3 * einsum('xYya,zwua,Xxuwyz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuwzy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuywz->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuyzw->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuzwy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/6 * einsum('xYya,zwua,Xxuzyw->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/2 * einsum('xYya,zwya,Xxwz->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += einsum('xyXa,zwYa,xyzw->XY', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzuxy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzuyx->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzxuy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('xyXa,zwua,Ywzxyu->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzyux->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyXa,zwua,Ywzyxu->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzuxy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzuyx->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzxuy->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] += 1/3 * einsum('xyYa,zwua,Xwzxyu->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzyux->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, ncore:ncore + ncas] -= 1/6 * einsum('xyYa,zwua,Xwzyxu->XY', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        
                        # EXT-EXT #
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 4 * einsum('ijAa,ijBa->AB', t1_ccee, t1_ccee, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 2 * einsum('ijAa,jiBa->AB', t1_ccee, t1_ccee, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 4 * einsum('ijxA,ijxB->AB', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 2 * einsum('ijxA,jixB->AB', L_t1_ccae, R_t1_ccae, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 2 * einsum('ijxA,ijyB,xy->AB', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += einsum('ijxA,jiyB,xy->AB', L_t1_ccae, R_t1_ccae, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixAa,iyBa,xy->AB', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixAa,iyaB,xy->AB', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixAy,izBw,yzxw->AB', L_t1_caea, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixAy,izBy,xz->AB', L_t1_caea, R_t1_caea, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixAy,izwB,yzxw->AB', L_t1_caea, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixAy,izyB,xz->AB', L_t1_caea, R_t1_caae, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixaA,iyBa,xy->AB', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixaA,iyaB,xy->AB', L_t1_caee, R_t1_caee, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixyA,izBw,yzxw->AB', L_t1_caae, R_t1_caea, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixyA,izBy,xz->AB', L_t1_caae, R_t1_caea, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= einsum('ixyA,izwB,yzwx->AB', L_t1_caae, R_t1_caae, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2 * einsum('ixyA,izyB,xz->AB', L_t1_caae, R_t1_caae, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += einsum('xyAa,zwBa,xyzw->AB', L_t1_aaee, R_t1_aaee, trdm_ccaa, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwvxy->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwvyx->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwxvy->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += 2/3 * einsum('xyzA,wuvB,zuwxyv->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwyvx->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] -= 1/3 * einsum('xyzA,wuvB,zuwyxv->AB', L_t1_aaae, R_t1_aaae, trdm_cccaaa, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore + ncas:ncore + ncas + nextern] += einsum('xyzA,wuzB,yxuw->AB', L_t1_aaae, R_t1_aaae, trdm_ccaa, optimize = einsum_type)

                    rdm_qd += np.conj(evec)[I, n] * rdm_mo * evec[J, m]
                    
                else:
                    if self.rdm_order == 2:
                        # OFF-DIAGS #
                        # COR-ACT #
                        rdm_mo[:ncore, ncore:ncore + ncas] += einsum('IxXy,yx->IX', R_t1_caaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, ncore:ncore + ncas] -= 1/2 * einsum('IxyX,yx->IX', R_t1_caaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, ncore:ncore + ncas] -= 1/2 * einsum('Ixyz,yzXx->IX', R_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        
                        # ACT-COR #
                        rdm_mo[ncore:ncore + ncas, :ncore] += einsum('IxXy,xy->XI', L_t1_caaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, :ncore] -= 1/2 * einsum('IxyX,xy->XI', L_t1_caaa, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore:ncore + ncas, :ncore] -= 1/2 * einsum('Ixyz,Xxyz->XI', L_t1_caaa, trdm_ccaa, optimize = einsum_type)
                        
                        # COR-EXT #
                        rdm_mo[:ncore, ncore + ncas:ncore + ncas + nextern] += einsum('IxAy,yx->IA', R_t1_caea, trdm_ca, optimize = einsum_type)
                        rdm_mo[:ncore, ncore + ncas:ncore + ncas + nextern] -= 1/2 * einsum('IxyA,yx->IA', R_t1_caae, trdm_ca, optimize = einsum_type)
                        
                        # EXT-COR #
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, :ncore] += einsum('IxAy,xy->AI', L_t1_caea, trdm_ca, optimize = einsum_type)
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, :ncore] -= 1/2 * einsum('IxyA,xy->AI', L_t1_caae, trdm_ca, optimize = einsum_type)
                        
                        # ACT-EXT #
                        rdm_mo[ncore:ncore + ncas, ncore + ncas:ncore + ncas + nextern] += 1/2 * einsum('xyzA,Xzyx->XA', R_t1_aaae, trdm_ccaa, optimize = einsum_type)
                        
                        # EXT-ACT #
                        rdm_mo[ncore + ncas:ncore + ncas + nextern, ncore:ncore + ncas] += 1/2 * einsum('xyzA,Xzyx->AX', L_t1_aaae, trdm_ccaa, optimize = einsum_type)
                    
                    rdm_qd += np.conj(evec)[I, n] * rdm_mo * evec[J, m]

        #self.rdm1 = rdm1
        return rdm_qd
    
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
        
        avail_shifts = ['real', 'imaginary', 'DSRG']
        
        if self.m1p_shift_type is not None and self.m1p_shift_type not in avail_shifts:
            raise ValueError(f"Invalid {'m1p_shift_type'}: '{self.m1p_shift_type}'. Available options are {avail_shifts}.")

        if self.p1p_shift_type is not None and self.p1p_shift_type not in avail_shifts:
            raise ValueError(f"Invalid {'p1p_shift_type'}: '{self.p1p_shift_type}'. Available options are {avail_shifts}.")
        
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

        return e_tot, e_corr, osc

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self, obj):
        self._verbose = obj
        self.log.verbose = obj
