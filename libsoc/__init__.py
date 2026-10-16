import sys
import os
import numpy as np
from prism.libsoc import general_somf
from prism.libsoc import magnetic
from prism.tools import trans_prop
import pyscf.dft.LebedevGrid
import csv
import pickle
import h5py


class SF_OBJECT:
    def __init__(self, mol, E_ref, rdm_aabb_ref, S_ref, mo, ms_ref=None):
        import prism.lib.logger as logger
        self.stdout = mol.stdout
        self.verbose = 4
        log = logger.Logger(self.stdout, self.verbose)
        self.log = log
        self.log.info("\nImporting spin-free object...")
        from pyscf.tools import molden
        self.molden = molden

        # General info
        self.nmo = len(mo)  
        self.mol = mol
        # Unit conversions
        self.hartree_to_ev = 27.2113862459817
        self.hartree_to_inv_cm = 219474.63136314
        # Constants
        from pyscf import lib
        self.light_speed = lib.parameters.LIGHT_SPEED
        self.g_free_elec = 2.002319
        self.kb = 1.3806483e-23 / 4.3597447222060e-18 #(Eh/K)
        self.mu_B_Eh =  5.7883817982e-5 / self.hartree_to_ev  #Bohr magneton(Eh/T)
        self.mu_B_erg = 9.27401549e-21
        self.T_to_G = 10000
        self.NA = 6.0221367e23 # Avogadro constant

        # Dipole moments
        self.dip_mom_ao = mol.intor_symmetric("int1e_r", comp = 3)

        #spin-free information
        self.deg_atol = 1e-5
        self.roots = len(E_ref)
        self.nmo = len(rdm_aabb_ref[0,0,0,0])
        self.mo = mo
        self.E_ref = E_ref
        self.rdm_aabb_ref = rdm_aabb_ref
        self.S_ref = S_ref
        self.ms_ref = ms_ref
        self.ms_ref_select = None
        self.analyze_evec_soc = True

        #create soc flag
        from prism.interface import PYSCF
        PYSCF.create_soc_flag(self)


        #Calculating result
        self.properties = {}  


    def kernel(self):
        
        if self.ms_ref is None:
            self.log.info("ms_ref is None. Obtaining ms_ref from rdm_aabb_ref...")
            self.rotate_rdm1s_to_ms()
            sys.stdout.flush()
            self.E_ref, self.rdm_aabb_ref, self.S_ref,  self.ms_ref = self.get_sf_ref()
            sys.stdout.flush()

        en_soc = self.compute_soc(self.E_ref, self.rdm_aabb_ref, self.S_ref,  self.ms_ref, self.soc)

        return en_soc, self.properties


    def compute_soc(self,E_ref, rdm_aabb_ref, S_ref,  ms_ref, soc):

        #Compute en_soc
        en_soc, evec_soc = general_somf.state_interaction_soc(self, E_ref, rdm_aabb_ref, S_ref,  ms_ref, soc)
 
        #Compute OSC
        #compute soc rdm
        rdm_sf = rdm_aabb_ref[0] + rdm_aabb_ref[1]
        I_total = []
        S_total = []
        ms_total = []
        for i in range(len(S_ref)):
            s = S_ref[i]
            n = int(2*s+1)
            for j in range(n):
                I_total.append(i)
                S_total.append(s)
                m = s-j
                ms_total.append(m)
    
        nstate_total = len(S_total)
        rdm_mo = np.zeros((nstate_total, nstate_total, self.nmo, self.nmo),dtype='complex')
        for I in range(nstate_total):
            for J in  range(nstate_total):
                if (np.abs(S_total[I]-S_total[J])<1e-8) and (np.abs(ms_total[I]-ms_total[J])<1e-8):
                    rdm_mo[I,J] = rdm_sf[I_total[I], I_total[J]]
        rdm_soc = np.einsum('ai,ibIJ,bj->ajIJ',np.conj(evec_soc).T , rdm_mo , evec_soc)

        # Calculate ground state degeneracy:
        deg_gs = 1
        for i in range(len(en_soc)-1):
            if (np.abs(en_soc[i+1] - en_soc[0])) < 1e-5:
                deg_gs += 1
            else:
                break
    

        # Calculate oscillator strengths for transitions from the first state
        osc_str_soc_full=[]
        osc_str_soc = np.zeros(len(en_soc)-1)
        for gs_index in range(deg_gs):
            e_diff = en_soc - en_soc[gs_index]
            e_diff = e_diff[gs_index+1:]
            osc = trans_prop.osc_strength(self, e_diff, rdm_soc[ gs_index, gs_index+1:])
            osc_str_soc_full.append(osc)
            osc_str_soc[gs_index:] += osc 

        self.properties["osc_strengths"] = osc_str_soc

        # Compute oscillator strengths starting from each state
        if self.verbose >= 5:
            for gs_index in range(deg_gs, len(en_soc)):  
                e_diff = en_soc - en_soc[gs_index]
                e_diff = e_diff[gs_index+1:]
                osc_str_soc_full.append(trans_prop.osc_strength(self, e_diff, rdm_mo[  gs_index, gs_index+1:]))

            self.properties["osc_strengths_full"] = osc_str_soc_full


        if self.analyze_evec_soc:
            population = evec_soc * np.conj(evec_soc)
            population = np.real(population)
            I = 0
            population_state = np.zeros((len(S_ref), len(evec_soc)))
            for i in range(len(S_ref)):
                multicity = int(S_ref[i]*2 +1)
                for j in range(multicity):
                    J = I + j
                    population_state[i] += population[J]
                I += multicity


            if len(population_state[:,0]) < 5:
                k = len(population_state[:,0])
            else:
                k = 5
            # Step 1: each column find large row index
            row_idx = np.argpartition(population_state, -k, axis=0)[-k:, :]
            # Step 2: value according to row index
            values = population_state[row_idx, np.arange(population_state.shape[1])]
            # Step 3: large to small order 
            order = np.argsort(-values, axis=0)
            # Step 4: after order row index
            sorted_row_idx = np.take_along_axis(row_idx, order, axis=0)
            values_sort = population_state[sorted_row_idx, np.arange(population_state.shape[1])]

            self.log.info("\nSOC index = [sf-index index]")
            for i in range(len(sorted_row_idx[0])):
                index_value = np.zeros((2,len(sorted_row_idx[:,i])))
                index_value[0] = sorted_row_idx[:,i] + 1
                index_value[1] = np.round(values_sort[:,i],2)
                self.log.info("%d = %s %s", i + 1, index_value[0], index_value[1])
            self.log.info("")

        #Compute magnetic properties
        if (self.gtensor or self.mag_av or  self.sus_av or  self.mag_vec or self.sus_tensor):
            properties_mag = magnetic.compute_properties(self, rdm_sf, en_soc, evec_soc, S_ref,  method = None)
            self.properties.update(properties_mag)

        self.print_results(en_soc,soc, osc_str_soc)

        if "osc_strengths_full" in self.properties:
            trans_prop.print_osc_strength(self, self.properties["osc_strengths_full"])

        if "magnetic_properties" in self.properties:
            magnetic.print_mag_properties(self, self.properties,  method = None)

        return en_soc


    def print_results(self, en_soc, soc, osc_str_soc):

        h2ev = self.hartree_to_ev
        h2cm = self.hartree_to_inv_cm

        self.log.info("\nSummary of SOC results for %s", soc.upper())
        self.log.info("------------------------------------------------------------------------------------------------------------------")
        self.log.info("  State    Degen.        E(total)            dE(a.u.)        dE(eV)      dE(nm)       dE(cm-1)      Osc Str.  ")
        self.log.info("------------------------------------------------------------------------------------------------------------------")

        e_gs = en_soc[0]
        e_tot = en_soc

        n_states = len(e_tot)

        for p in range(n_states):
            deg = 1
            de = e_tot[p] - e_gs
            de_ev = de * h2ev
            de_cm = de * h2cm
            if p == 0 or abs(de) < 1e-5:
                self.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12s %14.4f   %12s" % ((p+1), deg, e_tot[p], de, de_ev, " ", de_cm, " "))
            else:
                de_nm = 10000000 / de_cm
                self.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12.4f %14.4f    %12.8f" % ((p+1), deg, e_tot[p], de, de_ev, de_nm, de_cm, osc_str_soc[p-1]))

        self.log.info("----------------------------------------------------------------------------------------------------------------")


    def rotate_rdm1s_to_ms(self):
        '''
        Obtain Ms egienvector and their rdm_1s
        '''
        self.log.info("\n*****rotate_rdm1s_to_ms*****")
        #TO DO: make sure nmo_b is always equal nmo_a
        rdm_aabb = self.rdm_aabb_ref
        S_old_total = self.S_ref
        E_old_total = self.E_ref
        n_state = len(E_old_total)

        #Calculating ms by using rdm_aabb
        Sz_en_tot = np.zeros(n_state)
        Sz_evec_total = np.zeros((n_state,n_state))
        deg_list = self.degeneracy_list(E_old_total,  atol= self.deg_atol)
        self.log.info("deg_list=%s",deg_list)

        Ms_total = np.zeros((n_state,n_state))
        for I in range(n_state):
            for J in range(n_state):
                rdm_sz = 1/2 * ( rdm_aabb[0,I,J] - rdm_aabb[1,I,J] )
                sz = np.trace(rdm_sz)
                Ms_total[I,J] = sz

        self.log.extra("Ms_old_total=")
        self.log.extra("%s", Ms_total)

        for K in range(len(deg_list)):
            I = int(np.sum(deg_list[0:K]))
            J = I + deg_list[K]
            Ms = Ms_total[I:J, I:J]
            sz_en, sz_evec = np.linalg.eigh(Ms)
            Sz_evec_total[I:J,I:J] = sz_evec
            Sz_en_tot[I:J] = sz_en

        self.log.extra("sz_en_tot=")
        self.log.extra("%s", Sz_en_tot)

        self.log.extra("Sz_evec_total=")
        self.log.extra("%s", Sz_evec_total)

        self.rdm_aabb_ref = np.einsum('ai,KibIJ,bj->KajIJ',np.conj(Sz_evec_total).T , rdm_aabb , Sz_evec_total)

        S_total = np.einsum('ai,ib,bj->aj',np.conj(Sz_evec_total).T , np.diag(S_old_total) , Sz_evec_total)
        self.log.extra("Make sure S_total:")
        self.log.extra("%s", S_total)
        if not np.allclose(S_total, np.diag(np.diagonal(S_total)), atol=1e-7, rtol=0):
            self.log.warn("Warning: S_total is not diagonal")
        else:
            self.log.info("S_total is diagonal")

        self.S_ref = np.diagonal(S_total)

        E_new = np.einsum('ai,ib,bj->aj',np.conj(Sz_evec_total).T , np.diag(E_old_total) , Sz_evec_total)
        self.log.extra("Make sure E:")
        self.log.extra("%s", E_new)
        E_new = np.diagonal(E_new)
        if np.allclose(E_old_total, E_new, atol=1e-5, rtol=0):
            self.log.info("Energy is same after rotation")
            self.E_ref = E_new
        else:
            raise Exception("E_new change energy")

        return self.rdm_aabb_ref
    

    def get_sf_ref(self):
        '''
        select specifc Ms state for state-interaction calculation
        '''
        self.log.info("\n*****get_sf_ref*****")
        rdm_aabb = self.rdm_aabb_ref
        E = self.E_ref
        n_state = len(E)
        S_total = self.S_ref

        Ms=np.zeros((n_state,n_state))
        for I in range(n_state):
            for J in range(n_state):
                rdm_sz = 1/2 * ( rdm_aabb[0,I,J] - rdm_aabb[1,I,J] )
                sz = np.trace(rdm_sz)
                Ms[I,J] = sz

        self.log.extra("Make sure MS=")
        self.log.extra("%s", Ms)

        if not np.allclose(Ms, np.diag(np.diagonal(Ms)), atol=1e-5, rtol=0):
            self.log.warn("Warning: Ms is not diagonal")

        else:
            self.log.info("Ms is diagonal")

        sz = np.diagonal(Ms)
        sz = np.round(sz* 2) / 2
        sz = sz.tolist()
        sz = [round(elem,2) for elem in sz ]

        S_total = np.round(S_total* 2) / 2
        S_total = S_total.tolist()
        S_total = [round(elem,2) for elem in S_total ]

        #Choice ms as positive minimum
        if self.ms_ref_select is None:
            self.log.info("Choicing ms_ref as positive minimum")
            ms_ref = min((x for x in sz if x > 0), default=None) #Choice ms as positive minimum
        
            if ms_ref is None:
                self.log.info("detect ms_ref is None, it should be 0 ")
                ms_ref = 0
        else:
            self.log.info("User define ms_ref")
            ms_ref = self.ms_ref_select

        self.log.info("ms_ref=%s", ms_ref)

        ref_index = []
        for i in range(n_state):
            if (np.abs(sz[i]-ms_ref)) < 1e-5:
                ref_index.append(i)

        
        n_state_ref = len(ref_index)
  
        rdm_aabb_ref = np.zeros((2,n_state_ref, n_state_ref, len(rdm_aabb[0,0,0]),len(rdm_aabb[0,0,0])))
        E_ref = []
        S_ref = []
        E = E.tolist()
        for i in range(n_state_ref):
            I = ref_index[i]
            E_ref.append(E[I])
            S_ref.append(S_total[I])
            for j in range(n_state_ref):
                J = ref_index[j]
                rdm_aabb_ref[:,i,j] += rdm_aabb[:,I,J]

        self.log.info("roots_old=%s",n_state)
        self.log.info("roots_ref=%s",n_state_ref)
        self.log.info("S_ref=%s",S_ref)
        self.log.info("ms_ref=%s", ms_ref)
        self.log.info("E_ref=%s",E_ref)


        return E_ref, rdm_aabb_ref, S_ref, ms_ref


    def degeneracy_list(self,en, atol= 1e-5):
        '''
        determine degeneracy based on energy in list
        '''
        self.log.info("degeneracy_atol =%s", atol)

        deg = []
        count2, total_deg = 0,0
        for count1 in range(len(en)):
            if (np.abs(en[count1] - en[count2]) < atol):
                total_deg +=1
            else:
                deg.append(total_deg)
                total_deg = 1
                count2 = count1
        if (total_deg >=1):
            deg.append(total_deg)

        return deg