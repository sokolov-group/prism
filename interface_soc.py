import sys
import os
import numpy as np
import prism.libsoc.general_somf
import prism.libsoc.magnetic
import pyscf.dft.LebedevGrid
import csv
import pickle
import h5py

np.set_printoptions(threshold=np.inf, linewidth=200)


def read_sf(sf_name):

    print("\nLoad E_ref, rdm_aabb_ref, S_ref, ms_ref...")
    print("file name:"+ "_" +sf_name+ ".h5\n")
    with h5py.File(sf_name + ".h5", "r") as f:
        E_ref        = f["E_ref"][:]
        rdm_aabb_ref = f["rdm_aabb_ref"][:]
        S_ref        = f["S_ref"][:]
        ms_ref       = f["ms_ref"][()]
        mo           = f["mo"][:]

    return E_ref, rdm_aabb_ref, S_ref, ms_ref, mo


class PYSCF_ADC:
    def __init__(self,mf,adc,E,U,P,X):
        import prism.lib.logger as logger
        self.stdout = mf.mol.stdout
        self.verbose = 4
        log = logger.Logger(self.stdout, self.verbose)
        self.log = log
        from pyscf.tools import molden
        self.molden = molden

        log.info("\nImport ADC in PYSCF...")

        # General info
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

        #For SOC
        self.max_memory_soc = mf.mol.max_memory
        #from pyscf.x2c import x2c
        self.xmol = None 
        self.contr_coeff = None


        self.nmo = mf.mo_coeff.shape[1]  
        self.mol = mf.mol
        self.mf = mf

        #ADC information
        self.adc = adc
        self.method_type = adc.method_type.lower()
        self.adc.U = U #For uadc_ee.get_spin_square(self.adc) 
        self.U = U
        self.E = E
        self.P = P
        self.X = X
        self.roots = len(U[0])
        self.nocc_a = adc.nocc_a
        self.nocc_b = adc.nocc_b
        self.nvir_a = adc.nvir_a
        self.nvir_b = adc.nvir_b
        self.nmo_a = adc.nocc_a + adc.nvir_a
        self.nmo_b = adc.nocc_b + adc.nvir_b
        self.mo_a = adc.mo_coeff[0]
        self.mo_b = adc.mo_coeff[1]

        self.ncvs = adc.ncvs
        #MPn ref information
        self.e_corr_MPn = adc.e_corr
        self.scf_energy = adc.scf_energy
        self.E_ref_abs= self.e_corr_MPn + self.scf_energy
        log.info("e_corr_MPn=%s",self.e_corr_MPn)
        log.info("scf_energy=%s",self.scf_energy)
        log.info("E_ref_abs=%s", self.E_ref_abs)
        #To DO:
        self.nmo = self.nmo_a
        self.mo = self.mo_a

        
        if self.ncvs:
            log.info("Importing uadc_ip_cvs instead of uadc_ip...")
            from pyscf.adc import uadc_ip_cvs
            self.make_rdm1_eigenvectors = uadc_ip_cvs.make_rdm1_eigenvectors
        else:
            if self.method_type == "ea" :
                log.info("Importing uadc_ea...")
                from pyscf.adc import uadc_ea
                self.make_rdm1_eigenvectors = uadc_ea.make_rdm1_eigenvectors
            elif self.method_type == "ip" :
                from pyscf.adc import uadc_ip
                log.info("Importing uadc_ip...")
                self.make_rdm1_eigenvectors = uadc_ip.make_rdm1_eigenvectors
            elif self.method_type == "ee":
                from pyscf.adc import uadc_ee
                log.info("Importing uadc_ee...")
                self.make_rdm1_eigenvectors = uadc_ee.make_rdm1_eigenvectors

        #determine number of unpair electron in ADC
        spin = self.mol.spin
        if self.method_type in ("ea", "ip") :
            if spin == 0:
                self.spin_adc = spin+1
            else:
                #if unpair electon larger than zeoro, ea will add beta electron?
                self.spin_adc = spin-1

        elif self.method_type == "ee" :
            self.spin_adc = spin

        #flag
        if self.method_type == "ee":
            self.add_ref_ee = True
        else:
            self.add_ref_ee = False
        self.use_P = True
        self.P_thresh =  1e-6
        self.deg_atol = 1e-5
        self.soc = "bp"
        self.save_sf = False
        self.analyze_evec_soc = False
        self.gtensor = False
        self.origin_type = 'charge'
        self.target_index = 1
        self.ms_ref_select = None
        self.ignore_Ms_diag = False

        # For magnetic susceptibility
        self.mag_av = False
        self.sus_av = False
        self.mag_vec = False
        self.sus_tensor = False

        self.step_h_s = 0.001 

        ###Powder magnetization
        Bs_list = []
        for i in range(15):
            H = i * 0.5
            Bs_list.append(H)

        self.Bs_powder_M = Bs_list 
        self.T_powder_M = [1.8]
        ###Powder susceptibility
        T_list = []
        for i in range(21):
            T = 14.75 * i + 5
            T_list.append(T)
        self.T_powder_chi = T_list
        self.Bs_powder_chi = [0.1]
        ###Vector magnetization
        self.B_vec_M = [0,0,1]
        self.Bs_vec_M = Bs_list 
        self.T_vec_M = [1.8]
        ###Tensor  susceptibility
        self.B_vec_chi = [0,0,1]
        self.Bs_vec_chi = [0.1]
        self.T_vec_chi = [5,100,200,250]


        #Calculating result
        self.rdm_aabb = None
        self.e_tot = None
        self.S_total = None
        self.properties = {}  

    def kernel(self):
        self.log.info("Collect necessary information for SOC")
        if self.add_ref_ee is True:
            self.log.info("employ make_rdm1s_ee")
            self.make_rdm1s_ee()
        else:
            self.log.info("employ make_rdm1s_general")
            self.make_rdm1s_general()

        sys.stdout.flush()

        self.rotate_rdm1s_to_ms()

        sys.stdout.flush()


        E_ref, rdm_aabb_ref, S_ref,  ms_ref = self.get_sf_ref()

        sys.stdout.flush()

        self.log.info(" ")

        if self.save_sf:
            self.collect_sf(E_ref, rdm_aabb_ref, S_ref, ms_ref, self.mo)


        self.soc.lower()
        if self.soc == "all":
            self.log.info("Calculate all possible soc approach")

            en_soc_bp,  G_sq_en_bp = self.compute_soc(E_ref, rdm_aabb_ref, S_ref,  ms_ref, "bp")
            sys.stdout.flush()
            en_soc_dkh1,  G_sq_en_dkh1 = self.compute_soc(E_ref, rdm_aabb_ref, S_ref,  ms_ref, "dkh1")
            sys.stdout.flush()
            self.e_tot = []
            self.e_tot.append(en_soc_bp)
            self.e_tot.append(en_soc_dkh1)

            self.properties["g-factors"] = []
            self.properties["g-factors"].append(G_sq_en_bp)
            self.properties["g-factors"].append(G_sq_en_dkh1)

            

        else:
            en_soc,  G_sq_en = self.compute_soc(E_ref, rdm_aabb_ref, S_ref,  ms_ref, self.soc)
            sys.stdout.flush()
            self.e_tot = en_soc
            self.properties["g-factors"] = G_sq_en


        
        return self.e_tot, self.properties


    def compute_soc(self,E_ref, rdm_aabb_ref, S_ref,  ms_ref, soc):
        import prism.libsoc.general_somf
        en_soc, evec_soc = prism.libsoc.general_somf.state_interaction_soc(self, E_ref, rdm_aabb_ref, S_ref,  ms_ref, soc)

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
            #print(values_sort)

            #print(sorted_row_idx)
            self.log.info("\nSOC index = [sf-ADC index]")
            for i in range(len(sorted_row_idx[0])):
                index_value = np.zeros((2,len(sorted_row_idx[:,i])))
                index_value[0] = sorted_row_idx[:,i] + 1
                index_value[1] = np.round(values_sort[:,i],2)
                self.log.info("%d = %s %s", i + 1, index_value[0], index_value[1])


        self.print_results_ref(en_soc,soc)
        self.print_results(en_soc,soc)

        if self.gtensor is True:
            import prism.libsoc.magnetic
            rdm_sf = rdm_aabb_ref[0] + rdm_aabb_ref[1]
            Mu_sf = prism.libsoc.magnetic.mag_dip(self,rdm_sf,S_ref, origin_type = self.origin_type)
            Mu = np.einsum('ai,kib,bj->kaj',np.conj(evec_soc).T, Mu_sf, evec_soc)
            G_sq_en, G_evec = prism.libsoc.magnetic.gtensor(self, S_ref, Mu, target_index = self.target_index)

            ge = self.g_free_elec
            self.log.info("\nMagnetic g-factors (ge = %s):" % ge)
            self.log.info("%14.6f, %14.6f, %14.6f" % (G_sq_en[0], G_sq_en[1], G_sq_en[2]))
            self.log.info("%14.6f, %14.6f, %14.6f (g-shift)" % (G_sq_en[0] - ge, G_sq_en[1] - ge, G_sq_en[2] - ge))
            self.log.info("%14.3f, %14.3f, %14.3f (g-shift, ppt)" % (1000 * (G_sq_en[0] - ge), 1000 * (G_sq_en[1] - ge), 1000 * (G_sq_en[2] - ge)))
            #self.properties["g-factors"] = G_sq_en
        else:
            G_sq_en = 0
        


        return en_soc,  G_sq_en


    def make_rdm1s_ee(self):
        #TO DO: make sure nmo_b is always equal nmo_a
        self.log.info("\n*****make_rdm1s_ee*****")
        self.log.info("Condsider Ref state in rdm...")
        n_state = len(self.E)
        n_state += 1
        nmo_a = self.nmo_a
        nmo_b = self.nmo_b
        mo_a = self.mo_a
        mo_b = self.mo_b
        nocc_a = self.nocc_a
        nocc_b = self.nocc_b
        nvir_a = self.nvir_a
        nvir_b = self.nvir_b
        ovlp = self.mol.intor('cint1e_ovlp_sph')
        U = self.U
        P = self.P
        X = self.X
        E_ref_abs = self.E_ref_abs

        #Check ref Energy
        if np.abs(self.E[0]) > 0.01:
            self.log.info("Take MPn E_ref as ground state energy ")
            self.E = np.insert(self.E, 0, 0)
        else:
            self.log.info("Asume E_ref = lowest ADC energy ")
            E_g = self.E[0]
            self.E = np.insert(self.E, 0, E_g)
        
        #add UHF and MPn energy
        self.E += E_ref_abs
        self.log.info("self.E=%s",self.E)


        #get S quantum number:
        from pyscf.adc import uadc_ee
        spin_square, evec_ne = uadc_ee.get_spin_square(self.adc)
        S_old_total = (-1 + ( 1 + 4*spin_square)**(1/2) ) /2 
        
        #Include reference state
        # TO DO: Find real ref Spin from mf object 
        ss_ref, s_ref =  self.mf.spin_square() #ss_ref is S(S+1), s_ref = 2S+1
        S_old_ref = (-1 + ( 1 + 4*ss_ref)**(1/2) ) /2
        S_old_total = np.insert(S_old_total, 0, S_old_ref)
        self.log.info("S_old_total=")
        self.log.info("%s", S_old_total)
        self.S_total = S_old_total


        # Calculate Full RDM
        rdm_aabb = np.zeros((2, n_state, n_state, nmo_b, nmo_b))
        rdm_aa = np.zeros((n_state,n_state,nmo_a,nmo_a))
        rdm_bb = np.zeros((n_state,n_state,nmo_b,nmo_b))

        rdm1_ref = self.adc.make_ref_rdm1()
        rdm_aa[0,0] = rdm1_ref[0]
        rdm_bb[0,0] = rdm1_ref[1]


        for i in range(n_state-1):   
            #rdm1_a, rdm1_b = self.make_ref_rdm1s_ee(U[:,i])
            #accroding to NTO in prism, X should be <0| |i>
            rdm1_a = X[0][i]
            rdm1_b = X[1][i]
            rdm_aa[0,i+1] = rdm1_a
            rdm_bb[0,i+1] = rdm1_b

            rdm_aa[i+1,0] = rdm1_a.T
            rdm_bb[i+1,0] = rdm1_b.T

        for i in range(n_state-1):
            for j in range(n_state-1):
                rdm1_a, rdm1_b = self.make_rdm1_eigenvectors(self.adc,U[:,i], U[:,j])
                rdm_aa[i+1,j+1] = rdm1_a    
                rdm_bb[i+1,j+1] = rdm1_b


        #Transform rdm_bb in mo_a basis
        self.log.info("Transform rdm_bb in mo_a basis")
        rdm_bb_ao = np.einsum('ai,IJib,bj->IJaj',mo_b,rdm_bb,mo_b.T)
        CTS = mo_a.T @ ovlp
        SC =  ovlp @ mo_a
        rdm_bb_mo_a = np.einsum('ai,IJib,bj->IJaj',CTS, rdm_bb_ao ,SC)

        rdm_aabb[0] += rdm_aa
        rdm_aabb[1] += rdm_bb_mo_a
        self.mo_b = mo_a
        self.mo = mo_a
        self.rdm_aabb = rdm_aabb

        return self.rdm_aabb, self.S_total, self.E

    
    def make_rdm1s_general(self):
        self.log.info("\n*****make_rdm1s_general*****")
        #TO DO: make sure nmo_b is always equal nmo_a
        n_state = len(self.E)
        nmo_a = self.nmo_a
        nmo_b = self.nmo_b
        mo_a = self.mo_a
        mo_b = self.mo_b
        ovlp = self.mol.intor('cint1e_ovlp_sph')
        U = self.U
        P = self.P
        spin_adc = self.spin_adc


        rdm_aabb = np.zeros((2, n_state, n_state, nmo_b, nmo_b))
        rdm_aa = np.zeros((n_state,n_state,nmo_a,nmo_a))
        rdm_bb = np.zeros((n_state,n_state,nmo_b,nmo_b))
        for i in range(n_state):
            for j in range(n_state):
                if self.ncvs:
                    rdm1_a, rdm1_b = self.make_rdm1_eigenvectors(self.adc,U[:,i], U[:,j])
                else:
                    if self.method_type == "ea" :
                        rdm1_a, rdm1_b = self.make_rdm1_eigenvectors(self.adc,U[:,i], U[:,j])
                    elif self.method_type == "ee" :
                        rdm1_a, rdm1_b = self.make_rdm1_eigenvectors(self.adc,U[:,i], U[:,j])
                    elif self.method_type == 'ip':
                        #flip make_rdm
                        rdm1_a, rdm1_b = self.make_rdm1_eigenvectors(self.adc,U[:,j], U[:,i])

                
                rdm_aa[i,j] = rdm1_a    
                rdm_bb[i,j] = rdm1_b


        #Transform rdm_bb in mo_a basis
        self.log.info("Transform rdm_bb in mo_a basis")
        rdm_bb_ao = np.einsum('ai,IJib,bj->IJaj',mo_b,rdm_bb,mo_b.T)
        CTS = mo_a.T @ ovlp
        SC =  ovlp @ mo_a
        rdm_bb_mo_a = np.einsum('ai,IJib,bj->IJaj',CTS, rdm_bb_ao ,SC)

        rdm_aabb[0] += rdm_aa
        rdm_aabb[1] += rdm_bb_mo_a
        self.mo_b = mo_a
        self.mo = mo_a
        self.rdm_aabb = rdm_aabb

        #get S quantum number:
        S_old_total = np.zeros(n_state)
        if self.method_type == "ee":
            from pyscf.adc import uadc_ee
            spin_square, evec_ne = uadc_ee.get_spin_square(self.adc)
            S_old_total = (-1 + ( 1 + 4*spin_square)**(1/2) ) /2 
        else:
            S_old = round(spin_adc/2 , 2) 

            # If spec factor too low, spin should be different
            if self.use_P is True:
                self.log.info("Use P to find S quantum number")
                self.log.info("P_thresh=%s", self.P_thresh) 
                for i in range(n_state):
                    if P[i] > self.P_thresh:
                        S_old_total[i] = S_old
                    else:
                        S_old_total[i] = S_old + 1
            else:
                self.log.info("Use trdm to find S quantum number")
                self.log.info("P_thresh=%s", self.P_thresh)
                if P[0] > self.P_thresh:
                    S_old_total[0] = S_old
                else:
                    S_old_total[0] = S_old + 1
                
                for i in range(n_state-1):
                    trdm = rdm_aabb[0,0,i+1] + rdm_aabb[1,0,i+1]
                    if np.linalg.norm(trdm) < 1:
                        S_old_total[i+1] = S_old_total[0]+1
                    else:
                        S_old_total[i+1] = S_old_total[0]

        self.log.info("S_old_total=")
        self.log.info("%s", S_old_total)
        self.S_total = S_old_total

        return self.rdm_aabb, self.S_total, self.E
    

    def rotate_rdm1s_to_ms(self):
        self.log.info("\n*****rotate_rdm1s_to_ms*****")
        #TO DO: make sure nmo_b is always equal nmo_a
        rdm_aabb = self.rdm_aabb
        S_old_total = self.S_total
        E_old_total = self.E
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

        self.log.info("Ms_old_total=")
        self.log.info("%s", Ms_total)

        for K in range(len(deg_list)):
            I = int(np.sum(deg_list[0:K]))
            J = I + deg_list[K]
            Ms = Ms_total[I:J, I:J]
            sz_en, sz_evec = np.linalg.eigh(Ms)
            Sz_evec_total[I:J,I:J] = sz_evec
            Sz_en_tot[I:J] = sz_en

        self.log.info("sz_en_tot=")
        self.log.info("%s", Sz_en_tot)

        self.log.info("Sz_evec_total=")
        self.log.info("%s", Sz_evec_total)

        self.rdm_aabb = np.einsum('ai,KibIJ,bj->KajIJ',np.conj(Sz_evec_total).T , rdm_aabb , Sz_evec_total)

        S_total = np.einsum('ai,ib,bj->aj',np.conj(Sz_evec_total).T , np.diag(S_old_total) , Sz_evec_total)
        self.log.info("Make sure S_total:")
        self.log.info("%s", S_total)
        if not np.allclose(S_total, np.diag(np.diagonal(S_total)), atol=1e-7, rtol=0):
            self.log.info("Warning: S_total is not diagonal")
        else:
            self.log.info("S_total is diagonal")

        self.S_total = np.diagonal(S_total)

        E_new = np.einsum('ai,ib,bj->aj',np.conj(Sz_evec_total).T , np.diag(E_old_total) , Sz_evec_total)
        self.log.info("Make sure E:")
        self.log.info("%s", E_new)
        E_new = np.diagonal(E_new)
        if np.allclose(E_old_total, E_new, atol=1e-7, rtol=0):
            self.log.info("Energy is same after rotation")
            self.E = E_new
        else:
            raise Exception("E_new change energy")




        return self.rdm_aabb
    

    def get_sf_ref(self):
        self.log.info("\n*****get_sf_ref*****")
        rdm_aabb = self.rdm_aabb
        E = self.E
        n_state = len(E)
        S_total = self.S_total
        spin_adc = self.spin_adc 

        Ms=np.zeros((n_state,n_state))
        for I in range(n_state):
            for J in range(n_state):
                rdm_sz = 1/2 * ( rdm_aabb[0,I,J] - rdm_aabb[1,I,J] )
                sz = np.trace(rdm_sz)
                Ms[I,J] = sz

        self.log.info("Make sure MS=")
        self.log.info("%s", Ms)

        if not np.allclose(Ms, np.diag(np.diagonal(Ms)), atol=1e-10, rtol=0):
            self.log.info("Warning: Ms is not diagonal")
            if self.ignore_Ms_diag:
                self.log.info("ignore_Ms_diag")
            else:
                raise Exception("Ms is not diagonal")
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
            self.log.info("Choice ms_ref as positive minimum")
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

        #Discard unreasonable spin state 
        self.log.info("Discard unreasonable spin state ...")
        ref_index_S = []
        #even electron, S sohuld be 0,1,2,3....
        if spin_adc % 2 == 0:
            self.log.info("even electrons...")
            for i in range(len(ref_index)):
                I = ref_index[i]
                S_check = S_total[I]
                if abs(S_check % 1 - 0.5) < 1e-8:
                    self.log.info("discard S=%.1f state" %S_check)
                elif abs(S_check % 1) < 1e-8:    
                    ref_index_S.append(I)
                else:
                    self.log.info("discard no .5 .0  S=%.1f state" %S_check)
            
        #odd electron, S sohuld be 0.5, 1.5, 2.5, 3.5....
        else:
            self.log.info("odds electrons...")
            for i in range(len(ref_index)):
                I = ref_index[i]
                S_check = S_total[I]
                if abs(S_check % 1 - 0.5) < 1e-8:
                    ref_index_S.append(I)
                elif abs(S_check % 1) < 1e-8: 
                    self.log.info("discard S=%.1f state" %S_check)
                else:
                    self.log.info("discard no .5 .0 S=%.1f state" %S_check)
                

        
        n_state_ref = len(ref_index_S)
  
        rdm_aabb_ref = np.zeros((2,n_state_ref, n_state_ref, len(rdm_aabb[0,0,0]),len(rdm_aabb[0,0,0])))
        E_ref = []
        S_ref = []
        E = E.tolist()
        for i in range(n_state_ref):
            I = ref_index_S[i]
            E_ref.append(E[I])
            S_ref.append(S_total[I])
            for j in range(n_state_ref):
                J = ref_index_S[j]
                rdm_aabb_ref[:,i,j] += rdm_aabb[:,I,J]

        self.log.info("roots_old=%s",n_state)
        self.log.info("roots_ref=%s",n_state_ref)
        self.log.info("S_ref=%s",S_ref)
        self.log.info("ms_ref=%s", ms_ref)
        self.log.info("E_ref=%s",E_ref)

        self.log.info("set up xmol, contr_coeff")
        self.log.info("max_memory_soc =%s", self.max_memory_soc)
        from pyscf.x2c import x2c
        self.mol.max_memory = self.max_memory_soc
        self.xmol, self.contr_coeff = x2c.X2C(self.mol).get_xmol()

        return E_ref, rdm_aabb_ref, S_ref, ms_ref

    def collect_sf(self, E_ref, rdm_aabb_ref, S_ref, ms_ref, mo):

        self.log.info("\nSave E_ref, rdm_aabb_ref, S_ref, ms_ref... as csv file...")
        input_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        self.log.info("file name:"+input_file+ ".h5\n")

        with h5py.File(input_file + ".h5", "w") as f:
            f.create_dataset("E_ref", data=E_ref)
            f.create_dataset("rdm_aabb_ref", data=rdm_aabb_ref)
            f.create_dataset("S_ref", data=S_ref)
            f.create_dataset("ms_ref", data=ms_ref)
            f.create_dataset("mo", data=mo)


    def degeneracy_list(self,en, atol= 1e-5):

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


    def print_results(self,en_soc,soc):

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
                self.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12.4f %14.4f   %12s" % ((p+1), deg, e_tot[p], de, de_ev, de_nm, de_cm, "  "))

        self.log.info("----------------------------------------------------------------------------------------------------------------")


    def print_results_ref(self,en_soc,soc):

        h2ev = self.hartree_to_ev
        h2cm = self.hartree_to_inv_cm


        self.log.info("\nSummary of SOC results for %s", soc.upper())


        self.log.info("------------------------------------------------------------------------------------------------------------------")
        self.log.info("  State    Degen.        E(total)             E(a.u.)         E(eV)         E(nm)       E(cm-1)        Osc Str.  ")
        self.log.info("------------------------------------------------------------------------------------------------------------------")

        e_gs = en_soc[0]
        e_tot = en_soc

        n_states = len(e_tot)

        for p in range(n_states):
            deg = 1
            de = e_tot[p] 
            de_ev = de * h2ev
            de_cm = de * h2cm
            if p == 0 or abs(de) < 1e-5:
                self.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12s %14.4f   %12s" % ((p+1), deg, e_tot[p], de, de_ev, " ", de_cm, " "))
            else:
                de_nm = 10000000 / de_cm
                self.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12.4f %14.4f   %12s" % ((p+1), deg, e_tot[p], de, de_ev, de_nm, de_cm, "  "))

        self.log.info("----------------------------------------------------------------------------------------------------------------")


class SF_OBJECT:
    def __init__(self,mf, E_ref, rdm_aabb_ref, S_ref,  ms_ref, mo):
        self.log.info("\nImport spin-free object...")
        import prism.lib.logger as logger
        self.stdout = mf.mol.stdout
        self.verbose = 4
        log = logger.Logger(self.stdout, self.verbose)
        self.log = log
        from pyscf.tools import molden
        self.molden = molden

        # General info
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

        #For SOC
        self.max_memory_soc = mf.mol.max_memory
        #from pyscf.x2c import x2c
        self.xmol = None 
        self.contr_coeff = None


        self.nmo = mf.mo_coeff.shape[1]  
        self.mol = mf.mol
        self.mf = mf

        #ADC information
        #self.adc = adc
        #self.method_type = adc.method_type.lower()
        #self.adc.U = U #For uadc_ee.get_spin_square(self.adc) 
        #self.U = U
        #self.E = E
        #self.P = P
        #self.X = X
        self.roots = len(E_ref)
        #self.nocc_a = adc.nocc_a
        #self.nocc_b = adc.nocc_b
        #self.nvir_a = adc.nvir_a
        #self.nvir_b = adc.nvir_b
        #self.nmo_a = adc.nocc_a + adc.nvir_a
        #self.nmo_b = adc.nocc_b + adc.nvir_b
        #self.mo_a = adc.mo_coeff[0]
        #self.mo_b = adc.mo_coeff[1]
        #self.ncvs = adc.ncvs
        #MPn ref information
        #self.e_corr_MPn = adc.e_corr
        #self.scf_energy = adc.scf_energy
        #self.E_ref_abs= self.e_corr_MPn + self.scf_energy
        #print("e_corr_MPn=",self.e_corr_MPn)
        #print("scf_energy=",self.scf_energy)
        #print("E_ref_abs=" ,self.E_ref_abs)

        #SOC:
        self.nmo = len(rdm_aabb_ref[0,0,0,0])
        self.mo = mo
        self.E_ref = E_ref
        self.rdm_aabb_ref = rdm_aabb_ref
        self.S_ref = S_ref
        self.ms_ref = ms_ref
        self.soc = "bp"
        self.analyze_evec_soc = True

        self.log.info("set up xmol, contr_coeff")
        self.log.info("max_memory_soc =%s", self.max_memory_soc)
        from pyscf.x2c import x2c
        self.mol.max_memory = self.max_memory_soc
        self.xmol, self.contr_coeff = x2c.X2C(self.mol).get_xmol()

        
        #if self.ncvs:
        #    print("Importing uadc_ip_cvs instead of uadc_ip...")
        #    from pyscf.adc import uadc_ip_cvs
        #    self.make_rdm1_eigenvectors = uadc_ip_cvs.make_rdm1_eigenvectors
        #else:
        #    if self.method_type == "ea" :
        #        print("Importing uadc_ea...")
        #        from pyscf.adc import uadc_ea
        #        self.make_rdm1_eigenvectors = uadc_ea.make_rdm1_eigenvectors
        #    elif self.method_type == "ip" :
        #        from pyscf.adc import uadc_ip
        #        print("Importing uadc_ip...")
        #        self.make_rdm1_eigenvectors = uadc_ip.make_rdm1_eigenvectors
        #    elif self.method_type == "ee":
        #        from pyscf.adc import uadc_ee
        #        print("Importing uadc_ee...")
        #        self.make_rdm1_eigenvectors = uadc_ee.make_rdm1_eigenvectors

        #determine number of unpair electron in ADC
        #pin = self.mol.spin
        #f self.method_type in ("ea", "ip") :
        #   if spin == 0:
        #       self.spin_adc = spin+1
        #   else:
        #       #if unpair electon larger than zeoro, ea will add beta electron?
        #       self.spin_adc = spin-1

        #lif self.method_type == "ee" :
        #   self.spin_adc = spin

        #flag
        #if self.method_type == "ee":
        #    self.add_ref_ee = True
        #else:
        #    self.add_ref_ee = False
        #self.use_P = True
        #self.P_thresh =  1e-6
        #self.deg_atol = 1e-5
        #self.soc = "bp"
        #self.save_sf = False
        #self.analyze_evec_soc = False
        #self.gtensor = False
        #self.origin_type = 'charge'
        #self.target_index = 1
        #self.ms_ref_select = None
        #self.ignore_Ms_diag = False

        ## For magnetic susceptibility
        #self.mag_av = False
        #self.sus_av = False
        #self.mag_vec = False
        #self.sus_tensor = False

        #self.step_h_s = 0.001 

        ####Powder magnetization
        #Bs_list = []
        #for i in range(15):
        #    H = i * 0.5
        #    Bs_list.append(H)

        #self.Bs_powder_M = Bs_list 
        #self.T_powder_M = [1.8]
        ####Powder susceptibility
        #T_list = []
        #for i in range(21):
        #    T = 14.75 * i + 5
        #    T_list.append(T)
        #self.T_powder_chi = T_list
        #self.Bs_powder_chi = [0.1]
        ####Vector magnetization
        #self.B_vec_M = [0,0,1]
        #self.Bs_vec_M = Bs_list 
        #self.T_vec_M = [1.8]
        ####Tensor  susceptibility
        #self.B_vec_chi = [0,0,1]
        #self.Bs_vec_chi = [0.1]
        #self.T_vec_chi = [5,100,200,250]


        ##Calculating result
        #self.rdm_aabb = None
        #self.e_tot = None
        #self.S_total = None
        #self.properties = {}  

    def kernel(self):
        self.compute_soc(self.E_ref, self.rdm_aabb_ref, self.S_ref,  self.ms_ref, self.soc)
        return






    def compute_soc(self,E_ref, rdm_aabb_ref, S_ref,  ms_ref, soc):
        import prism.libsoc.general_somf
        en_soc, evec_soc = prism.libsoc.general_somf.state_interaction_soc(self, E_ref, rdm_aabb_ref, S_ref,  ms_ref, soc)
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
            #deg_list = self.degeneracy_list(E_ref)

            #print(population_state)
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
            #print(values_sort)
            #print(sorted_row_idx)
            print("\nSOC index = [sf-ADC index]")
            for i in range(len(sorted_row_idx[0])):
                index_value = np.zeros((2,len(sorted_row_idx[:,i])))
                index_value[0] = sorted_row_idx[:,i] + 1
                index_value[1] = np.round(values_sort[:,i],2)
                print(i+1, "=", index_value[0], index_value[1])
        self.print_results_ref(en_soc,soc)
        self.print_results(en_soc,soc)


    def print_results(self,en_soc,soc):

        h2ev = self.hartree_to_ev
        h2cm = self.hartree_to_inv_cm


        self.log.info("\nSummary of SOC results for",soc.upper())


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
                self.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12.4f %14.4f   %12s" % ((p+1), deg, e_tot[p], de, de_ev, de_nm, de_cm, "  "))

        self.log.info("----------------------------------------------------------------------------------------------------------------")


    def print_results_ref(self,en_soc,soc):

        h2ev = self.hartree_to_ev
        h2cm = self.hartree_to_inv_cm


        self.log.info("\nSummary of SOC results for", soc.upper())


        self.log.info("------------------------------------------------------------------------------------------------------------------")
        self.log.info("  State    Degen.        E(total)             E(a.u.)         E(eV)         E(nm)       E(cm-1)        Osc Str.  ")
        self.log.info("------------------------------------------------------------------------------------------------------------------")

        e_gs = en_soc[0]
        e_tot = en_soc

        n_states = len(e_tot)

        for p in range(n_states):
            deg = 1
            de = e_tot[p] 
            de_ev = de * h2ev
            de_cm = de * h2cm
            if p == 0 or abs(de) < 1e-5:
                self.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12s %14.4f   %12s" % ((p+1), deg, e_tot[p], de, de_ev, " ", de_cm, " "))
            else:
                de_nm = 10000000 / de_cm
                self.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12.4f %14.4f   %12s" % ((p+1), deg, e_tot[p], de, de_ev, de_nm, de_cm, "  "))

        self.log.info("----------------------------------------------------------------------------------------------------------------")













