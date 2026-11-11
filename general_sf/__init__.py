import sys
import os
import numpy as np
from prism.libsoc import general_somf
from prism.libsoc import magnetic
from prism.tools import trans_prop
from prism.libsoc import before_soc
from prism.libsoc import after_soc
import h5py


class SF_OBJECT:
    def __init__(self, interface, E_ref=None, rdm_aabb_ref=None, S_ref=None, mo=None, ms_ref=None, rdm_aabb_plus_ref=None):
        # General info
        self.interface = interface
        self.log = interface.log
        log = self.log
        log.info("Initializing SF_OBJECT...")
        self.stdout = interface.stdout
        self.verbose = interface.verbose
        self.max_memory = interface.max_memory
        self.ovlp = interface.ovlp
        self.spin = interface.mol.spin

        #Spin-free information
        #NECESSARY parapeter of state-interaction calculations
        self.E_ref = E_ref
        self.rdm_aabb_ref = rdm_aabb_ref
        self.S_ref = S_ref
        self.mo = mo
        self.ms_ref = ms_ref
        self.rdm_aabb_plus_ref = rdm_aabb_plus_ref

        #before_soc flag
        self.deg_atol = 1e-5 # For rotate_rdm1s_to_ms: Absolute tolerance for determining whether two states have the same energy.
        self.ms_ref_select = None # For get_sf_ref: Select ms=ms_ref_select for state-interaction calculation.

        #create soc flag
        before_soc.create_soc_flag(self)
    
    def import_mc(self):
        '''
        Import casscf from pyscf
        '''

        self.log.info("\nImporting mc in SF_OBJECT...")

        #If self has mc and adc, run_soc doesn't know should run soc-casscf or si-adc
        if hasattr(self, "adc"):
            raise Exception('mc object can not be loaded after loding adc')

        if self.interface.mc is None:
            raise Exception('mc should be loaded in interface')
        
        #CASSCF information
        self.mc = self.interface.mc
        self.nmo         = self.interface.nmo
        self.ncas        = self.interface.ncas
        self.ncore       = self.interface.ncore
        self.ref_nelecas = self.interface.ref_nelecas
        self.e_ref       = self.interface.e_ref
        self.mo          = self.interface.mo
        self.ref_wfn     = self.interface.ref_wfn
        self.ref_wfn_spin_mult = self.interface.ref_wfn_spin_mult


        return self

    def import_adc(self, adc):
        '''
        Import SRADC from pyscf
        '''

        self.log.info("\nImporting ADC in SF_OBJECT...")

        #If self has mc and adc, run_soc doesn't know should run soc-casscf or si-adc
        if hasattr(self, "mc"):
            raise Exception('ADC object can not be loaded after loding mc')
        
        #ADC information
        self.adc = adc
        self.method_type = adc.method_type.lower()
        self.adc.U = adc._adc_es.U #For uadc_ee.get_spin_square(self.adc) 
        self.U = adc._adc_es.U
        self.E = adc._adc_es.E
        self.P = adc._adc_es.P
        self.X = adc._adc_es.X
        self.roots = len(adc._adc_es.U[0])
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

        #MO:
        self.nmo = None
        self.mo = None

        #Load adc function from interface...
        self.make_rdm1_eigenvectors = self.interface.make_rdm1_eigenvectors
        self.get_spin_square = self.interface.get_spin_square


        #determine number of unpair electron in ADC
        spin = self.interface.mol.spin
        if self.method_type in ("ea", "ip") :
            if spin == 0:
                self.spin = spin+1
            else:
                #if unpair electon larger than zeoro, ea will add beta electron?
                #TODO: Make sure again after pyscf updata spinsquare function for IP EA
                self.spin = spin+1

        elif self.method_type == "ee" :
            self.spin = spin

        #flag
        if self.method_type == "ee":
            self.add_ref_ee = True  #determine whether add ref state or not in SI-EE-ADC calculation
        else:
            self.add_ref_ee = False
        self.use_P = True
        self.P_thresh =  1e-6

        return self

    def kernel_sf_soc(self):

        if self.rdm_aabb_plus_ref is not None: #Ms=0 condition
            E_ref = self.E_ref
            rdm_aabb_ref = self.rdm_aabb_ref
            rdm_aabb_plus_ref = self.rdm_aabb_plus_ref
            S_ref = self.S_ref

            if self.save_sf:
                before_soc.write_checkpoint_ms0(self.interface, E_ref, rdm_aabb_ref, rdm_aabb_plus_ref, S_ref, self.mo, name = self.name_sf) 

            #Compute en_soc
            en_soc, evec_soc = general_somf.state_interaction_soc_ms0(self.interface, E_ref, rdm_aabb_ref,  rdm_aabb_plus_ref, S_ref, self.mo, self.soc)

        else:    #Ms is not 0 condition
            if self.ms_ref is None:
                self.log.info("\n\nms_ref is None. Obtaining ms_ref from rdm_aabb_ref...\n")
                before_soc.rotate_rdm1s_to_ms(self)
                sys.stdout.flush()
                E_ref, rdm_aabb_ref, S_ref,  ms_ref = before_soc.get_sf_ref(self, self.spin)
                sys.stdout.flush()
            else:
                E_ref = self.E_ref
                rdm_aabb_ref = self.rdm_aabb_ref
                S_ref = self.S_ref
                ms_ref = self.ms_ref

            if self.save_sf:
                before_soc.write_checkpoint(self, E_ref, rdm_aabb_ref, S_ref, ms_ref, self.mo, name=self.name_sf)

            #Compute en_soc
            en_soc, evec_soc = general_somf.state_interaction_soc(self.interface, E_ref, rdm_aabb_ref, S_ref,  ms_ref, self.mo, self.soc)

        #Compute properties
        self.interface.mo = self.mo #Due to UHF's interface.mo = None. It is only has interface.mo_a and interface.mo_b 
        after_soc.compute_properties(self, en_soc, evec_soc, rdm_aabb_ref, S_ref)
        
        # Print results obtained from ADC
        after_soc.print_result(self, en_soc, evec_soc, S_ref, self.soc, "calculation" )

        #Print properties
        if "osc_strengths_full" in self.properties_soc:
            trans_prop.print_osc_strength(self.interface, self.properties_soc["osc_strengths_full"])
        if "magnetic_properties" in self.properties_soc:
            magnetic.print_mag_properties(self.interface, self.properties_soc,  method = self)

        return en_soc, self.properties_soc["osc_strengths"]

    def run_soc(self, soc_type=None):
        '''
        Run SOC for SR-ADC or CASSCF
        '''

        if soc_type:
            self.soc = soc_type

        if hasattr(self, "adc") and self.adc:
            import prism.libsoc.sradc
            #Collect self.E_ref, self.mo, self.rdm_aabb_ref, self.S_ref...
            prism.libsoc.sradc.collect_sf(self) 

        elif hasattr(self, "mc") and self.mc:
            import prism.libsoc.casscf
            #Collect self.E_ref, self.mo, self.rdm_aabb_ref, self.S_ref...
            prism.libsoc.casscf.collect_sf(self)
        
        en_soc, osc_str_soc = self.kernel_sf_soc()

        return  en_soc, osc_str_soc
        

