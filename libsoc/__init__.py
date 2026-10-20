import sys
import os
import numpy as np
from prism.libsoc import general_somf
from prism.libsoc import magnetic
from prism.tools import trans_prop
from prism.libsoc import before_soc
from prism.libsoc import after_soc
import h5py


def read_checkpoint(sf_name):

    print("\nLoad E_ref, rdm_aabb_ref, S_ref, ms_ref...")
    print("file name:"+sf_name+ "\n")
    with h5py.File(sf_name, "r") as f:
        E_ref        = f["E_ref"][:]
        rdm_aabb_ref = f["rdm_aabb_ref"][:]
        S_ref        = f["S_ref"][:]
        ms_ref       = f["ms_ref"][()]
        mo           = f["mo"][:]

    return E_ref, rdm_aabb_ref, S_ref, ms_ref, mo

def read_checkpoint_ms0(sf_name):

    print("\nE_ref, rdm_aabb_ref, rdm_aabb_plus_ref, S_ref...")
    print("file name:"+sf_name+ "\n")
    with h5py.File(sf_name, "r") as f:
        E_ref        = f["E_ref"][:]
        rdm_aabb_ref = f["rdm_aabb_ref"][:]
        rdm_aabb_plus_ref = f["rdm_aabb_plus_ref"][:]
        S_ref        = f["S_ref"][:]
        mo           = f["mo"][:]

    return E_ref, rdm_aabb_ref, rdm_aabb_plus_ref, S_ref, mo


class SF_OBJECT:
    def __init__(self, mol, E_ref, rdm_aabb_ref, S_ref, mo, ms_ref=None, rdm_aabb_plus_ref=None):

        #The code cannot determine whether to use state_interaction_soc or state_interaction_soc_ms0,
        #if ms_ref and rdm_aabb_plus_ref is specified simultaneously.
        if ms_ref is not None and rdm_aabb_plus_ref is not None:
            raise Exception("ms_ref and rdm_aabb_plus_ref cannot be specified simultaneously.")

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
        self.spin = mol.spin
        self.max_memory = mol.max_memory

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

  
        self.E_sf = E_ref
        self.rdm_aabb = rdm_aabb_ref
        self.S_total = S_ref

        self.ms_ref = ms_ref
        self.ms_ref_select = None

        self.rdm_aabb_plus_ref = rdm_aabb_plus_ref

        #create soc flag
        from prism.interface import PYSCF
        PYSCF.create_soc_flag(self)

    def run_soc(self, soc_type=None):

        if soc_type:
            self.soc = soc_type

        if self.rdm_aabb_plus_ref is not None: #Ms=0 condition
            E_ref = self.E_sf
            rdm_aabb_ref = self.rdm_aabb
            rdm_aabb_plus_ref = self.rdm_aabb_plus_ref
            S_ref = self.S_total

            #Compute en_soc
            en_soc, evec_soc = general_somf.state_interaction_soc_ms0(self, E_ref, rdm_aabb_ref,  rdm_aabb_plus_ref, S_ref, self.soc, verbose = self.verbose)

        else:    #Ms is not 0 condition
            if self.ms_ref is None:
                self.log.info("ms_ref is None. Obtaining ms_ref from rdm_aabb_ref...")
                before_soc.rotate_rdm1s_to_ms(self)
                sys.stdout.flush()
                E_ref, rdm_aabb_ref, S_ref,  ms_ref = before_soc.get_sf_ref(self, self.spin)
                sys.stdout.flush()
            else:
                E_ref = self.E_sf
                rdm_aabb_ref = self.rdm_aabb
                S_ref = self.S_total
                ms_ref = self.ms_ref

            #Compute en_soc
            en_soc, evec_soc = general_somf.state_interaction_soc(self, E_ref, rdm_aabb_ref, S_ref,  ms_ref, self.soc, verbose = self.verbose)

        #Compute properties
        after_soc.compute_properties(self, en_soc, evec_soc, rdm_aabb_ref, S_ref)
        
        # Print results obtained from ADC
        after_soc.print_results(self, en_soc, evec_soc, S_ref, self.soc, "sf_object")

        #Print properties
        if "osc_strengths_full" in self.properties_soc:
            trans_prop.print_osc_strength(self, self.properties_soc["osc_strengths_full"])
        if "magnetic_properties" in self.properties_soc:
            magnetic.print_mag_properties(self, self.properties_soc,  method = None)

        return en_soc, self.properties_soc["osc_strengths"]



