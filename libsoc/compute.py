import sys
import prism
import numpy as np
from prism.tools import trans_prop

def compute_somf_soc(interface):

    wfn = interface.ref_wfn

    # Calculate the total spin (S)
    spin_mult_wfn = interface.ref_wfn_spin_mult

    S = [round((spin_mult-1)/2,2) for spin_mult in spin_mult_wfn]

    # Calculate the spin projections Ms
    ms = []
    nstate = len(wfn)
    for I in range(nstate):
        sz = interface.apply_S_z(wfn[I],interface.ncas,interface.ref_nelecas[I])
        ms.append(np.dot(wfn[I].ravel(), sz.ravel()))

    ms = [round(elem,2) for elem in ms]

    from pyscf.fci.direct_spin1 import trans_rdm1s
    rdm= np.zeros((2, nstate, nstate, interface.nmo, interface.nmo))

    for I in range(nstate):
        for J in range(nstate):
            tmprdm_aabb = trans_rdm1s(wfn[J], wfn[I], interface.ncas, interface.ref_nelecas[I])
            rdm[0, I, J, interface.ncore:interface.ncore+interface.ncas, interface.ncore:interface.ncore+interface.ncas] = tmprdm_aabb[0]
            rdm[1, I, J, interface.ncore:interface.ncore+interface.ncas, interface.ncore:interface.ncore+interface.ncas] = tmprdm_aabb[1]

            if I == J:
                #uncorrelated diagonal terms
                rdm[:,I, J, :interface.ncore, :interface.ncore] =   np.identity(interface.ncore)  


    #generalSOC requires spin-free energy...
    en = interface.e_ref
    from prism.libsoc import general_somf
    en_soc, evec_soc = general_somf.state_interaction_soc(interface, en, rdm, S, ms)


    #compute osc
    #compute soc rdm
    rdm_sf = rdm[0] + rdm[1]
    I_total = []
    S_total = []
    ms_total = []
    for i in range(nstate):
        n = interface.ref_wfn_spin_mult[i]
        s = (n-1)/2
        for j in range(n):
            I_total.append(i)
            S_total.append(s)
            m = s-j
            ms_total.append(m)
    
    nstate_total = len(S_total)
    rdm_mo = np.zeros((nstate_total, nstate_total, interface.nmo, interface.nmo),dtype='complex')
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
        e_diff = en_soc- en_soc[gs_index]
        e_diff = e_diff[gs_index+1:]
        osc = trans_prop.osc_strength(interface, e_diff, rdm_soc[ gs_index, gs_index+1:])
        osc_str_soc_full.append(osc)
        osc_str_soc[gs_index:] += osc 

    # Print results obtained from soc-sa-casscf
    print_result_sa_casscf(interface, en_soc, osc_str_soc)

    return  


def print_result_sa_casscf(interface, en_soc, osc_str_soc):
    
    h2ev = interface.hartree_to_ev
    h2cm = interface.hartree_to_inv_cm
    
    interface.log.info("\nSummary of SOC-SA-CASSCF:")

    if interface.soc:
        interface.log.info("\nSummary of results for the %s calculation with the %s reference:" % (interface.soc.upper(), interface.reference.upper()))
    else:
        interface.log.info("\nSummary of results for the %s calculation with the %s reference:" % (interface.reference.upper()))

    interface.log.info("------------------------------------------------------------------------------------------------------------------")
    interface.log.info("  State    Degen.        E(total)            dE(a.u.)        dE(eV)      dE(nm)       dE(cm-1)      Osc Str.  ")
    interface.log.info("------------------------------------------------------------------------------------------------------------------")

    e_gs  = en_soc[0]
    e_tot = en_soc

    n_states = len(e_tot)

    for p in range(n_states):
        deg = 1
        if not interface.soc:
            deg = interface.spin_mult[p]
        de = e_tot[p] - e_gs
        de_ev = de * h2ev
        de_cm = de * h2cm
        if p == 0 or abs(de) < 1e-5:
            interface.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12s %14.4f   %12s" % ((p+1), deg, e_tot[p], de, de_ev, " ", de_cm, " "))
        else:
            de_nm = 10000000 / de_cm
            interface.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12.4f %14.4f    %12.8f" % ((p+1), deg, e_tot[p], de, de_ev, de_nm, de_cm, osc_str_soc[p-1]))

    interface.log.info("----------------------------------------------------------------------------------------------------------------")

    return
