import sys
import prism
import numpy as np

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
    rdm = np.zeros((2, nstate, nstate, interface.nmo, interface.nmo))

    for I in range(nstate):
        for J in range(nstate):
            tmprdm_aabb = trans_rdm1s(wfn[J], wfn[I], interface.ncas, interface.ref_nelecas[I])
            rdm[0, I, J, interface.ncore:interface.ncore+interface.ncas, interface.ncore:interface.ncore+interface.ncas] = tmprdm_aabb[0]
            rdm[1, I, J, interface.ncore:interface.ncore+interface.ncas, interface.ncore:interface.ncore+interface.ncas] = tmprdm_aabb[1]


    #generalSOC requires spin-free energy...
    en = interface.e_ref
    from prism.libsoc import general_somf
    #en_soc, evec_soc = general_somf.state_interaction_soc(interface, en, rdm, S, ms)
    general_somf.state_interaction_soc(interface, en, rdm, S, ms)


    return  

