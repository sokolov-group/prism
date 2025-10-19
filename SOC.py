import numpy as np


def Initialize_SOC(method):

    if method.evec_qdnevpt2 is None:
        print("It is NEVPT2")
    else:
        print("It is QDNEVPT2")


    ref_wfn =method.ref_wfn  
    ncas = method.ncas 
    ref_nelecas = method.ref_nelecas 
    ref_wfn_spin_mult = method.ref_wfn_spin_mult
    nstate = len(ref_wfn)
    evec = method.evec_qdnevpt2

    print(type(ref_wfn))
    #wfn = np.zeros((nstate, len(ref_wfn[0]), len(ref_wfn[1])))
    #for I in range(nstate)
    #evec[:,0] = np.array([0,0,1,0,0,0])
    #print(evec)
    wfn = np.einsum('ij,iab->jab',evec,ref_wfn)
    #print(wfn[0]-1*ref_wfn[2])
    #print("!!!!")
    
    ms = []
    print("nstate=",nstate)
    
    for I in range(nstate):
        sz = method.interface.apply_S_z(wfn[I],ncas,ref_nelecas[I])
        ms.append(np.dot(wfn[I].ravel(), sz.ravel()))

    ms = [round(elem,2) for elem in ms]

    print(ms)
    print(">?")


    
    



