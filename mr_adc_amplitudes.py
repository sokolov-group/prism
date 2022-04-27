import sys
import time
import numpy as np
# from functools import reduce
import prism.mr_adc_intermediates as mr_adc_intermediates
import prism.mr_adc_overlap as mr_adc_overlap
# import prism.mr_adc_integrals as mr_adc_integrals

# def compute_amplitudes(mr_adc):

#     start_time = time.time()

#     # First-order amplitudes
#     t1_amp = compute_t1_amplitudes(mr_adc)

#     # Second-order amplitudes
#     t2_amp = compute_t2_amplitudes(mr_adc, t1_amp)

#     t1_ce, t1_ca, t1_ae, t1_caea, t1_caaa, t1_aaea, t1_ccee, t1_ccea, t1_caee, t1_ccaa, t1_aaee = t1_amp
#     t2_ce, t2_ca, t2_ae, t2_caea, t2_caaa, t2_aaea, t2_ccee, t2_ccea, t2_caee, t2_ccaa, t2_aaee, t2_aa = t2_amp

#     print ("Time for computing amplitudes:                    %f sec\n" % (time.time() - start_time))

#     return (t1_ce, t1_ca, t1_ae, t1_caea, t1_caaa, t1_aaea, t1_ccee, t1_ccea, t1_caee, t1_ccaa, t1_aaee,
#             t2_ce, t2_ca, t2_ae, t2_caea, t2_caaa, t2_aaea, t2_ccee, t2_ccea, t2_caee, t2_ccaa, t2_aaee, t2_aa)

def compute_t1_p1(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca
    v_ccea = mr_adc.v2e.ccea

    ncore = mr_adc.ncore
    nextern = mr_adc.nextern

    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    # Computing K_ac
    K_ac = mr_adc_intermediates.compute_K_ac(mr_adc)

    # Orthogonalization and overlap truncation only in the active space
    S_p1_12_inv_act = mr_adc_overlap.compute_S12_p1(mr_adc, ignore_print = False)

    # Compute (S_12 K S_12)
    SKS = np.einsum("xy,yn->xn", K_ac, S_p1_12_inv_act)
    SKS = np.einsum("xm,xn->mn", S_p1_12_inv_act, SKS)

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    Vp1  = einsum('IJAX->IJAX', v_ccea, optimize = einsum_type).copy()
    Vp1 -= 0.5 * einsum('IJAy,yX->IJAX', v_ccea, rdm_ca, optimize = einsum_type)
    if mr_adc.debug_mode:
        print (">>> SA Vp1 norm: {:}".format(np.linalg.norm(Vp1)))

    Vp1 *= -1.0

    S_12_Vp1 = np.einsum("IJAX,Xm->IJAm", Vp1, S_p1_12_inv_act)

    # Multiply r.h.s. by U (e_a - e_i + e_mu)^-1 U^dag
    S_12_Vp1 = np.einsum("mp,IJAm->IJAp", evecs, S_12_Vp1)

    # Compute denominators
    d_ap = (e_extern[:,None] + evals).reshape(-1)
    d_ij = (e_core[:,None] + e_core).reshape(-1)

    d_apij = (d_ap[:,None] - d_ij).reshape(nextern, evals.shape[0], ncore, ncore)
    d_apij = d_apij**(-1)

    S_12_Vp1 = np.einsum("ApIJ,IJAp->IJAp", d_apij, S_12_Vp1)
    S_12_Vp1 = np.einsum("mp,IJAp->IJAm", evecs, S_12_Vp1)

    t_p1 = np.einsum("IJAm,Xm->IJAX", S_12_Vp1, S_p1_12_inv_act).copy()
    if mr_adc.debug_mode:
        print (">>> SA tp1 norm: {:}".format(np.linalg.norm(t_p1)))

    t1_ccea = t_p1.copy()
    e_p1  = 0.5 * einsum('ijax,ijax', t1_ccea, v_ccea, optimize = einsum_type)
    e_p1 -= 0.25 * einsum('ijax,ijay,xy', t1_ccea, v_ccea, rdm_ca, optimize = einsum_type)
    if mr_adc.debug_mode:
        print (">>> SA e_p1 norm: {:}".format(e_p1))

    return e_p1, t_p1

def compute_t1_m1(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca
    v_caee = mr_adc.v2e.caee

    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    ncore = mr_adc.ncore
    nextern = mr_adc.nextern

    # Computing K_ca
    K_ca = mr_adc_intermediates.compute_K_ca(mr_adc)

    # Orthogonalization and overlap truncation only in the active space
    S_m1_12_inv_act = mr_adc_overlap.compute_S12_m1(mr_adc, ignore_print = False)

    # Compute (S_12 K S_12)_{i a mu, j b nu}
    SKS = np.einsum("xy,yn->xn", K_ca, S_m1_12_inv_act)
    SKS = np.einsum("xm,xn->mn", S_m1_12_inv_act, SKS)

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    Vm1  = 0.5 * einsum('IyAB,Xy->IXAB', v_caee, rdm_ca, optimize = einsum_type)
    if mr_adc.debug_mode:
        print (">>> SA Vm1 norm: {:}".format(np.linalg.norm(Vm1)))

    Vm1 *= -1.0

    S_12_Vm1 = np.einsum("IXAB,Xm->ImAB", Vm1, S_m1_12_inv_act)

    # Multiply r.h.s. by U (e_a - e_i + e_mu)^-1 U^dag
    S_12_Vm1 = np.einsum("mp,ImAB->IpAB", evecs, S_12_Vm1)

    # Compute denominators
    d_ab = (e_extern[:,None] + e_extern).reshape(-1)
    d_ix = (e_core[:,None] - evals).reshape(-1)
    d_abix = (d_ab[:,None] - d_ix).reshape(nextern, nextern, ncore, evals.shape[0])
    d_abix = d_abix**(-1)

    S_12_Vm1 = np.einsum("ABIp,IpAB->IpAB", d_abix, S_12_Vm1)
    S_12_Vm1 = np.einsum("mp,IpAB->ImAB", evecs, S_12_Vm1)

    t_m1 = np.einsum("ImAB,Xm->IXAB", S_12_Vm1, S_m1_12_inv_act).copy()
    if mr_adc.debug_mode:
        print (">>> SA tm1 norm: {:}".format(np.linalg.norm(t_m1)))

    t1_caee = t_m1.copy()
    e_m1  = 0.25 * einsum('ixab,iyab,yx', t1_caee, v_caee, rdm_ca, optimize = einsum_type)
    if mr_adc.debug_mode:
        print (">>> SA e_m1 norm: {:}".format(e_m1))

    return e_m1, t_m1

