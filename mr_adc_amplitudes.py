import sys
import time
import numpy as np
from functools import reduce
import prism.mr_adc_intermediates as mr_adc_intermediates
import prism.mr_adc_overlap as mr_adc_overlap
import prism.mr_adc_integrals as mr_adc_integrals

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

    K_ac = mr_adc_intermediates.compute_K_ac(mr_adc)

    rdm_ca = mr_adc.rdm.ca

    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    # Orthogonalization and overlap truncation only in the active space
    S_p1_12_inv_act = mr_adc_overlap.compute_S12_p1(mr_adc, ignore_print = False)

    # Compute (S_12 K S_12)
    SKS = np.einsum("xy,yn->xn", K_ac, S_p1_12_inv_act)
    SKS = np.einsum("xm,xn->mn", S_p1_12_inv_act, SKS)

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    v_ccea = mr_adc.v2e.ccea
    v_ccae = -v_ccea.transpose(0,1,3,2).copy()

    Vp1 =- np.einsum('IJXA->IJAX', v_ccae).copy()
    Vp1 += np.einsum('IJxA, xX->IJAX', v_ccae, rdm_ca)

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

    tp1 = np.einsum("IJAm,Xm->IJAX", S_12_Vp1, S_p1_12_inv_act).copy()

    rdm_ac = np.identity(ncas) - rdm_ca
    e_p1 = 0.5 * np.einsum("ijay,ijax,xy", v_ccea, tp1, rdm_ac, optimize = True)

    return e_p1, tp1
