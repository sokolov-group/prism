import numpy as np

def compute_S12_p1(mr_adc, ignore_print = True):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas
    rdm_ca = mr_adc.rdm.ca

    S_p1  = einsum('XY->XY', np.identity(ncas), optimize = einsum_type).copy()
    S_p1 -= 0.5 * einsum('YX->XY', rdm_ca, optimize = einsum_type).copy()

    s_thresh = mr_adc.s_thresh_doubles
    S_eval, S_evec = np.linalg.eigh(S_p1)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]

    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])
    S_evec = S_evec[:, S_ind_nonzero]

    S_p1_12_inv = np.dot(S_evec, np.diag(S_inv_eval))

    if not ignore_print:
        print ("Dimension of the [+1] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
        if len(S_ind_nonzero) > 0:
            print ("Smallest eigenvalue of the [+1] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))

    return S_p1_12_inv