import os
import sys
import csv
import numpy as np
from prism import mr_adc 
from pyscf.tools import cubegen
from pyscf.tools import molden

def real_time_prop(mr_adc, evec, etot):

    # Check if real-time propagation is implemented for the selected method
    if not (mr_adc.method_type == "cvs-ip"): 
        raise Exception("Unrecognized methods for the Charge Migration!")

    # Eigenvectors 
    evec = np.array(evec)
    
    evec_shape = etot.shape[0]
    
    # Energy difference
    e_diff = etot - etot[0]

    init_cond = None
    if mr_adc.rt_init_cond is not None:
        init_cond = compute_init_cond_eigenstate(mr_adc, evec_shape)
    else:
        raise Exception("Initial conditions are not provided for the Charge Migration!")

    t = 0.0

    time_step = mr_adc.time_step

    print ("     Time             Norm(wfn)             E, a.u.     ")
    print ("--------------------------------------------------------")
    sys.stdout.flush()

    if mr_adc.rt_prop_method == "exact":
        wfn = init_cond.copy()
        H_eff  = np.diag(etot)

    # wavefunction at t=0
    wfn0 = wfn.copy()

    with open('auto_correlation.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Time (a.u.)', 'Auto-correlation \n'])

    all_dyson_mo     = []

    while t < mr_adc.rt_tmax:

        # Check the norm of the wavefunction, the energy
        wfn_norm = np.linalg.norm(wfn)

        # Calculate Energy at given time
        E = np.dot(wfn.T.conjugate(), np.dot(H_eff,wfn))    

        # Calculate Auto-correlation function => <wfn(t=0)|wfn(t)>
        A = np.dot(np.conj(wfn0.T),wfn)

        # Print propagation info
        print (" %10.6f         %10.6e           %10.6f" % (t, wfn_norm, E))
        sys.stdout.flush()
        
        if int(t // time_step) % mr_adc.print_step == 0:    

            # write autocorrelation function in every 50 steps
            with open('auto_correlation.csv', 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([t, abs(A)])

            # print td-dyson orbitals
            dyson_mo = td_dyson(mr_adc, wfn)

            if np.linalg.norm(dyson_mo) > 1e-6:
                all_dyson_mo.append(dyson_mo)

        if mr_adc.rt_prop_method == "exact":
            wfn = exact_propagator(mr_adc, wfn, H_eff)

        t = t + time_step 

    if all_dyson_mo:
        dyson_all_mo     = np.concatenate(all_dyson_mo, axis=1)
        molden.from_mo(mr_adc.interface.mol, 'dyson_rt.molden', dyson_all_mo)

    print ("--------------------------------------------------------\n")
    print ("Total number of time steps taken:       %d\n" % int(t / time_step))
    sys.stdout.flush()


def compute_init_cond_eigenstate(mr_adc, evec_shape):

    init_cond = mr_adc.rt_init_cond

    # Given a user-defined initial guess, calculate wavefunction at t = 0
    if type(init_cond) == tuple or type(init_cond) == list:
        init_cond = np.array(init_cond, dtype='complex')
    else:
        if type(init_cond) != np.ndarray:
            raise Exception("Initial conditions format is not recognized!")

    # Check if the initial conditions vector has the proper length
    if init_cond.shape[0] < evec_shape:
        new_init_cond = np.zeros(evec_shape, dtype='complex')
        new_init_cond[:init_cond.shape[0]] = init_cond
        init_cond = new_init_cond

    # Calculate the norm of initial conditions
    norm = np.linalg.norm(init_cond)
    init_cond /= norm
    mr_adc.rt_init_cond = init_cond

    return init_cond


def exact_propagator(mr_adc, wfn, H_eff):

    time_step = mr_adc.time_step

    A = -1j * np.diag(H_eff) * time_step
    A = np.exp(A)

    wfn = A * wfn

    return wfn

def td_dyson(mr_adc, wfn):

    X = mr_adc.compute_trans_moments()

    # Time-dependent spectroscopic amplitude
    X_t = np.dot(X, wfn)

    dyson_mo = np.dot(mr_adc.mo, X_t).reshape(-1, 1)

    # Seperate real and imaginary parts 
    dyson_mo_real = dyson_mo.real
    dyson_mo_imag = dyson_mo.imag

    dyson_mo = dyson_mo_real**2 + dyson_mo_imag**2

    return dyson_mo
