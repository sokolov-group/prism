import sys
import csv
import numpy as np

def real_time_prop(nevpt, evec, etot):

    # Check if real-time propagation is implemented for the selected method
    if not (nevpt.method_type == "qd"): 
        raise Exception("Unrecognized methods for the Charge Migration!")

    # Eigenvectors and transition moments
    evec = np.array(evec)
    evec_shape = evec.shape[0]

    init_cond = None
    if nevpt.rt_init_cond is not None:
        init_cond = compute_init_cond_eigenstate(nevpt, evec_shape)
    else:
        raise Exception("Initial conditions are not provided for the Charge Migration!")

    # Transform initial conditions from the eignstate basis to the MR-ADC excitation basis
    wfn = np.dot(evec, init_cond)

    t = 0.0

    time_step = nevpt.time_step

    print ("     Time             Norm(wfn)             E, a.u.     ")
    print ("--------------------------------------------------------")
    sys.stdout.flush()

    if nevpt.rt_prop_method == "exact":
        wfn = np.conj(evec.T) @ wfn
        #H_eff = np.conj(evec.T) @ H_eff @ evec
        H_eff  = np.diag(etot)

    # wavefunction at t=0
    wfn0 = wfn.copy()

    while t < nevpt.rt_tmax:

        # Check the norm of the wavefunction, the energy
        wfn_norm = np.linalg.norm(wfn)

        # Calculate Energy at given time
        E = np.dot(wfn.T.conjugate(), np.dot(H_eff,wfn))    

        # Calculate Auto-correlation function => <wfn(t=0)|wfn(t)>
        A = np.dot(np.conj(wfn0.T),wfn)

        # Print propagation info
        print (" %10.6f         %10.6e           %10.6f" % (t, wfn_norm, E))
        sys.stdout.flush()
        
        if int(t // time_step) % nevpt.print_step == 0:    

            # write autocorrelation function in every 50 steps
            with open('auto_correlation.csv', 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([t, abs(A)])

        if nevpt.rt_prop_method == "exact":
            wfn = exact_propagator(nevpt, wfn, H_eff)

        t = t + time_step 

    print ("--------------------------------------------------------\n")
    print ("Total number of time steps taken:       %d\n" % int(t / time_step))
    sys.stdout.flush()


def compute_init_cond_eigenstate(nevpt, evec_shape):

    init_cond = nevpt.rt_init_cond

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
    nevpt.rt_init_cond = init_cond

    return init_cond


def exact_propagator(nevpt, wfn, H_eff):

    time_step = nevpt.time_step

    A = -1j * np.diag(H_eff) * time_step
    A = np.exp(A)

    wfn = A * wfn

    return wfn

