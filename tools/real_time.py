import os
import sys
import csv
import numpy as np
from prism import nevpt
from pyscf.tools import cubegen

def real_time_prop(nevpt, evec, etot):

    # Check if real-time propagation is implemented for the selected method
    if not (nevpt.method_type == "qd"): 
        raise Exception("Unrecognized methods for the Charge Migration!")

    # Eigenvectors 
    evec = np.array(evec)
    print("print eigen vectors = ", evec)
    print("shape of the eigen vectors = ", evec.shape)
    print("print eigen vector 0 = ", evec[0])
    print("shape of the eigen vector 0 = ", evec[0].shape)
    #evec_shape = evec.shape[0]
    evec_shape = etot.shape[0]
    
    # Energy difference
    e_diff = etot - etot[0]

    init_cond = None
    if nevpt.rt_init_cond is not None:
        init_cond = compute_init_cond_eigenstate(nevpt, evec_shape)
    else:
        raise Exception("Initial conditions are not provided for the Charge Migration!")

    # Transform initial conditions from the eignstate basis to the QD-NEVPT2  basis
    #wfn = np.dot(evec, init_cond)
    #wfn = init_cond.copy()
    
    t = 0.0

    time_step = nevpt.time_step

    print ("     Time             Norm(wfn)             E, a.u.     ")
    print ("--------------------------------------------------------")
    sys.stdout.flush()

    if nevpt.rt_prop_method == "exact":
        wfn = init_cond.copy()
        H_eff  = np.diag(etot)

    # wavefunction at t=0
    wfn0 = wfn.copy()
    print()
    print("wfn0 = ", wfn0)
    print()

    with open('auto_correlation.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Time (a.u.)', 'Auto-correlation \n'])
   
    dipole_csv = open("dipole_moment.csv", "w", newline="")
    dipole_writer = csv.writer(dipole_csv)
    dipole_writer.writerow(["Time (a.u.)", "Dipole_Mom_X", "Dipole_Mom_Y", "Dipole_Mom_Z", "Total_Dipole_Mom"])
    
    requested_states = np.where(np.abs(nevpt.rt_init_cond) > 1e-12)[0]
    with open("state_population.csv", "w", newline="") as pop_file:
        pop_writer = csv.writer(pop_file)
        #header = ["Time_au"] + "State_{i+1}_Population" for i in requested_states]
        #pop_writer.writerow(header)

    while t < nevpt.rt_tmax:

        # Check the norm of the wavefunction, the energy
        wfn_norm = np.linalg.norm(wfn)

        # Calculate Energy at given time
        E = np.dot(wfn.T.conjugate(), np.dot(H_eff,wfn))    

        # Calculate Auto-correlation function => <wfn(t=0)|wfn(t)>
        A = np.dot(np.conj(wfn0.T),wfn)

        # Calculate the population in Eigen state basis => |c_i|**2
        p = np.abs(wfn)**2

        # Print only the states 
        with open("state_population.csv", "a", newline="") as pop_file:
            pop_writer = csv.writer(pop_file)
            row = [t] + [p[i] for i in requested_states]
            pop_writer.writerow(row)

        # time-dependent rdm 
        #if nevpt.density == True and int(t // time_step) % nevpt.print_step == 0:
        #if nevpt.density == True and any(abs(t - target_time) <= time_step / 2 for target_time in [0.0, 125.0, 210.0, 310.0, 412.0]):
        if nevpt.density and any(abs(t - target_time) <= time_step / 2 for target_time in nevpt.density_plot):

            # Time-dependent RDM for hole density
            #td_rdm1(nevpt, wfn, evec, t)
            #td_rdm1_spin(nevpt, wfn, t)
            td_rdm1_check(nevpt, wfn, evec, t)
    
            # Time-dependent dipole moment
            dipmom_x, dipmom_y, dipmom_z, td_dip_total = td_dip_mom(nevpt, e_diff, wfn, t)
        
            # Print dipole moments in CSV file
            dipole_writer.writerow([t, dipmom_x, dipmom_y, dipmom_z, td_dip_total])

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

def td_rdm1(nevpt, coeff_t, evec, t):

    mc = nevpt.interface.mc
    mf = nevpt.interface.mf
    mol = nevpt.interface.mf.mol 
    mo_coeff = mc.mo_coeff

    # Compute all 1-RDMs
    rdms = nevpt.make_rdm1(rdm_type="all")

    print("RDM dimension =", rdms.shape[-1])
    print("Number of MO columns =", mo_coeff.shape[1])
    print("norm of the rdms = ", np.linalg.norm(rdms))
    print()

    print()
    print("coeff_t = ", coeff_t)
    print()

    # Ground state RDM
    #rdm_init = rdms[0,0].copy().real
       
    print("initial wavefunction evec = ", evec)
    print("initial wavefunction evec norm = ", np.linalg.norm(evec))
    print("shape of the evec = ", evec.shape)
    print("shape of the rdms = ", rdms.shape)
    print("shape of the coeff_t = ", coeff_t.shape)

    print()

    #rdm_init = np.einsum("I,J,IJpq->pq", np.conj(evec0), evec0, rdms).real
    evec0 = evec[0]
    print("shape of evec0 = ", evec0.shape)
    rdm_init = np.einsum("I,J,IJpq->pq", np.conj(evec0), evec0, rdms, optimize=True)
   
    non_init, no_init = np.linalg.eigh(rdm_init)
    rdm_init_diagonal = no_init.conj().T @ rdm_init @ no_init

    print()
    print("type of no_init = ", no_init.dtype)
    print("shape of the no_init = ", no_init.shape)
    print("eigen values of rdm_init = ", non_init)
    print()

    print("shape of the rdm_init = ", rdm_init.shape)
    print("mo_coeff rt= ", mo_coeff)
    print("dtype of rdm_init =", rdm_init.dtype)
    print("rdm_init = ", rdm_init)
    print("shape of the mo_coeff = ", mo_coeff.shape)
    print("dtype of the mo_coeff = ", mo_coeff.dtype)
    print("shape of the rdms = ", rdms.shape)
    print()

    # Time dependent RDM
    coeff_t = evec.T @ coeff_t
    print("shape of coeff_t in td = ", coeff_t.shape)
    td_rdms = np.einsum("I,J,IJpq->pq", np.conj(coeff_t), coeff_t, rdms, optimize=True)

    non_td, no_td = np.linalg.eigh(td_rdms)
    td_rdms_diagonal = no_td.conj().T @ td_rdms @ no_td

    print()
    print("type of no_td = ", no_td.dtype)
    print("shape of the no_td = ", no_td.shape)
    print("eigen values of rdm_td = ", non_td)
    print()

    # Without einsum
#    n_states = coeff_t.shape[0]
#    td_rdms = np.zeros(rdms.shape[2:], dtype=complex)
#
#    for I in range(n_states):
#        for J in range(n_states):
#            td_rdms += np.conj(coeff_t[I]) * rdms[I, J] * coeff_t[J] 
#            #td_rdms += np.conj(coeff_t[I]).T * rdms[I, I] * coeff_t[I] 


    relative_error = (np.linalg.norm(td_rdms - td_rdms.conj().T))
    relative_error1 = np.linalg.norm(td_rdms) - np.linalg.norm(td_rdms.conj().T)
    
    print("Relative Hermiticity error =", relative_error)
    print("Relative Hermiticity error1 =", relative_error1)
   
    print("norm of the td_rdms = ", np.linalg.norm(td_rdms))
    print()
    #td_rdms = td_rdms.real

    print("dtype of td_rdms =", td_rdms.dtype)
    print("shape of the coeff_t = ", coeff_t.shape)
    print("shape of the td_rdms = ", td_rdms.shape)
    print()

    rdm_diff = td_rdms - rdm_init 
    rdm_diff = mo_coeff @ rdm_diff @ mo_coeff.conj().T

    rdm_diff_dia = td_rdms_diagonal - rdm_init_diagonal
    rdm_diff_dia = mo_coeff @ rdm_diff_dia @ mo_coeff.T


    print()
    print("rdmf_dif = ", rdm_diff)

    # diagonalize the time-dependent rdm
    non, no = np.linalg.eigh(rdm_diff)
   
    print()
    mask = np.abs(non) > 1.0e-8 
    print("occupation numbers (|non| > 1e-8) = ", non[mask])

    # print Hole density
    hole_dm = np.zeros_like(rdm_diff)
    particle_dm =  np.zeros_like(rdm_diff)

    for i, occ_change in enumerate(non):

        orb = no[:, i]

        # hole density
        if occ_change > 1.0e-8: #0:
            hole_dm += (occ_change) * np.outer(orb, orb.conj())

        else:
            particle_dm += (-occ_change) * np.outer(orb, orb.conj())

    # Transform densities to the AO basis for cube generation
    hole_dm = mo_coeff @ hole_dm @ mo_coeff.T
    particle_dm = mo_coeff @ particle_dm @ mo_coeff.T

    # cube file generations
    cubegen.density(mol, f"density_rt_{t:010.6f}.cube", rdm_diff, nx=100, ny=100, nz=100)
    cubegen.density(mol, f"hole_density_rt_{t:010.6f}.cube", hole_dm, nx=100, ny=100, nz=100)
    cubegen.density(mol, f"particle_density_rt_{t:010.6f}.cube", particle_dm, nx=100, ny=100, nz=100)

    return td_rdms

def td_dip_mom(nevpt, e_diff, coeff_t, t):

    n_micro_states = len(e_diff)
    dip_mom_ao = nevpt.interface.dip_mom_ao
    mo_coeff = nevpt.interface.mo
    mc = nevpt.interface.mc
    mf = nevpt.interface.mf
    mol = nevpt.interface.mf.mol
    mo_coeff = mc.mo_coeff

    # Compute all 1-RDMs
    rdms = nevpt.make_rdm1()

    rdm_init = rdms[0,0].copy().real
 
    # Time-dependent RDMS
    td_rdms = np.einsum("I,J,IJpq->pq", np.conj(coeff_t).T, coeff_t, rdms).real

    # TODO: need to check dipole moment and td_rdms need in AO or MO basis
    dip_mom_mo = np.zeros_like(dip_mom_ao)
    
    td_dip_total = []

    for d in range(dip_mom_ao.shape[0]):
        dip_mom_mo[d] = (mo_coeff.conj().T @ dip_mom_ao[d] @ mo_coeff)
    
    dip_evec_x = np.einsum('pq,pq->', dip_mom_mo[0], td_rdms)
    dip_evec_y = np.einsum('pq,pq->', dip_mom_mo[1], td_rdms)
    dip_evec_z = np.einsum('pq,pq->', dip_mom_mo[2], td_rdms)

    td_dip_total.append((dip_evec_x + dip_evec_y + dip_evec_z).real)

    print("td_dip_total.append = ", td_dip_total)

    return dip_evec_x, dip_evec_y, dip_evec_z, td_dip_total

def td_rdm1_spin(nevpt, coeff_t, t):

    mc = nevpt.interface.mc
    mol = nevpt.interface.mf.mol
    mo_coeff = mc.mo_coeff

    # Shape: (2, nstates, nstates, nmo, nmo)
    rdms = nevpt.make_rdm1s()

    # Initial alpha and beta RDMs
    init_alpha = rdms[0, 0, 0]
    init_beta = rdms[1, 0, 0]

    # Time-dependent alpha and beta RDMs
    td_rdms = np.einsum("I,J,sIJpq->spq", coeff_t.conj(), coeff_t, rdms, optimize=True)

    td_alpha = td_rdms[0]
    td_beta = td_rdms[1]

    # Density differences
    diff_total = (td_alpha + td_beta) - (init_alpha + init_beta) 

    # Transform from MO to AO basis
    diff_alpha_ao = mo_coeff @ diff_alpha.real @ mo_coeff.conj().T
    diff_beta_ao = mo_coeff @ diff_beta.real @ mo_coeff.conj().T
    diff_total_ao = mo_coeff @ diff_total.real @ mo_coeff.conj().T

    # Write cube files
    cubegen.density(mol, f"density_{t:.3f}.cube", diff_total_ao.real)

    return


def td_rdm1_check(nevpt, coeff_t, evec, t):

    mc = nevpt.interface.mc
    mf = nevpt.interface.mf
    mol = mf.mol
    mo_coeff = np.asarray(mc.mo_coeff)

    # ============================================================
    # 1. Get transition 1-RDMs
    # ============================================================

    rdms = np.asarray(nevpt.make_rdm1(rdm_type="all"))

    coeff_t = np.asarray(coeff_t, dtype=complex).copy()
    evec = np.asarray(evec, dtype=complex)

    print("\n" + "=" * 75)
    print(f"TIME-DEPENDENT RDM DEBUGGING, t = {t}")
    print("=" * 75)

    print("rdms shape       =", rdms.shape)
    print("evec shape       =", evec.shape)
    print("coeff_t shape    =", coeff_t.shape)
    print("MO shape         =", mo_coeff.shape)

    if rdms.ndim != 4:
        raise ValueError("Expected rdms with shape (nstate,nstate,nmo,nmo)")

    if evec.ndim != 2:
        raise ValueError("evec must be the FULL eigenvector matrix")

    if coeff_t.ndim != 1:
        raise ValueError("coeff_t must be a 1D coefficient vector")

    ns = rdms.shape[0]

    if rdms.shape[1] != ns or evec.shape != (ns, ns):
        raise ValueError("Incompatible state dimensions")

    if coeff_t.size != ns:
        raise ValueError("Coefficient vector has wrong length")

    if rdms.shape[2:] != (mo_coeff.shape[1],) * 2:
        raise ValueError("RDM orbital dimensions do not match MO coefficients")

    # ============================================================
    # 2. Inspect eigenvectors
    # ============================================================

    print("\n--- EIGENVECTOR DIAGNOSTICS ---")

    print("Eigenvector matrix:")
    print(evec)

    print("Row 0:")
    print(evec[0])

    print("Column 0:")
    print(evec[:, 0])

    print("Row 0 norm =", np.linalg.norm(evec[0]))
    print("Column 0 norm =", np.linalg.norm(evec[:, 0]))

    print(
        "Unitarity error =",
        np.linalg.norm(evec.conj().T @ evec - np.eye(ns))
    )

    # ============================================================
    # 3. Original initial RDM, unchanged
    # ============================================================

    print("\n--- INITIAL RDM ---")

    evec0 = evec[0]

    rdm_init = np.einsum(
        "I,J,IJpq->pq",
        evec0.conj(),
        evec0,
        rdms,
        optimize=True
    )

    eig_init = np.linalg.eigvalsh(rdm_init)

    print("Initial RDM eigenvalues:")
    print(eig_init)

    print("Initial RDM trace =", np.trace(rdm_init))

    print(
        "Initial RDM Hermiticity error =",
        np.linalg.norm(rdm_init - rdm_init.conj().T)
    )

    # ============================================================
    # 4. Check propagated coefficients
    # ============================================================

    print("\n--- PROPAGATED WAVEFUNCTION ---")

    print("Original coeff_t:")
    print(coeff_t)

    print("Real components:")
    print(coeff_t.real)

    print("Imaginary components:")
    print(coeff_t.imag)

    print("Norm before transformation =", np.linalg.norm(coeff_t))

    print(
        "Imaginary component norm =",
        np.linalg.norm(coeff_t.imag)
    )

    # ============================================================
    # 5. Compare possible basis transformations
    # ============================================================

    transformations = {
        "transpose": evec.T @ coeff_t,
        "direct": evec @ coeff_t,
        "identity": coeff_t.copy()
    }

    results = {}

    for name, coeff_ref in transformations.items():

        print("\n" + "-" * 65)
        print("TRANSFORMATION:", name)
        print("-" * 65)

        print("Transformed coefficients:")
        print(coeff_ref)

        print("Coefficient norm =", np.linalg.norm(coeff_ref))

        # Construct time-dependent 1-RDM
        td_rdms = np.einsum(
            "I,J,IJpq->pq",
            coeff_ref.conj(),
            coeff_ref,
            rdms,
            optimize=True
        )

        # Eigenvalues of TD RDM
        eig_td = np.linalg.eigvalsh(td_rdms)

        print("\nTD RDM eigenvalues:")
        print(eig_td)

        print("TD RDM trace =", np.trace(td_rdms))

        print(
            "TD RDM Hermiticity error =",
            np.linalg.norm(td_rdms - td_rdms.conj().T)
        )

        # Density difference relative to original initial RDM
        rdm_diff = td_rdms - rdm_init

        print(
            "Density difference norm =",
            np.linalg.norm(rdm_diff)
        )

        print(
            "Density difference trace =",
            np.trace(rdm_diff)
        )

        print(
            "Maximum occupation difference =",
            np.max(np.abs(eig_td - eig_init))
        )

        results[name] = {
            "coeff_ref": coeff_ref,
            "rdm": td_rdms,
            "rdm_diff": rdm_diff,
            "eigenvalues": eig_td
        }

    # ============================================================
    # 6. Compare transformations directly
    # ============================================================

    print("\n--- DIRECT COMPARISON ---")

    print(
        "Difference between evec.T and evec RDMs =",
        np.linalg.norm(
            results["transpose"]["rdm"]
            - results["direct"]["rdm"]
        )
    )

    print(
        "Difference between transpose and identity RDMs =",
        np.linalg.norm(
            results["transpose"]["rdm"]
            - results["identity"]["rdm"]
        )
    )

    print(
        "Difference between direct and identity RDMs =",
        np.linalg.norm(
            results["direct"]["rdm"]
            - results["identity"]["rdm"]
        )
    )

    # ============================================================
    # 7. Test first eigenstate at t = 0
    # ============================================================

    print("\n--- FIRST EIGENSTATE TEST ---")

    coeff_test = np.zeros(ns, dtype=complex)
    coeff_test[0] = 1.0

    coeff_test_T = evec.T @ coeff_test
    coeff_test_direct = evec @ coeff_test

    print(
        "Transpose versus evec[0] =",
        np.linalg.norm(coeff_test_T - evec[0])
    )

    print(
        "Direct versus evec[0] =",
        np.linalg.norm(coeff_test_direct - evec[0])
    )

    # This only tests consistency with using evec[0]
    # as a row-stored eigenvector. It does not independently
    # prove that PRISM stores eigenvectors as rows.

    # ============================================================
    # 8. Compare eigenvalues
    # ============================================================

    print("\n--- EIGENVALUE COMPARISON ---")

    print(
        "Initial       Transpose       Direct       Identity"
    )

    for i in range(len(eig_init)):
        print(
            f"{eig_init[i]:12.8f}  "
            f"{results['transpose']['eigenvalues'][i]:12.8f}  "
            f"{results['direct']['eigenvalues'][i]:12.8f}  "
            f"{results['identity']['eigenvalues'][i]:12.8f}"
        )

    # ============================================================
    # 9. Generate AO density differences for each transformation
    # ============================================================

    for name, result in results.items():

        rdm_diff_mo = result["rdm_diff"]

        # MO to AO transformation, performed only once
        rdm_diff_ao = (
            mo_coeff
            @ rdm_diff_mo
            @ mo_coeff.conj().T
        )

        imag_norm = np.linalg.norm(rdm_diff_ao.imag)

        print(
            f"{name} AO density imaginary norm =",
            imag_norm
        )

        # Cube density assumes a suitable real AO representation.
        cubegen.density(
            mol,
            f"density_debug_{name}_{t:010.6f}.cube",
            rdm_diff_ao.real,
            nx=100,
            ny=100,
            nz=100
        )

    print("\nDebugging complete.")
    print("=" * 75)

    # Preserve the original convention provisionally.
    # This choice must be validated against PRISM's
    # eigenvector orientation.
    return results["transpose"]["rdm"]

