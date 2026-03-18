import numpy as np

# Plot spectrum
def spectrum_plot(method, e_tot, osc_str):

    # define parameters
    broadening = 0.05
    omega_min = 0.0
    omega_max = 150.0
    omega_step = 0.01
    
    if method.__class__.__name__ == "QDNEVPT":

        # Calculate the energy difference 
        diffE = e_tot[1:] - e_tot[0]

    elif method.__class__.__name__ == "MRADC": 

        diffE = e_tot 
        
    chi = []
    omega_values = np.arange(omega_min, omega_max, omega_step)

    for omega in omega_values:

        chi_value = 0.0 + 1j * 0.0

        for pole in range(len(osc_str)):

            chi_value += osc_str[pole] / (omega - diffE[pole] + 1j * broadening)

        chi.append(-chi_value.imag / np.pi)

    chi = np.array(chi)

    # Save to a file
    table = np.hstack((omega_values.reshape(-1, 1), chi.reshape(-1, 1)))
    
    np.savetxt("spectrum.dat", table, fmt='%20.12f', delimiter='\t')


