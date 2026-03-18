import numpy as np

# Plot spectrum
def plot(energies, intensities, broadening = 0.05, omega_min = 0.0, omega_max = 30.0, omega_step = 0.01, filename = "spectrum", plot = False, x_label = "x", y_label = "y", title = "Spectrum"):

    chi = []
    omega_values = np.arange(omega_min, omega_max, omega_step)

    for omega in omega_values:

        chi_value = 0.0 + 1j * 0.0

        for pole in range(energies.shape[0]):

            chi_value += intensities[pole] / (omega - energies[pole] + 1j * broadening)

        chi.append(-chi_value.imag / np.pi)

    chi = np.array(chi)

    # Save to a file
    table = np.hstack((omega_values.reshape(-1, 1), chi.reshape(-1, 1)))

    np.savetxt(filename + ".txt", table, fmt='%20.12f', delimiter='\t')

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(omega_values, chi)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(filename + ".pdf")


