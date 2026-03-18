# Copyright 2026 Prism Developers. All Rights Reserved.
#
# Licensed under the GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.
#
# See the License file for the specific language governing
# permissions and limitations.
#
# Available at https://github.com/sokolov-group/prism
#
# Authors: Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#          Nishshanka M. Lakshan <lakshanweerasinghecc@gmail.com> 
#

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


