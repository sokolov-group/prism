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
# Authors: Carlos E. V. de Moura <carlosevmoura@gmail.com>
#          Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#          James D. Serna <jserna456@gmail.com>
#         

import os
import sys
import numpy as np
import re

import pyscf
import os
import glob


def get_geometries(directory=None, manual=False, min=None, max=None):
    """
    Identify XYZ geometries for active space selection.

    Parameters
    ----------
    directory : str
        Directory containing XYZ files.
    manual : bool, optional
        If True, manually restrict the geometries using min and max.
        If False, use all available XYZ files.
    min : int, optional
        First geometry index (inclusive).
    max : int, optional
        Last geometry index (inclusive).

    Returns
    -------
    xyz_files : list
        Sorted list of XYZ file paths.
    """

    # Validate directory
    if directory is None:
        raise ValueError(
            "No geometry directory provided. "
            "Please specify a directory containing XYZ files."
        )

    if not os.path.isdir(directory):
        raise ValueError(
            f"Geometry directory '{directory}' does not exist "
            "or is not a directory."
        )

    # Identify XYZ files
    xyz_files = glob.glob(os.path.join(directory, "*.xyz"))

    if not xyz_files:
        raise ValueError(
            f"No XYZ files found in geometry directory '{directory}'."
        )

    # Sort files numerically
    def numerical_key(filepath):
        filename = os.path.basename(filepath)
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.findall(r'\d+|\D+', filename)
        ]

    xyz_files.sort(key=numerical_key)

    # Validate manual selection
    if manual:
        if min is None or max is None:
            raise ValueError(
                "Manual geometry selection requires both 'min' and 'max'."
            )

        if not isinstance(min, int) or not isinstance(max, int):
            raise TypeError("'min' and 'max' must be integers.")

        if min < 0 or max < min:
            raise ValueError(
                "Invalid geometry range. "
                "'min' must be non-negative and 'max' must be >= 'min'."
            )

        if max >= len(xyz_files):
            raise ValueError(
                f"Requested geometry range ({min}, {max}) exceeds "
                f"the number of available XYZ files ({len(xyz_files)})."
            )

        xyz_files = xyz_files[min:max + 1]

    print(f"Geometries selected: {len(xyz_files)}")

    for xyz_file in xyz_files:
        print(f"  - {os.path.basename(xyz_file)}")

    print("Geometry identification complete.\n")

    return xyz_files

def build_molecule(xyz_file, basis):

    mol = pyscf.gto.Mole()
    mol.atom = xyz_file
    mol.basis = basis
    mol.verbose = 4
    mol.build()

    return mol

def run_guess_orbitals(xyz_files, method="rhf", basis="cc-pVDZ", alg_method='initial'):
    """
    Generate molecular orbital guesses for a collection of geometries.

    Parameters
    ----------
    xyz_files : list
        XYZ geometry files.
    method : str
        Electronic structure method used to generate orbitals.
    basis : str
        Basis set.

    Returns
    -------
    results : list
        Molecular orbital information for each geometry.
    """

    results = []

    if alg_method == 'initial':
        xyz_file = xyz_files[0]
        mol = build_molecule(xyz_file, basis)
                
        if method.lower() == "rhf":
            mf = pyscf.scf.RHF(mol)
        
        else:
            raise ValueError(
                f"Unsupported orbital guess method: {method}"
            )
        
        mf.kernel()
        
        results.append({
            "geometry": xyz_file,
            "mo_coeff": mf.mo_coeff,
            "mo_energy": mf.mo_energy,
            "mo_occ": mf.mo_occ,
            "energy": mf.e_tot,
        })
        
    elif alg_method == 'sequential':
        for xyz_file in xyz_files:

            # Read XYZ geometry
            # Build PySCF molecule
            mol = build_molecule(xyz_file, basis)
            
            if method.lower() == "rhf":
                mf = pyscf.scf.RHF(mol)

            else:
                raise ValueError(
                    f"Unsupported orbital guess method: {method}"
                )

            mf.kernel()

            results.append({
                "geometry": xyz_file,
                "mo_coeff": mf.mo_coeff,
                "mo_energy": mf.mo_energy,
                "mo_occ": mf.mo_occ,
                "energy": mf.e_tot,
            })

    return results

def run_algorithm(orbitals):

    print('HI')
    if len(orbitals) == 1:
        alg_type = 'initial'
        
    elif len(orbitals) > 1:
        alg_type = 'sequential'
        
    print(alg_type)