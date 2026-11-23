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

# XYZ READER
def read_xyz(xyz_file):

    with open(xyz_file, "r") as f:
        lines = f.readlines()

    n = int(lines[0])
    atoms = []
    coords = []

    for line in lines[2:2+n]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])

    return np.array(coords), atoms


# QUATERNION MATRIX (UNCHANGED)
def build_quaternion_matrix(M):
    S_W = np.zeros((4,4))
    
    S_W[0,0] = M[0,0] + M[1,1] + M[2,2]
    S_W[1,0] = M[1,2] - M[2,1]
    S_W[2,0] = M[2,0] - M[0,2]
    S_W[3,0] = M[0,1] - M[1,0]
    
    S_W[0,1] = M[1,2] - M[2,1]
    S_W[1,1] = M[0,0] - M[1,1] - M[2,2]
    S_W[2,1] = M[0,1] + M[1,0]
    S_W[3,1] = M[0,2] + M[2,0]
    
    S_W[0,2] = M[0,2] - M[2,0]
    S_W[1,2] = M[0,1] + M[1,0]
    S_W[2,2] = -M[0,0] + M[1,1] - M[2,2]
    S_W[3,2] = M[1,2] + M[2,1]
    
    S_W[0,3] = M[0,1] - M[1,0]
    S_W[1,3] = M[0,2] + M[2,0]
    S_W[2,3] = M[1,2] + M[2,1]
    S_W[3,3] = -M[0,0] - M[1,1] + M[2,2]
    
    return S_W


# QUAT → ROTATION MATRIX 
def quaternion_to_rotation_matrix(quat):
    from scipy.spatial.transform import Rotation as R
    
    quat = quat / np.linalg.norm(quat + 1e-15)
    return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


# ALIGNMENT 
def align_structure(X, Y):

    com_X = X.mean(axis=0)
    com_Y = Y.mean(axis=0)

    Xc = X - com_X
    Yc = Y - com_Y

    M = Xc.T @ Yc
    S_W = build_quaternion_matrix(M)

    eigval, eigvec = np.linalg.eigh(S_W)
    quat = eigvec[:, np.argmax(eigval)]

    R = quaternion_to_rotation_matrix(quat)

    # explicit translation (IMPORTANT FIX)
    t = com_Y - com_X @ R.T

    def rigid_transform(P):
        return P @ R.T + t

    X_aligned = rigid_transform(X)

    Gx = np.trace(Xc.T @ Xc)
    Gy = np.trace(Yc.T @ Yc)

    resid = (Gx + Gy - 2*np.trace(M @ R)) / Xc.shape[0]

    return X_aligned, R, resid, t


# PE TRANSFORM 
def transform_pot(in_pot, out_pot, R, t):

    with open(in_pot, "r") as f:
        lines = f.readlines()

    out_lines = []

    in_coords = False
    in_mult = False
    in_pol = False

    current_order = None
    skip_count = 0

    for line in lines:

        # SECTION SWITCHING
        if "@COORDINATES" in line:
            in_coords = True
            in_mult = False
            in_pol = False
            out_lines.append(line)
            continue

        if "@MULTIPOLES" in line:
            in_coords = False
            in_mult = True
            in_pol = False
            current_order = None
            skip_count = 0
            out_lines.append(line)
            continue

        if "@POLARIZABILITIES" in line:
            in_coords = False
            in_mult = False
            in_pol = True
            current_order = None
            skip_count = 0
            out_lines.append(line)
            continue

        # COORDINATES
        if in_coords:

            if line.strip().startswith("@") and "@COORDINATES" not in line:
                in_coords = False
                out_lines.append(line)
                continue

            parts = line.split()

            if len(parts) >= 4:
                try:
                    vec = np.array(list(map(float, parts[1:4])))
                    new = vec @ R.T + t

                    parts[1] = f"{new[0]:.8f}"
                    parts[2] = f"{new[1]:.8f}"
                    parts[3] = f"{new[2]:.8f}"

                    out_lines.append(" ".join(parts) + "\n")
                except ValueError:
                    out_lines.append(line)
            else:
                out_lines.append(line)

            continue

        # MULTIPOLES
        if in_mult:

            if line.strip().startswith("@") and "@MULTIPOLES" not in line:
                in_mult = False
                current_order = None
                out_lines.append(line)
                continue

            # ORDER header
            if line.startswith("ORDER"):
                out_lines.append(line)
                current_order = int(line.split()[1])
                skip_count = 1
                continue

            # number of sites
            if skip_count > 0:
                out_lines.append(line)
                skip_count -= 1
                continue

            parts = line.split()

            if len(parts) == 0:
                out_lines.append(line)
                continue

            # ORDER 0 (charges)
            if current_order == 0:
                out_lines.append(line)
                continue

            # ORDER 1 (dipoles)
            if current_order == 1 and len(parts) >= 4:

                mu = np.array(list(map(float, parts[1:4])))
                mu_new = R @ mu 

                parts[1] = f"{mu_new[0]:.8f}"
                parts[2] = f"{mu_new[1]:.8f}"
                parts[3] = f"{mu_new[2]:.8f}"

                out_lines.append(" ".join(parts) + "\n")
                continue

            # ORDER 2 (quadrupoles)
            if current_order == 2 and len(parts) >= 7:

                vals = list(map(float, parts[1:7]))

                Q = np.array([
                    [vals[0], vals[1], vals[2]],
                    [vals[1], vals[3], vals[4]],
                    [vals[2], vals[4], vals[5]]
                ])

                Q_new = R @ Q @ R.T

                vals_new = [
                    Q_new[0,0],
                    Q_new[0,1],
                    Q_new[0,2],
                    Q_new[1,1],
                    Q_new[1,2],
                    Q_new[2,2]
                ]

                for i in range(6):
                    parts[i+1] = f"{vals_new[i]:.8f}"

                out_lines.append(" ".join(parts) + "\n")
                continue

            out_lines.append(line)
            continue

        # POLARIZABILITIES
        if in_pol:

            if line.strip().startswith("@") and "@POLARIZABILITIES" not in line:
                in_pol = False
                current_order = None
                out_lines.append(line)
                continue

            if line.startswith("ORDER"):
                out_lines.append(line)
                current_order = None
                skip_count = 1
                continue

            if skip_count > 0:
                out_lines.append(line)
                skip_count -= 1
                continue

            parts = line.split()

            if len(parts) >= 7:

                vals = list(map(float, parts[1:7]))

                A = np.array([
                    [vals[0], vals[1], vals[2]],
                    [vals[1], vals[3], vals[4]],
                    [vals[2], vals[4], vals[5]]
                ])

                A_new = R @ A @ R.T

                vals_new = [
                    A_new[0,0],
                    A_new[0,1],
                    A_new[0,2],
                    A_new[1,1],
                    A_new[1,2],
                    A_new[2,2]
                ]

                for i in range(6):
                    parts[i+1] = f"{vals_new[i]:.8f}"

                out_lines.append(" ".join(parts) + "\n")
                continue

            out_lines.append(line)
            continue

        # DEFAULT
        out_lines.append(line)

    with open(out_pot, "w") as f:
        f.writelines(out_lines)

def stage_and_align(
    xyz_files,
    pe_enabled=False,
    output_directory="aligned_geometries",
):
    """
    Align molecular geometries to a common reference geometry.

    If PE is enabled, also rotate and translate the corresponding
    polarizable embedding potentials.

    Parameters
    ----------
    xyz_files : list
        List of XYZ geometry file paths.
    pe_enabled : bool
        Whether polarizable embedding is enabled.
    output_directory : str
        Directory where aligned geometries and PE potentials
        will be written.

    Returns
    -------
    aligned_files : list
        Paths to aligned XYZ geometries.
    """

    from pathlib import Path

    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    if not xyz_files:
        raise ValueError("No XYZ geometries provided.")

    # Reference geometry (first snapshot)
    P_ref, _ = read_xyz(xyz_files[0])

    aligned_files = []

    for xyz_file in xyz_files:

        xyz_file = Path(xyz_file)

        # Read geometry
        P_snap, atoms = read_xyz(xyz_file)

        # Calculate alignment
        P_aligned, R, resid, t = align_structure(
            P_snap,
            P_ref
        )

        # Output filenames
        output_xyz = output_directory / f"{xyz_file.stem}_aligned.xyz"

        # Write aligned geometry
        with open(output_xyz, "w") as f:

            f.write(f"{len(atoms)}\n")
            f.write("\n")

            for atom, coord in zip(atoms, P_aligned):
                f.write(
                    f"{atom} "
                    f"{coord[0]:.10f} "
                    f"{coord[1]:.10f} "
                    f"{coord[2]:.10f}\n"
                )

        aligned_files.append(str(output_xyz))

        # Transform PE potential if enabled
        if pe_enabled:

            pot_file = xyz_file.with_suffix(".pot")

            if not pot_file.exists():
                raise FileNotFoundError(
                    f"PE potential not found: {pot_file}"
                )

            output_pot = output_directory / f"{xyz_file.stem}_aligned.pot"

            transform_pot(
                pot_file,
                output_pot,
                R,
                t
            )

        print(
            f"[{xyz_file.stem}] aligned | "
            f"RMSD = {np.sqrt(resid):.6f}"
        )

    return aligned_files

def run_algorithm(
    orbitals,
    pe_enabled=False,
):
    """
    Run active-space selection algorithm.

    Parameters
    ----------
    orbitals : list
        Orbital information generated by run_guess_orbitals().
    pe_enabled : bool
        Whether polarizable embedding is enabled.
    """

    if not orbitals:
        raise ValueError("No orbital information provided.")

    # Determine algorithm type
    if len(orbitals) == 1:
        alg_type = "initial"

    else:
        alg_type = "sequential"

    print(f"Active space algorithm: {alg_type}")
    print(f"Polarizable embedding: {pe_enabled}")

    # Extract geometry files
    xyz_files = [
        orbital["geometry"]
        for orbital in orbitals
    ]

    # Stage and align geometries
    aligned_files = stage_and_align(
        xyz_files=xyz_files,
        pe_enabled=pe_enabled,
    )

    # ACTIVE SPACE SELECTION
    # To be implemented

    results = {
        "algorithm": alg_type,
        "pe_enabled": pe_enabled,
        "aligned_geometries": aligned_files,
    }

    return results