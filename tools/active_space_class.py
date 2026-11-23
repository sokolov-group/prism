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
#          Haden Dickerson <haden.dickerson423@outlook.com>


import glob
import os
import re

import numpy as np
import pyscf


class ActiveSpace:
    """Generate and align orbitals and geometries for active-space selection."""
    #TODO: Generalize so it works with SHARC/NAMD output
    def __init__(
        self,
        directory=None,
        manual=False,
        min_geometry=None,
        max_geometry=None,
        method="rhf",
        basis="cc-pVDZ",
        alg_method="initial",
        pe_enabled=False,
        output_directory="aligned_geometries",
    ):
        """
        Initialize an active-space calculation workflow.

        Parameters
        ----------
        directory : str, optional
            Directory containing XYZ geometry files.
        manual : bool, optional
            Whether to restrict geometry selection to a numerical range.
        min_geometry : int, optional
            First geometry index (inclusive).
        max_geometry : int, optional
            Last geometry index (inclusive).
        method : str, optional
            Electronic structure method used to generate orbitals.
        basis : str, optional
            Gaussian basis set.
        alg_method : str, optional
            Orbital generation mode: ``"initial"`` or ``"sequential"``.
        pe_enabled : bool, optional
            Whether to transform polarizable embedding potentials.
        output_directory : str, optional
            Directory for aligned geometries and potentials.
        """
        self.directory = directory
        self.manual = manual
        self.min = min_geometry
        self.max = max_geometry
        self.method = method
        self.basis = basis
        self.alg_method = alg_method
        self.pe_enabled = pe_enabled
        self.output_directory = output_directory
        self.xyz_files = []
        self.orbitals = []

    def get_geometries(self, directory=None, manual=None, min=None, max=None):
        """
        Identify XYZ geometries for active-space selection.

        Parameters
        ----------
        directory : str, optional
            Directory containing XYZ files.
        manual : bool, optional
            If True, manually restrict geometries using ``min`` and ``max``.
        min : int, optional
            First geometry index (inclusive).
        max : int, optional
            Last geometry index (inclusive).

        Returns
        -------
        list
            Sorted list of XYZ file paths.
        """
        if directory is None:
            directory = self.directory
        if manual is None:
            manual = self.manual
        if min is None:
            min = self.min
        if max is None:
            max = self.max

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

        xyz_files = glob.glob(os.path.join(directory, "*.xyz"))

        if not xyz_files:
            raise ValueError(
                f"No XYZ files found in geometry directory '{directory}'."
            )

        def numerical_key(filepath):
            filename = os.path.basename(filepath)
            return [
                int(part) if part.isdigit() else part.lower()
                for part in re.findall(r"\d+|\D+", filename)
            ]

        xyz_files.sort(key=numerical_key)

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

        self.xyz_files = xyz_files
        return xyz_files

    def build_molecule(self, xyz_file, basis=None):
        """Build a PySCF molecule from an XYZ file."""
        if basis is None:
            basis = self.basis

        mol = pyscf.gto.Mole()
        mol.atom = xyz_file
        mol.basis = basis
        mol.verbose = 4
        mol.build()
        return mol

    def run_guess_orbitals(
        self,
        xyz_files=None,
        method=None,
        basis=None,
        alg_method=None,
    ):
        """
        Generate molecular orbital guesses for a collection of geometries.

        Parameters
        ----------
        xyz_files : list, optional
            XYZ geometry files.
        method : str, optional
            Electronic structure method used to generate orbitals.
        basis : str, optional
            Gaussian basis set.
        alg_method : str, optional
            ``"initial"`` or ``"sequential"`` orbital generation mode.

        Returns
        -------
        list
            Molecular orbital information for each selected geometry.
        """
        if xyz_files is None:
            xyz_files = self.xyz_files
        if method is None:
            method = self.method
        if basis is None:
            basis = self.basis
        if alg_method is None:
            alg_method = self.alg_method

        if not xyz_files:
            raise ValueError("No XYZ geometries provided.")
        if alg_method not in ("initial", "sequential"):
            raise ValueError("alg_method must be initial or sequential.")
        results = []

        if alg_method == "initial":
            xyz_file = xyz_files[0]
            mol = self.build_molecule(xyz_file, basis)

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

        elif alg_method == "sequential":
            for xyz_file in xyz_files:
                mol = self.build_molecule(xyz_file, basis)

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

        self.orbitals = results
        return results

    @staticmethod
    def read_xyz(xyz_file):
        """Read coordinates and atom labels from an XYZ file."""
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

    @staticmethod
    def build_quaternion_matrix(M):
        """Build the quaternion matrix used for structure alignment."""
        S_W = np.zeros((4, 4))

        S_W[0, 0] = M[0, 0] + M[1, 1] + M[2, 2]
        S_W[1, 0] = M[1, 2] - M[2, 1]
        S_W[2, 0] = M[2, 0] - M[0, 2]
        S_W[3, 0] = M[0, 1] - M[1, 0]

        S_W[0, 1] = M[1, 2] - M[2, 1]
        S_W[1, 1] = M[0, 0] - M[1, 1] - M[2, 2]
        S_W[2, 1] = M[0, 1] + M[1, 0]
        S_W[3, 1] = M[0, 2] + M[2, 0]

        S_W[0, 2] = M[2, 0] - M[0, 2]
        S_W[1, 2] = M[0, 1] + M[1, 0]
        S_W[2, 2] = -M[0, 0] + M[1, 1] - M[2, 2]
        S_W[3, 2] = M[1, 2] + M[2, 1]

        S_W[0, 3] = M[0, 1] - M[1, 0]
        S_W[1, 3] = M[0, 2] + M[2, 0]
        S_W[2, 3] = M[1, 2] + M[2, 1]
        S_W[3, 3] = -M[0, 0] - M[1, 1] + M[2, 2]

        return S_W

    @staticmethod
    def quaternion_to_rotation_matrix(quat):
        """Convert a quaternion to a SciPy rotation matrix."""
        from scipy.spatial.transform import Rotation as R

        quat = quat / np.linalg.norm(quat + 1e-15)
        return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

    def align_structure(self, X, Y):
        """Rigidly align coordinates ``X`` onto reference coordinates ``Y``."""
        com_X = X.mean(axis=0)
        com_Y = Y.mean(axis=0)

        Xc = X - com_X
        Yc = Y - com_Y

        M = Xc.T @ Yc
        S_W = self.build_quaternion_matrix(M)

        eigval, eigvec = np.linalg.eigh(S_W)
        quat = eigvec[:, np.argmax(eigval)]

        R = self.quaternion_to_rotation_matrix(quat)

        t = com_Y - com_X @ R.T

        def rigid_transform(P):
            return P @ R.T + t

        X_aligned = rigid_transform(X)

        Gx = np.trace(Xc.T @ Xc)
        Gy = np.trace(Yc.T @ Yc)

        resid = np.mean(np.sum((X_aligned - Y)**2, axis=1))

        return X_aligned, R, resid, t

    @staticmethod
    def transform_pot(in_pot, out_pot, R, t):
        """Transform coordinates and tensors in a polarizable embedding file."""
        with open(in_pot, "r") as f:
            lines = f.readlines()

        out_lines = []

        in_coords = False
        in_mult = False
        in_pol = False

        current_order = None
        skip_count = 0

        for line in lines:
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

            if in_mult:
                if line.strip().startswith("@") and "@MULTIPOLES" not in line:
                    in_mult = False
                    current_order = None
                    out_lines.append(line)
                    continue

                if line.startswith("ORDER"):
                    out_lines.append(line)
                    current_order = int(line.split()[1])
                    skip_count = 1
                    continue

                if skip_count > 0:
                    out_lines.append(line)
                    skip_count -= 1
                    continue

                parts = line.split()

                if len(parts) == 0:
                    out_lines.append(line)
                    continue

                if current_order == 0:
                    out_lines.append(line)
                    continue

                if current_order == 1 and len(parts) >= 4:
                    mu = np.array(list(map(float, parts[1:4])))
                    mu_new = R @ mu

                    parts[1] = f"{mu_new[0]:.8f}"
                    parts[2] = f"{mu_new[1]:.8f}"
                    parts[3] = f"{mu_new[2]:.8f}"

                    out_lines.append(" ".join(parts) + "\n")
                    continue

                if current_order == 2 and len(parts) >= 7:
                    vals = list(map(float, parts[1:7]))

                    Q = np.array([
                        [vals[0], vals[1], vals[2]],
                        [vals[1], vals[3], vals[4]],
                        [vals[2], vals[4], vals[5]],
                    ])

                    Q_new = R @ Q @ R.T

                    vals_new = [
                        Q_new[0, 0], Q_new[0, 1], Q_new[0, 2],
                        Q_new[1, 1], Q_new[1, 2], Q_new[2, 2],
                    ]

                    for i in range(6):
                        parts[i+1] = f"{vals_new[i]:.8f}"

                    out_lines.append(" ".join(parts) + "\n")
                    continue

                out_lines.append(line)
                continue

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
                        [vals[2], vals[4], vals[5]],
                    ])

                    A_new = R @ A @ R.T

                    vals_new = [
                        A_new[0, 0], A_new[0, 1], A_new[0, 2],
                        A_new[1, 1], A_new[1, 2], A_new[2, 2],
                    ]

                    for i in range(6):
                        parts[i+1] = f"{vals_new[i]:.8f}"

                    out_lines.append(" ".join(parts) + "\n")
                    continue

                out_lines.append(line)
                continue

            out_lines.append(line)

        with open(out_pot, "w") as f:
            f.writelines(out_lines)

    def stage_and_align(self, xyz_files=None, pe_enabled=None, output_directory=None):
        """
        Align molecular geometries to a common reference geometry.

        If polarizable embedding is enabled, corresponding potentials are
        also rotated and translated.

        Parameters
        ----------
        xyz_files : list, optional
            XYZ geometry file paths.
        pe_enabled : bool, optional
            Whether polarizable embedding is enabled.
        output_directory : str, optional
            Directory for aligned geometries and potentials.

        Returns
        -------
        list
            Paths to aligned XYZ geometries.
        """
        from pathlib import Path

        if xyz_files is None:
            xyz_files = self.xyz_files
        if pe_enabled is None:
            pe_enabled = self.pe_enabled
        if output_directory is None:
            output_directory = self.output_directory

        output_directory = Path(output_directory)
        output_directory.mkdir(exist_ok=True, parents=True)

        if not xyz_files:
            raise ValueError("No XYZ geometries provided.")

        P_ref, reference_atoms = self.read_xyz(xyz_files[0])
        aligned_files = []

        for xyz_file in xyz_files:
            xyz_file = Path(xyz_file)

            P_snap, atoms = self.read_xyz(xyz_file)
            if atoms != reference_atoms:
                raise ValueError("All geometries must have the same atom ordering.")
            P_aligned, R, resid, t = self.align_structure(P_snap, P_ref)

            output_xyz = output_directory / f"{xyz_file.stem}_aligned.xyz"

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

            if pe_enabled:
                pot_file = xyz_file.with_suffix(".pot")

                if not pot_file.exists():
                    raise FileNotFoundError(
                        f"PE potential not found: {pot_file}"
                    )

                output_pot = output_directory / f"{xyz_file.stem}_aligned.pot"

                self.transform_pot(pot_file, output_pot, R, t)

            print(
                f"[{xyz_file.stem}] aligned | "
                f"RMSD = {np.sqrt(resid):.6f}"
            )

        self.aligned_files = aligned_files
        return aligned_files

    @staticmethod
    def analyze_overlap(overlap, ncore, ncas):
        """Section 2.2 of Cardenas and Nogueira, doi:10.1002/qua.26533.

        Rows label reference MOs; columns label sampled MOs. Both use the
        same contiguous active block. All returned indices are zero-based.
        Absolute overlaps (not squared overlaps) determine matches. Duplicate
        column matches retain the strongest candidate. Row repair visits active
        reference rows in order and stops when the swap lists balance.
        Unresolved cases are reported, never silently truncated or reassigned.
        """
        values = np.abs(np.asarray(overlap))
        if (values.ndim != 2 or values.shape[0] != values.shape[1]
                or not np.isfinite(values).all()):
            raise ValueError("Expected a finite square full-MO overlap matrix.")
        if (not isinstance(ncore, (int, np.integer))
                or not isinstance(ncas, (int, np.integer))
                or ncore < 0 or ncas < 1 or ncore + ncas > len(values)):
            raise ValueError("Invalid active-space bounds.")
        active = set(range(ncore, ncore + ncas))
        column_rows = values.argmax(axis=0)
        # Keep the strongest column for every reference row (stable tie break).
        matches = {}
        for col, row in enumerate(column_rows):
            if row not in matches or values[row, col] > values[row, matches[row]]:
                matches[int(row)] = col
        add, remove = set(), set()
        for row, col in matches.items():
            if row in active and col not in active:
                add.add(col)
            elif row not in active and col in active:
                remove.add(col)
        column_add, column_remove = sorted(add), sorted(remove)
        for row in sorted(active):
            if len(add) == len(remove):
                break
            col = int(values[row].argmax())
            if len(remove) > len(add):
                if col not in active:
                    add.add(col)
                else:
                    remove.discard(col)
            else:
                previous = matches.get(row)
                if previous is None:
                    continue
                if col in active and previous not in active:
                    add.discard(previous)
                elif col not in active and previous in active:
                    remove.add(previous)
        return {
            "add": sorted(add), "remove": sorted(remove),
            "balanced": len(add) == len(remove),
            "column_add": column_add, "column_remove": column_remove,
            "column_matches": matches,
        }

    @staticmethod
    def mo_overlap(reference, sample):
        """Equation (15), using PySCF cross-geometry AO integrals.

        Records contain ``mol`` and full ``mo_coeff`` arrays computed in that
        molecule's coordinate frame. Align geometries BEFORE computing orbitals;
        moving atoms afterward without rotating AO coefficients is invalid.
        """
        ref_mol, mol = reference["mol"], sample["mol"]
        if ([ref_mol.atom_symbol(i) for i in range(ref_mol.natm)] !=
                [mol.atom_symbol(i) for i in range(mol.natm)]):
            raise ValueError("Reference and sample must have identical atom ordering.")
        if ref_mol.cart != mol.cart or ref_mol._basis != mol._basis:
            raise ValueError("Reference and sample must use the same AO basis.")
        left, right = np.asarray(reference["mo_coeff"]), np.asarray(sample["mo_coeff"])
        if left.shape != (ref_mol.nao_nr(), ref_mol.nao_nr()) or right.shape != left.shape:
            raise ValueError("Full square MO coefficient matrices are required.")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("MO coefficients must be finite.")
        cross = pyscf.gto.intor_cross("int1e_ovlp", ref_mol, mol)
        return left.conj().T @ cross @ right

    def run_algorithm(self, orbitals=None, pe_enabled=None, *, reference=None,
                      ncore=None, ncas=None, optimize=None, max_cycle=10):
        """Check and optionally correct optimized CASSCF active spaces.

        ``orbitals`` is a list of records with ``mol``, ``mo_coeff`` and
        ``converged``; optional metadata (e.g. geometry) is preserved. ``reference``
        is a separately selected, optimized reference record (defaults to the
        first record). Its active space must already be chemically validated.
        ``ncore`` and ``ncas`` specify the common contiguous active block.

        ``optimize(record, swapped_coeff)`` must run a fresh CASSCF optimization
        and return an updated record, including its convergence flag. Configure
        identical electron counts/state weights and embedding in this callback.
        Without a callback this method only diagnoses/proposes swaps. The fixed
        reference never changes. At most ``max_cycle`` reoptimizations per sample
        are performed; unsuccessful cases return ``converged=False``.

        History entries include ``swap_pairs`` as zero-based (remove, add)
        coefficient-column indices before that correction and ``swaps_applied``
        indicating whether the permuted coefficients were passed to optimize.
        These indices are local to each iteration, not permanent MO identities.

        Alignment is deliberately separate: use ``stage_and_align`` before SCF
        and CASSCF. ``pe_enabled`` records metadata, not an embedding Hamiltonian.
        """
        if orbitals is None:
            orbitals = self.orbitals
        if not orbitals:
            raise ValueError("No optimized orbital records provided.")
        if ncore is None or ncas is None:
            raise ValueError("Specify ncore and ncas for the optimized CASSCF orbitals.")
        if not isinstance(max_cycle, int) or max_cycle < 0:
            raise ValueError("max_cycle must be a non-negative integer.")
        if reference is None:
            reference = orbitals[0]
        if not reference.get("converged", False):
            raise ValueError("The reference CASSCF calculation must be converged.")
        results = []
        for record in orbitals:
            current = dict(record)
            history = []
            for cycle in range(max_cycle + 1):
                if not current.get("converged", False):
                    status = "casscf_not_converged"
                    break
                overlap = self.mo_overlap(reference, current)
                analysis = self.analyze_overlap(overlap, ncore, ncas)
                analysis["swap_pairs"] = (list(zip(analysis["remove"], analysis["add"]))
                                          if analysis["balanced"] else [])
                analysis["swaps_applied"] = False
                history.append(analysis)
                if not analysis["balanced"]:
                    status = "unbalanced_matches"
                    break
                if not analysis["add"]:
                    status = "matched"
                    break
                if optimize is None or cycle == max_cycle:
                    status = "swaps_required" if optimize is None else "max_cycle"
                    break
                coeff = np.array(current["mo_coeff"], copy=True)
                for outgoing, incoming in analysis["swap_pairs"]:
                    print(f"[{current.get('geometry', 'sample')}] correction {cycle}: "
                          f"remove active MO {outgoing + 1}, add external MO {incoming + 1}; "
                          f"swap coefficient columns [{outgoing}, {incoming}] (zero-based)",
                          flush=True)
                    coeff[:, [outgoing, incoming]] = coeff[:, [incoming, outgoing]]
                analysis["swaps_applied"] = True
                current = dict(optimize(current, coeff))
            results.append({"orbitals": current, "history": history,
                            "status": status, "converged": status == "matched"})
        self.results = {"algorithm": "reference_overlap",
                        "pe_enabled": self.pe_enabled if pe_enabled is None else pe_enabled,
                        "geometries": results,
                        "converged": all(item["converged"] for item in results)}
        return self.results
