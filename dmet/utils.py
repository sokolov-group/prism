# Copyright 2026 Prism Developers. All Rights Reserved.
# Adapted from QC-DMET (Copyright 2015 Sebastian Wouters)
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
# Authors: Bryce Pickett <pickettosu@gmail.com>
#

import os
import sys
from contextlib import contextmanager

import numpy as np


@contextmanager
def silent_stdout():
    # Suppress C-level and Python-level stdout, restoring it safely on exceptions.
    sys.stdout.flush()
    old_fd = os.dup(sys.stdout.fileno())
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, sys.stdout.fileno())
        yield
    finally:
        sys.stdout.flush()
        os.dup2(old_fd, sys.stdout.fileno())
        os.close(old_fd)
        os.close(devnull)


def auto_nfrozen(mf, cutoff=-2.0):
    """Frozen-core count for an embedded problem, from the orbital energies.

    The embedded orbitals are impurity and bath orbitals, not atomic ones, so a
    per-atom core count does not apply to them. Orbitals below the cutoff are core.
    The default sits in the gap between the shallowest core and the deepest valence
    orbital for main-group elements: C 1s -11.3 against 2s -0.71, O 1s -20.7 against
    2s -1.24, Mg 2p -2.28 against 3s -0.25.
    """
    occupied = np.asarray(mf.mo_occ) > 0
    return int(np.sum(np.asarray(mf.mo_energy)[occupied] < cutoff))
