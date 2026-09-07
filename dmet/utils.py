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
