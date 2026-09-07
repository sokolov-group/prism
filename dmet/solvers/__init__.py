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
# Authors: Bryce Pickett <pickettosu@gmail.com>
#


# QD/PC-NEVPT2 excluded: their result objects cannot be pickled back from a worker.
PARALLEL_ELIGIBLE = frozenset({
    'ED', 'FCI', 'flag_rhf',
    'CASSCF',
})


def _run_rhf(task):
    from prism.dmet.solvers import rhf
    energy, rdm1 = rhf.execute(task)
    return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}


def _run_fci(task):
    from prism.dmet.solvers import fci
    energy, rdm1 = fci.execute(task)
    return {'counter': task['counter'], 'energy': energy, 'rdm1': rdm1}


def _run_casscf(task):
    from prism.dmet.solvers import casscf
    energy, rdm1, cas_res = casscf.execute(task)
    return {'counter': task['counter'], 'energy': energy,
            'rdm1': rdm1, 'cas_res': cas_res}


def _run_qdnevpt2(task):
    from prism.dmet.solvers import qdnevpt2
    energy, rdm1, qdnevpt2_res = qdnevpt2.execute(task)
    return {'counter': task['counter'], 'energy': energy,
            'rdm1': rdm1, 'qdnevpt2_res': qdnevpt2_res}


def _run_mradc(task):
    from prism.dmet.solvers import mradc
    energy, rdm1, mradc_res = mradc.execute(task)
    return {'counter': task['counter'], 'energy': energy,
            'rdm1': rdm1, 'mradc_res': mradc_res}


def _run_pcnevpt2(task):
    from prism.dmet.solvers import pcnevpt2
    energy, rdm1, pcnevpt2_res = pcnevpt2.execute(task)
    return {'counter': task['counter'], 'energy': energy,
            'rdm1': rdm1, 'pcnevpt2_res': pcnevpt2_res}


SOLVERS = {
    'flag_rhf': _run_rhf,
    'ED': _run_fci,
    'FCI': _run_fci,
    'CASSCF': _run_casscf,
    'QD-NEVPT2': _run_qdnevpt2,
    'PC-NEVPT2': _run_pcnevpt2,
    'MR-ADC': _run_mradc,
}


def execute(task):
    method = task['method']
    if method not in SOLVERS:
        raise ValueError(
            f"Unknown method='{method}'. Valid keys: {sorted(SOLVERS)}")
    return SOLVERS[method](task)
