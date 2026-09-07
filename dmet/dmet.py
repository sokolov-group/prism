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
import warnings
import concurrent.futures

import numpy as np
from scipy import optimize

import prism.lib.logger as logger
from prism.dmet import helper as dmet_helper
from prism.dmet import solvers
from prism.dmet.solvers import PARALLEL_ELIGIBLE
from prism.dmet.fragment_builder import FragmentBuilder


def _fragment_worker(task):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    return solvers.execute(task)


class DMET:

    # The keyword arguments are grouped by role in dmet/README.md.
    def __init__(self, integrals, fragments, is_translation_invariant, method='ED',
                 sc_method='LSTSQ', conv_tol=1e-5, max_cycle=200,
                 fit_impurity_and_bath=True, use_constrained_optimization=False,
                 use_density_embedding=False, use_density_embedding_no=False,
                 print_u=True, print_rdm=True,
                 ncas=None, nelecas=None, sa_nstates=1, sa_weights=None,
                 casscf_kwargs=None, qdnevpt2_kwargs=None,
                 pcnevpt2_kwargs=None, mradc_kwargs=None,
                 use_symmetry=False, symmetry_map=None,
                 parallel=False, max_workers=None, bath_tol=1e-13, n_bath_orbs=None,
                 bath_1rdm=None, core_occ_tol=None,
                 embedded_ref_dm=None, no_kernel=False,
                 keep_degenerate=False, deg_rtol=1e-6, print_bath_spectrum=False,
                 cas_select='energy',
                 embed_level_shift=0.0, scf_stability=False,
                 cas_spin=None, cas_spin_shift=0.2,
                 fragment_methods=None,
                 natorb_occ_thresh=0.02, natorb_max_superset=None,
                 deg_tol=1e-3, casci_conv_tol=1e-10,
                 verbose=None):

        self.ints = integrals
        self.norb = self.ints.norb
        self.fragments = fragments
        self.umat = np.zeros((self.norb, self.norb), dtype=float)
        self.relaxation = 0.0

        self.verbose = verbose if verbose is not None else self.ints.mol.verbose
        self.log = logger.Logger(self.ints.mol.stdout, self.verbose)

        self.method = method
        self.is_translation_invariant = is_translation_invariant
        self.sc_method = sc_method
        self.conv_tol = conv_tol        # u-matrix convergence for selfconsistent()
        self.max_cycle = max_cycle      # cap on selfconsistent() iterations
        self.ncas = ncas
        self.nelecas = nelecas
        self.sa_nstates = sa_nstates
        self.sa_weights = sa_weights
        self.casscf_kwargs = casscf_kwargs or {}
        self.cas_select = cas_select
        self.embed_level_shift = embed_level_shift   # static level shift on the embedded post-HF reference
        self.scf_stability = scf_stability
        self.cas_spin = cas_spin        # target 2S for CAS states; None disables the spin penalty
        self.cas_spin_shift = cas_spin_shift  # fix_spin_ penalty strength for off-target spins
        self.natorb_occ_thresh = natorb_occ_thresh    # natorb core-vs-active occupation cutoff
        self.natorb_max_superset = natorb_max_superset  # cap on the natorb superset window size
        self.deg_tol = deg_tol              # near-degenerate orbital energy tolerance
        self.casci_conv_tol = casci_conv_tol       # natorb superset CASCI fcisolver.conv_tol
        self.cas_results = []   # populated by do_exact() when method='CASSCF'
        self.qdnevpt2_kwargs = qdnevpt2_kwargs or {}
        self.qdnevpt2_results = []   # populated by do_exact() when method='QD-NEVPT2'
        self.pcnevpt2_kwargs = pcnevpt2_kwargs or {}
        self.pcnevpt2_results = []   # populated by do_exact() when method='PC-NEVPT2'
        self.mradc_kwargs = mradc_kwargs or {}
        self.mradc_results = []      # populated by do_exact() when method='MR-ADC'
        # Bath truncation: None = full symmetric bath; int = keep at most that many per fragment.
        if n_bath_orbs is None:
            self.num_bath_orbs = None
        elif hasattr(n_bath_orbs, '__len__'):
            if len(n_bath_orbs) != len(fragments):
                raise ValueError(
                    f"Length of n_bath_orbs ({len(n_bath_orbs)}) must match the number of "
                    f"fragments ({len(fragments)}); pass a single int to use one size for all.")
            self.num_bath_orbs = [int(x) for x in n_bath_orbs]
        else:
            self.num_bath_orbs = [int(n_bath_orbs)] * len(fragments)
        self.fit_imp_bath = fit_impurity_and_bath
        self.do_det = use_density_embedding
        self.do_det_no = use_density_embedding_no
        self.no_rotation = None
        self.alt_cost_func = use_constrained_optimization
        self.fragment_methods = dict(fragment_methods) if fragment_methods else {}
        for frag_idx, frag_method in self.fragment_methods.items():
            if not (0 <= frag_idx < len(fragments)):
                raise ValueError(
                    f"Invalid fragment_methods index {frag_idx}: out of range "
                    f"(have {len(fragments)} fragments).")
            if frag_method != 'RHF':
                raise ValueError(
                    f"Invalid fragment_methods value '{frag_method}' for fragment {frag_idx}: "
                    f"only 'RHF' is supported per fragment.")
        self.bath_tol = bath_tol
        self.core_occ_tol = core_occ_tol
        self.keep_degenerate = keep_degenerate
        self.deg_rtol = deg_rtol         # bath occupation degeneracy tolerance
        self.print_bath_spectrum = print_bath_spectrum
        # Externally supplied bath density in the local basis, consumed by do_exact().
        if bath_1rdm is not None:
            if sc_method != 'NONE':
                raise ValueError(
                    f"bath_1rdm requires sc_method='NONE', got '{sc_method}'. The "
                    f"correlation-potential fit requires a density that varies with umat.")
            trace = np.trace(bath_1rdm)
            if abs(trace - self.ints.nelec) > 1e-6:
                raise ValueError(
                    f"bath_1rdm trace {trace:.6f} != nelec {self.ints.nelec}. It must be "
                    f"in the orthonormal local basis: ao2loc.T @ S @ dm_ao @ S @ ao2loc.")
        self.bath_1rdm = bath_1rdm
        # Reference density in the AO basis, projected per fragment and used as the
        # embedded reference when no_kernel is set.
        if embedded_ref_dm is not None:
            if sc_method != 'NONE':
                raise ValueError(
                    f"embedded_ref_dm requires sc_method='NONE', got '{sc_method}'. The "
                    f"correlation-potential fit requires a reference that varies with umat.")
            nelec_ref = np.trace(embedded_ref_dm @ self.ints.ovlp)
            if abs(nelec_ref - self.ints.nelec) > 1e-6:
                raise ValueError(
                    f"embedded_ref_dm holds {nelec_ref:.6f} electrons, expected "
                    f"{self.ints.nelec}. It must be in the AO basis, so that "
                    f"Tr(dm @ S) is the electron count.")
            if not no_kernel:
                raise ValueError(
                    "embedded_ref_dm requires no_kernel=True. An SCF started from it "
                    "refills the orbitals by aufbau and discards the supplied reference.")
        self.embedded_ref_dm = embedded_ref_dm
        self.no_kernel = no_kernel
        self.parallel = parallel

        self._validate_config()
        self._warn_inert_params()

        self.use_symmetry = use_symmetry
        self.symmetry_map = symmetry_map  # user-provided {child_idx: parent_idx} or None
        if self.use_symmetry and self.symmetry_map is None and not is_translation_invariant:
            self.symmetry_map = self._auto_detect_symmetry()

        if max_workers is not None:
            self.max_workers = max_workers
        else:
            _env_threads = os.environ.get('PRISM_DMET_WORKERS',
                           os.environ.get('SLURM_CPUS_PER_TASK',
                           os.environ.get('OMP_NUM_THREADS', '1')))
            self.max_workers = max(1, int(_env_threads))

        maxiter_frags = 1 if is_translation_invariant else len(fragments)
        self.frag_caches = [None] * maxiter_frags

        self.min_func = None
        if self.alt_cost_func:
            self.min_func = 'FOCK_INIT'  # 'oei'
            if self.fit_imp_bath:
                raise ValueError(
                    "Invalid combination: use_constrained_optimization=True is incompatible with "
                    "fit_impurity_and_bath=True; set fit_impurity_and_bath=False instead.")
            if self.do_det:
                raise ValueError(
                    "Invalid combination: use_constrained_optimization=True is incompatible with "
                    "use_density_embedding=True.")
            if self.sc_method not in {'BFGS', 'NONE'}:
                raise ValueError(
                    "Invalid sc_method for use_constrained_optimization=True: requires 'BFGS' or 'NONE'.")

        if self.do_det:
            self.fit_imp_bath = False
            if self.do_det_no:
                self.no_vecs = None
                self.no_diag = None

        self.print_u = print_u
        self.print_rdm = print_rdm

        all_one = self._check_complete_tiling()
        if not all_one and self.is_translation_invariant:
            # Incomplete tiling: impurity orbitals must be the first in the Hamiltonian.
            raise ValueError(
                "Translation-invariant DMET requires the fragments to tile every orbital.")

        self.energy = 0.0
        self.imp_rdm1 = []
        self.dmet_orbs = []
        self.core_1rdm_loc = []
        self.bath_spectrum = []
        self.frag_energies = []
        self.imp_size = self.make_imp_size()
        self.mu_imp = 0.0
        self.mask = self.make_mask()
        self.helper = dmet_helper.DMETHelper(
            self.ints, self.make_h1_terms(), self.alt_cost_func, self.min_func, log=self.log)

    def _warn_inert_params(self):
        _cas_methods = {'CASSCF', 'QD-NEVPT2', 'PC-NEVPT2', 'MR-ADC'}
        if self.no_kernel:
            if self.method not in _cas_methods:
                warnings.warn(
                    f"no_kernel is inert for method='{self.method}' "
                    f"(only {sorted(_cas_methods)} build an embedded SCF).", UserWarning)
            elif self.scf_stability:
                warnings.warn(
                    "scf_stability is skipped when no_kernel=True; no embedded SCF "
                    "solution is produced to test.", UserWarning)
        if self.method in _cas_methods and self.cas_select == 'energy':
            warnings.warn(
                "cas_select='energy' (the default) selects the active space by "
                "orbital energy; for production use cas_select='natorb'.", UserWarning)

    def _validate_config(self):
        # Fail fast at construction on user-reachable misconfiguration.
        if len(self.fragments) == 0:
            raise ValueError("At least one fragment is required; got an empty list.")

        if self.is_translation_invariant and not self.ints.ti_ok:
            raise ValueError(
                "Translation-invariant DMET requires a TI-capable LocalIntegrals "
                "(ti_ok=True); the chosen localization sets ti_ok=False.")

        _valid_methods = {'ED', 'FCI', 'CASSCF', 'QD-NEVPT2', 'PC-NEVPT2', 'MR-ADC'}
        if self.method not in _valid_methods:
            raise ValueError(
                f"Unknown method='{self.method}'. Valid: {sorted(_valid_methods)}")

        if self.sc_method not in {'LSTSQ', 'BFGS', 'NONE'}:
            raise ValueError(
                f"Unknown sc_method='{self.sc_method}'. Valid: ['BFGS', 'LSTSQ', 'NONE']")

        _valid_cas = {'energy', 'natorb'}
        if self.cas_select not in _valid_cas:
            raise ValueError(
                f"Unknown cas_select='{self.cas_select}'. Valid: {sorted(_valid_cas)}")

        if self.method in ('QD-NEVPT2', 'PC-NEVPT2', 'MR-ADC') \
                and (self.ncas is None or self.nelecas is None):
            raise ValueError(
                f"Method '{self.method}' requires ncas and nelecas (active space size).")

        if self.method == 'QD-NEVPT2' and self.sa_nstates < 2:
            raise ValueError(
                "Method 'QD-NEVPT2' requires sa_nstates >= 2 for state-averaging.")

        if self.method == 'MR-ADC' and self.sa_nstates > 1:
            raise ValueError(
                "Method 'MR-ADC' requires sa_nstates=1. Prism MR-ADC takes a casscf or "
                "casci reference; a state-averaged CASSCF is reported as sa-casscf and "
                "is rejected. Request roots with mradc_kwargs={'nroots': n}.")

        if self.parallel and self.method in ('QD-NEVPT2', 'PC-NEVPT2', 'MR-ADC'):
            raise ValueError(
                f"Method '{self.method}' does not support parallel=True: it returns the "
                f"CASSCF and Prism objects needed for analysis, and those cannot be sent "
                f"back from a worker process. Use parallel=False.")

        for frag_idx, fragment in enumerate(self.fragments):
            if np.any(np.asarray(fragment) < 0):
                raise ValueError(
                    f"Fragment {frag_idx} has a negative orbital mask. Negative masks are "
                    f"not supported; pass fragment_methods={{{frag_idx}: 'RHF'}} to solve a "
                    f"fragment at the RHF level.")

    def _check_complete_tiling(self):
        covered = np.zeros((self.norb,), dtype=int)
        for fragment in self.fragments:
            covered += np.abs(fragment)
        if np.any(covered > 1):
            raise ValueError("Fragments overlap: an orbital belongs to more than one fragment.")
        all_one = np.all(covered == 1)
        return all_one

    def _auto_detect_symmetry(self):
        symmetry_map = {}
        seen = {}  # fingerprint -> first fragment index
        for idx, cluster in enumerate(self.fragments):
            fingerprint = int(np.sum(np.abs(cluster)))
            if fingerprint in seen:
                symmetry_map[idx] = seen[fingerprint]
            else:
                seen[fingerprint] = idx
        if symmetry_map:
            n_unique = len(self.fragments) - len(symmetry_map)
            self.log.info("Auto-detected %d unique fragment(s) out of %d total. "
                          "Skipping %d equivalent fragment solve(s)."
                          % (n_unique, len(self.fragments), len(symmetry_map)))
        return symmetry_map

    def make_imp_size(self):
        imp_sizes = []
        maxiter = len(self.fragments)
        if self.is_translation_invariant:
            maxiter = 1
        for frag_idx in range(maxiter):
            impurity_orbs = np.abs(self.fragments[frag_idx])
            num_imp_orbs = np.sum(impurity_orbs)
            imp_sizes.append(num_imp_orbs)
        imp_sizes = np.array(imp_sizes)
        return imp_sizes

    def make_h1_terms(self):
        h1_terms = []
        if self.do_det:  # DET (density embedding): fit the diagonal density only
            if self.is_translation_invariant:
                local_size = self.imp_size[0]
                for row in range(local_size):
                    h1 = np.zeros((self.norb, self.norb), dtype=int)
                    for jumper in range(self.norb // local_size):
                        jumpsquare = local_size * jumper
                        h1[jumpsquare + row, jumpsquare + row] = 1
                    h1_terms.append(h1)
            else:
                jumpsquare = 0
                for local_size in self.imp_size:
                    for row in range(local_size):
                        h1 = np.zeros((self.norb, self.norb), dtype=int)
                        h1[jumpsquare + row, jumpsquare + row] = 1
                        h1_terms.append(h1)
                    jumpsquare += local_size
        else:  # DMET (density-matrix embedding): fit the full density matrix
            if self.is_translation_invariant:
                local_size = self.imp_size[0]
                for row in range(local_size):
                    for col in range(row, local_size):
                        h1 = np.zeros((self.norb, self.norb), dtype=int)
                        for jumper in range(self.norb // local_size):
                            jumpsquare = local_size * jumper
                            h1[jumpsquare + row, jumpsquare + col] = 1
                            h1[jumpsquare + col, jumpsquare + row] = 1
                        h1_terms.append(h1)
            else:
                jumpsquare = 0
                for local_size in self.imp_size:
                    for row in range(local_size):
                        for col in range(row, local_size):
                            h1 = np.zeros((self.norb, self.norb), dtype=int)
                            h1[jumpsquare + row, jumpsquare + col] = 1
                            h1[jumpsquare + col, jumpsquare + row] = 1
                            h1_terms.append(h1)
                    jumpsquare += local_size
        return h1_terms

    def make_mask(self):
        mask = np.zeros((self.norb, self.norb), dtype=bool)
        if self.do_det:  # DET (density embedding)
            jump = 0
            for local_size in self.imp_size:  # imp_size has length 1 if is_translation_invariant
                for row in range(local_size):
                    mask[jump + row, jump + row] = True
                jump += local_size
        else:  # DMET (density-matrix embedding): fit the full density matrix
            jump = 0
            for local_size in self.imp_size:
                for row in range(local_size):
                    for col in range(row, local_size):
                        mask[jump + row, jump + col] = True
                jump += local_size
        return mask

    def do_exact(self, chempot_imp=0.0):
        one_rdm = (self.bath_1rdm if self.bath_1rdm is not None
                   else self.helper.construct_1rdm_loc(self.umat))
        self.energy = 0.0
        self.imp_rdm1 = []
        self.dmet_orbs = []
        self.core_1rdm_loc = []
        self.bath_spectrum = []
        self.frag_energies = []
        self.cas_results = []
        self.qdnevpt2_results = []
        self.pcnevpt2_results = []
        self.mradc_results = []
        if self.do_det and self.do_det_no:
            self.no_vecs = []
            self.no_diag = []

        maxiter = len(self.fragments)
        if self.is_translation_invariant:
            maxiter = 1

        remaining_orbs = np.ones((len(self.fragments[0]),), dtype=float)

        _frag_tasks = []
        _frag_meta = []

        # Spin-orbit coupling, NTOs and Dyson orbitals all need the full molecule.
        _nevpt_kwargs = {**(self.qdnevpt2_kwargs or {}), **(self.pcnevpt2_kwargs or {})}
        _needs_embedding_data = bool(
            _nevpt_kwargs.get('soc') or _nevpt_kwargs.get('compute_ntos')
            or (self.mradc_kwargs or {}).get('compute_dyson'))

        _builder = FragmentBuilder(
            ints = self.ints,
            helper = self.helper,
            fragments = self.fragments,
            method = self.method,
            num_bath_orbs = self.num_bath_orbs,
            bath_tol = self.bath_tol,
            fragment_methods = self.fragment_methods,
            core_occ_tol = self.core_occ_tol,
            keep_degenerate = self.keep_degenerate,
            deg_rtol = self.deg_rtol,
            needs_embedding_data = _needs_embedding_data,
            embedded_ref_dm = self.embedded_ref_dm,
        )

        for frag_idx in range(maxiter):

            # Symmetry skip: handled immediately without a solver.
            if (self.symmetry_map is not None and frag_idx in self.symmetry_map
                    and not self.is_translation_invariant):
                sym_desc = _builder.build_symmetry_bath(
                    frag_idx, one_rdm, self.symmetry_map[frag_idx])
                self.dmet_orbs.append(sym_desc['loc_2_dmet'][:, :sym_desc['norb_in_imp']])
                self.core_1rdm_loc.append(None)
                self.bath_spectrum.append(self.helper.bath_spectrum)
                if self.print_bath_spectrum:
                    self._dump_bath_spectrum(frag_idx, self.helper.bath_spectrum)
                _frag_meta.append({
                    'counter': frag_idx,
                    'sym_parent': sym_desc['sym_parent'],
                    'impurity_orbs': sym_desc['impurity_orbs'],
                })
                continue

            frag = _builder.build(frag_idx, one_rdm, chempot_imp)

            impurity_orbs = frag['impurity_orbs']
            num_imp_orbs = frag['num_imp_orbs']
            norb_in_imp = frag['norb_in_imp']
            nelec_in_imp = frag['nelec_in_imp']
            loc_2_dmet = frag['loc_2_dmet']
            core_1rdm_loc = frag['core_1rdm_loc']
            dmet_oei = frag['dmet_oei']
            dmet_fock = frag['dmet_fock']
            dmet_tei = frag['dmet_tei']
            dm_guess_rhf = frag['dm_guess_rhf']
            dip_mom_ao = frag['dip_mom_ao']
            _method_key = frag['method_key']

            # Populate dmet_orbs for bath-dump and cost-function use.
            self.dmet_orbs.append(loc_2_dmet[:, :norb_in_imp])
            self.core_1rdm_loc.append(core_1rdm_loc)
            self.bath_spectrum.append(frag['bath_spectrum'])
            if self.print_bath_spectrum:
                self._dump_bath_spectrum(frag_idx, frag['bath_spectrum'])

            self.log.note("Embedding a %d-orbital, %d-electron fragment cluster."
                          % (norb_in_imp, nelec_in_imp))

            _frag_meta.append({
                'counter': frag_idx,
                'sym_parent': None,
                'impurity_orbs': impurity_orbs,
                'num_imp_orbs': num_imp_orbs,
                'norb_in_imp': norb_in_imp,
                'nelec_in_imp': nelec_in_imp,
                'core_1rdm_loc': core_1rdm_loc,
                'method_key': _method_key,
                'loc_2_dmet': loc_2_dmet,
                'dmet_oei': dmet_oei,
                'dmet_fock': dmet_fock,
            })

            _mo_guess = None
            _ci_guess = None
            if _method_key == 'CASSCF' and self.frag_caches[frag_idx] is not None:
                cached = self.frag_caches[frag_idx]
                old_mo = cached['mo_coeff']
                _ci_guess = cached.get('ci', None)
                _ncas = self.ncas if self.ncas is not None else norb_in_imp
                _ncore = (nelec_in_imp - (self.nelecas if self.nelecas is not None
                                          else nelec_in_imp)) // 2
                if old_mo.shape == (norb_in_imp, norb_in_imp):
                    from prism.dmet.cas_selectors import project_amo_manually
                    _mo_guess, fidelity = project_amo_manually(
                        old_mo, _ncas, _ncore, dmet_fock, norb_in_imp, log=self.log)
                    if np.min(fidelity) < 0.5:
                        self.log.info("Low projection fidelity, discarding CI guess.")
                        _ci_guess = None
                else:
                    self.log.info("MO shape mismatch, starting fresh.")

            task = self._build_task(
                frag_idx, _method_key, dmet_oei, dmet_fock, dmet_tei,
                norb_in_imp, nelec_in_imp, num_imp_orbs, chempot_imp,
                dm_guess_rhf, mo_guess=_mo_guess, ci_guess=_ci_guess,
                dip_mom_ao=dip_mom_ao, embedding_data=frag['embedding_data'],
                embedded_ref=frag['embedded_ref'])

            _is_parallel_eligible = (
                self.parallel
                and _method_key in PARALLEL_ELIGIBLE
                and not self.do_det_no  # NO rotation requires in-process state
            )

            if _is_parallel_eligible:
                _frag_tasks.append(task)
            else:
                _frag_meta[-1]['sequential_result'] = self._run_fragment_sequential(task)

        _parallel_results = {}   # frag_idx -> result dict
        if _frag_tasks:
            _nw = min(self.max_workers, len(_frag_tasks))
            self.log.info("Submitting %d fragment(s) to %d worker process(es)."
                          % (len(_frag_tasks), _nw))
            with concurrent.futures.ProcessPoolExecutor(max_workers=_nw) as pool:
                futures = {pool.submit(_fragment_worker, t): t['counter'] for t in _frag_tasks}
                for fut in concurrent.futures.as_completed(futures):
                    res = fut.result()
                    _parallel_results[res['counter']] = res

        for meta in _frag_meta:
            frag_idx = meta['counter']
            impurity_orbs = meta['impurity_orbs']

            if meta.get('sym_parent') is not None:
                parent = meta['sym_parent']
                self.log.info("Fragment %d <- fragment %d (copied)."
                              % (frag_idx, parent))
                parent_energy = self.frag_energies[parent]
                parent_rdm = self.imp_rdm1[parent]
                self.energy += parent_energy
                self.frag_energies.append(parent_energy)
                self.imp_rdm1.append(parent_rdm.copy())
                remaining_orbs -= impurity_orbs
                continue

            num_imp_orbs = meta['num_imp_orbs']

            if frag_idx in _parallel_results:
                res = _parallel_results[frag_idx]
            else:
                res = meta['sequential_result']

            imp_energy = res['energy']
            imp_rdm1_frag = res['rdm1']

            if 'cas_res' in res:
                self.frag_caches[frag_idx] = {
                    'mo_coeff': res['cas_res']['mo_coeff'],
                    'ci': res['cas_res']['ci'],
                }
                self.cas_results.append(res['cas_res'])
            if 'qdnevpt2_res' in res:
                self.qdnevpt2_results.append(res['qdnevpt2_res'])
                self._attach_spin_pop_transform(res['qdnevpt2_res'], frag_idx)
            if 'mradc_res' in res:
                self.mradc_results.append(res['mradc_res'])
            if 'pcnevpt2_res' in res:
                self.pcnevpt2_results.append(res['pcnevpt2_res'])

            self.energy += imp_energy
            self.frag_energies.append(imp_energy)
            self.imp_rdm1.append(imp_rdm1_frag)
            if self.do_det and self.do_det_no:
                rdm_eigenvals, rdm_eigenvecs = np.linalg.eigh(
                    imp_rdm1_frag[:num_imp_orbs, :num_imp_orbs])
                self.no_vecs.append(rdm_eigenvecs)
                self.no_diag.append(rdm_eigenvals)

            remaining_orbs -= impurity_orbs

        if self.do_det and self.do_det_no:
            self.no_rotation = self.construct_no_rotation()

        n_electrons = 0.0
        for frag_idx in range(maxiter):
            n_electrons += np.trace(
                self.imp_rdm1[frag_idx][:self.imp_size[frag_idx], :self.imp_size[frag_idx]])
        if self.is_translation_invariant:
            n_electrons = n_electrons * len(self.fragments)
            self.energy = self.energy * len(self.fragments)
            remaining_orbs[:] = 0

        # Mean-field energy of orbitals no fragment covers (as in libDMET/Vayesta/mrh);
        # active_oei/active_fock already carry QM/MM and veff, so no environment SCF.
        if np.sum(remaining_orbs) != 0:
            if not np.array_equal(self.ints.active,
                                  np.ones((self.ints.mol.nao_nr(),), dtype=int)):
                raise RuntimeError(
                    "Incomplete-tiling energy correction requires a full active space.")

            imp_orbs = remaining_orbs == 1
            h_plus_f = self.ints.active_oei + self.ints.active_fock

            imp_energy_tail = \
                0.25 * np.einsum('ji,ij->', one_rdm[:, imp_orbs], h_plus_f[imp_orbs, :]) \
                + 0.25 * np.einsum('ji,ij->', one_rdm[imp_orbs, :], h_plus_f[:, imp_orbs])

            self.energy += imp_energy_tail
            n_electrons += np.trace(one_rdm[np.ix_(imp_orbs, imp_orbs)])

            remaining_orbs[remaining_orbs == 1] -= 1
        if not np.all(remaining_orbs == 0):
            raise RuntimeError("Some orbitals were not covered by any fragment.")

        self.energy += self.ints.const()
        return n_electrons

    def _attach_spin_pop_transform(self, qdnevpt2_res, frag_idx):
        # AO-basis data so nevpt.analyze() partitions the embedded spin density
        # over the atoms. Exact for a single fragment spanning every orbital.
        nevpt = qdnevpt2_res['nevpt']
        mo_emb = qdnevpt2_res['mc'].mo_coeff
        nevpt.spin_pop_mo = self.ints.ao2loc @ self.dmet_orbs[frag_idx] @ mo_emb
        nevpt.spin_pop_ovlp = self.ints.ovlp
        nevpt.spin_pop_mol = self.ints.mol

    def _build_task(self, frag_idx, method_key, dmet_oei, dmet_fock, dmet_tei,
                    norb_in_imp, nelec_in_imp, num_imp_orbs, chempot_imp,
                    dm_guess_rhf, mo_guess=None, ci_guess=None, dip_mom_ao=None,
                    embedding_data=None, embedded_ref=None):
        return {
            'counter': frag_idx,
            'method': method_key,
            'const': 0.0,
            'dmet_oei': dmet_oei,
            'dmet_fock': dmet_fock,
            'dmet_tei': dmet_tei,
            'dip_mom_ao': dip_mom_ao,
            'embedding_data': embedding_data,
            'embedded_ref': embedded_ref,
            'no_kernel': self.no_kernel,
            'norb': norb_in_imp,
            'nel': nelec_in_imp,
            'nimp': num_imp_orbs,
            'chempot_imp': chempot_imp,
            'dm_guess_rhf': dm_guess_rhf,
            'ncas': self.ncas,
            'nelecas': self.nelecas,
            'sa_nstates': self.sa_nstates,
            'sa_weights': self.sa_weights,
            'cas_select': self.cas_select,
            'natorb_occ_thresh': self.natorb_occ_thresh,
            'natorb_max_superset': self.natorb_max_superset,
            'deg_tol': self.deg_tol,
            'casci_conv_tol': self.casci_conv_tol,
            'embed_level_shift': self.embed_level_shift,
            'scf_stability': self.scf_stability,
            'cas_spin': self.cas_spin,
            'cas_spin_shift': self.cas_spin_shift,
            'casscf_kwargs': self.casscf_kwargs,
            'mo_guess': mo_guess,
            'ci_guess': ci_guess,
            'qdnevpt2_kwargs': self.qdnevpt2_kwargs,
            'pcnevpt2_kwargs': self.pcnevpt2_kwargs,
            'mradc_kwargs': self.mradc_kwargs,
            'spin': self.ints.mol.spin,
            'verbose': self.verbose,
        }

    def _run_fragment_sequential(self, task):
        result = solvers.execute(task)
        if 'cas_res' in result:
            cas_res = result['cas_res']
            self.frag_caches[task['counter']] = {
                'mo_coeff': cas_res['mo_coeff'],
                'ci': cas_res['ci'],
            }
        return result

    def construct_no_rotation(self):
        no_rotation_mat = np.zeros((self.norb, self.norb), dtype=float)
        jumpsquare = 0
        for frag_idx in range(len(self.imp_size)):  # imp_size has length 1 if is_translation_invariant
            no_rotation_mat[jumpsquare: jumpsquare + self.imp_size[frag_idx],
                         jumpsquare: jumpsquare + self.imp_size[frag_idx]] = self.no_vecs[frag_idx]
            jumpsquare += self.imp_size[frag_idx]
        for orb_idx in range(jumpsquare, self.norb):
            no_rotation_mat[orb_idx, orb_idx] = 1.0
        if self.is_translation_invariant:
            size = self.imp_size[0]
            for block in range(1, self.norb // size):
                no_rotation_mat[block*size:(block+1)*size, block*size:(block+1)*size] = no_rotation_mat[0:size, 0:size]
        return no_rotation_mat

    def cost_function(self, umat_flat):
        return np.linalg.norm(self.rdm_differences(umat_flat)) ** 2

    def alt_cost_function(self, umat_flat):
        umat_square_loc = self.flat2square(umat_flat)
        one_rdm_loc = self.helper.construct_1rdm_loc(umat_square_loc)

        errors = self.rdm_differences_masked(umat_flat)
        errors_sq = self.flat2square(errors)

        if self.min_func == 'oei':
            e_fun = np.trace(np.dot(self.ints.loc_oei(), one_rdm_loc))
        elif self.min_func == 'FOCK_INIT':
            e_fun = np.trace(np.dot(self.ints.loc_fock(), one_rdm_loc))
        e_cstr = np.sum(umat_square_loc * errors_sq)
        return -e_fun - e_cstr

    def cost_function_derivative(self, umat_flat):
        errors = self.rdm_differences(umat_flat)
        error_derivs = self.rdm_differences_derivative(umat_flat)
        gradient = np.zeros((len(umat_flat),), dtype=float)
        for elem_idx in range(len(umat_flat)):
            gradient[elem_idx] = 2 * np.sum(np.multiply(error_derivs[:, elem_idx], errors))
        return gradient

    def alt_cost_function_derivative(self, umat_flat):
        errors = self.rdm_differences_masked(umat_flat)
        return -errors

    def rdm_differences(self, umat_flat):
        umat_square_loc = self.flat2square(umat_flat)
        one_rdm_loc = self.helper.construct_1rdm_loc(umat_square_loc)

        n_cluster_orbs = 0
        for frag_idx in range(len(self.imp_size)):
            if self.do_det:  # DET (density embedding): fit the impurity block only
                n_cluster_orbs += self.imp_size[frag_idx]
                if self.fit_imp_bath:
                    raise RuntimeError("Density embedding is incompatible with fitting the bath.")
            else:  # DMET (density-matrix embedding): fit the full density matrix
                if self.fit_imp_bath:
                    n_cluster_orbs += self.dmet_orbs[frag_idx].shape[1] * self.dmet_orbs[frag_idx].shape[1]
                else:
                    n_cluster_orbs += self.imp_size[frag_idx] * self.imp_size[frag_idx]
        errors = np.zeros((n_cluster_orbs,), dtype=float)

        jump = 0
        for frag_idx in range(len(self.imp_size)):  # imp_size has length 1 if is_translation_invariant
            if self.fit_imp_bath:
                mf_rdm1 = self.dmet_orbs[frag_idx].T @ one_rdm_loc @ self.dmet_orbs[frag_idx]
                ed_rdm1 = self.imp_rdm1[frag_idx]
            else:
                mf_rdm1 = (one_rdm_loc[:, np.flatnonzero(self.fragments[frag_idx])])[np.flatnonzero(self.fragments[frag_idx]), :]
                ed_rdm1 = self.imp_rdm1[frag_idx][:self.imp_size[frag_idx], :self.imp_size[frag_idx]]
            if self.do_det:  # DET (density embedding)
                if self.do_det_no:  # Work in the NO basis
                    rdm_mismatch = np.diag(self.no_vecs[frag_idx].T @ mf_rdm1 @ self.no_vecs[frag_idx]) - self.no_diag[frag_idx]
                else:  # Work in the lattice basis
                    rdm_mismatch = np.diag(mf_rdm1 - ed_rdm1)
                errors[jump: jump + len(rdm_mismatch)] = rdm_mismatch
                jump += len(rdm_mismatch)
            else:  # DMET (density-matrix embedding): fit the full density matrix
                rdm_mismatch = mf_rdm1 - ed_rdm1
                square_size = rdm_mismatch.shape[0] * rdm_mismatch.shape[1]
                errors[jump: jump + square_size] = np.reshape(rdm_mismatch, square_size, order='F')
                jump += square_size
        if jump != n_cluster_orbs:
            raise RuntimeError("U-matrix fitting: cluster-orbital count mismatch.")

        return errors

    def rdm_differences_masked(self, umat_flat):
        umat_square_loc = self.flat2square(umat_flat)
        one_rdm_loc = self.helper.construct_1rdm_loc(umat_square_loc)

        n_cluster_orbs = 0
        jump = 0
        for frag_idx in range(len(self.imp_size)):
            mask_t = self.mask[np.ix_(list(range(jump, jump + self.imp_size[frag_idx])),
                                      list(range(jump, jump + self.imp_size[frag_idx])))]
            n_cluster_orbs += np.count_nonzero(mask_t)
            jump += self.imp_size[frag_idx]
        errors = np.zeros((n_cluster_orbs,), dtype=float)

        jump = 0
        jumpc = 0
        for frag_idx in range(len(self.imp_size)):  # imp_size has length 1 if is_translation_invariant
            mf_rdm1 = (one_rdm_loc[:, np.flatnonzero(self.fragments[frag_idx])])[np.flatnonzero(self.fragments[frag_idx]), :]
            ed_rdm1 = self.imp_rdm1[frag_idx][:self.imp_size[frag_idx], :self.imp_size[frag_idx]]
            rdm_mismatch = mf_rdm1 - ed_rdm1
            mask_t = self.mask[np.ix_(list(range(jumpc, jumpc + self.imp_size[frag_idx])),
                                      list(range(jumpc, jumpc + self.imp_size[frag_idx])))]
            square_size = np.count_nonzero(mask_t)
            errors[jump: jump + square_size] = np.reshape(rdm_mismatch[mask_t], square_size, order='F')
            jump  += square_size
            jumpc += self.imp_size[frag_idx]
        if jump != n_cluster_orbs:
            raise RuntimeError("U-matrix fitting: cluster-orbital count mismatch.")

        return errors

    def rdm_differences_derivative(self, umat_flat):
        umat_square_loc = self.flat2square(umat_flat)
        rdm_derivs_rot = self.helper.construct_1rdm_response(umat_square_loc, self.no_rotation)

        n_cluster_orbs = 0
        for frag_idx in range(len(self.imp_size)):
            if self.do_det:  # DET (density embedding): fit the impurity block only
                n_cluster_orbs += self.imp_size[frag_idx]
                if self.fit_imp_bath:
                    raise RuntimeError("Density embedding is incompatible with fitting the bath.")
            else:  # DMET (density-matrix embedding): fit the full density matrix
                if self.fit_imp_bath:
                    n_cluster_orbs += self.dmet_orbs[frag_idx].shape[1] * self.dmet_orbs[frag_idx].shape[1]
                else:
                    n_cluster_orbs += self.imp_size[frag_idx] * self.imp_size[frag_idx]

        gradient = []
        for elem_idx in range(len(umat_flat)):
            error_deriv = np.zeros((n_cluster_orbs,), dtype=float)
            jump = 0
            jumpsquare = 0
            for frag_idx in range(len(self.imp_size)):  # imp_size has length 1 if is_translation_invariant
                if self.fit_imp_bath:
                    local_derivative = self.dmet_orbs[frag_idx].T @ rdm_derivs_rot[elem_idx, :, :] @ self.dmet_orbs[frag_idx]
                else:
                    if self.do_det and self.do_det_no:
                        local_derivative = rdm_derivs_rot[elem_idx,
                                                         jumpsquare: jumpsquare + self.imp_size[frag_idx],
                                                         jumpsquare: jumpsquare + self.imp_size[frag_idx]]
                        jumpsquare += self.imp_size[frag_idx]
                    else:
                        local_derivative = ((rdm_derivs_rot[elem_idx, :, :])[:, np.flatnonzero(self.fragments[frag_idx])])[np.flatnonzero(self.fragments[frag_idx]), :]
                if self.do_det:  # DET (density embedding)
                    local_derivative = np.diag(local_derivative)
                    error_deriv[jump: jump + len(local_derivative)] = local_derivative
                    jump += len(local_derivative)
                else:  # DMET (density-matrix embedding): fit the full density matrix
                    square_size = local_derivative.shape[0] * local_derivative.shape[1]
                    error_deriv[jump: jump + square_size] = np.reshape(local_derivative, square_size, order='F')
                    jump += square_size
            if jump != n_cluster_orbs:
                raise RuntimeError("U-matrix fitting: cluster-orbital count mismatch.")
            gradient.append(error_deriv)
        gradient = np.array(gradient).T

        return gradient

    def flat2square(self, umat_flat):
        umat_square = np.zeros((self.norb, self.norb), dtype=float)
        umat_square[self.mask] = umat_flat
        umat_square = umat_square.T
        umat_square[self.mask] = umat_flat
        if self.is_translation_invariant:
            size = self.imp_size[0]
            for block in range(1, self.norb // size):
                umat_square[block*size:(block+1)*size, block*size:(block+1)*size] = umat_square[0:size, 0:size]

        if self.no_rotation is not None:
            umat_square = self.no_rotation @ umat_square @ self.no_rotation.T
        return umat_square

    def square2flat(self, umat_square):
        umat_square_copy = np.array(umat_square, copy=True)
        if self.no_rotation is not None:
            umat_square_copy = self.no_rotation.T @ umat_square_copy @ self.no_rotation
        umat_flat = umat_square_copy[self.mask]
        return umat_flat

    # Max physically reasonable chemical potential (Eh); beyond this Newton has diverged.
    _MU_MAX = 10.0

    def num_elec_cost_function(self, chempot_imp):
        if abs(chempot_imp) > self._MU_MAX:
            raise RuntimeError(
                f"Mu optimization diverged: |mu| = {abs(chempot_imp):.2f} Eh "
                f"exceeds threshold {self._MU_MAX} Eh."
            )
        nelec_dmet = self.do_exact(chempot_imp)
        nelec_target = self.ints.nelec
        self.log.note("      Chemical potential %s, electron count %s"
                      % (chempot_imp, nelec_dmet))
        return nelec_dmet - nelec_target

    def selfconsistent(self):
        if self.method in ('QD-NEVPT2', 'PC-NEVPT2', 'MR-ADC'):
            self.log.warn(
                "Self-consistent DMET repeats %s at every chemical-potential evaluation, "
                "which costs far more than oneshot()." % self.method)

        iteration = 0
        u_diff = 1.0
        self.log.note("Reference RHF energy: %s" % self.ints.e_hf)

        while u_diff > self.conv_tol and iteration < self.max_cycle:

            iteration += 1
            self.log.note("DMET iteration %d" % iteration)
            umat_old = np.array(self.umat, copy=True)
            rdm_old = self.transform_ed_1rdm()  # Zero at the very first iteration

            # Find the chemical potential for the correlated impurity problem.
            cput_ed = (logger.process_clock(), logger.perf_counter())
            try:
                self.mu_imp = optimize.newton(self.num_elec_cost_function, self.mu_imp)
            except RuntimeError:
                self.log.warn("Newton solver for chemical potential did not perfectly "
                              "converge. Proceeding with last evaluated chemical potential.")
            self.log.note("   Chemical potential: %s" % self.mu_imp)
            self.log.timer("embedding", *cput_ed)
            self.log.note("   Energy: %s" % self.energy)

            # Optimize the u-matrix.
            cput_cf = (logger.process_clock(), logger.perf_counter())
            if self.alt_cost_func and self.sc_method == 'BFGS':
                result = optimize.minimize(self.alt_cost_function, self.square2flat(self.umat),
                                           jac=self.alt_cost_function_derivative, options={'disp': False})
                self.umat = self.flat2square(result.x)
            elif self.sc_method == 'LSTSQ':
                result = optimize.leastsq(self.rdm_differences, self.square2flat(self.umat),
                                          Dfun=self.rdm_differences_derivative, factor=0.1)
                self.umat = self.flat2square(result[0])
            elif self.sc_method == 'BFGS':
                result = optimize.minimize(self.cost_function, self.square2flat(self.umat),
                                           jac=self.cost_function_derivative, options={'disp': False})
                self.umat = self.flat2square(result.x)
            self.umat = self.umat - np.eye(self.umat.shape[0]) * np.average(np.diag(self.umat))  # Remove arbitrary global shift
            if self.alt_cost_func:
                self.log.info("   Cost function after convergence: %s"
                              % self.alt_cost_function(self.square2flat(self.umat)))
            else:
                self.log.info("   Cost function after convergence: %s"
                              % self.cost_function(self.square2flat(self.umat)))
            self.log.timer("u-matrix optimization", *cput_cf)

            if self.print_u:
                self.print_umat()
            if self.print_rdm:
                self.print_1rdm()

            # Convergence check
            u_diff = np.linalg.norm(umat_old - self.umat)
            rdm_diff = np.linalg.norm(rdm_old - self.transform_ed_1rdm())
            self.umat = self.relaxation * umat_old + (1.0 - self.relaxation) * self.umat
            self.log.note("   u-matrix change (2-norm): %s" % u_diff)
            self.log.note("   1-RDM change (2-norm): %s" % rdm_diff)

            if self.sc_method == 'NONE':
                u_diff = 0.1 * self.conv_tol  # Do only 1 iteration

        if u_diff > self.conv_tol:
            self.log.warn("Self-consistency not converged in %d cycles: u-matrix change "
                          "%s exceeds conv_tol %s." % (self.max_cycle, u_diff, self.conv_tol))

        return self.energy

    def print_umat(self):
        self.log.info("Correlation potential (u-matrix):")
        square_jumper = 0
        for local_size in self.imp_size:  # imp_size has length 1 if is_translation_invariant
            self.log.info("%s" % self.umat[square_jumper:square_jumper + local_size,
                                           square_jumper:square_jumper + local_size])
            square_jumper += local_size

    def print_1rdm(self):
        self.log.info("Embedded 1-RDM (impurity + bath):")
        for frag_idx in range(len(self.imp_size)):  # imp_size has length 1 if is_translation_invariant
            self.log.info("%s" % self.imp_rdm1[frag_idx])

    def transform_ed_1rdm(self):
        result = np.zeros((self.umat.shape[0], self.umat.shape[0]), dtype=float)
        square_jumper = 0
        for frag_idx in range(len(self.imp_rdm1)):  # imp_size has length 1 if is_translation_invariant
            local_size = self.imp_size[frag_idx]
            result[square_jumper:square_jumper + local_size,
                   square_jumper:square_jumper + local_size] = \
                self.imp_rdm1[frag_idx][:local_size, :local_size]
            square_jumper += local_size
        return result

    def _dump_bath_spectrum(self, frag_idx, spectrum, ndisplay=10):
        nkeep = spectrum['num_bath_orbs']
        lo = max(0, nkeep - ndisplay)
        hi = min(len(spectrum['occupation']), nkeep + ndisplay)
        self.log.info("\nBath spectrum, fragment %d (kept %d of %d entangled):"
                      % (frag_idx, nkeep, spectrum['num_entangled']))
        self.log.info("   idx    occupation    dev. from 0/2      entropy   status")
        for i in range(lo, hi):
            self.log.info("  %4d  %12.8f     %12.6e  %11.6e   %s"
                          % (i, spectrum['occupation'][i], spectrum['occ_deviation'][i],
                             spectrum['entropy'][i], "bath" if i < nkeep else "core/virt"))

    def to_ao(self, mat_emb, impnumber=0):
        # Transform an embedded-basis matrix to the AO basis of the full molecule.
        coeff = self.ints.ao2loc @ self.dmet_orbs[impnumber]
        if mat_emb.shape[-1] != coeff.shape[1]:
            raise ValueError(
                f"Matrix dimension {mat_emb.shape[-1]} does not match the "
                f"{coeff.shape[1]}-orbital embedded basis of fragment {impnumber}.")
        return coeff @ mat_emb @ coeff.T

    def core_dm_ao(self, impnumber=0):
        # AO-basis density of the orbitals frozen out of a fragment's embedded problem.
        core = self.core_1rdm_loc[impnumber]
        if core is None:
            raise ValueError(f"Fragment {impnumber} was solved by symmetry and has no core "
                             f"density; use its symmetry parent instead.")
        return self.ints.ao2loc @ core @ self.ints.ao2loc.T

    def dump_bath_orbs(self, filename, impnumber=0):
        from pyscf.tools import molden
        with open(filename, 'w') as the_file:
            molden.header(self.ints.mol, the_file)
            molden.orbital_coeff(self.ints.mol, the_file,
                                 np.dot(self.ints.ao2loc, self.dmet_orbs[impnumber]))

    def dump_natural_orbitals(self, filename, impnumber=0, fmt='molden', orbital_indices=None):
        if not self.imp_rdm1:
            raise RuntimeError("No impurity 1-RDM available; run oneshot() or selfconsistent() first.")
        occ, vecs = np.linalg.eigh(self.imp_rdm1[impnumber])
        order = np.argsort(occ)[::-1]
        occ = occ[order]
        mo_ao = self.ints.ao2loc @ self.dmet_orbs[impnumber] @ vecs[:, order]
        if fmt == 'molden':
            from pyscf.tools import molden
            with open(filename, 'w') as the_file:
                molden.header(self.ints.mol, the_file)
                molden.orbital_coeff(self.ints.mol, the_file, mo_ao, occ=occ)
        elif fmt == 'cube':
            from pyscf.tools import cubegen
            if orbital_indices is None:
                orbital_indices = [i for i, n in enumerate(occ) if 1e-2 < n < 2.0 - 1e-2]
            base = filename[:-5] if filename.endswith('.cube') else filename
            for i in orbital_indices:
                cubegen.orbital(self.ints.mol, f"{base}_no{i}_occ{occ[i]:.3f}.cube", mo_ao[:, i])
        else:
            raise ValueError(f"Unknown fmt='{fmt}'. Use 'molden' or 'cube'.")
        return occ

    def dump_ntos(self, filename, impnumber=0, initial_state=0, target_state=1,
                  fmt='molden', n_pairs=None, nx=60, ny=60, nz=60):
        import pyscf.fci.direct_spin1 as fci_spin1
        _prism_results = self.qdnevpt2_results or self.pcnevpt2_results
        if _prism_results:
            if impnumber >= len(_prism_results):
                raise IndexError(f"Invalid impnumber={impnumber}: out of range.")
            res = _prism_results[impnumber]
            mc = res['mc']
            ci = mc.ci if isinstance(mc.ci, list) else [mc.ci]
            mo = mc.mo_coeff
            ncore = mc.ncore
            ncas = mc.ncas
            nelecas = mc.nelecas
        elif self.cas_results:
            if impnumber >= len(self.cas_results):
                raise IndexError(f"Invalid impnumber={impnumber}: out of range.")
            res = self.cas_results[impnumber]
            ci = res['ci'] if isinstance(res['ci'], list) else [res['ci']]
            mo = res['mo_coeff']
            ncore = res['ncore']
            ncas = res['ncas']
            nelecas = res['nelecas']
        else:
            raise RuntimeError(
                "No CASSCF or QD-NEVPT2 results available; run oneshot()/selfconsistent() first.")
        if initial_state >= len(ci) or target_state >= len(ci):
            raise ValueError(f"State index out of range (nstates={len(ci)}).")

        tdm = fci_spin1.trans_rdm1(ci[target_state], ci[initial_state], ncas, nelecas)
        U, s, Vh = np.linalg.svd(tdm, full_matrices=False)
        weights = s ** 2
        omega = np.sum(weights)
        w_norm = weights / (omega + 1e-30)
        pr = 1.0 / (np.sum(w_norm ** 2) + 1e-30)
        entropy = -np.sum(w_norm * np.log(w_norm + 1e-16))

        self.log.info("NTOs S%d -> S%d (impurity %d):" % (initial_state, target_state, impnumber))
        self.log.info("  Singular values:         %s" % np.round(s, 6).tolist())
        self.log.info("  Weights (s^2):           %s" % np.round(weights, 6).tolist())
        self.log.info("  Sum of weights (Omega):  %.6f" % omega)
        self.log.info("  Participation ratio:     %.6f" % pr)
        self.log.info("  Entanglement entropy:    %.6f" % entropy)
        self.log.info("  Entangled states (Z):    %.6f" % np.exp(entropy))

        # Active-space MOs back-transformed to real AOs (same chain as dump_natural_orbitals).
        cas_mo_ao = self.ints.ao2loc @ self.dmet_orbs[impnumber] @ mo[:, ncore:ncore + ncas]
        C_hole = np.dot(cas_mo_ao, U)        # hole NTOs (left singular vectors)
        C_particle = np.dot(cas_mo_ao, Vh.T)     # particle NTOs (right singular vectors)

        # Phase consistency: largest-magnitude AO sets hole sign (Prism convention).
        for k in range(C_hole.shape[1]):
            idx = np.argmax(np.abs(C_hole[:, k]))
            if C_hole[idx, k] < 0:
                C_hole[:, k]     *= -1
                C_particle[:, k] *= -1

        n_write = C_hole.shape[1] if n_pairs is None else min(n_pairs, C_hole.shape[1])

        if fmt == 'molden':
            from pyscf.tools import molden
            C_nto = np.zeros((C_hole.shape[0], 2 * n_write))
            C_nto[:, 0::2] = C_hole[:, :n_write]
            C_nto[:, 1::2] = C_particle[:, :n_write]
            occ_nto = np.repeat(weights[:n_write], 2)
            with open(filename, 'w') as the_file:
                molden.header(self.ints.mol, the_file)
                molden.orbital_coeff(self.ints.mol, the_file, C_nto, occ=occ_nto)
        elif fmt == 'cube':
            from pyscf.tools import cubegen
            base = filename[:-5] if filename.endswith('.cube') else filename
            for k in range(n_write):
                cubegen.orbital(self.ints.mol,
                                f"{base}_S{initial_state}S{target_state}_nto{k+1}_hole.cube",
                                C_hole[:, k], nx=nx, ny=ny, nz=nz)
                cubegen.orbital(self.ints.mol,
                                f"{base}_S{initial_state}S{target_state}_nto{k+1}_particle.cube",
                                C_particle[:, k], nx=nx, ny=ny, nz=nz)
        else:
            raise ValueError(f"Unknown fmt='{fmt}'. Use 'molden' or 'cube'.")
        return weights, C_hole, C_particle

    def oneshot(self, mu_imp=0.0, optimize_mu=False):
        if optimize_mu:
            try:
                self.mu_imp = optimize.newton(self.num_elec_cost_function, mu_imp)
            except RuntimeError:
                self.log.warn("Newton solver for mu_imp did not converge. "
                              "Falling back to initial mu.")
                self.mu_imp = mu_imp
        else:
            self.mu_imp = mu_imp

        self.do_exact(self.mu_imp)
        return self.energy


def make_fragments(mol, my_ints, atom_groups):
    ao_slices = mol.aoslice_by_atom()  # shape (natm, 4): (shl0, shl1, ao0, ao1)
    norb = my_ints.norb
    impurity_clusters = []
    covered = np.zeros(norb, dtype=int)

    for group in atom_groups:
        mask = np.zeros(norb, dtype=int)
        for atom_idx in group:
            ao_start = ao_slices[atom_idx, 2]
            ao_stop = ao_slices[atom_idx, 3]
            mask[ao_start:ao_stop] = 1
        impurity_clusters.append(mask)
        covered += mask

    if np.any(covered > 1):
        raise ValueError("Overlapping atom groups detected.")

    return impurity_clusters
