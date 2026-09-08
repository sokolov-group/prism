# prism.dmet

This module runs density matrix embedding theory (DMET) inside Prism.

DMET lets you treat one part of a molecule with an expensive method while the rest
of the molecule is held at the mean-field level.

First, run a mean-field calculation on the whole molecule. The molecular orbitals
from that calculation are then localized to each atom. Then, pick a set of atoms
whose orbitals form the impurity. Next, the mean-field density decides which of the
remaining orbitals are entangled with the impurity, through a Schmidt decomposition
of the mean-field wavefunction. Those orbitals form the bath, and by default there
are no more bath orbitals than impurity orbitals. The remaining orbitals form a
frozen core, which enters the correlated calculation as a mean-field potential.
Impurity plus bath is the cluster. The code builds a one-electron matrix and a
two-electron integral tensor for the cluster and gives them to the solver. The
cluster is much smaller than the molecule, so the solver is much cheaper.

The original papers are Knizia and Chan, Phys. Rev. Lett. 109, 186404 (2012) and
J. Chem. Theory Comput. 9, 1428 (2013). A practical guide is Wouters,
Jimenez-Hoyos, Sun and Chan, J. Chem. Theory Comput. 12, 2706 (2016).

## How to run a calculation

```python
import pyscf.gto
import pyscf.scf
from prism.dmet import DMET, LocalIntegrals, make_fragments

mol = pyscf.gto.M(atom='H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22',
                  basis='sto-3g', verbose=4)

mf = pyscf.scf.RHF(mol)
mf.kernel()

ints = LocalIntegrals(mf)
frags = make_fragments(mol, ints, [[0, 1], [2, 3]])

dmet = DMET(ints, frags, False, method='FCI')
energy = dmet.oneshot()
print("DMET energy: %.10f" % energy)
```

Four objects, in order: the full molecule mean field (`mf`), the local integrals
(`ints`), the fragments of interest (`frags`), and the DMET object (`dmet`).

### Full Molecule Mean Field (mf)

Run any PySCF mean field on the whole molecule. The mean field supplies the orbitals
that get localized and the density that decides which orbitals become bath. RHF,
ROHF, RKS and UKS all work. You can add `.density_fit()` and `.x2c()`.

If the mean field is density fitted, the cluster integrals are taken from its
three-index tensor instead of rebuilding the atomic-orbital integrals. The cluster
integrals then carry the fitting error of the mean field.

### Local Integrals (ints)

```python
ints = LocalIntegrals(mf, active_orbs=None, localization_type='meta_lowdin',
                      ao_rotation=None, localization_threshold=1e-6)
```

Both defaults are the normal choice, so `LocalIntegrals(mf)` is the usual call.

`active_orbs` is the list of orbital indices to use. `None` means every orbital in
the molecule, and is the default. Only `boys` accepts a smaller set. `meta_lowdin`,
`lowdin` and `iao` need every orbital.

`localization_type` is one of:

- `meta_lowdin`, the default choice, used in every example
- `lowdin`, plain symmetric orthogonalization
- `iao`, intrinsic atomic orbitals
- `boys`, Boys localization, slower and can be sensitive to the starting point

`ao_rotation` is an optional extra rotation applied after localizing.
`localization_threshold` only affects `boys`.

The `LocalIntegrals` object holds the integrals for the whole molecule in the
localized basis. It can be passed to several `DMET` objects.

### Fragments of Interest (frags)

```python
frags = make_fragments(mol, ints, atom_groups)
```

`atom_groups` is a list of lists of atom indices. Each inner list is one fragment.
`[[0, 1], [2, 3]]` means two fragments, the first holding atoms 0 and 1 and the
second holding atoms 2 and 3.

Fragments cannot overlap. They do not have to cover the whole molecule. Each
fragment is solved in turn, and for that one fragment every orbital outside it
becomes bath or core.

How much of the molecule to cover depends on the goal.

Cover the whole molecule when you want a total energy. Every region then gets its
own bath and its own solve, and the fragment energies sum to the molecular energy.
Examples 01 and 03 do this.

Cover only the region you care about when you want a local property, such as an
excitation energy, an ionization energy or a g-tensor. `[[0]]` on a metal complex
embeds the metal alone, with the ligands in the bath and the core. Example 04 does
this on Cu(NH3)4: one fragment on the copper, sixteen atoms left out.

Full coverage is required when `is_translation_invariant=True`, and leaving orbitals
out needs a full `active_orbs`.

### Density Matrix Embedding Method (dmet)

```python
dmet = DMET(ints, frags, is_translation_invariant, method='FCI')
```

Any of the keywords listed further down can be added to that call.
The first three arguments are positional. `is_translation_invariant` should be
`False` for molecules. Set it to `True` only when every fragment is a copy of the
first one, in which case the code solves one fragment and reuses the answer.

### One-Shot and Self-Consistent Runs (oneshot, selfconsistent)

There are two ways to run.

`dmet.oneshot()` builds the bath from the mean-field density, solves every fragment
once, and returns the total energy. This is the usual choice.

`dmet.selfconsistent()` adds an outer loop. It fits a one-body potential, called the
correlation potential or u-matrix, so that the mean-field density of each cluster
matches the correlated one. The bath is rebuilt at every step. It returns the total
energy.

Self-consistency costs much more. The solver runs once per chemical-potential
evaluation, and there can be many of those per iteration. With QD-NEVPT2, PC-NEVPT2
or MR-ADC the code prints a warning about this, but it will run.

## The solvers

Pick one with `method=`.

### FCI

`method='FCI'` runs a full configuration interaction on the cluster. Use it for
small clusters. It needs no active space.

### CASSCF

`method='CASSCF'` runs a complete active space SCF on the cluster. You must give
`ncas` and `nelecas`. Results land in `dmet.cas_results`.

### PC-NEVPT2

`method='PC-NEVPT2'` runs Prism's fully internally contracted NEVPT2 on top of an
embedded CASSCF. This is equivalent to partially contracted NEVPT2. Results land in
`dmet.pcnevpt2_results`.

### QD-NEVPT2

`method='QD-NEVPT2'` runs the quasidegenerate variant of NEVPT2. It needs
`sa_nstates` of 2 or more. Results land in `dmet.qdnevpt2_results`.

### MR-ADC

`method='MR-ADC'` runs multireference algebraic diagrammatic construction. The
reference must be single state, so `sa_nstates` stays at 1 and you ask for roots
through `mradc_kwargs`. Results land in `dmet.mradc_results`.

```python
dmet = DMET(ints, frags, False, method='MR-ADC', ncas=4, nelecas=4,
            mradc_kwargs={'method_type': 'cvs-ip', 'ncvs': 1, 'nroots': 3})
```

### One fragment at a lower level

Fragments do not all have to use the same solver. `method` sets what every fragment
uses, and `fragment_methods` overrides it for the fragments you name.

```python
dmet = DMET(ints, frags, False, method='CASSCF', ncas=4, nelecas=4,
            fragment_methods={1: 'RHF'})
```

Here fragment 0 gets CASSCF and fragment 1 gets plain RHF. `'RHF'` is the only
override accepted.

## Reading the results

Each solver fills a list, one entry per fragment.

`dmet.cas_results[i]` has `e_tot`, `e_states`, `e_imp`, `ncas`, `nelecas`, `ncore`,
`nstates`, `weights`, `ci`, `mo_coeff` and `cas_select`.

`dmet.pcnevpt2_results[i]` has `e_tot`, `e_corr`, `mc` and `nevpt`.

`dmet.qdnevpt2_results[i]` has `e_tot`, `e_corr`, `e_cas_states`, `mc` and `nevpt`.

`dmet.mradc_results[i]` has `e_exc`, `spec_factors`, `e_cas`, `mc` and `mradc`.

`mc` is the PySCF CASSCF object for that cluster. `nevpt` and `mradc` are the Prism
objects. You can call their methods directly, for example
`dmet.qdnevpt2_results[0]['nevpt'].analyze()`.

The MR-ADC `e_exc` values are in eV, because that is what the Prism MR-ADC
kernel returns. The NEVPT2 `e_tot` values are cluster energies without the nuclear
repulsion, so compare excitation energies rather than totals. The energy returned by
`oneshot` and `selfconsistent` is a total energy and includes nuclear repulsion.

`dmet.imp_rdm1[i]` holds the correlated one-particle density matrix of cluster `i`
in the embedded basis, for any solver.

## Keywords

All of these are optional keywords on `DMET`.

### Choosing the solver

| keyword | default | what it does |
|---|---|---|
| `method` | `'FCI'` | which solver to run |
| `fragment_methods` | `None` | dict of fragment index to `'RHF'` |

### The active space

| keyword | default | what it does |
|---|---|---|
| `ncas` | `None` | number of active orbitals |
| `nelecas` | `None` | number of active electrons |
| `cas_select` | `'energy'` | how to pick the active orbitals, `'energy'` or `'natorb'` |
| `sa_nstates` | `1` | how many states to average over |
| `sa_weights` | `None` | weights for those states, equal if not given |
| `cas_spin` | `None` | target 2S for the CAS states |
| `cas_spin_shift` | `0.2` | strength of the spin penalty |
| `natorb_occ_thresh` | `0.02` | occupation cutoff that separates active from core |
| `natorb_max_superset` | `None` | cap on the natorb search window |
| `deg_tol` | `1e-3` | orbital energies closer than this count as degenerate |
| `casci_conv_tol` | `1e-10` | convergence of the natorb search CASCI |

With `cas_select='energy'` the code takes `ncas` orbitals from around the Fermi
level. With `cas_select='natorb'` it runs a CASCI in a window, looks at the natural
occupations, and keeps the fractionally occupied orbitals. In that mode `ncas` sets
the size of the search window rather than the final active space.

### The bath

| keyword | default | what it does |
|---|---|---|
| `n_bath_orbs` | `None` | cap on bath size, an int or one int per fragment |
| `bath_tol` | `1e-13` | below this entanglement an orbital is not bath |
| `keep_degenerate` | `False` | extend the bath to finish a degenerate set |
| `deg_rtol` | `1e-6` | how close two occupations must be to count as degenerate |
| `core_occ_tol` | `None` | how far a core orbital may sit from 0 or 2 |
| `bath_1rdm` | `None` | build the bath from a density you supply |

By default the number of bath orbitals is the same as the number of impurity
orbitals. `n_bath_orbs` makes it smaller and cheaper, at the cost of throwing away
some entanglement. `keep_degenerate=True` extends the bath to finish a degenerate
set. Two orbitals with the same occupation belong in the bath together, so if the
cut falls between them the bath is extended to take both. The code reports this as
"Bath extended to N to complete a degenerate set."

### The embedded mean field

| keyword | default | what it does |
|---|---|---|
| `embed_level_shift` | `0.0` | level shift on the embedded SCF |
| `scf_stability` | `False` | run a stability analysis and follow any instability |
| `no_kernel` | `False` | skip the embedded SCF and take the reference from a density |
| `embedded_ref_dm` | `None` | AO-basis density to use as that reference |

By default a small SCF runs on the cluster before the correlated solver. It starts
from the projected mean-field density and reconverges.
`no_kernel=True` skips it and builds the reference from the natural orbitals of the
guess density. Use it where reconverging is unstable. `embedded_ref_dm` supplies
that density directly. It needs `no_kernel=True` and `sc_method='NONE'`, and the
density must be in the AO basis.

### Self-consistency

| keyword | default | what it does |
|---|---|---|
| `sc_method` | `'LSTSQ'` | how to fit the potential, `'LSTSQ'`, `'BFGS'` or `'NONE'` |
| `conv_tol` | `1e-5` | when the potential has stopped changing |
| `max_cycle` | `200` | cap on outer iterations |
| `fit_impurity_and_bath` | `True` | fit the cluster density, not the impurity block alone |
| `use_constrained_optimization` | `False` | use the alternative cost function with BFGS |
| `use_density_embedding` | `False` | fit only the diagonal, which is density embedding |
| `use_density_embedding_no` | `False` | do that in the natural orbital basis |

`sc_method='NONE'` runs a single iteration.

### Passing options to the solver

| keyword | default | what it does |
|---|---|---|
| `casscf_kwargs` | `None` | set on the PySCF CASSCF object |
| `pcnevpt2_kwargs` | `None` | set on the Prism NEVPT object |
| `qdnevpt2_kwargs` | `None` | set on the Prism QD-NEVPT object |
| `mradc_kwargs` | `None` | set on the Prism MR-ADC object |

Anything you put in these dicts is set as an attribute on the matching object. This
is how you reach Prism options that DMET does not wrap, including spin-orbit
coupling and the magnetic properties.

The NEVPT2 solvers also take `nfrozen`, the frozen core of the embedded problem. It
is sized to that problem, not to the whole cluster. `nfrozen='auto'` counts the
embedded orbitals below `nfrozen_cutoff`, which defaults to -2.0 Ha.

### Symmetry and speed

| keyword | default | what it does |
|---|---|---|
| `use_symmetry` | `False` | solve one fragment and copy the answer to its twins |
| `symmetry_map` | `None` | dict of child fragment to parent fragment |
| `parallel` | `False` | solve fragments in worker processes |
| `max_workers` | `None` | how many workers |

With `use_symmetry=True` and no map, fragments with the same number of orbitals are
treated as copies of the first one. That test is crude, so pass `symmetry_map` when
it matters.

`parallel=True` works with FCI, CASSCF and per-fragment RHF. It raises for
QD-NEVPT2, PC-NEVPT2 and MR-ADC, because their result objects cannot be sent back
from a worker process.

The cluster integrals are stored eightfold-packed, which is eight times smaller than
the full four-index array. A cluster of 600 orbitals holds them in 133 GiB.

### Printing

| keyword | default | what it does |
|---|---|---|
| `print_u` | `True` | print the correlation potential each iteration |
| `print_rdm` | `True` | print the cluster density each iteration |
| `print_bath_spectrum` | `False` | print the bath orbitals on either side of the cut |
| `verbose` | `None` | print level, taken from the molecule if not given |

## Properties and analysis

Prism's properties and analysis all work with DMET. Spin-orbit coupling and the
magnetic properties are requested through the solver kwargs before the run.
Everything else here is read off a finished calculation.

### Spin-orbit coupling and magnetic properties

These are Prism options, so you reach them through the solver kwargs.

```python
dmet = DMET(ints, frags, False, method='QD-NEVPT2',
            ncas=5, nelecas=9, sa_nstates=5,
            qdnevpt2_kwargs={'soc': 'breit-pauli', 'gtensor': True,
                             'mag_av': True, 'sus_av': True,
                             'Bs_powder_M': [0.5, 1.0, 2.0],
                             'T_powder_M': [1.8],
                             'Bs_powder_chi': [0.1],
                             'T_powder_chi': [5.0, 100.0, 300.0]})
dmet.oneshot()
props = dmet.qdnevpt2_results[0]['nevpt'].properties
print(props['g-factors'][0])
```

`soc` accepts `'breit-pauli'` or `'bp'` for the Breit-Pauli operator, and `'dkh1'`,
`'x2c-1'` or `'x2c1'` for the one-electron exact two-component operator. With
spin-orbit coupling requested, the real molecule and the frozen core density are
passed to the integral routines, so the integrals are built over the whole molecule.
The frozen core contributes to the answer.

Results come back in the `properties` dict of the Prism object: `g-factors`,
`g-eigenvectors`, `M_av`, `chi_av`, `M_xyz_all` and `chi_T_eval_all`, depending on
what was requested.

Susceptibility is a second derivative taken by finite differences, so how closely it
matches an unembedded run depends on `step_h_s`. The magnetization is a first
derivative and is not sensitive.

### Oscillator strengths and state analysis

`sa_nstates` of 2 or more gives excitation energies between the state-averaged
roots. QD-NEVPT2 and PC-NEVPT2 also return oscillator strengths for them. The dipole
integrals are taken over the real molecule and transformed into the embedded basis,
so the values are comparable with an unembedded run.

```python
res = dmet.qdnevpt2_results[0]
print(res['nevpt'].properties['osc_strengths'])
res['nevpt'].analyze()
```

`analyze()` prints the composition of each root: the alpha and beta occupations of
the leading determinants with their coefficients and weights, the natural
occupations of the active space, and, for an open-shell reference, Mulliken spin
populations over the atoms of the real molecule.

### Orbital files

Three methods write files you can open in a viewer.

- `dump_bath_orbs(filename, impnumber=0)` writes the cluster orbitals. Molden only.
- `dump_natural_orbitals(filename, impnumber=0, fmt='molden', orbital_indices=None)`
  writes the natural orbitals of the correlated density
- `dump_ntos(filename, impnumber=0, initial_state=0, target_state=1, fmt='molden',
  n_pairs=None, nx=60, ny=60, nz=60)` writes natural transition orbitals

Where there is a `fmt`, it accepts `'molden'` or `'cube'`. `dump_ntos` also lets you
set the cube grid with `nx`, `ny` and `nz`.

Prism's default orbital output also works. Set `compute_ntos` on the NEVPT object or
`compute_dyson` on the MR-ADC object to write molden files over the whole molecule.

### Densities in the AO basis

The cluster has its own orbitals, which are not atomic orbitals. Two methods map
back to the molecule.

```python
dm_ao = dmet.to_ao(dmet.imp_rdm1[0]) + dmet.core_dm_ao(0)
```

`to_ao(matrix, impnumber=0)` takes anything in the embedded basis and returns it in
the AO basis of the molecule. `core_dm_ao(impnumber=0)` returns the frozen core
density in the same basis. Adding them gives the full density of that cluster's
solution, and its trace against the overlap matrix is the electron count of the
whole molecule.

`core_dm_ao` raises for a fragment that was solved by symmetry, because such a
fragment has no core density of its own.

### The bath spectrum

Set `print_bath_spectrum=True` to see what orbitals the bath kept and what it
dropped.

```
Bath spectrum, fragment 0 (kept 3 of 3 entangled):
   idx    occupation    dev. from 0/2      entropy   status
     0    1.00000000     1.000000e+00  1.386294e+00   bath
     1    1.00000000     1.000000e+00  1.386294e+00   bath
     2    1.00000000     1.000000e+00  1.386294e+00   bath
     3    2.00000000    -8.881784e-16  0.000000e+00   core/virt
```

The columns are the occupation of the bath orbital, its distance from 0 or 2, and
its single-orbital entropy. An orbital at 0 or 2 is not entangled with the impurity
and does not need to be in the cluster. An orbital at 1 is as entangled as it can
be, and its entropy is ln 4, which is 1.386294.

The same numbers are available in code as `dmet.bath_spectrum[i]`, a dict with
`occupation`, `occ_deviation`, `entropy`, `num_bath_orbs` and `num_entangled`.

## Examples

The `examples/dmet` folder has four scripts. Each also runs the same calculation on
the whole molecule, so the error from the embedding is visible.

- `01-dmet-h4-fci.py`, the smallest example. One-shot DMET, self-consistent DMET and
  a density-fitted run, all checked against full-molecule FCI.
- `02-dmet-h2o-qdnevpt2.py`, excitation energies and oscillator strengths, checked
  against a direct QD-NEVPT2.
- `03-dmet-n2-solvers.py`, CASSCF, PC-NEVPT2 and CVS-IP-MR-ADC on the same system,
  with the bath spectrum and the AO-basis density.
- `04-dmet-cunh34-soc-gtensor.py`, a copper complex with spin-orbit coupling, a
  g-tensor and powder magnetization and susceptibility.

## Tests

The tests are in `tests/dmet`. Each one is a plain script, and can be run on its
own.

```
python tests/dmet/01-dmet-fci-h4.py
```

To run the whole suite, use the runner from the `tests` folder and enter `dmet` when
it asks which folders to scan.

```
cd tests && python run_tests.py
```