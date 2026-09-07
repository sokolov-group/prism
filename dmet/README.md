# prism.dmet

This module runs density matrix embedding theory (DMET) inside Prism.

DMET lets you treat one part of a molecule with an expensive method while the rest
of the molecule is held at the mean-field level. You pick the atoms you care about.
The module builds a small problem around them and hands it to a solver. Every Prism
solver works this way, so you can get NEVPT2 energies, MR-ADC core ionization
energies, spin-orbit coupling, g-tensors, magnetization and susceptibility from an
embedded calculation.

## What the code does

Start with a mean-field calculation on the whole molecule. The orbitals from that
calculation are spread over every atom, so the first step is to localize them. Each
localized orbital then belongs to one atom.

Pick a set of atoms. Their orbitals are the impurity.

The code then looks at the mean-field density and asks which of the remaining
orbitals are entangled with the impurity. Those orbitals are the bath. There are
never more bath orbitals than impurity orbitals. Everything left over is the core.
It is frozen at its mean-field density and enters the calculation only as a
potential.

Impurity plus bath is the cluster. The code builds a one-electron matrix and a
two-electron integral tensor for the cluster and gives them to the solver. The
cluster is much smaller than the molecule, so the solver is much cheaper.

This split comes from a Schmidt decomposition of the mean-field wavefunction. The
original papers are Knizia and Chan, Phys. Rev. Lett. 109, 186404 (2012) and
J. Chem. Theory Comput. 9, 1428 (2013). A practical guide is Wouters,
Jimenez-Hoyos, Sun and Chan, J. Chem. Theory Comput. 12, 2706 (2016).

## A first calculation

```python
import pyscf.gto
import pyscf.scf
from prism.dmet import DMET, LocalIntegrals, make_fragments

mol = pyscf.gto.M(atom='H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22',
                  basis='sto-3g', verbose=4)

mf = pyscf.scf.RHF(mol)
mf.kernel()

ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0, 1], [2, 3]])

dmet = DMET(ints, frags, False, method='FCI')
energy = dmet.oneshot()
print("DMET energy: %.10f" % energy)
```

That is the whole pattern. Four objects, in order: a mean field, a LocalIntegrals,
a fragment list, and a DMET.

## Step 1, the mean field

Run any PySCF mean field on the whole molecule. RHF, ROHF, RKS and UKS all work.
You can add `.density_fit()` and `.x2c()`.

The mean field does two jobs. It supplies the orbitals that get localized, and it
supplies the density that decides which orbitals become bath. If the mean-field
solution is wrong, the bath is wrong, so it is worth checking that it converged to
the state you wanted.

## Step 2, LocalIntegrals

```python
ints = LocalIntegrals(mf, active_orbs, localization_type,
                      ao_rotation=None, localization_threshold=1e-6)
```

`active_orbs` is the list of orbital indices to use. Pass
`list(range(mol.nao_nr()))` to use all of them, which is the normal choice.

`localization_type` is one of:

- `meta_lowdin`, a good default and the one used in every example
- `lowdin`, plain symmetric orthogonalization, needs all orbitals
- `iao`, intrinsic atomic orbitals, needs all orbitals
- `boys`, Boys localization, slower and can be sensitive to the starting point

`ao_rotation` is an optional extra rotation applied after localizing.
`localization_threshold` only affects `boys`.

This object holds the integrals for the whole molecule in the localized basis. It
is reusable. You can build one LocalIntegrals and hand it to several DMET objects.

## Step 3, fragments

```python
frags = make_fragments(mol, ints, atom_groups)
```

`atom_groups` is a list of lists of atom indices. Each inner list is one fragment.
`[[0, 1], [2, 3]]` means two fragments, the first holding atoms 0 and 1 and the
second holding atoms 2 and 3.

Every orbital must belong to exactly one fragment. If you leave atoms out, the code
raises an error.

A fragment can be a single atom. `[[0]]` on a metal complex gives you an embedding
around the metal, with the ligands in the bath and the core.

## Step 4, the DMET object

```python
dmet = DMET(ints, frags, is_translation_invariant, method='FCI')
```

Any of the keywords listed further down can be added to that call.
The first three arguments are positional. `is_translation_invariant` should be
`False` for molecules. Set it to `True` only when every fragment is a copy of the
first one, in which case the code solves one fragment and reuses the answer.

## Running a calculation

There are two ways to run.

`dmet.oneshot()` builds the bath from the mean-field density, solves every fragment
once, and returns the total energy. This is what you want most of the time.

`dmet.selfconsistent()` adds an outer loop. It fits a one-body potential, called the
correlation potential or u-matrix, so that the mean-field density of each cluster
matches the correlated one. The bath is rebuilt at every step. It returns the total
energy.

Self-consistency costs much more. The solver runs once per chemical-potential
evaluation, and there can be many of those per iteration. With QD-NEVPT2, PC-NEVPT2
or MR-ADC the code prints a warning about this, but it will run.

One thing to know about self-consistency. The energy it produces is stable, but the
u-matrix itself is not well determined. On small test systems the same run repeated
can give a u-matrix with a very different size while the energy moves by less than
1e-5 Ha. Treat the energy as the result and do not read much into the potential.

## The solvers

Pick one with `method=`.

### ED and FCI

`method='FCI'` runs a full configuration interaction on the cluster. `method='ED'`
is the same solver under a different name. Use it for small clusters. It needs no
active space.

### CASSCF

`method='CASSCF'` runs a complete active space SCF on the cluster. You must give
`ncas` and `nelecas`. Results land in `dmet.cas_results`.

### PC-NEVPT2

`method='PC-NEVPT2'` runs Prism's fully internally contracted NEVPT2 on top of an
embedded CASSCF. This is equivalent to partially contracted NEVPT2. Results land in
`dmet.pcnevpt2_results`.

### QD-NEVPT2

`method='QD-NEVPT2'` runs the quasidegenerate variant. It needs `sa_nstates` of 2 or
more. Results land in `dmet.qdnevpt2_results`.

### MR-ADC

`method='MR-ADC'` runs multireference algebraic diagrammatic construction. The
reference must be single state, so `sa_nstates` stays at 1 and you ask for roots
through `mradc_kwargs`. Results land in `dmet.mradc_results`.

```python
dmet = DMET(ints, frags, False, method='MR-ADC', ncas=4, nelecas=4,
            mradc_kwargs={'method_type': 'cvs-ip', 'ncvs': 1, 'nroots': 3})
```

### One fragment at a lower level

`fragment_methods={1: 'RHF'}` solves fragment 1 with plain RHF instead of the main
method. Only `'RHF'` is accepted.

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

Two unit notes. The MR-ADC `e_exc` values are in eV, because that is what the Prism
MR-ADC kernel returns. The NEVPT2 `e_tot` values are cluster energies and do not
include the nuclear repulsion, so compare excitation energies rather than totals.
The energy returned by `oneshot` and `selfconsistent` is a proper total energy and
does include nuclear repulsion.

`dmet.imp_rdm1[i]` holds the correlated one-particle density matrix of cluster `i`
in the embedded basis, for any solver.

## The keywords

All of these are optional keywords on `DMET`.

### Choosing the solver

| keyword | default | what it does |
|---|---|---|
| `method` | `'ED'` | which solver to run |
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

By default the bath is as large as it can be, which is the number of impurity
orbitals. `n_bath_orbs` makes it smaller and cheaper, at the cost of throwing away
some entanglement.

### The embedded mean field

| keyword | default | what it does |
|---|---|---|
| `embed_level_shift` | `0.0` | level shift on the embedded SCF |
| `scf_stability` | `False` | run a stability analysis and follow any instability |
| `no_kernel` | `False` | skip the embedded SCF and take the reference from a density |
| `embedded_ref_dm` | `None` | AO-basis density to use as that reference |

Normally the code runs a small SCF on the cluster before the correlated solver. That
SCF starts from the projected mean-field density and reconverges. Every DMET code
does this.

`no_kernel=True` skips it. The reference is built from the natural orbitals of the
guess density instead. This exists for cases where reconverging is unstable, for
example when a hard-won UKS solution is destroyed by an ROHF reconvergence.
`embedded_ref_dm` lets you supply that density yourself. It needs `no_kernel=True`
and `sc_method='NONE'`, and the density must be in the AO basis.

### Self-consistency

| keyword | default | what it does |
|---|---|---|
| `sc_method` | `'LSTSQ'` | how to fit the potential, `'LSTSQ'`, `'BFGS'` or `'NONE'` |
| `conv_tol` | `1e-5` | when the potential has stopped changing |
| `max_cycle` | `200` | cap on outer iterations |
| `fit_impurity_and_bath` | `True` | fit the whole cluster density, not just the impurity block |
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

### Symmetry and speed

| keyword | default | what it does |
|---|---|---|
| `use_symmetry` | `False` | solve one fragment and copy the answer to its twins |
| `symmetry_map` | `None` | dict of child fragment to parent fragment |
| `parallel` | `False` | solve fragments in worker processes |
| `max_workers` | `None` | how many workers |

With `use_symmetry=True` and no map, fragments with the same number of orbitals are
treated as copies of the first one. That is a crude test, so pass `symmetry_map`
yourself when it matters.

`parallel=True` works with ED, FCI, CASSCF and per-fragment RHF. It raises for
QD-NEVPT2, PC-NEVPT2 and MR-ADC, because their result objects cannot be sent back
from a worker process.

### Printing

| keyword | default | what it does |
|---|---|---|
| `print_u` | `True` | print the correlation potential each iteration |
| `print_rdm` | `True` | print the cluster density each iteration |
| `print_bath_spectrum` | `False` | print the bath orbitals on either side of the cut |
| `verbose` | `None` | print level, taken from the molecule if not given |

## Looking at the bath

Set `print_bath_spectrum=True` to see what the bath kept and what it dropped.

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

This is the diagnostic to look at when deciding whether `n_bath_orbs` is safe. If
the entanglement has already fallen to 1e-9 at the cut, little is being lost.

## Getting densities back in the AO basis

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

## Writing orbital files

Three methods write files you can open in a viewer.

- `dump_bath_orbs(filename, impnumber=0)` writes the cluster orbitals. Molden only.
- `dump_natural_orbitals(filename, impnumber=0, fmt='molden', orbital_indices=None)`
  writes the natural orbitals of the correlated density
- `dump_ntos(filename, impnumber=0, initial_state=0, target_state=1, fmt='molden',
  n_pairs=None, nx=60, ny=60, nz=60)` writes natural transition orbitals

Where there is a `fmt`, it accepts `'molden'` or `'cube'`. `dump_ntos` also lets you
set the cube grid with `nx`, `ny` and `nz`.

Prism's own orbital output also works. Setting `compute_ntos` on the NEVPT object or
`compute_dyson` on the MR-ADC object writes molden files over the whole molecule.

## Spin-orbit coupling and magnetic properties

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
`'x2c-1'` or `'x2c1'` for the one-electron exact two-component operator. When you
ask for spin-orbit coupling, the code notices and passes the real molecule and the
frozen core density through to the integral routines, so the integrals are built
over the whole molecule and not over
the cluster alone. The frozen core matters here. Leaving it out changes the answer.

Results come back in the `properties` dict of the Prism object. `g-factors`,
`g-eigenvectors`, `M_av`, `chi_av`, `M_xyz_all` and `chi_T_eval_all` are all
available depending on what you asked for.

One practical note on susceptibility. It is a second derivative taken by finite
differences, so how closely it matches an unembedded run depends on `step_h_s`. The
magnetization, a first derivative, does not have this sensitivity.

## Density fitting

If the mean field is density fitted, the cluster integrals are taken from its
three-index tensor instead of rebuilding the atomic-orbital integrals. Nothing extra
is needed. Just build the mean field with `.density_fit()`.

```python
mf = pyscf.scf.RHF(mol).density_fit()
mf.kernel()
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
```

This saves the most on large systems, where rebuilding the atomic-orbital integrals
for every fragment is the slowest part. The cluster integrals then carry the fitting
error of the mean field. On N2 in cc-pVDZ that error is about 3e-4 Ha in the DMET
energy. Whether that matters is your call.

If the mean field is not density fitted, the code reuses the atomic-orbital
integrals the mean field already holds when it can, and only rebuilds them when it
must.

## Limits and things to watch

The cluster is built on an empty PySCF molecule with the integrals supplied
directly. This is the standard PySCF pattern for a custom Hamiltonian. It means
anything that asks the cluster about atoms would fail, so the code passes the real
molecule through for spin-orbit coupling, magnetic properties and orbital output.
Any new Prism feature that needs atom positions will need the same treatment.

Polarizable embedding is not integrated with DMET.

The energies reported by the NEVPT2 solvers are cluster energies without nuclear
repulsion. Excitation energies are unaffected.

The correlation potential from a self-consistent run is not a well determined
quantity. The energy is.

Density fitting changes the answer by the size of the fitting error. It is a
different calculation, not a cheaper route to the same numbers.

## Examples

The `examples/dmet` folder has four scripts. Every one of them runs the same
calculation on the whole molecule as well, so you can see what the embedding costs.

- `01-dmet-h4-fci.py`, the smallest working example. One-shot DMET, self-consistent
  DMET and a density-fitted run, all checked against full-molecule FCI.
- `02-dmet-h2o-qdnevpt2.py`, excitation energies and oscillator strengths, checked
  against a direct QD-NEVPT2.
- `03-dmet-n2-solvers.py`, CASSCF, PC-NEVPT2 and CVS-IP-MR-ADC on the same system,
  with the bath spectrum and the AO-basis density.
- `04-dmet-cunh34-soc-gtensor.py`, a copper complex with spin-orbit coupling, a
  g-tensor and powder magnetization and susceptibility.

## Tests

The test suite lives in `dmet/tests` and runs with pytest.

```
python -m pytest dmet/tests -q
```

Many of the tests are identity tests. They put the whole molecule in the impurity,
which makes the embedding exact, and then check that the embedded answer matches the
unembedded one. That is the check to copy when adding a feature.
