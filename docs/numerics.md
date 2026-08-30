# Numerics, convergence, and validation

Correct execution is not by itself evidence that a transport result is
converged. Every published observable should be tested against the numerical
parameters that control its calculation.

## Fermi-surface resolution

`BandStructure.resolution` controls discretization in the three reciprocal
directions. Increase the in-plane and \(k_z\) resolutions separately and track
the relative change in the final observable, not only in the Fermi-surface
shape.

Recommended procedure:

1. Choose a baseline resolution that runs quickly.
2. Increase the in-plane resolution while keeping the \(k_z\) grid fixed.
3. Increase the \(k_z\) resolution after the in-plane result stabilizes.
4. Repeat near topology changes, van Hove points, or sharply anisotropic
   scattering, where convergence can deteriorate.

## Orbit-time integration

`Conductivity` integrates to eight times the longest lifetime returned on the
Fermi surface. `N_time` sets the number of stored time samples; `rtol` and
`atol` control the ODE solver.

Check convergence by increasing `N_time` and tightening both tolerances. A
stable trajectory is not sufficient: compare the final tensor components,
especially small off-diagonal components obtained through cancellation.

## Finite-temperature energy integral

At nonzero `T`, the calculation uses `N_epsilon` points over a window chosen by
`dfdE_cut_percent`. Converge both parameters. Thermoelectric coefficients are
more sensitive to the energy mesh than electrical conductivity because their
integrands contain additional powers of energy.

## Useful analytic and physical limits

The repository includes a free-electron example and hard-coded regression
values. A robust validation program should retain at least these checks:

- **Zero field:** off-diagonal conductivity vanishes when required by symmetry.
- **Drude limit:** a parabolic isotropic band reproduces Drude longitudinal and
  Hall transport.
- **Onsager relation:** compare \(\sigma_{ij}(\mathbf B)\) with
  \(\sigma_{ji}(-\mathbf B)\) for time-reversal-symmetric inputs.
- **Tensor inversion:** verify the sign and magnitude of Hall resistivity from
  the full inverse, not a diagonal approximation.
- **Constant lifetime:** confirm expected weak- and strong-field limits.
- **Mesh refinement:** demonstrate a documented tolerance for every reported
  observable.

## Existing regression tests

The current tests exercise Fermi-surface discretization and representative
zero- and finite-field conductivities. They are valuable regression guards,
but they do not yet constitute comprehensive physical validation across all
models and Python/dependency versions.

Run the current suite with:

```bash
python -m unittest discover -s tests -v
```

The fitting regression is currently skipped because its experimental input
data are not distributed with the repository. This gap should be closed with a
small, redistributable dataset or a synthetic fitting benchmark.

For a release, record the environment and archive the test report. Any intended
change in a reference value should be explained physically and numerically in
the changelog.

## Reproducible output

Raw output files should carry enough metadata to reconstruct the calculation.
Do not encode the only copy of parameters in a filename. Prefer a structured
sidecar file (JSON, YAML, or TOML) containing all parameters and the software
version; this is a recommended future addition because implementing it would
change output behavior.
