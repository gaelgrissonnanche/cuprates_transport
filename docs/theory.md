# Theory and implementation

## Scope of the model

The package solves the semiclassical Boltzmann transport equation in the
relaxation-time approximation. It assumes that transport can be described by
quasiparticles with a band energy \(\varepsilon(\mathbf{k})\), group velocity
\(\mathbf{v}=\hbar^{-1}\nabla_{\mathbf{k}}\varepsilon\), and a possibly
momentum-, energy-, and temperature-dependent lifetime \(\tau\).

Within these assumptions, the magnetic field is included nonperturbatively:
the orbit integration is not an expansion in powers of \(B\).

## Chambers conductivity

For a momentum-independent lifetime, the Chambers solution can be written

\[
\sigma_{ij}=\frac{e^2}{4\pi^3}\int d^3k
\left(-\frac{\partial f_0}{\partial\varepsilon}\right)
v_i[\mathbf{k}(0)]\int_{-\infty}^{0}
v_j[\mathbf{k}(t)]e^{t/\tau}\,dt.
\]

For \(\tau=\tau[\mathbf{k}(t)]\), the exponential is replaced by the survival
factor

\[
\exp\left[-\int_0^t\frac{dt'}{\tau[\mathbf{k}(t')]}\right]
\]

when the past trajectory is represented using positive integration time.

The orbit follows the semiclassical Lorentz equation

\[
\hbar\frac{d\mathbf{k}}{dt}=-e\,\mathbf{v}\times\mathbf{B},
\]

with \(e>0\) denoting the elementary-charge magnitude. The implementation
reverses the magnetic field in the positive-time integration so that the
computed trajectory represents the past history in the conventional Chambers
formula.

## Mapping equations to the package

| Physical operation | Implementation |
| --- | --- |
| Parse \(\varepsilon(\mathbf{k})\) | `BandStructure.energy_velocity_definition` |
| Compute \(\mathbf{v}(\mathbf{k})\) | `BandStructure.velocity_func` |
| Discretize the Fermi surface | `BandStructure.discretize_fermi_surface` |
| Evaluate \(\tau(\mathbf{k},\varepsilon,T)\) | `Scattering.tau_tot_func` |
| Integrate the Lorentz orbit | `Conductivity.solve_movement` |
| Accumulate \(\int dt'/\tau\) | `Conductivity.t_o_tau_func` |
| Form velocity-history products | `Conductivity.velocity_product` |
| Integrate the conductivity tensor | `Conductivity.chambers_func` |
| Sweep field angles and invert \(\sigma\) | `ADMR.runADMR` |

## Fermi-surface reduction

At zero temperature, \(-\partial f_0/\partial\varepsilon\) becomes a delta
function at the Fermi energy. The volume integral is then reduced to a
Fermi-surface integral with the appropriate density-of-states weight. The
`BandStructure` object stores the discretized momenta (`kf`), velocities
(`vf`), surface weights (`dkf`), and local density-of-states factor (`dos_k`).

At finite temperature, `Conductivity` evaluates the transport tensor on an
energy grid and integrates it against the derivative of the Fermi function.
The same energy-resolved conductivity is combined with additional energy
factors to obtain `alpha` and `beta`.

## Multiple scattering channels

The configured scattering models are added as rates,

\[
\Gamma_{\mathrm{tot}}=\frac{1}{\tau_{\mathrm{tot}}}
=\sum_a\Gamma_a,
\]

which is the implementation of Matthiessen's rule in
`Scattering.tau_tot_func`. The order of entries in `scattering_models` does not
change the sum, but each named model requires its corresponding parameters in
`scattering_params`.

## Multiple bands

Independent bands contribute additively to conductivity, not resistivity:

\[
\boldsymbol{\sigma}_{\mathrm{total}}=\sum_b
\boldsymbol{\sigma}_b,
\qquad
\boldsymbol{\rho}=\boldsymbol{\sigma}_{\mathrm{total}}^{-1}.
\]

`ADMR.runADMR` follows this order for every magnetic-field orientation.

## References

1. R. G. Chambers, *Proc. Phys. Soc. A* **65**, 458 (1952).
2. G. Grissonnanche *et al.*, *Nature* **595**, 667-672 (2021),
   [doi:10.1038/s41586-021-03697-8](https://doi.org/10.1038/s41586-021-03697-8).
