# Units and conventions

The public inputs intentionally use units common in condensed-matter physics,
while the returned transport tensors use SI units.

## Input and output units

| Quantity | Attribute or parameter | Unit |
| --- | --- | --- |
| Lattice constants | `a`, `b`, `c` | angstrom |
| Momentum | `kf`, `kx`, `ky`, `kz` | inverse angstrom |
| Overall band scale | `energy_scale` | meV |
| Dimensionless hoppings | entries of `band_params` | fraction of `energy_scale` |
| Temperature | `T` | K |
| Magnetic field | `Bamp` | T |
| Field angles | `Btheta`, `Bphi` | degrees |
| Time and lifetime | `time_array`, `tau_tot` | ps |
| Scattering rate | scattering amplitudes such as `gamma_0` | ps^-1 (THz) |
| Velocity | `vf`, `vft` | m s^-1 |
| Electrical conductivity | `sigma` | S m^-1 |
| Electrical resistivity | `inv(sigma)` | ohm m |

`beta` and `alpha` follow the internal thermoelectric convention used by
`Conductivity.integrand_coeff`. When reporting derived Seebeck or thermal
coefficients, record the exact tensor combination and charge-sign convention
rather than relying only on a symbol name.

## Reciprocal-space convention

The default tight-binding expression receives physical momenta in inverse
angstrom and forms dimensionless arguments such as `a*kx`, `b*ky`, and `c*kz`.
`k_max` specifies dimensionless Brillouin-zone bounds; for example,
`[pi, pi, 2*pi]` corresponds to \(k_x\in[-\pi/a,\pi/a]\),
\(k_y\in[-\pi/b,\pi/b]\), and \(k_z\in[-2\pi/c,2\pi/c]\).

## Charge and field convention

The constant `e` is the positive elementary-charge magnitude. The orbit solver
uses the electron equation of motion and reverses the magnetic field while
integrating forward in numerical time to reconstruct the past trajectory in
the Chambers expression. Hall-sign validation should therefore be performed
against a band with a known carrier type whenever a new dispersion is added.

## Tensor indices

The Cartesian index mapping is `x = 0`, `y = 1`, and `z = 2`. Thus
`sigma[0, 1]` is \(\sigma_{xy}\), and `sigma[2, 2]` is \(\sigma_{zz}\).
The physical resistivity tensor is obtained with:

```python
rho = numpy.linalg.inv(conductivity.sigma)
```

Do not approximate \(\rho_{ii}=1/\sigma_{ii}\) when off-diagonal terms are
non-negligible.

## Conversion examples

```python
rho_ohm_m = numpy.linalg.inv(transport.sigma)
rho_microohm_cm = rho_ohm_m * 1e8
hall_mm3_per_c = (rho_ohm_m[1, 0] / transport.Bamp) * 1e9
```

The older derivation in `docs/readme_units_conversion_ADMR.pdf` is retained as
historical supporting documentation. This page should become the authoritative
human-readable unit reference as equations and APIs are reviewed.
