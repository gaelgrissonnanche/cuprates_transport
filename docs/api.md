# API guide

This page documents the intended high-level workflow. Several attributes are
currently public for historical reasons; code outside the package should avoid
depending on names beginning with an underscore.

## `BandStructure`

```python
BandStructure(
    a,
    b,
    c,
    energy_scale,
    band_params=...,
    band_model=...,
    band_name="band_1",
    resolution=[20, 20, 3],
    k_max=[pi, pi, 2*pi],
    number_of_bz=1,
    parallel=True,
    march_square=False,
)
```

The `band_model` is a SymPy-compatible expression in `kx`, `ky`, `kz`, the
lattice constants, and the keys of `band_params`. Energies are expressed as a
multiple of `energy_scale`.

Main methods:

- `runBandStructure(epsilon=0, printDoping=False)` computes the discretized
  constant-energy surface and derived weights.
- `energy_func(kx, ky, kz)` evaluates the dispersion.
- `velocity_func(kx, ky, kz)` evaluates the three velocity components.
- `doping(...)` and `filling(...)` estimate carrier counts on a grid.
- `set_mu_to_doping(p_target, ptol=0.001)` adjusts the chemical potential.

Important outputs after `runBandStructure` are `kf`, `vf`, `dkf`, `dos_k`,
`dos_epsilon`, `p`, and `n`.

Changing a band parameter through `band[key] = value` invalidates the existing
Fermi-surface discretization. Run `runBandStructure` again before transport.

## `Scattering`

Scattering behavior is configured indirectly through `Conductivity`:

```python
scattering_models=["isotropic", "cos2phi"]
scattering_params={"gamma_0": 15.1, "gamma_k": 66, "power": 12}
```

Currently implemented model names include:

| Model | Form | Required parameters |
| --- | --- | --- |
| `isotropic` | \(\Gamma_0\) | `gamma_0` |
| `cos2phi` | \(\Gamma_k|\cos 2\phi|^p\) | `gamma_k`, `power` |
| `cos2phi_coskz` | angular model with \(k_z\) modulation | `gamma_k`, `power`, `power_z` |
| `sin2phi` | \(\Gamma_k|\sin 2\phi|^p\) | `gamma_k_sin`, `power_sin` |
| `gamma_dos_func` | rate proportional to local DOS | `gamma_dos_max` |
| `polynomial` | absolute fifth-order polynomial in \(\phi\) | `a1` through `a5` |
| `gamma_skew_marginal_fl` | energy- and temperature-dependent model | `a_epsilon`, `a_abs_epsilon`, `a_T` |

Rates are expressed in ps^-1 and are summed before inversion to obtain the
total lifetime.

## `Conductivity`

```python
Conductivity(
    bandObject,
    Bamp,
    Bphi=0,
    Btheta=0,
    N_time=500,
    rtol=1e-4,
    atol=1e-4,
    T=0,
    dfdE_cut_percent=0.001,
    N_epsilon=20,
    scattering_models=...,
    scattering_params=...,
)
```

Call `runTransport()` after the supplied `BandStructure` has been discretized.
Results are written to:

- `sigma`: electrical-conductivity tensor;
- `alpha`: thermoelectric tensor at finite temperature;
- `beta`: thermal transport tensor at finite temperature;
- `kft`, `vft`: momentum and velocity histories;
- `tau_tot`, `t_o_tau`: lifetimes and integrated scattering exponent.

The magnetic-field properties `Bamp`, `Btheta`, and `Bphi` may be updated
between calls. Each update refreshes the Cartesian field vector.

## `ADMR`

```python
ADMR(
    condObject_list,
    Btheta_min=0,
    Btheta_max=110,
    Btheta_step=5,
    Btheta_norm=0,
    Bphi_array=[0, 15, 30, 45],
    show_progress=True,
)
```

`condObject_list` contains one `Conductivity` instance per independent band.
All bands are evaluated at each angle and their conductivity tensors are added
before inversion.

- `runADMR()` fills `rhozz_array` and normalized `rzz_array`.
- `fileADMR(...)` writes numerical data and calculation metadata.
- `figADMR(...)` creates the standard angular plots.

## Fitting modules

The package presently retains separate single-temperature, multi-temperature,
serial, and parallel fitting modules. Their APIs grew around specific ADMR data
workflows and should be considered provisional before version 1.0:

- `fitting_admr_series_oneT`
- `fitting_admr_series_multi_T`
- `fitting_admr_parallel_oneT`
- `fitting_admr_parallel_multiT`

For a published fit, archive the input data mapping, bounds, initial values,
objective definition, optimizer settings, random seed where applicable, and
the final parameter covariance or uncertainty procedure.
