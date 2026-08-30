# cuprates-transport

`cuprates-transport` calculates semiclassical Boltzmann transport from the
Chambers solution of the relaxation-time Boltzmann equation. Starting from an
arbitrary symbolic tight-binding dispersion, it discretizes the Fermi surface,
evolves quasiparticle trajectories in a magnetic field, and evaluates the full
electrical-conductivity tensor. It also supports thermoelectric coefficients,
angle-dependent magnetoresistance (ADMR), multiple bands, momentum- and
energy-dependent scattering rates, and ADMR fitting.

The project originated in studies of cuprate metals, but the transport engine
is not restricted to cuprate band structures. A future release may adopt the
more general project name **BoltzmannChambers**; the current import name remains
`cuprates_transport`.

> **Project status:** research software approaching its first broadly supported
> release. The numerical implementation is established, but the public API may
> be refined before version 1.0. Published calculations should record the exact
> package version or Git commit and all numerical convergence parameters.

## Capabilities

- Symbolic tight-binding dispersions with analytically generated velocities
- Two- and three-dimensional Fermi-surface discretization
- Chambers conductivity to all orders in magnetic field within the
  semiclassical relaxation-time approximation
- Momentum-, energy-, and temperature-dependent scattering models
- Electrical, thermoelectric, and thermal response tensors
- Single- and multiband ADMR
- Serial and parallel fitting workflows
- Reference calculations for a free-electron band and Nd-LSCO

## Installation

The package requires Python 3.9 or newer. From a checkout of this repository:

```bash
python -m pip install .
```

For an editable development installation with test and documentation tools:

```bash
python -m pip install -e ".[dev,docs]"
```

The repository also contains [`chambers.yml`](chambers.yml), which describes
the Conda environment historically used for the calculations. The package
metadata in `pyproject.toml` is the authoritative dependency list for new
installations.

## Quick start

The following example computes the conductivity and resistivity tensors for a
single warped cylindrical Fermi surface:

```python
import numpy as np
from numpy import pi

from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity

band = BandStructure(
    band_name="large_pocket",
    a=3.74767,
    b=3.74767,
    c=13.2,
    energy_scale=190,  # meV
    band_params={
        "mu": -0.826,
        "t": 1.0,
        "tp": -0.14,
        "tpp": 0.07,
        "tz": 0.07,
    },
    resolution=[21, 21, 7],
    k_max=[pi, pi, 2 * pi],
)
band.runBandStructure()

transport = Conductivity(
    band,
    Bamp=45,       # tesla
    Btheta=0,      # degrees
    Bphi=0,        # degrees
    T=0,           # kelvin
    scattering_models=["isotropic", "cos2phi"],
    scattering_params={
        "gamma_0": 15.1,
        "gamma_k": 66.0,
        "power": 12,
    },
)
transport.runTransport()

sigma = transport.sigma       # S m^-1
rho = np.linalg.inv(sigma)    # ohm m
print(rho)
```

For production calculations, increase the Fermi-surface and time resolutions
until the requested observables are converged. See the
[numerical guidance](docs/numerics.md) before interpreting a result.

## Documentation map

- [Physical theory and its implementation](docs/theory.md)
- [Units and conventions](docs/units.md)
- [Public API guide](docs/api.md)
- [Numerical convergence and validation](docs/numerics.md)
- [Worked scripts](examples/)
- [Contributing and development](CONTRIBUTING.md)
- [Support policy](SUPPORT.md)

## Scientific background

The numerical formulation follows the Chambers solution of the semiclassical
Boltzmann equation. The implementation and its use for ADMR are described in:

> G. Grissonnanche *et al.*, “Linear-in temperature resistivity from an
> isotropic Planckian scattering rate,” *Nature* **595**, 667–672 (2021).
> [doi:10.1038/s41586-021-03697-8](https://doi.org/10.1038/s41586-021-03697-8)

The original Chambers reference is:

> R. G. Chambers, “The kinetic formulation of conduction problems,”
> *Proceedings of the Physical Society. Section A* **65**, 458 (1952).

## Reproducibility checklist

When reporting a calculation, preserve at least:

- the package version or Git commit;
- the complete band dispersion and parameter dictionary;
- lattice constants and energy scale;
- Fermi-surface resolution and Brillouin-zone bounds;
- temperature, magnetic-field magnitude and orientation;
- scattering models and all scattering parameters;
- time and energy integration settings (`N_time`, tolerances, `N_epsilon`, and
  `dfdE_cut_percent`);
- the convergence study used for the reported observable.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Until a
versioned software DOI is available, cite the repository and the exact release
or commit used, together with the relevant physics publication.

## License

This project is distributed under the GNU General Public License v3.0. See
[`LICENSE.txt`](LICENSE.txt).
