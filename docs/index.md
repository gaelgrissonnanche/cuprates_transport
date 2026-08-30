# cuprates-transport documentation

`cuprates-transport` is a research package for evaluating semiclassical
transport tensors from the Chambers solution of the Boltzmann equation.

Start with the [README](../README.md) for installation and a minimal example.
The remaining pages separate the physics, conventions, API, and numerical
checks so that calculations can be understood and reproduced independently.

## Suggested reading order

1. [Theory](theory.md): the transport equation and its mapping to the code.
2. [Units and conventions](units.md): the dimensions and sign conventions.
3. [API guide](api.md): the main objects and their lifecycle.
4. [Numerics and validation](numerics.md): convergence and analytic checks.

## Core workflow

```text
tight-binding dispersion
        |
        v
BandStructure.runBandStructure()
        |
        | Fermi-surface points, velocities, integration weights
        v
Conductivity.runTransport()
        |
        | sigma, alpha, beta tensors
        v
tensor inversion or ADMR angular sweep
```

The separation between the band, scattering model, orbit solver, and ADMR
driver is important: it allows each part of a calculation to be validated
independently.
