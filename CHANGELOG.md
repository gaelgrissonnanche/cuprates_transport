# Changelog

This project uses a human-readable changelog. Versions before 1.0 may refine
the public API; scientific results should always record the exact release or
Git commit used.

## 0.7.0 - Unreleased

### Added

- Modern `pyproject.toml` package metadata and reproducible build configuration.
- Installation, theory, units, API, numerical-validation, contribution,
  support, and citation documentation.
- NumPy-style public API docstrings covering parameters, units, array shapes,
  returns, state mutations, numerical assumptions, and file side effects.
- Continuous-integration checks for package builds and core numerical tests.
- A test-suite boundary that explicitly skips the fitting regression when its
  non-distributed experimental dataset is unavailable.

### Changed

- Declared the complete import-time dependency set, including `sympy` and
  `psutil`, which was incomplete in the legacy packaging metadata.

### Removed

- Legacy `setup.py` and `setup.cfg` metadata in favor of `pyproject.toml`.
- The repository-wide Flake8 unit test, which scanned generated and ignored
  files and did not provide an actionable code-quality signal.

## 0.6 - 2021-01-20

- Historical research release containing band-structure, Chambers transport,
  ADMR, scattering, and fitting implementations.
