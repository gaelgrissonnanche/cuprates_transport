# Contributing

Contributions that improve correctness, documentation, validation, portability,
or usability are welcome. For scientific changes, explain the physical model,
units, assumptions, and expected numerical effect.

## Before opening a change

Use an issue to discuss changes that alter equations, conventions, public APIs,
or numerical results. Small documentation and test corrections can be submitted
directly as pull requests.

## Development setup

```bash
git clone https://github.com/gaelgrissonnanche/cuprates_transport.git
cd cuprates_transport
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

The historical fitting regression is skipped because its Nd-LSCO input data
are not distributed. Re-enable it only after a redistributable test dataset has
been added with provenance and licensing information.

Build and validate distribution artifacts with:

```bash
python -m build
python -m twine check dist/*
```

Build the documentation locally with:

```bash
mkdocs serve
```

## Scientific-change checklist

A change that can affect numerical output should include:

1. A derivation or authoritative reference for the implemented expression.
2. An explicit statement of dimensions and unit conversions.
3. A focused test with an analytic limit, independent implementation, or
   previously validated reference value.
4. A convergence test for any new discretization or integration parameter.
5. A changelog entry explaining whether existing results can change.

Do not silently change signs, tensor-index conventions, default resolutions,
integration limits, or scattering-rate definitions.

## Style and documentation

- Prefer descriptive physical names and document their units.
- Add docstrings to public classes, methods, and functions.
- Use NumPy-style sections (`Parameters`, `Returns`, `Notes`, and `References`)
  when expanding API docstrings.
- Keep examples executable and use reduced resolution only when explicitly
  labelled as a demonstration.
- Avoid committing generated figures, caches, editor settings, or private data.

## Reporting results

Pull requests should report the Python version, platform, test command, and the
relevant before/after numerical values. If output changes only at finite
precision, state both the absolute and relative difference.
